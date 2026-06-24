"""
engine/agents/reflection.py — S2 Reflection Generator (Layer 1 LLM, Layer 2 zero-LLM)

Spec: docs/spec_agent_reflection_memory.md v1.0 (frozen 2026-05-04).

This module turns a (decision, realized_outcome, factor_context) triple into a
persisted AgentReflection row with:
  * 4-section structured memo (LLM-generated, prompt frozen at spec §3.2)
  * rule-based hit/miss flag (no LLM, per feedback_no_llm_as_judge.md)
  * 384-dim embedding (sentence-transformers if installed; deterministic hash
    fallback otherwise — production retrieval requires real ST install, but
    smoke tests and offline pipelines stay green either way)

Public API:
  compute_hit_flag(direction, realized_outcome) -> str
  compute_embedding(text)                       -> list[float]
  generate_reflection_text(...)                 -> str        (LLM call)
  build_and_persist_reflection(...)             -> int        (returns id)
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

EMBEDDING_DIM        = 384
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HIT_THRESHOLD        = 0.005
PROMPT_VERSION       = "reflection_v1.0_frozen_2026-05-04"

# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — rule-based hit/miss (zero LLM)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hit_flag(direction: str | None, realized_outcome: float | None) -> str:
    """
    Spec §3.3. Direction ∈ {long, short, neutral, None}; realized_outcome may
    be None (pre-backfill) — returns "pending" so retrieval can filter.
    """
    if realized_outcome is None:
        return "pending"
    if direction is None or direction == "neutral":
        return "neutral"

    predicted_sign = +1 if direction == "long" else -1
    if realized_outcome == 0.0:
        return "miss"
    realized_sign = +1 if realized_outcome > 0 else -1

    if realized_sign != predicted_sign:
        return "miss"
    if abs(realized_outcome) > HIT_THRESHOLD:
        return "hit"
    return "partial"


# ─────────────────────────────────────────────────────────────────────────────
# Embedding — sentence-transformers preferred, deterministic hash fallback
# ─────────────────────────────────────────────────────────────────────────────

_st_model = None
_st_tried = False

def _get_st_model():
    global _st_model, _st_tried
    if _st_tried:
        return _st_model
    _st_tried = True
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("reflection: sentence-transformers loaded (%s)", EMBEDDING_MODEL_NAME)
    except Exception as exc:
        logger.warning(
            "reflection: sentence-transformers unavailable (%s); using hash fallback. "
            "Install for production retrieval: pip install sentence-transformers",
            exc,
        )
        _st_model = None
    return _st_model


def _hash_embedding(text: str) -> list[float]:
    """
    Deterministic 384-dim unit vector derived from SHA-256 of text.
    Not semantically meaningful — keeps schema/round-trip working when
    sentence-transformers is unavailable. Same text → same vector, but
    cosine similarity carries no semantic signal.
    """
    out: list[float] = []
    counter = 0
    while len(out) < EMBEDDING_DIM:
        h = hashlib.sha256(f"{text}|{counter}".encode("utf-8")).digest()
        # 8 floats per 32-byte hash (4 bytes each)
        for i in range(0, 32, 4):
            (val,) = struct.unpack("<i", h[i:i+4])
            out.append(val / 2147483648.0)  # → roughly [-1, 1]
            if len(out) >= EMBEDDING_DIM:
                break
        counter += 1
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


def compute_embedding(text: str) -> list[float]:
    """
    384-dim L2-normalized list[float]. Real ST if installed, deterministic
    hash otherwise. Normalization keeps the retrieval contract simple:
    dot(a, b) == cosine_similarity(a, b) for any two stored embeddings.
    """
    text = text or ""
    model = _get_st_model()
    if model is not None:
        try:
            vec = model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            return [float(x) for x in vec.tolist()]
        except Exception as exc:
            logger.warning("reflection: ST encode failed (%s); falling back to hash", exc)
    return _hash_embedding(text)  # hash fallback already L2-normalized


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — LLM reflection generator (prompt frozen per spec §3.2)
# ─────────────────────────────────────────────────────────────────────────────

REFLECTION_PROMPT_TEMPLATE = """You are a reflection agent. Below is a past decision \
and its realized outcome. Write a structured reflection in 4 sections \
(50-80 chars each, total 200-400 chars).

Decision context (JSON):
{decision_summary}

Realized outcome:
{realized_outcome_str}

Factor context (B++ snapshot at decision time, JSON):
{factor_context}

Recent prior reflections (most recent last; may be empty):
{prior_reflections_block}

Write exactly four sections in this order, each prefixed with the bracketed tag:

[CONTEXT]   What was the decision context?
[DECISION]  What did I predict?
[OUTCOME]   What actually happened?
[LESSON]    What pattern should I learn for future similar contexts?

Constraints:
- Total length 200-400 chars
- Use English (cross-language consistency)
- No markdown, no bullets — narrative prose
- Refer to factor IC / ICIR / beta numbers if relevant
- Do not output anything outside the four sections
"""


def _format_prior_reflections(prior: list[str] | None) -> str:
    if not prior:
        return "(none)"
    lines = []
    for i, r in enumerate(prior[-5:], 1):
        lines.append(f"{i}. {r.strip()[:240]}")
    return "\n".join(lines)


def _format_realized_outcome(val: float | None) -> str:
    if val is None:
        return "(pending — backfill required)"
    return f"{val:+.4f}"


def generate_reflection_text(
    decision_summary: dict,
    realized_outcome: float | None,
    factor_context: dict | None,
    prior_reflections: list[str] | None = None,
    model: Any | None = None,
) -> str:
    """
    Call LLM with frozen prompt. Returns reflection text (string).
    Raises RuntimeError if model is None or LLM call fails — callers decide
    whether to skip (pre-backfill) or persist with placeholder.
    """
    if model is None:
        raise RuntimeError("generate_reflection_text: model required (no LLM provided)")

    prompt = REFLECTION_PROMPT_TEMPLATE.format(
        decision_summary=json.dumps(decision_summary, ensure_ascii=False, sort_keys=True),
        realized_outcome_str=_format_realized_outcome(realized_outcome),
        factor_context=json.dumps(factor_context or {}, ensure_ascii=False, sort_keys=True),
        prior_reflections_block=_format_prior_reflections(prior_reflections),
    )

    try:
        raw = model.generate_content(prompt).text
    except Exception as exc:
        logger.warning("generate_reflection_text: LLM call failed: %s", exc)
        raise RuntimeError(f"llm_call_failed: {exc}") from exc

    text = (raw or "").strip()
    if not text:
        raise RuntimeError("llm_returned_empty")

    return text


def validate_reflection_schema(text: str) -> bool:
    """
    Rule-based check that text contains all 4 section tags in order
    and total length within 150-600 (slightly looser than 200-400 to
    tolerate LLM whitespace).
    """
    if not text:
        return False
    n = len(text)
    if n < 150 or n > 800:
        return False
    expected = ["[CONTEXT]", "[DECISION]", "[OUTCOME]", "[LESSON]"]
    pos = -1
    for tag in expected:
        idx = text.find(tag, pos + 1)
        if idx == -1 or idx <= pos:
            return False
        pos = idx
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — wires into engine.memory.AgentReflection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReflectionInput:
    agent_id:         str
    decision_date:    datetime.date
    decision_summary: dict
    realized_outcome: float | None     = None
    factor_context:   dict | None      = None
    decision_ref_id:  int | None       = None
    prior_reflections: list[str]       = field(default_factory=list)


def build_and_persist_reflection(
    inp: ReflectionInput,
    model: Any | None = None,
    session: Any | None = None,
) -> int:
    """
    End-to-end:
      LLM call (frozen prompt) → schema validate → embedding → ORM row insert.

    Returns: AgentReflection.id (int).
    Raises: on LLM failure or schema invalid (caller decides whether to skip).
    """
    from engine.memory import AgentReflection, SessionFactory

    text = generate_reflection_text(
        decision_summary  = inp.decision_summary,
        realized_outcome  = inp.realized_outcome,
        factor_context    = inp.factor_context,
        prior_reflections = inp.prior_reflections,
        model             = model,
    )
    if not validate_reflection_schema(text):
        raise RuntimeError(f"reflection_schema_invalid (len={len(text)})")

    direction = (inp.decision_summary or {}).get("direction")
    hit_flag  = compute_hit_flag(direction, inp.realized_outcome)
    embedding = compute_embedding(text)

    row = AgentReflection(
        agent_id         = inp.agent_id,
        decision_ref_id  = inp.decision_ref_id,
        decision_date    = inp.decision_date,
        decision_summary = json.dumps(inp.decision_summary, ensure_ascii=False, sort_keys=True),
        realized_outcome = inp.realized_outcome,
        hit_flag         = hit_flag,
        factor_context   = json.dumps(inp.factor_context or {}, ensure_ascii=False, sort_keys=True),
        reflection_text  = text,
        embedding        = json.dumps(embedding),
        embedding_model  = EMBEDDING_MODEL_NAME,
        created_at       = datetime.datetime.utcnow(),
    )

    own_session = session is None
    sess = session if session is not None else SessionFactory()
    try:
        sess.add(row)
        sess.commit()
        rid = row.id
        logger.info(
            "reflection persisted: id=%s agent=%s date=%s hit=%s len=%d",
            rid, inp.agent_id, inp.decision_date, hit_flag, len(text),
        )
        return rid
    except Exception:
        sess.rollback()
        raise
    finally:
        if own_session:
            sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# RAG Retrieval (S2.4) — spec §4
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TOP_K          = 5
DEFAULT_LOOKBACK_MONTHS = 18


def _cosine(a: list[float], b: list[float]) -> float:
    """Both stored embeddings are L2-normalized → dot product == cosine."""
    return sum(x * y for x, y in zip(a, b))


def retrieve_relevant_reflections(
    agent_id:        str,
    query_text:      str,
    k:               int = DEFAULT_TOP_K,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    exclude_ids:     list[int] | None = None,
    as_of:           datetime.date | None = None,
    session:         Any | None = None,
) -> list:
    """
    Top-K cosine-similar reflections from same agent, recency-filtered.

    Returns list of AgentReflection rows in descending similarity order.
    Empty list if no candidates / empty query.

    Semantics:
      * agent_id filter → no cross-agent contamination
      * decision_date >= as_of - lookback_months*30 days → factor-regime
        relevance (older reflections may reference dead regimes)
      * exclude_ids skips self when called during a re-reflection refresh
      * embedding NULL rows skipped (pending state)

    Spec docs/spec_agent_reflection_memory.md §4.2 (frozen v1.0).
    """
    from engine.memory import AgentReflection, SessionFactory

    if not query_text:
        return []
    if as_of is None:
        as_of = datetime.date.today()
    cutoff = as_of - datetime.timedelta(days=30 * lookback_months)

    query_emb = compute_embedding(query_text)

    own_session = session is None
    sess = session if session is not None else SessionFactory()
    try:
        candidates = (
            sess.query(AgentReflection)
                .filter(AgentReflection.agent_id == agent_id)
                .filter(AgentReflection.decision_date >= cutoff)
                .filter(AgentReflection.embedding.isnot(None))
                .all()
        )

        scored: list[tuple[float, Any]] = []
        for c in candidates:
            if exclude_ids and c.id in exclude_ids:
                continue
            try:
                c_emb = json.loads(c.embedding)
            except Exception:
                continue
            if len(c_emb) != EMBEDDING_DIM:
                continue
            scored.append((_cosine(query_emb, c_emb), c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]
    finally:
        if own_session:
            sess.close()


def format_reflections_for_prompt(reflections: list, agent_id: str = "") -> str:
    """
    Render top-K reflections for prompt injection per spec §4.3.
    Returns empty string if no reflections (so prompt has no dead block).
    """
    if not reflections:
        return ""

    lines: list[str] = []
    lines.append(f"=== Past Reflections (Top-{len(reflections)} relevant, agent={agent_id}) ===")
    lines.append("")

    for i, r in enumerate(reflections, 1):
        try:
            fc = json.loads(r.factor_context) if r.factor_context else {}
        except Exception:
            fc = {}
        ic_top = ""
        ic_list = fc.get("factor_ic_top3") if isinstance(fc, dict) else None
        if isinstance(ic_list, list) and ic_list:
            head = ic_list[0]
            if isinstance(head, dict) and head.get("name"):
                ic_top = f", factor_ic_top: {head['name']}"

        lines.append(f"[{i}] {r.decision_date}: {r.reflection_text}")
        lines.append(f"    Outcome: {r.hit_flag}{ic_top}")
        lines.append("")

    lines.append("=== End Reflections ===")
    return "\n".join(lines)


def build_reflection_query(
    decision_summary: dict | None = None,
    factor_context:   dict | None = None,
    extra_text:       str | None  = None,
) -> str:
    """
    Compose a short query string used as embedding seed for retrieval.
    Drops nothing — structured but free-form. The exact composition is not
    semantically critical (MiniLM tolerates concatenation), but stable
    composition keeps retrieval reproducible across runs.
    """
    parts: list[str] = []
    if decision_summary:
        sector = decision_summary.get("sector")
        direction = decision_summary.get("direction")
        rationale = decision_summary.get("rationale_excerpt")
        if sector:    parts.append(f"sector={sector}")
        if direction: parts.append(f"direction={direction}")
        if rationale: parts.append(str(rationale))
    if factor_context:
        ic_list = factor_context.get("factor_ic_top3")
        if isinstance(ic_list, list) and ic_list:
            names = [str(x.get("name")) for x in ic_list if isinstance(x, dict) and x.get("name")]
            if names:
                parts.append("top_factors=" + ",".join(names))
    if extra_text:
        parts.append(str(extra_text))
    return " | ".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────────────────
# Backfill loop (S2.6) — spec §6
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BACKFILL_BATCH    = 10   # spec §6.3: 5-10 per call
DEFAULT_BACKFILL_DAILY_CAP = 20  # spec §6.3: max 20 per day safety guard


def _decision_to_reflection_input(dec) -> "ReflectionInput":
    """
    Map a DecisionLog row into the ReflectionInput shape used by S2.3.

    Direction is derived from the structured `direction` field when present
    (超配/低配/标配/中性), falling back to "neutral".
    Realized outcome uses active_return (ETF − SPY).
    Factor context surfaces the quant snapshot already on the row.
    """
    direction_map = {
        "超配": "long", "低配": "short", "标配": "neutral", "中性": "neutral",
        "long": "long", "short": "short", "neutral": "neutral",
    }
    direction = direction_map.get((dec.direction or "").strip(), "neutral")

    decision_summary = {
        "sector":            dec.sector_name,
        "direction":         direction,
        "confidence":        (dec.confidence_score or 0) / 100.0,
        "rationale_excerpt": (dec.ai_conclusion or "")[:300],
        "horizon":           dec.horizon,
        "regime":            dec.macro_regime,
    }

    factor_context = {
        "quant_p_noise":  dec.quant_p_noise,
        "quant_val_r2":   dec.quant_val_r2,
        "quant_test_r2":  dec.quant_test_r2,
        "quant_active":   dec.quant_active,
        "weight_adj_pct": dec.weight_adjustment_pct,
        "active_return":  dec.active_return,
    }

    return ReflectionInput(
        agent_id         = "sector_pipeline",
        decision_date    = dec.decision_date or dec.created_at.date(),
        decision_summary = decision_summary,
        realized_outcome = float(dec.active_return) if dec.active_return is not None else None,
        factor_context   = factor_context,
        decision_ref_id  = dec.id,
    )


def generate_reflections_for_pending(
    as_of:        datetime.date | None = None,
    model:        Any | None  = None,
    max_per_call: int  = DEFAULT_BACKFILL_BATCH,
    daily_cap:    int  = DEFAULT_BACKFILL_DAILY_CAP,
) -> dict:
    """
    Spec §6 backfill loop. For sector_pipeline DecisionLog rows where the
    realized outcome (active_return) is filled but no reflection exists yet,
    generate a reflection memo via LLM and persist.

    Returns:
        {
          "processed": int,        # successfully persisted reflections
          "failed":    int,        # rows that errored (LLM / schema / DB)
          "skipped_daily_cap": bool,
          "candidates": int,       # pending rows seen this call
        }

    Failure handling:
      * If model is None and no key_pool is available, returns 0 processed.
      * Per-row exceptions are logged and counted as `failed` — never raised.
      * Daily cap is enforced by counting today's AgentReflection rows.

    Spec docs/spec_agent_reflection_memory.md §6 (frozen v1.0).
    """
    from sqlalchemy import exists, and_
    from engine.memory import (
        AgentReflection,
        DecisionLog,
        SessionFactory,
    )

    if as_of is None:
        as_of = datetime.date.today()

    summary = {
        "processed": 0,
        "failed":    0,
        "skipped_daily_cap": False,
        "candidates": 0,
    }

    # Lazy LLM resolution — defer to key_pool only if caller didn't supply one.
    if model is None:
        try:
            from engine.key_pool import get_pool
            model = get_pool().get_model()
        except Exception as exc:
            logger.warning(
                "generate_reflections_for_pending: no LLM available (%s); skipping",
                exc,
            )
            return summary

    with SessionFactory() as sess:
        # Daily-cap check: how many sector_pipeline reflections written today?
        today_count = sess.query(AgentReflection).filter(
            AgentReflection.agent_id == "sector_pipeline",
            AgentReflection.created_at >= datetime.datetime.combine(
                as_of, datetime.time.min
            ),
        ).count()
        remaining_today = max(0, daily_cap - today_count)
        if remaining_today == 0:
            summary["skipped_daily_cap"] = True
            return summary
        budget = min(max_per_call, remaining_today)

        # NOT EXISTS reflection for this DecisionLog.id (spec §6.2)
        no_reflection_exists = ~exists().where(
            and_(
                AgentReflection.decision_ref_id == DecisionLog.id,
                AgentReflection.agent_id == "sector_pipeline",
            )
        )

        candidates = (
            sess.query(DecisionLog)
                .filter(DecisionLog.tab_type == "sector")
                .filter(DecisionLog.active_return.isnot(None))
                .filter(DecisionLog.superseded.is_(False))
                .filter(no_reflection_exists)
                .order_by(DecisionLog.decision_date.asc(), DecisionLog.id.asc())
                .limit(budget * 2)  # over-fetch slightly; budget enforced below
                .all()
        )

        summary["candidates"] = len(candidates)
        for dec in candidates:
            if summary["processed"] >= budget:
                break
            try:
                inp = _decision_to_reflection_input(dec)
                build_and_persist_reflection(inp, model=model)
                summary["processed"] += 1
            except Exception as exc:
                logger.warning(
                    "generate_reflections_for_pending: decision_id=%s failed: %s",
                    dec.id, exc,
                )
                summary["failed"] += 1

    logger.info(
        "generate_reflections_for_pending(as_of=%s): processed=%d failed=%d "
        "candidates=%d daily_cap_hit=%s",
        as_of, summary["processed"], summary["failed"],
        summary["candidates"], summary["skipped_daily_cap"],
    )
    return summary
