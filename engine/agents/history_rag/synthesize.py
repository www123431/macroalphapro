"""
LLM synthesis with strict citation enforcement (P2.4 deliverable).

Wraps Gemini 2.5 Flash structured-output to turn a query + retrieved
docs into a cited answer. Enforces invariants:

  1. **Generation-only role**. LLM answers FROM the retrieved evidence;
     it never scores, ranks, or filters which docs are relevant. The
     0-LLM-in-evaluation invariant lives in retrieve.py (deterministic).
     This module is the *generation* half of the same red line.

  2. **Citation discipline**. Every claim in the answer must reference
     a citation_id that maps back to one of the retrieved docs. After
     parsing the LLM output we verify all cited IDs exist in the input
     evidence — synthetic / hallucinated cites are rejected.

  3. **Daily cost cap**. Tracks USD spend in a JSON state file under
     `.streamlit/`. Refuses synthesis when the daily budget is
     exhausted, returning a structured error so the UI can fall back
     to retrieval-only mode gracefully.

Public API
----------
  synthesize_answer(query, retrieval_results) -> SynthesizedAnswer

Returns SynthesizedAnswer with .answer, .citations, .cost_usd,
.budget_remaining, and .status ("ok" | "budget_exhausted" | "no_evidence"
| "llm_error").
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from engine.agents.history_rag.config import (
    SYNTHESIS_DAILY_BUDGET,
    SYNTHESIS_MAX_TOKENS,
    SYNTHESIS_MODEL,
    SYNTHESIS_TEMPERATURE,
)
from engine.agents.history_rag.retrieve import RetrievalResult
from engine.agents.history_rag.schema import SourceType

logger = logging.getLogger(__name__)

# ── Cost tracker ─────────────────────────────────────────────────────────────
_STREAMLIT_DIR = Path(__file__).resolve().parents[3] / ".streamlit"
_COST_STATE_PATH = _STREAMLIT_DIR / "rag_synthesis_cost.json"

# Gemini 2.5 Flash pricing per 1M tokens (matches engine/config.py).
_COST_PER_1M_INPUT  = 0.30
_COST_PER_1M_OUTPUT = 2.50


def _load_cost_state() -> dict:
    if not _COST_STATE_PATH.exists():
        return {"daily": {}, "calls_total": 0, "usd_total": 0.0}
    try:
        with open(_COST_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"daily": {}, "calls_total": 0, "usd_total": 0.0}


def _save_cost_state(state: dict) -> None:
    _STREAMLIT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _COST_STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, _COST_STATE_PATH)


def _today_iso() -> str:
    return datetime.date.today().isoformat()


def _today_spend_usd() -> float:
    return float(_load_cost_state()["daily"].get(_today_iso(), 0.0))


def get_synthesis_cost_status() -> dict:
    """Return current cost telemetry. Cheap; safe to call from UI.

    2026-05-08: budget reads through engine.llm_budget runtime helper so
    supervisor adjustments via System Console > LLM Budget tab propagate
    to all displays (executive_brief / research_console / spec_drafter
    page). Default 0.05/day from history_rag.config.SYNTHESIS_DAILY_BUDGET
    when no SystemConfig override is set.
    """
    from engine.llm_budget import get_rag_synthesis_daily_budget_usd
    state = _load_cost_state()
    today = _today_iso()
    today_spend = float(state["daily"].get(today, 0.0))
    daily_budget = get_rag_synthesis_daily_budget_usd()
    return {
        "today_usd":        today_spend,
        "today_budget_usd": daily_budget,
        "today_remaining":  max(0.0, daily_budget - today_spend),
        "calls_total":      int(state.get("calls_total", 0)),
        "usd_total":        float(state.get("usd_total", 0.0)),
    }


def _record_call(usd: float) -> None:
    state = _load_cost_state()
    today = _today_iso()
    state["daily"][today] = float(state["daily"].get(today, 0.0)) + usd
    state["calls_total"] = int(state.get("calls_total", 0)) + 1
    state["usd_total"]  = float(state.get("usd_total", 0.0)) + usd
    _save_cost_state(state)


# ── Output schema ────────────────────────────────────────────────────────────
# Gemini structured-output enforces this shape. The model fills citation_ids
# from the {{C0}}, {{C1}}, ... markers we put on each input doc.
_SYNTH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "Concise, factual answer grounded in the provided evidence. "
                "Inline-cite every claim using [C0] / [C1] / ... matching the "
                "evidence labels. If the evidence is insufficient, say so explicitly."
            ),
        },
        "citations": {
            "type": "array",
            "description": "Subset of evidence labels actually used in the answer.",
            "items": {
                "type": "object",
                "properties": {
                    "citation_id": {
                        "type": "string",
                        "description": "Label like 'C0', 'C1', ... (must exist in input evidence).",
                    },
                    "claim": {
                        "type": "string",
                        "description": "The specific claim from the answer this citation supports.",
                    },
                },
                "required": ["citation_id", "claim"],
            },
        },
        "evidence_quality": {
            "type": "string",
            "enum": ["strong", "partial", "thin", "irrelevant"],
            "description": (
                "Self-assessment of whether the retrieved evidence actually "
                "answers the query. 'irrelevant' = retrieval missed the question."
            ),
        },
    },
    "required": ["answer", "citations", "evidence_quality"],
}


# ── Output dataclass ─────────────────────────────────────────────────────────
@dataclass
class Citation:
    """One verified citation in a synthesized answer."""
    citation_id:   str             # "C0" / "C1" / ...
    claim:         str
    doc_id:        str             # mapped back to RetrievalResult.doc_id
    source_type:   SourceType
    title:         str
    deep_link:     str | None = None


@dataclass
class SynthesizedAnswer:
    """LLM-synthesized response over retrieved evidence.

    status meanings:
      "ok"                  — answer + citations populated; ready to display
      "no_evidence"         — empty retrieval input; LLM not called
      "budget_exhausted"    — would exceed daily synthesis budget
      "llm_error"           — Gemini call failed; check error_msg
      "citation_invalid"    — LLM cited a non-existent doc; suspect hallucination
    """
    query:             str
    answer:            str
    citations:         list[Citation]      = field(default_factory=list)
    evidence_quality:  str                 = "irrelevant"
    status:            str                 = "ok"
    cost_usd:          float               = 0.0
    input_tokens:      int                 = 0
    output_tokens:     int                 = 0
    budget_remaining:  float               = 0.0
    error_msg:         str | None          = None


# ── Prompt assembly ──────────────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = (
    "你是 Macro Alpha Pro 项目的研究助理。\n\n"
    "你的工作类型分两类，行为不同：\n\n"
    "(A) Content question — 用户问项目历史的具体内容（'why was BAB shipped' / "
    "'list amendments to spec_b_plus_mass_fdr_search.md' / 'when was regime "
    "overlay disabled'）。这类问题：必须严格基于 evidence 作答，每个 claim 引用 "
    "[C0]/[C1] 等具体 doc。如果 evidence 真的不沾边，evidence_quality=irrelevant + "
    "诚实告诉用户 'evidence 中没有直接证据回答此问题'，不要硬编。\n\n"
    "(B) Meta question — 用户问 console / 项目本身能干什么（'what can you do' / "
    "'你能帮我查什么' / 'how do I find spec amendments'）。这类问题：优先使用 "
    "evidence 中 source_type=system_help 的 doc 作答（它们就是为这类问题准备的 "
    "capability 描述）。如果 system_help evidence 命中，照常作答 + 引用对应 [C]。"
    "如果 evidence 全是 content 类（decision_log/spec/amendment 等）但用户问的是 meta，"
    "evidence_quality=irrelevant，answer 要明确说 'evidence 是关于具体决策内容的，跟你的"
    "问题不沾边' + 简要列出 console 可以查询的话题方向（spec amendments / pending "
    "approvals / past hypothesis verdicts / Reflexion lessons / audit findings 等），"
    "引导用户改问。\n\n"
    "答题格式硬规则（两类都适用）：\n"
    "1. 纯散文，禁止 markdown 装饰：不要用 ** 加粗、不要用 # 标题、不要用 - 列表、"
    "不要用 > 引用。强调用普通中文（'尤其是'/'关键是'）。\n"
    "2. Citation 节制：每个 claim 引用 1-2 个最具代表性的 source；禁止 dump 全部 "
    "citation（绝对不要写 [C0, C1, C2, C3, C4, C5, C6, C7] 这种全引用）。如果一个 "
    "观点跨多个 source，挑信息量最高的 1-2 个引用即可。\n"
    "3. 不要用 underscore 形式的内部标识符做行文用语（'rule_spec_hash_vs_code_drift' "
    "'spec_amendment'）；改写为自然语言（'spec 哈希漂移规则' '规范修订'）。\n"
    "4. answer 控制在 250 字以内（含中英），简洁、就事论事。\n"
    "5. evidence_quality 字段必须诚实反映 evidence 的真实匹配度："
    "strong / partial / thin / irrelevant —— 不许为了让答案显得有依据就虚报 strong。"
)


def _build_evidence_block(results: Sequence[RetrievalResult]) -> str:
    """Format retrieval hits into a numbered evidence block for the LLM."""
    lines: list[str] = []
    for i, r in enumerate(results):
        cid = f"C{i}"
        occurred = r.occurred_at.isoformat() if r.occurred_at else "unknown"
        # Cap each evidence chunk at 600 chars to keep total prompt bounded.
        text = r.text if len(r.text) <= 600 else (r.text[:600] + "…")
        lines.append(
            f"[{cid}] {r.title}\n"
            f"  source: {r.source_type.value}#{r.source_id}  occurred: {occurred}\n"
            f"  text:   {text}"
        )
    return "\n\n".join(lines)


def _build_prompt(query: str, results: Sequence[RetrievalResult]) -> str:
    """Assemble the user-side prompt (system instruction goes via Gemini config)."""
    return (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"=== Evidence ===\n"
        f"{_build_evidence_block(results)}\n\n"
        f"=== User question ===\n"
        f"{query.strip()}\n"
    )


# ── Main entry point ─────────────────────────────────────────────────────────
def synthesize_answer(
    query: str,
    retrieval_results: Sequence[RetrievalResult],
    *,
    daily_budget_usd:  float | None = None,    # None → read from llm_budget helper
    temperature:       float = SYNTHESIS_TEMPERATURE,
    max_output_tokens: int   = SYNTHESIS_MAX_TOKENS,
) -> SynthesizedAnswer:
    """Synthesize a cited answer from retrieved evidence.

    Args
    ----
    query : the user's question (mixed Chinese / English OK).
    retrieval_results : output of engine.agents.history_rag.retrieve().
    daily_budget_usd : refuse to call LLM if today's spend would exceed this.
    temperature : Gemini temperature (default low for grounded synthesis).
    max_output_tokens : output cap for the response.

    Behavior
    --------
    - Empty retrieval_results → status='no_evidence', no LLM call, $0.
    - Budget exhausted → status='budget_exhausted', no LLM call, $0.
    - LLM failure → status='llm_error', $0 charged for failed call.
    - LLM cites unknown doc → status='citation_invalid' but answer
      preserved for diagnosis (do not display to user).
    - All-clear → status='ok', citations resolved with deep_links.

    Budget resolution (2026-05-08):
      - If daily_budget_usd is None (default) → read from
        engine.llm_budget.get_rag_synthesis_daily_budget_usd() which is
        SystemConfig-backed (runtime tunable via System Console > LLM Budget).
      - If daily_budget_usd is explicit → use it (override; e.g. test harness).
    """
    # Resolve budget at call time so SystemConfig overrides take effect
    if daily_budget_usd is None:
        from engine.llm_budget import get_rag_synthesis_daily_budget_usd
        daily_budget_usd = get_rag_synthesis_daily_budget_usd()

    if not query or not query.strip():
        return SynthesizedAnswer(
            query=query, answer="(empty query)",
            status="no_evidence",
            budget_remaining=daily_budget_usd - _today_spend_usd(),
        )
    if not retrieval_results:
        return SynthesizedAnswer(
            query=query,
            answer="本项目历史索引中没有匹配此问题的证据。请考虑放宽 source 过滤、扩大日期范围，或换一种问法。",
            status="no_evidence",
            budget_remaining=daily_budget_usd - _today_spend_usd(),
        )

    today_spend = _today_spend_usd()
    if today_spend >= daily_budget_usd:
        return SynthesizedAnswer(
            query=query,
            answer=(
                f"今天 LLM synthesis 预算 ${daily_budget_usd:.2f} 已用完"
                f"（已花 ${today_spend:.4f}）。改用 retrieval-only 模式查看上方原始 evidence。"
            ),
            status="budget_exhausted",
            cost_usd=0.0,
            budget_remaining=0.0,
        )

    prompt = _build_prompt(query, retrieval_results)

    # ── Call Gemini ──────────────────────────────────────────────────────────
    try:
        from engine.key_pool import get_pool
        pool = get_pool()
        model = pool.get_model(
            model_name=SYNTHESIS_MODEL,
            response_schema=_SYNTH_RESPONSE_SCHEMA,
            temperature=temperature,
        )
        resp = model.generate_content(prompt)
        pool.report_success(has_content=True)
    except Exception as exc:
        logger.exception("history_rag.synthesize_answer: Gemini call failed")
        return SynthesizedAnswer(
            query=query,
            answer="LLM synthesis 失败；请回退到 retrieval-only 模式查看 evidence。",
            status="llm_error",
            error_msg=str(exc)[:200],
            budget_remaining=daily_budget_usd - today_spend,
        )

    raw_text = getattr(resp, "text", None) or str(resp)
    usage    = getattr(resp, "usage_metadata", None)
    in_tok   = int(getattr(usage, "prompt_token_count",     0) or 0)
    out_tok  = int(
        (getattr(usage, "candidates_token_count", 0) or 0)
        + (getattr(usage, "thoughts_token_count", 0) or 0)
    )
    cost = (in_tok * _COST_PER_1M_INPUT + out_tok * _COST_PER_1M_OUTPUT) / 1_000_000.0
    _record_call(cost)

    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("history_rag.synthesize_answer: JSON parse failed: %s", str(exc)[:120])
        return SynthesizedAnswer(
            query=query,
            answer=raw_text[:300] if raw_text else "(empty LLM response)",
            status="llm_error",
            cost_usd=cost,
            input_tokens=in_tok, output_tokens=out_tok,
            error_msg="response_schema_parse_failed",
            budget_remaining=daily_budget_usd - today_spend - cost,
        )

    # ── Citation verification ────────────────────────────────────────────────
    # Build label → RetrievalResult map. LLM cites by "C0" / "C1" / ...
    # If LLM cites a label that isn't in this map, it's a hallucination.
    valid_labels = {f"C{i}": r for i, r in enumerate(retrieval_results)}
    raw_cites    = parsed.get("citations") or []
    citations:   list[Citation] = []
    invalid_ids: list[str]      = []
    for c in raw_cites:
        if not isinstance(c, dict):
            continue
        cid   = str(c.get("citation_id", "")).strip()
        claim = str(c.get("claim", "")).strip()
        if cid not in valid_labels:
            invalid_ids.append(cid)
            continue
        ref = valid_labels[cid]
        citations.append(Citation(
            citation_id=cid,
            claim=claim,
            doc_id=ref.doc_id,
            source_type=ref.source_type,
            title=ref.title,
            deep_link=ref.deep_link,
        ))

    status = "ok"
    if invalid_ids:
        # Hallucinated citation = caller should display "evidence verification
        # failed" rather than the answer text. We keep the parsed answer for
        # diagnostic purposes but flag the status.
        logger.warning(
            "history_rag.synthesize_answer: %d invalid citations: %s",
            len(invalid_ids), invalid_ids,
        )
        status = "citation_invalid"

    return SynthesizedAnswer(
        query=query,
        answer=str(parsed.get("answer", "")).strip(),
        citations=citations,
        evidence_quality=str(parsed.get("evidence_quality", "irrelevant")),
        status=status,
        cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
        budget_remaining=daily_budget_usd - today_spend - cost,
        error_msg=(f"invalid_citations: {invalid_ids}" if invalid_ids else None),
    )
