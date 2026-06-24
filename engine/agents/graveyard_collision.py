"""engine.agents.graveyard_collision — second autonomous reactive agent.

Subscribes to the EventBus event `intent_filed` (published when a
research_test intent is filed via /api/intents/file). For every such
intent, scans the RED lessons graveyard for collisions with the
proposed candidate's family + name + claim, and writes a typed warning
row to data/graveyard_collision/warnings.jsonl.

Why this exists
---------------
User pushback on "现在这个项目太庞大了" — concretely the "我会不会重复
测一个 RED" anxiety. The graveyard surfaces RED verdicts but only on
the candidate page (after the test was approved). By then the user
already committed mental energy to the candidate.

This agent runs at intent-file time (the moment PM approval translates
into "Claude, go test this") so collisions surface BEFORE Claude / the
pipeline burns cycles on a candidate that's already been killed in
substance.

Collision dimensions (3 separate scores, soft-thresholded)
---------------------------------------------------------
S1  same family + name-token overlap ≥ 2 tokens (high signal)
S2  same family + same mechanism_subtype (very high signal — usually
    a deliberate re-test, but worth flagging)
S3  same family + claim-text trigram overlap > 0.30

Verdict:
  CLEAN — no collision dimension fires
  WARN  — exactly one fires; surface but don't block
  RISK  — 2+ fire; the user should re-read the RED lesson before
          spending Claude cycles. (Still doesn't BLOCK — agent is
          advisory, not gate. Compliance with the warning is the
          user's judgement.)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import datetime as _dt
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT       = Path(__file__).resolve().parent.parent.parent
_WARNINGS_DIR    = _REPO_ROOT / "data" / "graveyard_collision"
_WARNINGS_FILE   = _WARNINGS_DIR / "warnings.jsonl"
_WRITE_LOCK      = threading.Lock()

# Tokens shorter than this are too generic to overlap usefully (e.g. "of", "the")
_TOKEN_MIN_LEN     = 4
_NAME_OVERLAP_BAR  = 2
_CLAIM_TRIGRAM_BAR = 0.30

# T4.5 / T4.6 (2026-06-05): semantic dim S4 thresholds.
# Recalibrated downward after T4.6 empirical run: MiniLM (general-
# purpose L6-v2) only hit 0.27-0.37 cosine for "BAB factor" claim
# vs the 68-RED corpus including K1_BAB. The model isn't strong
# enough on financial-acronym semantics to clear 0.55. Lowering
# the WARN bar to 0.45 accepts more false positives (better to ask
# the reviewer "look at this RED" than to silently miss). RISK bar
# at 0.65 still requires genuinely strong overlap.
# A future R8-tier embedding swap (FinBERT, e5-large, or domain-
# tuned) should let us raise these back to the original 0.55/0.72.
_SEM_WARN_BAR   = 0.45    # >= 0.45 -> S4 fires (WARN-level signal alone)
_SEM_RISK_BAR   = 0.65    # >= 0.65 -> S4 counts as 2 dims (RISK by itself)

# RED-summary embedding cache: {event_id -> 384-d float32 vec}. Reset
# on import; rebuilt lazily as RED lessons are encountered. RED set
# changes rarely (a new RED is a manual emit), so the cache hit rate
# is very high after warmup.
_RED_EMBED_CACHE: "dict[str, object]" = {}


# ── Utilities ──────────────────────────────────────────────────────


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _tokenize(text: str) -> set[str]:
    """Cheap word tokenizer. Filters short stop-word-ish tokens."""
    if not text:
        return set()
    raw = text.lower().replace("/", " ").replace("-", " ").replace("_", " ")
    return {w for w in raw.split() if len(w) >= _TOKEN_MIN_LEN}


# T4.6 (2026-06-05) S5 helper: extract "meaningful" tokens from a
# RED subject_id for direct substring match in candidate claim. Catches
# the BAB <-> "Betting-Against-Beta (BAB)" case the semantic embedding
# misses (general MiniLM doesn't connect financial acronyms strongly).
_ACRONYM_STOPWORDS = {
    "v1", "v2", "v3", "v4", "v5", "v6",
    "k1", "k2", "k3", "k4", "ge2", "ge3", "ge4",
    "ls", "longonly", "shortonly", "loadall", "alpha",
    "hold1", "hold3", "hold6", "hold12",
    "form4", "form13f",
    "default", "design", "filter", "drift", "decay",
    "ibes", "rev", "comp", "expanded", "change",
    "q5", "d10",
    "monthly", "daily", "weekly",
    "test", "candidate", "factor",
}


def _meaningful_name_tokens(name: str) -> set[str]:
    """Split a subject_id like 'K1_BAB' / 'insider_clust3_hold6' /
    'g10_xc_carry_em_extension' into the substantive identifier tokens
    (BAB / clust3 / xc / em / extension / etc.). Filters version
    suffixes and structural artifacts so the substring check below
    fires on actual factor mentions, not 'v1' coincidences."""
    out: set[str] = set()
    raw = name.lower().replace("-", "_").split("_")
    for t in raw:
        t = t.strip()
        if not t or t in _ACRONYM_STOPWORDS:
            continue
        if len(t) < 2:
            continue
        # Skip purely-numeric tokens
        if t.isdigit():
            continue
        out.add(t)
    return out


def _trigrams(text: str) -> set[str]:
    s = (text or "").lower()
    if len(s) < 3:
        return set()
    return {s[i:i+3] for i in range(len(s) - 2)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _append_warning(row: dict) -> None:
    _WARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=str)
    with _WRITE_LOCK:
        with _WARNINGS_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ── Core checker ───────────────────────────────────────────────────


def check_collision(
    candidate_name:    str,
    family:            Optional[str],
    mechanism_subtype: Optional[str],
    claim_text:        Optional[str],
) -> dict:
    """Pure function: returns dict with verdict + per-dimension scores +
    matching RED lessons. Callable directly for testing / from the UI
    pre-warn flow.

    Does NOT write the warnings.jsonl row — that's the subscriber's job.
    """
    if not family:
        return {
            "verdict": "CLEAN",
            "reason":  "no family supplied; cannot scan graveyard",
            "matches": [],
        }
    try:
        from engine.research_store import store
        red_lessons = store.filter_events(
            event_type="factor_verdict_filed",
            family=family,
            verdict="RED",
            limit=200,
        )
        # T4.5 (2026-06-05): family taxonomy mismatch — RED events in
        # research_store use legacy family names (path_aa, position_weighting,
        # macro, ...) while forward_vectors use FamilyV2 enum (LOW_VOL,
        # CARRY, ...). When family-filter returns empty, fall back to
        # scanning ALL RED events and rely on semantic S4 to find
        # cross-taxonomy collisions (e.g. a "BAB" claim under family=LOW_VOL
        # vs a "betting-against-beta residualized" RED under family=path_x).
        scan_scope = f"family={family}"
        if not red_lessons:
            red_lessons = store.filter_events(
                event_type="factor_verdict_filed",
                verdict="RED",
                limit=200,
            )
            scan_scope = (f"all-family fallback (no RED on family={family}; "
                          f"semantic S4 picks cross-taxonomy collisions)")
    except Exception as exc:
        logger.exception("graveyard_collision: store filter failed")
        return {
            "verdict": "CLEAN",
            "reason":  f"store_unreachable:{exc}",
            "matches": [],
        }

    if not red_lessons:
        return {
            "verdict": "CLEAN",
            "reason":  f"no RED verdicts in store at all (scope={scan_scope})",
            "matches": [],
        }

    cand_tokens   = _tokenize(candidate_name)
    cand_trigrams = _trigrams(claim_text or "")

    # T4.5 (2026-06-05): embed the candidate claim once (or 0 times if
    # claim_text empty / embeddings unavailable). Encode RED summaries
    # on-demand into the module cache. semantic_available=False falls
    # back to S1+S2+S3 only — no behavior regression vs pre-T4.5.
    cand_vec, semantic_available = _embed_or_none(claim_text or "")

    matches = []
    for L in red_lessons:
        red_name      = L.subject_id or ""
        red_subtype   = (L.metrics or {}).get("mechanism_subtype")
        red_summary   = L.summary or ""

        red_tokens   = _tokenize(red_name)
        red_trigrams = _trigrams(red_summary)

        # S1: name-token overlap
        overlap = len(cand_tokens & red_tokens)
        s1_hit  = overlap >= _NAME_OVERLAP_BAR

        # S2: subtype match
        s2_hit  = (mechanism_subtype is not None
                   and red_subtype is not None
                   and mechanism_subtype == red_subtype)

        # S3: claim trigram similarity
        sim     = _jaccard(cand_trigrams, red_trigrams)
        s3_hit  = sim > _CLAIM_TRIGRAM_BAR

        # T4.5 S4: semantic cosine similarity (MiniLM embeddings)
        sem_sim = 0.0
        s4_hit_warn = False
        s4_hit_risk = False
        if semantic_available:
            # T4.6 (2026-06-05): embed "subject_id + summary" not just
            # summary. Many RED events (especially T4.6 backfills from
            # factory_ledger) have STATISTICAL summaries ("deflated SR
            # collapses ...") that don't carry the factor's mechanism
            # name. Prepending the subject_id ("K1_BAB ...") gives the
            # MiniLM vector a fighting chance to match a candidate
            # claim that mentions "betting-against-beta" by name.
            red_embed_text = (red_name + ". " + red_summary).strip()
            if red_embed_text:
                red_vec = _red_embed_cached(L.event_id, red_embed_text)
                if red_vec is not None:
                    sem_sim = float(_cosine(cand_vec, red_vec))
                    s4_hit_warn = sem_sim >= _SEM_WARN_BAR
                    s4_hit_risk = sem_sim >= _SEM_RISK_BAR

        # T4.6 S5: meaningful-name substring match. Each substantive
        # token in the RED's subject_id that appears literally in the
        # candidate claim contributes a hit. Catches BAB <-> "BAB"
        # type acronyms that semantic embedding misses on a general
        # model. Threshold: >=1 substantive token match. The token
        # extractor filters version suffixes / structural artifacts.
        claim_low = (claim_text or "").lower()
        name_toks = _meaningful_name_tokens(red_name)
        # Word-boundary match: avoid "ba" matching inside "abandon"
        import re as _re
        s5_overlap = sum(1 for t in name_toks
                          if _re.search(rf"\b{_re.escape(t)}\b", claim_low))
        s5_hit = s5_overlap >= 1

        if s1_hit or s2_hit or s3_hit or s4_hit_warn or s5_hit:
            # Each fired dim contributes 1, except S4_risk contributes 2
            # (a high semantic match is itself enough to escalate)
            s4_dims = 2 if s4_hit_risk else (1 if s4_hit_warn else 0)
            matches.append({
                "red_event_id":    L.event_id,
                "red_candidate":   red_name,
                "red_subtype":     red_subtype,
                "red_summary":     red_summary[:160],
                "s1_name_overlap": overlap,
                "s2_subtype_eq":   s2_hit,
                "s3_claim_sim":    round(sim, 3),
                "s4_semantic_sim": round(sem_sim, 3),
                "s5_name_substr":  s5_overlap,
                "dims_fired":      sum([s1_hit, s2_hit, s3_hit, s5_hit]) + s4_dims,
            })

    if not matches:
        return {
            "verdict": "CLEAN",
            "reason":  (f"scanned {len(red_lessons)} RED lessons on family={family}, "
                        f"no collision (semantic={'on' if semantic_available else 'off'})"),
            "matches": [],
            "semantic_enabled": semantic_available,
        }

    # Worst case across all matches drives the verdict
    max_dims = max(m["dims_fired"] for m in matches)
    if max_dims >= 2:
        verdict = "RISK"
    else:
        verdict = "WARN"

    # Sort matches strongest-first so UI can show top 3
    matches.sort(key=lambda m: (m["dims_fired"], m.get("s4_semantic_sim", 0), m["s3_claim_sim"]),
                 reverse=True)
    return {
        "verdict":   verdict,
        "reason":    (f"{len(matches)} RED match(es), worst fires {max_dims} dims "
                      f"(semantic={'on' if semantic_available else 'off'})"),
        "matches":   matches[:5],
        "n_scanned": len(red_lessons),
        "semantic_enabled": semantic_available,
    }


# ── T4.5 semantic helpers ──────────────────────────────────────────


def _embed_or_none(text: str):
    """Encode `text` to a 384-d MiniLM vec. Returns (vec, True) on
    success, (None, False) on any failure (model not installed,
    text empty, etc.). Defensive: a semantic-path failure must NEVER
    break the collision check — we just fall back to S1-S3."""
    if not text or not text.strip():
        return None, False
    try:
        from engine.research.embeddings import encode
        vec = encode(text)
        return vec, True
    except Exception as exc:
        logger.warning("graveyard_collision: semantic disabled — encode failed: %s", exc)
        return None, False


def _red_embed_cached(event_id: str, summary: str):
    """Cached encode for a RED-lesson summary. Module-level cache hits
    after the first call per event_id; misses encode once + populate."""
    if event_id in _RED_EMBED_CACHE:
        return _RED_EMBED_CACHE[event_id]
    try:
        from engine.research.embeddings import encode
        vec = encode(summary)
        _RED_EMBED_CACHE[event_id] = vec
        return vec
    except Exception:
        return None


def _cosine(a, b) -> float:
    """Cosine similarity. Both inputs are L2-normalized by MiniLM so
    cosine = dot product. Falls back to manual norm if not normalized."""
    import numpy as _np
    a = _np.asarray(a).ravel()
    b = _np.asarray(b).ravel()
    dot = float(_np.dot(a, b))
    # When normalized, dot is already cosine. Defensive normalize for safety.
    na = float(_np.linalg.norm(a))
    nb = float(_np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    # If both are roughly unit norm, dot is fine; else divide.
    if 0.95 <= na <= 1.05 and 0.95 <= nb <= 1.05:
        return dot
    return dot / (na * nb)


# ── EventBus handler ───────────────────────────────────────────────


def handle_intent_filed(agent_event) -> dict:
    """Subscriber for event_type=intent_filed.

    Payload shape (set by routes_intents publishing call):
      {
        "intent_id": str,
        "kind":      str,      # only research_test / pipeline_test relevant
        "subject_type": str,
        "subject_id":   str,
        "payload":      dict,  # includes mechanism_family, claim, etc
      }
    """
    payload = getattr(agent_event, "payload", {}) or {}
    kind = payload.get("kind", "")
    # Only research_test / pipeline_test signals "we're about to spend
    # Claude cycles on a candidate". Other kinds (audit, doctrine) are
    # not graveyard-comparable.
    if kind not in ("research_test", "pipeline_test"):
        return {"verdict": "SKIP", "reason": f"kind={kind} not graveyard-relevant"}

    intent_payload = payload.get("payload", {}) or {}
    candidate_name    = (intent_payload.get("proposal_name")
                         or intent_payload.get("subject_id")
                         or payload.get("subject_id", ""))
    family            = (intent_payload.get("mechanism_family")
                         or intent_payload.get("family")
                         or "")
    mechanism_subtype = intent_payload.get("mechanism_subtype")
    claim_text        = (intent_payload.get("claim")
                         or intent_payload.get("ask")
                         or "")

    result = check_collision(
        candidate_name    = candidate_name,
        family            = family,
        mechanism_subtype = mechanism_subtype,
        claim_text        = claim_text,
    )

    row = {
        "warning_id":     f"gc_{payload.get('intent_id', '')[:8]}_{int(_dt.datetime.utcnow().timestamp())}",
        "checked_ts":     _utc_iso(),
        "intent_id":      payload.get("intent_id"),
        "candidate_name": candidate_name,
        "family":         family,
        "subtype":        mechanism_subtype,
        "verdict":        result["verdict"],
        "reason":         result.get("reason"),
        "matches":        result.get("matches", []),
        "n_scanned":      result.get("n_scanned", 0),
        "agent":          "engine.agents.graveyard_collision",
        "agent_version":  1,
    }
    try:
        _append_warning(row)
    except Exception:
        logger.exception("graveyard_collision: failed to write warning row")

    logger.info("graveyard_collision: intent=%s family=%s -> %s",
                (payload.get("intent_id") or "")[:8], family, result["verdict"])
    return row


# ── Subscription ───────────────────────────────────────────────────


_SUBSCRIBED = False


def subscribe_to_bus() -> None:
    global _SUBSCRIBED
    if _SUBSCRIBED:
        return
    try:
        from engine.agents.event_bus import get_event_bus
        bus = get_event_bus()
        bus.subscribe("intent_filed", handle_intent_filed)
        _SUBSCRIBED = True
        logger.info("graveyard_collision: subscribed to intent_filed")
    except Exception as exc:
        logger.warning("graveyard_collision: subscribe_to_bus failed: %s", exc, exc_info=True)


if os.environ.get("GRAVEYARD_COLLISION_NO_AUTOSUBSCRIBE") != "1":
    subscribe_to_bus()
