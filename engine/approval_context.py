"""
P-AUDIT v1 — Supervisor Approval Audit Panel backend (deterministic context).

Spec: docs/spec_supervisor_approval_panel_v1.md (forward-registered 2026-05-04).

Contracts (3 public functions, all return JSON-friendly dicts/lists):

    get_approval_context(approval_id)        -> dict   # Tier 2 base
    get_similar_past_approvals(approval_id)  -> list   # 3b RAG-hybrid
    get_decision_replay(approval_id)         -> list   # 3a timeline

Hard rules:
  - 0 LLM enters this layer. All retrieval is deterministic SQL +
    sentence-transformer cosine (Layer 1 generation, Layer 2 ranking is
    deterministic dot product).
  - Returns are graceful: empty/insufficient inputs surface as "insufficient
    data" sentinels, not exceptions.

Cross-refs:
  - S2 retriever (engine/agents/reflection.py:retrieve_relevant_reflections)
  - SpecRegistry (engine/preregistration.py)
  - PendingApproval (engine/memory.py + P-AUDIT-1 review_rationale/category cols)
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any


REVIEW_CATEGORIES: tuple[str, ...] = (
    "signal_confirmed",
    "regime_driven",
    "supervisor_discretion",
    "risk_override",
    "cash_flow_routine",
    "other",
)
"""Fixed enum (D2 in spec). Free-text rationale is mandatory but the
category is constrained so post-hoc analytics group cleanly."""


_RATIONALE_MIN_CHARS = 10


def validate_review_inputs(rationale: str | None, category: str | None) -> tuple[bool, str]:
    """
    UI gate helper. Returns (ok, reason). ok=False blocks Approve/Reject buttons.
    """
    if not rationale or len(rationale.strip()) < _RATIONALE_MIN_CHARS:
        return (False, f"rationale ≥ {_RATIONALE_MIN_CHARS} chars required")
    if category not in REVIEW_CATEGORIES:
        return (False, f"category must be one of {REVIEW_CATEGORIES}")
    return (True, "ok")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 base — get_approval_context
# ─────────────────────────────────────────────────────────────────────────────

def _get_approval_context_cached(approval_id: int) -> dict:
    """
    Streamlit-cached entry point. Caching key = approval_id; TTL = 60s.
    Avoids re-running 7 SQL aggregations every chip switch in the
    multi-alert tabular workflow. Cache invalidates on:
      - 60s expiration
      - st.cache_data.clear() called from _apply_decision after resolve
    """
    try:
        import streamlit as _st
        @_st.cache_data(ttl=60, show_spinner=False)
        def _inner(_aid: int) -> dict:
            return get_approval_context(int(_aid))
        return _inner(int(approval_id))
    except Exception:
        # Outside Streamlit runtime (verify scripts, AppTest etc.) — passthrough
        return get_approval_context(int(approval_id))


def get_approval_context(approval_id: int, *, session: Any | None = None) -> dict:
    """
    Aggregate all 8 modules of context for one pending approval row.

    Returns dict with keys:
      base       - one-line action + spec + deadline
      cb_status  - circuit-breaker last-N events
      harking    - active HARKing flags relevant to the spec
      quant_ctx  - decision-time quant signal numbers (from linked DecisionLog)
      reject_preview - rule-based cascade preview if rejected
      decision_context - 7-layer decision support (M3-corrected-ext-full)

    Missing pieces are surfaced as None / empty list, never raise.
    UI callers that need cache: use `_get_approval_context_cached(id)`.
    """
    from engine.memory import (
        PendingApproval,
        SessionFactory,
        WatchlistEntry,
        DecisionLog,
        CircuitBreakerLog,
        HARKingFlag,
        SimulatedPosition,
        SpecRegistry,
    )

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        row: PendingApproval | None = sess.get(PendingApproval, int(approval_id))
        if row is None:
            return {"approval_id": int(approval_id), "found": False}

        wl = (
            sess.get(WatchlistEntry, int(row.watchlist_entry_id))
            if row.watchlist_entry_id is not None else None
        )

        decision = (
            sess.query(DecisionLog)
                .filter(DecisionLog.id == int(wl.decision_log_id))
                .first()
            if (wl is not None and wl.decision_log_id is not None) else None
        )
        if decision is None and row.sector and row.triggered_date:
            decision = (
                sess.query(DecisionLog)
                    .filter(DecisionLog.sector_name == row.sector)
                    .filter(DecisionLog.tab_type == "sector")
                    .filter(DecisionLog.decision_date <= row.triggered_date)
                    .order_by(DecisionLog.decision_date.desc())
                    .first()
            )

        spec = _resolve_governing_spec(sess, decision)
        cb_status = _summarize_cb_status(sess)
        harking = _active_harking_for_spec(sess, spec.get("path") if spec else None)
        quant_ctx = _quant_ctx_from_decision(decision)
        reject_preview = _reject_cascade_preview(sess, row, wl)

        # ── M3-corrected-ext-full DECISION CONTEXT (clarification 2026-05-04)
        # 7 layers + 5 EXT, all deterministic / 0 LLM. See spec § Amendment.
        decision_context = _build_decision_context(
            sess=sess, approval_row=row, watchlist=wl, decision=decision,
            quant_ctx_payload=quant_ctx,
        )

        days_left: int | None = None
        if row.approval_deadline is not None:
            today = datetime.date.today()
            days_left = (row.approval_deadline - today).days

        base = {
            "approval_id":            int(row.id),
            "approval_type":          row.approval_type,
            "priority":               row.priority,
            "status":                 row.status,
            "sector":                 row.sector,
            "ticker":                 row.ticker,
            "amount_or_weight":       row.suggested_weight,
            "triggered_condition":    row.triggered_condition,
            "triggered_date":         _date_to_str(row.triggered_date),
            "triggered_price":        row.triggered_price,
            "approval_deadline":      _date_to_str(row.approval_deadline),
            "deadline_days_left":     days_left,
            "governing_spec_path":    spec.get("path") if spec else None,
            "governing_spec_hash":    spec.get("hash") if spec else None,
            "last_amend_days":        spec.get("last_amend_days") if spec else None,
            "spec_excerpt_first_200_chars":
                                      spec.get("excerpt") if spec else None,
            "linked_decision_log_id": int(decision.id) if decision else None,
            "linked_watchlist_id":    int(wl.id) if wl else None,
            "contradicts_quant":      bool(row.contradicts_quant or False),
            "llm_confidence":         row.llm_confidence,
        }

        return {
            "found":             True,
            "approval_id":       int(row.id),
            "base":              base,
            "cb_status":         cb_status,
            "harking":           harking,
            "quant_ctx":         quant_ctx,
            "reject_preview":    reject_preview,
            "decision_context":  decision_context,   # M3-corrected-ext-full
        }
    finally:
        if own:
            sess.close()


def _build_decision_context(
    sess: Any,
    approval_row: Any,
    watchlist: Any,
    decision: Any,
    quant_ctx_payload: dict,
) -> dict:
    """
    Aggregate 7 layers into one dict. Each layer returns gracefully when
    inputs are missing — never raises. UI is expected to render conditionally
    on `available` flags inside each layer.
    """
    from engine import decision_context as dc

    sector = approval_row.sector if approval_row else None
    ticker = approval_row.ticker if approval_row else None
    sw = float(approval_row.suggested_weight or 0.0) if approval_row else 0.0
    direction = (
        watchlist.direction if (watchlist and watchlist.direction)
        else (decision.direction if decision else None)
    )

    l1 = dc.get_watchlist_origin(int(approval_row.id), session=sess)
    l2 = dc.get_quant_posture(ticker, sector, session=sess)
    l3 = dc.get_regime_context(
        ticker, sector, l1.get("created_date") if l1.get("available") else None,
        session=sess,
    )
    l4 = dc.get_portfolio_posture(int(approval_row.id), sw, session=sess)
    l5 = dc.get_conditional_history(
        sector, direction, l3.get("regime_label"), session=sess,
    )

    # If project ex-ante is insufficient (n<min_n), attempt historical replay
    # fallback (yfinance + walk-forward MSM regime proxy).
    # Anti-anchoring guards enforced UI-side in _tab_history.
    if direction is None and approval_row is not None:
        # Best-effort fallback: infer direction from suggested_weight sign
        sw_val = float(approval_row.suggested_weight or 0.0)
        if sw_val > 0:
            direction = "long"
        elif sw_val < 0:
            direction = "short"
    if l5.get("insufficient_data") and ticker and direction and l3.get("regime_label"):
        try:
            from engine.historical_replay import get_historical_conditional_hit_rate
            target_regime = l3.get("regime_label")
            # MSM walk-forward only outputs risk-on / risk-off / transition.
            # Historical replay accepts risk-on / neutral / risk-off — map.
            if target_regime == "transition":
                target_regime = "neutral"
            d_norm = "long" if direction in ("long", "超配") else "short" if direction in ("short", "低配") else None
            if d_norm is not None and target_regime in ("risk-on", "neutral", "risk-off"):
                replay = get_historical_conditional_hit_rate(
                    ticker=ticker,
                    direction=d_norm,
                    target_regime=target_regime,
                    horizon_days=21,
                    lookback_years=15,
                    regime_proxy="vix_simple",  # MSM walk-forward is heavy; use as default
                )
                l5["_replay_payload"] = replay
        except Exception:
            pass
    l6 = dc.compose_thesis(
        decision_log_payload=quant_ctx_payload,
        watchlist_origin=l1, quant_posture=l2, regime_context=l3,
    )
    l7a = dc.get_forward_preview(int(approval_row.id), sw, session=sess)

    return {
        "watchlist_origin":     l1,
        "quant_posture":        l2,
        "regime_context":       l3,
        "portfolio_posture":    l4,
        "conditional_history":  l5,
        "thesis_module":        l6,
        "forward_preview":      l7a,
    }


def _date_to_str(d: Any) -> str | None:
    if d is None:
        return None
    if isinstance(d, datetime.date):
        return d.isoformat()
    return str(d)


def _resolve_governing_spec(sess: Any, decision: Any) -> dict | None:
    """
    Return {path, hash, last_amend_days, excerpt} for the spec that governs
    this approval, or None when no governing spec is identifiable.

    Resolution order:
      1. decision.spec_hash → SpecRegistry.current_hash match
      2. fallback: docs/spec_sector_pipeline_unification.md (the live default
         for sector-pipeline-driven approvals)
    """
    from engine.memory import SpecRegistry

    candidate_hash = getattr(decision, "spec_hash", None) if decision else None
    spec_row = None
    if candidate_hash:
        spec_row = (
            sess.query(SpecRegistry)
                .filter(SpecRegistry.current_hash == candidate_hash)
                .first()
        )
    if spec_row is None:
        spec_row = (
            sess.query(SpecRegistry)
                .filter(SpecRegistry.spec_path.like("%sector_pipeline%"))
                .filter(SpecRegistry.status == "active")
                .first()
        )
    if spec_row is None:
        return None

    last_amend_days = None
    try:
        ledger = json.loads(spec_row.amendment_log or "[]")
        if ledger:
            last_at = ledger[-1].get("at")
            if last_at:
                last_dt = datetime.datetime.fromisoformat(
                    last_at.replace("Z", "+00:00")
                )
                last_amend_days = (
                    datetime.datetime.utcnow() - last_dt.replace(tzinfo=None)
                ).days
    except Exception:
        pass

    excerpt = _read_spec_excerpt(spec_row.spec_path)
    return {
        "path":            spec_row.spec_path,
        "hash":            (spec_row.current_hash or "")[:16],
        "last_amend_days": last_amend_days,
        "excerpt":         excerpt,
    }


def _read_spec_excerpt(rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full = os.path.join(repo_root, rel_path)
        if not os.path.exists(full):
            return None
        with open(full, "r", encoding="utf-8") as f:
            text = f.read(600)
        return text[:200]
    except Exception:
        return None


def _summarize_cb_status(sess: Any) -> dict:
    from engine.memory import CircuitBreakerLog

    last = (
        sess.query(CircuitBreakerLog)
            .order_by(CircuitBreakerLog.triggered_at.desc())
            .first()
    )
    last_severe = (
        sess.query(CircuitBreakerLog)
            .filter(CircuitBreakerLog.level == "severe")
            .order_by(CircuitBreakerLog.triggered_at.desc())
            .first()
    )

    return {
        "last_event_at":      last.triggered_at.isoformat() if last else None,
        "last_event_level":   last.level if last else None,
        "last_event_resolved": bool(last.resolved_at) if last else None,
        "last_severe_at":
            last_severe.triggered_at.isoformat() if last_severe else None,
        "manual_reset_at":
            last.resolved_at.isoformat()
            if (last and last.resolved_at and
                (last.resolved_by or "").lower() != "auto") else None,
    }


def _active_harking_for_spec(sess: Any, spec_path: str | None) -> list[dict]:
    from engine.memory import HARKingFlag

    if not spec_path:
        return []
    rows = (
        sess.query(HARKingFlag)
            .filter(HARKingFlag.spec_path == spec_path)
            .filter(HARKingFlag.resolved_at.is_(None))
            .order_by(HARKingFlag.detected_at.desc())
            .limit(5)
            .all()
    )
    return [
        {
            "rule":        r.rule,
            "severity":    r.severity,
            "detected_at": r.detected_at.isoformat() if r.detected_at else None,
            "notes":       r.notes,
        }
        for r in rows
    ]


def _quant_ctx_from_decision(decision: Any) -> dict:
    if decision is None:
        return {"available": False}
    debate_excerpt: str | None = None
    if getattr(decision, "debate_transcript", None):
        try:
            tx = json.loads(decision.debate_transcript or "[]")
            if isinstance(tx, list) and tx:
                last = tx[-1]
                if isinstance(last, dict):
                    debate_excerpt = (last.get("content") or "")[:300]
                else:
                    debate_excerpt = str(last)[:300]
            elif isinstance(tx, dict):
                debate_excerpt = json.dumps(tx)[:300]
        except Exception:
            debate_excerpt = (decision.debate_transcript or "")[:300]

    return {
        "available":            True,
        "decision_id":          int(decision.id),
        "decision_date":        _date_to_str(decision.decision_date),
        "direction":            decision.direction,
        "confidence_score":     decision.confidence_score,
        "macro_regime":         decision.macro_regime,
        "quant_p_noise":        decision.quant_p_noise,
        "quant_val_r2":         decision.quant_val_r2,
        "quant_test_r2":        decision.quant_test_r2,
        "weight_adjustment_pct": decision.weight_adjustment_pct,
        "weight_before":        decision.weight_before,
        "weight_after":         decision.weight_after,
        "key_thesis":           decision.key_thesis,
        "primary_risk":         decision.primary_risk,
        "debate_summary_excerpt": debate_excerpt,
    }


def _reject_cascade_preview(sess: Any, row: Any, wl: Any) -> dict:
    """
    Rule-based estimate of what gets affected if this approval is rejected.
    No simulation: just count downstream rows impacted under fixed rules.
    """
    from engine.memory import SimulatedPosition

    sectors_affected: list[str] = []
    position_value_at_risk = 0.0
    watchlist_invalidated = 0
    downstream_actions: list[str] = []

    if row.approval_type == "entry":
        sectors_affected = [row.sector] if row.sector else []
        watchlist_invalidated = 1 if wl is not None else 0
        downstream_actions = ["WatchlistEntry → status=watching (revert)"]
    elif row.approval_type == "risk_control":
        latest_pos_date = (
            sess.query(SimulatedPosition.snapshot_date)
                .filter(SimulatedPosition.track == "main")
                .order_by(SimulatedPosition.snapshot_date.desc())
                .first()
        )
        if latest_pos_date is not None:
            positions = (
                sess.query(SimulatedPosition)
                    .filter(SimulatedPosition.snapshot_date == latest_pos_date[0])
                    .filter(SimulatedPosition.track == "main")
                    .filter(SimulatedPosition.sector == row.sector)
                    .all()
            )
            sectors_affected = list({p.sector for p in positions})
            position_value_at_risk = float(sum(
                (p.position_value or 0.0) for p in positions
            ))
        downstream_actions = ["RiskOverrideLog write + position retained"]
    elif row.approval_type == "rebalance":
        downstream_actions = ["Rebalance skipped → drift carries to next cycle"]
    elif row.approval_type == "cash_flow":
        downstream_actions = ["CashFlow → status=cancelled; NAV unchanged"]
    else:
        downstream_actions = ["No-op"]

    return {
        "approval_type":             row.approval_type,
        "sectors_affected":          sectors_affected,
        "position_value_at_risk":    round(position_value_at_risk, 2),
        "watchlist_invalidated_count": watchlist_invalidated,
        "downstream_actions":        downstream_actions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3b — get_similar_past_approvals (rule-based + S2 RAG hybrid)
# ─────────────────────────────────────────────────────────────────────────────

def get_similar_past_approvals(
    approval_id: int,
    top_k: int = 3,
    *,
    session: Any | None = None,
) -> list[dict]:
    """
    Top-K most similar past PendingApproval rows (already resolved). Ordered
    by recency × similarity. Each item carries retrieval_method telling the
    UI whether match came from rule-based filter, S2 RAG, or both.

    Algorithm (D4 in spec):
      1. Rule-based candidates: same sector & approval_type, status in
         (approved, rejected), resolved within last 720 days, exclude self.
      2. S2 RAG candidates: build query string from approval row, call
         retrieve_relevant_reflections(agent_id="sector_pipeline", k=2*top_k).
         Map each retrieved reflection to its decision_ref_id → linked
         WatchlistEntry → PendingApproval (best-effort).
      3. Merge: union ⇒ score = rule_match_bonus + recency_decay; keep top_k.
    """
    from engine.memory import (
        PendingApproval, WatchlistEntry, DecisionLog, SessionFactory,
    )

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        row = sess.get(PendingApproval, int(approval_id))
        if row is None:
            return []

        cutoff = datetime.date.today() - datetime.timedelta(days=720)

        rule_rows = (
            sess.query(PendingApproval)
                .filter(PendingApproval.id != row.id)
                .filter(PendingApproval.sector == row.sector)
                .filter(PendingApproval.approval_type == row.approval_type)
                .filter(PendingApproval.status.in_(("approved", "rejected")))
                .filter(PendingApproval.triggered_date >= cutoff)
                .order_by(PendingApproval.triggered_date.desc())
                .limit(top_k * 4)
                .all()
        )

        rag_pa_ids: set[int] = set()
        try:
            from engine.agents.reflection import retrieve_relevant_reflections

            query_text = " | ".join([
                str(row.sector or ""),
                str(row.triggered_condition or ""),
                str(row.approval_type or ""),
            ]).strip()
            reflections = retrieve_relevant_reflections(
                agent_id="sector_pipeline",
                query_text=query_text,
                k=top_k * 2,
                session=sess,
            ) if query_text else []
            for refl in reflections:
                dec_id = getattr(refl, "decision_ref_id", None)
                if dec_id is None:
                    continue
                wl = (
                    sess.query(WatchlistEntry)
                        .filter(WatchlistEntry.decision_log_id == int(dec_id))
                        .first()
                )
                if wl is None:
                    continue
                pa_match = (
                    sess.query(PendingApproval)
                        .filter(PendingApproval.watchlist_entry_id == wl.id)
                        .filter(PendingApproval.id != row.id)
                        .first()
                )
                if pa_match is not None:
                    rag_pa_ids.add(int(pa_match.id))
        except Exception:
            rag_pa_ids = set()

        merged: dict[int, dict] = {}
        today = datetime.date.today()

        def _score(pa_row: Any, methods: list[str]) -> float:
            base = 1.0 if "rule_based" in methods else 0.0
            base += 0.6 if "rag" in methods else 0.0
            if pa_row.triggered_date is not None:
                days_old = max(1, (today - pa_row.triggered_date).days)
                base += 1.0 / (1.0 + days_old / 90.0)
            return base

        for pa in rule_rows:
            methods = ["rule_based"]
            if int(pa.id) in rag_pa_ids:
                methods.append("rag")
            merged[int(pa.id)] = _emit_similar_row(sess, pa, methods, _score(pa, methods))

        for rid in rag_pa_ids - {int(p.id) for p in rule_rows}:
            pa = sess.get(PendingApproval, int(rid))
            if pa is None:
                continue
            methods = ["rag"]
            merged[int(pa.id)] = _emit_similar_row(sess, pa, methods, _score(pa, methods))

        ordered = sorted(merged.values(), key=lambda d: d["_score"], reverse=True)[:top_k]
        for d in ordered:
            d.pop("_score", None)
        return ordered
    finally:
        if own:
            sess.close()


def _emit_similar_row(sess: Any, pa: Any, methods: list[str], score: float) -> dict:
    from engine.memory import WatchlistEntry, DecisionLog

    decision = None
    if pa.watchlist_entry_id is not None:
        wl = sess.get(WatchlistEntry, int(pa.watchlist_entry_id))
        if wl is not None and wl.decision_log_id is not None:
            decision = sess.get(DecisionLog, int(wl.decision_log_id))

    active_return = decision.active_return if decision else None
    accuracy = decision.accuracy_score if decision else None
    hit_flag = "hit" if (accuracy is not None and accuracy >= 0.5) else (
        "miss" if accuracy is not None else "pending"
    )

    return {
        "approval_id":      int(pa.id),
        "decision_id":      int(decision.id) if decision else None,
        "decision_date":    _date_to_str(decision.decision_date) if decision else None,
        "approval_date":    _date_to_str(pa.triggered_date),
        "sector":           pa.sector,
        "ticker":           pa.ticker,
        "direction":        decision.direction if decision else None,
        "amount":           pa.suggested_weight,
        "verdict":          pa.status,
        "review_category":  pa.review_category,
        "review_rationale": (pa.review_rationale or "")[:120] or None,
        "active_return":    active_return,
        "hit_flag":         hit_flag,
        "retrieval_method": "+".join(methods),
        "_score":           score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3a — get_decision_replay (deterministic timeline reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

def get_decision_replay(approval_id: int, *, session: Any | None = None) -> list[dict]:
    """
    Reconstruct event chain that led to the approval. Pure SQL aggregation
    over AgentRun + AgentEventRow + DecisionLog + WatchlistEntry +
    PendingApproval timestamps. Each step is one dict with:

        ts             ISO8601 string
        type           e.g. "trigger" / "spec_lookup" / "llm_debate" /
                       "quant_audit" / "approval_gate_created"
        actor          "system" / "rule" / "llm" / "quant"
        payload_summary short str
        run_id_link    AgentRun.run_id when sourceable, else None
        reconstructed  True when timestamp is reverse-inferred from adjacent
                       rows (visible to UI as "[reconstructed]")
    """
    from engine.memory import (
        PendingApproval, WatchlistEntry, DecisionLog, SessionFactory,
    )
    # AgentRun/AgentEventRow were an older agent event-bus schema since removed from
    # engine.memory (signature-drift fix 2026-05-24). The replay reconstructs fully from
    # DecisionLog + WatchlistEntry + PendingApproval timestamps; the optional event-bus steps
    # are skipped when that table is absent. (AgentRun was imported but never used — dropped.)
    try:
        from engine.memory import AgentEventRow  # type: ignore
    except Exception:
        AgentEventRow = None

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        row = sess.get(PendingApproval, int(approval_id))
        if row is None:
            return []

        wl = sess.get(WatchlistEntry, int(row.watchlist_entry_id)) \
            if row.watchlist_entry_id is not None else None

        decision = (
            sess.get(DecisionLog, int(wl.decision_log_id))
            if (wl is not None and wl.decision_log_id is not None) else None
        )

        steps: list[dict] = []

        if decision is not None and decision.created_at is not None:
            steps.append({
                "ts":              decision.created_at.isoformat(),
                "type":            "decision_logged",
                "actor":           "system",
                "payload_summary": f"DecisionLog.id={decision.id} "
                                   f"sector={decision.sector_name} "
                                   f"direction={decision.direction}",
                "run_id_link":     None,
                "reconstructed":   False,
            })

            if decision.debate_transcript:
                steps.append({
                    "ts":              decision.created_at.isoformat(),
                    "type":            "llm_debate",
                    "actor":           "llm",
                    "payload_summary": _excerpt_debate(decision.debate_transcript),
                    "run_id_link":     None,
                    "reconstructed":   True,
                })

            if (decision.quant_p_noise is not None
                    or decision.quant_val_r2 is not None):
                steps.append({
                    "ts":              decision.created_at.isoformat(),
                    "type":            "quant_audit",
                    "actor":           "quant",
                    "payload_summary":
                        f"p_noise={decision.quant_p_noise} "
                        f"val_r2={decision.quant_val_r2} "
                        f"weight_adj={decision.weight_adjustment_pct}",
                    "run_id_link":     None,
                    "reconstructed":   True,
                })

            if decision.spec_hash:
                steps.append({
                    "ts":              decision.created_at.isoformat(),
                    "type":            "spec_lookup",
                    "actor":           "rule",
                    "payload_summary":
                        f"spec_hash={decision.spec_hash[:16]}",
                    "run_id_link":     None,
                    "reconstructed":   True,
                })

        if wl is not None and wl.created_date is not None:
            steps.append({
                "ts":              datetime.datetime.combine(
                                       wl.created_date,
                                       datetime.time(0, 0, 0)).isoformat(),
                "type":            "watchlist_created",
                "actor":           "system",
                "payload_summary": f"WatchlistEntry.id={wl.id} "
                                   f"status={wl.status} "
                                   f"weight={wl.suggested_weight}",
                "run_id_link":     None,
                "reconstructed":   False,
            })
            if wl.triggered_date is not None:
                steps.append({
                    "ts":              datetime.datetime.combine(
                                           wl.triggered_date,
                                           datetime.time(0, 0, 0)).isoformat(),
                    "type":            "trigger_fired",
                    "actor":           "rule",
                    "payload_summary": f"price={wl.triggered_price} "
                                       f"signal={wl.entry_tsmom_signal}",
                    "run_id_link":     None,
                    "reconstructed":   False,
                })

        if row.created_at is not None:
            steps.append({
                "ts":              row.created_at.isoformat(),
                "type":            "approval_gate_created",
                "actor":           "system",
                "payload_summary": f"approval_type={row.approval_type} "
                                   f"priority={row.priority} "
                                   f"deadline={_date_to_str(row.approval_deadline)}",
                "run_id_link":     None,
                "reconstructed":   False,
            })

        if AgentEventRow is not None and decision is not None and decision.created_at is not None:
            try:
                bus_rows = (
                    sess.query(AgentEventRow)
                        .filter(AgentEventRow.occurred_at
                                >= decision.created_at - datetime.timedelta(hours=2))
                        .filter(AgentEventRow.occurred_at
                                <= (row.created_at or datetime.datetime.utcnow())
                                    + datetime.timedelta(hours=2))
                        .order_by(AgentEventRow.occurred_at.asc())
                        .limit(10)
                        .all()
                )
                for ev in bus_rows:
                    steps.append({
                        "ts":              ev.occurred_at.isoformat(),
                        "type":            f"event:{ev.event_type}",
                        "actor":           ev.source_agent or "system",
                        "payload_summary":
                            (ev.payload or "")[:160],
                        "run_id_link":     None,
                        "reconstructed":   False,
                    })
            except Exception:
                pass

        if row.resolved_at is not None:
            steps.append({
                "ts":              row.resolved_at.isoformat(),
                "type":            f"resolved:{row.status}",
                "actor":           row.resolved_by or "system",
                "payload_summary":
                    f"category={row.review_category or '(none)'} "
                    f"rationale={(row.review_rationale or '')[:80]!r}",
                "run_id_link":     None,
                "reconstructed":   False,
            })

        steps.sort(key=lambda s: s["ts"])
        return steps
    finally:
        if own:
            sess.close()


def _excerpt_debate(transcript: str | None) -> str:
    if not transcript:
        return ""
    try:
        tx = json.loads(transcript)
        if isinstance(tx, list) and tx:
            last = tx[-1]
            if isinstance(last, dict):
                return (last.get("content") or "")[:160]
            return str(last)[:160]
        if isinstance(tx, dict):
            return json.dumps(tx)[:160]
    except Exception:
        pass
    return transcript[:160]
