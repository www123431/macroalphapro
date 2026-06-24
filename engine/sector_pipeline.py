"""
Sector Pipeline — single source of truth for sector debate execution.

P0-0 (2026-05-02): Unifies sector debate ETL + decision-write across the
three trigger paths (UI Tab 2, orchestrator signal-flip, daily_batch
auto-debate), eliminating training-serving skew. See
docs/spec_sector_pipeline_unification.md.

Two pure functions:

    prepare_sector_inputs(sector, t_day, vix) -> dict
        Pure ETL. Pulls news + macro + valuation + quant + quant_gate +
        historical context + state_vector. No DB writes. No session_state.

    run_sector_pipeline(model, sector, t_day, vix, decision_source, ...) -> dict
        Full pipeline: prepare_sector_inputs → run_sector_debate →
        run_quant_coherence_check → confidence-scaled adjustment →
        save_decision → supersede / PendingApproval.

Note (regime overlay decoupling): the SimulatedPosition write triggered by
save_decision (engine/memory._upsert_simulated_position_from_decision) sets
target_weight = base_weight + adj, clamped to [-0.20, 0.20]. It does not
pass through engine.portfolio.construct_portfolio nor reference
REGIME_SCALE, so the 2026-05-02 overlay-disabled baseline (REGIME_SCALE=1.0)
has no effect on this path.
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Maps the UI-style regime label (returned by _infer_macro_regime) to the
# risk-on / transition / risk-off bucket that get_quant_gates expects.
_REGIME_LABEL_MAP = {
    "低波动/牛市": "risk-on",
    "温和波动":   "risk-on",
    "震荡期":     "transition",
    "高波动/危机": "risk-off",
}

# Markers that indicate debate failed due to API quota / rate-limit. We must
# never persist these as decisions — they pollute Clean Zone stats and feed
# garbage into Memory Curator's pattern extraction.
_DEBATE_QUOTA_MARKERS = ("生成失败:", "429", "RESOURCE_EXHAUSTED", "quota", "Quota")


def _debate_output_is_error(text: str) -> bool:
    if not text or not text.strip():
        return True
    return any(m in text for m in _DEBATE_QUOTA_MARKERS)


def _infer_macro_regime(vix: float) -> str:
    """Replicate ui.tabs._infer_macro_regime to keep engine layer UI-free."""
    if vix >= 30:
        return "高波动/危机"
    if vix >= 20:
        return "震荡期"
    if vix >= 15:
        return "温和波动"
    return "低波动/牛市"


def _get_secret(name: str) -> str:
    """Env first, then st.secrets fallback (mirrors engine.news_fetcher pattern)."""
    val = os.environ.get(name, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name, "")
    except Exception:
        return ""


def _build_macro_context(t_day: datetime.date) -> str:
    """
    Single source of truth for sector debate macro context.

    Reads DailyBriefSnapshot first (same field that agent.py:_build_macro_only_context
    consumes). Falls back to empty string. UI's session_state.macro_memo is no
    longer consulted here — that breaks the implicit "Tab 1 button → sector debate"
    coupling on purpose; see spec §6 risk note 3.
    """
    try:
        from engine.memory import get_daily_brief_snapshot
        snap = get_daily_brief_snapshot(t_day)
        if not snap:
            return ""
        parts: list[str] = []
        if snap.regime:
            parts.append(f"当前制度: {snap.regime}")
        if snap.p_risk_on is not None:
            parts.append(f"P(risk-on): {snap.p_risk_on:.1%}")
        if snap.narrative:
            parts.append(f"今日宏观摘要: {snap.narrative[:300]}")
        return "\n".join(parts)
    except Exception as e:
        logger.warning("macro_context build failed for %s: %s", t_day, e)
        return ""


def _build_news_context(sector: str, ticker: str, regime_label_ui: str) -> str:
    """
    Mirror ui.tabs:_run_sector_analysis news pipeline:
      Layer 1 fetch_sector_news + build_weighted_news_summary
      → fall back to NewsPerceiver.build_context
      → append spillover_context.
    """
    news_ctx = ""
    try:
        from engine.news_fetcher import fetch_sector_news, build_weighted_news_summary
        items = fetch_sector_news(sector, ticker, days=3, max_total=8)
        if items:
            news_ctx = build_weighted_news_summary(items, max_chars=1200)
    except Exception as e:
        logger.warning("fetch_sector_news failed for %s: %s", sector, e)

    perceiver = None
    if not news_ctx:
        try:
            from engine.news import NewsPerceiver
            perceiver = NewsPerceiver(
                av_key=_get_secret("AV_KEY"),
                gnews_key=_get_secret("GNEWS_KEY"),
            )
            news_ctx = perceiver.build_context(
                sector, ticker, n=6, macro_regime=regime_label_ui,
            )
        except Exception as e:
            logger.warning("NewsPerceiver.build_context failed for %s: %s", sector, e)
            news_ctx = ""

    try:
        from engine.news import NewsPerceiver
        if perceiver is None:
            perceiver = NewsPerceiver(
                av_key=_get_secret("AV_KEY"),
                gnews_key=_get_secret("GNEWS_KEY"),
            )
        spill = perceiver.build_spillover_context(sector, macro_regime=regime_label_ui)
        if spill:
            news_ctx = (news_ctx + "\n\n" + spill) if news_ctx else spill
    except Exception as e:
        logger.debug("spillover_context skipped for %s: %s", sector, e)

    return news_ctx


def prepare_sector_inputs(
    sector_name: str,
    t_day:       datetime.date,
    vix:         float,
) -> dict[str, Any]:
    """
    Pure ETL: pull every input the sector debate needs.

    Returns dict with keys aligned to run_sector_debate signature plus
    auxiliary fields (state_vector, ticker_for_news, regime_label) needed
    downstream by save_decision.
    """
    from engine.history import get_active_sector_etf
    from engine.scanner import AUDIT_TICKERS
    from engine.memory import get_historical_context
    from engine.signal import get_quant_gates
    from engine.quant import (
        compute_quant_metrics, compute_state_vector, get_valuation_snapshot,
    )

    sector_etf_map = get_active_sector_etf()
    audit_tickers = AUDIT_TICKERS.get(sector_name) or []
    ticker_for_news = (
        sector_etf_map.get(sector_name)
        or (audit_tickers[0] if audit_tickers else sector_name)
    )

    regime_label_ui = _infer_macro_regime(vix)
    regime_bucket   = _REGIME_LABEL_MAP.get(regime_label_ui, "transition")

    # News
    news_context = _build_news_context(sector_name, ticker_for_news, regime_label_ui)

    # Historical
    try:
        historical_context = get_historical_context(
            "sector", sector_name=sector_name,
            macro_regime=regime_label_ui, n=5,
        )
    except Exception as e:
        logger.warning("historical_context failed for %s: %s", sector_name, e)
        historical_context = ""

    # Macro (DailyBriefSnapshot — same source agent uses)
    macro_context = _build_macro_context(t_day)

    # Valuation
    etf_for_valuation = sector_etf_map.get(sector_name) or ticker_for_news
    try:
        valuation_context = (
            get_valuation_snapshot(etf_for_valuation) if etf_for_valuation else ""
        )
    except Exception as e:
        logger.warning("valuation_snapshot failed for %s: %s", sector_name, e)
        valuation_context = ""

    # Quant metrics — AUDIT_TICKERS is canonical (matches Tab3 audit agent)
    quant_tickers = tuple(audit_tickers) if audit_tickers else ()
    try:
        quant_context = (
            compute_quant_metrics(quant_tickers, vix) if quant_tickers else {}
        )
    except Exception as e:
        logger.warning("compute_quant_metrics failed for %s: %s", sector_name, e)
        quant_context = {}

    # Quant gate
    try:
        all_gates = get_quant_gates(as_of=t_day, regime_label=regime_bucket)
        quant_gate = dict(all_gates.get(sector_name, {}))
        if quant_context and "a_vol" in quant_context:
            quant_gate["ann_vol"] = quant_context["a_vol"]
    except Exception as e:
        logger.warning("get_quant_gates failed for %s: %s", sector_name, e)
        quant_gate = {}

    # State vector for debate_transcript
    try:
        state_vector = compute_state_vector(ticker_for_news, t_day)
    except Exception as e:
        logger.debug("compute_state_vector failed for %s: %s", ticker_for_news, e)
        state_vector = {}

    return {
        "news_context":       news_context,
        "macro_context":      macro_context,
        "historical_context": historical_context,
        "valuation_context":  valuation_context,
        "quant_context":      quant_context,
        "quant_gate":         quant_gate,
        "state_vector":       state_vector,
        "ticker_for_news":    ticker_for_news,
        "regime_label":       regime_label_ui,
    }


def run_sector_pipeline(
    model,
    sector_name:        str,
    t_day:              datetime.date,
    vix:                float,
    decision_source:    str,
    parent_decision_id: int | None = None,
    revision_reason:    str = "",
    overwrite:          bool = False,
    history_prefix:     str = "",
) -> dict[str, Any]:
    """
    Back-compat wrapper.

    Implementation moved to engine/agents/sector_pipeline/SectorPipelineAgent
    on 2026-05-03 (P0 step 2 of agent-infra adoption sweep — observability
    only, NOT a behavior change). This is observability/audit infrastructure,
    not agentic capability. See memory/project_agentic_orchestration_v1.md.

    All three call sites preserved unchanged:
      - engine/daily_batch.py (auto-debate)
      - engine/orchestrator.py (signal-flip)
      - engine/paper_trading.py (Arm B forward run)

    Returns the same dict shape as before.
    """
    from engine.agents.base import Trigger
    from engine.agents.sector_pipeline import SectorPipelineAgent

    agent = SectorPipelineAgent(model=model)
    trigger = Trigger(
        type="manual" if "ui" in decision_source else "scheduled",
        source=decision_source,
        payload={
            "sector":             sector_name,
            "vix":                vix,
            "parent_decision_id": parent_decision_id,
            "revision_reason":    revision_reason,
            "overwrite":          overwrite,
            "history_prefix":     history_prefix,
        },
    )
    agent.run(trigger, t_day)
    return agent._last_result or {
        "saved_id":            None,
        "debate":              {},
        "qc_flags":            [],
        "scaled_adj":          0.0,
        "pending_approval_id": None,
        "inputs":              {},
    }
