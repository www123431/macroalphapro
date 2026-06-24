"""
engine/era.py — External Reality Audit (ERA)
==============================================
Quarterly LLM audit that evaluates past Track B decisions against realized
macro outcomes. Verdict is written back to AlphaMemory.

Design (P6 blueprint / M4 resolution):
  - Temperature  : 0.1 (low — disciplined, not creative)
  - Input        : macro data only — no price series, no P&L
  - Verdict set  : logic_correct / lucky_guess / logic_wrong
  - Cadence      : quarterly (first NYSE trading day of Q1/Q2/Q3/Q4)
  - Decoupled    : ERA has no feedback loop into the decision model;
                   verdicts are retrospective metadata only
  - Gate         : GATE_ERA must be "true" in SystemConfig

Called from: daily_batch.py quarterly async thread (Phase 5)
"""
from __future__ import annotations

import datetime
import json
import logging

logger = logging.getLogger(__name__)

_GATE_KEY      = "GATE_ERA"
_LOOKBACK_DAYS = 90    # review decisions made in the prior quarter
_VALID_VERDICTS = frozenset({"logic_correct", "lucky_guess", "logic_wrong"})


# ── Gate ──────────────────────────────────────────────────────────────────────

def _gate_enabled() -> bool:
    try:
        from engine.memory import get_system_config
        return str(get_system_config(_GATE_KEY, "false")).lower() == "true"
    except Exception:
        return False


# ── Macro context fetcher ──────────────────────────────────────────────────────

def _fetch_macro_context(as_of: datetime.date) -> str:
    """
    Fetch macro indicators available as of as_of (no price data).
    Returns a human-readable string for the LLM prompt.
    Macro data only: rates, credit spreads, economic regime — no sector prices.
    """
    lines: list[str] = []
    cutoff = as_of.isoformat()
    try:
        import yfinance as yf
        # 10Y Treasury yield (^TNX)
        tnx = yf.download("^TNX", period="3mo", progress=False, auto_adjust=True)
        if not tnx.empty:
            y_start = float(tnx["Close"].iloc[0])
            y_end   = float(tnx["Close"].iloc[-1])
            lines.append(f"10Y Treasury: {y_start:.2f}% → {y_end:.2f}% (过去3个月)")
    except Exception:
        pass
    try:
        import yfinance as yf
        # 2Y Treasury (^IRX proxy for short end)
        irx = yf.download("^IRX", period="3mo", progress=False, auto_adjust=True)
        if not irx.empty:
            lines.append(
                f"13W T-Bill: {float(irx['Close'].iloc[-1]):.2f}%（当前无风险利率）"
            )
    except Exception:
        pass
    try:
        import yfinance as yf
        # HYG as credit spread proxy (yield spread vs treasuries)
        hyg = yf.download("HYG", period="3mo", progress=False, auto_adjust=True)
        if not hyg.empty:
            ret_3m = float(hyg["Close"].iloc[-1] / hyg["Close"].iloc[0] - 1)
            lines.append(f"HYG(高收益债) 3M回报: {ret_3m:+.2%}（信用代理）")
    except Exception:
        pass
    try:
        import yfinance as yf
        # VIX
        vix = yf.download("^VIX", period="3mo", progress=False, auto_adjust=True)
        if not vix.empty:
            vix_now   = float(vix["Close"].iloc[-1])
            vix_start = float(vix["Close"].iloc[0])
            lines.append(f"VIX: {vix_start:.1f} → {vix_now:.1f}")
    except Exception:
        pass

    return "\n".join(lines) if lines else f"宏观数据获取失败（截止 {cutoff}）"


# ── Past decisions fetcher ────────────────────────────────────────────────────

def _get_recent_decisions(as_of: datetime.date) -> list[dict]:
    """
    Return Track B AlphaMemory decisions from the prior quarter that have
    not yet been ERA-audited (era_verdict IS NULL).
    """
    from engine.memory import AlphaMemory, SessionFactory
    cutoff = as_of - datetime.timedelta(days=_LOOKBACK_DAYS)
    with SessionFactory() as s:
        rows = (
            s.query(AlphaMemory)
             .filter(
                 AlphaMemory.source == "track_b",
                 AlphaMemory.decision_date >= cutoff,
                 AlphaMemory.decision_date <= as_of,
                 AlphaMemory.era_verdict.is_(None),
             )
             .order_by(AlphaMemory.decision_date)
             .all()
        )
    return [
        {
            "id":             r.id,
            "decision_date":  str(r.decision_date),
            "sector":         r.sector,
            "quant_weight":   r.quant_weight,
            "llm_delta":      r.llm_delta,
            "adjusted_weight":r.adjusted_weight,
            "logic_chain":    r.logic_chain or "",
            "confidence":     r.confidence,
        }
        for r in rows
    ]


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_era_prompt(
    as_of: datetime.date,
    decisions: list[dict],
    macro_context: str,
) -> str:
    decision_lines = []
    for i, d in enumerate(decisions, 1):
        decision_lines.append(
            f"  [{i}] {d['decision_date']} | {d['sector']} | "
            f"Δ={float(d['llm_delta'] or 0):+.2%} | "
            f"理由: {d['logic_chain'][:80] if d['logic_chain'] else 'N/A'}"
        )

    return f"""你是量化基金的外部现实审计员（ERA）。你的职责是评估Track B在过去一个季度的决策，
判断其逻辑是否在事后被宏观现实证实。

审计日期：{as_of}

过去一季度宏观环境（仅使用这些数据，不得使用个股/ETF价格）：
{macro_context}

待审计的Track B决策：
{chr(10).join(decision_lines)}

对每个决策给出以下JSON格式的审计结论：
{{
  "audits": [
    {{
      "id": 决策ID（整数）,
      "verdict": "logic_correct" 或 "lucky_guess" 或 "logic_wrong",
      "era_score": 0到1之间的置信度（1=确定，0=无法判断）,
      "reasoning": "一句话说明理由（中文，不超过80字）"
    }}
  ]
}}

判断标准：
- logic_correct : 决策逻辑有据可查且宏观环境确实按预期方向发展
- lucky_guess   : 方向正确但逻辑站不住脚，或运气成分大
- logic_wrong   : 宏观环境走势与决策逻辑相悖
- 无法判断时  : 输出 lucky_guess 并 era_score=0.1"""


# ── Write verdicts back to DB ─────────────────────────────────────────────────

def _write_verdicts(
    as_of: datetime.date,
    audits: list[dict],
) -> int:
    """Update AlphaMemory rows with ERA verdicts. Returns count updated."""
    from engine.memory import AlphaMemory, SessionFactory
    updated = 0
    with SessionFactory() as s:
        for audit in audits:
            row = s.query(AlphaMemory).filter(AlphaMemory.id == audit["id"]).first()
            if row is None:
                continue
            verdict = str(audit.get("verdict", "lucky_guess"))
            if verdict not in _VALID_VERDICTS:
                verdict = "lucky_guess"
            row.era_verdict   = verdict
            row.era_score     = float(audit.get("era_score", 0.1))
            row.era_reasoning = str(audit.get("reasoning", ""))[:300]
            row.verified_at   = datetime.datetime.utcnow()
            updated += 1
        s.commit()
    return updated


# ── Main entry point ───────────────────────────────────────────────────────────

def run_era(
    as_of: datetime.date,
    model,
) -> dict:
    """
    Run ERA quarterly audit. Returns summary dict with counts and verdict breakdown.

    Requires GATE_ERA = "true" in SystemConfig and a non-None model.
    Safe to call even if no unaudited decisions exist.
    """
    summary = {
        "as_of":           str(as_of),
        "decisions_found": 0,
        "updated":         0,
        "verdicts":        {},
        "skipped_reason":  "",
    }

    if not _gate_enabled():
        summary["skipped_reason"] = "GATE_ERA disabled"
        logger.debug("era: GATE_ERA disabled — skipping")
        return summary
    if model is None:
        summary["skipped_reason"] = "no model"
        return summary

    decisions = _get_recent_decisions(as_of)
    summary["decisions_found"] = len(decisions)
    if not decisions:
        logger.debug("era: no unaudited Track B decisions found for %s", as_of)
        return summary

    macro_context = _fetch_macro_context(as_of)
    prompt        = _build_era_prompt(as_of, decisions, macro_context)

    # ── LLM call (temperature=0.1 — M4 resolution) ───────────────────────────
    try:
        from engine.key_pool import get_pool
        _pool = get_pool()
        _m    = _pool.get_model(temperature=0.1)
        raw   = _m.generate_content(prompt).text.strip()
        _pool.report_success(has_content=bool(raw))
    except Exception:
        try:
            raw = model.generate_content(prompt).text.strip()
        except Exception as exc:
            logger.warning("era: LLM call failed: %s", exc)
            summary["skipped_reason"] = f"LLM error: {exc}"
            return summary

    # ── Parse JSON ────────────────────────────────────────────────────────────
    try:
        _s, _e = raw.find("{"), raw.rfind("}") + 1
        if _s == -1 or _e == 0:
            raise ValueError("no JSON found")
        data   = json.loads(raw[_s:_e])
        audits = data.get("audits", [])
        if not isinstance(audits, list):
            raise ValueError("audits is not a list")
    except Exception as exc:
        logger.warning("era: JSON parse failed: %s | raw=%s", exc, raw[:200])
        summary["skipped_reason"] = f"parse error: {exc}"
        return summary

    updated = _write_verdicts(as_of, audits)
    summary["updated"] = updated

    # Tally verdict distribution
    for audit in audits:
        v = audit.get("verdict", "unknown")
        summary["verdicts"][v] = summary["verdicts"].get(v, 0) + 1

    logger.info(
        "era: audited %d decisions on %s — updated=%d verdicts=%s",
        len(decisions), as_of, updated, summary["verdicts"],
    )
    return summary
