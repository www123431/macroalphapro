"""
engine/track_b.py — LLM Sector Weight Overlay (Track B)
=========================================================
A 5% absolute budget overlay that adjusts quant-derived weights based on
qualitative macro reasoning. All outputs require human approval.

Design constraints (P6 blueprint):
  - Budget  : ≤5% gross absolute deviation from quant weights
  - Per-sector: ±25% relative (e.g. 10% quant → [7.5%, 12.5%])
  - Scope   : only adjusts quant-held positions — no new entries or exits
  - Gate    : GATE_TRACK_B must be "true" in SystemConfig before any LLM call
  - Audit   : all decisions written to AlphaMemory + PendingApproval(track_b)

Called from: daily_batch.py monthly M2 job (Phase 5)
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_GATE_KEY       = "GATE_TRACK_B"
_BUDGET_ABS     = 0.05   # 5% gross absolute weight budget
_SECTOR_REL_CAP = 0.25   # ±25% relative per-sector cap


# ── Gate ──────────────────────────────────────────────────────────────────────

def _gate_enabled() -> bool:
    try:
        from engine.memory import get_system_config
        return str(get_system_config(_GATE_KEY, "false")).lower() == "true"
    except Exception:
        return False


# ── Data helpers ──────────────────────────────────────────────────────────────

def _get_quant_positions() -> dict[str, tuple[str, float]]:
    """Return {sector: (ticker, weight)} for non-zero main-track positions."""
    from engine.memory import SimulatedPosition, SessionFactory
    with SessionFactory() as s:
        latest_date = (
            s.query(SimulatedPosition.snapshot_date)
             .order_by(SimulatedPosition.snapshot_date.desc())
             .limit(1).scalar()
        )
        if latest_date is None:
            return {}
        positions = (
            s.query(SimulatedPosition)
             .filter(
                 SimulatedPosition.snapshot_date == latest_date,
                 SimulatedPosition.track == "main",
             )
             .all()
        )
    return {
        p.sector: (p.ticker, float(p.actual_weight or 0.0))
        for p in positions
        if (p.actual_weight or 0.0) != 0.0
    }


# ── Constraint enforcement ─────────────────────────────────────────────────────

def _enforce_constraints(
    proposals: dict[str, float],
    quant_weights: dict[str, float],
) -> dict[str, float]:
    """
    Apply per-sector relative cap then rescale to fit gross budget.
    Proposals for sectors not currently held are silently dropped.
    """
    adjusted: dict[str, float] = {}
    for sector, delta in proposals.items():
        qw = quant_weights.get(sector, 0.0)
        if qw == 0.0:
            continue
        rel_cap = abs(qw) * _SECTOR_REL_CAP
        adjusted[sector] = max(-rel_cap, min(rel_cap, float(delta)))

    gross = sum(abs(d) for d in adjusted.values())
    if gross > _BUDGET_ABS and gross > 1e-9:
        scale = _BUDGET_ABS / gross
        adjusted = {s: d * scale for s, d in adjusted.items()}

    return {s: d for s, d in adjusted.items() if abs(d) >= 1e-4}


# ── LLM prompt builder ─────────────────────────────────────────────────────────

def _build_prompt(
    as_of: datetime.date,
    regime_label: str,
    quant_positions: dict[str, tuple[str, float]],
    signal_df: "pd.DataFrame | None",
) -> str:
    position_lines = "\n".join(
        f"  {sector}: ticker={ticker}, weight={weight:+.2%}"
        for sector, (ticker, weight) in sorted(quant_positions.items())
    )
    signal_lines = "N/A"
    if signal_df is not None and not signal_df.empty:
        rows = []
        for sector in quant_positions:
            if sector in signal_df.index:
                r = signal_df.loc[sector]
                rows.append(
                    f"  {sector}: composite={float(r.get('composite_score', 50)):.0f} "
                    f"tsmom={float(r.get('tsmom', 0)):+.0f} "
                    f"carry_norm={float(r.get('carry_norm', 50)):.0f} "
                    f"csmom_rank={float(r.get('csmom_rank', 50)):.0f}"
                )
        if rows:
            signal_lines = "\n".join(rows)

    return f"""你是宏观量化基金的Track B叠加层。你的职责是在量化模型基础上，基于定性宏观判断提出微幅权重调整建议。

当前制度：{regime_label.upper() or 'UNKNOWN'}
日期：{as_of}

量化当前持仓（仅列出非零仓位）：
{position_lines}

信号详情：
{signal_lines}

约束条件（必须严格遵守）：
1. 仅对上述量化持仓进行调整，不得新增或清空仓位
2. 每个标的调整幅度不超过当前权重的±25%（相对）
3. 所有调整幅度绝对值之和不超过5%（预算约束）
4. 仅在有明确宏观理由时才提出调整；若无特殊情况请返回空 adjustments

输出JSON格式（严格遵守，不要附加文字）：
{{
  "adjustments": {{
    "sector名称": delta_weight（小数，如0.01表示+1%）
  }},
  "reasoning": "一句话说明调整理由（中文，不超过100字）",
  "confidence": 0到1之间的置信度
}}"""


# ── Main entry point ───────────────────────────────────────────────────────────

def run_track_b(
    as_of: datetime.date,
    model,
    regime_label: str = "",
    signal_df: "pd.DataFrame | None" = None,
) -> list[str]:
    """
    Run Track B LLM overlay. Returns list of sectors where adjustments were proposed.

    All proposals are written to AlphaMemory + PendingApproval (type='track_b').
    Requires GATE_TRACK_B = "true" in SystemConfig and a non-None model.
    """
    if not _gate_enabled():
        logger.debug("track_b: GATE_TRACK_B disabled — skipping")
        return []
    if model is None:
        logger.debug("track_b: no model available — skipping")
        return []

    quant_positions = _get_quant_positions()
    if not quant_positions:
        logger.debug("track_b: no active positions — skipping")
        return []

    prompt = _build_prompt(as_of, regime_label, quant_positions, signal_df)

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        from engine.key_pool import get_pool
        _pool = get_pool()
        _m    = _pool.get_model(temperature=0.3)
        raw   = _m.generate_content(prompt).text.strip()
        _pool.report_success(has_content=bool(raw))
    except Exception:
        # Fallback to passed model
        try:
            raw = model.generate_content(prompt).text.strip()
        except Exception as exc:
            logger.warning("track_b: LLM call failed: %s", exc)
            return []

    # ── Parse JSON ────────────────────────────────────────────────────────────
    try:
        _s, _e = raw.find("{"), raw.rfind("}") + 1
        if _s == -1 or _e == 0:
            logger.warning("track_b: no JSON in LLM response")
            return []
        data       = json.loads(raw[_s:_e])
        raw_props  = {str(k): float(v) for k, v in data.get("adjustments", {}).items()}
        reasoning  = str(data.get("reasoning", ""))[:300]
        confidence = float(data.get("confidence", 0.5))
    except Exception as exc:
        logger.warning("track_b: JSON parse failed: %s", exc)
        return []

    if not raw_props:
        logger.debug("track_b: LLM proposed no adjustments on %s", as_of)
        return []

    # ── Enforce constraints ───────────────────────────────────────────────────
    quant_weights = {s: w for s, (_, w) in quant_positions.items()}
    constrained   = _enforce_constraints(raw_props, quant_weights)
    if not constrained:
        return []

    # ── Persist: AlphaMemory + PendingApproval ────────────────────────────────
    sectors_proposed: list[str] = []
    try:
        from engine.memory import AlphaMemory, PendingApproval, SessionFactory
        now = datetime.datetime.utcnow()
        with SessionFactory() as s:
            for sector, delta in constrained.items():
                ticker, qw = quant_positions[sector]
                new_w = round(qw + delta, 4)

                s.add(AlphaMemory(
                    decision_date=as_of,
                    sector=sector,
                    source="track_b",
                    quant_weight=qw,
                    llm_delta=delta,
                    adjusted_weight=new_w,
                    logic_chain=reasoning,
                    confidence=confidence,
                    created_at=now,
                ))
                s.add(PendingApproval(
                    approval_type="track_b",
                    priority="normal",
                    sector=sector,
                    ticker=ticker,
                    triggered_condition=(
                        f"Track B: quant={qw:+.2%} → proposed={new_w:+.2%} "
                        f"(Δ={delta:+.2%}) | {reasoning}"
                    ),
                    triggered_date=as_of,
                    suggested_weight=new_w,
                ))
                sectors_proposed.append(sector)
            s.commit()

        logger.info(
            "track_b: proposed adjustments for %d sectors on %s: %s",
            len(sectors_proposed), as_of, sectors_proposed,
        )
    except Exception as exc:
        logger.warning("track_b: DB write failed: %s", exc)

    return sectors_proposed
