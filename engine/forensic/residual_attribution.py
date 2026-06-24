"""
engine/forensic/residual_attribution.py — Brinson-Hood-Beebower decomposition.

Tier-1 audit #4 follow-up / Forensic redesign Phase 2 (2026-05-14).

Purpose
-------
Decompose a strategy-day's realized net return into 3 deterministic
components plus a residual:

    realized_net = β · F + TC_drag + ε

  β · F     — factor-explained return (FF5 betas × realized factor returns)
  TC_drag   — execution friction (from engine.execution.cost_model via
              PaperTradeStrategyLog.tc_drag_today populated by Step 6 backfill)
  ε         — unexplained residual

This is the ONLY scope where LLM forensic narrative is admissible:
LLM should explain ε, not the components already explained by factors
or execution. If |ε|/|realized| < 0.40, the day is "factor-explained" and
LLM narrative adds nothing — narrative only fires when there is a real
residual to investigate.

Design choice — no α intercept term
-----------------------------------
Daily α can't be estimated from a single day's data; including a separate
α intercept here would just shift noise between α and ε without adding
information. Book-level α is a separate, longer-window estimate (rolling
60-day OLS in engine.risk_metrics.compute_ff5_factor_tilt) and is reported
in the FF5 expander on the Positions page. For per-day residual
attribution we use the 3-term decomposition deliberately.

References
----------
  - Brinson-Hood-Beebower 1986 "Determinants of Portfolio Performance"
  - Almgren-Chriss 2000 (TC model used here from engine.execution.cost_model)
  - Fama-French 2015 (FF5 factors)
"""
from __future__ import annotations

import datetime
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ResidualBreakdown:
    """3-component decomposition of a strategy-day's realized net return."""
    date:              datetime.date
    strategy_name:     str

    realized_net:      float           # observed daily_net_return
    factor_explained:  float           # Σ β_i × F_i (FF5)
    tc_drag:           float           # signed (negative for friction)
    residual:          float           # realized - factor - tc_drag

    # Diagnostic metrics
    abs_residual_share: float          # |ε| / |realized|
    llm_eligible:       bool           # abs_residual_share >= LLM_ELIGIBILITY_THRESHOLD

    # Breakdown detail
    factor_betas:      dict            # {Mkt, SMB, HML, RMW, CMA}
    factor_returns:    dict            # {Mkt, SMB, HML, RMW, CMA} realized on day
    factor_contribs:   dict            # {Mkt, SMB, HML, RMW, CMA} β × F

    # Data quality
    n_positions_used:  int             # n positions covered by FF5 calc
    n_obs_factor:      int             # rolling-window obs for β estimation
    weighted_r_squared: float          # asset-level weighted R² from FF5 OLS
    notes:             list[str] = field(default_factory=list)


# ε-magnitude threshold above which LLM forensic narrative is admissible.
# Below this, the return is sufficiently explained by factors + TC; running
# LLM would generate narrative on noise. Calibrated against typical
# institutional factor-attribution coverage (50-70% explained by FF5 for
# single-stock books).
LLM_ELIGIBILITY_THRESHOLD = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# Core decomposition
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_realized_factor_returns(
    date: datetime.date,
) -> Optional[dict]:
    """Fetch realized FF5 factor returns on a specific trading day.

    Uses the same proxy ETFs as engine.risk_metrics._ff5_factor_returns:
      Mkt = SPY
      SMB = IWM - IWB
      HML = IWD - IWF
      RMW = QUAL - SPY
      CMA = USMV - SPY
    """
    import yfinance as _yf
    import pandas as _pd
    tickers = ("SPY", "IWM", "IWB", "IWD", "IWF", "QUAL", "USMV")
    try:
        # Fetch a 5-day window to ensure the target date is covered and we
        # can compute pct_change (need 2 consecutive trading days).
        start = date - datetime.timedelta(days=10)
        end   = date + datetime.timedelta(days=2)
        data = _yf.download(
            list(tickers), start=start.isoformat(), end=end.isoformat(),
            auto_adjust=True, progress=False, multi_level_index=False,
        )
        close = data["Close"] if "Close" in data.columns else data
        if isinstance(close, _pd.Series):
            close = close.to_frame(name=tickers[0])
        close.index = _pd.to_datetime(close.index).date
    except Exception as exc:
        logger.warning("FF5 factor returns fetch failed for %s: %s", date, exc)
        return None

    avail = sorted(d for d in close.index if d <= date)
    if len(avail) < 2:
        return None
    today, prev = avail[-1], avail[-2]
    try:
        rets = {}
        for tk in tickers:
            if tk not in close.columns:
                return None
            p_today = float(close.at[today, tk])
            p_prev  = float(close.at[prev,  tk])
            if not (p_prev > 0 and not math.isnan(p_today) and not math.isnan(p_prev)):
                return None
            rets[tk] = p_today / p_prev - 1.0
        return {
            "Mkt": rets["SPY"],
            "SMB": rets["IWM"]  - rets["IWB"],
            "HML": rets["IWD"]  - rets["IWF"],
            "RMW": rets["QUAL"] - rets["SPY"],
            "CMA": rets["USMV"] - rets["SPY"],
        }
    except Exception:
        return None


def decompose_strategy_day(
    strategy_name: str,
    date:          datetime.date,
    session:       Optional[object] = None,
) -> Optional[ResidualBreakdown]:
    """Decompose realized_net for a single (strategy, date) row.

    Returns None if data is missing (no positions / no realized return /
    no factor returns available). Caller distinguishes None (insufficient
    data) from a populated ResidualBreakdown (decomposition complete).
    """
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeStrategyLog
    from engine.risk_metrics import compute_ff5_factor_tilt
    import pandas as _pd

    init_db()
    own_session = session is None
    sess = session if session is not None else SessionFactory()
    try:
        row = (
            sess.query(PaperTradeStrategyLog)
                .filter(PaperTradeStrategyLog.strategy_name == strategy_name,
                        PaperTradeStrategyLog.date == date)
                .first()
        )
    finally:
        if own_session:
            sess.close()

    if row is None:
        return None
    if row.daily_net_return is None and row.daily_gross_return is None:
        return None
    if not row.positions_json:
        return None
    try:
        positions = json.loads(row.positions_json)
    except Exception:
        return None
    if not positions:
        return None

    realized_net = (
        float(row.daily_net_return)
        if row.daily_net_return is not None
        else float(row.daily_gross_return)
    )
    tc_drag = float(row.tc_drag_today or 0.0)

    # Compute strategy-position FF5 betas via the existing helper.
    book_df = _pd.DataFrame(
        [{"ticker": tk, "actual_weight": float(w)} for tk, w in positions.items()]
    )
    tilt = compute_ff5_factor_tilt(book_df, period="2y")
    if tilt["n_obs"] == 0 or tilt["n_assets"] == 0:
        # Can't compute factor decomposition — return None so caller knows
        # to fall back to simpler explanation.
        return None

    realized_factors = _fetch_realized_factor_returns(date)
    if realized_factors is None:
        return None

    factor_contribs = {
        f: float(tilt[f]) * float(realized_factors[f])
        for f in ("Mkt", "SMB", "HML", "RMW", "CMA")
    }
    factor_explained = sum(factor_contribs.values())
    # tc_drag is a positive cost-decimal (per fill_daily_tc convention);
    # in net-return space it's already subtracted via daily_net_return =
    # daily_gross_return - tc_drag_today. So in the residual identity:
    #     realized_net = β·F + (-tc_drag) + ε
    # we represent TC contribution as -tc_drag (negative). To keep the
    # invariant (realized = sum of components + residual), we use a
    # signed tc_contribution.
    tc_contribution = -tc_drag
    residual = realized_net - factor_explained - tc_contribution

    abs_share = (abs(residual) / abs(realized_net)) if abs(realized_net) > 1e-9 else 0.0
    llm_eligible = abs_share >= LLM_ELIGIBILITY_THRESHOLD

    notes: list[str] = []
    if not row.daily_net_return and row.daily_gross_return is not None:
        notes.append("daily_net_return not populated; used daily_gross_return as fallback")
    if tc_drag == 0:
        notes.append("tc_drag_today = 0 (non-rebalance day or pre-Step-6 row)")

    return ResidualBreakdown(
        date=date,
        strategy_name=strategy_name,
        realized_net=realized_net,
        factor_explained=float(factor_explained),
        tc_drag=float(tc_contribution),
        residual=float(residual),
        abs_residual_share=float(abs_share),
        llm_eligible=bool(llm_eligible),
        factor_betas={k: float(tilt[k]) for k in ("Mkt", "SMB", "HML", "RMW", "CMA")},
        factor_returns={k: float(realized_factors[k]) for k in ("Mkt", "SMB", "HML", "RMW", "CMA")},
        factor_contribs={k: float(factor_contribs[k]) for k in ("Mkt", "SMB", "HML", "RMW", "CMA")},
        n_positions_used=int(tilt["n_assets"]),
        n_obs_factor=int(tilt["n_obs"]),
        weighted_r_squared=float(tilt["r_squared"]),
        notes=notes,
    )


def explain_decomposition(rb: ResidualBreakdown) -> str:
    """Human-readable summary line for logs / UI."""
    pct = lambda v: f"{v*100:+.3f}%"
    contribs_str = " + ".join(
        f"{f}:{pct(c)}" for f, c in rb.factor_contribs.items() if abs(c) > 1e-5
    ) or "negligible"
    return (
        f"{rb.date} {rb.strategy_name}  realized={pct(rb.realized_net)}  "
        f"= factors({pct(rb.factor_explained)}: {contribs_str})  "
        f"+ TC({pct(rb.tc_drag)})  + epsilon({pct(rb.residual)})  "
        f"|epsilon|/|r|={rb.abs_residual_share*100:.0f}%  "
        f"LLM_eligible={rb.llm_eligible}"
    )
