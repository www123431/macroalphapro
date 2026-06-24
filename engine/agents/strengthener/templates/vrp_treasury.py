"""engine.agents.strengthener.templates.vrp_treasury — variance risk premium
on US Treasuries (Bond-VRP, MVP).

Built 2026-06-22 to dispatch the synthesizer's Bond-VRP candidate (W4-E2E
follow-up). Mirrors the vrp_spx template's Carr-Wu 2009 short-vol logic,
substituting:
  VIX  → MOVE  (ICE BofA MOVE = 1m forward-looking implied vol on US
                 Treasury futures, reported in BASIS POINTS of yield)
  SPX  → TLT   (iShares 20+ Year Treasury Bond ETF — most-liquid
                 long-duration Treasury price proxy)

MOVE-to-TLT scaling
===================
MOVE is yield-volatility in basis points. TLT is price. To compare,
approximate TLT price vol via duration: dP/P ≈ -D × dY. TLT duration is
~17 years (as of 2024). So 100bp MOVE roughly corresponds to ~17% TLT
price vol annualized.

Implementation: scale MOVE / 100 by TLT effective duration (~17) to
get TLT-implied price vol equivalent. Conservative pick D=17 captures
the post-2002 average; not adjusted for time-varying duration.

Strategy (MVP, monthly rebalance)
==================================
At month-end t:
  - implied_var = (MOVE_t / 100 × D_TLT / 100)² × (21/252)
    i.e., square of monthly-equivalent IV in price terms
  - realized_var = sum over next 21 trading days of TLT log-return²
  - PnL = implied_var - realized_var (short-vol convention; positive
    iff realized < implied = vol-seller wins)

Verdict
=======
Sharpe + NW-t with HAC lag 6 on monthly PnL series. Mean must be
positive for GREEN (VRP economic claim is "vol-seller profits on
average"; negative mean → RED by sign alone). BUG-3 thresholds via
n_trials.

Known limitations (MVP, defer)
==============================
- Effective duration D=17 hardcoded; real TLT duration ranges 16-19
  across rate cycles. Affects implied-vol scaling by ±10% in variance
  units.
- No transaction cost adjustment (real Treasury options have wider
  spreads than SPX). Sharpe inflated relative to live trading.
- MOVE measures option-implied yield vol; we approximate TLT price
  vol via duration. The proper measure would be option chain on TLT
  itself (deep but illiquid). MOVE proxy introduces an additional
  duration-mismatch noise term.
"""
from __future__ import annotations

import dataclasses as _dc
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.agents.strengthener.factor_spec_extractor import FactorSpec

logger = logging.getLogger(__name__)

_TEMPLATE_VERSION = "v1.0_mvp_2026-06-22"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MOVE_TLT_PATH = _REPO_ROOT / "data" / "cache" / "_move_tlt_daily.parquet"

_MIN_OBS_MONTHS = 60
_TRADING_DAYS_PER_MONTH = 21
# TLT effective duration ~17 years post-2002 average. Used to scale
# MOVE (yield vol bps) to TLT (price vol pct).
_TLT_EFFECTIVE_DURATION_YEARS = 17.0


def _load_move_tlt_daily() -> Optional[pd.DataFrame]:
    if not _MOVE_TLT_PATH.is_file():
        return None
    df = pd.read_parquet(_MOVE_TLT_PATH)
    return df.dropna(how="all")


def _build_monthly_vrp_pnl(daily: pd.DataFrame) -> Optional[pd.Series]:
    """Construct monthly short-vol PnL series (variance-swap convention),
    Treasury edition. See module docstring for math.
    """
    if "MOVE" not in daily.columns or "TLT" not in daily.columns:
        return None
    df = daily[["MOVE", "TLT"]].dropna()
    if len(df) < 252:
        return None

    log_ret = np.log(df["TLT"] / df["TLT"].shift(1)).dropna()
    # Realized variance over rolling N trading days
    rolling_var = (log_ret ** 2).rolling(_TRADING_DAYS_PER_MONTH).sum()
    # MOVE → TLT-implied annualized vol via duration scale
    # MOVE bps × (D/100) = TLT vol bps; then / 100 to decimal; then square
    tlt_implied_vol = df["MOVE"] / 100.0 * (_TLT_EFFECTIVE_DURATION_YEARS / 100.0)
    # implied annualized variance, scaled to 21-day window
    implied_var = tlt_implied_vol ** 2 * (_TRADING_DAYS_PER_MONTH / 252.0)
    # implied at month start (lag 21 trading days), realized over month → end
    implied_at_start = implied_var.shift(_TRADING_DAYS_PER_MONTH)
    daily_pnl = implied_at_start - rolling_var
    monthly = daily_pnl.resample("ME").last().dropna()
    return monthly


def _classify_verdict(
    sharpe: float, nw_t: float, mean_pnl: float, n_trials: int,
) -> tuple[str, str]:
    """Same logic as vrp_spx: positive mean required for GREEN."""
    if not math.isfinite(nw_t):
        return "INSUFFICIENT_HISTORY", "NW-t non-finite"
    if mean_pnl <= 0:
        return "RED", (
            f"mean monthly PnL {mean_pnl:.5f} ≤ 0; short-vol Bond-VRP "
            f"did NOT earn positive premium in this sample"
        )

    try:
        from engine.research.verdict_thresholds import (
            t_green_threshold, t_marginal_threshold,
        )
        t_g = t_green_threshold(n_trials)
        t_m = t_marginal_threshold(n_trials)
    except Exception:
        t_g, t_m = 1.96, 1.65

    if nw_t >= t_g:
        return "GREEN", f"NW-t={nw_t:.2f} >= {t_g:.2f}; positive Bond-VRP confirmed"
    if nw_t >= t_m:
        return "MARGINAL", f"NW-t={nw_t:.2f} in [{t_m:.2f}, {t_g:.2f})"
    return "RED", f"NW-t={nw_t:.2f} < {t_m:.2f}; mean positive but not significant"


def template_vrp_treasury(spec: FactorSpec):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    # 1. Load data
    daily = _load_move_tlt_daily()
    if daily is None:
        return TemplateResult(
            verdict          = "INSUFFICIENT_DATA",
            summary          = ("MOVE/TLT daily cache missing — run "
                                "scripts/oneoff/_fetch_move_tlt_daily_*.py first"),
            metrics          = {},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # 2. Build monthly PnL
    monthly_pnl = _build_monthly_vrp_pnl(daily)
    if monthly_pnl is None or len(monthly_pnl) < _MIN_OBS_MONTHS:
        n = 0 if monthly_pnl is None else len(monthly_pnl)
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = f"only {n} monthly obs (min {_MIN_OBS_MONTHS})",
            metrics          = {"n_obs_months": n},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    n_obs = len(monthly_pnl)
    mean_pnl = float(monthly_pnl.mean())
    std_pnl  = float(monthly_pnl.std(ddof=1))
    if std_pnl <= 0 or not math.isfinite(std_pnl):
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = "PnL series has degenerate variance",
            metrics          = {"n_obs_months": n_obs},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    sharpe = mean_pnl / std_pnl * math.sqrt(12.0)

    # NW-t with HAC lag 6
    try:
        import statsmodels.api as sm
        x = np.ones(n_obs)
        ols = sm.OLS(monthly_pnl.values, x).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6},
        )
        nw_t = float(ols.tvalues[0])
    except Exception:
        nw_t = mean_pnl / (std_pnl / math.sqrt(n_obs))

    # n_trials for BUG-3 thresholds
    n_trials = 0
    try:
        from engine.research.strategy_family_classifier import (
            strategy_family_for_spec,
        )
        from engine.agents.strengthener.factor_dispatcher import (
            _family_n_trials_now,
        )
        n_trials = _family_n_trials_now(strategy_family_for_spec(spec))
    except Exception:
        pass

    verdict, note = _classify_verdict(sharpe, nw_t, mean_pnl, n_trials)

    # MaxDD on cumulative PnL
    cum = monthly_pnl.cumsum()
    running_max = cum.cummax()
    max_dd = float((cum - running_max).min())

    summary = (
        f"Bond-VRP short-vol Treasury ({monthly_pnl.index[0].strftime('%Y-%m')}~"
        f"{monthly_pnl.index[-1].strftime('%Y-%m')}, n={n_obs}mo): "
        f"mean_pnl={mean_pnl*10000:+.2f}vp² (variance-points-squared), "
        f"Sharpe={sharpe:+.2f}, NW-t={nw_t:+.2f}, "
        f"MaxDD={max_dd*10000:+.2f}vp² → {verdict}. {note}"
    )

    _pnl_df = pd.DataFrame({
        "pnl_gross":     monthly_pnl,
        "pnl_net_13bp":  monthly_pnl,  # cost model TBD
        "turnover":      pd.Series(0.0, index=monthly_pnl.index),
    })

    return TemplateResult(
        verdict          = verdict,
        summary          = summary,
        metrics          = {
            "mean_pnl_monthly":     mean_pnl,
            "std_pnl_monthly":      std_pnl,
            "sharpe_gross":         sharpe,
            "nw_t_gross":           nw_t,
            "max_drawdown":         max_dd,
            "n_obs_months":         n_obs,
            "n_trials_at_dispatch": n_trials,
            "tlt_duration_assumed": _TLT_EFFECTIVE_DURATION_YEARS,
        },
        artifacts        = {
            "pnl_series_df":   _pnl_df,
            "pnl_default_col": "pnl_net_13bp",
            "pnl_gross_col":   "pnl_gross",
        },
        template_version = _TEMPLATE_VERSION,
    )
