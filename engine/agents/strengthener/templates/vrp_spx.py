"""engine.agents.strengthener.templates.vrp_spx — variance risk premium
on SPX (Carr-Wu 2009 canonical).

Tests claims of the form:
  "Implied vol systematically exceeds realized vol; short-vol strategy
   earns positive risk premium" / "The variance risk premium is
   significantly positive in SPX" / similar.

Scope (narrow MVP)
==================
  signal_kind  : vrp
  universe     : us_equities_spx_options
  data         : VIX index daily + SPX index daily (1990-2026)
  strategy     : monthly short-variance proxy.
                 At month-end t observe VIX_t (forward-looking 30-day
                 implied vol). Over the next ~21 trading days compute
                 realized vol from SPX daily returns. Short-vol PnL per
                 month = (VIX_t/100)² × (21/252) - RV²_{t,t+21} × (21/252)
                 i.e. variance-swap-style payoff in variance units.

Verdict
=======
Sharpe + Newey-West t-stat on monthly PnL series. Multi-testing-
corrected GREEN threshold per strategy_family n_trials (BUG-3).
Average PnL must be POSITIVE for GREEN — Carr-Wu 2009 documented
positive VRP; if our sample shows negative mean PnL, claim fails by
sign alone before threshold check.

M2 anchor (Carr-Wu 2009): mean monthly PnL > 0 in 1990-2007
sub-sample. Replication anchor test guards against template math
drift.

Why short-vol convention (not long-vol)
========================================
Carr-Wu 2009 frames VRP as "vol risk premium" = systematic positive
return to writing vol insurance. PnL convention is from variance
swap dealer perspective: receive realized, pay implied. Mean
positive = realized < implied on average = insurance writers profit.
We reverse the sign to match "short-vol strategy PnL > 0" reading.

Known limitations (defer)
=========================
- No transaction cost adjustment (real variance swaps have bid-ask
  spreads ~2-5 vol pts; ignored here)
- 1990-2026 sample includes 2008 GFC + 2020 COVID where short-vol
  blew up catastrophically — Sharpe will look worse than Carr-Wu's
  1990-2007 sample. Honest result.
- VIX measures 30-day BS-implied vol; canonical variance swap is
  log-contract not BS. Approximation introduces small bias.
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

_TEMPLATE_VERSION = "v1.0_2026-06-13"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_VIX_SPX_PATH = _REPO_ROOT / "data" / "cache" / "_vix_spx_daily.parquet"

_MIN_OBS_MONTHS = 60
_TRADING_DAYS_PER_MONTH = 21


def _load_vix_spx_daily() -> Optional[pd.DataFrame]:
    if not _VIX_SPX_PATH.is_file():
        return None
    df = pd.read_parquet(_VIX_SPX_PATH)
    return df.dropna(how="all")


def _build_monthly_vrp_pnl(daily: pd.DataFrame) -> Optional[pd.Series]:
    """Construct monthly short-vol PnL series (variance-swap convention).

    For each month-end t:
      - implied variance over next 21 trading days = (VIX_t / 100)²  × (21/252)
      - realized variance from SPX daily log returns over next 21 days
      - short-vol PnL = implied_variance - realized_variance
        (positive iff realized < implied → insurance writer wins)

    Returns monthly Series indexed by month-end of PNL OBSERVATION date
    (i.e., end of the 21-day window the PnL realizes over).
    """
    if "VIX" not in daily.columns or "SPX" not in daily.columns:
        return None
    df = daily[["VIX", "SPX"]].dropna()
    if len(df) < 252:
        return None

    log_ret = np.log(df["SPX"] / df["SPX"].shift(1)).dropna()
    # Realized variance over rolling N trading days
    rolling_var = (log_ret ** 2).rolling(_TRADING_DAYS_PER_MONTH).sum()
    # Implied variance at month start = (VIX/100)² × (21/252)
    implied_var = (df["VIX"] / 100.0) ** 2 * (_TRADING_DAYS_PER_MONTH / 252.0)
    # PnL at end of period t = implied_var at start - realized_var over period
    # implied at month start, realized over the month → PnL at month end
    # Need implied_at_start = implied_var.shift(_TRADING_DAYS_PER_MONTH)
    implied_at_start = implied_var.shift(_TRADING_DAYS_PER_MONTH)
    daily_pnl = implied_at_start - rolling_var
    # Resample to month-end — take last value of the rolling daily series
    monthly = daily_pnl.resample("ME").last().dropna()
    return monthly


def _classify_verdict(
    sharpe: float, nw_t: float, mean_pnl: float, n_trials: int,
) -> tuple[str, str]:
    """Returns (verdict, note). GREEN requires POSITIVE mean PnL by
    construction (Carr-Wu doctrine: VRP is a risk premium for insurance
    writers, not a random alpha). Negative mean → RED regardless of NW-t.
    """
    if not math.isfinite(nw_t):
        return "INSUFFICIENT_HISTORY", "NW-t non-finite"
    if mean_pnl <= 0:
        return "RED", (
            f"mean monthly PnL {mean_pnl:.5f} ≤ 0; short-vol strategy "
            f"did NOT earn positive VRP in this sample"
        )

    # BUG-3 corrected thresholds
    try:
        from engine.research.verdict_thresholds import (
            t_green_threshold, t_marginal_threshold,
        )
        t_g = t_green_threshold(n_trials)
        t_m = t_marginal_threshold(n_trials)
    except Exception:
        t_g, t_m = 1.96, 1.65

    if nw_t >= t_g:
        return "GREEN", f"NW-t={nw_t:.2f} >= {t_g:.2f}; positive VRP confirmed"
    if nw_t >= t_m:
        return "MARGINAL", f"NW-t={nw_t:.2f} in [{t_m:.2f}, {t_g:.2f})"
    return "RED", f"NW-t={nw_t:.2f} < {t_m:.2f}; mean positive but not statistically significant"


def template_vrp_spx(spec: FactorSpec):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    # 1. Load data
    daily = _load_vix_spx_daily()
    if daily is None:
        return TemplateResult(
            verdict          = "INSUFFICIENT_DATA",
            summary          = "VIX/SPX daily cache missing",
            metrics          = {},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # 2. Build monthly PnL series
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
    std_pnl = float(monthly_pnl.std(ddof=1))
    if std_pnl <= 0 or not math.isfinite(std_pnl):
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = "PnL series has degenerate variance",
            metrics          = {"n_obs_months": n_obs},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    sharpe = mean_pnl / std_pnl * math.sqrt(12.0)

    # NW-t with HAC SE lag 6
    try:
        import statsmodels.api as sm
        x = np.ones(n_obs)
        ols = sm.OLS(monthly_pnl.values, x).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6},
        )
        nw_t = float(ols.tvalues[0])
    except Exception:
        nw_t = mean_pnl / (std_pnl / math.sqrt(n_obs))

    # 3. Verdict
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
        f"VRP short-vol SPX ({monthly_pnl.index[0].strftime('%Y-%m')}~"
        f"{monthly_pnl.index[-1].strftime('%Y-%m')}, n={n_obs}mo): "
        f"mean_pnl={mean_pnl*10000:+.1f}vp² (variance-points-squared), "
        f"Sharpe={sharpe:+.2f}, NW-t={nw_t:+.2f}, "
        f"MaxDD={max_dd*10000:+.1f}vp² → {verdict}. {note}"
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
        },
        artifacts        = {
            "pnl_series_df":   _pnl_df,
            "pnl_default_col": "pnl_net_13bp",
            "pnl_gross_col":   "pnl_gross",
        },
        template_version = _TEMPLATE_VERSION,
    )
