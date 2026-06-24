"""
engine/path_e/pre_fomc_backtest.py — Backtest + verdict for Pre-FOMC drift.

Pre-registration: docs/spec_path_e_pre_fomc_drift_v1.md (id=64) §2.4 + §2.5

Two parallel Sharpe computations:
  Method A — event-time series (PRIMARY): 80 events × annualized √8 multiplier
  Method B — daily TS series (SECONDARY): 2520 daily, 99% zeros (framework consistency)

5-gate post-audit framework: §2.5 gates 1-5.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Spec §六 locked
FOMC_EVENTS_PER_YEAR_LOCKED = 8
NW_LAG_EVENT_TIME_LOCKED    = 1
NW_LAG_DAILY_TS_LOCKED      = 60
N_BOOTSTRAP_RESAMPLES       = 1000
BOOTSTRAP_ALPHA             = 0.05

# 5-gate thresholds (spec §2.5)
GATE_1_SHARPE_THRESHOLD     = 0.5
GATE_1_NW_T_THRESHOLD       = 2.0
GATE_3_OOS_PCT_THRESHOLD    = 0.60   # OOS Sharpe ≥ 60% of in-sample
GATE_5_RESIDUAL_T_THRESHOLD = 1.5
GATE_5_RESIDUAL_SHARPE_TH   = 0.3

# OOS hold-out split
IN_SAMPLE_FRACTION_LOCKED   = 0.70

# Random rolling sub-period
RANDOM_ROLLING_N_WINDOWS    = 6
RANDOM_ROLLING_YEARS        = 3
RANDOM_ROLLING_SEED         = 20260512


def newey_west_t(returns: pd.Series, lag: int) -> float:
    """NW HAC t-stat for mean ≠ 0."""
    r = returns.dropna()
    if len(r) < lag + 5:
        return float("nan")
    mu = r.mean()
    n = len(r)
    gamma_0 = ((r - mu) ** 2).mean()
    nw_var = gamma_0
    for h in range(1, min(lag + 1, n)):
        w = 1.0 - h / (lag + 1)
        gamma_h = ((r - mu).iloc[h:].values * (r - mu).iloc[:n-h].values).mean()
        nw_var += 2 * w * gamma_h
    if nw_var <= 0:
        return float("nan")
    se = np.sqrt(nw_var / n)
    return float(mu / se) if se > 0 else float("nan")


def event_time_sharpe(event_returns: pd.Series, n_events_per_year: int) -> float:
    """Annualized event-time Sharpe = mean/std × sqrt(N events/year)."""
    r = event_returns.dropna()
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(mu / sd * np.sqrt(n_events_per_year))


def daily_ts_sharpe(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized daily Sharpe (mostly 0s for event strategy)."""
    r = daily_returns.dropna()
    if len(r) < 5:
        return float("nan")
    mu = r.mean() * periods_per_year
    sd = r.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mu / sd) if sd > 0 else float("nan")


def bootstrap_ci_event_sharpe(event_returns: pd.Series, n_events_per_year: int,
                              n_resamples: int = N_BOOTSTRAP_RESAMPLES,
                              alpha: float = BOOTSTRAP_ALPHA) -> tuple[float, float]:
    """Politis-Romano stationary bootstrap on event returns; return 95% CI for Sharpe."""
    r = event_returns.dropna().values
    if len(r) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(20260512)
    sharpes = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(r), size=len(r))
        boot = r[idx]
        mu = boot.mean()
        sd = boot.std(ddof=1)
        if sd > 0:
            sharpes.append(mu / sd * np.sqrt(n_events_per_year))
    if not sharpes:
        return (float("nan"), float("nan"))
    return (float(np.percentile(sharpes, alpha/2 * 100)),
            float(np.percentile(sharpes, (1 - alpha/2) * 100)))


def subperiod_regime_anchored(event_returns: pd.DataFrame) -> dict:
    """Regime-anchored: Pre-COVID 2014-2019 / COVID 2020-2021 / Post-COVID 2022-2023.
    Each sub-period: count events + event-time Sharpe + sign.
    """
    er = event_returns.copy()
    er['year'] = er['statement_release_date'].apply(lambda d: d.year)

    periods = {
        "pre_covid":  (2014, 2019),
        "covid":      (2020, 2021),
        "post_covid": (2022, 2023),
    }
    out = {}
    for label, (lo, hi) in periods.items():
        mask = (er['year'] >= lo) & (er['year'] <= hi)
        sub = er.loc[mask, 'basket_return_net']
        if len(sub) < 2:
            out[label] = {"n_events": int(len(sub)), "sharpe": None, "all_positive_sign": None}
            continue
        sh = event_time_sharpe(sub, FOMC_EVENTS_PER_YEAR_LOCKED)
        out[label] = {
            "n_events":         int(len(sub)),
            "sharpe":           float(sh) if not np.isnan(sh) else None,
            "all_positive_sign": bool(sh > 0) if not np.isnan(sh) else None,
            "mean_return":      float(sub.mean()),
        }

    all_positive = all(out[p]["all_positive_sign"] is True for p in periods)
    out["regime_anchored_all_positive"] = all_positive
    return out


def subperiod_random_rolling(daily_returns: pd.Series,
                             n_windows: int = RANDOM_ROLLING_N_WINDOWS,
                             window_years: int = RANDOM_ROLLING_YEARS,
                             seed: int = RANDOM_ROLLING_SEED) -> dict:
    """Random rolling 6 × 3-year windows on daily TS; all Sharpe ≥ 0?"""
    r = daily_returns.dropna()
    if not isinstance(r.index, pd.DatetimeIndex):
        return {"warning": "non-datetime index", "all_positive": None}
    window_size = 252 * window_years
    if len(r) < window_size + 5:
        return {"warning": "insufficient data", "all_positive": None}

    rng = np.random.default_rng(seed)
    max_start = len(r) - window_size
    starts = sorted(rng.integers(0, max_start, size=n_windows))

    window_sharpes = []
    for s in starts:
        window = r.iloc[s:s + window_size]
        sh = daily_ts_sharpe(window, periods_per_year=252)
        window_sharpes.append(float(sh) if not np.isnan(sh) else None)

    valid = [s for s in window_sharpes if s is not None]
    all_positive = all(s > 0 for s in valid) if valid else None
    return {
        "n_windows":         n_windows,
        "window_years":      window_years,
        "window_sharpes":    window_sharpes,
        "all_positive":      bool(all_positive) if all_positive is not None else None,
        "min_window_sharpe": float(min(valid)) if valid else None,
    }


def oos_hold_out_split(event_returns: pd.DataFrame,
                       in_sample_frac: float = IN_SAMPLE_FRACTION_LOCKED) -> dict:
    """70% in-sample / 30% OOS hold-out split by event order (time-respecting)."""
    er = event_returns.sort_values("statement_release_date").reset_index(drop=True)
    n = len(er)
    cut = int(n * in_sample_frac)
    in_sample = er.iloc[:cut]
    oos       = er.iloc[cut:]

    sh_in  = event_time_sharpe(in_sample['basket_return_net'], FOMC_EVENTS_PER_YEAR_LOCKED)
    sh_oos = event_time_sharpe(oos['basket_return_net'],       FOMC_EVENTS_PER_YEAR_LOCKED)

    ratio_oos_to_in = sh_oos / sh_in if (sh_in is not None and not np.isnan(sh_in) and sh_in > 0) else None
    pass_gate_3 = (ratio_oos_to_in is not None and ratio_oos_to_in >= GATE_3_OOS_PCT_THRESHOLD)

    return {
        "in_sample_n_events":  int(len(in_sample)),
        "oos_n_events":        int(len(oos)),
        "in_sample_sharpe":    float(sh_in)  if not np.isnan(sh_in)  else None,
        "oos_sharpe":          float(sh_oos) if not np.isnan(sh_oos) else None,
        "ratio_oos_to_in":     float(ratio_oos_to_in) if ratio_oos_to_in is not None else None,
        "gate_3_threshold":    GATE_3_OOS_PCT_THRESHOLD,
        "gate_3_pass":         bool(pass_gate_3),
    }


def incremental_alpha_vs_baseline(daily_strategy: pd.Series,
                                  baseline_daily: pd.Series,
                                  lag: int = NW_LAG_DAILY_TS_LOCKED) -> dict:
    """Regress strategy daily on baseline daily; residual α + t-stat + Sharpe."""
    df = pd.DataFrame({'y': daily_strategy, 'x': baseline_daily}).dropna()
    if len(df) < 30:
        return {"residual_alpha": None, "t_stat": None,
                "residual_sharpe": None, "n_obs": int(len(df)), "rho": None,
                "gate_5_pass": False}
    x = df['x'].values
    y = df['y'].values
    rho = float(np.corrcoef(x, y)[0, 1])
    slope, intercept, _, _, _ = stats.linregress(x, y)
    residuals = pd.Series(y - (slope * x + intercept), index=df.index)
    ann_alpha = float(intercept * 252)
    residual_sharpe = daily_ts_sharpe(residuals, 252)
    nw_t = newey_west_t(residuals, lag=lag)
    gate_5 = (nw_t is not None and not np.isnan(nw_t) and nw_t >= GATE_5_RESIDUAL_T_THRESHOLD and
              residual_sharpe is not None and not np.isnan(residual_sharpe) and
              residual_sharpe >= GATE_5_RESIDUAL_SHARPE_TH)
    return {
        "residual_alpha_ann":  ann_alpha,
        "t_stat":              float(nw_t) if not np.isnan(nw_t) else None,
        "residual_sharpe":     float(residual_sharpe) if not np.isnan(residual_sharpe) else None,
        "n_obs":               int(len(df)),
        "rho":                 rho,
        "beta":                float(slope),
        "gate_5_threshold_t":  GATE_5_RESIDUAL_T_THRESHOLD,
        "gate_5_threshold_sharpe": GATE_5_RESIDUAL_SHARPE_TH,
        "gate_5_pass":         bool(gate_5),
    }


@dataclass
class PreFOMCVerdict:
    decision:                  str
    spec_hash:                  str
    wave:                       str
    universe_source:            str
    window_start:               str
    window_end:                 str
    n_events:                   int
    # Method A — primary
    method_A_sharpe_net:        Optional[float]
    method_A_nw_t:              Optional[float]
    method_A_ci_lower:          Optional[float]
    method_A_ci_upper:          Optional[float]
    method_A_ann_return:        Optional[float]
    method_A_ann_vol:           Optional[float]
    # Method B — secondary
    method_B_sharpe_net:        Optional[float]
    method_B_nw_t:              Optional[float]
    # Sub-period
    subperiod_regime:           dict
    subperiod_random_rolling:   dict
    # OOS
    oos_hold_out:               dict
    # Incremental α
    incremental_alpha_vs_K1:    dict
    # 5-gate summary
    gate_1_individual_pass:     bool
    gate_2_selective_bhy:       str
    gate_3_oos_pass:            bool
    gate_4_subperiod_pass:      bool
    gate_5_incremental_pass:    bool
    cumulative_return:          float
    max_drawdown:               float
    honest_disclose:            list[str]


def build_pre_fomc_verdict(
    event_returns:        pd.DataFrame,
    daily_strategy:       pd.DataFrame,
    baseline_daily:       pd.Series,
    spec_hash:            str,
    universe_source:      str,
    window_start:         datetime.date,
    window_end:           datetime.date,
) -> PreFOMCVerdict:
    """Build comprehensive Pre-FOMC verdict with 5-gate post-audit evaluation."""

    # Method A
    ret_A = event_returns['basket_return_net']
    sh_A = event_time_sharpe(ret_A, FOMC_EVENTS_PER_YEAR_LOCKED)
    nw_A = newey_west_t(ret_A, lag=NW_LAG_EVENT_TIME_LOCKED)
    ci_lo, ci_hi = bootstrap_ci_event_sharpe(ret_A, FOMC_EVENTS_PER_YEAR_LOCKED)
    ann_ret_A = float(ret_A.mean() * FOMC_EVENTS_PER_YEAR_LOCKED)
    ann_vol_A = float(ret_A.std(ddof=1) * np.sqrt(FOMC_EVENTS_PER_YEAR_LOCKED))

    # Method B
    ret_B = daily_strategy['strategy_return']
    sh_B = daily_ts_sharpe(ret_B, 252)
    nw_B = newey_west_t(ret_B, lag=NW_LAG_DAILY_TS_LOCKED)

    # Sub-period
    subp_regime = subperiod_regime_anchored(event_returns)
    subp_rolling = subperiod_random_rolling(ret_B)

    # OOS
    oos = oos_hold_out_split(event_returns)

    # Incremental α
    incr_alpha = incremental_alpha_vs_baseline(ret_B, baseline_daily)

    # Gates
    gate_1 = bool(sh_A is not None and not np.isnan(sh_A) and sh_A >= GATE_1_SHARPE_THRESHOLD and
                  nw_A is not None and not np.isnan(nw_A) and nw_A >= GATE_1_NW_T_THRESHOLD)
    gate_3 = bool(oos["gate_3_pass"])
    gate_4 = bool(subp_regime.get("regime_anchored_all_positive") and
                  subp_rolling.get("all_positive") is True)
    gate_5 = bool(incr_alpha["gate_5_pass"])

    # Decision
    if not gate_1:
        decision = "FAIL"
    elif not (gate_3 and gate_4 and gate_5):
        decision = "INDIVIDUAL_PASS_BUT_NON_INDEPENDENT"
    else:
        decision = "PASS_INDEPENDENT"

    # Cumulative + DD on daily TS
    daily_cum = (1 + ret_B).cumprod()
    cum_ret = float(daily_cum.iloc[-1] - 1.0) if len(daily_cum) > 0 else 0.0
    rolling_max = daily_cum.cummax()
    drawdown = (daily_cum - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    return PreFOMCVerdict(
        decision=decision,
        spec_hash=spec_hash,
        wave="E-pre-fomc-drift",
        universe_source=universe_source,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        n_events=int(len(event_returns)),
        method_A_sharpe_net=float(sh_A) if not np.isnan(sh_A) else None,
        method_A_nw_t=float(nw_A) if not np.isnan(nw_A) else None,
        method_A_ci_lower=ci_lo if not np.isnan(ci_lo) else None,
        method_A_ci_upper=ci_hi if not np.isnan(ci_hi) else None,
        method_A_ann_return=ann_ret_A,
        method_A_ann_vol=ann_vol_A,
        method_B_sharpe_net=float(sh_B) if not np.isnan(sh_B) else None,
        method_B_nw_t=float(nw_B) if not np.isnan(nw_B) else None,
        subperiod_regime=subp_regime,
        subperiod_random_rolling=subp_rolling,
        oos_hold_out=oos,
        incremental_alpha_vs_K1=incr_alpha,
        gate_1_individual_pass=gate_1,
        gate_2_selective_bhy="DEMOTED_SINGLE_TEST",
        gate_3_oos_pass=gate_3,
        gate_4_subperiod_pass=gate_4,
        gate_5_incremental_pass=gate_5,
        cumulative_return=cum_ret,
        max_drawdown=max_dd,
        honest_disclose=[
            "Lucca-Moench 2015 original window 1994-2014; we extend 2014-2023 with significant post-publication arbitrage period overlap",
            "Post-publication decay risk: paper published 2015, 2014-2023 effectively half post-publication OOS",
            "2020 emergency FOMC (Mar 3, Mar 15) EXCLUDED to preserve event regularity",
            "80 events moderate sample; OOS hold-out 24 events small (wide CI expected)",
            "Equity-only universe (K1 subset); does not test pre-FOMC drift on bonds/commodities/FX",
            "Daily-close to daily-close approximation of t-24h-to-announcement; intraday version (Lucca-Moench original) requires minute-level data out of scope",
        ],
    )
