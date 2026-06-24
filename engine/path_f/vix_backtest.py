"""
engine/path_f/vix_backtest.py — Path F backtest + 5-gate verdict.

Pre-registration: docs/spec_path_f_vix_term_structure_v1.md (id=65) §2.5-2.7
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Spec §六 locked
NW_LAG_DAILY_LOCKED                   = 60
BOOTSTRAP_N_RESAMPLES                 = 1000
BOOTSTRAP_ALPHA                       = 0.05
GATE_1_SHARPE_THRESHOLD               = 0.4    # ETF sleeve per feedback_sleeve_specific_pass_gates standing rule (was 0.5 single-stock paradigm)
GATE_1_NW_T_THRESHOLD                 = 1.8    # ETF sleeve (was 2.0)
GATE_3_OOS_PCT_THRESHOLD              = 0.60
GATE_5_RESIDUAL_T_THRESHOLD           = 1.5
GATE_5_RESIDUAL_SHARPE_THRESHOLD      = 0.3
IN_SAMPLE_FRACTION_LOCKED             = 0.70
RANDOM_ROLLING_N_WINDOWS              = 6
RANDOM_ROLLING_YEARS                  = 3
RANDOM_ROLLING_SEED                   = 20260512


def newey_west_t(returns: pd.Series, lag: int) -> float:
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


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 5:
        return float("nan")
    mu = r.mean() * periods_per_year
    sd = r.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mu / sd) if sd > 0 else float("nan")


def bootstrap_ci_sharpe(returns: pd.Series, periods_per_year: int = 252,
                        n_resamples: int = BOOTSTRAP_N_RESAMPLES,
                        alpha: float = BOOTSTRAP_ALPHA) -> tuple[float, float]:
    r = returns.dropna().values
    if len(r) < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(20260512)
    sharpes = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(r), size=len(r))
        boot = r[idx]
        mu = boot.mean() * periods_per_year
        sd = boot.std(ddof=1) * np.sqrt(periods_per_year)
        if sd > 0:
            sharpes.append(mu / sd)
    if not sharpes:
        return (float("nan"), float("nan"))
    return (float(np.percentile(sharpes, alpha/2 * 100)),
            float(np.percentile(sharpes, (1 - alpha/2) * 100)))


def regime_sub_period(daily_returns: pd.Series) -> dict:
    """Pre-COVID / COVID / Post-COVID Sharpe per spec §2.6 Gate 4 Method A."""
    r = daily_returns.copy()
    r.index = pd.to_datetime(r.index)
    periods = {
        "pre_covid":  (datetime.date(2014, 1, 1), datetime.date(2019, 12, 31)),
        "covid":      (datetime.date(2020, 1, 1), datetime.date(2021, 12, 31)),
        "post_covid": (datetime.date(2022, 1, 1), datetime.date(2023, 12, 31)),
    }
    out = {}
    for label, (lo, hi) in periods.items():
        mask = (r.index.date >= lo) & (r.index.date <= hi)
        seg = r[mask].dropna()
        if len(seg) < 5:
            out[label] = {"n_obs": int(len(seg)), "sharpe": None}
            continue
        sh = annualized_sharpe(seg, 252)
        out[label] = {"n_obs": int(len(seg)), "sharpe": float(sh) if not np.isnan(sh) else None}
    all_positive = all(out[p]["sharpe"] is not None and out[p]["sharpe"] >= 0 for p in periods)
    out["regime_all_positive"] = bool(all_positive)
    return out


def random_rolling_sub_period(daily_returns: pd.Series,
                              n_windows: int = RANDOM_ROLLING_N_WINDOWS,
                              window_years: int = RANDOM_ROLLING_YEARS,
                              seed: int = RANDOM_ROLLING_SEED) -> dict:
    r = daily_returns.dropna()
    window_size = 252 * window_years
    if len(r) < window_size + 5:
        return {"warning": "insufficient data", "all_positive": None}
    rng = np.random.default_rng(seed)
    starts = sorted(rng.integers(0, len(r) - window_size, size=n_windows))
    sharpes = []
    for s in starts:
        window = r.iloc[s:s + window_size]
        sharpes.append(float(annualized_sharpe(window, 252)) if not np.isnan(annualized_sharpe(window, 252)) else None)
    valid = [s for s in sharpes if s is not None]
    all_positive = all(s >= 0 for s in valid) if valid else None
    return {"n_windows": n_windows, "window_years": window_years,
            "window_sharpes": sharpes, "all_positive": bool(all_positive) if all_positive is not None else None,
            "min_window_sharpe": float(min(valid)) if valid else None}


def oos_hold_out(daily_returns: pd.Series) -> dict:
    n = len(daily_returns)
    cut = int(n * IN_SAMPLE_FRACTION_LOCKED)
    in_sample = daily_returns.iloc[:cut]
    oos       = daily_returns.iloc[cut:]
    sh_in  = annualized_sharpe(in_sample, 252)
    sh_oos = annualized_sharpe(oos, 252)
    ratio = sh_oos / sh_in if (not np.isnan(sh_in) and sh_in > 0) else None
    gate_3 = ratio is not None and ratio >= GATE_3_OOS_PCT_THRESHOLD
    return {
        "in_sample_n":  int(len(in_sample)),
        "oos_n":        int(len(oos)),
        "in_sample_sharpe":  float(sh_in) if not np.isnan(sh_in) else None,
        "oos_sharpe":        float(sh_oos) if not np.isnan(sh_oos) else None,
        "ratio_oos_to_in":   float(ratio) if ratio is not None else None,
        "gate_3_pass":       bool(gate_3),
    }


def incremental_alpha_vs_baseline(strategy_daily: pd.Series, baseline_daily: pd.Series,
                                  lag: int = NW_LAG_DAILY_LOCKED) -> dict:
    """Compute incremental alpha test (CAPM-style intercept test + IR).

    Bug fix 2026-05-12: original implementation used Sharpe(residuals) but
    OLS with intercept makes mean(residuals) = 0 by construction → always 0.
    Correct academic standard is Information Ratio = α / σ(residuals),
    where α = intercept (annualized). NW t-stat for α uses HAC SE.
    """
    df = pd.DataFrame({'y': strategy_daily, 'x': baseline_daily}).dropna()
    n = len(df)
    if n < 30:
        return {"gate_5_pass": False, "n_obs": int(n)}
    x = df['x'].values
    y = df['y'].values
    rho = float(np.corrcoef(x, y)[0, 1])
    slope, intercept, *_ = stats.linregress(x, y)
    residuals = pd.Series(y - (slope * x + intercept), index=df.index)

    ann_alpha = float(intercept * 252)
    residual_std_daily = float(residuals.std(ddof=1))
    residual_std_ann = residual_std_daily * np.sqrt(252)
    # Information Ratio (academic standard for incremental alpha magnitude)
    information_ratio = ann_alpha / residual_std_ann if residual_std_ann > 0 else float("nan")

    # NW HAC standard error for intercept (α)
    # Var(α_HAC) ≈ NW variance of residuals / n (when x is small/orthogonal to α)
    mu_resid = 0.0  # OLS residuals mean by construction
    gamma_0 = ((residuals - mu_resid) ** 2).mean()
    nw_var = gamma_0
    for h in range(1, min(lag + 1, n)):
        w = 1.0 - h / (lag + 1)
        gamma_h = ((residuals - mu_resid).iloc[h:].values *
                   (residuals - mu_resid).iloc[:n-h].values).mean()
        nw_var += 2 * w * gamma_h
    se_alpha_daily = np.sqrt(nw_var / n) if nw_var > 0 else float("nan")
    nw_t_alpha = intercept / se_alpha_daily if (se_alpha_daily > 0 and not np.isnan(se_alpha_daily)) else float("nan")

    gate_5 = (not np.isnan(nw_t_alpha) and nw_t_alpha >= GATE_5_RESIDUAL_T_THRESHOLD and
              not np.isnan(information_ratio) and information_ratio >= GATE_5_RESIDUAL_SHARPE_THRESHOLD)

    return {
        "n_obs":              int(n),
        "rho":                rho,
        "beta":               float(slope),
        "residual_alpha_ann": ann_alpha,
        "information_ratio":  float(information_ratio) if not np.isnan(information_ratio) else None,
        "t_stat":             float(nw_t_alpha) if not np.isnan(nw_t_alpha) else None,
        "gate_5_pass":        bool(gate_5),
        "bug_fix_note":       "v2 IR formula (α/σ_residual annual); v1 used Sharpe(residuals)=0 by OLS",
    }


@dataclass
class PathFVerdict:
    decision:                        str
    spec_hash:                       str
    wave:                            str
    universe_source:                 str
    window_start:                    str
    window_end:                      str
    n_daily_obs:                     int
    n_trades:                        int
    n_stop_loss_triggers:            int
    method_A_sharpe_net:             Optional[float]
    method_A_nw_t:                   Optional[float]
    method_A_ci_lower:               Optional[float]
    method_A_ci_upper:               Optional[float]
    method_A_ann_return:             Optional[float]
    method_A_ann_vol:                Optional[float]
    method_B_n_trades:               int
    method_B_mean_trade_return:      Optional[float]
    subperiod_regime:                dict
    subperiod_random_rolling:        dict
    oos_hold_out:                    dict
    incremental_alpha_vs_K1:         dict
    gate_1_individual_pass:          bool
    gate_2_selective_bhy:            str
    gate_3_oos_pass:                 bool
    gate_4_subperiod_pass:           bool
    gate_5_incremental_pass:         bool
    cumulative_return:               float
    max_drawdown:                    float
    stop_loss_events:                list
    honest_disclose:                 list
