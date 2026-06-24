"""
engine/b_plus_phase_d.py — Phase D: Fama-MacBeth Factor Decomposition
=======================================================================

Spec: docs/spec_b_plus_mass_fdr_search.md v2.0 §15.

Decomposes per-strategy weekly returns into Fama-French factor exposures + Jensen's
alpha (residual). Identifies how much of each strategy's "alpha" is actually
factor exposure (market β / size / value / momentum / quality).

Implementation choice (per spec §15.4):
  ETF-proxy factors are used in lieu of Kenneth French data library to keep
  data dependencies internal to the project's universe.

  Mkt-RF  ≈ SPY excess return  (use SPY directly; risk-free assumed 0 for weekly)
  SMB     ≈ IWM (small-cap) - IWB (large-cap proxy via SPY)
  HML     ≈ IWN (value) - IWO (growth)
  MOM     ≈ MTUM (momentum factor ETF) - low-mom proxy (USMV)
  QMJ     ≈ QUAL (quality factor) - low-quality proxy (rest of universe avg)

This approximation introduces noise vs canonical Kenneth French data but is
academically defensible at master's-project scope. Documented in decision doc.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = "data/b_plus_results"


# ─────────────────────────────────────────────────────────────────────────────
# ETF-proxy Fama-French factor construction
# ─────────────────────────────────────────────────────────────────────────────

FF_PROXY_TICKERS = {
    "market":       "SPY",   # market excess return
    "size_long":    "IWM",   # small-cap (Russell 2000)
    "size_short":   "SPY",   # large-cap proxy (use SPY again as benchmark)
    "value_long":   "IWN",   # Russell 2000 Value
    "value_short":  "IWO",   # Russell 2000 Growth
    "mom_long":     "MTUM",  # MSCI USA Momentum
    "mom_short":    "USMV",  # MSCI USA Min Vol (low-momentum proxy)
    "qual_long":    "QUAL",  # MSCI USA Quality
    "qual_short":   "USMV",  # again as quality complement (imperfect proxy)
}


def fetch_ff_factor_returns(
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    """
    Fetch weekly returns for FF-proxy ETFs over the given window.
    Returns wide DataFrame: index = Friday dates, columns = factor name + "_long"/"_short".
    """
    from engine.signal import _fetch_closes

    tickers_needed = list(set(FF_PROXY_TICKERS.values()))
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=14)).date()
    fetch_end   = pd.Timestamp(end_date).date()

    closes = _fetch_closes(tickers_needed, fetch_start, fetch_end)
    if closes.empty:
        return pd.DataFrame()

    # Resample to weekly Friday close, compute weekly returns
    weekly_closes = closes.resample("W-FRI").last()
    weekly_rets = weekly_closes.pct_change(fill_method=None).dropna(how="all")
    return weekly_rets


def construct_ff_factors_from_etfs(
    etf_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct FF-style factors from ETF weekly returns.

    Returns DataFrame with columns:
      MKT_RF: market excess (SPY return; rf assumed 0 weekly)
      SMB:    IWM - SPY
      HML:    IWN - IWO
      MOM:    MTUM - USMV
      QMJ:    QUAL - USMV
    """
    out = pd.DataFrame(index=etf_returns.index)

    if "SPY" in etf_returns.columns:
        out["MKT_RF"] = etf_returns["SPY"]
    if "IWM" in etf_returns.columns and "SPY" in etf_returns.columns:
        out["SMB"] = etf_returns["IWM"] - etf_returns["SPY"]
    if "IWN" in etf_returns.columns and "IWO" in etf_returns.columns:
        out["HML"] = etf_returns["IWN"] - etf_returns["IWO"]
    if "MTUM" in etf_returns.columns and "USMV" in etf_returns.columns:
        out["MOM"] = etf_returns["MTUM"] - etf_returns["USMV"]
    if "QUAL" in etf_returns.columns and "USMV" in etf_returns.columns:
        out["QMJ"] = etf_returns["QUAL"] - etf_returns["USMV"]

    return out.dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────────
# Time-series factor regression (per strategy)
# ─────────────────────────────────────────────────────────────────────────────

def regress_strategy_on_factors(
    strategy_returns: pd.Series,
    factor_returns:   pd.DataFrame,
) -> dict:
    """
    Run OLS time-series regression:
      strategy_returns_t = α + β_MKT × MKT_RF_t + β_SMB × SMB_t + β_HML × HML_t
                              + β_MOM × MOM_t + β_QMJ × QMJ_t + ε_t

    Returns dict:
      alpha_per_period        : intercept α
      alpha_annualized        : α × 52
      betas                   : dict factor_name → β
      t_stats                 : dict factor_name → t-stat (NW HAC)
      r2                      : R² of regression
      n                       : sample size
      residual_sharpe         : annualised Sharpe of residual ε (pure alpha)
      residual_t_stat         : NW HAC t-stat of residual mean
    """
    from engine.backtest import newey_west_sharpe_se

    # Align
    aligned = pd.concat({"y": strategy_returns}, axis=1)
    for col in factor_returns.columns:
        aligned[col] = factor_returns[col]
    aligned = aligned.dropna()
    n = len(aligned)
    if n < 12:
        return {"error": "insufficient_n", "n": n}

    Y = aligned["y"].values
    factor_names = [c for c in factor_returns.columns if c in aligned.columns]
    X_mat = aligned[factor_names].values
    X_with_intercept = np.column_stack([np.ones(n), X_mat])

    # OLS
    XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
    coefs = XtX_inv @ X_with_intercept.T @ Y
    alpha = float(coefs[0])
    betas = {name: float(c) for name, c in zip(factor_names, coefs[1:])}

    # Residuals
    y_pred = X_with_intercept @ coefs
    residuals = Y - y_pred

    # R²
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # NW HAC standard errors for coefs (Newey-West with lag = floor(4(n/100)^(2/9)))
    sigma2 = ss_res / max(n - len(factor_names) - 1, 1)
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    nw_var_coef = sigma2 * np.diag(XtX_inv)
    # NW HAC enhancement (Bartlett kernel)
    if L > 0 and n > L * 2:
        for lag in range(1, L + 1):
            w = 1.0 - lag / (L + 1)
            for i in range(len(factor_names) + 1):
                # Computing full HAC on cov matrix is heavy; use simplified variance scaling
                cov_lag = float(np.cov(residuals[lag:], residuals[:-lag])[0, 1])
                nw_var_coef[i] += 2 * w * cov_lag * XtX_inv[i, i]
    se_coef = np.sqrt(np.maximum(nw_var_coef, 1e-18))

    t_stats = {
        "alpha": float(alpha / max(se_coef[0], 1e-12)),
    }
    for i, name in enumerate(factor_names, start=1):
        t_stats[name] = float(coefs[i] / max(se_coef[i], 1e-12))

    # Residual Sharpe (pure alpha after factor exposure)
    residual_series = pd.Series(residuals, index=aligned.index)
    res_mu = float(residual_series.mean())
    res_sd = float(residual_series.std(ddof=1))
    residual_sharpe = (res_mu / res_sd) * np.sqrt(52) if res_sd > 1e-12 else 0.0
    residual_nw = newey_west_sharpe_se(residual_series.values, periods_per_year=52)

    return {
        "alpha_per_period":     alpha,
        "alpha_annualized":     float(alpha * 52),
        "alpha_t_stat":         t_stats["alpha"],
        "betas":                betas,
        "t_stats":              t_stats,
        "factor_names":         factor_names,
        "r2":                   r2,
        "n":                    n,
        "residual_sharpe":      residual_sharpe,
        "residual_nw_t":        float(residual_nw.get("t_stat", float("nan"))),
        "residual_ann_return":  float(res_mu * 52),
        "residual_ann_vol":     float(res_sd * np.sqrt(52)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional Fama-MacBeth (advanced, per spec §15.2)
# ─────────────────────────────────────────────────────────────────────────────

def fama_macbeth_cross_sectional(
    panel_returns:  pd.DataFrame,    # T × N matrix of asset returns
    factor_betas:   pd.DataFrame,    # N × K matrix of asset factor exposures (time-invariant)
) -> dict:
    """
    Classical Fama-MacBeth two-pass:
      Pass 1: For each asset i, regress returns on factors → get factor loadings β_i
      Pass 2: For each time t, cross-sectional regression of returns on β → get factor premia λ_t
              Average λ_t across t → factor risk premium
              t-stat = mean / (std/√T)

    Returns dict with factor_premia, t_stats, n_periods, n_assets.

    NOTE: For the B++ project, we use the simpler time-series regression
    (regress_strategy_on_factors) which is sufficient. This function is
    provided for completeness per spec §15.2 but not invoked by default.
    """
    if panel_returns.empty or factor_betas.empty:
        return {"error": "empty_input"}

    common_assets = panel_returns.columns.intersection(factor_betas.index)
    if len(common_assets) < 5:
        return {"error": "insufficient_assets"}

    panel = panel_returns[common_assets]
    betas = factor_betas.loc[common_assets]

    # Pass 2: cross-sectional regression each period
    T = len(panel)
    K = betas.shape[1]
    lambda_t = np.zeros((T, K))

    for t_idx in range(T):
        Y_t = panel.iloc[t_idx].values
        # Drop NaN
        valid = ~np.isnan(Y_t)
        if valid.sum() < K + 1:
            lambda_t[t_idx, :] = np.nan
            continue
        X_t = betas.values[valid]
        Y_t_v = Y_t[valid]
        # Add intercept
        X_with_int = np.column_stack([np.ones(valid.sum()), X_t])
        try:
            coefs = np.linalg.pinv(X_with_int.T @ X_with_int) @ X_with_int.T @ Y_t_v
            lambda_t[t_idx, :] = coefs[1:]   # skip intercept
        except Exception:
            lambda_t[t_idx, :] = np.nan

    # Time-series mean of factor premia
    factor_premia = np.nanmean(lambda_t, axis=0)
    factor_std    = np.nanstd(lambda_t, axis=0, ddof=1)
    valid_t       = (~np.isnan(lambda_t)).sum(axis=0)
    t_stats       = factor_premia / (factor_std / np.sqrt(valid_t))

    return {
        "factor_premia":   {betas.columns[k]: float(factor_premia[k]) for k in range(K)},
        "factor_premia_annualized": {betas.columns[k]: float(factor_premia[k] * 52) for k in range(K)},
        "t_stats":         {betas.columns[k]: float(t_stats[k]) for k in range(K)},
        "n_periods":       T,
        "n_assets":        len(common_assets),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase D orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_d(
    results_dir: str = DEFAULT_RESULTS_DIR,
    train_start: str = "2010-01-01",
    end_date:    str = "2024-12-31",
) -> dict:
    """
    Full Phase D pipeline: fetch FF factors + per-strategy decomposition.

    Persists:
      data/b_plus_results/phase_d_factors.csv          (FF factor weekly returns)
      data/b_plus_results/phase_d_decomposition.csv    (per-strategy alpha + βs + R²)
    """
    from engine.b_plus_phase_c import load_per_spec, load_all_oos_returns

    # 1) Fetch ETF-proxy FF factor returns
    etf_rets = fetch_ff_factor_returns(train_start, end_date)
    if etf_rets.empty:
        return {"error": "ff_etf_fetch_failed"}
    ff_factors = construct_ff_factors_from_etfs(etf_rets)
    ff_factors.to_csv(os.path.join(results_dir, "phase_d_factors.csv"))

    # 2) Load all strategy OOS returns
    returns_wide = load_all_oos_returns(results_dir)
    if returns_wide.empty:
        return {"error": "no_oos_returns"}

    # 3) Per-strategy time-series decomposition
    rows = []
    for spec_label in returns_wide.columns:
        s_returns = returns_wide[spec_label].dropna()
        if len(s_returns) < 12:
            continue
        result = regress_strategy_on_factors(s_returns, ff_factors)
        if "error" in result:
            rows.append({"spec_label": spec_label, "error": result["error"]})
            continue
        row = {
            "spec_label":            spec_label,
            "alpha_annualized":      result["alpha_annualized"],
            "alpha_t_stat":          result["alpha_t_stat"],
            "r2":                    result["r2"],
            "n":                     result["n"],
            "residual_sharpe":       result["residual_sharpe"],
            "residual_nw_t":         result["residual_nw_t"],
            "residual_ann_return":   result["residual_ann_return"],
            "residual_ann_vol":      result["residual_ann_vol"],
        }
        # Add per-factor βs
        for fname, beta in result["betas"].items():
            row[f"beta_{fname}"] = beta
            row[f"t_{fname}"] = result["t_stats"].get(fname, float("nan"))
        rows.append(row)

    decomp_df = pd.DataFrame(rows)
    decomp_df.to_csv(os.path.join(results_dir, "phase_d_decomposition.csv"), index=False)

    # 4) Summary stats
    valid = decomp_df[decomp_df["alpha_annualized"].notna()] if "alpha_annualized" in decomp_df else pd.DataFrame()
    return {
        "n_specs_decomposed":      len(valid),
        "median_r2":               float(valid["r2"].median()) if not valid.empty else float("nan"),
        "median_alpha_ann":        float(valid["alpha_annualized"].median()) if not valid.empty else float("nan"),
        "median_residual_sharpe":  float(valid["residual_sharpe"].median()) if not valid.empty else float("nan"),
        "n_alpha_significant_5pct": int((valid["alpha_t_stat"] > 1.96).sum()) if not valid.empty else 0,
        "factor_columns_in_decomp": [c for c in decomp_df.columns if c.startswith("beta_")],
        "output_files": [
            "data/b_plus_results/phase_d_factors.csv",
            "data/b_plus_results/phase_d_decomposition.csv",
        ],
    }
