"""
engine/b_plus_phase_c.py — Phase C: Multi-Strategy Combination Layer
====================================================================

Spec: docs/spec_b_plus_mass_fdr_search.md v2.0 §14.

Operates on output of `engine.b_plus_search.run_mass_search`:
  - data/b_plus_results/per_spec.csv
  - data/b_plus_results/{spec_label}_oos_returns.csv

Implements:
  C.1 IC + ICIR per strategy (already in per_spec.csv)
  C.2 Strategy correlation matrix on weekly OOS returns
  C.3 IC-weighted meta-strategy via mean-variance optimization
  C.4 Beta-neutralized long-short (per spec §14.4)

All operations are POST-BACKTEST aggregation; do not re-run backtests.
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
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_per_spec(results_dir: str = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Load per-spec aggregated results."""
    path = os.path.join(results_dir, "per_spec.csv")
    return pd.read_csv(path)


def load_oos_returns(spec_label: str, results_dir: str = DEFAULT_RESULTS_DIR) -> pd.Series:
    """Load weekly OOS returns for a single spec."""
    path = os.path.join(results_dir, f"{spec_label}_oos_returns.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["return"]


def load_all_oos_returns(results_dir: str = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Load all OOS returns into a wide DataFrame (date index, spec_label columns)."""
    spec_df = load_per_spec(results_dir)
    out = {}
    for spec_label in spec_df["spec_label"]:
        s = load_oos_returns(spec_label, results_dir)
        if not s.empty:
            out[spec_label] = s
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# C.2 Strategy Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_strategy_correlation(
    returns_wide: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix of strategy weekly returns.

    method ∈ {"pearson", "spearman", "kendall"}
    """
    return returns_wide.corr(method=method)


def identify_redundant_strategies(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """
    Identify strategy pairs with |correlation| > threshold (potentially redundant).
    Returns list of (spec_a, spec_b, correlation) sorted descending.
    """
    pairs = []
    n = corr_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix.iloc[i, j]
            if not np.isnan(r) and abs(r) > threshold:
                pairs.append((corr_matrix.index[i], corr_matrix.columns[j], float(r)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# C.3 IC-Weighted Meta-Strategy (mean-variance optimization)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic_weighted_meta(
    returns_wide: pd.DataFrame,
    spec_df:      pd.DataFrame,
    target_vol:   float = 0.10,
    use_ic_for_alpha: bool = True,
    ridge_lambda: float = 1e-4,
) -> dict:
    """
    Compute IC-weighted meta-strategy weights via mean-variance optimization.

    Per spec §14.3 — Markowitz (1952) + Grinold-Kahn (1999) IR optimization:
        w* ∝ inv(Σ) × IC_vector
        scaled to portfolio target vol = 10% annualised

    Args:
        returns_wide: T×K matrix of strategy weekly returns
        spec_df:     per-spec results (must include 'oos_ic_mean' column)
        target_vol:  annualised target vol for the meta-strategy (default 10%)
        use_ic_for_alpha: True → use IC mean as alpha vector (Grinold-Kahn);
                          False → use OOS Sharpe as alpha
        ridge_lambda: ridge regularization on Σ for numerical stability

    Returns dict with:
        weights      : pd.Series of meta weights (sum to non-zero, can be negative)
        meta_returns : pd.Series of meta-strategy weekly returns
        meta_sharpe  : annualised Sharpe of meta
        meta_nw_t    : NW HAC t-stat of meta
        ic_vector    : alpha vector used
        cov_matrix   : Σ used
        n_strategies : count
    """
    from engine.backtest import newey_west_sharpe_se

    if returns_wide.empty:
        return {"error": "no returns"}

    # Drop strategies with insufficient data
    valid_specs = returns_wide.columns[returns_wide.count() >= 12].tolist()
    if not valid_specs:
        return {"error": "no_valid_strategies"}

    R = returns_wide[valid_specs].fillna(0.0)
    K = len(valid_specs)

    # Covariance matrix of strategy returns
    Sigma = R.cov(ddof=1).values
    Sigma += ridge_lambda * np.eye(K)   # ridge regularization

    # Alpha vector
    if use_ic_for_alpha:
        ic_lookup = spec_df.set_index("spec_label")["oos_ic_mean"].to_dict()
        ic_vec = np.array([ic_lookup.get(s, 0.0) for s in valid_specs])
    else:
        sharpe_lookup = spec_df.set_index("spec_label")["oos_sharpe"].to_dict()
        ic_vec = np.array([sharpe_lookup.get(s, 0.0) for s in valid_specs])

    ic_vec = np.nan_to_num(ic_vec, nan=0.0)

    # Mean-variance optimal weights: w ∝ Σ^-1 × ic
    Sigma_inv = np.linalg.pinv(Sigma)
    raw_weights = Sigma_inv @ ic_vec

    # Compute meta returns at raw weights to determine vol
    meta_raw = R.values @ raw_weights
    raw_ann_vol = float(np.std(meta_raw, ddof=1) * np.sqrt(52))
    if raw_ann_vol < 1e-6:
        return {"error": "raw_meta_zero_vol"}

    # Scale to target vol
    scale = target_vol / raw_ann_vol
    weights = raw_weights * scale
    meta_returns = pd.Series(R.values @ weights, index=R.index, name="meta_return")

    # Stats
    mu = float(meta_returns.mean())
    sd = float(meta_returns.std(ddof=1))
    meta_sharpe = (mu / sd) * np.sqrt(52) if sd > 1e-12 else 0.0
    nw = newey_west_sharpe_se(meta_returns.values, periods_per_year=52)

    return {
        "weights":      pd.Series(weights, index=valid_specs).sort_values(key=abs, ascending=False),
        "meta_returns": meta_returns,
        "meta_sharpe":  meta_sharpe,
        "meta_ann_return": float(mu * 52),
        "meta_ann_vol": float(sd * np.sqrt(52)),
        "meta_nw_t":    float(nw.get("t_stat", float("nan"))),
        "meta_nw_ci":   (float(nw.get("ci_low", float("nan"))), float(nw.get("ci_high", float("nan")))),
        "ic_vector":    pd.Series(ic_vec, index=valid_specs),
        "n_strategies": K,
    }


def compute_erc_meta(returns_wide: pd.DataFrame, target_vol: float = 0.10) -> dict:
    """
    Equal Risk Contribution meta-strategy across strategies.

    Each strategy contributes equal risk to the meta-portfolio. Simpler alternative
    to mean-variance — does not require alpha vector. Pure diversification play.

    Returns same dict structure as compute_ic_weighted_meta.
    """
    from engine.backtest import newey_west_sharpe_se

    if returns_wide.empty:
        return {"error": "no_returns"}

    valid_specs = returns_wide.columns[returns_wide.count() >= 12].tolist()
    if not valid_specs:
        return {"error": "no_valid_strategies"}

    R = returns_wide[valid_specs].fillna(0.0)
    K = len(valid_specs)
    Sigma = R.cov(ddof=1).values

    # ERC: solve w_i × (Σw)_i = constant, all i. SLSQP iteration.
    from scipy.optimize import minimize

    def objective(w):
        Sw = Sigma @ w
        rc = w * Sw
        target_rc = float(np.mean(rc))
        return float(np.sum((rc - target_rc) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * K
    w0 = np.ones(K) / K
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    raw_weights = res.x

    meta_raw = R.values @ raw_weights
    raw_ann_vol = float(np.std(meta_raw, ddof=1) * np.sqrt(52))
    if raw_ann_vol < 1e-6:
        return {"error": "raw_meta_zero_vol"}
    scale = target_vol / raw_ann_vol
    weights = raw_weights * scale
    meta_returns = pd.Series(R.values @ weights, index=R.index, name="erc_meta_return")

    mu = float(meta_returns.mean())
    sd = float(meta_returns.std(ddof=1))
    erc_sharpe = (mu / sd) * np.sqrt(52) if sd > 1e-12 else 0.0
    nw = newey_west_sharpe_se(meta_returns.values, periods_per_year=52)

    return {
        "weights":         pd.Series(weights, index=valid_specs).sort_values(ascending=False),
        "meta_returns":    meta_returns,
        "meta_sharpe":     erc_sharpe,
        "meta_ann_return": float(mu * 52),
        "meta_ann_vol":    float(sd * np.sqrt(52)),
        "meta_nw_t":       float(nw.get("t_stat", float("nan"))),
        "n_strategies":    K,
        "convergence":     bool(res.success),
    }


# ─────────────────────────────────────────────────────────────────────────────
# C.4 Beta-Neutralized Long-Short Construction
# ─────────────────────────────────────────────────────────────────────────────

def compute_strategy_beta_to_market(
    strategy_returns: pd.Series,
    market_returns:   pd.Series,
) -> dict:
    """
    Compute strategy β to market via OLS regression on weekly returns.
    """
    aligned = pd.concat({"r": strategy_returns, "m": market_returns}, axis=1).dropna()
    if len(aligned) < 12:
        return {"beta": float("nan"), "alpha": float("nan"), "r2": float("nan"), "n": len(aligned)}

    x = aligned["m"].values
    y = aligned["r"].values

    # OLS y = α + β x + ε
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    var_x = (x_centered ** 2).sum()
    if var_x < 1e-12:
        return {"beta": float("nan"), "alpha": float("nan"), "r2": 0.0, "n": len(aligned)}

    beta = float((x_centered * y_centered).sum() / var_x)
    alpha = float(y.mean() - beta * x.mean())
    y_pred = alpha + beta * x
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {"beta": beta, "alpha": alpha, "r2": r2, "n": len(aligned)}


def beta_neutralize_strategy(
    strategy_returns: pd.Series,
    market_returns:   pd.Series,
) -> dict:
    """
    Construct β-neutral version of strategy: subtract β × market.

    Returns dict with:
        original_returns, beta, alpha, r2, neutralized_returns,
        neutralized_sharpe, neutralized_ann_return, neutralized_ann_vol
    """
    from engine.backtest import newey_west_sharpe_se

    reg = compute_strategy_beta_to_market(strategy_returns, market_returns)
    if np.isnan(reg["beta"]):
        return {"error": "regression_failed", **reg}

    aligned = pd.concat({"r": strategy_returns, "m": market_returns}, axis=1).dropna()
    neutralized = aligned["r"] - reg["beta"] * aligned["m"]

    mu = float(neutralized.mean())
    sd = float(neutralized.std(ddof=1))
    sharpe = (mu / sd) * np.sqrt(52) if sd > 1e-12 else 0.0
    nw = newey_west_sharpe_se(neutralized.values, periods_per_year=52)

    return {
        "beta":                       reg["beta"],
        "alpha_per_period":            reg["alpha"],
        "alpha_annualized":            float(reg["alpha"] * 52),
        "r2_to_market":                reg["r2"],
        "n":                           reg["n"],
        "neutralized_returns":         neutralized,
        "neutralized_sharpe":          sharpe,
        "neutralized_ann_return":      float(mu * 52),
        "neutralized_ann_vol":         float(sd * np.sqrt(52)),
        "neutralized_nw_t":            float(nw.get("t_stat", float("nan"))),
    }


def beta_neutralize_all_strategies(
    returns_wide:    pd.DataFrame,
    market_returns:  pd.Series,
) -> pd.DataFrame:
    """
    Apply β-neutralization to every strategy.

    Returns DataFrame with columns:
      spec_label, beta, alpha_annualized, r2_to_market,
      neutralized_sharpe, neutralized_nw_t
    """
    rows = []
    for spec_label in returns_wide.columns:
        result = beta_neutralize_strategy(returns_wide[spec_label], market_returns)
        if "error" in result:
            rows.append({"spec_label": spec_label, "error": result["error"]})
            continue
        rows.append({
            "spec_label":             spec_label,
            "beta":                   result["beta"],
            "alpha_annualized":       result["alpha_annualized"],
            "r2_to_market":           result["r2_to_market"],
            "neutralized_sharpe":     result["neutralized_sharpe"],
            "neutralized_ann_return": result["neutralized_ann_return"],
            "neutralized_ann_vol":    result["neutralized_ann_vol"],
            "neutralized_nw_t":       result["neutralized_nw_t"],
            "n":                      result["n"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Phase C orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_c(
    results_dir:    str = DEFAULT_RESULTS_DIR,
    market_ticker:  str = "SPY",
) -> dict:
    """
    Full Phase C pipeline: correlation matrix + IC-weighted meta + ERC meta + β-neutralization.

    Persists results to:
      data/b_plus_results/phase_c_correlation.csv
      data/b_plus_results/phase_c_ic_meta.json
      data/b_plus_results/phase_c_erc_meta.json
      data/b_plus_results/phase_c_beta_neutral.csv
    """
    spec_df = load_per_spec(results_dir)
    returns_wide = load_all_oos_returns(results_dir)

    if returns_wide.empty:
        return {"error": "no_oos_returns_loaded"}

    out: dict = {}

    # C.2 Correlation
    corr = compute_strategy_correlation(returns_wide, method="pearson")
    corr.to_csv(os.path.join(results_dir, "phase_c_correlation.csv"))
    redundant = identify_redundant_strategies(corr, threshold=0.7)
    out["correlation_matrix_shape"] = corr.shape
    out["redundant_pairs"] = redundant[:10]   # top 10

    # C.3 IC-weighted meta
    ic_meta = compute_ic_weighted_meta(returns_wide, spec_df, use_ic_for_alpha=True)
    if "error" not in ic_meta:
        with open(os.path.join(results_dir, "phase_c_ic_meta.json"), "w") as f:
            json.dump({
                "weights":          ic_meta["weights"].to_dict(),
                "meta_sharpe":      ic_meta["meta_sharpe"],
                "meta_ann_return":  ic_meta["meta_ann_return"],
                "meta_ann_vol":     ic_meta["meta_ann_vol"],
                "meta_nw_t":        ic_meta["meta_nw_t"],
                "n_strategies":     ic_meta["n_strategies"],
            }, f, indent=2, default=str)
        ic_meta["meta_returns"].to_csv(
            os.path.join(results_dir, "phase_c_ic_meta_returns.csv"), header=["return"]
        )
    out["ic_meta_summary"] = ({
        "sharpe":  ic_meta.get("meta_sharpe"),
        "nw_t":    ic_meta.get("meta_nw_t"),
        "n":       ic_meta.get("n_strategies"),
    } if "error" not in ic_meta else {"error": ic_meta.get("error")})

    # C.3 ERC meta
    erc_meta = compute_erc_meta(returns_wide)
    if "error" not in erc_meta:
        with open(os.path.join(results_dir, "phase_c_erc_meta.json"), "w") as f:
            json.dump({
                "weights":          erc_meta["weights"].to_dict(),
                "meta_sharpe":      erc_meta["meta_sharpe"],
                "meta_ann_return":  erc_meta["meta_ann_return"],
                "meta_ann_vol":     erc_meta["meta_ann_vol"],
                "meta_nw_t":        erc_meta["meta_nw_t"],
                "n_strategies":     erc_meta["n_strategies"],
                "convergence":      erc_meta["convergence"],
            }, f, indent=2, default=str)
        erc_meta["meta_returns"].to_csv(
            os.path.join(results_dir, "phase_c_erc_meta_returns.csv"), header=["return"]
        )
    out["erc_meta_summary"] = ({
        "sharpe":  erc_meta.get("meta_sharpe"),
        "nw_t":    erc_meta.get("meta_nw_t"),
        "n":      erc_meta.get("n_strategies"),
    } if "error" not in erc_meta else {"error": erc_meta.get("error")})

    # C.4 β-neutralization (need market returns)
    try:
        from engine.signal import _fetch_closes
        start = returns_wide.index.min()
        end   = returns_wide.index.max()
        market_closes = _fetch_closes(
            [market_ticker], start.date(), end.date()
        )
        if not market_closes.empty:
            spy = market_closes[market_ticker].dropna()
            # Resample to weekly Friday close
            spy_weekly = spy.resample("W-FRI").last().pct_change(fill_method=None).dropna()
            spy_weekly = spy_weekly.reindex(returns_wide.index, method="ffill").dropna()
            beta_neutral_df = beta_neutralize_all_strategies(returns_wide, spy_weekly)
            beta_neutral_df.to_csv(
                os.path.join(results_dir, "phase_c_beta_neutral.csv"), index=False
            )
            out["beta_neutral_n_specs"] = len(beta_neutral_df)
        else:
            out["beta_neutral_error"] = "no_market_data"
    except Exception as exc:
        logger.warning("beta-neutralization failed: %s", exc)
        out["beta_neutral_error"] = str(exc)

    out["status"] = "ok"
    return out
