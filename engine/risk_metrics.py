"""
engine/risk_metrics.py — Portfolio-level risk computations for Position page.

Academic references:
  - VaR three methods (parametric / historical / Cornish-Fisher 1937)
  - ES / CVaR (Artzner et al. 1999) — coherent risk measure, mandatory companion to VaR
  - Component VaR (Garman 1996) — marginal contribution to portfolio VaR
  - Stress test: historical scenario replay (Lehman 2008-09, COVID 2020-02, Fed 2022-06)
  - Carhart 4-factor (Carhart 1997) — Mkt-RF + SMB + HML + MOM
  - Active exposure vs SPY sector benchmark — active management standard
  - HHI (Herfindahl-Hirschman) — concentration metric

All functions degrade gracefully on missing data:
  - Insufficient history (n<MIN_OBS) → return NaN + flag in `meta` dict
  - yfinance failure → return empty / zero-filled
  - Single-position portfolio → VaR uses asset return directly
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from engine._streamlit_shim import streamlit as st   # headless-safe; see shim docstring
import yfinance as yf

from engine.quant import QuantEngine

# ── Constants ────────────────────────────────────────────────────────────────
SPY_SECTOR_PROXIES = {
    # SPDR sector ETF basket; equal-weighted as zero-information SPY benchmark proxy
    "Technology": "XLK", "Financials": "XLF", "Energy": "XLE",
    "Health Care": "XLV", "Industrials": "XLI", "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP", "Utilities": "XLU", "Materials": "XLB",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
# Approximate SPY sector weights (S&P 500 official, periodically re-checked vs S&P)
SPY_SECTOR_WEIGHTS = {
    "Technology": 0.295, "Financials": 0.135, "Health Care": 0.122,
    "Consumer Discretionary": 0.105, "Communication Services": 0.090,
    "Industrials": 0.082, "Consumer Staples": 0.058, "Energy": 0.038,
    "Utilities": 0.025, "Real Estate": 0.022, "Materials": 0.022,
}
CARHART_PROXIES = {
    "Mkt": "SPY",   # market
    "SMB": "IWM",   # small-cap
    "Big": "IWB",   # large-cap (used for SMB = IWM - IWB)
    "HML": "IWD",   # value
    "Growth": "IWF",  # growth (used for HML = IWD - IWF)
    "MOM": "MTUM",  # momentum
}
STRESS_SCENARIOS = {
    # (label, start, end) — historically realized shocks; not predictive
    "Lehman 2008": ("2008-09-15", "2008-10-15"),
    "COVID 2020": ("2020-02-19", "2020-03-23"),
    "Fed Shock 2022": ("2022-06-10", "2022-06-17"),
}
MIN_HIST_OBS = 252           # minimum daily obs for parametric stats
MIN_HIST_VAR_OBS = 1000       # historical VaR — avoid quantile noise
LOOKBACK_DAYS = "2y"         # default yfinance history fetch
RISK_FREE_RATE = 0.04        # annualized; for Sharpe / IR


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def fetch_returns(tickers: tuple[str, ...], period: str = LOOKBACK_DAYS) -> pd.DataFrame:
    """Daily simple returns for `tickers`. Returns empty DataFrame on failure."""
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            list(tickers), period=period, auto_adjust=True,
            progress=False, multi_level_index=False,
        )
        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            close = data["Close"]
        elif "Close" in data:
            close = data["Close"]
        else:
            close = data
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        rets = close.pct_change().dropna(how="all")
        return rets
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_window_close(ticker: str, start: str, end: str) -> pd.Series:
    """Single-ticker close prices for a date window — for stress test replay."""
    try:
        data = yf.download(
            ticker, start=start, end=end, auto_adjust=True,
            progress=False, multi_level_index=False,
        )
        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            close = data["Close"]
            # yfinance edge case: even with multi_level_index=False, certain
            # tickers/versions return a DataFrame for data["Close"]. Squeeze
            # to first column so downstream float(close.iloc[-1]/...) holds.
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.dropna()
        if "Close" in data:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.dropna()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


# ── Portfolio return synthesis ───────────────────────────────────────────────
def synthesize_portfolio_returns(
    positions: pd.DataFrame,
    period: str = LOOKBACK_DAYS,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """
    Build portfolio daily returns from position weights × ETF returns.

    Returns:
        port_ret: pd.Series  — daily portfolio returns (live weight * asset return)
        rets:     pd.DataFrame — daily returns matrix (assets in cols)
        meta:     dict        — {n_obs, n_assets, missing_tickers}
    """
    meta = {"n_obs": 0, "n_assets": 0, "missing_tickers": []}
    if positions is None or positions.empty:
        return pd.Series(dtype=float), pd.DataFrame(), meta

    pos = positions[positions["actual_weight"].abs() > 1e-6].copy()
    if pos.empty:
        return pd.Series(dtype=float), pd.DataFrame(), meta

    tickers = tuple(pos["ticker"].dropna().unique())
    rets = fetch_returns(tickers, period)
    if rets.empty:
        meta["missing_tickers"] = list(tickers)
        return pd.Series(dtype=float), pd.DataFrame(), meta

    # Reindex weights to rets columns
    weight_map = pos.set_index("ticker")["actual_weight"].to_dict()
    cols = [c for c in rets.columns if c in weight_map]
    rets = rets[cols].dropna(how="all")
    weights = np.array([weight_map[c] for c in cols])
    if rets.empty or len(cols) == 0:
        return pd.Series(dtype=float), rets, meta

    port_ret = rets.fillna(0).dot(weights)
    meta.update({
        "n_obs": int(len(port_ret)),
        "n_assets": int(len(cols)),
        "missing_tickers": [t for t in tickers if t not in cols],
    })
    return port_ret, rets, meta


# ── VaR / ES three methods ───────────────────────────────────────────────────
@dataclass
class VarBlock:
    """Container for VaR / ES output across three estimation methods."""
    parametric: float = float("nan")
    historical: float = float("nan")
    cornish_fisher: float = float("nan")
    es_parametric: float = float("nan")
    es_historical: float = float("nan")
    n_obs: int = 0
    insufficient: bool = True
    bootstrap_ci: tuple[float, float] = (float("nan"), float("nan"))
    divergence_warning: bool = False  # True if 3 methods disagree by > 30%


def compute_var_block(port_ret: pd.Series, alpha: float = 0.05) -> VarBlock:
    """
    Compute VaR three ways + ES; flag if methods diverge (suggests fat-tailed).

    Convention: returns are in *daily decimal form*; VaR/ES negative = loss.
    """
    n = len(port_ret)
    block = VarBlock(n_obs=n, insufficient=(n < MIN_HIST_OBS))
    if n < 30:
        return block

    # Parametric (Normal)
    mu = float(port_ret.mean())
    sigma = float(port_ret.std())
    z = -1.6449 if alpha == 0.05 else -2.3263
    block.parametric = mu + z * sigma

    # Historical (sample quantile)
    block.historical = float(np.quantile(port_ret, alpha))

    # Cornish-Fisher (skew/kurt adjusted)
    try:
        block.cornish_fisher = QuantEngine.cornish_fisher_var(port_ret, alpha=alpha)
    except Exception:
        block.cornish_fisher = block.parametric

    # ES (CVaR)
    block.es_historical = QuantEngine.expected_shortfall(port_ret, alpha=alpha)
    # Parametric ES under normal: ES = μ - σ·φ(z)/α
    phi_z = float(np.exp(-z * z / 2) / np.sqrt(2 * np.pi))
    block.es_parametric = mu - sigma * phi_z / alpha

    # Bootstrap CI for historical VaR (only if n sufficient)
    if n >= 100:
        rng = np.random.default_rng(42)
        boots = []
        arr = port_ret.values
        for _ in range(500):
            sample = rng.choice(arr, size=n, replace=True)
            boots.append(np.quantile(sample, alpha))
        block.bootstrap_ci = (float(np.quantile(boots, 0.025)),
                              float(np.quantile(boots, 0.975)))

    # Divergence flag — fat-tail signal when methods diverge significantly
    vals = [v for v in (block.parametric, block.historical, block.cornish_fisher)
            if not np.isnan(v) and v < 0]
    if len(vals) >= 2:
        spread = (max(vals) - min(vals)) / abs(np.mean(vals))
        block.divergence_warning = spread > 0.30
    return block


# ── Component VaR (Garman 1996) ──────────────────────────────────────────────
def compute_component_var(
    rets: pd.DataFrame,
    weights: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Component VaR for each asset = wᵢ · ∂VaR/∂wᵢ
    Σ Component VaR = Portfolio VaR (by Euler theorem on homogeneous fns).

    Linear (Gaussian) approximation:
      Component VaR_i = wᵢ · (Σw)ᵢ / σ_p · z_α
    where Σ is the asset covariance matrix.

    Returns: {ticker: component_var_decimal}
    """
    if rets.empty or not weights:
        return {}
    cols = [c for c in rets.columns if c in weights]
    if not cols:
        return {}
    w = np.array([weights[c] for c in cols])
    cov = rets[cols].cov().values
    cov_w = cov @ w
    sigma_p = float(np.sqrt(w @ cov_w))
    if sigma_p < 1e-10:
        return {c: 0.0 for c in cols}
    z = -1.6449 if alpha == 0.05 else -2.3263
    comp = w * cov_w / sigma_p * z   # negative = loss contribution
    return {c: float(comp[i]) for i, c in enumerate(cols)}


# ── Stress test ──────────────────────────────────────────────────────────────
@dataclass
class StressResult:
    label: str
    pnl_pct: float
    contributions: dict[str, float] = field(default_factory=dict)
    missing_assets: list[str] = field(default_factory=list)


def run_stress_scenarios(
    positions: pd.DataFrame,
    scenarios: dict | None = None,
) -> list[StressResult]:
    """
    Replay historical shocks. Each scenario = window-period total return per asset,
    weighted by current portfolio weights.
    """
    scenarios = scenarios or STRESS_SCENARIOS
    if positions is None or positions.empty:
        return []
    pos = positions[positions["actual_weight"].abs() > 1e-6].copy()
    if pos.empty:
        return []

    out: list[StressResult] = []
    weight_map = pos.set_index("ticker")["actual_weight"].to_dict()

    for label, (start, end) in scenarios.items():
        contribs: dict[str, float] = {}
        missing: list[str] = []
        for tk, w in weight_map.items():
            close = fetch_window_close(tk, start, end)
            if len(close) < 2:
                missing.append(tk)
                continue
            window_ret = float(close.iloc[-1] / close.iloc[0] - 1)
            contribs[tk] = w * window_ret
        out.append(StressResult(
            label=label,
            pnl_pct=sum(contribs.values()),
            contributions=contribs,
            missing_assets=missing,
        ))
    return out


# ── Active exposure vs SPY ───────────────────────────────────────────────────
def compute_active_exposure(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Active over/underweight vs SPY sector benchmark.

    Returns DataFrame with columns: sector | port_wgt | spy_wgt | active_wgt
    Sorted by |active_wgt| descending.
    """
    if positions is None or positions.empty:
        return pd.DataFrame(columns=["sector", "port_wgt", "spy_wgt", "active_wgt"])

    pos = positions[positions["actual_weight"].abs() > 1e-6].copy()
    if pos.empty:
        return pd.DataFrame(columns=["sector", "port_wgt", "spy_wgt", "active_wgt"])

    # Aggregate portfolio weights by sector
    if "sector" not in pos.columns:
        pos = pos.reset_index().rename(columns={"index": "sector"})
    port_by_sec = pos.groupby("sector")["actual_weight"].sum().to_dict()

    rows = []
    all_sectors = set(port_by_sec.keys()) | set(SPY_SECTOR_WEIGHTS.keys())
    for sec in all_sectors:
        port_w = float(port_by_sec.get(sec, 0.0))
        spy_w = float(SPY_SECTOR_WEIGHTS.get(sec, 0.0))
        rows.append({
            "sector": sec,
            "port_wgt": port_w,
            "spy_wgt": spy_w,
            "active_wgt": port_w - spy_w,
        })
    df = pd.DataFrame(rows).sort_values(
        "active_wgt", key=lambda s: s.abs(), ascending=False,
    )
    return df


def compute_active_share(positions: pd.DataFrame) -> float:
    """
    Active Share (Cremers & Petajisto 2009): half of |w_p - w_b| sum.
    Range [0,1]; 1 = fully active, 0 = pure index. < 0.6 = closet indexer.
    """
    df = compute_active_exposure(positions)
    if df.empty:
        return 0.0
    return float(df["active_wgt"].abs().sum() / 2.0)


# ── Carhart 4-factor exposure ────────────────────────────────────────────────
def compute_factor_tilt(
    positions: pd.DataFrame,
    period: str = LOOKBACK_DAYS,
) -> dict[str, float]:
    """
    Portfolio Carhart 4-factor exposure via OLS regression of asset returns on
    factor proxies, then weighting by position.

    Factors:
      Mkt-Rf  ≈ SPY
      SMB     ≈ IWM - IWB
      HML     ≈ IWD - IWF
      MOM     ≈ MTUM (excess over SPY)

    Returns: {"Mkt": β_mkt, "SMB": β_smb, "HML": β_hml, "MOM": β_mom, "n_obs": int}
    """
    out = {"Mkt": 0.0, "SMB": 0.0, "HML": 0.0, "MOM": 0.0, "n_obs": 0}
    if positions is None or positions.empty:
        return out
    pos = positions[positions["actual_weight"].abs() > 1e-6].copy()
    if pos.empty:
        return out

    # Fetch factor proxies + asset returns in one batch
    asset_tks = tuple(pos["ticker"].dropna().unique())
    factor_tks = tuple(set(CARHART_PROXIES.values()))
    all_tks = tuple(sorted(set(asset_tks) | set(factor_tks)))
    rets = fetch_returns(all_tks, period)
    if rets.empty:
        return out

    # Build factor returns
    try:
        f_mkt = rets.get("SPY")
        if f_mkt is None:
            return out
        smb = rets.get("IWM", pd.Series()) - rets.get("IWB", pd.Series())
        hml = rets.get("IWD", pd.Series()) - rets.get("IWF", pd.Series())
        mom = rets.get("MTUM", pd.Series()) - f_mkt
        factors = pd.DataFrame({
            "Mkt": f_mkt, "SMB": smb, "HML": hml, "MOM": mom,
        }).dropna()
    except Exception:
        return out

    if len(factors) < 60:
        out["n_obs"] = len(factors)
        return out

    weight_map = pos.set_index("ticker")["actual_weight"].to_dict()
    port_betas = {"Mkt": 0.0, "SMB": 0.0, "HML": 0.0, "MOM": 0.0}
    F = factors.values
    F_aug = np.column_stack([np.ones(len(F)), F])
    try:
        # Pseudo-inverse for stability
        XtX_inv = np.linalg.pinv(F_aug.T @ F_aug)
    except Exception:
        return out

    for tk, w in weight_map.items():
        if tk not in rets.columns:
            continue
        y = rets[tk].reindex(factors.index).fillna(0).values
        if len(y) != len(F_aug):
            continue
        try:
            beta = XtX_inv @ F_aug.T @ y   # [α, β_mkt, β_smb, β_hml, β_mom]
        except Exception:
            continue
        port_betas["Mkt"] += w * float(beta[1])
        port_betas["SMB"] += w * float(beta[2])
        port_betas["HML"] += w * float(beta[3])
        port_betas["MOM"] += w * float(beta[4])

    port_betas["n_obs"] = len(factors)
    return port_betas


# ── Fama-French 5-factor exposure (Fama-French 2015) ─────────────────────────
# Proxy choices for RMW (profitability) and CMA (investment) are approximate
# because true FF5 uses portfolio sorts on accounting data not derivable from
# ETF prices. Standard practice when only price data is available:
#   - RMW ≈ QUAL (iShares MSCI USA Quality Factor ETF) - SPY
#   - CMA ≈ USMV (iShares MSCI USA Min Vol Factor ETF) - SPY
# (Low-investment firms are empirically lower-vol.) Tracking error vs academic
# FF5 ≈ 30-40% on betas; useful for direction & magnitude, not for paper-
# publication precision. UI must disclose proxy nature.
FF5_PROXIES = {
    "Mkt": "SPY",
    "SMB_small":  "IWM",
    "SMB_big":    "IWB",
    "HML_value":  "IWD",
    "HML_growth": "IWF",
    "RMW_quality": "QUAL",  # proxy
    "CMA_minvol":  "USMV",  # proxy
}


def _ff5_factor_returns(period: str = LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """Build FF5 daily factor return DataFrame from proxy ETFs. Returns None on failure."""
    tks = tuple(set(FF5_PROXIES.values()))
    rets = fetch_returns(tks, period)
    if rets.empty:
        return None
    spy = rets.get("SPY")
    if spy is None:
        return None
    try:
        factors = pd.DataFrame({
            "Mkt": spy,
            "SMB": rets.get("IWM", pd.Series()) - rets.get("IWB", pd.Series()),
            "HML": rets.get("IWD", pd.Series()) - rets.get("IWF", pd.Series()),
            "RMW": rets.get("QUAL", pd.Series()) - spy,
            "CMA": rets.get("USMV", pd.Series()) - spy,
        }).dropna()
    except Exception:
        return None
    return factors if len(factors) >= 60 else None


def compute_ff5_factor_tilt(
    positions: pd.DataFrame,
    period: str = LOOKBACK_DAYS,
) -> dict:
    """Fama-French 5-factor portfolio exposure via OLS asset-level regression.

    Returns: dict with keys
      Mkt, SMB, HML, RMW, CMA   — portfolio-level betas (float)
      alpha_daily               — portfolio daily alpha intercept
      r_squared                 — book-aggregate weighted-mean asset R²
      n_obs                     — days of overlapping factor history
      n_assets                  — non-zero positions used
      proxy_disclosure          — short note for UI disclosure
    Returns dict with NaN betas + n_obs=0 if insufficient data.
    """
    out = {
        "Mkt": float("nan"), "SMB": float("nan"), "HML": float("nan"),
        "RMW": float("nan"), "CMA": float("nan"),
        "alpha_daily": float("nan"), "r_squared": float("nan"),
        "n_obs": 0, "n_assets": 0,
        "proxy_disclosure": "RMW≈QUAL-SPY · CMA≈USMV-SPY (ETF proxies; exact FF5 needs accounting sorts)",
    }
    if positions is None or positions.empty:
        return out
    pos = positions[positions["actual_weight"].abs() > 1e-6].copy()
    if pos.empty:
        return out

    factors = _ff5_factor_returns(period)
    if factors is None:
        return out

    asset_tks = tuple(pos["ticker"].dropna().unique())
    rets = fetch_returns(asset_tks, period)
    if rets.empty:
        return out

    weight_map = pos.set_index("ticker")["actual_weight"].to_dict()
    F = factors.values
    F_aug = np.column_stack([np.ones(len(F)), F])
    try:
        XtX_inv = np.linalg.pinv(F_aug.T @ F_aug)
    except Exception:
        return out

    port_betas = {"Mkt": 0.0, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0}
    port_alpha = 0.0
    weighted_r2_num = 0.0
    weighted_w = 0.0
    n_assets = 0
    for tk, w in weight_map.items():
        if tk not in rets.columns:
            continue
        y = rets[tk].reindex(factors.index).fillna(0).values
        if len(y) != len(F_aug):
            continue
        try:
            beta = XtX_inv @ F_aug.T @ y   # [α, β_mkt, β_smb, β_hml, β_rmw, β_cma]
        except Exception:
            continue
        port_alpha       += w * float(beta[0])
        port_betas["Mkt"] += w * float(beta[1])
        port_betas["SMB"] += w * float(beta[2])
        port_betas["HML"] += w * float(beta[3])
        port_betas["RMW"] += w * float(beta[4])
        port_betas["CMA"] += w * float(beta[5])
        # asset-level R²
        yhat = F_aug @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot > 0:
            weighted_r2_num += abs(w) * (1.0 - ss_res / ss_tot)
            weighted_w      += abs(w)
        n_assets += 1

    out.update(port_betas)
    out["alpha_daily"] = float(port_alpha)
    out["r_squared"]   = float(weighted_r2_num / weighted_w) if weighted_w > 0 else float("nan")
    out["n_obs"]       = int(len(factors))
    out["n_assets"]    = int(n_assets)
    return out


def compute_ff5_rolling_stability(
    positions: pd.DataFrame,
    period: str = LOOKBACK_DAYS,
    window_days: int = 60,
    step_days: int = 10,
) -> dict:
    """Compute std of portfolio FF5 betas across rolling sub-windows.

    Lower std = more stable factor exposure (the desired property).
    Higher std on a factor = book has been rotating into / out of that tilt.

    Returns: dict {factor: std_of_beta_across_windows} + n_windows. NaN if
    insufficient data (< 2 windows fit in `period`).
    """
    out = {
        "Mkt_std": float("nan"), "SMB_std": float("nan"), "HML_std": float("nan"),
        "RMW_std": float("nan"), "CMA_std": float("nan"),
        "n_windows": 0,
    }
    if positions is None or positions.empty:
        return out

    factors = _ff5_factor_returns(period)
    if factors is None or len(factors) < window_days * 2:
        return out

    pos = positions[positions["actual_weight"].abs() > 1e-6].copy()
    if pos.empty:
        return out
    asset_tks = tuple(pos["ticker"].dropna().unique())
    rets = fetch_returns(asset_tks, period)
    if rets.empty:
        return out
    rets = rets.reindex(factors.index).fillna(0)

    weight_map = pos.set_index("ticker")["actual_weight"].to_dict()
    starts = list(range(0, len(factors) - window_days + 1, step_days))
    betas_series = {"Mkt": [], "SMB": [], "HML": [], "RMW": [], "CMA": []}
    for s in starts:
        e = s + window_days
        F_w = factors.iloc[s:e].values
        F_aug = np.column_stack([np.ones(len(F_w)), F_w])
        try:
            XtX_inv = np.linalg.pinv(F_aug.T @ F_aug)
        except Exception:
            continue
        port_betas = {"Mkt": 0.0, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0}
        for tk, w in weight_map.items():
            if tk not in rets.columns:
                continue
            y = rets[tk].iloc[s:e].values
            if len(y) != len(F_aug):
                continue
            try:
                beta = XtX_inv @ F_aug.T @ y
            except Exception:
                continue
            port_betas["Mkt"] += w * float(beta[1])
            port_betas["SMB"] += w * float(beta[2])
            port_betas["HML"] += w * float(beta[3])
            port_betas["RMW"] += w * float(beta[4])
            port_betas["CMA"] += w * float(beta[5])
        for k in betas_series:
            betas_series[k].append(port_betas[k])

    n_w = len(betas_series["Mkt"])
    if n_w < 2:
        return out
    out["n_windows"] = n_w
    for k in ("Mkt", "SMB", "HML", "RMW", "CMA"):
        arr = np.asarray(betas_series[k], dtype=float)
        out[f"{k}_std"] = float(np.std(arr, ddof=1))
    return out


# ── Beta vs SPY (with Newey-West CI) ─────────────────────────────────────────
def compute_beta_vs_spy(
    asset_returns: pd.Series,
    spy_returns: pd.Series,
    nw_lag: int = 5,
) -> dict[str, float]:
    """
    Single-asset beta vs SPY with Newey-West HAC standard error.
    Returns {beta, se_nw, ci_low, ci_high, n_obs}
    """
    s = pd.concat([asset_returns, spy_returns], axis=1, join="inner").dropna()
    if len(s) < 60:
        return {"beta": float("nan"), "se_nw": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n_obs": len(s)}
    s.columns = ["y", "x"]
    x = s["x"].values
    y = s["y"].values
    x_demeaned = x - x.mean()
    y_demeaned = y - y.mean()
    var_x = float((x_demeaned ** 2).mean())
    if var_x < 1e-12:
        return {"beta": float("nan"), "se_nw": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n_obs": len(s)}
    beta = float((x_demeaned * y_demeaned).mean() / var_x)
    resid = y_demeaned - beta * x_demeaned
    n = len(s)
    # Newey-West variance with Bartlett kernel
    omega = float((resid ** 2).mean())
    for lag in range(1, nw_lag + 1):
        w_lag = 1.0 - lag / (nw_lag + 1)
        cov_lag = float((resid[lag:] * resid[:-lag] * x_demeaned[lag:]
                          * x_demeaned[:-lag]).sum()) / n
        omega += 2.0 * w_lag * cov_lag
    se_nw = float(np.sqrt(omega / (n * var_x ** 2))) if var_x > 0 else float("nan")
    return {
        "beta": beta,
        "se_nw": se_nw,
        "ci_low": beta - 1.96 * se_nw,
        "ci_high": beta + 1.96 * se_nw,
        "n_obs": n,
    }


# ── Concentration metrics ────────────────────────────────────────────────────
def compute_concentration(positions: pd.DataFrame) -> dict[str, float]:
    """
    Portfolio concentration:
      hhi      — Herfindahl-Hirschman = Σ w² (range [1/N, 1])
      top1_pct — largest single-position weight
      top5_pct — top 5 positions sum
      n_pos    — number of non-zero positions
    """
    out = {"hhi": 0.0, "top1_pct": 0.0, "top5_pct": 0.0, "n_pos": 0,
           "long_n": 0, "short_n": 0}
    if positions is None or positions.empty:
        return out
    pos = positions[positions["actual_weight"].abs() > 1e-6].copy()
    if pos.empty:
        return out
    w_abs = pos["actual_weight"].abs().sort_values(ascending=False)
    out["hhi"] = float((w_abs ** 2).sum())
    out["top1_pct"] = float(w_abs.iloc[0]) if len(w_abs) else 0.0
    out["top5_pct"] = float(w_abs.head(5).sum())
    out["n_pos"] = int(len(w_abs))
    out["long_n"] = int((pos["actual_weight"] > 1e-6).sum())
    out["short_n"] = int((pos["actual_weight"] < -1e-6).sum())
    return out


# ── Liquidity profile ────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def fetch_adv(tickers: tuple[str, ...]) -> dict[str, float]:
    """Average daily volume × price (60d) — proxy for ADV in $."""
    out = {}
    if not tickers:
        return out
    for tk in tickers:
        try:
            d = yf.download(tk, period="3mo", auto_adjust=True,
                            progress=False, multi_level_index=False)
            if d.empty:
                out[tk] = 0.0
                continue
            v = d["Volume"].tail(60).mean() if "Volume" in d.columns else 0.0
            p = d["Close"].tail(60).mean() if "Close" in d.columns else 0.0
            out[tk] = float(v * p)
        except Exception:
            out[tk] = 0.0
    return out


def compute_liquidity_profile(
    positions: pd.DataFrame, nav: float,
) -> pd.DataFrame:
    """
    Days-to-exit = position MV / (ADV × max participation rate).
    Conservative: 10% participation rate.
    """
    if positions is None or positions.empty:
        return pd.DataFrame(columns=["ticker", "mv", "adv", "days_to_exit"])
    pos = positions[positions["actual_weight"].abs() > 1e-6].copy()
    if pos.empty:
        return pd.DataFrame(columns=["ticker", "mv", "adv", "days_to_exit"])
    tks = tuple(pos["ticker"].dropna().unique())
    adv = fetch_adv(tks)
    rows = []
    for _, r in pos.iterrows():
        tk = r["ticker"]
        mv = abs(float(r.get("actual_weight") or 0)) * nav
        a = adv.get(tk, 0.0)
        dte = mv / (a * 0.10) if a > 0 else float("inf")
        rows.append({"ticker": tk, "mv": mv, "adv": a, "days_to_exit": dte})
    return pd.DataFrame(rows).sort_values("days_to_exit", ascending=False)


# ── Portfolio header strip metrics ───────────────────────────────────────────
@dataclass
class PortfolioHeaderMetrics:
    nav: float = 0.0
    dtd_pnl_pct: float = 0.0
    mtd_pnl_pct: float = float("nan")
    ytd_pnl_pct: float = float("nan")
    sharpe: float = float("nan")
    vol_ann: float = float("nan")
    beta_spy: float = float("nan")
    tracking_error: float = float("nan")
    info_ratio: float = float("nan")
    var_95: float = float("nan")
    es_95: float = float("nan")
    dd_curr: float = float("nan")
    dd_max: float = float("nan")
    hhi: float = 0.0
    active_share: float = 0.0
    n_obs: int = 0


def compute_header_metrics(
    positions: pd.DataFrame,
    nav: float,
    dtd_pnl_pct: float,
    monthly_returns: pd.DataFrame | None = None,
) -> PortfolioHeaderMetrics:
    """
    Single canonical computation for the 14-metric Portfolio Header Strip.
    Falls back to NaN where data insufficient (caller must render '—').
    """
    m = PortfolioHeaderMetrics(nav=nav, dtd_pnl_pct=dtd_pnl_pct)
    if positions is None or positions.empty:
        return m

    # Concentration / active share
    conc = compute_concentration(positions)
    m.hhi = conc["hhi"]
    m.active_share = compute_active_share(positions)

    # Synthesize portfolio returns for risk metrics
    port_ret, rets, meta = synthesize_portfolio_returns(positions)
    m.n_obs = meta.get("n_obs", 0)

    if m.n_obs >= 60:
        # Sharpe / Vol
        mu_d = float(port_ret.mean())
        sigma_d = float(port_ret.std())
        if sigma_d > 1e-9:
            m.vol_ann = sigma_d * np.sqrt(252)
            m.sharpe = (mu_d * 252 - RISK_FREE_RATE) / m.vol_ann

        # VaR / ES
        var_block = compute_var_block(port_ret, alpha=0.05)
        m.var_95 = var_block.historical if not np.isnan(var_block.historical) else var_block.parametric
        m.es_95 = var_block.es_historical

        # Drawdown
        cum = (1 + port_ret).cumprod()
        dd_series = (cum / cum.cummax()) - 1
        m.dd_curr = float(dd_series.iloc[-1])
        m.dd_max = float(dd_series.min())

        # Beta vs SPY + tracking error + IR
        spy_rets = fetch_returns(("SPY",))
        if not spy_rets.empty and "SPY" in spy_rets.columns:
            beta_info = compute_beta_vs_spy(port_ret, spy_rets["SPY"])
            m.beta_spy = beta_info["beta"]
            joined = pd.concat([port_ret, spy_rets["SPY"]], axis=1, join="inner").dropna()
            if len(joined) > 60:
                joined.columns = ["p", "b"]
                active = joined["p"] - joined["b"]
                te_d = float(active.std())
                if te_d > 1e-9:
                    m.tracking_error = te_d * np.sqrt(252)
                    m.info_ratio = (float(active.mean()) * 252) / m.tracking_error

    # MTD / YTD from monthly returns
    if monthly_returns is not None and not monthly_returns.empty:
        try:
            md = monthly_returns.copy()
            if "return_month" in md.columns:
                md["return_month"] = pd.to_datetime(md["return_month"])
                today = datetime.date.today()
                this_month = pd.Timestamp(today.replace(day=1))
                this_year = pd.Timestamp(today.replace(month=1, day=1))
                # MTD requires intra-month return (we don't have daily), approximate as 0 or use dtd
                # YTD = sum of monthly contributions for current year
                ytd = md[md["return_month"] >= this_year]["contribution"].sum()
                m.ytd_pnl_pct = float(ytd) if pd.notna(ytd) else float("nan")
        except Exception:
            pass

    return m
