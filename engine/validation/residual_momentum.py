"""engine/validation/residual_momentum.py — Residual Momentum POC (Blitz-Huij-Martens 2011).

Phase 2 §I.A.2 of research_agenda_2026-05-29. After Quality (I.A.1) RED today
(commit f9e5824, junk premium era), this tests another well-documented equity
factor that is structurally orthogonal to both:
  - D-PEAD / analyst revision (earnings-information family, our existing equity sleeve)
  - Novy-Marx Quality (RED — junk-wins 2013-2024)

Academic anchor
---------------
Blitz, Huij, Martens 2011 "Residual Momentum" (J. of Banking & Finance):
- Raw momentum (12-1) earns ~0.8% / month in 1929-2009 US but has high crash risk
- Their finding: AFTER stripping factor exposure (Mkt+SMB+HML), the residual
  momentum signal still works AND has dramatically lower drawdown
- The intuition: traditional momentum often loads on whatever factor was
  trending; residual momentum captures the idiosyncratic part

Construction (pre-committed BHM 2011 standard)
----------------------------------------------
1. For each (stock, month t), compute rolling factor betas on past 36 months:
       r_i,s = α_i + β1·Mkt-RF_s + β2·SMB_s + β3·HML_s + ε_i,s
   using shrunk OLS (shrinkage λ=0.5 toward 0 to handle short panels).
2. Compute residual return:  e_i,s = r_i,s - (β1·Mkt-RF_s + β2·SMB_s + β3·HML_s)
3. Standardize: z_i,s = e_i,s / σ_i,resid (rolling 36m residual vol)
4. Signal at t = sum(z_i,s) for s in [t-12, t-2]  (1-month skip per momentum convention)
5. Cross-sectional ranking: long top 20% / short bottom 20%, equal-weighted
6. Monthly rebalance, 30bp/side cost
7. Output: monthly L/S return series

Strict-gate framing
-------------------
- Pre-committed parameters above; NO grid search
- Sample 2013-2024 (limited by Compustat funda + 6-month lag warmup +
  36-month factor regression warmup)
- The same hostile-equity-factor sample where Quality RED'd. Probably RED.
- BHM 2011 reports Sharpe ~0.8 on 1929-2009 sample; their 1990s subsample is
  weakest. Our 2013-2024 may be even weaker than their weakest subsample.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Pre-committed parameters ─────────────────────────────────────────────────
BETA_WINDOW = 36           # months for rolling factor regression
SHRINK_LAMBDA = 0.5        # ridge-like shrinkage toward 0 betas
FORMATION = 12             # months in signal window
SKIP = 1                   # skip latest month (12-1 standard)
TOP_Q = 0.20               # top 20% long (quintile)
BOT_Q = 0.20               # bottom 20% short
RT_EQ_BPS = 30.0           # single-side execution cost
MIN_NAMES_PER_LEG = 10
MIN_OBS_FOR_REGRESSION = 24    # minimum months to estimate betas
RESIDUAL_VOL_FLOOR = 0.01      # avoid div-by-near-zero standardization


@lru_cache(maxsize=1)
def _load_panels():
    """Load CRSP daily returns + Compustat funda universe + FF5 factors."""
    funda = pd.read_parquet("data/cache/_compustat_funda.parquet")
    rets  = pd.read_parquet("data/cache/crsp_hist_daily_ret.parquet")
    mp    = pd.read_parquet("data/cache/_cik_gvkey_permno_map.parquet")
    ff    = pd.read_parquet("data/cache/ff_factors_weekly.parquet")
    return funda, rets, mp, ff


def _build_monthly_returns_panel() -> pd.DataFrame:
    """Compound CRSP daily → monthly returns per permno. Wide format
    (rows=month, cols=permno) for vectorized residual computation."""
    _, rets, _, _ = _load_panels()
    rets = rets.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    rets["month"] = rets["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = rets.groupby(["permno", "month"])["ret"].apply(
        lambda x: float((1.0 + x).prod() - 1.0)).reset_index()
    wide = monthly.pivot(index="month", columns="permno", values="ret").sort_index()
    return wide


def _build_ff_monthly() -> pd.DataFrame:
    """Aggregate FF5 weekly factors to monthly. We only need Mkt-RF, SMB, HML
    for BHM 2011 (they use FF3, not FF5)."""
    _, _, _, ff = _load_panels()
    ff = ff.copy()
    ff.index = pd.to_datetime(ff.index)
    needed = ["Mkt-RF", "SMB", "HML"]
    monthly = (1.0 + ff[needed]).resample("ME").prod() - 1.0
    return monthly


def _shrunk_betas(y: np.ndarray, X: np.ndarray, lam: float = SHRINK_LAMBDA) -> np.ndarray:
    """Ridge-style shrinkage toward 0 betas (no intercept shrink). Stable when
    panel is short. Returns (k,) array of betas where X is (n, k)."""
    XtX = X.T @ X
    XtY = X.T @ y
    # Ridge: (X'X + lam I) ^-1 X'y; identity rows for slope-only shrinkage
    p = XtX.shape[0]
    reg = np.eye(p) * lam
    reg[0, 0] = 0.0   # don't shrink the intercept (column 0)
    try:
        beta = np.linalg.solve(XtX + reg, XtY)
    except np.linalg.LinAlgError:
        beta = np.zeros(p)
    return beta


def _build_residual_panel() -> pd.DataFrame:
    """For each (permno, month), compute the standardized residual return
    using shrunk rolling 36-month FF3 betas. Returns wide DataFrame."""
    returns_wide = _build_monthly_returns_panel()
    ff = _build_ff_monthly()

    # Align month indexes
    common = returns_wide.index.intersection(ff.index)
    R = returns_wide.loc[common]
    F = ff.loc[common]

    # Restrict to stocks with at least MIN_OBS_FOR_REGRESSION non-NaN months
    enough = R.notna().sum(axis=0) >= MIN_OBS_FOR_REGRESSION
    R = R.loc[:, enough]

    # F augmented with intercept column
    F_aug = pd.concat([pd.Series(1.0, index=F.index, name="intercept"), F], axis=1)
    F_aug_np = F_aug.values   # (T, 4)

    months = R.index.tolist()
    permnos = R.columns.tolist()
    residual_z = pd.DataFrame(index=months, columns=permnos, dtype=float)

    # For each month t, take past BETA_WINDOW months to estimate betas, then
    # compute residual at t and standardize by rolling residual vol
    for col in permnos:
        y_all = R[col].values
        # Compute rolling beta + residual vectorized across time
        for t in range(BETA_WINDOW, len(months)):
            window = slice(t - BETA_WINDOW, t)
            y_w = y_all[window]
            X_w = F_aug_np[window]
            mask = ~np.isnan(y_w)
            if mask.sum() < MIN_OBS_FOR_REGRESSION:
                continue
            beta = _shrunk_betas(y_w[mask], X_w[mask])
            # Residual at month t (uses beta estimated through t-1)
            y_t = y_all[t]
            if np.isnan(y_t):
                continue
            x_t = F_aug_np[t]
            r_t = float(y_t - x_t @ beta)
            # Standardize by residual vol in the window
            resid_window = y_w[mask] - X_w[mask] @ beta
            sigma = float(resid_window.std(ddof=1)) if len(resid_window) > 1 else np.nan
            if np.isnan(sigma) or sigma < RESIDUAL_VOL_FLOOR:
                continue
            residual_z.iloc[t, residual_z.columns.get_loc(col)] = r_t / sigma

    return residual_z


def _build_signal_panel() -> pd.DataFrame:
    """12-1 month sum of standardized residuals → cross-sectional signal."""
    z = _build_residual_panel()
    # Rolling 11-month sum (since skip=1) of z, shifted so signal at month t
    # uses z from t-12 to t-2.
    # roll_window = FORMATION - SKIP = 11 months, ending at t-1
    rolled = z.shift(SKIP).rolling(FORMATION - SKIP).sum()
    return rolled


def build_residual_momentum_ls() -> pd.Series:
    """Monthly L/S return series for BHM 2011 residual momentum, NET of
    30bp/side cost. Equal-weight top 20% L / bottom 20% S."""
    signal = _build_signal_panel()
    returns_wide = _build_monthly_returns_panel()

    rows: list[dict] = []
    for t_idx, month in enumerate(signal.index):
        # Trade at month+1; need signal at month, return at month+1
        if t_idx + 1 >= len(signal.index):
            break
        traded_month = signal.index[t_idx + 1]
        sig_row = signal.loc[month].dropna()
        ret_row = returns_wide.loc[traded_month].dropna() if traded_month in returns_wide.index else pd.Series()
        common = sig_row.index.intersection(ret_row.index)
        if len(common) < MIN_NAMES_PER_LEG * 2 / max(TOP_Q, BOT_Q):
            continue
        sig_row = sig_row.loc[common]
        ret_row = ret_row.loc[common]
        hi_cut = sig_row.quantile(1.0 - TOP_Q)
        lo_cut = sig_row.quantile(BOT_Q)
        longs = sig_row[sig_row >= hi_cut].index
        shorts = sig_row[sig_row <= lo_cut].index
        if len(longs) < MIN_NAMES_PER_LEG or len(shorts) < MIN_NAMES_PER_LEG:
            continue
        long_ret = float(ret_row.loc[longs].mean())
        short_ret = float(ret_row.loc[shorts].mean())
        cost = 2.0 * RT_EQ_BPS / 10000.0    # 100% turnover assumption
        ls_net = long_ret - short_ret - cost
        rows.append({"month": traded_month, "net_ret": ls_net,
                      "n_long": int(len(longs)), "n_short": int(len(shorts))})

    out = pd.DataFrame(rows).set_index("month")["net_ret"].sort_index()
    return out.rename("residual_momentum")


def diagnostic_summary() -> dict:
    r = build_residual_momentum_ls().dropna()
    n = len(r)
    if n < 12:
        return {"n_months": n, "error": "insufficient data"}
    sh = float(r.mean() * 12 / (r.std() * np.sqrt(12)))
    vol = float(r.std() * np.sqrt(12))
    cum = (1.0 + r).cumprod()
    dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "n_months": n,
        "range":    f"{r.index.min().date()} -> {r.index.max().date()}",
        "ann_ret":  round(float(r.mean() * 12), 4),
        "ann_vol":  round(vol, 4),
        "sharpe":   round(sh, 3),
        "t_stat":   round(sh * np.sqrt(n / 12), 2),
        "maxdd":    round(dd, 4),
        "best":     round(float(r.max()), 4),
        "worst":    round(float(r.min()), 4),
        "hit_rate": round(float((r > 0).mean()), 3),
    }
