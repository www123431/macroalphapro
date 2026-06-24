"""engine/validation/fin_accruals.py — FIN (accruals half) orthogonality test.

The user asked whether deploying the DHS FIN factor (spec id=62) is meaningful. Verdict logic:
FIN's standalone alpha is ~0 (Sloan/Daniel-Titman are arbitraged); its ONLY value would be
DIVERSIFICATION — i.e. IF it is orthogonal to the existing book. So we test orthogonality, we
don't deploy blind.

NSI (the issuance half) needs shares-outstanding (csho) which isn't cached + WRDS is down, so this
tests the ACCRUALS half (Sloan 1996 balance-sheet accruals) — the larger, more-robust half. If
accruals is NOT orthogonal (likely, same-asset-class), full FIN almost certainly isn't either.

Look-ahead safe: annual accruals (fiscal datadate) are used only `lag_months` (default 6) AFTER
datadate, so the 10-K is public; held 12 months (annual rebalance), marked on CRSP monthly returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_FUNDA = "data/cache/_compustat_funda.parquet"
_LINK = "data/cache/_pead_ts_panel_2014_2023.parquet"   # gvkey<->permno link
_RET = "data/cache/crsp_hist_daily_ret.parquet"
_FF = "data/cache/ff_factors_weekly.parquet"


def _norm_gvkey(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def build_accruals_ls(q: float = 0.3, lag_months: int = 6) -> pd.Series:
    """Monthly long(low-accruals)/short(high-accruals) return series. Sloan balance-sheet
    accruals = (ΔACT−ΔCHE) − (ΔLCT−ΔDLC−ΔTXP) − DP, scaled by avg total assets."""
    fa = pd.read_parquet(_FUNDA)
    fa["datadate"] = pd.to_datetime(fa["datadate"])
    fa["gvkey"] = _norm_gvkey(fa["gvkey"])
    fa = fa.dropna(subset=["gvkey", "at"]).sort_values(["gvkey", "datadate"])

    g = fa.groupby("gvkey")
    for c in ["act", "che", "lct", "dlc", "txp", "at"]:
        fa[f"d_{c}"] = g[c].diff()
    fa["at_avg"] = (fa["at"] + g["at"].shift(1)) / 2.0
    fa["accruals"] = (((fa["d_act"] - fa["d_che"])
                       - (fa["d_lct"] - fa["d_dlc"].fillna(0) - fa["d_txp"].fillna(0)))
                      - fa["dp"]) / fa["at_avg"]
    fa = fa.dropna(subset=["accruals"])
    fa = fa[fa["at_avg"] > 0]
    # the date the annual figure is tradeable (10-K public)
    fa["known"] = fa["datadate"] + pd.DateOffset(months=lag_months)

    link = pd.read_parquet(_LINK)[["gvkey", "permno"]].drop_duplicates()
    link["gvkey"] = _norm_gvkey(link["gvkey"])
    fa = fa.merge(link.dropna(), on="gvkey", how="inner")

    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    mret = mret.where(daily.resample("ME").count() > 5)
    months = sorted(mret.index)

    rows = []
    for m in months:
        avail = fa[fa["known"] <= m].sort_values("known")
        if avail.empty:
            continue
        latest = avail.groupby("permno").tail(1)            # most recent known accruals per name
        latest = latest[latest["known"] >= m - pd.DateOffset(months=18)]  # not stale > 18mo
        a = latest.set_index("permno")["accruals"].dropna()
        if len(a) < 40:
            continue
        lo = a[a <= a.quantile(q)].index          # LOW accruals = LONG
        hi = a[a >= a.quantile(1 - q)].index       # HIGH accruals = SHORT
        r = mret.loc[m]
        rl = r.reindex(lo).dropna(); rs = r.reindex(hi).dropna()
        if len(rl) < 10 or len(rs) < 10:
            continue
        rows.append((m, float(rl.mean() - rs.mean())))
    return pd.Series(dict(rows)).sort_index().rename("accruals_ls")


def _ff_monthly() -> pd.DataFrame:
    ff = pd.read_parquet(_FF)
    ff.index = pd.to_datetime(ff.index)
    fac = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]
    fac = [c for c in fac if c in ff.columns]
    return ((1 + ff[fac]).resample("ME").prod() - 1)


def _ols_alpha_t(y: pd.Series, X: pd.DataFrame) -> tuple[float, float, float]:
    """Return (annualized alpha, alpha t-stat, R2). X without const; const added."""
    Z = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(Z) < 24:
        return (float("nan"), float("nan"), float("nan"))
    yv = Z["y"].values
    Xm = np.column_stack([np.ones(len(Z)), Z[X.columns].values])
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    resid = yv - Xm @ beta
    dof = len(Z) - Xm.shape[1]
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.inv(Xm.T @ Xm)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    alpha_m, alpha_se = beta[0], se[0]
    r2 = 1 - (resid @ resid) / (((yv - yv.mean()) ** 2).sum())
    return (float(alpha_m * 12), float(alpha_m / alpha_se), float(r2))


def orthogonality_test() -> dict:
    """Residual-α of the accruals factor vs FF5+UMD and vs FF5+UMD+PEAD, plus its correlation with
    the deployed PEAD leg. Verdict: orthogonal diversifier only if residual-α t ≥ ~3 AND low corr."""
    acc = build_accruals_ls()
    ffm = _ff_monthly()
    rf = ffm["RF"] if "RF" in ffm.columns else 0.0
    acc_x = (acc - rf).dropna()
    ff5umd = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"] if c in ffm.columns]

    a_ff, t_ff, r2_ff = _ols_alpha_t(acc_x, ffm[ff5umd])

    # add PEAD (the deployed equity leg) as a control
    from engine.portfolio.dpead_recon import build_dpead_recon_returns
    pead = (1 + build_dpead_recon_returns(long_short=True).clip(-0.2, 0.2)).resample("ME").prod() - 1
    Xp = pd.concat([ffm[ff5umd], pead.rename("PEAD")], axis=1)
    a_fp, t_fp, r2_fp = _ols_alpha_t(acc_x, Xp)

    j = pd.concat([acc.rename("acc"), pead.rename("pead")], axis=1).dropna()
    corr_pead = float(j["acc"].corr(j["pead"])) if len(j) > 12 else float("nan")

    def shp(x):
        x = x.dropna(); return float(x.mean() * 12 / (x.std() * np.sqrt(12))) if x.std() > 0 else float("nan")

    orthogonal = (t_fp == t_fp) and (abs(t_fp) >= 3.0) and (abs(corr_pead) < 0.3)
    return {
        "n_months": int(acc.dropna().size),
        "accruals_standalone_sharpe": round(shp(acc), 3),
        "alpha_vs_ff5umd_ann": round(a_ff, 4) if a_ff == a_ff else None,
        "alpha_t_vs_ff5umd": round(t_ff, 3) if t_ff == t_ff else None,
        "alpha_t_vs_ff5umd_pead": round(t_fp, 3) if t_fp == t_fp else None,
        "r2_vs_ff5umd_pead": round(r2_fp, 3) if r2_fp == r2_fp else None,
        "corr_with_pead": round(corr_pead, 3) if corr_pead == corr_pead else None,
        "orthogonal_diversifier": bool(orthogonal),
        "verdict": ("ORTHOGONAL — worth deploying" if orthogonal
                    else "NOT orthogonal / no residual alpha — deployment not worth it"),
        "note": "Accruals half of FIN only (NSI needs csho + WRDS, down). Gate: residual-α t≥3 AND |corr|<0.3.",
    }
