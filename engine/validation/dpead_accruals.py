"""engine/validation/dpead_accruals.py — D_PEAD axis A.4: earnings-quality conditioning.

The last untried conditioning lever on the roadmap A-list (after A.1 short-leg
tilt = WIN, A.2 small-cap = WIN, A.3 reaction = NULL, dispersion = immaterial).

Mechanism (Sloan 1996 × Bernard-Thomas 1989): earnings surprises backed by CASH
(low accruals) are higher-quality and persist; surprises inflated by ACCRUALS
revert. So the post-earnings drift should be CLEANER for low-accrual high-SUE
firms. This conditions D_PEAD on earnings quality, not on a separate signal.

Accruals = Sloan balance-sheet definition (from cached comp.funda annual):
  ACC = (Δact − Δche) − (Δlct − Δdlc − Δtxp) − dp,  scaled by avg total assets
  (lower = more cash-backed = higher quality). Merged to each PEAD event by the
  most recent fiscal year ending >= 120 days before rdq (point-in-time). All from
  cache — no WRDS connection needed.

Conditioned L/S: long high-SUE-decile ∩ low-accrual half; short low-SUE-decile ∩
high-accrual half; vs the SUE-only baseline.

VERDICT (2026-05-21, top-1500): the BEST conditioning lever found besides the
A.1 tilt / A.2 small-cap, but a CANDIDATE not a shipped win:
  - on the crude monthly SUE lens: SUE-only net deflSR 0.36 / residual t=1.06 ->
    SUE×low-accrual net deflSR 0.60 / residual alpha 4.41%/yr t=1.78. Conditioning
    nearly DOUBLES the residual t and lifts alpha ~60%, direction = Sloan×BT.
  - ROBUST: the improvement (cond − base) holds in BOTH subsample halves
    (+2.74%/yr 2014-19, +2.16%/yr 2019-24); cond t 2.00 vs base 1.37 full-sample.
    Not one-period luck (unlike dispersion conditioning, which was immaterial and
    amplified the wrong slow-drift window per A.3).
  - BUT still RED (net deflSR 0.60, residual t=1.78 — both under the bars); shown
    only on the CRUDE monthly lens (which under-powers SUE to t~0.9 vs the
    production DHS-tilted t=4.64). Whether it lifts the ALREADY-GREEN production
    D_PEAD (net deflSR 0.99) is untested, and the median-accrual split HALVES the
    names per leg = real capacity cost on an already small-cap-dense, capacity-
    constrained alpha (A.2). => mechanism-validated refinement, integration into
    the production DHS 60-day-hold pipeline is the next step before it can be a
    banked win; NOT shipped. Did not sign-flip or tune past the honest result.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FUNDA = "data/cache/_compustat_funda.parquet"
_PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"


def compute_accruals() -> pd.DataFrame:
    """Sloan balance-sheet accruals per gvkey-fyear (lower = more cash-backed)."""
    f = pd.read_parquet(_FUNDA).copy()
    f["datadate"] = pd.to_datetime(f["datadate"])
    f = f.sort_values(["gvkey", "fyear"])
    for c in ["act", "che", "lct", "dlc", "txp", "dp", "at"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    g = f.groupby("gvkey")
    d_ca, d_cash = g["act"].diff(), g["che"].diff()
    d_cl, d_std, d_tp = g["lct"].diff(), g["dlc"].diff(), g["txp"].diff()
    avg_at = (f["at"] + g["at"].shift(1)) / 2
    f["accruals"] = ((d_ca - d_cash) - (d_cl - d_std.fillna(0) - d_tp.fillna(0)) - f["dp"]) / avg_at
    acc = f.dropna(subset=["accruals"])
    acc = acc[np.isfinite(acc["accruals"])][["gvkey", "datadate", "accruals"]]
    acc["gvkey"] = acc["gvkey"].astype(int)
    return acc


def build_conditioned_panel() -> pd.DataFrame:
    """PEAD events + point-in-time accruals (most recent fy >=120d before rdq)."""
    acc = compute_accruals()
    panel = pd.read_parquet(_PANEL).dropna(subset=["permno", "rdq", "sue", "gvkey"]).copy()
    panel["rdq"] = pd.to_datetime(panel["rdq"])
    panel["gvkey"] = panel["gvkey"].astype(int)
    panel["permno"] = panel["permno"].astype(int)
    m = panel.merge(acc, on="gvkey", how="inner")
    m = m[m["datadate"] <= m["rdq"] - pd.Timedelta(days=120)]
    return m.sort_values("datadate").groupby(["permno", "rdq"], as_index=False).last()


def _wide_monthly() -> pd.DataFrame:
    r = pd.read_parquet(_RET); r["date"] = pd.to_datetime(r["date"])
    daily = r.pivot_table(index="date", columns="permno", values="ret").sort_index()
    m = (1 + daily.fillna(0)).resample("ME").prod() - 1
    return m.where(daily.resample("ME").count() > 5)


def build_conditioned_ls(condition: bool, decile: float = 0.1, hold: int = 2) -> pd.Series:
    """Calendar-month L/S. condition=False: SUE-only decile baseline.
    condition=True: high-SUE ∩ low-accrual long / low-SUE ∩ high-accrual short."""
    ev = build_conditioned_panel()
    ev["rdq_m"] = ev["rdq"].dt.to_period("M").dt.to_timestamp("M")
    mret = _wide_monthly()
    rows = []
    for mo in mret.index:
        a = ev[(ev.rdq_m <= mo) & (ev.rdq_m > mo - pd.DateOffset(months=hold))]
        if len(a) < 60:
            continue
        hi = a[a.sue >= a.sue.quantile(1 - decile)]
        lo = a[a.sue <= a.sue.quantile(decile)]
        if condition:
            am = a.accruals.median()
            hi = hi[hi.accruals <= am]; lo = lo[lo.accruals >= am]
        nx = mo + pd.offsets.MonthEnd(1)
        if nx not in mret.index:
            continue
        rl = mret.loc[nx].reindex(hi.permno.unique()).dropna()
        rs = mret.loc[nx].reindex(lo.permno.unique()).dropna()
        if len(rl) < 5 or len(rs) < 5:
            continue
        rows.append((nx, float(rl.mean() - rs.mean())))
    return pd.Series(dict(rows)).sort_index().rename("dpead_lowaccrual" if condition else "dpead_sue")
