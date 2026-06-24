"""engine/validation/revenue_surprise.py — Tier-1 2nd-alpha: revenue-surprise drift (SURGE).

Per the multi-alpha agenda ([[project-multi-alpha-research-agenda-2026-05-20]]),
Tier 1 = squeeze the EARNINGS-EVENT family on the SAME announcements with a
DIFFERENT signal. Revenue (sales) surprise (Jegadeesh-Livnat 2006, JAR):
revenue surprises predict drift INCREMENTALLY to EPS surprises — revenue is
harder to manage and more persistent, so the signal is partially INDEPENDENT of
the EPS-based D_PEAD. The sharp test: is SURGE low-correlation with SUE? If yes,
it diversifies the book; if it's just SUE again, it doesn't.

Construction mirrors the SUE panel exactly (Bernard-Thomas seasonal random walk):
  delta_sale  = saleq − saleq_{q-4}            (seasonal Δ)
  sigma_8q    = std(delta_sale over [q-8..q-1])
  surge_raw   = delta_sale / sigma_8q
Then a calendar-month cross-sectional L/S (long top decile, short bottom) over
the 60-day post-rdq window — the SAME construction is applied to SUE so the
two series are apples-to-apples comparable for the correlation test.

Data: Compustat comp.fundq saleq/revtq + rdq, via the configured pgpass (direct
psycopg2 — the wrds wrapper prompts for credentials non-interactively).
Screened through alpha_factory.gate(); GREEN-only deploys.

VERDICT (2026-05-21, top-1500, WRDS restored): RED. SURGE standalone L/S Sharpe
-0.03 / t=-0.10 (no revenue-surprise drift). The JL-2006 INCREMENTAL claim also
fails here: SUE x SURGE double-confirmation HURTS (SUE-only quintile Sharpe 0.28
t=0.88 -> confirmed 0.03 t=0.11). corr(SURGE,SUE)=0.47, corr(SURGE,D_PEAD)=0.067.
Revenue surprise neither stands alone nor strengthens the EPS-based edge in the
large-cap universe (same arbitraged-in-large-cap pattern as the price factors;
the crude monthly constructor also under-powers SUE itself to t~0.9 vs the
production DHS-tilted t=4.64, so this is a weak lens — but standalone t=-0.10 +
negative increment is fairly conclusive). Small-cap version untested (lower
priority given the standalone deadness).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"   # has permno/gvkey/rdq/sue/mcap
_SALEQ = "data/cache/_compustat_saleq.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine(
        "postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
        connect_args={"sslmode": "require"})


def fetch_saleq(force: bool = False) -> pd.DataFrame:
    """Pull Compustat quarterly sales (saleq/revtq) + rdq for the panel's gvkeys.
    Cached. Needs WRDS (pgpass)."""
    if os.path.exists(_SALEQ) and not force:
        return pd.read_parquet(_SALEQ)
    from sqlalchemy import text
    gvkeys = sorted(pd.read_parquet(_PANEL)["gvkey"].dropna().astype(int).unique())
    gv6 = ",".join("'%s'" % str(g).zfill(6) for g in gvkeys)
    eng = _pg_engine()
    q = ("select gvkey, datadate, rdq, fyearq, fqtr, saleq, revtq "
         "from comp.fundq where datadate between '2011-06-01' and '2024-06-30' "
         "and gvkey in (%s) and (saleq is not null or revtq is not null)" % gv6)
    f = pd.read_sql(text(q), eng)
    eng.dispose()
    f.to_parquet(_SALEQ, index=False)
    logger.info("saleq: %d rows, %d gvkeys", len(f), f["gvkey"].nunique())
    return f


def build_surge_panel() -> pd.DataFrame:
    """Compute SURGE (standardized unexpected revenue) mirroring SUE, joined to
    the PEAD panel's permno/rdq/market_cap. One row per gvkey-quarter."""
    f = fetch_saleq().copy()
    f["sales"] = f["saleq"].fillna(f["revtq"])
    f = f.dropna(subset=["sales", "datadate"])
    f["gvkey"] = f["gvkey"].astype(int)
    f["datadate"] = pd.to_datetime(f["datadate"])
    f = f.sort_values(["gvkey", "datadate"])

    out = []
    for gv, g in f.groupby("gvkey"):
        g = g.drop_duplicates("datadate").sort_values("datadate").reset_index(drop=True)
        g["sales_lag4"] = g["sales"].shift(4)
        g["delta_sale"] = g["sales"] - g["sales_lag4"]
        # sigma over 8 PRIOR quarters of delta (exclude current), like sigma_8q
        g["sigma_8q_s"] = g["delta_sale"].shift(1).rolling(8, min_periods=4).std()
        g["surge_raw"] = g["delta_sale"] / g["sigma_8q_s"]
        out.append(g)
    s = pd.concat(out, ignore_index=True)
    s = s.dropna(subset=["surge_raw"])
    s = s[np.isfinite(s["surge_raw"])]

    # join permno + rdq + market_cap from the PEAD panel (match on gvkey + rdq)
    panel = pd.read_parquet(_PANEL)[["permno", "gvkey", "rdq", "sue", "market_cap_at_q"]].copy()
    panel["gvkey"] = panel["gvkey"].astype(int)
    panel["rdq"] = pd.to_datetime(panel["rdq"])
    s["rdq"] = pd.to_datetime(s["rdq"])
    m = s.merge(panel, on=["gvkey", "rdq"], how="inner")
    m["permno"] = m["permno"].astype(int)
    return m[["permno", "gvkey", "rdq", "sue", "surge_raw", "market_cap_at_q"]]


def _wide_monthly_returns(ret_path: str = _RET) -> pd.DataFrame:
    r = pd.read_parquet(ret_path)
    r["date"] = pd.to_datetime(r["date"])
    daily = r.pivot_table(index="date", columns="permno", values="ret").sort_index()
    m = (1.0 + daily.fillna(0.0)).resample("ME").prod() - 1.0
    return m.where(daily.resample("ME").count() > 5)


def build_event_ls_monthly(events: pd.DataFrame, signal_col: str,
                           hold_months: int = 2, decile: float = 0.1) -> pd.Series:
    """Calendar-month cross-sectional L/S: each month hold firms whose rdq was in
    the last `hold_months` and that were in the top (long) / bottom (short)
    `signal_col` decile at announcement; equal-weight, next-month return.

    The SAME function is applied to 'sue' and 'surge_raw' so the two series are
    apples-to-apples for the correlation test."""
    ev = events.dropna(subset=[signal_col, "rdq", "permno"]).copy()
    ev["rdq"] = pd.to_datetime(ev["rdq"])
    ev["rdq_month"] = ev["rdq"].dt.to_period("M").dt.to_timestamp("M")
    mret = _wide_monthly_returns()
    months = mret.index

    # decile cuts computed per rdq-cohort (cross-section at announcement)
    rows = []
    for m in months:
        active = ev[(ev["rdq_month"] <= m) &
                    (ev["rdq_month"] > m - pd.DateOffset(months=hold_months))]
        if len(active) < 50:
            continue
        hi = active[active[signal_col] >= active[signal_col].quantile(1 - decile)]["permno"]
        lo = active[active[signal_col] <= active[signal_col].quantile(decile)]["permno"]
        nxt_idx = m + pd.offsets.MonthEnd(1)
        if nxt_idx not in mret.index:
            continue
        nxt = mret.loc[nxt_idx]
        rl = nxt.reindex(hi.unique()).dropna()
        rs = nxt.reindex(lo.unique()).dropna()
        if len(rl) < 5 or len(rs) < 5:
            continue
        rows.append((nxt_idx, float(rl.mean() - rs.mean())))
    return pd.Series(dict(rows), name=signal_col).sort_index()
