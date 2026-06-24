"""composer.components.equity_data — shared loaders for EQUITY components.

Centralized so the UNIVERSE / SIGNAL / WEIGHTING components for equity
all read the same panels with consistent semantics. Pure offline — all
data sourced from data/cache/_crsp_msf_insider_* + _compustat_funda +
_crsp_ccm_link parquets (substrate committed 2026-06-05 in the
backfill_wrds_ccm_link commit).

What's exposed
--------------
crsp_mcap_wide()    [date × permno] of month-end market cap ($MM)
crsp_returns_wide() [date × permno] of monthly total returns (decimal)
ccm_link_map()      permno → gvkey (most recent active link only)
compustat_funda()   long-form (gvkey, datadate, ...) of annual fundamentals
book_to_market_wide(min_history_months=24) → [date × permno] of B/M

Sample selection notes
----------------------
- CRSP cache is a single-name "insider" universe panel (2013-10 onward,
  ~7000 unique permnos). Likely top of US equity market, not full CRSP.
- Compustat cache subset covers ~2300 gvkeys with annual fundamentals
  2011-2024. Join coverage with CRSP universe ~30-35% per cross-section
  (the cache was originally cut for an insider-trading study, so it's
  large-cap biased). Documented honestly here; downstream specs should
  note the universe restriction in their summary.
- This is enough to demonstrate the component pattern + run end-to-end
  through composer. For broader coverage, a follow-up substrate refresh
  would expand compustat cache to the full universe.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CACHE = _REPO_ROOT / "data" / "cache"

# Compustat lag: fiscal-year-end → public availability ~120 days later
# (10-K filing window). Add 60d buffer for conservatism per Fama-French
# practice (use FY-end + 6 months for July rebalance).
_COMPUSTAT_LAG_DAYS = 180


@functools.lru_cache(maxsize=1)
def crsp_returns_wide() -> pd.DataFrame:
    """[date × permno] monthly total returns. NaN where missing."""
    df = pd.read_parquet(_CACHE / "_crsp_msf_insider_universe.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    wide = df.pivot_table(index="date", columns="permno",
                            values="ret", aggfunc="last").sort_index()
    # Month-end normalization: ensure index is month-end timestamps.
    wide.index = wide.index.to_period("M").to_timestamp("M")
    return wide.astype("float64")


@functools.lru_cache(maxsize=1)
def crsp_mcap_wide() -> pd.DataFrame:
    """[date × permno] monthly market cap (raw units, typically $MM)."""
    df = pd.read_parquet(_CACHE / "_crsp_msf_insider_mcap.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["mcap"] = pd.to_numeric(df["mcap"], errors="coerce")
    wide = df.pivot_table(index="date", columns="permno",
                            values="mcap", aggfunc="last").sort_index()
    wide.index = wide.index.to_period("M").to_timestamp("M")
    return wide.astype("float64")


@functools.lru_cache(maxsize=1)
def ccm_link_map() -> dict:
    """permno → gvkey map. Picks the LATEST active link per permno
    (linkenddt is NaT if still active). Conservative: 1 gvkey per permno."""
    df = pd.read_parquet(_CACHE / "_crsp_ccm_link.parquet")
    df["linkdt"] = pd.to_datetime(df["linkdt"])
    df["linkenddt"] = pd.to_datetime(df["linkenddt"])
    # Keep latest-linkdt row per permno (active > closed, then most recent)
    df = df.sort_values(["permno", "linkenddt", "linkdt"], na_position="last")
    df = df.dropna(subset=["permno", "gvkey"]).drop_duplicates("permno", keep="last")
    return dict(zip(df["permno"].astype(int), df["gvkey"].astype(str)))


@functools.lru_cache(maxsize=1)
def compustat_funda() -> pd.DataFrame:
    """Long-form Compustat annual fundamentals. Coerces datadate to
    pandas Timestamp."""
    df = pd.read_parquet(_CACHE / "_compustat_funda.parquet")
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    return df


@functools.lru_cache(maxsize=1)
def book_to_market_wide() -> pd.DataFrame:
    """[date × permno] book-to-market ratio with PIT discipline.

    For each (permno, month):
      1. Look up the permno's gvkey via ccm_link_map.
      2. Find the most recent compustat row with datadate ≤ month - 180d
         (the public-availability lag).
      3. B/M = ceq / mcap_at_month.
      4. Filter out non-positive book equity (per FF convention).

    Sign convention: HIGHER B/M = more 'value' = LONG signal.
    """
    mcap = crsp_mcap_wide()
    funda = compustat_funda()
    link = ccm_link_map()

    # 1. Compustat: keep just gvkey, datadate, ceq; drop bad rows.
    fd = funda[["gvkey", "datadate", "ceq"]].dropna(subset=["ceq"])
    fd = fd[fd["ceq"] > 0]  # FF convention: drop negative book equity
    fd = fd.sort_values(["gvkey", "datadate"])

    # 2. Build a permno→[(available_from, ceq)] mapping for as-of lookup.
    # Index by gvkey for fast lookup per permno.
    fd_by_gv: dict[str, pd.DataFrame] = {
        g: grp[["datadate", "ceq"]].reset_index(drop=True)
        for g, grp in fd.groupby("gvkey")
    }

    # 3. For each (permno, date), look up B/M.
    out = pd.DataFrame(index=mcap.index, columns=mcap.columns,
                        dtype="float64")
    lag = pd.Timedelta(days=_COMPUSTAT_LAG_DAYS)
    n_resolved = 0
    for permno in mcap.columns:
        gv = link.get(int(permno))
        if not gv or gv not in fd_by_gv:
            continue
        gv_rows = fd_by_gv[gv]
        # For each month, take latest ceq with datadate+lag ≤ month
        mcap_col = mcap[permno]
        for dt in mcap_col.dropna().index:
            available = gv_rows[gv_rows["datadate"] + lag <= dt]
            if available.empty:
                continue
            ceq = float(available["ceq"].iloc[-1])
            m = float(mcap_col.loc[dt])
            if m > 0:
                out.at[dt, permno] = ceq / m
                n_resolved += 1
    logger.info("book_to_market_wide: %d (permno, month) values resolved",
                 n_resolved)
    return out


@functools.lru_cache(maxsize=1)
def gross_profitability_wide() -> pd.DataFrame:
    """[date × permno] gross profitability GP/A = (revt - cogs) / at, with
    PIT discipline (compustat datadate + 180d).

    Per Novy-Marx 2013 JFE "The Other Side of Value": gross profits
    scaled by total assets has predictive power for cross-sectional
    equity returns comparable to or stronger than book-to-market,
    especially within the value tilt. Sign: HIGHER = MORE profitable
    = LONG.

    Raw signal (not z-scored). Cross-sectional ranking happens
    downstream in the weighting component (top-q vs bottom-q).
    """
    mcap = crsp_mcap_wide()
    funda = compustat_funda()
    link = ccm_link_map()

    fd = funda[["gvkey", "datadate", "revt", "cogs", "at"]].dropna(subset=["at"])
    fd = fd[fd["at"] > 0]
    fd["gpoa"] = (fd["revt"].fillna(0) - fd["cogs"].fillna(0)) / fd["at"]
    fd = fd.sort_values(["gvkey", "datadate"])
    fd_by_gv = {g: grp[["datadate", "gpoa"]].reset_index(drop=True)
                for g, grp in fd.groupby("gvkey")}

    out = pd.DataFrame(index=mcap.index, columns=mcap.columns, dtype="float64")
    lag = pd.Timedelta(days=_COMPUSTAT_LAG_DAYS)
    n_resolved = 0
    for permno in mcap.columns:
        gv = link.get(int(permno))
        if not gv or gv not in fd_by_gv:
            continue
        gv_rows = fd_by_gv[gv]
        mcap_col = mcap[permno]
        for dt in mcap_col.dropna().index:
            available = gv_rows[gv_rows["datadate"] + lag <= dt]
            if available.empty:
                continue
            out.at[dt, permno] = float(available["gpoa"].iloc[-1])
            n_resolved += 1
    logger.info("gross_profitability_wide: %d (permno, month) values resolved",
                 n_resolved)
    return out


@functools.lru_cache(maxsize=1)
def quality_qmj_wide() -> pd.DataFrame:
    """[date × permno] simplified Quality-Minus-Junk composite signal.

    Per Asness-Frazzini-Pedersen 2019 RAS, full QMJ has 4 dimensions:
      Profitability / Growth / Safety / Payout
    each with multiple sub-measures and z-scored cross-sectionally.

    This implementation is SIMPLIFIED to 2 dimensions using fields the
    cached compustat funda actually has:
      profitability: GP/A = (revt - cogs) / at
      safety:        1 - (at - ceq) / at   # 1 minus debt-to-asset proxy
                                           # (total_liab ≈ at - ceq when
                                           #  non-controlling interest small)

    Each dimension z-scored cross-sectionally per month, then averaged.
    Sign: HIGHER = MORE quality = LONG. Junk (low quality) = SHORT.

    The "simplified" caveat is surfaced in the signal component's metadata
    so any verdict trained on this knows it's not the full AFP 2019 spec.
    A future C2.1 could add growth + payout when more compustat fields are
    cached. For now, GP/A + safety captures the 2 dimensions AFP describe
    as the dominant drivers of cross-sectional quality returns.
    """
    mcap = crsp_mcap_wide()
    funda = compustat_funda()
    link = ccm_link_map()

    # Pick fields, drop bad rows
    fd = funda[["gvkey", "datadate", "revt", "cogs", "at", "ceq"]].dropna(
        subset=["at"]
    )
    fd = fd[fd["at"] > 0]
    # Compute raw quality components per (gvkey, datadate)
    fd["gp"] = (fd["revt"].fillna(0) - fd["cogs"].fillna(0))
    fd["gpoa"] = fd["gp"] / fd["at"]                                     # profitability
    fd["leverage"] = (fd["at"] - fd["ceq"].fillna(0)) / fd["at"]
    fd["safety"] = 1.0 - fd["leverage"]                                  # safety
    fd = fd.sort_values(["gvkey", "datadate"])

    fd_by_gv: dict[str, pd.DataFrame] = {
        g: grp[["datadate", "gpoa", "safety"]].reset_index(drop=True)
        for g, grp in fd.groupby("gvkey")
    }

    # Build raw [date × permno] panels of gpoa + safety with PIT lag
    gpoa = pd.DataFrame(index=mcap.index, columns=mcap.columns,
                         dtype="float64")
    safety = pd.DataFrame(index=mcap.index, columns=mcap.columns,
                           dtype="float64")
    lag = pd.Timedelta(days=_COMPUSTAT_LAG_DAYS)
    for permno in mcap.columns:
        gv = link.get(int(permno))
        if not gv or gv not in fd_by_gv:
            continue
        gv_rows = fd_by_gv[gv]
        mcap_col = mcap[permno]
        for dt in mcap_col.dropna().index:
            available = gv_rows[gv_rows["datadate"] + lag <= dt]
            if available.empty:
                continue
            gpoa.at[dt, permno] = float(available["gpoa"].iloc[-1])
            safety.at[dt, permno] = float(available["safety"].iloc[-1])

    # Cross-sectional z-score per month, then average
    def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, pd.NA), axis=0)
    z_gpoa = _zscore_cs(gpoa)
    z_safety = _zscore_cs(safety)
    quality = (z_gpoa + z_safety) / 2.0
    logger.info("quality_qmj_wide: %d (permno, month) values resolved",
                 int(quality.notna().sum().sum()))
    return quality


@functools.lru_cache(maxsize=1)
def market_excess_returns_monthly() -> pd.Series:
    """Fama-French Mkt-RF monthly series, aligned to our CRSP monthly
    panel's month-end index. Source: cached FF weekly factors resampled
    via compound product (FF semantics: log/excess returns aggregate
    via (1+r) product to month).

    Used by BAB beta estimation and any other CAPM-style components.
    Cached for the process lifetime (lru_cache)."""
    from engine.factor_regression.ken_french import fetch_ff5_mom_weekly
    ff = fetch_ff5_mom_weekly()
    ff = ff.copy()
    ff.index = pd.to_datetime(ff.index)
    # Resample weekly excess market returns to monthly via compound
    mkt_w = pd.to_numeric(ff["MKT_RF"], errors="coerce").dropna()
    mkt_m = (1 + mkt_w).resample("ME").prod() - 1
    return mkt_m


@functools.lru_cache(maxsize=1)
def beta_wide(window_months: int = 36, min_periods: int = 24) -> pd.DataFrame:
    """[date × permno] rolling-window beta vs Fama-French market.

    For each permno: beta_t = cov(r_i, r_m, window) / var(r_m, window).
    PIT: use the beta from t-1 onward (shift after compute).

    Computed per-permno (vectorized via pandas rolling); ~3-5s on the
    7000-permno universe. lru_cache so first call pays, subsequent are
    free. Sign convention: HIGHER beta = riskier = BAB SHORT.
    """
    rw = crsp_returns_wide()
    mkt = market_excess_returns_monthly()
    mkt = mkt.reindex(rw.index).dropna()
    # Restrict r to dates where market exists
    rw_aligned = rw.loc[mkt.index]
    # Build excess returns (subtract RF? FF MKT_RF is already excess of RF
    # since it's the market minus the risk-free rate. We use raw permno
    # returns as a proxy for total return — the regression intercept
    # absorbs the average RF / alpha.)
    out = pd.DataFrame(index=rw_aligned.index, columns=rw_aligned.columns,
                        dtype="float64")
    mkt_var = mkt.rolling(window_months, min_periods=min_periods).var()
    for permno in rw_aligned.columns:
        r = rw_aligned[permno]
        if r.notna().sum() < min_periods:
            continue
        cov = r.rolling(window_months, min_periods=min_periods).cov(mkt)
        # beta_t known AT END of t but for trading at t+1 we shift later
        out[permno] = cov / mkt_var
    # PIT: signal_t uses beta from t-1
    return out.shift(1)


def universe_top_by_mcap(top_n: int) -> pd.DataFrame:
    """[date × permno] boolean: True iff permno is in the top_n by mcap
    on that month. Used as the EQUITY__US_LARGE / __SP500 / __RUSSELL_*
    subset proxy when finer universe definitions aren't available."""
    mcap = crsp_mcap_wide()
    out = pd.DataFrame(False, index=mcap.index, columns=mcap.columns)
    for dt, row in mcap.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        top = valid.nlargest(top_n).index
        out.loc[dt, top] = True
    return out
