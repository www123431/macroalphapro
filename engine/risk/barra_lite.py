"""engine/risk/barra_lite.py — BARRA-style factor exposure check, Scope A.1.

Senior-quant gap (3) per [[project-end-to-end-vision-2026-05-30]]. Full
BARRA USE4 replication (10 styles x 60 industries, 15-20h) deliberately
NOT pursued — diminishing returns vs L4 self-research / cross-asset risk
model / IBES+Compustat fetcher wiring. This module is the high-ROI
v1 that answers the senior-quant questions:

  Q1. "Is D_PEAD just momentum in disguise?"
  Q2. "Do carry / TSMOM sleeves actually have ~0 equity-factor exposure?"
  Q3. "What's a sleeve's idiosyncratic alpha after MKT/SMB/MOM control?"

SCOPE A.1 (this module):
  - 3 factors constructed FROM OUR OWN CRSP CACHE (no ETF look-ahead):
    * MKT  = CRSP value-weighted index daily (vwretd)
    * SMB  = small-minus-big sort by market_cap_at_q (top vs bottom 30%)
    * MOM  = winners-minus-losers sort by past 12-1 month return
  - Time-series regression: sleeve_ret_m = alpha + beta_MKT * MKT_m
                            + beta_SMB * SMB_m + beta_MOM * MOM_m + eps
  - Newey-West HAC standard errors (lag 6 for monthly).

A.2 deferred (after Compustat fetcher wiring):
  - + HML (book-to-price), QMJ (quality), 11 GICS sector dummies.

OUTPUT: FactorExposureReport dataclass with betas, t-stats, alpha,
R^2, and a human-readable verdict for each sleeve.

Universe: CRSP top-1500 (matches D_PEAD universe). Factor returns are
computed cross-sectionally from CRSP daily, monthly-resampled.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CRSP_RET_PATH = REPO_ROOT / "data" / "cache" / "crsp_hist_daily_ret.parquet"
VWRETD_PATH = REPO_ROOT / "data" / "cache" / "crsp_vwretd_daily.parquet"
PEAD_PANEL_PATH = REPO_ROOT / "data" / "cache" / "_pead_ts_panel_2014_2023.parquet"
COMPUSTAT_FUNDA_PATH = REPO_ROOT / "data" / "cache" / "_compustat_funda_for_barra.parquet"
COMPUSTAT_GICS_PATH = REPO_ROOT / "data" / "cache" / "_compustat_company_gics.parquet"

# 11 GICS sectors (the institutional standard taxonomy as of 2018)
GICS_SECTORS = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}

# Factor-construction defaults
UNIVERSE_TOP_N = 1500
SORT_TOP_FRAC = 0.30
SORT_BOTTOM_FRAC = 0.30
MOM_LOOKBACK_MONTHS = 12
MOM_SKIP_MONTHS = 1
HAC_LAGS = 6

# PIT lag from Compustat datadate (fiscal year end) to public availability.
# 6 months is the conservative bracket — funda is typically filed 60-120 days
# post-quarter-end but some lagging filings push beyond. Per Fama-French
# 2015 standard practice + Hou-Xue-Zhang 2020 sensitivity check.
COMPUSTAT_PIT_LAG_DAYS = 180


@dataclasses.dataclass
class FactorExposureReport:
    """Output of regress_sleeve_on_factors()."""
    sleeve_name: str
    n_months: int
    alpha_monthly: float      # intercept (monthly)
    alpha_annualized: float   # alpha * 12 (since regress on monthly)
    alpha_t_hac: float        # HAC t-stat for alpha (Newey-West lag 6)
    betas: dict[str, float]   # {'MKT': ..., 'SMB': ..., 'MOM': ...}
    t_stats_hac: dict[str, float]
    r_squared: float
    factor_means_pct_per_yr: dict[str, float]  # for diagnostic
    verdict: str              # human-readable summary

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# -- Factor return construction ------------------------------------------

def _load_crsp_daily_returns() -> pd.DataFrame:
    if not CRSP_RET_PATH.exists():
        raise FileNotFoundError(f"missing {CRSP_RET_PATH}")
    df = pd.read_parquet(CRSP_RET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_market_caps_quarterly() -> pd.DataFrame:
    """From PEAD panel: (permno, rdq, market_cap_at_q). We use rdq to
    pin point-in-time market caps. Forward-fill within each permno
    until next rdq."""
    p = pd.read_parquet(PEAD_PANEL_PATH)
    p["rdq"] = pd.to_datetime(p["rdq"])
    return p[["permno", "rdq", "market_cap_at_q"]].sort_values(["permno", "rdq"])


def build_mkt_factor() -> pd.Series:
    """MKT = CRSP value-weighted index daily, monthly-resampled to monthly
    log-equivalent return (compounded)."""
    v = pd.read_parquet(VWRETD_PATH)
    v.index = pd.to_datetime(v.index)
    mkt_m = ((1 + v["vwretd"]).resample("ME").prod() - 1).rename("MKT")
    return mkt_m


def _pivot_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Pivot to (date, permno) -> ret. Compound to monthly."""
    pivot = daily.pivot_table(index="date", columns="permno",
                                  values="ret", aggfunc="first").sort_index()
    monthly = (1 + pivot).resample("ME").prod() - 1
    return monthly


def build_smb_factor(monthly_returns: pd.DataFrame,
                          market_caps: pd.DataFrame) -> pd.Series:
    """SMB = mean(small return) - mean(big return).
    Each month: rank universe by most recent market_cap_at_q,
    take bottom SORT_BOTTOM_FRAC as 'small', top SORT_TOP_FRAC as 'big'.
    Universe constrained to UNIVERSE_TOP_N by market cap.
    """
    rows = []
    for t in monthly_returns.index:
        # Use rdq <= t to pin point-in-time market cap
        recent_mc = market_caps[market_caps["rdq"] <= t].sort_values("rdq")
        last_mc = recent_mc.groupby("permno").last()
        if len(last_mc) < UNIVERSE_TOP_N // 3:
            continue
        universe = last_mc.nlargest(min(UNIVERSE_TOP_N, len(last_mc)),
                                            "market_cap_at_q")
        n = len(universe)
        n_bot = int(np.ceil(n * SORT_BOTTOM_FRAC))
        n_top = int(np.ceil(n * SORT_TOP_FRAC))
        small = universe.nsmallest(n_bot, "market_cap_at_q").index
        big = universe.nlargest(n_top, "market_cap_at_q").index
        r_small = monthly_returns.loc[t].reindex(small).dropna()
        r_big = monthly_returns.loc[t].reindex(big).dropna()
        if len(r_small) < 10 or len(r_big) < 10:
            continue
        rows.append((t, float(r_small.mean() - r_big.mean())))
    return pd.Series(dict(rows)).rename("SMB").sort_index()


def build_mom_factor(monthly_returns: pd.DataFrame,
                          market_caps: pd.DataFrame) -> pd.Series:
    """MOM = winners minus losers.
    For each month t: rank universe by past (t-12, t-1] cumulative return
    (skip month t to avoid microstructure reversal). Top SORT_TOP_FRAC
    = winners, bottom SORT_BOTTOM_FRAC = losers.
    """
    cum_lookback = (
        (1 + monthly_returns).rolling(MOM_LOOKBACK_MONTHS).apply(np.prod, raw=True) - 1
    )
    # Skip the most recent month: shift by MOM_SKIP_MONTHS
    cum_lookback = cum_lookback.shift(MOM_SKIP_MONTHS)

    rows = []
    for t in monthly_returns.index:
        recent_mc = market_caps[market_caps["rdq"] <= t].sort_values("rdq")
        last_mc = recent_mc.groupby("permno").last()
        if len(last_mc) < UNIVERSE_TOP_N // 3:
            continue
        universe_permnos = (
            last_mc.nlargest(min(UNIVERSE_TOP_N, len(last_mc)),
                                 "market_cap_at_q").index
        )
        # Past performance at t
        if t not in cum_lookback.index:
            continue
        past = cum_lookback.loc[t].reindex(universe_permnos).dropna()
        if len(past) < 100:
            continue
        n_top = int(np.ceil(len(past) * SORT_TOP_FRAC))
        n_bot = int(np.ceil(len(past) * SORT_BOTTOM_FRAC))
        winners = past.nlargest(n_top).index
        losers = past.nsmallest(n_bot).index
        r_win = monthly_returns.loc[t].reindex(winners).dropna()
        r_los = monthly_returns.loc[t].reindex(losers).dropna()
        if len(r_win) < 10 or len(r_los) < 10:
            continue
        rows.append((t, float(r_win.mean() - r_los.mean())))
    return pd.Series(dict(rows)).rename("MOM").sort_index()


def _load_compustat_funda() -> pd.DataFrame:
    """Load cached Compustat funda subset for the BARRA universe."""
    if not COMPUSTAT_FUNDA_PATH.exists():
        raise FileNotFoundError(f"missing {COMPUSTAT_FUNDA_PATH}; "
                                  f"run wrds_compustat.fetch_funda first")
    f = pd.read_parquet(COMPUSTAT_FUNDA_PATH)
    f["datadate"] = pd.to_datetime(f["datadate"])
    # gvkey may be stored as string '001690' or int 1690 — normalize to int
    f["gvkey"] = pd.to_numeric(f["gvkey"], errors="coerce").astype("Int64")
    return f


def _build_fundamentals_panel(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """For each month_end and permno, return the latest Compustat funda
    row whose datadate + COMPUSTAT_PIT_LAG_DAYS <= month_end.

    Returns: (month_end, permno, gvkey, ceq, ni, at, sale, b_to_m, roe)
    where b_to_m is computed from ceq / (last-known market cap at
    funda datadate, from PEAD panel market_cap_at_q) — used because
    we don't have monthly market caps cached.

    PIT discipline: a funda row with datadate=2020-12-31 becomes
    AVAILABLE for any month_end >= 2021-06-29 (180-day lag). This
    eliminates look-ahead bias from premature use of fiscal-end data.
    """
    funda = _load_compustat_funda()
    pead = pd.read_parquet(PEAD_PANEL_PATH)
    pead["gvkey"] = pd.to_numeric(pead["gvkey"], errors="coerce").astype("Int64")
    pead["rdq"] = pd.to_datetime(pead["rdq"])

    # gvkey -> permno mapping + ALL rdq/market_cap events from PEAD panel.
    # Keep duplicates: we need every rdq event to look up market cap near
    # each fiscal year end. Earlier (buggy) drop_duplicates kept only ONE
    # rdq per permno which caused b_to_m to vanish after 2016.
    gvk_to_permno = (
        pead.dropna(subset=["gvkey", "permno", "rdq", "market_cap_at_q"])
            [["gvkey", "permno", "market_cap_at_q", "rdq"]]
            .sort_values("rdq")
    )

    # Available-from date = datadate + lag
    funda = funda.copy()
    funda["available_from"] = (
        funda["datadate"] + pd.Timedelta(days=COMPUSTAT_PIT_LAG_DAYS)
    )
    funda["b_to_m_raw"] = funda["ceq"]    # numerator only; market_cap from PEAD join
    funda["roe_raw"] = funda["ni"] / funda["ceq"].replace(0, pd.NA)

    rows = []
    permnos_to_keep = set(monthly_returns.columns.tolist())
    for gvkey, sub in funda.groupby("gvkey"):
        # Pick matching permno (use the most-recent gvkey->permno map)
        permno_match = gvk_to_permno[gvk_to_permno["gvkey"] == gvkey]
        if permno_match.empty:
            continue
        permno = int(permno_match["permno"].iloc[-1])
        if permno not in permnos_to_keep:
            continue
        sub_sorted = sub.sort_values("available_from")
        # Approximate market cap at fiscal year end: pick the PEAD
        # market_cap_at_q from the rdq nearest to datadate. PEAD coverage
        # starts 2014, so funda rows before then will have NaN b_to_m
        # (excluded from HML cross-section). For later years the nearest-
        # rdq lookup is robust within ~1 year.
        permno_match_sorted = permno_match.sort_values("rdq")
        rdq_arr = permno_match_sorted["rdq"].values
        for _, row in sub_sorted.iterrows():
            d = row["datadate"]
            if len(rdq_arr) == 0:
                mc = float("nan")
            else:
                idx = np.searchsorted(rdq_arr, np.datetime64(d), side="right") - 1
                if idx < 0:
                    idx = 0
                nearest_rdq = permno_match_sorted.iloc[idx]
                # accept if within 1 year of fiscal year end (forward or backward)
                gap_days = abs((d - nearest_rdq["rdq"]).days)
                mc = (nearest_rdq["market_cap_at_q"]
                       if gap_days <= 365 else float("nan"))
            ceq = row["ceq"] if pd.notna(row["ceq"]) else float("nan")
            b_to_m = (ceq / mc) if (mc and mc > 0 and pd.notna(ceq) and ceq > 0) else float("nan")
            rows.append({
                "permno":         permno,
                "available_from": row["available_from"],
                "datadate":       d,
                "ceq":            ceq,
                "ni":             row["ni"],
                "b_to_m":         b_to_m,
                "roe":            row["roe_raw"],
            })

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    # ensure numeric dtype on b_to_m and roe (mixed nan/float can promote to object)
    panel["b_to_m"] = pd.to_numeric(panel["b_to_m"], errors="coerce")
    panel["roe"] = pd.to_numeric(panel["roe"], errors="coerce")
    panel = panel.sort_values(["permno", "available_from"])
    # For each (month_end, permno), the active row is the most recent
    # available_from <= month_end. Expand:
    expanded = []
    for permno, sub in panel.groupby("permno"):
        sub_sorted = sub.sort_values("available_from")
        af = sub_sorted["available_from"].values
        for t in monthly_returns.index:
            idx = np.searchsorted(af, np.datetime64(t), side="right") - 1
            if idx < 0:
                continue
            r = sub_sorted.iloc[idx]
            expanded.append({
                "month_end": t,
                "permno":    permno,
                "b_to_m":    r["b_to_m"],
                "roe":       r["roe"],
            })
    return pd.DataFrame(expanded)


def build_hml_factor(monthly_returns: pd.DataFrame,
                          fund_panel: pd.DataFrame,
                          market_caps: pd.DataFrame) -> pd.Series:
    """HML = high B/M (value) minus low B/M (growth).

    At each month t: take the universe (top-N by market cap), look up
    each name's PIT-corrected b_to_m, sort, take top SORT_TOP_FRAC as
    value, bottom SORT_BOTTOM_FRAC as growth. HML_t = mean(R_value_t) -
    mean(R_growth_t).
    """
    rows = []
    for t in monthly_returns.index:
        recent_mc = market_caps[market_caps["rdq"] <= t].sort_values("rdq")
        last_mc = recent_mc.groupby("permno").last()
        if len(last_mc) < UNIVERSE_TOP_N // 3:
            continue
        universe = last_mc.nlargest(min(UNIVERSE_TOP_N, len(last_mc)),
                                            "market_cap_at_q").index
        fp = fund_panel[(fund_panel["month_end"] == t)
                          & (fund_panel["permno"].isin(universe))
                          & (fund_panel["b_to_m"].notna())]
        if len(fp) < 100:
            continue
        n_top = int(np.ceil(len(fp) * SORT_TOP_FRAC))
        n_bot = int(np.ceil(len(fp) * SORT_BOTTOM_FRAC))
        value_permnos = fp.nlargest(n_top, "b_to_m")["permno"].values
        growth_permnos = fp.nsmallest(n_bot, "b_to_m")["permno"].values
        r_value = monthly_returns.loc[t].reindex(value_permnos).dropna()
        r_growth = monthly_returns.loc[t].reindex(growth_permnos).dropna()
        if len(r_value) < 10 or len(r_growth) < 10:
            continue
        rows.append((t, float(r_value.mean() - r_growth.mean())))
    return pd.Series(dict(rows)).rename("HML").sort_index()


def build_qmj_factor(monthly_returns: pd.DataFrame,
                          fund_panel: pd.DataFrame,
                          market_caps: pd.DataFrame) -> pd.Series:
    """QMJ = high quality (ROE) minus low quality, equal-weighted.

    Asness-Frazzini-Pedersen 2019 full QMJ has 4 sub-components
    (profitability + growth + safety + payout). This Phase-2 proxy
    uses ROE only as the most-dominant component. Phase 5 should
    upgrade to the full 4-component QMJ.
    """
    rows = []
    for t in monthly_returns.index:
        recent_mc = market_caps[market_caps["rdq"] <= t].sort_values("rdq")
        last_mc = recent_mc.groupby("permno").last()
        if len(last_mc) < UNIVERSE_TOP_N // 3:
            continue
        universe = last_mc.nlargest(min(UNIVERSE_TOP_N, len(last_mc)),
                                            "market_cap_at_q").index
        fp = fund_panel[(fund_panel["month_end"] == t)
                          & (fund_panel["permno"].isin(universe))
                          & (fund_panel["roe"].notna())]
        if len(fp) < 100:
            continue
        n_top = int(np.ceil(len(fp) * SORT_TOP_FRAC))
        n_bot = int(np.ceil(len(fp) * SORT_BOTTOM_FRAC))
        quality_permnos = fp.nlargest(n_top, "roe")["permno"].values
        junk_permnos = fp.nsmallest(n_bot, "roe")["permno"].values
        r_q = monthly_returns.loc[t].reindex(quality_permnos).dropna()
        r_j = monthly_returns.loc[t].reindex(junk_permnos).dropna()
        if len(r_q) < 10 or len(r_j) < 10:
            continue
        rows.append((t, float(r_q.mean() - r_j.mean())))
    return pd.Series(dict(rows)).rename("QMJ").sort_index()


def _load_gics_mapping() -> pd.DataFrame:
    """Load gvkey -> gsector mapping from cached Compustat company table.

    Returns: (gvkey, gsector) where gsector is the 2-digit GICS code.
    """
    if not COMPUSTAT_GICS_PATH.exists():
        raise FileNotFoundError(f"missing {COMPUSTAT_GICS_PATH}; "
                                  f"run scripts/fetch_compustat_gics.py first")
    g = pd.read_parquet(COMPUSTAT_GICS_PATH)
    g["gvkey"] = pd.to_numeric(g["gvkey"], errors="coerce").astype("Int64")
    g["gsector"] = g["gsector"].astype("string")
    return g[["gvkey", "gsector"]].dropna()


def _build_permno_sector_map(monthly_returns: pd.DataFrame) -> dict[int, str]:
    """For each permno in our universe, return its GICS sector code.

    Joins via PEAD panel's gvkey<->permno mapping. Returns dict
    {permno: '10'/'15'/.../'60'}.
    """
    gics = _load_gics_mapping()
    pead = pd.read_parquet(PEAD_PANEL_PATH)
    pead["gvkey"] = pd.to_numeric(pead["gvkey"], errors="coerce").astype("Int64")
    gvk_to_permno = (
        pead.dropna(subset=["gvkey", "permno"])
            .drop_duplicates(subset=["gvkey", "permno"])[["gvkey", "permno"]]
    )
    j = gvk_to_permno.merge(gics, on="gvkey", how="inner")
    out = {}
    for _, row in j.iterrows():
        p = int(row["permno"])
        if p in monthly_returns.columns and pd.notna(row["gsector"]):
            out[p] = str(row["gsector"])
    return out


def build_sector_factors(monthly_returns: pd.DataFrame,
                              market_caps: pd.DataFrame,
                              permno_to_sector: dict[int, str]) -> pd.DataFrame:
    """11 GICS sector factor returns: equal-weighted sector portfolio MINUS
    equal-weighted full universe. Output column names: SEC_10, SEC_15, ...

    Sector excess return formulation isolates the within-universe sector
    bet. Phase 5 will replace with cross-sectional regression-derived
    industry factor returns (proper BARRA USE4 method).
    """
    rows = {}
    sector_codes = sorted(GICS_SECTORS.keys())
    for t in monthly_returns.index:
        recent_mc = market_caps[market_caps["rdq"] <= t].sort_values("rdq")
        last_mc = recent_mc.groupby("permno").last()
        if len(last_mc) < UNIVERSE_TOP_N // 3:
            continue
        universe_permnos = last_mc.nlargest(
            min(UNIVERSE_TOP_N, len(last_mc)), "market_cap_at_q"
        ).index.tolist()
        # Universe mean return (equal-weighted)
        r_universe = monthly_returns.loc[t].reindex(universe_permnos).dropna()
        if len(r_universe) < 100:
            continue
        univ_mean = r_universe.mean()
        out_row = {"t": t}
        for sec in sector_codes:
            sec_permnos = [p for p in universe_permnos
                           if permno_to_sector.get(p) == sec]
            if len(sec_permnos) < 5:
                out_row[f"SEC_{sec}"] = float("nan")
                continue
            r_sec = monthly_returns.loc[t].reindex(sec_permnos).dropna()
            if len(r_sec) < 5:
                out_row[f"SEC_{sec}"] = float("nan")
                continue
            out_row[f"SEC_{sec}"] = float(r_sec.mean() - univ_mean)
        rows[t] = out_row
    df = pd.DataFrame.from_dict(rows, orient="index").drop(columns=["t"],
                                                                errors="ignore")
    df.index.name = None
    return df


def build_factor_returns(phase: int = 1) -> pd.DataFrame:
    """Construct factor monthly return panel from cached data.

    phase=1: MKT + SMB + MOM (no Compustat needed)
    phase=2: + HML + QMJ (needs Compustat funda cache)
    phase=3: + 11 GICS sector excess returns (SEC_10 .. SEC_60)
             (needs Compustat company-gics cache)
    """
    daily = _load_crsp_daily_returns()
    monthly_ret = _pivot_returns(daily)
    market_caps = _load_market_caps_quarterly()

    mkt = build_mkt_factor()
    smb = build_smb_factor(monthly_ret, market_caps)
    mom = build_mom_factor(monthly_ret, market_caps)

    if phase == 1:
        return pd.concat([mkt, smb, mom], axis=1).dropna()

    fund_panel = _build_fundamentals_panel(monthly_ret)
    hml = build_hml_factor(monthly_ret, fund_panel, market_caps)
    qmj = build_qmj_factor(monthly_ret, fund_panel, market_caps)

    if phase == 2:
        return pd.concat([mkt, smb, mom, hml, qmj], axis=1).dropna()

    # Phase 3+
    permno_to_sector = _build_permno_sector_map(monthly_ret)
    sectors = build_sector_factors(monthly_ret, market_caps, permno_to_sector)
    factors = pd.concat([mkt, smb, mom, hml, qmj, sectors], axis=1).dropna()
    return factors


# -- Regression with HAC SE -----------------------------------------------

def _hac_se(X: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC standard errors. Returns SE vector for each coef."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    # S = sum_t (resid_t * X_t)(resid_t * X_t)' + Bartlett-weighted lags
    u = residuals[:, None] * X    # shape (n, k)
    S = u.T @ u / n
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1)
        Gamma = u[L:].T @ u[:-L] / n
        S += w * (Gamma + Gamma.T)
    var_beta = n * XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(var_beta))


def regress_sleeve_by_regime(
    sleeve_returns: pd.Series,
    regime_series: pd.Series,
    factors: pd.DataFrame | None = None,
    sleeve_name: str = "sleeve",
    hac_lags: int = HAC_LAGS,
    min_months_per_regime: int = 18,
) -> dict[str, FactorExposureReport]:
    """FLAW 5 fix per [[project-loop-design-flaws-discovered-2026-05-30]].

    Run regress_sleeve_on_factors SEPARATELY for each regime label in
    regime_series. Surfaces time-varying factor exposures that aggregated
    regression masks — critical for evaluating insurance candidates
    (their value is in STRESS regimes; aggregated test under-reports it).

    Args:
      sleeve_returns: monthly Series indexed by date
      regime_series: monthly Series of regime labels (e.g.,
        {'CALM', 'NORMAL', 'STRESS'} from build_vix_regime_monthly())
      factors: factor return panel; defaults to build_factor_returns()
      min_months_per_regime: regimes with fewer obs are skipped (too noisy)

    Returns: dict {regime_label: FactorExposureReport}.

    Notes:
      Per-regime n_months is small (typically 20-80 months for CALM/STRESS,
      40-100 for NORMAL). HAC SE may be unreliable for regimes with n < 24.
      Verdict text in each report reflects that regime's local profile.
    """
    if factors is None:
        factors = build_factor_returns()

    s = sleeve_returns.copy()
    s.index = pd.to_datetime(s.index)
    s = s.resample("ME").last() if not s.index.equals(
        s.index.to_period("M").to_timestamp("M")) else s
    r = regime_series.copy()
    r.index = pd.to_datetime(r.index)
    r = r.resample("ME").last() if not r.index.equals(
        r.index.to_period("M").to_timestamp("M")) else r

    J = pd.concat([s.rename("y"), r.rename("regime"), factors],
                       axis=1).dropna(subset=["y", "regime"])
    out: dict[str, FactorExposureReport] = {}
    for regime_label in sorted(J["regime"].unique()):
        sub = J[J["regime"] == regime_label]
        if len(sub) < min_months_per_regime:
            continue
        sub_s = sub["y"]
        sub_f = sub[factors.columns]
        try:
            rep = regress_sleeve_on_factors(
                sub_s, sub_f, sleeve_name=f"{sleeve_name}@{regime_label}",
                hac_lags=min(hac_lags, max(2, len(sub) // 4)),
                min_obs=min_months_per_regime,
            )
            out[str(regime_label)] = rep
        except ValueError as exc:
            logger.warning("regime %s regression failed for %s: %s",
                              regime_label, sleeve_name, exc)
    return out


def regress_sleeve_on_factors(
    sleeve_returns: pd.Series,
    factors: pd.DataFrame | None = None,
    sleeve_name: str = "sleeve",
    hac_lags: int = HAC_LAGS,
    min_obs: int = 24,
) -> FactorExposureReport:
    """Time-series regression of sleeve monthly return on the supplied
    factor panel with Newey-West HAC SE.

    Auto-detects whether factors is Phase-1 (MKT/SMB/MOM) or Phase-2
    (+HML/QMJ) or Phase-3 (+sectors) by which columns are present.

    min_obs: minimum overlap months. Default 24 (institutional standard
    for HAC reliability). Regime-stratified callers may lower to 18-20
    when working with short regime sub-samples; results should be
    interpreted with caution at low n.
    """
    if factors is None:
        factors = build_factor_returns()

    s = sleeve_returns.copy()
    s.index = pd.to_datetime(s.index)
    s = s.resample("ME").last() if not s.index.equals(s.index.to_period("M").to_timestamp("M")) else s
    J = pd.concat([s.rename("y"), factors], axis=1).dropna()
    if len(J) < min_obs:
        raise ValueError(f"too few overlap months: {len(J)} < {min_obs}")

    factor_cols = [c for c in factors.columns if c in J.columns]
    y = J["y"].values
    X = np.column_stack(
        [np.ones(len(J))] + [J[c].values for c in factor_cols]
    )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    se = _hac_se(X, resid, lags=hac_lags)
    t = beta / se

    alpha_m = float(beta[0])
    betas = {c: float(beta[i + 1]) for i, c in enumerate(factor_cols)}
    t_stats = {"alpha": float(t[0])}
    for i, c in enumerate(factor_cols):
        t_stats[c] = float(t[i + 1])

    factor_means_pct_per_yr = {
        c: float(factors[c].dropna().mean() * 12.0 * 100.0)
        for c in factor_cols
    }

    verdict_lines = []
    # Highlight significant exposures (|t| >= 2.0)
    for c in factor_cols:
        tval = t_stats[c]
        if abs(tval) >= 2.0:
            verdict_lines.append(
                f"{c} exposure |t|={abs(tval):.2f} sig (beta={betas[c]:+.3f})"
            )
    if t_stats["alpha"] >= 2.0:
        n_fac = len(factor_cols)
        verdict_lines.append(
            f"alpha t={t_stats['alpha']:.2f} survives {n_fac}-factor control"
        )
    elif t_stats["alpha"] < 1.0:
        n_fac = len(factor_cols)
        verdict_lines.append(
            f"alpha t={t_stats['alpha']:.2f} subsumed by {n_fac} factors"
        )

    return FactorExposureReport(
        sleeve_name=sleeve_name,
        n_months=len(J),
        alpha_monthly=alpha_m,
        alpha_annualized=alpha_m * 12.0,
        alpha_t_hac=t_stats["alpha"],
        betas=betas,
        t_stats_hac=t_stats,
        r_squared=r2,
        factor_means_pct_per_yr=factor_means_pct_per_yr,
        verdict=" | ".join(verdict_lines) if verdict_lines else "no significant exposures",
    )
