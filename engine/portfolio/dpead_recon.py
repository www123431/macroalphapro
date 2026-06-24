"""engine/portfolio/dpead_recon.py — reproducible D_PEAD return reconstruction.

Closes audit residual #3 (docs/live_delivers_backtest_audit_2026-05-25.md): the original
`_dpead_recon_base.parquet` (the D_PEAD daily return series feeding the combined-book 1.04)
had NO writer in the code tree — so the 1.04 was only "inferred clean", not reproducible.

This module rebuilds that series FROM SOURCE using the SAME clean, look-ahead-free B-T 1989
SUE PEAD logic the live signal uses (engine.portfolio.paper_trade_combined.get_d_pead_signal):
  - firms with rdq in (t - PEAD_WINDOW_DAYS, t]  → only PAST announcements (no look-ahead)
  - top-N by point-in-time market_cap_at_q (cshoq×prccq, survivorship-free panel)
  - cross-section rank by SUE; long top decile (+ short bottom decile if long_short)
  - daily return = mean(long daily ret) − mean(short daily ret), from CRSP daily returns

Inputs are cached (no WRDS pull): the long SUE panel (2014-2023) + CRSP daily returns
(survivorship-free, incl. delisted). The reproduced series is validated against the existing
artifact by correlation + Sharpe (test_dpead_recon.py); if it tracks, the 1.04 is reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"
_GICS = "data/cache/_compustat_company_gics.parquet"
_SICH_PIT = "data/cache/_compustat_funda_sich_pit.parquet"
_RECON_OUT = "data/cache/_dpead_recon_base_reproduced.parquet"
_SECTOR_NEUTRAL_OUT = "data/cache/_dpead_sector_neutral.parquet"
_SECTOR_NEUTRAL_PIT_OUT = "data/cache/_dpead_sector_neutral_pit.parquet"

# SIC 2-digit -> Fama-French 12-industry mapping. Standard academic
# classification (Kenneth French Data Library FF12). Used by
# build_dpead_sector_neutral_returns_pit to assign PIT sector buckets.
def _sic_to_ff12(sic: int) -> int:
    if not sic or sic <= 0:
        return 12   # Other (catch-all)
    if (100 <= sic <= 999) or (2000 <= sic <= 2399) or (2700 <= sic <= 2749) \
       or (2770 <= sic <= 2799) or (3100 <= sic <= 3199) \
       or (3940 <= sic <= 3989):
        return 1    # NoDur
    if (2500 <= sic <= 2519) or (2590 <= sic <= 2599) \
       or (3630 <= sic <= 3659) or (3710 <= sic <= 3711) \
       or (3714 <= sic <= 3714) or (3716 <= sic <= 3716) \
       or (3750 <= sic <= 3751) or (3792 <= sic <= 3792) \
       or (3900 <= sic <= 3939) or (3990 <= sic <= 3999):
        return 2    # Durbl
    if (2520 <= sic <= 2589) or (2600 <= sic <= 2699) \
       or (2750 <= sic <= 2769) or (3000 <= sic <= 3099) \
       or (3200 <= sic <= 3569) or (3580 <= sic <= 3629) \
       or (3700 <= sic <= 3709) or (3712 <= sic <= 3713) \
       or (3715 <= sic <= 3715) or (3717 <= sic <= 3749) \
       or (3752 <= sic <= 3791) or (3793 <= sic <= 3799) \
       or (3830 <= sic <= 3839) or (3860 <= sic <= 3899):
        return 3    # Manuf
    if (1200 <= sic <= 1399) or (2900 <= sic <= 2999):
        return 4    # Enrgy
    if (2800 <= sic <= 2829) or (2840 <= sic <= 2899):
        return 5    # Chems
    if (3570 <= sic <= 3579) or (3660 <= sic <= 3692) \
       or (3694 <= sic <= 3699) or (3810 <= sic <= 3829) \
       or (7370 <= sic <= 7379):
        return 6    # BusEq
    if 4800 <= sic <= 4899:
        return 7    # Telcm
    if 4900 <= sic <= 4949:
        return 8    # Utils
    if (5000 <= sic <= 5999) or (7200 <= sic <= 7299) \
       or (7600 <= sic <= 7699):
        return 9    # Shops
    if (2830 <= sic <= 2839) or (3693 <= sic <= 3693) \
       or (3840 <= sic <= 3859) or (8000 <= sic <= 8099):
        return 10   # Hlth
    if 6000 <= sic <= 6999:
        return 11   # Money
    return 12       # Other

PEAD_WINDOW_DAYS: int = 90      # ~60 trading days drift window (Path D spec)
UNIVERSE_TOP_N: int = 1500
DECILE: float = 0.10
MIN_LEG_NAMES: int = 3
SECTOR_MIN_NAMES: int = 8   # minimum names per sector to apply per-sector ranking


def build_dpead_recon_returns(pead_window_days: int = PEAD_WINDOW_DAYS,
                              universe_top_n: int = UNIVERSE_TOP_N,
                              decile: float = DECILE,
                              long_short: bool = True,
                              ret_path: str = _RET) -> pd.Series:
    """Reproduce the D_PEAD daily return series from cached source. long_short=True →
    market-neutral top-vs-bottom decile (matches the recon-base low-vol/≈0-mean profile);
    long_short=False → long-only top decile (the live deployed leg). ret_path defaults to the
    registered crsp.dsf panel; pass the delisting-spliced panel to measure the dlret-refined
    series (engine/validation/delisting_merge.py) WITHOUT changing the default/live behavior."""
    panel = pd.read_parquet(_PANEL)
    panel["rdq"] = pd.to_datetime(panel["rdq"])
    panel = panel.dropna(subset=["sue", "market_cap_at_q", "rdq"]).sort_values("rdq")

    ret = pd.read_parquet(ret_path)
    ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()

    # Rolling window via searchsorted on the sorted rdq array (fast vs per-day boolean filter).
    rdq_vals = panel["rdq"].values
    rows: list[tuple] = []
    for t in daily.index:
        lo_i = np.searchsorted(rdq_vals, np.datetime64(t - pd.Timedelta(days=pead_window_days)), "right")
        hi_i = np.searchsorted(rdq_vals, np.datetime64(t), "right")
        if hi_i - lo_i < 20:
            continue
        win = panel.iloc[lo_i:hi_i]
        # latest SUE + market cap per firm in the window
        act = win.groupby("permno").last()
        act = act.nlargest(min(universe_top_n, len(act)), "market_cap_at_q")
        sue = act["sue"]
        if len(sue) < 20:
            continue
        r = daily.loc[t]
        long_r = r.reindex(sue[sue >= sue.quantile(1 - decile)].index).dropna()
        if len(long_r) < MIN_LEG_NAMES:
            continue
        if long_short:
            short_r = r.reindex(sue[sue <= sue.quantile(decile)].index).dropna()
            if len(short_r) < MIN_LEG_NAMES:
                continue
            rows.append((t, float(long_r.mean() - short_r.mean())))
        else:
            rows.append((t, float(long_r.mean())))

    return pd.Series(dict(rows)).sort_index().rename("base")


def regenerate_and_save(long_short: bool = True) -> str:
    """Build + persist the reproduced recon series (separate path; does not overwrite the
    original artifact until verified)."""
    s = build_dpead_recon_returns(long_short=long_short)
    s.to_frame("base").to_parquet(_RECON_OUT)
    return _RECON_OUT


# ── Sector-neutral variant (per [[project-barra-phase-chain-2026-05-30]] B.1) ──

def _load_permno_to_sector() -> dict[int, str]:
    """Build {permno: gsector} dict from cached PEAD panel + Compustat
    company-gics mapping. Returns 2-digit GICS codes ('10'..'60')."""
    pead = pd.read_parquet(_PANEL)
    pead["gvkey"] = pd.to_numeric(pead["gvkey"], errors="coerce").astype("Int64")
    gics = pd.read_parquet(_GICS)
    gics["gvkey"] = pd.to_numeric(gics["gvkey"], errors="coerce").astype("Int64")
    gics["gsector"] = gics["gsector"].astype("string")
    pmap = (pead.dropna(subset=["gvkey", "permno"])
                 .drop_duplicates(subset=["gvkey", "permno"])[["gvkey", "permno"]])
    j = pmap.merge(gics[["gvkey", "gsector"]], on="gvkey", how="inner")
    out = {}
    for _, row in j.iterrows():
        if pd.notna(row["gsector"]):
            out[int(row["permno"])] = str(row["gsector"])
    return out


def build_dpead_sector_neutral_returns(
    pead_window_days: int = PEAD_WINDOW_DAYS,
    universe_top_n: int = UNIVERSE_TOP_N,
    decile: float = DECILE,
    ret_path: str = _RET,
    sector_min_names: int = SECTOR_MIN_NAMES,
) -> pd.Series:
    """Build D_PEAD daily L/S return SECTOR-NEUTRAL by construction.

    Method per institutional standard (BARRA / Fama-French sector-
    neutralized portfolio):
      1. Take same top-1500 by market_cap_at_q universe as plain D_PEAD.
      2. For each GICS sector with >= sector_min_names firms in window:
         - rank by SUE within the sector
         - long the top `decile` of the sector
         - short the bottom `decile` of the sector
      3. Combine: long leg = mean across all sector longs, short leg =
         mean across all sector shorts, equal-weighted across sectors.
      4. Return = mean(long leg) - mean(short leg).

    Effect: removes the sector concentration that the Phase 3 BARRA
    audit revealed was contributing ~30% of D_PEAD's reported alpha.
    Trade-off: we give up the cross-sector SUE-spread premium (sectors
    with systematically higher SUE were paying us). Expected:
      Sharpe falls modestly (0.05-0.15 in net Sharpe)
      R^2 with sectors collapses (close to 0 instead of ~0.10-0.15)
      Residual alpha t-stat improves (closer to pure PEAD)
    """
    panel = pd.read_parquet(_PANEL)
    panel["rdq"] = pd.to_datetime(panel["rdq"])
    panel = panel.dropna(subset=["sue", "market_cap_at_q", "rdq", "permno"]).sort_values("rdq")

    ret = pd.read_parquet(ret_path)
    ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()

    permno_to_sector = _load_permno_to_sector()
    rdq_vals = panel["rdq"].values

    rows: list[tuple] = []
    for t in daily.index:
        lo_i = np.searchsorted(rdq_vals, np.datetime64(t - pd.Timedelta(days=pead_window_days)), "right")
        hi_i = np.searchsorted(rdq_vals, np.datetime64(t), "right")
        if hi_i - lo_i < 20:
            continue
        win = panel.iloc[lo_i:hi_i]
        act = win.groupby("permno").last()
        act = act.nlargest(min(universe_top_n, len(act)), "market_cap_at_q")
        if len(act) < 20:
            continue
        # Attach sector
        act = act.copy()
        act["sector"] = act.index.map(lambda p: permno_to_sector.get(int(p)))
        act = act.dropna(subset=["sector"])
        if len(act) < 50:
            continue

        r = daily.loc[t]
        sector_long_returns = []
        sector_short_returns = []
        for sector, sub in act.groupby("sector"):
            if len(sub) < sector_min_names:
                continue
            sue = sub["sue"]
            top = sue[sue >= sue.quantile(1 - decile)].index
            bot = sue[sue <= sue.quantile(decile)].index
            l = r.reindex(top).dropna()
            s = r.reindex(bot).dropna()
            if len(l) < 1 or len(s) < 1:
                continue
            sector_long_returns.append(float(l.mean()))
            sector_short_returns.append(float(s.mean()))
        if len(sector_long_returns) < 3:
            continue
        # Equal-weight across sectors (this is what makes it sector-neutral)
        long_ret = np.mean(sector_long_returns)
        short_ret = np.mean(sector_short_returns)
        rows.append((t, float(long_ret - short_ret)))

    return pd.Series(dict(rows)).sort_index().rename("base")


def regenerate_and_save_sector_neutral() -> str:
    """Build + persist the sector-neutral D_PEAD daily series."""
    s = build_dpead_sector_neutral_returns()
    s.to_frame("base").to_parquet(_SECTOR_NEUTRAL_OUT)
    return _SECTOR_NEUTRAL_OUT


# ── PIT sector-neutral D_PEAD (per [[project-sector-neutral-dpead-real-2x-improvement-2026-05-31]] P-D1 fix) ──

def _load_permno_to_sector_pit() -> pd.DataFrame:
    """Build PIT permno-sector mapping from Compustat funda.sich.

    Returns DataFrame with columns: [permno, datadate, ff12].
    Caller looks up sector by finding the most recent datadate <=
    current asof date for each permno.
    """
    sich = pd.read_parquet(_SICH_PIT)
    sich["gvkey"] = pd.to_numeric(sich["gvkey"], errors="coerce").astype("Int64")
    sich["datadate"] = pd.to_datetime(sich["datadate"])
    sich["sich"] = pd.to_numeric(sich["sich"], errors="coerce")
    sich["ff12"] = sich["sich"].apply(
        lambda x: _sic_to_ff12(int(x)) if pd.notna(x) and x > 0 else 12
    )

    pead = pd.read_parquet(_PANEL)
    pead["gvkey"] = pd.to_numeric(pead["gvkey"], errors="coerce").astype("Int64")
    pmap = (pead.dropna(subset=["gvkey", "permno"])
                 .drop_duplicates(subset=["gvkey", "permno"])[["gvkey", "permno"]])
    j = sich.merge(pmap, on="gvkey", how="inner")
    j["permno"] = j["permno"].astype(int)
    return j[["permno", "datadate", "ff12"]].sort_values(["permno", "datadate"])


def _sector_at(permno_sector_panel: pd.DataFrame, permno: int,
                  asof: pd.Timestamp) -> int | None:
    """Find FF12 sector for a permno as-of a date (PIT-correct)."""
    sub = permno_sector_panel[permno_sector_panel["permno"] == permno]
    if sub.empty:
        return None
    sub = sub[sub["datadate"] <= asof]
    if sub.empty:
        # Use earliest available if as-of is before history
        sub = permno_sector_panel[permno_sector_panel["permno"] == permno]
        return int(sub["ff12"].iloc[0])
    return int(sub["ff12"].iloc[-1])


def build_dpead_sector_neutral_returns_pit(
    pead_window_days: int = PEAD_WINDOW_DAYS,
    universe_top_n: int = UNIVERSE_TOP_N,
    decile: float = DECILE,
    ret_path: str = _RET,
    sector_min_names: int = SECTOR_MIN_NAMES,
) -> pd.Series:
    """PIT version of sector-neutral D_PEAD.

    Replaces current-snapshot GICS sectors with point-in-time SIC -> FF12
    mapping (Compustat funda.sich for each fiscal year). Eliminates the
    look-ahead bias documented in P-D1.

    Per [[project-sector-neutral-dpead-real-2x-improvement-2026-05-31]]
    P-D1: 15.4% of gvkeys changed SIC during 2008-2024, so the prior
    current-snapshot version had ~15% lookahead-tainted sector assignments.

    Methodology otherwise identical to build_dpead_sector_neutral_returns:
    within each FF12 bucket with >= sector_min_names, take top decile (long)
    and bottom decile (short) by SUE.
    """
    panel = pd.read_parquet(_PANEL)
    panel["rdq"] = pd.to_datetime(panel["rdq"])
    panel = panel.dropna(subset=["sue", "market_cap_at_q", "rdq", "permno"]).sort_values("rdq")

    ret = pd.read_parquet(ret_path)
    ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()

    sector_panel = _load_permno_to_sector_pit()
    rdq_vals = panel["rdq"].values

    rows: list[tuple] = []
    # Cache permno -> sorted (datadate, ff12) array for fast searchsorted lookup
    sector_by_permno: dict[int, tuple] = {}
    for permno, sub in sector_panel.groupby("permno"):
        sub_sorted = sub.sort_values("datadate")
        sector_by_permno[int(permno)] = (
            sub_sorted["datadate"].values,
            sub_sorted["ff12"].values,
        )

    for t in daily.index:
        lo_i = np.searchsorted(rdq_vals, np.datetime64(t - pd.Timedelta(days=pead_window_days)), "right")
        hi_i = np.searchsorted(rdq_vals, np.datetime64(t), "right")
        if hi_i - lo_i < 20:
            continue
        win = panel.iloc[lo_i:hi_i]
        act = win.groupby("permno").last()
        act = act.nlargest(min(universe_top_n, len(act)), "market_cap_at_q")
        if len(act) < 20:
            continue
        # Attach PIT sector
        act = act.copy()
        t_np = np.datetime64(t)
        sectors = []
        for permno in act.index:
            entry = sector_by_permno.get(int(permno))
            if entry is None:
                sectors.append(None)
                continue
            ddates, ffs = entry
            idx = np.searchsorted(ddates, t_np, "right") - 1
            if idx < 0:
                sectors.append(int(ffs[0]))
            else:
                sectors.append(int(ffs[idx]))
        act["sector"] = sectors
        act = act.dropna(subset=["sector"])
        if len(act) < 50:
            continue

        r = daily.loc[t]
        sector_long = []
        sector_short = []
        for sec, sub in act.groupby("sector"):
            if len(sub) < sector_min_names:
                continue
            sue = sub["sue"]
            top = sue[sue >= sue.quantile(1 - decile)].index
            bot = sue[sue <= sue.quantile(decile)].index
            l = r.reindex(top).dropna()
            s = r.reindex(bot).dropna()
            if len(l) < 1 or len(s) < 1:
                continue
            sector_long.append(float(l.mean()))
            sector_short.append(float(s.mean()))
        if len(sector_long) < 3:
            continue
        rows.append((t, float(np.mean(sector_long) - np.mean(sector_short))))

    return pd.Series(dict(rows)).sort_index().rename("base")


def regenerate_and_save_sector_neutral_pit() -> str:
    s = build_dpead_sector_neutral_returns_pit()
    s.to_frame("base").to_parquet(_SECTOR_NEUTRAL_PIT_OUT)
    return _SECTOR_NEUTRAL_PIT_OUT
