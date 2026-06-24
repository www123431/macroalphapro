"""engine/portfolio/dpead_pit_sn_ibes_combo.py — combines PIT sector
neutralization (FF12) with IBES revisions booster signal.

Per [[project-ibes-booster-real-vs-phack-2026-05-31]] high-leverage
experiment: do the two improvement axes stack independently?

  PIT SN improves UNIVERSE (within-sector ranking, less noise)
  IBES improves SIGNAL (SUE + revision confirmation)

If they tap different sources of alpha → multiplicative stacking
If they tap same noise reduction → no incremental gain

Methodology:
  Within each PIT FF12 sector with >= 8 firms:
    1. Compute combined_score = sue_weight × SUE_z + (1-sue_weight) × revision_z
       (z-score WITHIN sector cross-section to remove sector-level mean)
    2. Long top decile of combined_score within sector
    3. Short bottom decile within sector
  Equal-weight across sectors (sector-neutral by construction)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
_PANEL = REPO_ROOT / "data" / "cache" / "_pead_ts_panel_2014_2023.parquet"
_RET = REPO_ROOT / "data" / "cache" / "crsp_hist_daily_ret.parquet"
_IBES = REPO_ROOT / "data" / "cache" / "_ibes_statsum_for_pead.parquet"
_SICH = REPO_ROOT / "data" / "cache" / "_compustat_funda_sich_pit.parquet"
_OUT_DAILY = REPO_ROOT / "data" / "cache" / "_dpead_pit_sn_ibes_combo_daily.parquet"
_OUT_MONTHLY = REPO_ROOT / "data" / "cache" / "_dpead_pit_sn_ibes_combo_monthly.parquet"

PEAD_WINDOW_DAYS = 90
UNIVERSE_TOP_N = 1500
DECILE = 0.10
SECTOR_MIN_NAMES = 8
REVISION_LOOKBACK_DAYS = 60
SUE_WEIGHT = 0.6


def _sic_to_ff12(sic: int) -> int:
    """Standard Ken French FF12 mapping (same as dpead_recon)."""
    if not sic or sic <= 0: return 12
    if (100 <= sic <= 999) or (2000 <= sic <= 2399) or (2700 <= sic <= 2749) \
       or (2770 <= sic <= 2799) or (3100 <= sic <= 3199) \
       or (3940 <= sic <= 3989): return 1
    if (2500 <= sic <= 2519) or (2590 <= sic <= 2599) \
       or (3630 <= sic <= 3659) or (3710 <= sic <= 3711) \
       or (3714 <= sic <= 3714) or (3716 <= sic <= 3716) \
       or (3750 <= sic <= 3751) or (3792 <= sic <= 3792) \
       or (3900 <= sic <= 3939) or (3990 <= sic <= 3999): return 2
    if (2520 <= sic <= 2589) or (2600 <= sic <= 2699) \
       or (2750 <= sic <= 2769) or (3000 <= sic <= 3099) \
       or (3200 <= sic <= 3569) or (3580 <= sic <= 3629) \
       or (3700 <= sic <= 3709) or (3712 <= sic <= 3713) \
       or (3715 <= sic <= 3715) or (3717 <= sic <= 3749) \
       or (3752 <= sic <= 3791) or (3793 <= sic <= 3799) \
       or (3830 <= sic <= 3839) or (3860 <= sic <= 3899): return 3
    if (1200 <= sic <= 1399) or (2900 <= sic <= 2999): return 4
    if (2800 <= sic <= 2829) or (2840 <= sic <= 2899): return 5
    if (3570 <= sic <= 3579) or (3660 <= sic <= 3692) \
       or (3694 <= sic <= 3699) or (3810 <= sic <= 3829) \
       or (7370 <= sic <= 7379): return 6
    if 4800 <= sic <= 4899: return 7
    if 4900 <= sic <= 4949: return 8
    if (5000 <= sic <= 5999) or (7200 <= sic <= 7299) \
       or (7600 <= sic <= 7699): return 9
    if (2830 <= sic <= 2839) or (3693 <= sic <= 3693) \
       or (3840 <= sic <= 3859) or (8000 <= sic <= 8099): return 10
    if 6000 <= sic <= 6999: return 11
    return 12


def _build_pit_sector_lookup() -> dict:
    """{permno: [(datadate, ff12), ...]} sorted ascending."""
    sich = pd.read_parquet(_SICH)
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
    j = j.sort_values(["permno", "datadate"])
    out: dict = {}
    for permno, sub in j.groupby("permno"):
        out[int(permno)] = (
            sub["datadate"].values, sub["ff12"].values,
        )
    return out


def _build_revision_panel(ibes: pd.DataFrame, pead: pd.DataFrame,
                              lookback_days: int = REVISION_LOOKBACK_DAYS) -> dict:
    """{(permno, rdq): revision_delta} for fast lookup."""
    ibes = ibes.copy()
    ibes["statpers"] = pd.to_datetime(ibes["statpers"])
    ibes["meanest"] = pd.to_numeric(ibes["meanest"], errors="coerce")
    ibes = ibes[ibes["fpi"].astype(str) == "1"].dropna(subset=["meanest"])

    pead = pead.dropna(subset=["ticker", "permno", "rdq"]).copy()
    pead["rdq"] = pd.to_datetime(pead["rdq"])
    ibes_by_ticker = {tk: sub.sort_values("statpers")
                       for tk, sub in ibes.groupby("ticker")}

    rev_map: dict = {}
    for _, row in pead.iterrows():
        sub = ibes_by_ticker.get(row["ticker"])
        if sub is None: continue
        rdq = row["rdq"]
        win = sub[(sub["statpers"] >= rdq - pd.Timedelta(days=lookback_days))
                  & (sub["statpers"] < rdq)]
        if len(win) < 2: continue
        e0, e1 = win["meanest"].iloc[0], win["meanest"].iloc[-1]
        if pd.isna(e0) or pd.isna(e1) or abs(e0) < 1e-6: continue
        rev_map[(int(row["permno"]), rdq)] = (e1 - e0) / abs(e0)
    return rev_map


def build_combo_returns(
    sue_weight: float = SUE_WEIGHT,
    pead_window_days: int = PEAD_WINDOW_DAYS,
    universe_top_n: int = UNIVERSE_TOP_N,
    decile: float = DECILE,
    sector_min_names: int = SECTOR_MIN_NAMES,
) -> pd.Series:
    """Build PIT SN + IBES booster combined daily return series."""
    panel = pd.read_parquet(_PANEL)
    panel["rdq"] = pd.to_datetime(panel["rdq"])
    panel = panel.dropna(subset=["sue", "market_cap_at_q", "rdq", "permno"]).sort_values("rdq")

    ret = pd.read_parquet(_RET)
    ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()

    ibes = pd.read_parquet(_IBES)
    rev_map = _build_revision_panel(ibes, panel)
    sector_lookup = _build_pit_sector_lookup()

    rdq_vals = panel["rdq"].values
    rows = []

    for t in daily.index:
        lo_i = np.searchsorted(rdq_vals, np.datetime64(t - pd.Timedelta(days=pead_window_days)), "right")
        hi_i = np.searchsorted(rdq_vals, np.datetime64(t), "right")
        if hi_i - lo_i < 20: continue
        win = panel.iloc[lo_i:hi_i]
        act = win.groupby("permno").last()
        act = act.nlargest(min(universe_top_n, len(act)), "market_cap_at_q")
        if len(act) < 20: continue

        # Add PIT sector + revision_delta + numeric coercion
        act = act.copy()
        act["sue"] = pd.to_numeric(act["sue"], errors="coerce")
        t_np = np.datetime64(t)
        sectors = []
        revisions = []
        for permno, row in act.iterrows():
            # Sector
            entry = sector_lookup.get(int(permno))
            if entry is None:
                sectors.append(None)
            else:
                ddates, ffs = entry
                idx = np.searchsorted(ddates, t_np, "right") - 1
                sectors.append(int(ffs[max(0, idx)]))
            # Revision
            rev = rev_map.get((int(permno), row["rdq"]))
            revisions.append(rev if rev is not None else np.nan)
        act["sector"] = sectors
        act["revision_delta"] = revisions
        act = act.dropna(subset=["sector", "sue"])
        if len(act) < 50: continue

        r = daily.loc[t]
        sector_long = []
        sector_short = []

        # Within each FF12 sector, build combined signal + decile sort
        for sec, sub in act.groupby("sector"):
            if len(sub) < sector_min_names: continue
            sue_z = (sub["sue"] - sub["sue"].mean()) / max(sub["sue"].std(), 1e-9)
            rev_clean = sub["revision_delta"].dropna()
            if len(rev_clean) >= max(5, sector_min_names // 2):
                rev_z = (sub["revision_delta"] - rev_clean.mean()) / max(rev_clean.std(), 1e-9)
                rev_z = rev_z.fillna(0.0)
                combined = sue_weight * sue_z + (1 - sue_weight) * rev_z
            else:
                combined = sue_z   # fall back to pure SUE in sector with few revisions
            if combined.dropna().empty: continue
            top_thr = combined.quantile(1 - decile)
            bot_thr = combined.quantile(decile)
            long_p = combined[combined >= top_thr].index
            short_p = combined[combined <= bot_thr].index
            l = r.reindex(long_p).dropna()
            s = r.reindex(short_p).dropna()
            if len(l) < 1 or len(s) < 1: continue
            sector_long.append(float(l.mean()))
            sector_short.append(float(s.mean()))
        if len(sector_long) < 3: continue
        rows.append((t, float(np.mean(sector_long) - np.mean(sector_short))))

    return pd.Series(dict(rows)).sort_index().rename("base")


def regenerate_and_save() -> tuple[str, str]:
    s_daily = build_combo_returns()
    s_daily.to_frame("base").to_parquet(_OUT_DAILY)
    monthly = ((1 + s_daily.clip(-0.2, 0.2)).resample("ME").prod() - 1).rename("combo")
    monthly.to_frame("combo").to_parquet(_OUT_MONTHLY)
    return str(_OUT_DAILY), str(_OUT_MONTHLY)
