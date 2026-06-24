"""engine/portfolio/dpead_abnormal_sue.py — Path B abnormal-SUE
strategy: SUE relative to peer-group expectation.

Academic basis:
  Abarbanell & Bushee 1997 "Fundamental Analysis, Future Earnings,
    and Stock Prices" (Journal of Accounting Research): earnings
    surprise INTERPRETED relative to cross-sectional + time-series
    peer benchmarks is more predictive than raw SUE
  Frankel & Lee 1998 "Accounting valuation, market expectation, and
    cross-sectional stock returns" (Journal of Accounting & Economics):
    peer-relative valuation drives returns
  Chordia & Shivakumar 2006 "Earnings and price momentum" (Journal of
    Financial Economics): earnings vs price momentum interaction

Hypothesis: SUE_abnormal = SUE - mean(SUE | peer_group) extracts the
FIRM-SPECIFIC surprise component (vs the sector-cohort effect already
captured by PIT SN within-sector ranking).

  peer_group(i, t) = same FF12 sector
                   × same size quintile (within month, within universe)
                   × same announcement quarter

Pre-committed falsification criteria (per Path B doctrine 2026-05-31):
  C1. Sharpe(abnormal_signal) > Sharpe(PIT_SN_parent) in shared window
  C2. Cosine(abnormal_returns, PIT_SN_returns) < 0.7 (orthogonality)
  C3. Combined signal Sharpe > max(abnormal, PIT_SN) — additive stack
  Failing ANY of C1/C2/C3 → STRATEGY REJECTED per IBES-combo lesson
  (project_combo_test_axes_not_independent_2026-05-31).
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
_SICH = REPO_ROOT / "data" / "cache" / "_compustat_funda_sich_pit.parquet"
_OUT_MONTHLY = REPO_ROOT / "data" / "cache" / "_dpead_abnormal_sue_monthly.parquet"

PEAD_WINDOW_DAYS = 90
UNIVERSE_TOP_N = 1500
DECILE = 0.10
N_SIZE_QUINTILES = 5
MIN_PEER_GROUP_SIZE = 5    # require ≥5 firms in peer group to compute reliable mean


def _sic_to_ff12(sic: int) -> int:
    """Standard Ken French FF12 mapping — same as in dpead_pit_sn_ibes_combo."""
    if not sic or sic <= 0:
        return 12
    if (100 <= sic <= 999) or (2000 <= sic <= 2399) or (2700 <= sic <= 2749) \
       or (2770 <= sic <= 2799) or (3100 <= sic <= 3199) \
       or (3940 <= sic <= 3989):
        return 1
    if (2500 <= sic <= 2519) or (2590 <= sic <= 2599) \
       or (3630 <= sic <= 3659) or (3710 <= sic <= 3711) \
       or (3714 <= sic <= 3714) or (3716 <= sic <= 3716) \
       or (3750 <= sic <= 3751) or (3792 <= sic <= 3792) \
       or (3900 <= sic <= 3939) or (3990 <= sic <= 3999):
        return 2
    if (2520 <= sic <= 2589) or (2600 <= sic <= 2699) \
       or (2750 <= sic <= 2769) or (3000 <= sic <= 3099) \
       or (3200 <= sic <= 3569) or (3580 <= sic <= 3629) \
       or (3700 <= sic <= 3709) or (3712 <= sic <= 3713) \
       or (3715 <= sic <= 3715) or (3717 <= sic <= 3749) \
       or (3752 <= sic <= 3791) or (3793 <= sic <= 3799) \
       or (3830 <= sic <= 3839) or (3860 <= sic <= 3899):
        return 3
    if (1200 <= sic <= 1399) or (2900 <= sic <= 2999):
        return 4
    if (2800 <= sic <= 2829) or (2840 <= sic <= 2899):
        return 5
    if (3570 <= sic <= 3579) or (3660 <= sic <= 3692) \
       or (3694 <= sic <= 3699) or (3810 <= sic <= 3829) \
       or (7370 <= sic <= 7379):
        return 6
    if 4800 <= sic <= 4899:
        return 7
    if 4900 <= sic <= 4949:
        return 8
    if (5000 <= sic <= 5999) or (7200 <= sic <= 7299) \
       or (7600 <= sic <= 7699):
        return 9
    if (2830 <= sic <= 2839) or (3693 <= sic <= 3693) \
       or (3840 <= sic <= 3859) or (8000 <= sic <= 8099):
        return 10
    if 6000 <= sic <= 6999:
        return 11
    return 12


def _build_pit_sector_lookup() -> dict:
    """Same as dpead_pit_sn_ibes_combo._build_pit_sector_lookup —
    returns {permno: (sorted dates array, ff12 array)} for PIT lookup."""
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
        out[int(permno)] = (sub["datadate"].values, sub["ff12"].values)
    return out


def _annotate_panel_with_sector_and_size(
    panel: pd.DataFrame, sector_lookup: dict,
) -> pd.DataFrame:
    """Add ff12 sector + size_quintile + announce_quarter columns."""
    panel = panel.copy()
    panel["sue"] = pd.to_numeric(panel["sue"], errors="coerce")
    panel["market_cap_at_q"] = pd.to_numeric(
        panel["market_cap_at_q"], errors="coerce",
    )
    panel = panel.dropna(subset=["sue", "market_cap_at_q", "rdq", "permno"])
    panel["rdq"] = pd.to_datetime(panel["rdq"])

    # FF12 sector via PIT lookup
    sectors = []
    for _, row in panel.iterrows():
        entry = sector_lookup.get(int(row["permno"]))
        if entry is None:
            sectors.append(12)  # Other
            continue
        ddates, ffs = entry
        idx = np.searchsorted(ddates, np.datetime64(row["rdq"]), "right") - 1
        sectors.append(int(ffs[max(0, idx)]))
    panel["ff12"] = sectors

    # Size quintile WITHIN MONTH (so peer groups are time-coherent)
    panel["announce_month"] = panel["rdq"].dt.to_period("M")
    panel["size_quintile"] = panel.groupby("announce_month")["market_cap_at_q"].transform(
        lambda x: pd.qcut(x, q=N_SIZE_QUINTILES, labels=False, duplicates="drop")
    )
    panel["size_quintile"] = panel["size_quintile"].fillna(-1).astype(int)
    panel = panel[panel["size_quintile"] >= 0]
    return panel


def _compute_abnormal_sue(panel: pd.DataFrame) -> pd.DataFrame:
    """For each (sector, size_quintile, announce_month) bucket compute
    peer_mean_SUE; output SUE_abnormal = SUE - peer_mean.

    Peer groups with fewer than MIN_PEER_GROUP_SIZE firms are flagged
    with sue_abnormal = NaN (insufficient peer data — fall back to raw
    SUE downstream if desired)."""
    panel = panel.copy()
    group_keys = ["ff12", "size_quintile", "announce_month"]
    peer_stats = panel.groupby(group_keys)["sue"].agg(["mean", "count"]).reset_index()
    peer_stats = peer_stats.rename(columns={"mean": "peer_mean_sue",
                                              "count": "peer_size"})
    panel = panel.merge(peer_stats, on=group_keys, how="left")
    # Mask sparse peer groups
    panel["sue_abnormal"] = np.where(
        panel["peer_size"] >= MIN_PEER_GROUP_SIZE,
        panel["sue"] - panel["peer_mean_sue"],
        np.nan,
    )
    return panel


def build_abnormal_sue_returns(
    pead_window_days: int = PEAD_WINDOW_DAYS,
    universe_top_n: int = UNIVERSE_TOP_N,
    decile: float = DECILE,
) -> pd.Series:
    """Build monthly returns of long-top-decile / short-bottom-decile
    SUE_abnormal portfolio.

    Symmetric with engine.portfolio.dpead_pit_sn_ibes_combo for fair
    head-to-head comparison against PIT SN.
    """
    sector_lookup = _build_pit_sector_lookup()
    panel = pd.read_parquet(_PANEL)
    panel = _annotate_panel_with_sector_and_size(panel, sector_lookup)
    panel = _compute_abnormal_sue(panel)

    ret = pd.read_parquet(_RET)
    ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()

    panel = panel.sort_values("rdq")
    rdq_vals = panel["rdq"].values

    rows = []
    for t in daily.index:
        lo_i = np.searchsorted(rdq_vals,
                                np.datetime64(t - pd.Timedelta(days=pead_window_days)),
                                "right")
        hi_i = np.searchsorted(rdq_vals, np.datetime64(t), "right")
        if hi_i - lo_i < 20:
            continue
        win = panel.iloc[lo_i:hi_i]
        act = win.groupby("permno").last()
        act = act.nlargest(min(universe_top_n, len(act)), "market_cap_at_q")
        act = act.dropna(subset=["sue_abnormal"])
        if len(act) < 50:
            continue

        # Rank by SUE_abnormal across the universe (NOT within sector —
        # we want to test if peer-adjustment alone adds value)
        thr_top = act["sue_abnormal"].quantile(1 - decile)
        thr_bot = act["sue_abnormal"].quantile(decile)
        long_p = act[act["sue_abnormal"] >= thr_top].index
        short_p = act[act["sue_abnormal"] <= thr_bot].index

        r = daily.loc[t]
        l = r.reindex(long_p).dropna()
        s = r.reindex(short_p).dropna()
        if len(l) < 5 or len(s) < 5:
            continue
        rows.append((t, float(l.mean() - s.mean())))

    if not rows:
        raise RuntimeError("no return rows produced — check data inputs")
    daily_returns = pd.Series(dict(rows)).sort_index().rename("abnormal_sue_daily")
    monthly = ((1 + daily_returns.clip(-0.2, 0.2)).resample("ME").prod() - 1)
    return monthly.rename("abnormal_sue")


def regenerate_and_save() -> str:
    monthly = build_abnormal_sue_returns()
    _OUT_MONTHLY.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_frame("abnormal_sue").to_parquet(_OUT_MONTHLY)
    return str(_OUT_MONTHLY)


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    out = regenerate_and_save()
    print(f"saved: {out}")
    df = pd.read_parquet(out)
    s = df.iloc[:, 0]
    sharpe = (s.mean() * 12) / (s.std() * (12 ** 0.5))
    print(f"n_months: {len(s)}")
    print(f"gross Sharpe (annualized): {sharpe:+.3f}")
    print(f"ann return: {s.mean()*12:+.2%}  ann vol: {s.std()*(12**0.5):.2%}")
