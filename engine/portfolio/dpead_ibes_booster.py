"""engine/portfolio/dpead_ibes_booster.py — Strategy B per loop-robustness
roadmap [[feedback-loop-is-robustness-doctrine-2026-05-31]].

ENHANCED D_PEAD with IBES analyst revisions as confirming/dis-confirming
signal. Academic basis:
  - Stickel 1991 (JF) "Effect of Earnings Information on Stock Price"
  - Chordia-Shivakumar 2006 (JFE) "Earnings and price momentum"
  - Lo-Patel 2014: revisions + SUE combine for stronger signal

LOGIC:
  For each rdq event in PEAD panel:
    1. Look at IBES statsum_epsus revisions in [-60 days, -1 day] window
       before rdq for fpi=1 (current quarter forecast)
    2. revision_delta = (meanest_at_rdq - meanest_60d_prior) / |meanest_60d_prior|
    3. z-score revision_delta cross-sectionally each rdq
    4. Combined signal = β × SUE_z + (1-β) × revision_delta_z
       Default β = 0.6 (SUE weighted slightly higher; tunable)
  Long top-decile of combined signal cross-sectionally each day
  Short bottom-decile
  Equal weight within leg

OUTPUT: daily L/S return series, persisted to
data/cache/_dpead_ibes_booster.parquet
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
_OUT = REPO_ROOT / "data" / "cache" / "_dpead_ibes_booster_daily.parquet"

PEAD_WINDOW_DAYS = 90
UNIVERSE_TOP_N = 1500
DECILE = 0.10
MIN_LEG_NAMES = 3
REVISION_LOOKBACK_DAYS = 60
SUE_WEIGHT = 0.6     # combined = SUE_WEIGHT × SUE_z + (1-SUE_WEIGHT) × rev_z


def _build_revision_panel(ibes: pd.DataFrame,
                              pead: pd.DataFrame,
                              lookback_days: int = REVISION_LOOKBACK_DAYS) -> pd.DataFrame:
    """For each (permno, rdq) in PEAD, compute revision_delta from IBES.

    Returns DataFrame with columns: permno, rdq, revision_delta.
    """
    ibes = ibes.copy()
    ibes["statpers"] = pd.to_datetime(ibes["statpers"])
    ibes["meanest"] = pd.to_numeric(ibes["meanest"], errors="coerce")
    ibes = ibes[ibes["fpi"].astype(str) == "1"]   # current period only
    ibes = ibes.dropna(subset=["meanest"])

    pead = pead.dropna(subset=["ticker", "permno", "rdq"]).copy()
    pead["rdq"] = pd.to_datetime(pead["rdq"])

    out_rows = []
    # Group ibes by ticker for fast lookup
    ibes_by_ticker = {tk: sub.sort_values("statpers")
                       for tk, sub in ibes.groupby("ticker")}
    for _, row in pead.iterrows():
        ticker = row["ticker"]
        rdq = row["rdq"]
        sub = ibes_by_ticker.get(ticker)
        if sub is None:
            continue
        # statpers in [rdq - lookback, rdq - 1]
        prior_window = sub[
            (sub["statpers"] >= rdq - pd.Timedelta(days=lookback_days))
            & (sub["statpers"] < rdq)
        ]
        if len(prior_window) < 2:
            continue
        # First in window vs last in window
        est_start = prior_window["meanest"].iloc[0]
        est_end = prior_window["meanest"].iloc[-1]
        if pd.isna(est_start) or pd.isna(est_end) or abs(est_start) < 1e-6:
            continue
        revision_delta = (est_end - est_start) / abs(est_start)
        out_rows.append({
            "permno":         int(row["permno"]),
            "rdq":            rdq,
            "revision_delta": float(revision_delta),
        })
    return pd.DataFrame(out_rows)


def build_dpead_ibes_booster_returns(
    sue_weight: float = SUE_WEIGHT,
    pead_window_days: int = PEAD_WINDOW_DAYS,
    universe_top_n: int = UNIVERSE_TOP_N,
    decile: float = DECILE,
) -> pd.Series:
    """Build daily L/S return series of D_PEAD enhanced by IBES revisions.

    Same universe + drift-window logic as build_dpead_recon_returns; only
    the cross-sectional ranking signal differs: instead of pure SUE,
    use combined SUE_z + revision_z.
    """
    panel = pd.read_parquet(_PANEL)
    panel["rdq"] = pd.to_datetime(panel["rdq"])
    panel = panel.dropna(subset=["sue", "market_cap_at_q", "rdq", "permno"]).sort_values("rdq")

    ret = pd.read_parquet(_RET)
    ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()

    ibes = pd.read_parquet(_IBES)
    revisions = _build_revision_panel(ibes, panel)
    revisions["rdq"] = pd.to_datetime(revisions["rdq"])
    # Dedupe (permno, rdq) keys to ensure rev_map.get returns scalar
    revisions = revisions.drop_duplicates(subset=["permno", "rdq"], keep="last")
    rev_map = revisions.set_index(["permno", "rdq"])["revision_delta"].to_dict()

    rdq_vals = panel["rdq"].values
    rows = []
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

        # Lookup revision_delta for each (permno, rdq)
        revisions_for_act = []
        for permno, row in act.iterrows():
            rev = rev_map.get((int(permno), row["rdq"]))
            revisions_for_act.append(rev if rev is not None else np.nan)
        act = act.copy()
        act["revision_delta"] = revisions_for_act

        # Z-score SUE and revision_delta cross-sectionally
        act["sue"] = pd.to_numeric(act["sue"], errors="coerce")
        sue_z = (act["sue"] - act["sue"].mean()) / act["sue"].std()
        rev_clean = act["revision_delta"].dropna()
        if len(rev_clean) < 30:
            # Fallback to pure SUE if too few revisions
            combined = sue_z
        else:
            rev_mean = rev_clean.mean()
            rev_std = rev_clean.std()
            rev_z = (act["revision_delta"] - rev_mean) / max(rev_std, 1e-9)
            rev_z = rev_z.fillna(0.0)   # missing → neutral
            combined = sue_weight * sue_z + (1 - sue_weight) * rev_z

        if combined.dropna().empty:
            continue
        threshold_top = combined.quantile(1 - decile)
        threshold_bot = combined.quantile(decile)
        long_permnos = combined[combined >= threshold_top].index
        short_permnos = combined[combined <= threshold_bot].index

        r = daily.loc[t]
        long_r = r.reindex(long_permnos).dropna()
        short_r = r.reindex(short_permnos).dropna()
        if len(long_r) < MIN_LEG_NAMES or len(short_r) < MIN_LEG_NAMES:
            continue
        rows.append((t, float(long_r.mean() - short_r.mean())))

    return pd.Series(dict(rows)).sort_index().rename("dpead_ibes_booster")


def regenerate_and_save() -> str:
    s = build_dpead_ibes_booster_returns()
    s.to_frame("base").to_parquet(_OUT)
    return str(_OUT)
