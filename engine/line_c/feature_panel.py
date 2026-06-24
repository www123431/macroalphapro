"""engine/line_c/feature_panel.py — assemble the event-level analysis panel.

For each earnings-call event (permno, rdq, transcript_id) build:
  TARGETS    fwd_ret_21, fwd_ret_63  (cum return from next trading day after rdq)
  CONTROLS   sue, car_3d (abnormal CAR[-1,+1] = the priced surprise),
             mom_12_1, log_size, sector (FF12)
  TEXT       lm_* + finbert_* (merged by transcript_id) + delta_tone (call-over-call)
  + sector-neutralized level variants (text level features demeaned within
    sector×quarter — the #1 guard against a tone-level signal being a sector bet)

Returns + market proxy come from data/line_c/_crsp_daily_ret_2011_2024.parquet
(equal-weight universe mean = market). Output: data/line_c/_event_panel.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine.line_c import wrds_direct

logger = logging.getLogger(__name__)

CACHE = Path("data/line_c")
SUE_PATH = CACHE / "_sue_panel_2011_2024.parquet"
IDX_PATH = CACHE / "_transcripts_index_2011_2024.parquet"
RET_PATH = CACHE / "_crsp_daily_ret_2011_2024.parquet"
SIC_PATH = CACHE / "_permno_sic.parquet"
PANEL_OUT = CACHE / "_event_panel.parquet"

LEVEL_TEXT_FEATS = [
    "lm_net_tone", "lm_pos_prop", "lm_neg_prop", "lm_uncertainty_prop",
    "lm_litigious_prop", "lm_constraining_prop", "lm_modal_prop",
    "numeric_density", "finbert_tone", "finbert_pos", "finbert_neg",
]


def pull_sic(permnos: list[int]) -> pd.DataFrame:
    """permno -> SIC (most recent name record) via crsp.msenames; cached."""
    if SIC_PATH.exists():
        return pd.read_parquet(SIC_PATH)
    conn = wrds_direct.connect("${WRDS_USER_1}")
    try:
        df = pd.read_sql(
            "SELECT DISTINCT ON (permno) permno, siccd FROM crsp.msenames "
            "WHERE permno IN %(p)s ORDER BY permno, nameendt DESC NULLS FIRST",
            conn, params={"p": tuple(sorted(set(int(p) for p in permnos)))},
        )
    finally:
        conn.close()
    df["permno"] = df["permno"].astype(int)
    df["siccd"] = pd.to_numeric(df["siccd"], errors="coerce")
    df.to_parquet(SIC_PATH)
    return df


def ff12(sic: float) -> str:
    """Fama-French 12-industry from SIC (compact)."""
    if pd.isna(sic):
        return "Other"
    s = int(sic)
    def inr(*rng):
        return any(a <= s <= b for a, b in rng)
    if inr((100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)): return "NoDur"
    if inr((2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
           (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)): return "Durbl"
    if inr((2520, 2589), (2600, 2699), (2750, 2769), (2800, 2829), (2840, 2899), (3000, 3099),
           (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749),
           (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)): return "Manuf"
    if inr((1200, 1399), (2900, 2999)): return "Enrgy"
    if inr((2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)): return "Chems"  # approx Hlth/Chems blend
    if inr((3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)): return "BusEq"
    if inr((4800, 4899),): return "Telcm"
    if inr((4900, 4949),): return "Utils"
    if inr((5000, 5999), (7200, 7299), (7600, 7699)): return "Shops"
    if inr((2830, 2831), (3693, 3693), (3840, 3851), (8000, 8099)): return "Hlth"
    if inr((6000, 6999),): return "Money"
    return "Other"


def _wide_returns():
    r = pd.read_parquet(RET_PATH)
    r["date"] = pd.to_datetime(r["date"])
    R = r.pivot_table(index="date", columns="permno", values="ret").sort_index()
    cal = R.index.values
    permno_pos = {int(p): i for i, p in enumerate(R.columns)}
    Rv = R.values.astype(np.float64)
    mkt = np.nanmean(Rv, axis=1)                      # equal-weight universe = market
    # cumulative log return per column (NaN->0 for compounding gaps)
    Rf = np.nan_to_num(Rv, nan=0.0)
    clog = np.cumsum(np.log1p(np.clip(Rf, -0.99, None)), axis=0)
    abn = Rf - mkt[:, None]
    clog_abn = np.cumsum(np.log1p(np.clip(abn, -0.99, None)), axis=0)
    return cal, permno_pos, clog, clog_abn, abn


def _win_ret(clog, a, b, col):
    """compounded return over global-calendar [a, b] inclusive for column col."""
    if a < 0 or b >= clog.shape[0] or col is None or a > b:
        return np.nan
    base = clog[a - 1, col] if a > 0 else 0.0
    return float(np.expm1(clog[b, col] - base))


def build_panel(text_feat_df: pd.DataFrame) -> pd.DataFrame:
    sue = pd.read_parquet(SUE_PATH)
    sue["rdq"] = pd.to_datetime(sue["rdq"])
    idx = pd.read_parquet(IDX_PATH)
    idx["rdq"] = pd.to_datetime(idx["rdq"])

    ev = idx.merge(text_feat_df, on="transcript_id", how="inner")
    ev = ev.merge(sue[["permno", "rdq", "sue", "market_cap_at_q"]], on=["permno", "rdq"], how="inner")
    ev = ev.dropna(subset=["sue", "market_cap_at_q"])
    ev["log_size"] = np.log(ev["market_cap_at_q"].clip(lower=1e-3))
    logger.info("events after merge text+sue: %d", len(ev))

    cal, ppos, clog, clog_abn, abn = _wide_returns()
    cal_ts = pd.DatetimeIndex(cal)

    fwd21, fwd63, car3, mom = [], [], [], []
    for r in ev.itertuples(index=False):
        col = ppos.get(int(r.permno))
        g = int(np.searchsorted(cal, np.datetime64(r.rdq), side="left"))  # rdq position (or insertion)
        entry = int(np.searchsorted(cal, np.datetime64(r.rdq), side="right"))  # first day after rdq
        fwd21.append(_win_ret(clog, entry, entry + 20, col))
        fwd63.append(_win_ret(clog, entry, entry + 62, col))
        # abnormal CAR[-1,+1] = sum abnormal daily returns over [g-1, g+1]
        if col is not None and 1 <= g < clog.shape[0] - 1:
            car3.append(float(np.nansum(abn[g - 1:g + 2, col])))
        else:
            car3.append(np.nan)
        mom.append(_win_ret(clog, entry - 252, entry - 21, col))
    ev["fwd_ret_21"] = fwd21
    ev["fwd_ret_63"] = fwd63
    ev["car_3d"] = car3
    ev["mom_12_1"] = mom

    # sector (FF12)
    sic = pull_sic(ev["permno"].unique().tolist())
    ev = ev.merge(sic, on="permno", how="left")
    ev["sector"] = ev["siccd"].apply(ff12)

    # quarter key + call-over-call delta tone (within permno, ordered by rdq)
    ev["quarter"] = ev["rdq"].dt.to_period("Q").astype(str)
    ev = ev.sort_values(["permno", "rdq"])
    for c in ["finbert_tone", "lm_net_tone"]:
        if c in ev.columns:
            ev[f"d_{c}"] = ev.groupby("permno")[c].diff()

    # sector-neutralize level text features within sector×quarter
    for c in [c for c in LEVEL_TEXT_FEATS if c in ev.columns]:
        grp = ev.groupby(["sector", "quarter"])[c]
        ev[f"{c}_sn"] = (ev[c] - grp.transform("mean"))
    return ev.reset_index(drop=True)


if __name__ == "__main__":
    import sys, warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tf = sys.argv[1] if len(sys.argv) > 1 else None
    if tf is None:
        raise SystemExit("usage: feature_panel.py <text_features.parquet> [out.parquet]")
    out = sys.argv[2] if len(sys.argv) > 2 else str(PANEL_OUT)
    text_feat = pd.read_parquet(tf)
    panel = build_panel(text_feat)
    panel.to_parquet(out)
    print(f"event panel: {panel.shape} -> {out}")
    rdq = pd.to_datetime(panel["rdq"])
    print("rdq", rdq.min().date(), "->", rdq.max().date(), "| events", len(panel),
          "| permnos", panel["permno"].nunique())
    print("\nfwd/control coverage (non-null):")
    for c in ["fwd_ret_21", "fwd_ret_63", "car_3d", "mom_12_1", "sue", "finbert_tone", "lm_net_tone"]:
        if c in panel.columns:
            print(f"  {c:14s} {panel[c].notna().mean()*100:5.1f}%  mean={panel[c].mean():+.4f}")
    print("\nsector dist:"); print(panel["sector"].value_counts().to_string())
