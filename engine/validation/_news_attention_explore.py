"""Exploratory: news ATTENTION-shock signal (orthogonal to the already-RED ESS-level).

Pre-registered PRIMARY hypothesis (declared before running):
  Abnormal news attention (this-month story count vs the firm's own trailing-12m
  baseline) induces temporary price pressure that REVERSES next month, and this
  reversal is stronger / un-arbitraged in SMALL-cap names (limits-to-arb).
  => signal = -abnormal_attention; long low-attention, short high-attention; small-cap.

We report the FULL pre-specified grid (direction x universe x raw-vs-abnormal),
not just the best cell, and count n_trials honestly. Monthly, t -> t+1, EW.
No look-ahead: signal at end of month t uses only data through t.
"""
import numpy as np
import pandas as pd

SENT = "data/cache/_rpna_daily_sentiment.parquet"
RET  = "data/line_c/_crsp_daily_ret_2011_2024.parquet"
MCAP = "data/cache/_crsp_msf_insider_mcap.parquet"

def _ym(s):
    return s.dt.year * 12 + s.dt.month

# ---- monthly news panel: stories + mean ess per permno-month ----
sent = pd.read_parquet(SENT)
sent["ym"] = _ym(sent["d"])
news_m = (sent.groupby(["permno", "ym"])
          .agg(stories=("n_stories", "sum"), ess=("ess", "mean"))
          .reset_index())

# ---- abnormal attention: log stories vs own trailing-12m mean (>=6 prior obs) ----
news_m = news_m.sort_values(["permno", "ym"])
news_m["lstories"] = np.log1p(news_m["stories"])
g = news_m.groupby("permno")["lstories"]
news_m["base"] = g.transform(lambda x: x.shift(1).rolling(12, min_periods=6).mean())
news_m["basestd"] = g.transform(lambda x: x.shift(1).rolling(12, min_periods=6).std())
news_m["abn_att"] = (news_m["lstories"] - news_m["base"]) / news_m["basestd"].replace(0, np.nan)
# sentiment SURPRISE: ess vs own trailing-6m baseline (analogous to revision, not level)
news_m["ess_base"] = news_m.groupby("permno")["ess"].transform(
    lambda x: x.shift(1).rolling(6, min_periods=3).mean())
news_m["d_ess"] = news_m["ess"] - news_m["ess_base"]
news_m = news_m.dropna(subset=["abn_att"])

# ---- monthly returns per permno (compounded from daily) ----
ret = pd.read_parquet(RET)
ret["ym"] = _ym(ret["date"])
mret = (ret.groupby(["permno", "ym"])["ret"]
        .apply(lambda x: (1 + x).prod() - 1).reset_index(name="mret"))

# ---- size (month-end mcap) ----
mc = pd.read_parquet(MCAP)
mc["ym"] = _ym(mc["date"])
mc = mc[["permno", "ym", "mcap"]]

# ---- assemble: signal at t, predict return at t+1 ----
panel = news_m.merge(mc, on=["permno", "ym"], how="inner")
panel["ym1"] = panel["ym"] + 1
fwd = mret.rename(columns={"ym": "ym1", "mret": "fret"})
panel = panel.merge(fwd[["permno", "ym1", "fret"]], on=["permno", "ym1"], how="inner")
panel = panel.dropna(subset=["fret", "abn_att", "mcap", "stories"])
print("panel rows:", len(panel), "months:", panel["ym"].nunique(),
      "permnos:", panel["permno"].nunique())

def ls_series(df, sigcol, ascending):
    """Decile L/S monthly EW return series. ascending=True => long LOW signal."""
    out = {}
    for ym, gdf in df.groupby("ym1"):
        gg = gdf.dropna(subset=[sigcol]).copy()
        if len(gg) < 30:
            continue
        gg["dec"] = pd.qcut(gg[sigcol].rank(method="first"), 10, labels=False)
        lo = gg.loc[gg["dec"] == 0, "fret"].mean()
        hi = gg.loc[gg["dec"] == 9, "fret"].mean()
        out[ym] = (lo - hi) if ascending else (hi - lo)
    return pd.Series(out).sort_index()

def stats(s):
    s = s.dropna()
    if len(s) < 12:
        return dict(n=len(s), ann=np.nan, t=np.nan, sharpe=np.nan)
    ann = s.mean() * 12
    sharpe = (s.mean() / s.std()) * np.sqrt(12)
    t = s.mean() / (s.std() / np.sqrt(len(s)))
    return dict(n=len(s), ann=round(ann * 100, 2), t=round(t, 2), sharpe=round(sharpe, 2))

# size terciles (within each cross-section) for limits-to-arb conditioning
panel["sz"] = panel.groupby("ym1")["mcap"].transform(
    lambda x: pd.qcut(x.rank(method="first"), 3, labels=["small", "mid", "large"]))

universes = {"ALL": panel,
             "small": panel[panel["sz"] == "small"],
             "large": panel[panel["sz"] == "large"]}
print("\n=== GRID (decile L/S, EW, monthly t->t+1) ===")
print(f"{'signal':14s}{'univ':7s}{'dir':10s}{'n':>5s}{'ann%':>8s}{'t':>7s}{'sharpe':>8s}")
results = {}
for sigcol, signame in [("abn_att", "abn_att"), ("lstories", "raw_att"),
                        ("d_ess", "ess_chg"), ("ess", "ess_lvl")]:
    for uname, udf in universes.items():
        for asc, dname in [(True, "rev(longLow)"), (False, "mom(longHigh)")]:
            st = stats(ls_series(udf, sigcol, asc))
            results[(signame, uname, dname)] = st
            print(f"{signame:14s}{uname:7s}{dname:10s}{st['n']:>5}{str(st['ann']):>8}{str(st['t']):>7}{str(st['sharpe']):>8}")

print("\nn_trials in this grid:", len(results))
prim = results[("abn_att", "small", "rev(longLow)")]
print("PRIMARY (abn_att/small/reversal):", prim)

# ---- formal gate on the pre-registered PRIMARY -> records to campaign ledger ----
def _ym_to_ts(ym):
    yr = (ym - 1) // 12
    mo = (ym - 1) % 12 + 1
    return pd.Timestamp(yr, mo, 1) + pd.offsets.MonthEnd(0)

from engine.research.pipeline import run_gate
import json

def _to_ts(s):
    s = s.copy()
    s.index = [_ym_to_ts(int(y)) for y in s.index]
    return s.sort_index()

# the flicker: sentiment-CHANGE reversal (best of grid). honest n_trials = full grid (24).
for uname in ["small", "ALL"]:
    s = _to_ts(ls_series(universes[uname], "d_ess", True))
    v = run_gate(s, name=f"news_ess_change_reversal_{uname}",
                 mechanism="news_sentiment_change", n_trials=24,
                 log=(uname == "small"))   # log the strongest as the campaign trial
    print(f"\n=== GATE ess_chg/{uname}/reversal (n_trials=24) ===")
    print(json.dumps(v, indent=2, ensure_ascii=False))
