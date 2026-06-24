"""engine/line_c/gtrends_attention.py — Google-Trends investor-attention (DEG ASVI).

Da-Engelberg-Gao (2011, JF) "In Search of Attention": investor attention proxied by
Google search volume of the TICKER SYMBOL (ticker = investor search, unambiguous,
not consumer interest). Abnormal SVI predicts a short-run price increase then
reversal. A NON-earnings, retail-attention mechanism -> potential low correlation
with the D_PEAD / revision earnings-information family (the kind of DIFFERENT
mechanism a real 3rd alpha needs).

DISCIPLINE: free but the decisive test is the TURNOVER WALL (attention/reversal
effects are high-turnover; this project has killed every reversal on cost). So we
report cost-aware NET Sharpe + turnover + deflated Sharpe + correlation vs D_PEAD,
not just gross. Monthly resolution (1 Trends query / ticker over the full window)
keeps turnover lower AND the pull tractable; weekly (stitched) is a refinement only
if monthly shows something.

Pull is THROTTLED + 429-backoff + RESUMABLE (Trends rate-limits hard).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE = Path("data/line_c")
SVI_PATH = CACHE / "_gtrends_svi.parquet"
UNIV = CACHE / "_universe_top1500_2011_2024.parquet"
SUE_PATH = CACHE / "_sue_panel_2011_2024.parquet"
RET_PATH = CACHE / "_crsp_daily_ret_2011_2024.parquet"
TIMEFRAME = "2011-01-01 2024-06-30"


def _top_n_tickers(n: int) -> pd.DataFrame:
    """Top-N by latest mcap from the SUE panel; returns (permno, ticker)."""
    sue = pd.read_parquet(SUE_PATH)
    sue["rdq"] = pd.to_datetime(sue["rdq"])
    latest = sue.sort_values("rdq").groupby("permno").tail(1)
    top = latest.sort_values("market_cap_at_q", ascending=False).head(n)
    return top[["permno", "ticker"]].dropna().drop_duplicates("ticker").reset_index(drop=True)


def pull_svi(tickers: list[str], *, sleep=2.5, max_backoff=120) -> pd.DataFrame:
    """Monthly SVI per ticker over TIMEFRAME; throttled + 429-backoff + resumable."""
    from pytrends.request import TrendReq
    done: set[str] = set()
    if SVI_PATH.exists():
        done = set(pd.read_parquet(SVI_PATH, columns=["ticker"])["ticker"].unique())
    todo = [t for t in tickers if t not in done]
    logger.info("SVI pull: %d total, %d cached, %d to fetch", len(tickers), len(done), len(todo))
    pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
    rows = []
    backoff = sleep
    for i, tk in enumerate(todo, 1):
        for attempt in range(5):
            try:
                pt.build_payload([tk], timeframe=TIMEFRAME)
                df = pt.interest_over_time()
                if not df.empty and tk in df.columns:
                    sub = df[[tk]].reset_index().rename(columns={tk: "svi", "date": "date"})
                    sub["ticker"] = tk
                    rows.append(sub[["ticker", "date", "svi"]])
                backoff = sleep
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    backoff = min(backoff * 2, max_backoff)
                    logger.warning("429 on %s, backoff %.0fs", tk, backoff)
                    time.sleep(backoff)
                else:
                    logger.warning("skip %s: %s", tk, str(e)[:60])
                    break
        time.sleep(sleep)
        if i % 25 == 0 or i == len(todo):
            if rows:
                _append_svi(rows); rows = []
            logger.info("  SVI %d/%d fetched", i, len(todo))
    if rows:
        _append_svi(rows)
    return pd.read_parquet(SVI_PATH)


def _append_svi(rows):
    new = pd.concat(rows, ignore_index=True)
    if SVI_PATH.exists():
        new = pd.concat([pd.read_parquet(SVI_PATH), new], ignore_index=True).drop_duplicates(["ticker", "date"])
    new.to_parquet(SVI_PATH)


def build_asvi_panel() -> pd.DataFrame:
    """ASVI = log(svi+1) - log(rolling 8-month prior median +1), per ticker -> permno + next-month ret."""
    svi = pd.read_parquet(SVI_PATH)
    svi["date"] = pd.to_datetime(svi["date"])
    svi = svi.sort_values(["ticker", "date"])
    svi["med8"] = svi.groupby("ticker")["svi"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=4).median())
    svi["asvi"] = np.log(svi["svi"] + 1) - np.log(svi["med8"] + 1)
    svi = svi.dropna(subset=["asvi"])

    univ = pd.read_parquet(UNIV)[["permno", "ticker"]].drop_duplicates("ticker")
    svi = svi.merge(univ, on="ticker", how="inner")
    svi["month"] = svi["date"].dt.to_period("M").dt.to_timestamp("M")

    # monthly returns from daily
    r = pd.read_parquet(RET_PATH); r["date"] = pd.to_datetime(r["date"])
    daily = r.pivot_table(index="date", columns="permno", values="ret")
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    mret = mret.where(daily.resample("ME").count() > 5)
    # next-month return aligned to signal month
    nxt = mret.shift(-1).stack().rename("fwd_ret_1m").reset_index()
    nxt.columns = ["month", "permno", "fwd_ret_1m"]
    panel = svi.merge(nxt, on=["permno", "month"], how="inner").dropna(subset=["fwd_ret_1m"])
    return panel[["permno", "ticker", "month", "svi", "asvi", "fwd_ret_1m"]]


def evaluate(panel: pd.DataFrame, n_trials=24):
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    # cross-sectional decile L/S each month, BOTH directions reported
    rows_hi, rows_lo, ent, prevL = [], [], [], set()
    for m, g in panel.groupby("month"):
        if len(g) < 50:
            continue
        g = g.assign(d=pd.qcut(g["asvi"].rank(method="first"), 10, labels=False))
        hi = g.loc[g["d"] == 9, "fwd_ret_1m"].mean()       # high attention
        lo = g.loc[g["d"] == 0, "fwd_ret_1m"].mean()       # low attention
        rows_hi.append((m, hi - lo))
        L = set(g.loc[g["d"] == 9, "permno"])
        ent.append(len(L - prevL) / max(len(L), 1)); prevL = L
    ls = pd.Series(dict(rows_hi)).sort_index()             # long high-attention − low
    turn = float(np.mean(ent) * 12)
    gross_sr = (ls.mean() / ls.std()) * np.sqrt(12) if ls.std() > 0 else np.nan
    # cost: monthly one-way turnover × 2 legs × 10bps (large-cap); net
    cost_m = turn / 12 * 2 * 0.0010
    net = ls - cost_m
    net_sr = (net.mean() / net.std()) * np.sqrt(12) if net.std() > 0 else np.nan
    dsr_g = deflated_sharpe_ratio(ls.values, n_trials=n_trials, periods_per_year=12)
    dsr_n = deflated_sharpe_ratio(net.values, n_trials=n_trials, periods_per_year=12)
    print(f"\nDEG ASVI attention L/S (long hi-attn − lo, monthly, n={len(ls)} months):")
    print(f"  GROSS  ann_ret={ls.mean()*12*100:+.2f}%  Sharpe={gross_sr:+.2f}  deflSR={dsr_g.deflated_sr:.3f}")
    print(f"  turnover≈{turn:.1f}x/yr -> NET Sharpe={net_sr:+.2f}  deflSR={dsr_n.deflated_sr:.3f}")
    print(f"  (DEG: high attention -> short-run UP then reversal; sign here is hi−lo next-month)")
    return {"n": len(ls), "gross_sr": gross_sr, "net_sr": net_sr, "turnover": turn,
            "gross_deflSR": dsr_g.deflated_sr, "net_deflSR": dsr_n.deflated_sr}


if __name__ == "__main__":
    import sys, warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        tks = _top_n_tickers(10)["ticker"].tolist()
        print("test tickers:", tks)
        t0 = time.time()
        svi = pull_svi(tks, sleep=2.0)
        print(f"pulled {svi['ticker'].nunique()} tickers, {len(svi)} rows in {time.time()-t0:.0f}s")
        print(svi.groupby("ticker").size().to_string())
    elif cmd == "pull":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
        pull_svi(_top_n_tickers(n)["ticker"].tolist())
    elif cmd == "eval":
        evaluate(build_asvi_panel())
