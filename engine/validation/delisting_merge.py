"""engine/validation/delisting_merge.py — splice CRSP delisting returns into the daily panel.

Closes audit residual #1 (docs/live_delivers_backtest_audit_2026-05-25.md + delisting_bias.py):
the daily return panel (crsp_hist_daily_ret, from crsp.dsf) omits CRSP delisting returns (dlret)
— a name that delists has a final move (often a big bankruptcy loss, sometimes a merger premium)
that the panel drops by simply ending the series. This module pulls crsp.dsedelist, applies the
Shumway-1997 / Shumway-Warther-1999 fallback for missing performance-delisting returns, and splices
each delisting return onto the first trading day AFTER the name's last observed return.

Shumway fallback (for missing dlret only):
  - performance/liquidation delisting (dlstcd in [400, 600)) → -0.30 for NYSE/AMEX (hexcd 1/2),
    -0.55 for NASDAQ (hexcd 3)  [Shumway 1997; Shumway-Warther 1999]
  - non-performance (merger 200s / exchange 300s) with missing dlret → 0.0 (no catastrophic move)
  - present dlret → used as-is (covers acquisition premiums etc.)

Output is a SEPARATE file (crsp_hist_daily_ret_dladj.parquet); the original panel is NOT
overwritten, so the live/registered path is byte-unchanged. Use via
dpead_recon.build_dpead_recon_returns(ret_path=spliced) to measure the refined D_PEAD.

Per delisting_bias.py the gap is CONSERVATIVE (delisted names skew low-SUE/short side), so this
splice REFINES (mostly improves the short leg), it cannot reveal the 1.04 to be inflated.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_RET = "data/cache/crsp_hist_daily_ret.parquet"
_DL = "data/cache/_crsp_dsedelist.parquet"
_OUT = "data/cache/crsp_hist_daily_ret_dladj.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine("postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
                         connect_args={"sslmode": "require", "connect_timeout": 25})


def fetch_delisting_returns(d0: str = "2013-10-01", d1: str = "2024-06-30",
                            force: bool = False) -> pd.DataFrame:
    """Pull crsp.dsedelist (permno, dlstdt, dlstcd, dlret, hexcd) for the sample. Cached."""
    if os.path.exists(_DL) and not force:
        return pd.read_parquet(_DL)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        dl = pd.read_sql(text(
            "select permno, dlstdt, dlstcd, dlret, hexcd from crsp.dsedelist "
            f"where dlstdt between '{d0}' and '{d1}'"), eng)
    finally:
        eng.dispose()
    dl.to_parquet(_DL, index=False)
    return dl


def _fill_dlret(dlret, dlstcd, hexcd) -> float:
    """Shumway fallback for MISSING dlret only."""
    if pd.notna(dlret):
        return float(dlret)
    if 400 <= dlstcd < 600:                 # performance / liquidation
        return -0.55 if hexcd == 3 else -0.30   # NASDAQ vs NYSE/AMEX
    return 0.0                              # merger/exchange missing → no catastrophic move


def build_spliced_panel(save: bool = True) -> pd.DataFrame:
    """Return the daily panel with delisting returns spliced onto the day after each name's
    last observed return. Names still trading at panel end (right-censored) are left untouched."""
    panel = pd.read_parquet(_RET)
    panel["date"] = pd.to_datetime(panel["date"])
    dl = fetch_delisting_returns()
    dl = dl[dl["permno"].isin(panel["permno"].unique())].copy()
    dl["dlstdt"] = pd.to_datetime(dl["dlstdt"])
    dl["dlret_fill"] = [
        _fill_dlret(r.dlret, r.dlstcd, r.hexcd) for r in dl.itertuples()]

    cal = np.sort(panel["date"].unique())
    gmax = cal.max()
    last = panel.groupby("permno")["date"].max()

    rows = []
    for r in dl.itertuples():
        p = r.permno
        if p not in last.index:
            continue
        ld = np.datetime64(last[p])
        if ld >= gmax:                      # right-censored: can't place after panel end
            continue
        idx = int(np.searchsorted(cal, ld, "right"))
        if idx >= len(cal):
            continue
        # sanity: delisting date should be near the stop (within ~45 calendar days)
        if abs((r.dlstdt - pd.Timestamp(ld)).days) > 45:
            continue
        if r.dlret_fill == 0.0:
            continue
        rows.append((p, pd.Timestamp(cal[idx]), float(r.dlret_fill)))

    add = pd.DataFrame(rows, columns=["permno", "date", "ret"])
    merged = (pd.concat([panel, add], ignore_index=True)
              .drop_duplicates(["permno", "date"], keep="first")
              .sort_values(["permno", "date"]).reset_index(drop=True))
    if save:
        merged.to_parquet(_OUT, index=False)
    return merged


def splice_summary() -> dict:
    """How many delisting returns get spliced + their direction."""
    panel = pd.read_parquet(_RET, columns=["permno", "date"])
    panel["date"] = pd.to_datetime(panel["date"])
    merged = build_spliced_panel(save=True)
    add_n = len(merged) - len(panel)
    spliced = merged.merge(panel[["permno", "date"]], on=["permno", "date"], how="left",
                           indicator=True)
    only_new = spliced[spliced["_merge"] == "left_only"]
    return {
        "delisting_rows_spliced": int(add_n),
        "mean_spliced_ret": round(float(only_new["ret"].mean()), 4) if add_n else None,
        "n_negative": int((only_new["ret"] < 0).sum()),
        "n_positive": int((only_new["ret"] > 0).sum()),
        "min_ret": round(float(only_new["ret"].min()), 4) if add_n else None,
        "out_path": _OUT,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(splice_summary(), indent=2))
