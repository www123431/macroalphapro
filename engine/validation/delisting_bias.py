"""engine/validation/delisting_bias.py — direction/magnitude of the missing-delisting-return bias.

Audit residual #1 (docs/live_delivers_backtest_audit_2026-05-25.md): the CRSP daily-return panel
(crsp_hist_daily_ret) uses raw `ret` without CRSP delisting returns (dlret). The audit question
is NOT "is there a gap" (there is) but "does it make the backtest OPTIMISTIC (a concern) or
CONSERVATIVE (benign)?" — which depends on whether delisted names cluster on the LONG or SHORT
side of the SUE signal.

This is answerable OFFLINE (no WRDS / no dlret needed): a name that delists has a final loss the
panel omits; if it was SHORTED (low SUE), the strategy MISSES a gain → backtest is conservative;
if it was LONG (high SUE), the strategy misses a loss → backtest is inflated.

Finding (2026-05-25): delisted names skew LOW-SUE (short side) → the bias is CONSERVATIVE. The
full fix (merge actual dlret + Shumway fallback) needs a WRDS pull and would only REFINE (mostly
improve) the backtest — it cannot reveal the 1.04 to be inflated by this gap.
"""
from __future__ import annotations

import pandas as pd

_RET = "data/cache/crsp_hist_daily_ret.parquet"
_SUE_PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"


def quantify_delisting_bias(min_gap_days: int = 90) -> dict:
    """Identify names that delisted/stopped trading during the sample and test whether they
    skew to the SHORT (low-SUE) side. Returns the bias direction + supporting stats."""
    ret = pd.read_parquet(_RET, columns=["permno", "date"])
    ret["date"] = pd.to_datetime(ret["date"])
    gmax = ret["date"].max()
    last = ret.groupby("permno")["date"].max()
    delisted = set(last[last < gmax - pd.Timedelta(days=min_gap_days)].index)

    sue = pd.read_parquet(_SUE_PANEL, columns=["permno", "sue"]).dropna()
    sue_avg = sue.groupby("permno")["sue"].mean()
    dl = sue_avg.reindex([p for p in delisted if p in sue_avg.index]).dropna()
    if dl.empty:
        return {"available": False, "reason": "no delisted names with SUE"}

    q25, q75 = sue_avg.quantile(0.25), sue_avg.quantile(0.75)
    mean_all, mean_dl = float(sue_avg.mean()), float(dl.mean())
    return {
        "available": True,
        "panel_end": str(gmax)[:10],
        "n_total_permno": int(ret["permno"].nunique()),
        "n_delisted": int(len(delisted)),
        "n_delisted_with_sue": int(len(dl)),
        "mean_sue_all": round(mean_all, 4),
        "mean_sue_delisted": round(mean_dl, 4),
        "delisted_pct_short_side": round(float((dl <= q25).mean()), 4),   # bottom-quartile SUE
        "delisted_pct_long_side": round(float((dl >= q75).mean()), 4),    # top-quartile SUE
        # delisted names lower-SUE than universe ⇒ over-represented on the SHORT leg ⇒
        # their omitted final loss is a MISSED short gain ⇒ backtest understates ⇒ conservative.
        "bias_direction": "conservative" if mean_dl < mean_all else "inflating",
    }
