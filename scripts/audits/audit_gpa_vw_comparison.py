"""scripts/audit_gpa_vw_comparison.py — GP/A self-doubt B4 audit.

Reproduces the 2026-06-08 GP/A backtest (subject
tier_c_auto_seed_gpa_cross_sectional_rank, spec_hash dc4cf6beaa247880)
under VALUE-WEIGHTED bucketing and compares to the original
EQUAL-WEIGHTED verdict.

Why: self-doubt #B4 flagged "EW-only reporting (B4) is a material
concern for a profitability factor where large-cap alpha is typically
weaker". Novy-Marx 2013 reports BOTH EW and VW; we only ran EW. If VW
also passes (Sharpe >= 0.5, |t| >= 1.96), GP/A is a clean
PROMOTE-to-paper-trade candidate. If VW fails, B4 is a legitimate
concern and PROMOTE should wait.

No LLM call. Reuses the cross_sec template's CRSP loaders + signal
builder; only swaps `np.mean(vals)` → `weighted average by lagged
mktcap` in the bucket-return step.

Run:
    python scripts/audit_gpa_vw_comparison.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from engine.agents.strengthener.templates.cross_sec_us_equities import (
    _load_crsp_msf, _load_crsp_delisting, _build_signal,
    _UNIVERSE_TOP_N, _N_QUINTILES, _TC_BP_PER_RT,
)
from engine.agents.strengthener._safety_constants import (
    MIN_STOCKS_PER_BUCKET as _MIN_STOCKS_PER_BUCKET,
)
from engine.research.ablation.metrics import (
    annualized_sharpe, newey_west_sharpe_se,
)


SIGNAL_KEY      = "gp_at"
DATE_RANGE      = ("1992-01-01", "2024-12-31")
WARMUP_MONTHS   = 14

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "gpa_vw_2026_06_16"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bucket_ret_vw(perms: set,
                    ret_t_next:    pd.Series,
                    mc_t:          pd.Series,
                    delist_lookup: dict) -> tuple[float, int]:
    """Value-weighted bucket return.

    Weight = lagged mktcap @ t (mc_t — same panel used for universe rank).
    Substitutes delisting return for permnos that delist between t and
    t+1, mirroring the EW path. Weights for delisted permnos still use
    the prior mc_t.
    """
    vals: list[float] = []
    weights: list[float] = []
    for p in perms:
        w = mc_t.get(p)
        if not pd.notna(w) or w <= 0:
            continue
        dl = delist_lookup.get((int(p), ret_t_next.name))
        if dl is not None:
            vals.append(float(dl))
            weights.append(float(w))
        else:
            r = ret_t_next.get(p)
            if pd.notna(r):
                vals.append(float(r))
                weights.append(float(w))
    if not vals or sum(weights) <= 0:
        return float("nan"), 0
    return float(np.average(vals, weights=weights)), len(vals)


def _vw_backtest(signal_panel: pd.DataFrame,
                  return_panel: pd.DataFrame,
                  mktcap_panel: pd.DataFrame,
                  delist_panel: pd.DataFrame,
                  *,
                  top_n:     int,
                  n_buckets: int,
                  tc_bp:     float) -> tuple[pd.Series, dict]:
    mktcap_lagged = mktcap_panel.shift(1)

    delist_lookup = {
        (int(r.permno), r.dlst_month_end): float(r.dlret)
        for r in delist_panel.itertuples(index=False)
        if pd.notna(r.dlret)
    }

    pnl: list[float]               = []
    pnl_dates: list[pd.Timestamp]  = []
    turnover: list[float]          = []
    n_stocks: list[int]            = []
    prev_long_set:  set            = set()
    prev_short_set: set            = set()

    sorted_dates = sorted(signal_panel.index)
    for i, t in enumerate(sorted_dates[:-1]):
        t_next = sorted_dates[i + 1]
        sig_t = signal_panel.loc[t]
        mc_t  = mktcap_lagged.loc[t] if t in mktcap_lagged.index else None
        if mc_t is None or mc_t.notna().sum() == 0:
            continue
        universe_mask = mc_t.notna() & sig_t.notna()
        if universe_mask.sum() < top_n // 4:
            continue
        mc_in_universe = mc_t[universe_mask]
        top_perms = mc_in_universe.nlargest(top_n).index
        sig_in_top = sig_t.loc[top_perms].dropna()
        if len(sig_in_top) < n_buckets * _MIN_STOCKS_PER_BUCKET:
            continue

        ranks = sig_in_top.rank(method="first")
        try:
            buckets = pd.qcut(ranks, n_buckets, labels=False,
                                duplicates="drop")
        except ValueError:
            continue

        q1_perms = set(sig_in_top.index[buckets == 0])
        q5_perms = set(sig_in_top.index[buckets == n_buckets - 1])
        if (len(q1_perms) < _MIN_STOCKS_PER_BUCKET
                or len(q5_perms) < _MIN_STOCKS_PER_BUCKET):
            continue

        ret_t_next = return_panel.loc[t_next] if t_next in return_panel.index else None
        if ret_t_next is None:
            continue
        ret_t_next.name = t_next

        r_long,  _ = _bucket_ret_vw(q5_perms, ret_t_next, mc_t, delist_lookup)
        r_short, _ = _bucket_ret_vw(q1_perms, ret_t_next, mc_t, delist_lookup)
        if not (math.isfinite(r_long) and math.isfinite(r_short)):
            continue

        gross = r_long - r_short

        if prev_long_set or prev_short_set:
            chg_long  = len(q5_perms.symmetric_difference(prev_long_set))  / max(len(q5_perms), 1)
            chg_short = len(q1_perms.symmetric_difference(prev_short_set)) / max(len(q1_perms), 1)
            to = chg_long + chg_short
        else:
            to = 2.0
        prev_long_set  = q5_perms
        prev_short_set = q1_perms

        pnl.append(gross)
        pnl_dates.append(t_next)
        turnover.append(to)
        n_stocks.append(len(sig_in_top))

    idx = pd.DatetimeIndex(pnl_dates)
    pnl_gross  = pd.Series(pnl, index=idx)
    turn_s     = pd.Series(turnover, index=idx)
    pnl_net_13 = pnl_gross - turn_s * (tc_bp / 10_000.0)
    pnl_net_80 = pnl_gross - turn_s * (80.0  / 10_000.0)
    diag = {
        "n_months":          int(len(pnl_gross)),
        "avg_turnover":      float(np.mean(turnover)) if turnover else float("nan"),
        "avg_universe_size": float(np.mean(n_stocks))  if n_stocks else float("nan"),
        "pnl_gross_series":  pnl_gross,
        "pnl_net_13_series": pnl_net_13,
        "pnl_net_80_series": pnl_net_80,
    }
    return pnl_net_13, diag


def _stats(pnl: pd.Series) -> dict:
    if pnl.empty:
        return {"sharpe": None, "nw_t": None, "n": 0}
    s = annualized_sharpe(pnl)
    se = newey_west_sharpe_se(pnl)
    t = (s / se) if (math.isfinite(s) and math.isfinite(se) and se > 0) else float("nan")
    return {
        "sharpe":     float(s)  if math.isfinite(s) else None,
        "nw_se":      float(se) if math.isfinite(se) else None,
        "nw_t":       float(t)  if math.isfinite(t) else None,
        "ann_return": float(pnl.mean()) * 12.0,
        "ann_vol":    float(pnl.std(ddof=1)) * math.sqrt(12.0),
        "n_months":   int(len(pnl)),
    }


def main():
    print("Loading CRSP MSF + delisting...")
    msf    = _load_crsp_msf()
    delist = _load_crsp_delisting()

    start, end = pd.Timestamp(DATE_RANGE[0]), pd.Timestamp(DATE_RANGE[1])
    fetch_start = start - pd.DateOffset(months=WARMUP_MONTHS)
    panel = msf[(msf["month_end"] >= fetch_start) & (msf["month_end"] <= end)]
    print(f"  CRSP panel rows={len(panel):,}  "
          f"months={panel['month_end'].nunique()}  "
          f"permnos={panel['permno'].nunique():,}")

    print(f"Building signal {SIGNAL_KEY!r} (funda join + PIT lag)...")
    signal_panel = _build_signal(panel, SIGNAL_KEY)
    return_panel = panel.pivot(index="month_end", columns="permno", values="ret")
    mktcap_panel = panel.pivot(index="month_end", columns="permno", values="mktcap")
    signal_panel = signal_panel.loc[signal_panel.index >= start]
    print(f"  signal_panel shape={signal_panel.shape}  "
          f"start={signal_panel.index.min().date() if len(signal_panel) else None}")

    print("Running VW backtest...")
    pnl_vw_net13, diag_vw = _vw_backtest(
        signal_panel  = signal_panel,
        return_panel  = return_panel,
        mktcap_panel  = mktcap_panel,
        delist_panel  = delist,
        top_n         = _UNIVERSE_TOP_N,
        n_buckets     = _N_QUINTILES,
        tc_bp         = _TC_BP_PER_RT,
    )

    print()
    print("=" * 64)
    print("GP/A AUDIT — VW vs EW (2026-06-16)")
    print("=" * 64)
    print(f"Window:           {DATE_RANGE[0]} → {DATE_RANGE[1]}")
    print(f"Universe:         top {_UNIVERSE_TOP_N} CRSP by lagged mktcap")
    print(f"Avg universe:     {diag_vw['avg_universe_size']:.0f}")
    print(f"N months:         {diag_vw['n_months']}")
    print(f"Avg turnover:     {diag_vw['avg_turnover']:.3f}")
    print()
    print("VW results:")
    for label, series in [("GROSS",  diag_vw["pnl_gross_series"]),
                          ("13bp",   diag_vw["pnl_net_13_series"]),
                          ("80bp",   diag_vw["pnl_net_80_series"])]:
        s = _stats(series)
        if s["sharpe"] is None:
            print(f"  {label}: empty")
            continue
        verdict = "GREEN" if abs(s["nw_t"] or 0) >= 1.96 else (
                  "MARGINAL" if abs(s["nw_t"] or 0) >= 1.65 else "RED")
        print(f"  {label:>5}: Sharpe={s['sharpe']:.3f}  NW-t={s['nw_t']:.3f}  "
              f"ann_ret={s['ann_return']:.2%}  ann_vol={s['ann_vol']:.2%}  "
              f"→ {verdict}")

    print()
    print("EW comparator (from 2026-06-08 verdict event 704b792e):")
    print("        13bp: Sharpe=0.670  NW-t=3.567  ann_ret=6.92%  ann_vol=10.32%  → GREEN")
    print("        80bp: Sharpe=0.535  NW-t=2.949  ann_ret=5.51%  ann_vol=10.31%  → GREEN")

    # Persist
    pnl_df = pd.DataFrame({
        "pnl_gross":    diag_vw["pnl_gross_series"],
        "pnl_net_13bp": diag_vw["pnl_net_13_series"],
        "pnl_net_80bp": diag_vw["pnl_net_80_series"],
    }).dropna(how="all")
    out_parquet = OUT_DIR / "gpa_vw_pnl_series.parquet"
    pnl_df.to_parquet(out_parquet)
    print()
    print(f"PnL series saved → {out_parquet}")

    out_json = OUT_DIR / "gpa_vw_stats.json"
    out_json.write_text(json.dumps({
        "subject":   "tier_c_auto_seed_gpa_cross_sectional_rank",
        "parent_verdict_event_id": "704b792e-fb8c-4f93-95df-585f6818ab20",
        "signal":    SIGNAL_KEY,
        "window":    list(DATE_RANGE),
        "universe":  f"top {_UNIVERSE_TOP_N} CRSP",
        "weighting": "value_weighted_by_lagged_mktcap",
        "vw_gross":  _stats(diag_vw["pnl_gross_series"]),
        "vw_13bp":   _stats(diag_vw["pnl_net_13_series"]),
        "vw_80bp":   _stats(diag_vw["pnl_net_80_series"]),
        "ew_13bp_from_2026_06_08_verdict": {
            "sharpe": 0.670, "nw_t": 3.567, "ann_return": 0.0692,
        },
        "ew_80bp_from_2026_06_08_verdict": {
            "sharpe": 0.535, "nw_t": 2.949, "ann_return": 0.0551,
        },
        "avg_universe_size": diag_vw["avg_universe_size"],
        "n_months":          diag_vw["n_months"],
        "avg_turnover":      diag_vw["avg_turnover"],
    }, indent=2, default=str))
    print(f"Stats saved → {out_json}")


if __name__ == "__main__":
    main()
