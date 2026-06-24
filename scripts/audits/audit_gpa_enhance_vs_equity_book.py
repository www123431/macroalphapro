"""scripts/audit_gpa_enhance_vs_equity_book.py — GP/A enhance test (Step B).

Tests whether adding GP/A as a weighted overlay to the deployed equity_book
sleeve produces a statistically significant Sharpe improvement via paired
block bootstrap (Politis-Romano 1994 + Jobson-Korkie 1981 / Memmel 2003).

Why this matters for the Sharpe 1.32 → 1.5+ goal:
  - Forward pipeline (FF5+MOM spanning) said: GP/A α-t = 1.89 — MARGINAL
  - That number cannot answer "does this improve OUR book"
  - Only paired test vs deployed PnL series can
  - This is the correct adjudicator per forward-vs-enhance doctrine

Test design:
  - Baseline:  build_equity_book() — the deployed cash equity sleeve PnL
  - Variants:  baseline blended with GP/A at 5%, 10%, 20% weight
               (all vol-targeted to ~10% to match book sizing)
  - For each weight: paired block bootstrap, IMPROVEMENT/NOISE/DEGRADATION

Run:
    python scripts/audit_gpa_enhance_vs_equity_book.py
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

from engine.portfolio.combined_book import build_equity_book
from engine.research.enhance import dispatch_enhance_hypothesis

GPA_PNL_PARQUET = _REPO_ROOT / "data" / "research_store" / "tier_c_pnl" / "dc4cf6beaa247880_GREEN.parquet"
OUT_DIR         = _REPO_ROOT / "data" / "research_store" / "audit" / "gpa_enhance_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENHANCE_LOG_PATH = OUT_DIR / "enhance_verdict_log.jsonl"


def _load_gpa_pnl_monthly() -> pd.Series:
    df = pd.read_parquet(GPA_PNL_PARQUET)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    s = df["pnl_net_13bp"].dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    return s


def _vol_target(series: pd.Series, target_ann_vol: float = 0.10) -> pd.Series:
    """Scale the series so its annualized vol matches target. One-shot scalar."""
    monthly_vol = series.std(ddof=1)
    if not math.isfinite(monthly_vol) or monthly_vol <= 0:
        return series
    ann_vol = monthly_vol * math.sqrt(12)
    return series * (target_ann_vol / ann_vol)


def main():
    print("Building deployed equity_book PnL...")
    equity = build_equity_book()
    equity.index = pd.to_datetime(equity.index).to_period("M").to_timestamp("M")
    print(f"  equity_book: n={len(equity)}  "
          f"range={equity.index.min().date()} → {equity.index.max().date()}  "
          f"Sharpe={equity.mean()/equity.std(ddof=1)*math.sqrt(12):.3f}")

    print("Loading GP/A PnL...")
    gpa = _load_gpa_pnl_monthly()
    print(f"  GP/A: n={len(gpa)}  "
          f"range={gpa.index.min().date()} → {gpa.index.max().date()}  "
          f"Sharpe={gpa.mean()/gpa.std(ddof=1)*math.sqrt(12):.3f}")

    # Vol-target both to 10% so blending weights are interpretable
    equity_vt = _vol_target(equity, 0.10)
    gpa_vt    = _vol_target(gpa,    0.10)
    print(f"  vol-targeted both to 10%")
    print()

    # Restrict to overlapping window (equity_book starts later than GP/A)
    common = equity_vt.index.intersection(gpa_vt.index)
    baseline = equity_vt.loc[common]
    gpa_aligned = gpa_vt.loc[common]
    print(f"Overlapping window: {common.min().date()} → {common.max().date()} "
          f"({len(common)} months)")
    print(f"  baseline Sharpe (vt 10%): {baseline.mean()/baseline.std(ddof=1)*math.sqrt(12):.3f}")
    print(f"  GP/A (alone, vt 10%):     {gpa_aligned.mean()/gpa_aligned.std(ddof=1)*math.sqrt(12):.3f}")
    print(f"  corr(baseline, GP/A):     {baseline.corr(gpa_aligned):.3f}")
    print()

    weights = [0.05, 0.10, 0.20, 0.30]
    all_results = []

    print(f"{'wt':<6}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}{'CI low':>10}{'CI high':>10}{'corr':>8}")
    print("=" * 95)
    for w in weights:
        variant = (1 - w) * baseline + w * gpa_aligned
        # Vol-target the variant back to 10% (so we compare strategies of
        # the same total risk; this is the "swap" not "add" framing).
        variant_vt = _vol_target(variant, 0.10)

        result = dispatch_enhance_hypothesis(
            hypothesis_id    = f"gpa_overlay_w{int(w*100)}",
            sleeve_id        = "equity_book",
            variant_returns  = variant_vt,
            baseline_returns = baseline,
            cron_run_id      = None,
            cron_source      = "manual_audit",
            n_iterations     = 2000,
            block_size       = 6,
            log_path         = ENHANCE_LOG_PATH,
            seed             = 42,
        )
        b = result.bootstrap_result or {}
        ds = b.get("sharpe_diff_observed")
        t  = b.get("sharpe_diff_t_stat")
        p  = b.get("sharpe_diff_p_value")
        lo = b.get("sharpe_diff_ci_lo")
        hi = b.get("sharpe_diff_ci_hi")
        c  = b.get("correlation")

        fmt_n = lambda x, f="{:>+8.4f}": f.format(x) if x is not None else "n/a"
        ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
        t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
        p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
        lo_s = f"{lo:>+9.4f}" if lo is not None else "    n/a"
        hi_s = f"{hi:>+9.4f}" if hi is not None else "    n/a"
        c_s  = f"{c:>+7.3f}"  if c  is not None else "   n/a"
        v_label = result.refusal_reason or result.verdict
        print(f"{w:<6.0%}{v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}{c_s}")
        all_results.append({
            "weight":         w,
            "verdict":        result.verdict,
            "refusal":        result.refusal_reason,
            "refusal_detail": result.refusal_detail,
            "bootstrap":      b,
            "summary":        result.summary,
        })

    print()
    print("Bootstrap config: B=2000, block_size=6 months, seed=42")
    print(f"Baseline = build_equity_book() (PEAD-PIT-SN + analyst revision, vol-targeted)")
    print(f"Variant  = (1-w) * baseline + w * GP/A_vt, re-vol-targeted to 10%")
    print(f"Verdict log: {ENHANCE_LOG_PATH}")

    # Persist
    out_json = OUT_DIR / "gpa_enhance_results.json"
    out_json.write_text(json.dumps({
        "subject":          "tier_c_auto_seed_gpa_cross_sectional_rank",
        "parent_verdict_event_id": "704b792e-fb8c-4f93-95df-585f6818ab20",
        "baseline_sleeve":  "equity_book",
        "baseline_window":  [str(common.min().date()), str(common.max().date())],
        "n_paired_months":  len(common),
        "method":           "Politis-Romano 1994 paired circular block bootstrap, "
                              "B=2000, block_size=6mo, seed=42",
        "results_by_weight": all_results,
    }, indent=2, default=str))
    print(f"Results JSON:  {out_json}")


if __name__ == "__main__":
    main()
