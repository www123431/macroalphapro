"""scripts/audit_book_level_ff5_overlay_vs_deployed.py — first FEGD-driven enhance.

The FEGD cross-sleeve scan (commit 7ef..., this session) identified
HML, RMW, CMA as gap factors in ALL THREE alpha sleeves (equity_book,
cross_asset_carry, cross_asset_tsmom). The book has effectively ZERO
loading on these three FF5 dimensions.

This audit tests whether adding small book-level overlays of these
factors (using Ken French cached series directly) improves book Sharpe
via paired Politis-Romano 1994 block bootstrap.

Variants tested (overlay weight × factor)
=========================================
  HML_5pct  : book + 5% Ken French HML factor
  CMA_5pct  : book + 5% Ken French CMA factor
  RMW_5pct  : book + 5% Ken French RMW factor
  3factor_blend_3pct : book + 1% HML + 1% RMW + 1% CMA

Each variant re-vol-targeted to baseline book vol (swap framing).

Why book level (not single sleeve)
==================================
The FEGD scan showed the gaps are CONSISTENT across all 3 alpha
sleeves. So we're not enhancing a single sleeve — we're adding a
factor to which the WHOLE book is structurally blind. Book-level
overlay is the cleanest test.

Why small weights (3-5%)
========================
Conservative — the FEGD gaps have |t| < 1.65 (not far from zero), so
the expected book Sharpe lift per unit of overlay is small. Testing
small weights probes whether the marginal effect is statistically
meaningful before committing to large allocation.

Run:
    python scripts/audit_book_level_ff5_overlay_vs_deployed.py
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

from engine.portfolio.combined_book import build_combined_book
from engine.research.enhance import dispatch_enhance_hypothesis

KF_DAILY = _REPO_ROOT / "data" / "cache" / "ken_french_ff5_mom_daily.parquet"
OUT_DIR  = _REPO_ROOT / "data" / "research_store" / "audit" / "book_ff5_overlay_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG_PATH = OUT_DIR / "enhance_verdict_log.jsonl"


def _load_kf_monthly() -> pd.DataFrame:
    df = pd.read_parquet(KF_DAILY)
    return ((1.0 + df).resample("ME").prod() - 1.0).dropna(how="all")


def _ann_sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 12 or s.std(ddof=1) <= 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * math.sqrt(12))


def _vol_target(s: pd.Series, target_ann: float = 0.10) -> pd.Series:
    s = s.dropna()
    v = s.std(ddof=1) * math.sqrt(12)
    if not math.isfinite(v) or v <= 0:
        return s
    return s * (target_ann / v)


def main():
    print("Building deployed combined book (5-sleeve regime-conditional)...")
    book = build_combined_book(
        crisis_risk_weight    = 0.05,
        mom_hedge_risk_weight = 0.02,
        regime_conditional    = True,
        book_vol_target       = 0.10,
    ).dropna()
    book.index = pd.to_datetime(book.index).to_period("M").to_timestamp("M")
    print(f"  book: n={len(book)} "
          f"range={book.index.min().date()} → {book.index.max().date()}  "
          f"Sharpe={_ann_sharpe(book):+.3f}")

    print("Loading Ken French monthly factor series...")
    kf = _load_kf_monthly()
    kf.index = pd.to_datetime(kf.index).to_period("M").to_timestamp("M")
    print(f"  KF cols={list(kf.columns)}  range={kf.index.min().date()} → {kf.index.max().date()}")
    print()

    # Vol-target each factor to 10% so overlay weights are interpretable
    hml_vt = _vol_target(kf["HML"].dropna(), 0.10)
    cma_vt = _vol_target(kf["CMA"].dropna(), 0.10)
    rmw_vt = _vol_target(kf["RMW"].dropna(), 0.10)

    common = book.index.intersection(hml_vt.index)
    book_c = book.loc[common]
    print(f"Overlap window: {common.min().date()} → {common.max().date()} "
          f"({len(common)} months)")
    print(f"Baseline book (in overlap, vt 10%):  Sharpe={_ann_sharpe(book_c):+.3f}")
    print()

    # Build overlays
    variants = {}
    for name, factor, w in [
        ("HML_5pct",          hml_vt, 0.05),
        ("CMA_5pct",          cma_vt, 0.05),
        ("RMW_5pct",          rmw_vt, 0.05),
        ("3factor_blend_3pct", None,  None),
    ]:
        if name == "3factor_blend_3pct":
            f = (hml_vt.loc[common] + cma_vt.loc[common] + rmw_vt.loc[common]) / 3
            blended = (1 - 0.03) * book_c + 0.03 * f
        else:
            blended = (1 - w) * book_c + w * factor.loc[common]
        # Re-vol-target to baseline vol (swap framing)
        variants[name] = _vol_target(blended.dropna(), 0.10)

    print("Standalone vol-targeted Sharpe (10% target):")
    print(f"  {'baseline_book':<22} Sharpe={_ann_sharpe(book_c):+.3f}")
    for name, s in variants.items():
        print(f"  {name:<22} Sharpe={_ann_sharpe(s):+.3f}")
    print()

    print("Paired enhance test (Politis-Romano 1994, B=2000, block=6mo):")
    print("=" * 95)
    print(f"{'variant':<22}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}{'CI low':>11}{'CI high':>11}{'corr':>8}")
    print("-" * 95)

    baseline = _vol_target(book_c, 0.10)
    results = []
    for name, variant in variants.items():
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"book_ff5_overlay_{name}",
            sleeve_id        = "combined_book",
            variant_returns  = variant,
            baseline_returns = baseline,
            cron_run_id      = None,
            cron_source      = "manual_audit",
            n_iterations     = 2000,
            block_size       = 6,
            log_path         = ENHANCE_LOG_PATH,
            seed             = 42,
        )
        b = r.bootstrap_result or {}
        ds = b.get("sharpe_diff_observed"); t = b.get("sharpe_diff_t_stat")
        p  = b.get("sharpe_diff_p_value");  lo= b.get("sharpe_diff_ci_lo")
        hi = b.get("sharpe_diff_ci_hi");    c = b.get("correlation")
        v_label = r.refusal_reason or r.verdict
        ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
        t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
        p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
        lo_s = f"{lo:>+10.4f}" if lo is not None else "     n/a"
        hi_s = f"{hi:>+10.4f}" if hi is not None else "     n/a"
        c_s  = f"{c:>+7.3f}"  if c  is not None else "   n/a"
        print(f"{name:<22}{v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}{c_s}")
        results.append({
            "variant": name, "verdict": r.verdict, "refusal": r.refusal_reason,
            "bootstrap": b, "summary": r.summary,
        })

    out_json = OUT_DIR / "book_ff5_overlay_results.json"
    out_json.write_text(json.dumps({
        "subject":          "combined_book",
        "baseline":         "build_combined_book(5-sleeve regime-conditional, vt 10%)",
        "fegd_source":      "All 3 alpha sleeves show HML/RMW/CMA gap (|t|<1.65)",
        "method":           "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":  int(len(baseline)),
        "results":          results,
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
