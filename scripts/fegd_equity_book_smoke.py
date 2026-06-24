"""scripts/fegd_equity_book_smoke.py — FEGD Phase 1 smoke test.

Runs the factor exposure gap detector against the deployed equity_book
sleeve to demonstrate the closed loop: data-driven gap identification
→ proposed improvement_directions.

Expected outcome (per Phase 1 hypothesis): equity_book should have ZERO
significant BAB loading (because PEAD-PIT-SN + analyst revision capture
earnings momentum, not low-vol anomaly). FEGD should auto-propose
"low-vol BAB variants" — exactly what the yaml `improvement_directions`
DOESN'T list (the cognitive blindspot identified in the 2026-06-17
deployment_demand_emitter session).

Run:
    python scripts/fegd_equity_book_smoke.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.portfolio.combined_book import build_equity_book
from engine.research.factor_exposure_gap_detector import (
    GAP_T_THRESHOLD,
    build_canonical_factor_matrix,
    deployed_factor_exposure,
    propose_improvement_directions,
)


OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "fegd_equity_book_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Building canonical factor matrix...")
    fm = build_canonical_factor_matrix()
    print(f"  factor matrix: cols={list(fm.columns)}")
    print(f"  range: {fm.index.min().date()} → {fm.index.max().date()}")
    print()

    print("Building deployed equity_book PnL...")
    equity = build_equity_book().dropna()
    print(f"  equity_book: n={len(equity)} "
          f"range={equity.index.min().date()} → {equity.index.max().date()}")
    print()

    print(f"Running factor exposure regression "
          f"(HAC lag 6, gap threshold |t| < {GAP_T_THRESHOLD})...")
    report = deployed_factor_exposure(
        equity, sleeve_id="equity_book", factor_matrix=fm,
    )

    print()
    print("=" * 80)
    print(f"FACTOR EXPOSURE — equity_book")
    print("=" * 80)
    print(f"  window: {report.window_start} → {report.window_end}  "
          f"n={report.n_obs}  R²={report.r_squared:.3f}")
    print()
    print(f"{'factor':<10}{'beta':>10}{'t-stat':>10}{'|t|':>9}{'gap?':>8}")
    print("-" * 50)
    for exp in report.exposures:
        is_gap = exp.is_gap()
        marker = "★ GAP" if is_gap else "       "
        print(f"  {exp.factor:<8}{exp.beta:>+10.3f}{exp.t_stat:>+10.3f}"
              f"{abs(exp.t_stat):>9.3f}  {marker}")

    print()
    print("=" * 80)
    print(f"PROPOSED improvement_directions (data-driven, NOT in yaml)")
    print("=" * 80)
    proposals = propose_improvement_directions(report)
    if not proposals:
        print("  (none — equity_book covers all canonical factors)")
    else:
        for p in proposals:
            print(f"  ★ {p.gap_factor:<8} → {p.direction_text}")
            print(f"        family: {p.mechanism_family}")
            print(f"        rationale: {p.rationale[:120]}")
            print()

    out_json = OUT_DIR / "equity_book_exposure_report.json"
    out_json.write_text(json.dumps({
        "sleeve_id":      report.sleeve_id,
        "report":         report.to_dict(),
        "proposals":      [{
            "sleeve_id":         p.sleeve_id,
            "gap_factor":        p.gap_factor,
            "direction_text":    p.direction_text,
            "mechanism_family":  p.mechanism_family,
            "rationale":         p.rationale,
        } for p in proposals],
        "method":         f"HAC-OLS lag 6, gap threshold |t|<{GAP_T_THRESHOLD}",
        "factor_matrix_columns": list(fm.columns),
    }, indent=2, default=str))
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
