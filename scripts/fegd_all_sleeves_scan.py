"""scripts/fegd_all_sleeves_scan.py — FEGD scan across all 3 alpha sleeves.

Extends the 2026-06-17 equity_book smoke (commit 7ef...) to ALSO run
on cross_asset_carry and cross_asset_tsmom sleeves. Uses exclude_factors
to skip self-regression where the sleeve is itself one of the proxy
factors (XA_CARRY for carry sleeve, XA_TSMOM for TSMOM sleeve).

Goal: surface data-driven gap factors that nobody manually identified,
prioritize a next candidate to enhance-test based on EMPIRICAL gaps
rather than human intuition.

Run:
    python scripts/fegd_all_sleeves_scan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.portfolio.combined_book import (
    build_equity_book, build_carry_book, build_tsmom_book,
)
from engine.research.factor_exposure_gap_detector import (
    GAP_T_THRESHOLD,
    build_canonical_factor_matrix,
    deployed_factor_exposure,
    propose_improvement_directions,
)


OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "fegd_all_sleeves_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SLEEVES = [
    # (sleeve_id, builder, exclude_self_factors)
    ("equity_book",        build_equity_book, ()),
    ("cross_asset_carry",  build_carry_book,  ("XA_CARRY",)),
    ("cross_asset_tsmom",  build_tsmom_book,  ("XA_TSMOM",)),
]


def main():
    print("Building canonical factor matrix once (reuse across sleeves)...")
    fm = build_canonical_factor_matrix()
    print(f"  factors: {list(fm.columns)}")
    print(f"  range:   {fm.index.min().date()} → {fm.index.max().date()}")
    print()

    all_reports: list[dict] = []
    all_proposals: list[dict] = []

    for sleeve_id, builder, excl in SLEEVES:
        print(f"=== {sleeve_id} ===")
        s = builder().dropna()
        print(f"  n={len(s)} range={s.index.min().date()} → {s.index.max().date()}")
        report = deployed_factor_exposure(
            s, sleeve_id=sleeve_id, factor_matrix=fm,
            exclude_factors=excl,
        )
        print(f"  window: {report.window_start} → {report.window_end} "
              f"n={report.n_obs}  R²={report.r_squared:.3f}")
        print(f"  {'factor':<10}{'beta':>10}{'t':>10}  gap?")
        for e in report.exposures:
            marker = "★ GAP" if e.is_gap() else "      "
            print(f"  {e.factor:<10}{e.beta:>+10.3f}{e.t_stat:>+10.3f}  {marker}")
        print()
        proposals = propose_improvement_directions(report)
        if proposals:
            print(f"  PROPOSED directions:")
            for p in proposals:
                print(f"    ★ {p.gap_factor:<10} → [{p.mechanism_family}] "
                       f"{p.direction_text}")
        else:
            print(f"  (no proposed directions)")
        print()
        all_reports.append({
            "sleeve_id": sleeve_id,
            "exclude_factors": list(excl),
            "report": report.to_dict(),
        })
        for p in proposals:
            all_proposals.append({
                "sleeve_id":         p.sleeve_id,
                "gap_factor":        p.gap_factor,
                "direction_text":    p.direction_text,
                "mechanism_family":  p.mechanism_family,
                "rationale":         p.rationale,
            })

    # Cross-sleeve summary
    print("=" * 80)
    print("CROSS-SLEEVE GAP SUMMARY")
    print("=" * 80)
    from collections import Counter
    gap_counter: Counter = Counter()
    for r in all_reports:
        for fac in r["report"]["gap_factors"]:
            gap_counter[fac] += 1
    print("Factor gap frequency (# sleeves missing this loading):")
    for fac, n in gap_counter.most_common():
        print(f"  {fac:<10} : {n}/{len(SLEEVES)} sleeves")

    # Persist
    out_json = OUT_DIR / "all_sleeves_exposure_scan.json"
    out_json.write_text(json.dumps({
        "reports":   all_reports,
        "proposals": all_proposals,
        "factor_matrix_columns": list(fm.columns),
        "gap_frequency": dict(gap_counter),
        "n_sleeves":     len(SLEEVES),
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
