"""scripts/fegd_emit_demand_all_sleeves.py — FEGD Stage 1 CLI.

Runs FEGD on all 3 deployed alpha sleeves and emits the detected gap
factors as capability_gaps rows tagged source:fegd_factor_gap.

The burndown_ranker reads load_demand_families from capability_gaps;
any family in the demand set gets a ×1.5 multiplier on demand_score.
The ranker doesn't distinguish FEGD vs yaml source — both contribute
to the demand set equally — but the source tag is preserved for audit.

Idempotent: re-runs skip already-present signatures (per-(sleeve,
family,gap_factor)). Safe to schedule monthly.

Run:
    python scripts/fegd_emit_demand_all_sleeves.py            # dry-run
    python scripts/fegd_emit_demand_all_sleeves.py --write    # commit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.portfolio.combined_book import (
    build_equity_book, build_carry_book, build_tsmom_book,
)
from engine.research.factor_exposure_gap_detector import (
    build_canonical_factor_matrix, emit_fegd_demand,
)

SLEEVES = [
    ("equity_book",       build_equity_book, ()),
    ("cross_asset_carry", build_carry_book,  ("XA_CARRY",)),
    ("cross_asset_tsmom", build_tsmom_book,  ("XA_TSMOM",)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                     help="commit FEGD gap rows to capability_gaps.jsonl")
    args = ap.parse_args()
    dry = not args.write

    print(f"Mode: {'WRITE' if not dry else 'DRY-RUN (preview only)'}")
    print()
    print("Building canonical factor matrix (reused across sleeves)...")
    fm = build_canonical_factor_matrix()
    print(f"  cols={list(fm.columns)}")
    print()

    total_parsed = 0
    total_present = 0
    total_written = 0
    for sleeve_id, builder, excl in SLEEVES:
        print(f"=== {sleeve_id} ===")
        s = builder().dropna()
        result = emit_fegd_demand(
            sleeve_id=sleeve_id, sleeve_pnl=s,
            factor_matrix=fm, exclude_factors=excl, dry_run=dry,
        )
        print(f"  parsed gaps:      {result['parsed']}")
        print(f"  already present:  {result['already_present']}")
        print(f"  written:          {result['written']}")
        if result["new_rows"]:
            label = "WOULD write" if dry else "wrote"
            print(f"  rows {label}:")
            for row in result["new_rows"]:
                print(f"    [{row['family']:<22}] {row['gap_factor']:<10} "
                      f"β={row['beta']:+.3f}  t={row['t_stat']:+.3f}  "
                      f"({row['direction']})")
        else:
            print(f"  (no new gaps to write)")
        print()
        total_parsed  += result["parsed"]
        total_present += result["already_present"]
        total_written += result["written"]

    print("=" * 70)
    print(f"TOTAL: parsed={total_parsed} already_present={total_present} written={total_written}")
    if dry:
        print()
        print("(--dry-run mode — pass --write to commit rows to capability_gaps.jsonl)")


if __name__ == "__main__":
    main()
