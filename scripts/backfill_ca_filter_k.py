"""scripts/backfill_ca_filter_k.py — Phase 5.7 operational follow-up.

Honest backfill of cost_model.ca_filter_k for 5 deployed sleeves.
Sleeves do NOT yet expose internal signal series, so we can't run
the 5.5 PBB-validated k-sweep against real history per-sleeve. So
we backfill with the paper's default k=2.0 + explicit
method=paper_default flag — when a sleeve gets signal-series
exposure (architectural change), re-run with the scaffold and
write back a calibrated k + method=pbb_sweep_calibrated.

This is what "shipping a placeholder honestly" looks like — explicit
about what's measured vs assumed, NEVER ships fake calibrated numbers.

Also writes a tcost_round_trip_bps field consistent with each
sleeve's existing Almgren-Chriss audit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / "data" / "research" / "mechanism_library"

# (sleeve_id (yaml stem),  paper_default_k,  tcost_round_trip_bps,
#  signal_type_for_audit_block,  notes)
BACKFILL_PLAN: list[tuple[str, float, int, str, str]] = [
    (
        "cross_asset_carry", 2.0, 8, "point_forecast",
        "Carry yield IS expected return (KMPV 2018). Direct identity "
        "calibration via signal_taxonomy. Paper k=2.0 default until "
        "PBB k-sweep on roll-yield history validates per-asset k."
    ),
    (
        "post_earnings_drift", 2.0, 30, "cross_sect_rank",
        "SUE decile in FF12. Cross-sect rank requires decile→forward-"
        "return calibration on SUE panel; PBB k-sweep deferred until "
        "panel signal series is exposed by SUE backtest."
    ),
    (
        "crisis_hedge_tlt_gld", 2.0, 5, "regime_indicator",
        "VIX 1y z regime (CALM/NORMAL/STRESS). Regime-conditional mean "
        "calibration. ETF-only universe (TLT/GLD), low tcost ~5bp RT. "
        "PBB k-sweep deferred until regime classifier exposed."
    ),
    (
        "mom_hedge_overlay", 2.0, 10, "vol_norm_zscore",
        "Beta-residual z vs MTUM. OLS calibration. PBB k-sweep deferred "
        "until residual signal exposed."
    ),
    (
        "time_series_momentum", 2.0, 8, "vol_norm_zscore",
        "MOP 2012 12-1 z-score per asset. OLS calibration. PBB k-sweep "
        "deferred until per-asset z series exposed."
    ),
    # PENDING_DEPLOY sleeves — same SG5 enforcement bar applies
    (
        "post_earnings_drift_pit_sn", 2.0, 30, "cross_sect_rank",
        "PIT-clean Sales-Net SUE decile (replacement candidate for "
        "post_earnings_drift). Same SUE rank family → cross-sect calibration. "
        "PBB k-sweep deferred until SUE panel signal exposed."
    ),
    (
        "tail_hedge_put_spread", 2.0, 50, "binary_trigger",
        "SPX put-spread (delta -25/-10) monthly roll, 5% notional. Binary "
        "active/inactive trigger. Higher tcost (~50bp RT) for option spreads "
        "vs futures. PBB k-sweep deferred until roll-history signal exposed."
    ),
]


CA_BLOCK_TEMPLATE = """\
  # ── Phase 5.7 Cost-Aware Execution Filter (CA) ──
  # Paper formula: |expected_return| > ca_filter_k × |Δposition| × tcost_round_trip
  # Engine: engine.portfolio.execution_filter.should_trade()
  # Signal calibration via engine.portfolio.signal_taxonomy
  ca_filter_k: {k}
  ca_filter_k_method: paper_default
  ca_filter_k_audit_date: "2026-06-01"
  ca_filter_k_audit_note: |
    {note}
  ca_signal_type: {signal_type}
  tcost_round_trip_bps: {tcost_bps}
"""


def backfill_yaml(
    yaml_path: Path, k: float, tcost_bps: int,
    signal_type: str, note: str,
) -> str:
    """Insert the CA block into the cost_model section, idempotent.

    Returns: "added" / "already_present" / "no_cost_model_block".
    """
    text = yaml_path.read_text(encoding="utf-8")
    # Idempotency: skip if already has ca_filter_k
    if re.search(r"^\s*ca_filter_k:\s", text, re.MULTILINE):
        return "already_present"
    # Find the cost_model block; insert before the next top-level key
    # (a yaml line at col 0 that isn't part of cost_model)
    lines = text.splitlines(keepends=True)
    cost_start: int | None = None
    insert_at: int | None = None
    for i, line in enumerate(lines):
        if line.startswith("cost_model:"):
            cost_start = i
            continue
        if cost_start is not None:
            # End of cost_model block = first line at col 0 that's a
            # new top-level key (letter/_) OR comment block separator
            stripped = line.lstrip("﻿")
            if (stripped and not stripped[0].isspace()
                  and not stripped.startswith("#")):
                insert_at = i
                break
    if cost_start is None:
        return "no_cost_model_block"
    if insert_at is None:
        insert_at = len(lines)
    block = CA_BLOCK_TEMPLATE.format(
        k=k, tcost_bps=tcost_bps,
        signal_type=signal_type,
        # Indent the multi-line note properly
        note=note.replace("\n", "\n    "),
    )
    new_text = "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])
    yaml_path.write_text(new_text, encoding="utf-8")
    return "added"


def main() -> int:
    results: dict[str, str] = {}
    for stem, k, tcost_bps, sig_type, note in BACKFILL_PLAN:
        yp = LIB / f"{stem}.yaml"
        if not yp.is_file():
            results[stem] = "MISSING_YAML"
            continue
        results[stem] = backfill_yaml(yp, k, tcost_bps, sig_type, note)

    print("CA filter backfill results:")
    for stem, status in results.items():
        print(f"  {stem:24s} → {status}")
    bad = [s for s in results.values()
           if s not in ("added", "already_present")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
