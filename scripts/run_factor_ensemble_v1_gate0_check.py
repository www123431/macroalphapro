"""
scripts/run_factor_ensemble_v1_gate0_check.py — Gate 0 baseline reproducibility.

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50) §五 Gate 0 + §rule-9 N9

Purpose (revised 2026-05-09 per Sprint-Week-4 pre-flight audit Fix #2)
----------------------------------------------------------------------
Gate 0 PRIMARY logic: validate the new walk-forward harness produces a
NON-PATHOLOGICAL baseline (BAB-only) Sharpe under monthly rebalance, so the
SAME harness's ensemble run can be compared apples-to-apples against this
internal baseline.

Apples-to-oranges issue (caught pre-implementation):
  Initial Gate 0 spec referenced B++ Mass FDR Tier 1 OOS Sharpe (0.985) as
  match target with ±0.001 tolerance. **B++ Tier 1 used WEEKLY rebalance**;
  our walk-forward uses MONTHLY. Different rebalance frequency = different
  Sharpe number; ±0.001 strict match is impossible by design, not by harness
  defect.

Revised Gate 0 (2026-05-09):
  1. HARNESS INTERNAL CONSISTENCY (mandatory pass): harness baseline-only run
     produces a finite, non-NaN, reasonable Sharpe (range [-0.5, 2.0])
  2. BAB DIRECTIONAL SANITY (informational): harness baseline Sharpe within
     ±0.5 of B++ Tier 1 0.985 reference (broad band, not strict match;
     differences expected due to monthly-vs-weekly rebalance + harness
     simplifications)
  3. ENSEMBLE VS BASELINE COMPARABILITY: BOTH runs use SAME harness, SAME
     date range, SAME universe → ΔSharpe is apples-to-apples within harness

Output
------
  data/factor_ensemble_v1/gate0_baseline_check.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("gate0_baseline_check")


# Reference: B++ Mass FDR Tier 1 OOS BAB Sharpe (project's marginal verdict baseline)
# Methodology note: B++ Tier 1 used WEEKLY rebalance; this harness uses MONTHLY.
# Used here as INFORMATIONAL directional check only, NOT as strict match target.
_BPP_REFERENCE_SHARPE: float = 0.985
_BPP_REFERENCE_SOURCE: str = (
    "B++ Mass FDR Tier 1 OOS verdict (project memory project_b_plus_marginal_2026-05-04.md). "
    "WEEKLY rebalance methodology (not directly comparable to monthly harness)."
)

# Gate criteria (revised 2026-05-09 Fix #2)
_HARNESS_SHARPE_RANGE: tuple[float, float] = (-0.5, 2.0)  # mandatory non-pathological range
_DIRECTIONAL_SANITY_BAND: float = 0.5  # ±0.5 vs B++ reference (broad, informational)


def main(verbose: bool = False) -> dict:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from engine.factor_ensemble_walk_forward import (
        OOS_START_DATE,
        DEFAULT_END_DATE,
        run_walk_forward,
        _DATA_DIR,
    )

    logger.info("=" * 70)
    logger.info("Gate 0: harness baseline (BAB-only, monthly rebalance) sanity check")
    logger.info("OOS window: %s → %s", OOS_START_DATE, DEFAULT_END_DATE)
    logger.info("Mandatory non-pathological range: [%.2f, %.2f]", *_HARNESS_SHARPE_RANGE)
    logger.info("Informational B++ directional band: %.4f ± %.2f",
                _BPP_REFERENCE_SHARPE, _DIRECTIONAL_SANITY_BAND)
    logger.info("=" * 70)

    result = run_walk_forward(
        start_date=OOS_START_DATE,
        end_date=DEFAULT_END_DATE,
        baseline_only=True,
        use_cache=True,
        persist=False,  # don't pollute walk_forward.parquet with baseline-only run
    )

    harness_sharpe = result.annualized_sharpe

    # MANDATORY: non-pathological range
    range_lo, range_hi = _HARNESS_SHARPE_RANGE
    in_range = range_lo <= harness_sharpe <= range_hi
    finite = (
        result.n_periods > 0 and
        not (harness_sharpe != harness_sharpe)  # NaN check
    )
    mandatory_pass = in_range and finite

    # INFORMATIONAL: directional vs B++ Tier 1 reference
    bpp_delta = harness_sharpe - _BPP_REFERENCE_SHARPE
    bpp_directional_pass = abs(bpp_delta) <= _DIRECTIONAL_SANITY_BAND

    if not finite:
        gate_status = "FAIL_PATHOLOGICAL"
    elif not mandatory_pass:
        gate_status = "FAIL_OUT_OF_RANGE"
    elif not bpp_directional_pass:
        gate_status = "PASS_WITH_DIRECTIONAL_CAVEAT"  # mandatory pass, informational warn
    else:
        gate_status = "PASS"

    summary = {
        "spec_id":               50,
        "gate":                  "Gate 0 harness baseline reproducibility (revised 2026-05-09)",
        "harness_sharpe":        round(harness_sharpe, 4),
        "harness_n_periods":     result.n_periods,
        "harness_annualized_vol": round(result.annualized_vol, 4),
        "harness_cumulative_return": round(result.cumulative_return, 4),
        "harness_max_drawdown":  round(result.max_drawdown, 4),
        "mandatory_range":       _HARNESS_SHARPE_RANGE,
        "mandatory_in_range":    in_range,
        "mandatory_finite":      finite,
        "mandatory_pass":        mandatory_pass,
        "bpp_reference_sharpe":  _BPP_REFERENCE_SHARPE,
        "bpp_reference_source":  _BPP_REFERENCE_SOURCE,
        "bpp_delta":             round(bpp_delta, 4),
        "bpp_directional_band":  _DIRECTIONAL_SANITY_BAND,
        "bpp_directional_pass":  bpp_directional_pass,
        "status":                gate_status,
        "completed_at":          datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "honest_disclosure": (
            "B++ reference used WEEKLY rebalance; harness uses MONTHLY rebalance. "
            "These are NOT apples-to-apples — directional band check is "
            "informational only. PRIMARY Gate 0 logic is internal harness "
            "consistency: mandatory non-pathological Sharpe AND finite n_periods. "
            "Subsequent ensemble vs baseline ΔSharpe in same harness IS "
            "apples-to-apples and that's what verdict framework uses."
        ),
    }

    output_path = _DATA_DIR / "gate0_baseline_check.json"
    try:
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Persisted to %s", output_path)
    except Exception as exc:
        logger.warning("Persist failed: %s", exc)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("=" * 70)
    logger.info("Gate 0 status: %s | harness Sharpe=%.4f | B++ delta=%+.4f",
                gate_status, harness_sharpe, bpp_delta)
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate 0 harness baseline reproducibility check (revised)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    try:
        main(verbose=args.verbose)
        sys.exit(0)
    except Exception as exc:
        logger.error("Gate 0 check failed: %s", exc, exc_info=True)
        sys.exit(1)
