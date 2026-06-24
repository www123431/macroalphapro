"""
scripts/run_dual_track_snapshot.py — Rebalance-day dual-track snapshot.

Pre-registration: docs/spec_etf_holdings_llm_risk_monitor.md (id=49)
Spec section: §2.9 Counterfactual Tracking — at each rebalance, persist BOTH
Track A (caps applied) and Track B (caps disabled) portfolio weights as a
snapshot for downstream daily P&L delta computation.

Usage
-----
Invoked by cron on rebalance days (typically last business day of month) AFTER
the monthly LLM screening run (run_etf_holdings_monitor_monthly.py) so that
cap_state.json reflects current month's caps before snapshot is taken.

  python scripts/run_dual_track_snapshot.py
  python scripts/run_dual_track_snapshot.py --as-of 2026-05-30
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

logger = logging.getLogger("dual_track_snapshot")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(as_of: Optional[datetime.date] = None, *, verbose: bool = False) -> dict:
    _setup_logging(verbose)
    if as_of is None:
        as_of = datetime.date.today()

    logger.info("=" * 70)
    logger.info("Dual-track snapshot as_of=%s (spec id=49 §2.9)", as_of)
    logger.info("=" * 70)

    # Build signal + regime
    try:
        from engine.signal import get_signal_dataframe
        from engine.regime import get_regime_on
    except Exception as exc:
        logger.error("Dependency import failed: %s", exc)
        return {"status": "import_failed", "error": str(exc)}

    logger.info("Step 1: build signal_df ...")
    try:
        signal_df = get_signal_dataframe(as_of=as_of)
        if signal_df is None or len(signal_df) == 0:
            logger.warning("signal_df is empty — skipping snapshot")
            return {"status": "empty_signal", "as_of": as_of.isoformat()}
        logger.info("  → signal_df shape: %s", signal_df.shape)
    except Exception as exc:
        logger.error("signal_df build failed: %s", exc)
        return {"status": "signal_failed", "error": str(exc)}

    logger.info("Step 2: get regime ...")
    try:
        regime = get_regime_on(as_of)
        logger.info("  → regime: %s p_risk_on=%.3f",
                    getattr(regime, "regime", "?"),
                    getattr(regime, "p_risk_on", 0))
    except Exception as exc:
        logger.warning("regime fetch failed (continuing with None): %s", exc)
        regime = None

    # Compute dual-track snapshot
    logger.info("Step 3: compute dual-track snapshot (Track A caps applied, "
                "Track B caps disabled) ...")
    from engine.etf_holdings_counterfactual import (
        compute_dual_track_snapshot,
        persist_dual_track_snapshot,
    )

    snapshot = compute_dual_track_snapshot(
        as_of=as_of,
        signal_df=signal_df,
        regime=regime,
    )
    if snapshot.get("status") != "ok":
        logger.error("Snapshot computation failed: %s", snapshot)
        return snapshot

    logger.info("  → Track A: %d positions, Track B: %d positions, %d ETFs differ (capped: %s)",
                len(snapshot["track_a_weights"]),
                len(snapshot["track_b_weights"]),
                snapshot["n_capped"],
                ", ".join(snapshot["capped_etfs"]) or "none")

    # Persist
    logger.info("Step 4: persist snapshot to dual_track_snapshots.parquet ...")
    persisted = persist_dual_track_snapshot(snapshot)
    if persisted:
        logger.info("  → Snapshot persisted")
    else:
        logger.warning("  → Persist failed")

    summary = {
        "status":           "ok",
        "as_of":            as_of.isoformat(),
        "spec_id":          49,
        "n_track_a":        len(snapshot["track_a_weights"]),
        "n_track_b":        len(snapshot["track_b_weights"]),
        "n_capped":         snapshot["n_capped"],
        "capped_etfs":      snapshot["capped_etfs"],
        "persisted":        persisted,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-track snapshot for ETF Holdings counterfactual")
    parser.add_argument("--as-of", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    as_of_arg = (
        datetime.date.fromisoformat(args.as_of) if args.as_of else None
    )
    try:
        main(as_of_arg, verbose=args.verbose)
        sys.exit(0)
    except Exception as exc:
        logger.error("Dual-track snapshot failed: %s", exc, exc_info=True)
        sys.exit(1)
