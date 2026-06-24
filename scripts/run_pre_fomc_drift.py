"""
scripts/run_pre_fomc_drift.py — Path E Pre-FOMC Announcement Drift walk-forward CLI.

Pre-registration: docs/spec_path_e_pre_fomc_drift_v1.md (id=64) §八.

Pipeline:
  1. Validate spec hash
  2. Load K1 equity subset universe (~34 ETFs)
  3. Fetch yfinance daily price panel 2014-2023 (+90d buffer for safety)
  4. Compute pre-FOMC event returns (80 events)
  5. Build daily TS strategy returns
  6. Load K1 baseline daily TS for incremental α regression
  7. Build verdict with 5-gate post-audit eval
  8. Persist verdict + event_returns + daily_ts to data/path_e/

Usage:
  py -3.11 scripts/run_pre_fomc_drift.py --start 2014-01-01 --end 2023-12-31 --run-id v1_pre_fomc_10y
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger("path_e.run_pre_fomc")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run_pre_fomc_drift.py",
                                description="Path E Pre-FOMC Drift walk-forward")
    p.add_argument("--start",  default="2014-01-01")
    p.add_argument("--end",    default="2023-12-31")
    p.add_argument("--run-id", default="v1_pre_fomc_10y")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def fetch_yfinance_panel(tickers: list[str], start: datetime.date,
                        end: datetime.date) -> pd.DataFrame:
    """Bulk-fetch yfinance daily closes for tickers."""
    # yfinance returns multi-index DataFrame when multiple tickers
    end_extended = end + datetime.timedelta(days=10)  # buffer for non-trading days
    data = yf.download(
        tickers=" ".join(tickers),
        start=start.isoformat(),
        end=end_extended.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        raise RuntimeError("yfinance returned empty data")

    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        # single-ticker case
        closes = data[['Close']]
        closes.columns = [tickers[0]]

    # Reindex to keep only working tickers + sort
    closes = closes.dropna(axis=1, how='all')  # drop tickers with NO data
    closes = closes.sort_index()
    closes.index = pd.DatetimeIndex([d.normalize() if hasattr(d, 'normalize') else pd.Timestamp(d).normalize() for d in closes.index])
    return closes


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from engine.preregistration import (
        validate_reference, _compute_git_blob_hash, _resolve_to_abs,
    )
    from engine.universe_manager import get_active_universe
    from engine.path_e.fomc_calendar import (
        FOMC_EVENTS_2014_2023, get_fomc_events_in_window,
    )
    from engine.path_e.pre_fomc_signal import (
        compute_event_returns, build_daily_strategy_returns,
    )
    from engine.path_e.pre_fomc_backtest import build_pre_fomc_verdict

    SPEC_PATH = "docs/spec_path_e_pre_fomc_drift_v1.md"

    # Step 0: spec validation
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason)
        return 2
    logger.info("Spec validated.")
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))
    logger.info("Spec hash: %s", spec_hash)

    start_date = datetime.date.fromisoformat(args.start)
    end_date   = datetime.date.fromisoformat(args.end)

    # Step 1: Load K1 equity subset universe
    logger.info("Loading K1 equity subset universe...")
    universe_dict = get_active_universe(asset_classes=['equity_sector', 'equity_factor'])
    tickers = sorted(set(universe_dict.values()))
    logger.info("Universe: %d equity ETFs", len(tickers))
    logger.debug("Tickers: %s", tickers)

    # Step 2: Filter FOMC events to window
    fomc_events = get_fomc_events_in_window(start_date, end_date)
    logger.info("FOMC events in window [%s, %s]: %d", start_date, end_date, len(fomc_events))

    # Step 3: Fetch yfinance daily price panel
    logger.info("Fetching yfinance daily closes for %d tickers...", len(tickers))
    price_panel = fetch_yfinance_panel(tickers, start_date, end_date)
    logger.info("Price panel: %d daily obs × %d tickers (dropped %d empty)",
                price_panel.shape[0], price_panel.shape[1], len(tickers) - price_panel.shape[1])

    # Step 4: Compute event returns
    logger.info("Computing pre-FOMC event returns...")
    event_returns = compute_event_returns(fomc_events, price_panel)
    logger.info("Event returns: %d events computed", len(event_returns))
    if len(event_returns) == 0:
        logger.error("0 event returns computed — aborting")
        return 3

    # Step 5: Build daily TS strategy returns
    logger.info("Building daily TS strategy returns...")
    daily_strategy = build_daily_strategy_returns(event_returns, price_panel.index)
    logger.info("Daily TS: %d days (mostly 0s; %d non-zero events)",
                len(daily_strategy), int((daily_strategy['strategy_return'] != 0).sum()))

    # Step 6: Load K1 baseline daily TS for incremental α
    # K1 paired_returns parquet has weekly returns; convert to daily-aligned for regression
    k1_path = REPO_ROOT / "data/path_c_k1/v1_k1_size_expanded_paired_returns.parquet"
    if not k1_path.exists():
        logger.warning("K1 baseline parquet not found; incremental α skipped (will fail Gate 5)")
        k1_baseline_daily = pd.Series(dtype=float)
    else:
        k1_paired = pd.read_parquet(k1_path)
        # k1_weekly_returns is weekly; we need to broadcast to daily for regression
        # Simple approach: distribute weekly return evenly across 5 trading days
        # (approximation; better: actual daily would need re-running K1)
        n_weeks = len(k1_paired)
        weekly_returns = k1_paired['k1_weekly_returns'].values
        # Align to start of K1 window (2014-01-01)
        weekly_dates = pd.date_range(start='2014-01-06', periods=n_weeks, freq='W-MON')
        # For each week return, spread across 5 daily values: (1+r)^(1/5) - 1
        # Then align to trading days
        daily_index = price_panel.index
        k1_daily = pd.Series(0.0, index=daily_index)
        for i, wd in enumerate(weekly_dates):
            if i >= n_weeks:
                break
            week_r = weekly_returns[i]
            daily_eq = (1 + week_r) ** (1/5) - 1
            # Distribute across the 5 trading days starting at wd
            for offset in range(5):
                target = wd + pd.Timedelta(days=offset)
                # Snap to next trading day
                future_trading = daily_index[daily_index >= target]
                if len(future_trading) > 0:
                    k1_daily.loc[future_trading[0]] = daily_eq
        k1_baseline_daily = k1_daily
        logger.info("K1 baseline daily TS: %d obs (broadcast from %d weekly)",
                    int((k1_baseline_daily != 0).sum()), n_weeks)

    # Step 7: Build verdict
    logger.info("Building post-audit 5-gate verdict...")
    verdict = build_pre_fomc_verdict(
        event_returns=event_returns,
        daily_strategy=daily_strategy,
        baseline_daily=k1_baseline_daily,
        spec_hash=spec_hash,
        universe_source=f"k1_equity_subset_{len(price_panel.columns)}etfs",
        window_start=start_date,
        window_end=end_date,
    )

    # Step 8: Persist
    out_dir = REPO_ROOT / "data/path_e"
    out_dir.mkdir(parents=True, exist_ok=True)

    verdict_path = out_dir / f"{args.run_id}_verdict.json"
    from dataclasses import asdict
    verdict_dict = asdict(verdict)
    verdict_dict["run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    verdict_path.write_text(json.dumps(verdict_dict, indent=2, default=str), encoding="utf-8")

    event_returns_path = out_dir / f"{args.run_id}_event_returns.parquet"
    event_returns.drop(columns=["ticker_returns_json"], errors='ignore').to_parquet(event_returns_path)

    daily_path = out_dir / f"{args.run_id}_daily_ts.parquet"
    daily_strategy.to_parquet(daily_path)

    logger.info("=" * 70)
    logger.info("FINAL DECISION: %s", verdict.decision)
    logger.info("Method A (event-time):")
    logger.info("  Sharpe:      %.4f", verdict.method_A_sharpe_net or 0)
    logger.info("  NW t:        %.4f", verdict.method_A_nw_t or 0)
    logger.info("  CI 95%%:      [%.4f, %.4f]", verdict.method_A_ci_lower or 0,
                verdict.method_A_ci_upper or 0)
    logger.info("  Ann return:  %.4f (= %.2f%% per year)", verdict.method_A_ann_return or 0,
                (verdict.method_A_ann_return or 0) * 100)
    logger.info("Method B (daily TS):")
    logger.info("  Sharpe:      %.4f", verdict.method_B_sharpe_net or 0)
    logger.info("  NW t:        %.4f", verdict.method_B_nw_t or 0)
    logger.info("5-Gate summary:")
    logger.info("  Gate 1 (individual):       %s", "PASS" if verdict.gate_1_individual_pass else "FAIL")
    logger.info("  Gate 2 (Selective BHY):    %s", verdict.gate_2_selective_bhy)
    logger.info("  Gate 3 (OOS hold-out):     %s", "PASS" if verdict.gate_3_oos_pass else "FAIL")
    logger.info("  Gate 4 (Sub-period dual):  %s", "PASS" if verdict.gate_4_subperiod_pass else "FAIL")
    logger.info("  Gate 5 (Incremental α):    %s", "PASS" if verdict.gate_5_incremental_pass else "FAIL")
    logger.info("Cumulative 10y: %.4f", verdict.cumulative_return)
    logger.info("Max DD: %.4f", verdict.max_drawdown)
    logger.info("Artifacts: %s", verdict_path)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
