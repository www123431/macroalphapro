"""
scripts/run_asset_growth_walk_forward.py — Path J Asset Growth CLI orchestrator.

Pre-registration: docs/spec_path_j_asset_growth_drift_v1.md (id=60) §4.1 + §八.

Pipeline:
  1. Validate spec id=60
  2. Load Russell-2000-proxy universe (CRSP rank 1001-3000 at 3 sampling dates, union)
  3. Fetch asset_growth_signal_panel (Compustat fundq atq + CRSP linkage)
  4. NO additional top-N filter (universe IS already rank 1001-3000)
  5. Compute asset_growth_signal + decile-leg
  6. Fetch CRSP daily returns (auto-extended +90d past last rdq)
  7. Walk-forward (REUSE pead_backtest)
  8. Build verdict with wave="C-assetgrowth"

Usage:
  py -3.11 scripts/run_asset_growth_walk_forward.py
  py -3.11 scripts/run_asset_growth_walk_forward.py --start 2014-01-01 --end 2023-12-31 \
      --rank-min 1001 --rank-max 3000 --run-id v1_asset_growth_russell2000_10y
  py -3.11 scripts/run_asset_growth_walk_forward.py --mock --rank-max 50 --start 2014-01-01 --end 2015-12-31
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

logger = logging.getLogger("path_c.run_asset_growth")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_asset_growth_walk_forward.py",
        description="Path J Asset Growth Russell-2000-proxy walk-forward orchestrator",
    )
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end",   default="2023-12-31")
    p.add_argument("--rank-min", type=int, default=1001)
    p.add_argument("--rank-max", type=int, default=3000)
    p.add_argument("--run-id", default="v1_asset_growth_russell2000_10y")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    start_date = datetime.date.fromisoformat(args.start)
    end_date   = datetime.date.fromisoformat(args.end)
    mock_mode  = args.mock

    from engine.path_c.asset_growth_signal_panel import (
        bulk_fetch_asset_growth_signal_panel,
        is_wrds_available,
    )
    from engine.path_c.asset_growth_signal import build_asset_growth_signal_panel
    from engine.path_c.pead_backtest import (
        run_walk_forward_pead,
        persist_walk_forward_result,
        HOLD_TRADING_DAYS_LOCKED,
    )
    from engine.path_c.verdict import build_pead_verdict, persist_verdict
    from engine.preregistration import (
        validate_reference, compute_pre_registration_n_trials,
        _compute_git_blob_hash, _resolve_to_abs,
    )

    SPEC_PATH = "docs/spec_path_j_asset_growth_drift_v1.md"

    # Step 0: pre-reg validation
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason)
        return 2
    logger.info("Spec validated.")
    n_trials = compute_pre_registration_n_trials()
    logger.info("Project cumulative n_trials = %d (audit only)", n_trials)
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))

    # Step 1: Russell-2000-proxy universe
    if mock_mode:
        # In mock mode, n_synthetic = rank_max if rank_min default, else range size
        n_synthetic = max(1, args.rank_max - args.rank_min + 1) if args.rank_max > args.rank_min else max(20, args.rank_max)
        tickers = [f"MOCK{i:03d}" for i in range(n_synthetic)]
        logger.info("MOCK MODE: %d synthetic tickers", len(tickers))
    else:
        if not is_wrds_available():
            logger.error("WRDS not configured. Pass --mock for synthetic smoke.")
            return 3
        from engine.universe_singlename.constituents_loader import (
            load_russell2000_proxy_at_date,
        )
        sampling_dates = [start_date, datetime.date(2018, 6, 30), end_date]
        try:
            tickers_set: set = set()
            for s_date in sampling_dates:
                r = load_russell2000_proxy_at_date(
                    as_of=s_date, rank_min=args.rank_min, rank_max=args.rank_max,
                )
                tickers_set.update(r.tickers)
                logger.info("  %s: %d firms in rank %d-%d",
                            s_date, r.n_constituents, args.rank_min, args.rank_max)
            tickers = sorted(tickers_set)
            logger.info("Russell-2000-proxy universe union: %d unique tickers", len(tickers))
        except Exception as exc:
            logger.error("Russell 2000 loader failed: %s", exc)
            return 4

    # Step 2: asset growth panel
    logger.info("Fetching asset_growth_signal_panel for %d tickers, [%s, %s]",
                len(tickers), start_date, end_date)
    panel_result = bulk_fetch_asset_growth_signal_panel(
        tickers=tickers, start_date=start_date, end_date=end_date,
        mock_mode=mock_mode,
    )
    logger.info("Panel: %d firm-quarters (mode=%s)",
                panel_result.n_firm_quarters, panel_result.mode)
    if panel_result.panel.empty:
        logger.error("Empty panel — aborting.")
        return 5

    # NOTE: NO top-N filter here. The Russell-2000-proxy universe IS already
    # rank 1001-3000 by mkt cap; further filtering would be redundant.
    panel = panel_result.panel

    # Step 3: asset growth signal + decile-leg
    logger.info("Computing asset_growth_signal + decile-leg")
    signal_panel = build_asset_growth_signal_panel(panel)
    leg_counts = signal_panel["leg"].value_counts().to_dict()
    logger.info("Leg counts: %s", leg_counts)

    # Step 4: CRSP daily returns (extend +90d past last rdq)
    last_rdq = signal_panel[signal_panel["leg"].isin(["long", "short"])]["rdq"].max()
    if hasattr(last_rdq, "date"):
        last_rdq = last_rdq.date()
    returns_end = max(end_date, last_rdq + datetime.timedelta(days=90)) if last_rdq else end_date
    logger.info("Fetching daily returns [%s, %s] (extended %d days past window_end)",
                start_date, returns_end, (returns_end - end_date).days)

    if mock_mode:
        import numpy as np
        bdates = pd.bdate_range(start=start_date, end=returns_end)
        rng = np.random.default_rng(42)
        cols = {t: rng.normal(0.0001, 0.015, size=len(bdates)) for t in tickers}
        # Small-cap higher vol (0.015 vs SP500 0.01)
        returns_panel = pd.DataFrame(cols, index=bdates)
        returns_panel.index.name = "date"
    else:
        from engine.universe_singlename.crsp_loader import bulk_fetch_crsp_daily_panel
        price_panel = bulk_fetch_crsp_daily_panel(
            tickers=tickers, start_date=start_date, end_date=returns_end,
            mock_mode=False,
        )
        returns_panel = price_panel.pct_change(fill_method=None).dropna(how="all")

    logger.info("Returns panel: %d daily obs × %d tickers", *returns_panel.shape)

    # Step 5: walk-forward (REUSE pead_backtest; rename ticker → ticker_ibes)
    signal_panel_for_backtest = signal_panel.rename(columns={"ticker": "ticker_ibes"})
    logger.info("Walk-forward (hold=%d trading days)", HOLD_TRADING_DAYS_LOCKED)
    wf_result = run_walk_forward_pead(
        signal_panel=signal_panel_for_backtest,
        returns_panel=returns_panel,
        window_start=start_date, window_end=end_date,
        checkpoint_run_id=args.run_id,
        spec_hash_at_run=spec_hash,
    )

    if wf_result.daily_returns.empty:
        logger.error("Walk-forward produced empty daily_returns — aborting.")
        return 6

    ag_data_dir = REPO_ROOT / "data" / "path_c_asset_growth"
    ag_data_dir.mkdir(parents=True, exist_ok=True)
    ag_parquet = ag_data_dir / "walk_forward_asset_growth.parquet"
    persist_walk_forward_result(wf_result, parquet_path=ag_parquet)

    logger.info("Walk-forward: %d quarters, %d firm-quarters active, %d daily obs",
                wf_result.n_quarters_processed, wf_result.n_firm_quarters_active,
                len(wf_result.daily_returns))

    # Step 6: verdict
    logger.info("Building verdict")
    verdict = build_pead_verdict(
        wf_result, signal_panel_for_backtest,
        spec_hash=spec_hash, spec_path=SPEC_PATH,
        effective_n_trials=n_trials,
        wave="C-assetgrowth",
        universe_source=f"russell2000_proxy_crsp_rank_{args.rank_min}_{args.rank_max}",
    )

    verdict_path = ag_data_dir / "v1_asset_growth_10y_verdict.json"
    persist_verdict(verdict, verdict_path)
    logger.info(
        "Verdict: %s (Sharpe gross=%.3f net=%.3f / NW t=%.3f / BHY %s / CI [%.3f, %.3f])",
        verdict.decision, verdict.sharpe_gross, verdict.sharpe_net,
        verdict.nw_t_stat,
        "pass" if verdict.bhy_fdr_passes else "fail",
        verdict.bootstrap_ci_lower, verdict.bootstrap_ci_upper,
    )
    logger.info("Artifacts: %s", verdict_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
