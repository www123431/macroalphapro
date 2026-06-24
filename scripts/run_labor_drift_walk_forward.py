"""
scripts/run_labor_drift_walk_forward.py — Path C Labor Signal Drift CLI orchestrator.

Pre-registration: docs/spec_path_c_labor_signal_drift_v1.md (id=58) §4.1 + §八.

Pipeline (mirrors scripts/run_pead_walk_forward.py with labor signal pipeline
substituted; backtest + verdict modules 100% reused):

  1. Load top-N CRSP vintage S&P 500 universe (default top-200)
  2. Fetch Revelio + Compustat + CRSP linkage → firm-quarter labor-signal panel
  3. Compute labor_signal + cross-section rank + decile-leg assignment
  4. Fetch CRSP daily price panel, compute pct_change → daily returns
  5. Walk-forward: aggregate daily long-short P&L + apply TC drag (REUSED)
  6. Build verdict.json + persist (REUSED with bhy demoted gates)
  7. Persist per-quarter checkpoint JSONL (REUSED)

Usage:
  py -3.11 scripts/run_labor_drift_walk_forward.py
  py -3.11 scripts/run_labor_drift_walk_forward.py --start 2014-01-01 --end 2023-12-31 \
                                                   --top-n 200 --run-id v1_labor_10y
  py -3.11 scripts/run_labor_drift_walk_forward.py --mock --top-n 20 --start 2014-01-01 --end 2015-12-31
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

logger = logging.getLogger("path_c.run_labor_drift")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_labor_drift_walk_forward.py",
        description="Path C Labor Signal Drift walk-forward orchestrator",
    )
    p.add_argument("--start", default="2014-01-01",
                   help="Window start (spec-locked 2014-01-01)")
    p.add_argument("--end",   default="2023-12-31",
                   help="Window end (spec-locked 2023-12-31)")
    p.add_argument("--top-n", type=int, default=200,
                   help="Universe top-N by market cap (spec-locked 200)")
    p.add_argument("--run-id", default="v1_labor_drift_10y",
                   help="Checkpoint run_id")
    p.add_argument("--mock", action="store_true",
                   help="Mock panels (no WRDS) for smoke test")
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

    from engine.path_c.labor_signal_panel import (
        bulk_fetch_labor_signal_panel,
        is_wrds_available,
    )
    from engine.path_c.labor_signal import build_labor_signal_panel
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

    SPEC_PATH = "docs/spec_path_c_labor_signal_drift_v1.md"

    # Step 0: pre-registration validation
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason)
        return 2
    logger.info("Spec validated; reference recorded.")
    n_trials = compute_pre_registration_n_trials()
    logger.info("Project cumulative pre-registration n_trials = %d (audit only)", n_trials)
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))

    # Step 1: universe
    if mock_mode:
        tickers = [f"MOCK{i:03d}" for i in range(args.top_n)]
        logger.info("MOCK MODE: %d synthetic tickers", len(tickers))
    else:
        if not is_wrds_available():
            logger.error("WRDS not configured. Pass --mock for synthetic smoke.")
            return 3
        from engine.universe_singlename.constituents_loader import (
            load_sp500_constituents_at_date,
        )
        try:
            sampling_dates = [
                start_date,
                datetime.date(2018, 6, 30),
                end_date,
            ]
            tickers_set: set = set()
            for s_date in sampling_dates:
                r = load_sp500_constituents_at_date(as_of=s_date, source="crsp_vintage")
                tickers_set.update(r.tickers)
            tickers = sorted(tickers_set)
            logger.info("Universe: %d CRSP-vintage tickers", len(tickers))
        except Exception as exc:
            logger.error("Universe loader failed: %s", exc)
            return 4

    # Step 2: labor signal panel (Revelio + Compustat + CRSP linkage)
    logger.info("Fetching labor signal panel for %d tickers, [%s, %s]",
                len(tickers), start_date, end_date)
    panel_result = bulk_fetch_labor_signal_panel(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        mock_mode=mock_mode,
    )
    logger.info("Labor panel: %d firm-quarters (mode=%s)",
                panel_result.n_firm_quarters, panel_result.mode)
    if panel_result.panel.empty:
        logger.error("Empty labor panel — aborting.")
        return 5

    # Step 3: top-N filter per quarter (spec §2.1 hard-locked top-200)
    panel = panel_result.panel
    if "market_cap_at_q" in panel.columns and panel["market_cap_at_q"].notna().any():
        keep_indices = []
        for quarter, group in panel.groupby("fiscal_yearq"):
            top_n_idx = group.nlargest(args.top_n, "market_cap_at_q").index
            keep_indices.extend(top_n_idx)
        panel = panel.loc[keep_indices].reset_index(drop=True)
        logger.info("Top-N filter: %d firm-quarters retained (top %d per quarter)",
                    len(panel), args.top_n)
    else:
        logger.warning(
            "market_cap_at_q missing → top-N filter SKIPPED. Spec §2.1 hard-locks top-%d; "
            "disclose deviation in verdict.", args.top_n,
        )

    # Step 4: labor signal + decile-leg assignment
    logger.info("Computing labor signal + cross-section rank + decile-leg")
    signal_panel = build_labor_signal_panel(panel)
    leg_counts = signal_panel["leg"].value_counts().to_dict()
    logger.info("Leg counts: %s", leg_counts)

    # Step 5: daily returns panel (extend window_end + 90 calendar days for full hold)
    last_rdq = signal_panel[signal_panel["leg"].isin(["long", "short"])]["rdq"].max()
    if hasattr(last_rdq, "date"):
        last_rdq = last_rdq.date()
    returns_end = max(end_date, last_rdq + datetime.timedelta(days=90)) if last_rdq else end_date
    logger.info("Fetching daily returns [%s, %s] (extended %d days past window_end)",
                start_date, returns_end,
                (returns_end - end_date).days)

    if mock_mode:
        import numpy as np
        bdates = pd.bdate_range(start=start_date, end=returns_end)
        rng = np.random.default_rng(42)
        cols = {t: rng.normal(0.0001, 0.01, size=len(bdates)) for t in tickers}
        returns_panel = pd.DataFrame(cols, index=bdates)
        returns_panel.index.name = "date"
    else:
        from engine.universe_singlename.crsp_loader import bulk_fetch_crsp_daily_panel
        price_panel = bulk_fetch_crsp_daily_panel(
            tickers=tickers, start_date=start_date, end_date=returns_end,
            mock_mode=False,
        )
        returns_panel = price_panel.pct_change().dropna(how="all")

    logger.info("Returns panel: %d daily obs × %d tickers", *returns_panel.shape)

    # Step 6: walk-forward (REUSE pead_backtest)
    # Rename ticker → ticker_ibes for pead_backtest compatibility (ticker_col default)
    signal_panel_for_backtest = signal_panel.rename(columns={"ticker": "ticker_ibes"})

    logger.info("Walk-forward aggregation (hold=%d trading days)",
                HOLD_TRADING_DAYS_LOCKED)
    wf_result = run_walk_forward_pead(
        signal_panel=signal_panel_for_backtest,
        returns_panel=returns_panel,
        window_start=start_date,
        window_end=end_date,
        checkpoint_run_id=args.run_id,
        spec_hash_at_run=spec_hash,
    )

    if wf_result.daily_returns.empty:
        logger.error("Walk-forward produced empty daily_returns — aborting.")
        return 6

    # Use labor-specific output dir + parquet path
    labor_data_dir = REPO_ROOT / "data" / "path_c_labor"
    labor_data_dir.mkdir(parents=True, exist_ok=True)
    labor_parquet = labor_data_dir / "walk_forward_labor.parquet"
    persist_walk_forward_result(wf_result, parquet_path=labor_parquet)

    logger.info("Walk-forward: %d quarters, %d firm-quarters active, %d daily obs",
                wf_result.n_quarters_processed, wf_result.n_firm_quarters_active,
                len(wf_result.daily_returns))

    # Step 7: verdict (REUSE)
    logger.info("Building verdict (Sharpe / NW t lag=60 / industry single-test gates, BHY reporting)")
    verdict = build_pead_verdict(
        wf_result, signal_panel_for_backtest,
        spec_hash=spec_hash,
        spec_path=SPEC_PATH,
        effective_n_trials=n_trials,
        wave="C-labor",
        universe_source="crsp_vintage_top200_revelio_linked",
    )

    verdict_path = labor_data_dir / "v1_labor_drift_10y_verdict.json"
    persist_verdict(verdict, verdict_path)
    logger.info(
        "Verdict: %s (Sharpe gross=%.3f net=%.3f / NW t=%.3f / BHY %s reporting / CI [%.3f, %.3f])",
        verdict.decision, verdict.sharpe_gross, verdict.sharpe_net,
        verdict.nw_t_stat,
        "pass" if verdict.bhy_fdr_passes else "fail",
        verdict.bootstrap_ci_lower, verdict.bootstrap_ci_upper,
    )
    logger.info("Artifacts: %s", verdict_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
