"""
scripts/run_pead_walk_forward.py — Path C #1 PEAD walk-forward CLI orchestrator.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57) §4.1 + §八.

Pipeline orchestrator that wires together Sprint 2-5 modules into a single
reproducible command:

  1. Load top-N CRSP vintage S&P 500 universe (default top-200)
  2. Fetch I/B/E/S + Compustat + CRSP linkage → firm-quarter earnings panel
  3. Compute SUE + cross-section rank + decile-leg assignment
  4. Fetch CRSP daily price panel, compute pct_change → daily returns
  5. Walk-forward: aggregate daily long-short P&L + apply TC drag
  6. Build verdict.json + persist to data/path_c/v1_pead_10y_verdict.json
  7. Persist per-quarter checkpoint JSONL

Usage:
  py -3.11 scripts/run_pead_walk_forward.py
  py -3.11 scripts/run_pead_walk_forward.py --start 2014-01-01 --end 2023-12-31 \
                                            --top-n 200 --run-id v1_pead_10y
  py -3.11 scripts/run_pead_walk_forward.py --mock   # synthetic panel for smoke

Requires WRDS configured (~/.pgpass or APPDATA postgresql/pgpass.conf with
${WRDS_USER_1} credentials) unless --mock passed.

End-of-window drift coverage: caller's returns_panel auto-extends ≥60
trading days past `--end` to capture full hold windows for last-quarter
announcements (avoids rigor audit finding A truncation bias).
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

# Ensure repo root on sys.path when invoked as `py -3.11 scripts/...`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

logger = logging.getLogger("path_c.run_walk_forward")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pead_walk_forward.py",
        description="Path C #1 PEAD walk-forward orchestrator",
    )
    p.add_argument("--start", default="2014-01-01",
                   help="Window start date (YYYY-MM-DD; spec-locked to 2014-01-01)")
    p.add_argument("--end",   default="2023-12-31",
                   help="Window end date (YYYY-MM-DD; spec-locked to 2023-12-31)")
    p.add_argument("--top-n", type=int, default=200,
                   help="Universe top-N by market cap (spec-locked to 200)")
    p.add_argument("--run-id", default="v1_pead_10y",
                   help="Checkpoint run_id for resume + audit")
    p.add_argument("--mock", action="store_true",
                   help="Use mock panels (synthetic; no WRDS required) for smoke")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose logging")
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

    from engine.path_c.earnings_panel import bulk_fetch_earnings_panel, is_wrds_available
    from engine.path_c.sue_signal import build_sue_signal_panel
    from engine.path_c.pead_backtest import (
        run_walk_forward_pead, persist_walk_forward_result, HOLD_TRADING_DAYS_LOCKED,
    )
    from engine.path_c.verdict import build_pead_verdict, persist_verdict
    from engine.preregistration import (
        validate_reference, compute_pre_registration_n_trials,
    )

    SPEC_PATH = "docs/spec_path_c_earnings_pead_v1.md"

    # ── Step 0: pre-registration validation ────────────────────────────────
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s (run `py -3.11 -m engine.preregistration validate %s`)", reason, SPEC_PATH)
        return 2
    logger.info("Spec validated OK; reference recorded.")
    n_trials = compute_pre_registration_n_trials()
    logger.info("Project cumulative pre-registration n_trials = %d", n_trials)

    # Spec hash for verdict artifact
    from engine.preregistration import _compute_git_blob_hash, _resolve_to_abs
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))

    # ── Step 1: universe (top-N CRSP vintage SP500) ────────────────────────
    if mock_mode:
        # Synthetic top-N tickers for smoke testing
        tickers = [f"MOCK{i:03d}" for i in range(args.top_n)]
        logger.info("MOCK MODE: %d synthetic tickers", len(tickers))
    else:
        if not is_wrds_available():
            logger.error(
                "WRDS not configured. Pass --mock for synthetic smoke, or configure "
                "~/.pgpass (Unix) / APPDATA postgresql/pgpass.conf (Windows) with "
                "${WRDS_USER_1} credentials."
            )
            return 3
        # Real path: load CRSP-vintage SP500 constituents at sampling dates,
        # union (no top-N here — top-N filter applied per-quarter post-panel
        # using market_cap_at_q from earnings_panel; see Step 5 below).
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
            logger.info("Universe: %d unique CRSP-vintage tickers across %d sampling dates",
                        len(tickers), len(sampling_dates))
        except Exception as exc:
            logger.error("Universe loader failed: %s. Verify "
                         "engine.universe_singlename.constituents_loader CRSP "
                         "vintage path activated.", exc)
            return 4

    # ── Step 2: earnings panel ─────────────────────────────────────────────
    logger.info("Fetching earnings panel for %d tickers, [%s, %s]",
                len(tickers), start_date, end_date)
    panel_result = bulk_fetch_earnings_panel(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        mock_mode=mock_mode,
    )
    logger.info("Earnings panel: %d firm-quarters (mode=%s)",
                panel_result.n_firm_quarters, panel_result.mode)
    if panel_result.panel.empty:
        logger.error("Empty earnings panel — aborting.")
        return 5

    # ── Step 3: top-N filter per quarter (spec §2.1 + §六) ─────────────────
    # Apply BEFORE SUE ranking so that decile cuts are over the locked
    # top-N universe (not over all firms with I/B/E/S coverage).
    panel = panel_result.panel
    if "market_cap_at_q" in panel.columns and panel["market_cap_at_q"].notna().any():
        keep_indices = []
        for quarter, group in panel.groupby("fiscal_yearq"):
            top_n_idx = group.nlargest(args.top_n, "market_cap_at_q").index
            keep_indices.extend(top_n_idx)
        panel = panel.loc[keep_indices].reset_index(drop=True)
        logger.info("Top-N filter applied: %d firm-quarters retained (top %d per quarter)",
                    len(panel), args.top_n)
    else:
        logger.warning(
            "market_cap_at_q missing/all-NaN → top-N filter SKIPPED. Using full "
            "universe. Spec §2.1 hard-locks top-%d; deviation must be disclosed "
            "in verdict metadata.", args.top_n,
        )

    # ── Step 4: SUE signal + decile legs ───────────────────────────────────
    logger.info("Computing SUE + cross-section rank + decile-leg assignment")
    signal_panel = build_sue_signal_panel(panel)
    leg_counts = signal_panel["leg"].value_counts().to_dict()
    logger.info("Leg counts: %s", leg_counts)

    # ── Step 5: daily returns panel (extend end+90 calendar days for full hold) ──
    last_rdq = signal_panel[signal_panel["leg"].isin(["long", "short"])]["rdq"].max()
    if hasattr(last_rdq, "date"):
        last_rdq = last_rdq.date()
    # Extend ≥ 60 trading days ≈ 90 calendar days past last rdq to avoid
    # end-of-window drift truncation (rigor audit finding A 2026-05-12).
    returns_end = max(end_date, last_rdq + datetime.timedelta(days=90))
    logger.info("Fetching daily returns panel [%s, %s] (extended %d days "
                "past window_end for full drift coverage)",
                start_date, returns_end,
                (returns_end - end_date).days)

    if mock_mode:
        # Synthetic constant-return panel for smoke
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

    # ── Step 6: walk-forward ──────────────────────────────────────────────
    logger.info("Running walk-forward aggregation (hold=%d trading days)",
                HOLD_TRADING_DAYS_LOCKED)
    wf_result = run_walk_forward_pead(
        signal_panel=signal_panel,
        returns_panel=returns_panel,
        window_start=start_date,
        window_end=end_date,
        checkpoint_run_id=args.run_id,
        spec_hash_at_run=spec_hash,
    )

    if wf_result.daily_returns.empty:
        logger.error("Walk-forward produced empty daily_returns — aborting.")
        return 6

    persist_walk_forward_result(wf_result)
    logger.info(
        "Walk-forward: %d quarters, %d firm-quarters active, %d daily obs",
        wf_result.n_quarters_processed, wf_result.n_firm_quarters_active,
        len(wf_result.daily_returns),
    )

    # ── Step 7: verdict ────────────────────────────────────────────────────
    logger.info("Building verdict (Sharpe / NW t lag=60 / bootstrap CI / BHY-FDR)")
    verdict = build_pead_verdict(
        wf_result, signal_panel,
        spec_hash=spec_hash,
        effective_n_trials=n_trials,
    )

    verdict_path = REPO_ROOT / "data" / "path_c" / "v1_pead_10y_verdict.json"
    persist_verdict(verdict, verdict_path)
    logger.info(
        "Verdict: %s (Sharpe gross=%.3f net=%.3f / NW t=%.3f / BHY %s / "
        "CI [%.3f, %.3f])",
        verdict.decision, verdict.sharpe_gross, verdict.sharpe_net,
        verdict.nw_t_stat, "pass" if verdict.bhy_fdr_passes else "fail",
        verdict.bootstrap_ci_lower, verdict.bootstrap_ci_upper,
    )
    logger.info("Artifacts: %s", verdict_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
