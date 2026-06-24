"""
scripts/run_factor_mining_lab.py — Tier 1 single-stock factor mining lab CLI.

Tier 1 mining lab parallel to Tier 2 ETF candidate gate:
  Tier 2 (existing): scripts/run_b_plus_search_*.py + factor_lab.runner
  Tier 1 (this):     this script + factor_lab.mining_runner

Per architecture lock (memory project_final_vision_hybrid_2026-05-10):
  - factor_kind="infrastructure_spec" (P-LAB exempt → +0 EFFECTIVE_N_TRIALS)
  - Verdict raw (NOT BHY-corrected); promotion to Tier 2 requires manual
    review + re-registration as research_hypothesis spec
  - Universe: vintage S&P 500 (Wave A retail proxy or Wave B CRSP)
  - 0 LLM imports (deterministic mining)

Usage:
  # Default: all registered Tier 1 candidates on Wave A retail (5y window)
  python scripts/run_factor_mining_lab.py

  # Single factor, custom window
  python scripts/run_factor_mining_lab.py --factor-id ivol_singlestock \
      --start-date 2018-01-01 --end-date 2023-12-31

  # Wave B CRSP universe (post-WRDS approval; mock fallback if unavailable)
  python scripts/run_factor_mining_lab.py --universe-source crsp_vintage

Output:
  data/factor_mining_lab/<factor_id>_<YYYY-MM-DD>.json   — session record
  docs/decisions/factor_mining_<factor_id>_<YYYY-MM-DD>.md — verdict markdown
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("factor_mining_lab")


# ── Default window: trailing 5 years ending 1 month ago ────────────────────
def _default_end_date() -> datetime.date:
    """End 1 month ago (give universe + panel time to settle)."""
    today = datetime.date.today()
    if today.month == 1:
        return datetime.date(today.year - 1, 12, 31)
    last_day_prev = (today.replace(day=1) - datetime.timedelta(days=1))
    return last_day_prev


def _default_start_date(end: datetime.date) -> datetime.date:
    """5 years before end (Tier 1 mining: ~60 monthly observations enough for
    raw NW t-stat ≥ 1.65 to mean directional, ≥ 2.5 to mean promotable)."""
    try:
        return end.replace(year=end.year - 5)
    except ValueError:    # Feb 29
        return end.replace(year=end.year - 5, day=28)


def _resolve_factor_ids(requested: list[str]) -> list[str]:
    """Resolve --factor-id list → canonical Tier 1 factor IDs.

    Special token 'all' expands to all registered factors.
    """
    from engine.factor_library_singlename import list_factors

    available = list_factors()
    if not requested or "all" in requested:
        return available

    out: list[str] = []
    for fid in requested:
        if fid not in available:
            raise SystemExit(
                f"factor-id {fid!r} not registered. Available: {available}. "
                f"Add to engine/factors_singlename/<name>.py + register_factor()."
            )
        if fid not in out:
            out.append(fid)
    return out


def main(
    factor_ids:       list[str],
    start_date:       datetime.date,
    end_date:         datetime.date,
    universe_source:  str,
    persist:          bool,
    verbose:          bool,
) -> list[dict]:
    """Run Tier 1 mining for each factor; return summary list."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from engine.factor_ensemble_singlename.panel_fetcher import (
        bulk_fetch_singlestock_panel,
    )
    from engine.factor_lab.mining_runner import run_mining_session
    from engine.universe_singlename import load_sp500_constituents_at_date

    resolved_factors = _resolve_factor_ids(factor_ids)

    logger.info("=" * 72)
    logger.info("Tier 1 Factor Mining Lab — single-stock walk-forward")
    logger.info("Factors:         %s", ", ".join(resolved_factors))
    logger.info("Window:          %s → %s", start_date, end_date)
    logger.info("Universe source: %s", universe_source)
    logger.info("Persist:         %s", persist)
    logger.info("=" * 72)

    # Universe + panel are shared across all factors in this session
    def universe_at_date_fn(d: datetime.date) -> list:
        return load_sp500_constituents_at_date(
            as_of=d, source=universe_source,
        ).tickers

    logger.info("Sampling universe at end_date to size panel fetch …")
    end_universe = universe_at_date_fn(end_date)
    if not end_universe:
        raise SystemExit(
            f"universe empty at {end_date} — universe_source {universe_source!r} "
            f"may be misconfigured (e.g. crsp_vintage without WRDS approval)."
        )

    # Always include SPY for IVOL CAPM benchmark
    panel_tickers = sorted(set(end_universe + ["SPY"]))
    logger.info("Fetching daily panel for %d tickers ...", len(panel_tickers))
    panel = bulk_fetch_singlestock_panel(
        tickers=panel_tickers,
        start_date=start_date,
        end_date=end_date,
    )
    if panel.empty:
        raise SystemExit(
            "panel fetch returned empty DataFrame. yfinance may be rate-limited "
            "or offline. Retry later."
        )
    logger.info("Panel: %d dates × %d tickers", panel.shape[0], panel.shape[1])

    # Run mining for each factor
    summaries: list[dict] = []
    for factor_id in resolved_factors:
        logger.info("-" * 72)
        logger.info(">>> Mining: %s", factor_id)
        try:
            result = run_mining_session(
                factor_id=factor_id,
                universe_at_date_fn=universe_at_date_fn,
                panel=panel,
                start_date=start_date,
                end_date=end_date,
                persist_artifacts=persist,
            )
        except Exception as exc:
            logger.error("Mining failed for %s: %s", factor_id, exc, exc_info=True)
            summaries.append({
                "factor_id": factor_id,
                "verdict":   "execution_error",
                "error":     f"{type(exc).__name__}: {exc}",
            })
            continue

        summary = {
            "factor_id":             result.factor_id,
            "verdict":               result.verdict,
            "n_periods":             result.n_periods,
            "annualized_sharpe_net": round(result.annualized_sharpe_net, 4),
            "annualized_vol_net":    round(result.annualized_vol_net, 4),
            "cumulative_return_net": round(result.cumulative_return_net, 4),
            "nw_t_stat_net":         round(result.nw_t_stat_net, 3),
            "sign_match":            result.sign_match,
            "expected_sign":         result.expected_sign,
            "mean_n_active":         round(result.mean_n_active, 1),
        }
        summaries.append(summary)
        logger.info(
            ">>> %s: verdict=%s | Sharpe=%+.3f | NW t=%+.3f | n=%d",
            factor_id, result.verdict, result.annualized_sharpe_net,
            result.nw_t_stat_net, result.n_periods,
        )

    logger.info("=" * 72)
    logger.info("Tier 1 mining session complete — %d factor(s) processed", len(summaries))
    logger.info("=" * 72)

    print(json.dumps({
        "session_summary": {
            "start_date":      start_date.isoformat(),
            "end_date":        end_date.isoformat(),
            "universe_source": universe_source,
            "factors":         resolved_factors,
        },
        "results": summaries,
    }, ensure_ascii=False, indent=2, default=str))

    return summaries


if __name__ == "__main__":
    default_end   = _default_end_date()
    default_start = _default_start_date(default_end)

    parser = argparse.ArgumentParser(
        description=(
            "Tier 1 single-stock factor mining lab CLI. Tier 1 verdicts are "
            "raw (NOT BHY-corrected); promotion to Tier 2 requires manual "
            "review + re-registration."
        ),
    )
    parser.add_argument(
        "--factor-id", action="append", default=[],
        help=(
            "Tier 1 factor ID to mine (repeatable). Special: 'all' (default) "
            "= every registered factor. Examples: --factor-id ivol_singlestock "
            "--factor-id strev_singlestock"
        ),
    )
    parser.add_argument(
        "--start-date", default=default_start.isoformat(),
        help=f"walk-forward start (default: {default_start})",
    )
    parser.add_argument(
        "--end-date", default=default_end.isoformat(),
        help=f"walk-forward end (default: {default_end})",
    )
    parser.add_argument(
        "--universe-source",
        choices=["mktcap_top500_proxy", "wikipedia_archive", "crsp_vintage"],
        default="mktcap_top500_proxy",
        help=(
            "vintage universe source. proxy=Wave A primary (survivorship-biased "
            "lower bound). wikipedia=Wave A robustness. crsp_vintage=Wave B "
            "(WRDS-pending; falls back to proxy if unavailable)."
        ),
    )
    parser.add_argument(
        "--no-persist", dest="persist", action="store_false", default=True,
        help="skip writing JSON + verdict markdown (CI / dry-run mode)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    try:
        start = datetime.date.fromisoformat(args.start_date)
        end   = datetime.date.fromisoformat(args.end_date)
    except ValueError as exc:
        raise SystemExit(f"invalid date format (expected YYYY-MM-DD): {exc}")
    if start >= end:
        raise SystemExit(f"start ({start}) must be < end ({end})")

    try:
        main(
            factor_ids      = args.factor_id,
            start_date      = start,
            end_date        = end,
            universe_source = args.universe_source,
            persist         = args.persist,
            verbose         = args.verbose,
        )
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Tier 1 mining lab failed: %s", exc, exc_info=True)
        sys.exit(1)
