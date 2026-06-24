"""
scripts/run_etf_holdings_monitor_monthly.py — Monthly ETF Holdings LLM Risk Monitor.

Pre-registration: docs/spec_etf_holdings_llm_risk_monitor.md (id=49, hash 0c3696fc4145)

Purpose
-------
Monthly orchestration entry point for ETF Holdings Risk Monitor (Sprint Week 4
deliverable). Called by cron on the last business day of each month
(rebalance day). Performs:

  1. Fetch top 10 holdings for 24 equity ETFs via engine.etf_holdings_ingestion
  2. Deduplicate to unique names (~120-211 per current universe)
  3. Per-name LLM screen with ex-ante context (recent SEC 8-K + news +
     30d return + next earnings + sector)
  4. Per-ETF aggregation → cap trigger detection
  5. Persist cap state (severity-priority) + DecisionLog
  6. Print summary report

Resilience: never crashes — partial failures logged + continue with
reduced coverage (caller can retry).

Usage
-----
  python scripts/run_etf_holdings_monitor_monthly.py
  python scripts/run_etf_holdings_monitor_monthly.py --as-of 2026-05-31
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from typing import Optional

logger = logging.getLogger("etf_holdings_monitor")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _gather_inputs_for_name(
    name:        str,
    as_of:       datetime.date,
    news_pool:   list[dict],
) -> dict:
    """
    Gather ex-ante context for per-name LLM screening.

    Returns dict with keys: sector, recent_8k_filings, recent_news,
    price_30d_return, next_earnings_date.

    Resilience: each fetch wrapped in try/except, returns None on failure.
    """
    out = {
        "sector":              None,
        "recent_8k_filings":   [],
        "recent_news":         [],
        "price_30d_return":    None,
        "next_earnings_date":  None,
    }

    # 1. Sector + 30d return + next earnings via yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(name)
        info = t.info or {}
        out["sector"] = info.get("sector")

        hist = t.history(period="60d")
        if not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if len(closes) >= 22:
                out["price_30d_return"] = float(
                    closes.iloc[-1] / closes.iloc[-22] - 1
                )

        cal = getattr(t, "calendar", None)
        if cal is not None and len(cal) > 0:
            try:
                # yfinance calendar shape varies; try DataFrame + dict
                if hasattr(cal, "iloc"):
                    earnings_dates = cal.iloc[0].get("Earnings Date")
                    if earnings_dates:
                        ed = (
                            earnings_dates[0]
                            if isinstance(earnings_dates, (list, tuple))
                            else earnings_dates
                        )
                        if hasattr(ed, "date"):
                            out["next_earnings_date"] = ed.date()
                elif isinstance(cal, dict):
                    eds = cal.get("Earnings Date") or []
                    if eds and hasattr(eds[0], "date"):
                        out["next_earnings_date"] = eds[0].date()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("ehrm: yfinance info fetch failed for %s: %s", name, exc)

    # 2. Recent news from pre-fetched news_pool (filter by ticker)
    try:
        upper = name.upper()
        matched = [
            n.to_dict() if hasattr(n, "to_dict") else n
            for n in news_pool
            if upper in (
                [t.upper() for t in (n.get("tickers", []) if isinstance(n, dict) else getattr(n, "tickers", []))]
                if (isinstance(n, dict) or hasattr(n, "tickers"))
                else []
            )
        ]
        out["recent_news"] = matched[:8]  # cap at 8 per name (per spec build_prompt)
    except Exception as exc:
        logger.debug("ehrm: news filter failed for %s: %s", name, exc)

    # 3. SEC 8-K filings — defer for v1 MVP (would require EDGAR scraper integration);
    # honest disclosure: spec §一 acknowledges fundamental data partial coverage in free-data
    # path. EDGAR 8-K integration is a v1+ enhancement candidate.
    out["recent_8k_filings"] = []

    return out


def _fetch_news_pool(
    unique_names: set[str],
    days_back:    int = 30,
    max_items:    int = 500,
) -> list[dict]:
    """
    Pre-fetch news for all unique names in a single batch (faster than per-name).
    Resilience: returns empty list on failure; downstream LLM gets no news context.
    """
    try:
        from engine.news_fetchers import fetch_all_for_portfolio
        news_items_obj = fetch_all_for_portfolio(
            tickers=list(unique_names),
            sectors=[],
            days_back=days_back,
            max_items=max_items,
        )
        # Convert to list[dict]
        return [
            n.to_dict() if hasattr(n, "to_dict") else dict(n)
            for n in news_items_obj
        ]
    except Exception as exc:
        logger.warning("ehrm: news pool fetch failed (continuing without): %s", exc)
        return []


def main(as_of: Optional[datetime.date] = None, *, verbose: bool = False) -> dict:
    """
    Main orchestration. Returns summary dict (also printed as JSON).
    """
    _setup_logging(verbose)

    if as_of is None:
        as_of = datetime.date.today()

    logger.info("=" * 70)
    logger.info("ETF Holdings Risk Monitor — monthly run as_of=%s", as_of)
    logger.info("Spec: id=49 v3 hash=9cc868d2 (forward-locked 2026-05-14)")
    logger.info("=" * 70)

    # Phase A3 (2026-05-14) — purge expired cap_state.json entries first
    # Defense against stale entries persisting past their TTL (HARD_CAP_DURATION_DAYS=5
    # trading days; cleanup uses 3 calendar-day buffer for audit traceability).
    from engine.etf_holdings_risk_monitor import cleanup_expired_cap_state
    n_purged = cleanup_expired_cap_state(as_of=as_of)
    if n_purged > 0:
        logger.info("Pre-run cleanup: purged %d expired cap_state entries", n_purged)

    from engine.etf_holdings_ingestion import (
        fetch_all_equity_etf_holdings,
        deduplicate_holdings_to_unique_names,
    )
    from engine.etf_holdings_risk_monitor import (
        screen_name,
        aggregate_etf_risk,
        trigger_etf_cap,
        _persist_cap_trigger,
        _write_decision_log_cap_activation,
        get_cost_status,
    )

    # 1. Fetch holdings
    logger.info("Step 1: fetch top-10 holdings for 24 equity ETFs ...")
    holdings_by_etf = fetch_all_equity_etf_holdings(as_of)
    n_etfs = len(holdings_by_etf)
    n_with_holdings = sum(1 for h in holdings_by_etf.values() if h)
    logger.info("  → %d ETFs, %d with holdings (coverage %.1f%%)",
                n_etfs, n_with_holdings, n_with_holdings / max(n_etfs, 1) * 100)

    # 2. Dedup
    unique_names = deduplicate_holdings_to_unique_names(holdings_by_etf)
    logger.info("Step 2: dedup → %d unique names", len(unique_names))

    # 3. Pre-fetch news pool
    logger.info("Step 3: pre-fetch news pool (last 30 days, up to 500 items) ...")
    news_pool = _fetch_news_pool(unique_names, days_back=30, max_items=500)
    logger.info("  → %d news items fetched", len(news_pool))

    # 4. Per-name LLM screen
    logger.info("Step 4: per-name LLM screen (~%d calls, $0.05 typical each) ...", len(unique_names))
    name_scores: dict[str, int] = {}
    n_screened = 0
    n_fallbacks = 0
    n_high_score = 0
    for name in sorted(unique_names):
        ctx = _gather_inputs_for_name(name, as_of, news_pool)
        try:
            result = screen_name(
                name=name,
                as_of=as_of,
                sector=ctx["sector"],
                recent_8k_filings=ctx["recent_8k_filings"],
                recent_news=ctx["recent_news"],
                price_30d_return=ctx["price_30d_return"],
                next_earnings_date=ctx["next_earnings_date"],
            )
            name_scores[name] = result["risk_score"]
            n_screened += 1
            if result.get("fallback"):
                n_fallbacks += 1
            if result["risk_score"] >= 3:
                n_high_score += 1
        except Exception as exc:
            logger.warning("ehrm: screen_name failed for %s: %s — defaulting to 1", name, exc)
            name_scores[name] = 1
            n_fallbacks += 1
    logger.info(
        "  → %d screened, %d fallbacks, %d high-score (≥3)",
        n_screened, n_fallbacks, n_high_score,
    )

    # 5. Per-ETF aggregation + cap trigger
    logger.info("Step 5: per-ETF aggregation + cap trigger detection ...")
    cap_activations: list[dict] = []
    etf_aggregates: dict[str, float] = {}
    for etf, holdings in holdings_by_etf.items():
        score = aggregate_etf_risk(holdings, name_scores)
        etf_aggregates[etf] = round(score, 4)
        # v2 amendment: pass holdings + name_scores for max-of fallback
        if trigger_etf_cap(score, holdings=holdings, name_scores=name_scores):
            top_contributors = sorted(
                holdings,
                key=lambda h: -name_scores.get(str(h.get("name", "")).upper(), 1)
                              * h.get("weight", 0.0),
            )[:3]
            top_names = [str(h.get("name", "")).upper() for h in top_contributors]
            n_high = sum(
                1 for h in holdings
                if name_scores.get(str(h.get("name", "")).upper(), 1) >= 3
            )
            rationale = (
                f"Aggregate risk {score:.2f} ≥ 3.5; top contributors: "
                f"{', '.join(top_names)}; {n_high} holdings ≥ score 3"
            )
            _persist_cap_trigger(etf, as_of, score, rationale)
            decision_id = _write_decision_log_cap_activation(
                etf=etf, aggregate_score=score, rationale=rationale,
                n_holdings_above_3=n_high, triggered_at=as_of,
            )
            cap_activations.append({
                "etf":              etf,
                "aggregate_score":  round(score, 4),
                "rationale":        rationale,
                "decision_log_id":  decision_id,
            })

    logger.info("  → %d cap activations", len(cap_activations))

    # 6. Cost status
    cost_status = get_cost_status(as_of)
    logger.info(
        "Step 6: cost burn %.4f / %.2f cap (%.1f%%)",
        cost_status["trailing_365d_total_usd"],
        cost_status["annual_cap_usd"],
        cost_status["fraction_of_annual_cap"] * 100,
    )

    summary = {
        "as_of":              as_of.isoformat(),
        "spec_id":            49,
        "spec_hash_prefix":   "0c3696fc4145",
        "n_etfs":             n_etfs,
        "n_with_holdings":    n_with_holdings,
        "n_unique_names":     len(unique_names),
        "n_news_items":       len(news_pool),
        "n_screened":         n_screened,
        "n_fallbacks":        n_fallbacks,
        "n_high_score_names": n_high_score,
        "etf_aggregates":     etf_aggregates,
        "cap_activations":    cap_activations,
        "n_cap_activations":  len(cap_activations),
        "cost_status":        cost_status,
        "completed_at":       datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    # Persist for UI consumption (pages/etf_holdings_monitor.py reads this)
    try:
        from engine.etf_holdings_risk_monitor import _DATA_DIR as _EHRM_DATA_DIR
        _summary_path = _EHRM_DATA_DIR / "last_run_summary.json"
        _summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Summary persisted: %s", _summary_path)
    except Exception as exc:
        logger.warning("Failed to persist summary: %s", exc)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    logger.info("=" * 70)
    logger.info("ETF Holdings Risk Monitor — DONE (%d caps fired)", len(cap_activations))
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF Holdings Risk Monitor monthly run")
    parser.add_argument("--as-of", type=str, default=None,
                        help="ISO date (YYYY-MM-DD); default = today")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose DEBUG logging")
    parser.add_argument("--force", action="store_true",
                        help="Bypass day-of-month gate (run regardless of date). "
                             "Default: only runs on 1st of month (matches monthly cadence "
                             "when daily-scheduled — see docs/etf_holdings_monthly_scheduler_setup.md).")
    args = parser.parse_args()

    as_of_arg = (
        datetime.date.fromisoformat(args.as_of) if args.as_of else None
    )

    # Day-of-month gate (Phase A1, 2026-05-14):
    # Task Scheduler MacroAlphaPro_ETFHoldings fires DAILY at 06:30 SGT for portability,
    # but spec §2.3 mandates monthly cadence. Skip on days 2-31 unless --force or --as-of override.
    today = as_of_arg or datetime.date.today()
    if today.day != 1 and not args.force and args.as_of is None:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        logging.info(
            "ETF Holdings Monitor: skipping day-%d (not 1st of month). "
            "Use --force to override or --as-of to test specific date. "
            "Next monthly run: %s",
            today.day,
            (today.replace(day=1) + datetime.timedelta(days=32)).replace(day=1).isoformat(),
        )
        sys.exit(0)

    try:
        result = main(as_of_arg, verbose=args.verbose)
        sys.exit(0)
    except Exception as exc:
        logger.error("ETF Holdings Monitor failed: %s", exc, exc_info=True)
        sys.exit(1)
