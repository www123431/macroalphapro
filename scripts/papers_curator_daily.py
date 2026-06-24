"""scripts/papers_curator_daily.py — Employee A daily ingestion (2026-06-14).

DAILY HEARTBEAT for research substrate. This is the cron that should fire
EVERY DAY — burndown / verdict production is a different cadence
(weekly / on-demand) per the cron architecture reset of 2026-06-14.

Pipeline (each step independent; failures degrade gracefully):

  1. crawl_arxiv_qfin       → new arXiv q-fin candidates → cache.jsonl
  2. crawl_nber_rss         → new NBER working papers     → cache.jsonl
  3. judge unjudged         → DeepSeek filter (relevance) → judgments.jsonl
  4. summarize accepted     → DeepSeek summary (claims)   → summaries.jsonl
  5. run_synthesis_pipeline → Sonnet synthesize hypotheses → hypotheses.jsonl
                                                            + research_store event

Cost per day (typical):
  - Crawl: $0
  - Filter: ~$0.002 × N new papers (N ≈ 5-15 per day)
  - Summarize: ~$0.005 × M accepted (M ≈ 3-8 per day)
  - Synthesize: ~$0.10-0.30 (depends on summaries window)
  - Total: $0.20-0.80 / day, $6-24 / month

Cadence rationale (industry parallel — AQR / Two Sigma / Renaissance):
  Research pipeline has asymmetric cadence by design:
    - Ingestion = continuous/daily (paper feeds drop daily)
    - Hypothesis extraction = same cadence as ingestion (LLM fresh-context advantage)
    - Verdict / backtest = batched weekly (Bailey-LdP n_trials caps verdict
      throughput at ~9/wk regardless of how often you push the button)
    - Capital decision = monthly / quarterly (human)

The PREVIOUS daily-burndown design was inverted — it ran consumption at
daily cadence while ingestion sat dormant. The queue went stale.

Usage:
  python scripts/papers_curator_daily.py                  # full pipeline
  python scripts/papers_curator_daily.py --dry-run        # no LLM cost
  python scripts/papers_curator_daily.py --skip-crawl     # synthesize only
  python scripts/papers_curator_daily.py --max-filter 20  # cost cap
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("papers_curator_daily")


def _load_secrets_into_env() -> None:
    """Load API keys from .streamlit/secrets.toml into os.environ so the
    LLM call layer + DeepSeek provider see them in headless cron context."""
    import os
    secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.is_file():
        logger.warning("secrets.toml not found at %s", secrets_path)
        return
    try:
        try:
            import tomllib as tom   # py3.11+
            with secrets_path.open("rb") as fh:
                s = tom.load(fh)
        except ModuleNotFoundError:
            import tomli as tom     # py3.10 fallback
            with secrets_path.open("rb") as fh:
                s = tom.load(fh)
    except Exception as exc:
        logger.warning("secrets load failed: %s", exc)
        return
    for k in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
                "SEMANTIC_SCHOLAR_API_KEY"):
        if s.get(k):
            os.environ[k] = s[k]


def _step_crawl(arxiv_max: int, nber: bool) -> dict:
    """Step 1+2: crawl new papers from enabled sources."""
    from engine.agents.papers_curator import (
        crawl_all, save_new_candidates, load_cache,
    )
    n_before = len(load_cache())
    cands = crawl_all(arxiv_max=arxiv_max)
    n_new_arxiv = save_new_candidates(cands)
    n_new_nber = 0
    if nber:
        try:
            from engine.agents.papers_curator.nber_rss_crawler import (
                crawl_and_persist_nber,
            )
            nber_result = crawl_and_persist_nber()
            n_new_nber = int(nber_result.get("new_count") or 0)
        except Exception as exc:
            logger.warning("NBER crawl failed (non-fatal): %s", exc)
    return {
        "arxiv_fetched":  len(cands),
        "arxiv_new":      n_new_arxiv,
        "nber_new":       n_new_nber,
        "cache_total":    n_before + n_new_arxiv + n_new_nber,
    }


def _step_filter(max_filter: int) -> dict:
    """Step 3: judge unjudged papers with the filter (DeepSeek)."""
    from engine.agents.papers_curator import load_cache
    from engine.agents.papers_curator.filter import judge_paper
    from engine.agents.papers_curator.judgments_store import (
        latest_by_paper, append_judgment,
    )

    candidates = load_cache()
    judged_map = latest_by_paper()
    unjudged = [
        c for c in candidates
        if (c.source, c.source_id) not in judged_map
    ]
    if not unjudged:
        return {"unjudged_total": 0, "judged_now": 0, "yes": 0, "no": 0,
                  "errors": 0}
    # Cost cap: only judge first max_filter unjudged this run
    batch = unjudged[:max_filter]
    yes_count = no_count = err_count = 0
    for c in batch:
        try:
            j = judge_paper(c)
            if j is None:
                err_count += 1
                continue
            append_judgment(j)
            if j.is_tradable_factor:
                yes_count += 1
            else:
                no_count += 1
        except Exception as exc:
            logger.warning("filter exception on %s/%s: %s",
                            c.source, c.source_id, exc)
            err_count += 1
    return {
        "unjudged_total": len(unjudged),
        "judged_now":     len(batch),
        "yes":            yes_count,
        "no":             no_count,
        "errors":         err_count,
    }


def _step_summarize(max_summary: int) -> dict:
    """Step 4: summarize accepted-but-not-summarized papers (DeepSeek)."""
    from engine.agents.papers_curator import load_cache
    from engine.agents.papers_curator.summarizer import summarize_paper
    from engine.agents.papers_curator.judgments_store import (
        latest_by_paper as latest_judgments,
    )
    from engine.agents.papers_curator.summaries_store import (
        latest_by_paper as latest_summaries, append_summary,
    )

    candidates = load_cache()
    judged_map = latest_judgments()
    summarized_map = latest_summaries()

    accepted_unsummarized = []
    for c in candidates:
        key = (c.source, c.source_id)
        j = judged_map.get(key)
        if j is None or not j.is_tradable_factor:
            continue
        if key in summarized_map:
            continue
        accepted_unsummarized.append((c, j))
    if not accepted_unsummarized:
        return {"accepted_unsummarized_total": 0, "summarized_now": 0,
                  "errors": 0}
    batch = accepted_unsummarized[:max_summary]
    ok = err = 0
    for c, j in batch:
        try:
            s = summarize_paper(c, j)
            if s is None:
                err += 1
                continue
            append_summary(s)
            ok += 1
        except Exception as exc:
            logger.warning("summarizer exception on %s/%s: %s",
                            c.source, c.source_id, exc)
            err += 1
    return {
        "accepted_unsummarized_total": len(accepted_unsummarized),
        "summarized_now":              len(batch),
        "ok":                           ok,
        "errors":                       err,
    }


def _step_synthesize(dry_run: bool) -> dict:
    """Step 5: Sonnet synthesize hypotheses from summaries window."""
    from engine.agents.papers_curator.synthesis_runner import (
        run_synthesis_pipeline,
    )
    result = run_synthesis_pipeline(dry_run=dry_run)
    return {
        "n_candidates": result.get("n_candidates", 0),
        "n_written":    result.get("n_written", 0),
        "errors":       len(result.get("errors") or []),
        "snapshot":     result.get("snapshot", {}),
        "event_id":     result.get("event_id"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arxiv-max",  type=int, default=50)
    ap.add_argument("--max-filter", type=int, default=20,
                     help="cap on filter LLM calls per run (~$0.04/run @ 20)")
    ap.add_argument("--max-summary", type=int, default=15,
                     help="cap on summary LLM calls per run (~$0.08/run @ 15)")
    ap.add_argument("--no-nber", action="store_true",
                     help="skip NBER crawl (arXiv only)")
    ap.add_argument("--skip-crawl", action="store_true",
                     help="skip crawl step (synthesize only)")
    ap.add_argument("--skip-filter", action="store_true",
                     help="skip filter step")
    ap.add_argument("--skip-summarize", action="store_true",
                     help="skip summarize step")
    ap.add_argument("--skip-synthesize", action="store_true",
                     help="skip synthesize step")
    ap.add_argument("--dry-run", action="store_true",
                     help="synthesize without writing hypotheses")
    args = ap.parse_args()

    _load_secrets_into_env()

    print(f"=== papers_curator_daily @ {Path(__file__).name} ===")
    print()

    if args.skip_crawl:
        print("[1+2] crawl SKIPPED")
    else:
        crawl_stats = _step_crawl(args.arxiv_max, nber=not args.no_nber)
        print(f"[1+2] crawl: arxiv fetched={crawl_stats['arxiv_fetched']} "
                f"new={crawl_stats['arxiv_new']}, nber new={crawl_stats['nber_new']}, "
                f"cache total={crawl_stats['cache_total']}")

    if args.skip_filter:
        print("[3] filter SKIPPED")
    else:
        filter_stats = _step_filter(args.max_filter)
        print(f"[3] filter: unjudged_total={filter_stats['unjudged_total']} "
                f"judged_now={filter_stats['judged_now']} "
                f"(yes={filter_stats['yes']} no={filter_stats['no']} "
                f"errors={filter_stats['errors']})")

    if args.skip_summarize:
        print("[4] summarize SKIPPED")
    else:
        summ_stats = _step_summarize(args.max_summary)
        print(f"[4] summarize: accepted_unsummarized_total="
                f"{summ_stats['accepted_unsummarized_total']} "
                f"summarized_now={summ_stats['summarized_now']} "
                f"(ok={summ_stats.get('ok',0)} errors={summ_stats.get('errors',0)})")

    if args.skip_synthesize:
        print("[5] synthesize SKIPPED")
    else:
        synth_stats = _step_synthesize(args.dry_run)
        print(f"[5] synthesize: n_candidates={synth_stats['n_candidates']} "
                f"n_written={synth_stats['n_written']} "
                f"errors={synth_stats['errors']} "
                f"event_id={synth_stats['event_id']}")
        snap = synth_stats.get("snapshot", {})
        if snap:
            print(f"     snapshot: summaries={snap.get('recent_summaries')} "
                    f"sleeves={snap.get('deployed_sleeves')} "
                    f"events={snap.get('recent_events')} "
                    f"doctrine={snap.get('doctrine_snippets')}")

    print()
    print("=== daily ingest complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
