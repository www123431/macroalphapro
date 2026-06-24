"""Phase 7 — Paper Discovery Runner (new-flow + historical-backfill).

Two cadences, both supported per user 2026-05-30 ("过去的也要啊...双线推进"):

  NEW-FLOW (daily/weekly cron):
    python -m scripts.run_paper_discovery --new-flow \
      --max-per-source 50 --confidence 0.5

  HISTORICAL-BACKFILL (slower cadence, e.g. weekly/monthly):
    python -m scripts.run_paper_discovery --backfill \
      --start 2018-01-01 --end 2026-05-30 \
      --max-per-year 200 --sources arxiv,nber

  DRY-RUN (no LLM call — useful for sanity-checking the pipeline shape):
    python -m scripts.run_paper_discovery --new-flow --no-llm

Output:
  * data/research/discovery_log.jsonl    — every paper outcome (audit trail)
  * data/research/discovery_queue.jsonl  — queue_for_review entries
  * Console summary with stage counts (Stage 5 graveyard hits, etc.)

Windows Task Scheduler examples:
  Daily 06:00:
    schtasks /Create /SC DAILY /TN "macro-alpha\\paper-discovery-new" \
      /TR "python C:\\path\\to\\intern\\scripts\\run_paper_discovery.py --new-flow" \
      /ST 06:00
  Weekly Sunday 04:00:
    schtasks /Create /SC WEEKLY /D SUN /TN "macro-alpha\\paper-discovery-backfill" \
      /TR "python C:\\path\\to\\intern\\scripts\\run_paper_discovery.py \
            --backfill --start 2018-01-01 --end 2026-05-30 --max-per-year 100" \
      /ST 04:00

Cron equivalents (Unix):
  0 6 * * *   cd /path/intern && python -m scripts.run_paper_discovery --new-flow
  0 4 * * 0   cd /path/intern && python -m scripts.run_paper_discovery --backfill \
              --start 2018-01-01 --end 2026-05-30 --max-per-year 100
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/run_paper_discovery.py` without PYTHONPATH=.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def discover_new_flow(
    *,
    max_per_source: int = 50,
    use_llm: bool = True,
    confidence_threshold: float = 0.5,
    use_llm_rescue: bool = False,
) -> dict:
    """Daily/weekly new-flow discovery cadence.

    Returns: dict with sources_fetched + papers_total + stage_counts.
    """
    from engine.research.discovery import multi_source_dispatch
    from engine.research.discovery import discovery_pipeline

    t0 = time.time()
    papers_df = multi_source_dispatch.fetch_new_flow(
        max_results_per_source=max_per_source,
    )
    elapsed = time.time() - t0

    result = {
        "mode":              "new_flow",
        "papers_fetched":    int(len(papers_df)),
        "fetch_elapsed_sec": round(elapsed, 1),
        "summary":           None,
        "by_source":         {},
    }

    if len(papers_df):
        result["by_source"] = (
            papers_df["source"].value_counts().to_dict()
            if "source" in papers_df.columns else {}
        )
        summary = discovery_pipeline.run_discovery_batch(
            papers_df,
            use_llm=use_llm,
            confidence_threshold=confidence_threshold,
            log=True,
            use_llm_rescue=use_llm_rescue,
        )
        result["summary"] = summary

    return result


def discover_backfill(
    start_date: str,
    end_date: str,
    *,
    max_per_year: int = 200,
    sources: list[str] | None = None,
    use_llm: bool = True,
    confidence_threshold: float = 0.5,
    use_llm_rescue: bool = False,
) -> dict:
    """Historical backfill — slice year-by-year.

    Per user 2026-05-30: "过去的也要啊...双线推进" + "论文越齐全越好".
    """
    from engine.research.discovery import multi_source_dispatch
    from engine.research.discovery import discovery_pipeline

    t0 = time.time()
    papers_df = multi_source_dispatch.fetch_historical_backfill(
        start_date, end_date,
        max_results_per_year=max_per_year,
        sources=sources,
    )
    elapsed = time.time() - t0

    result = {
        "mode":               "backfill",
        "start_date":         start_date,
        "end_date":           end_date,
        "papers_fetched":     int(len(papers_df)),
        "fetch_elapsed_sec":  round(elapsed, 1),
        "summary":            None,
        "by_source":          {},
    }

    if len(papers_df):
        result["by_source"] = (
            papers_df["source"].value_counts().to_dict()
            if "source" in papers_df.columns else {}
        )
        summary = discovery_pipeline.run_discovery_batch(
            papers_df,
            use_llm=use_llm,
            confidence_threshold=confidence_threshold,
            log=True,
            use_llm_rescue=use_llm_rescue,
        )
        result["summary"] = summary

    return result


def _write_run_log(result: dict) -> None:
    """Append a run-level summary to data/research/discovery_runs.jsonl."""
    log_path = Path("data/research/discovery_runs.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        **result,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new-flow", action="store_true",
                        help="Daily/weekly cadence — pull last 14 days from all sources")
    mode.add_argument("--backfill", action="store_true",
                        help="Historical cadence — year-by-year over date range")

    parser.add_argument("--start", default="2018-01-01",
                        help="Backfill start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default=str(datetime.date.today()),
                        help="Backfill end date (YYYY-MM-DD)")
    parser.add_argument("--max-per-source", type=int, default=50,
                        help="New-flow: max papers per source")
    parser.add_argument("--max-per-year", type=int, default=200,
                        help="Backfill: max papers per source per year")
    parser.add_argument("--sources", default="arxiv",
                        help="Backfill: comma-separated subset of {arxiv,nber}. "
                              "NBER historical is currently unavailable (no public API).")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="LLM confidence threshold for queue_for_review")
    parser.add_argument("--no-llm", action="store_true",
                        help="Dry-run without LLM extraction (regex-only)")
    parser.add_argument("--use-llm-rescue", action="store_true",
                        help=("Opt-in: for papers below borderline floor, call "
                              "LLM bool-feature extractor to rescue missed markers "
                              "(senior B). Adds ~$0.0008 per rescued paper. "
                              "Recommended for low-volume cross-disciplinary sources "
                              "where abstracts often phrase markers non-canonically."))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    use_llm = not args.no_llm
    if args.new_flow:
        result = discover_new_flow(
            max_per_source=args.max_per_source,
            use_llm=use_llm,
            confidence_threshold=args.confidence,
            use_llm_rescue=args.use_llm_rescue,
        )
    else:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        result = discover_backfill(
            args.start, args.end,
            max_per_year=args.max_per_year,
            sources=sources,
            use_llm=use_llm,
            confidence_threshold=args.confidence,
            use_llm_rescue=args.use_llm_rescue,
        )

    _write_run_log(result)

    print("=" * 72)
    print(f"DISCOVERY RUN — mode={result['mode']}")
    print("=" * 72)
    print(f"Papers fetched:    {result['papers_fetched']}")
    print(f"Fetch elapsed:     {result['fetch_elapsed_sec']}s")
    if result["by_source"]:
        print("By source:")
        for src, n in sorted(result["by_source"].items(), key=lambda kv: -kv[1]):
            print(f"  {src:<24} {n}")
    if result["summary"]:
        s = result["summary"]
        print(f"Total processed:   {s.get('total', 0)}")
        print(f"Queued for review: {s.get('queued', 0)}")
        print(f"Review w/ caveat:  {s.get('review_with_caveat', 0)}")
        if s.get("stage_counts"):
            print("Stage breakdown:")
            for stage, n in sorted(s["stage_counts"].items(), key=lambda kv: -kv[1]):
                print(f"  {stage:<24} {n}")

    return 0 if result["papers_fetched"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
