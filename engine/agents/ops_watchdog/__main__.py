"""
engine/agents/ops_watchdog/__main__.py — CLI entry for the Ops Watchdog Agent.

Per spec §八 reproducibility section:

    # Manual debug run (no LLM spend)
    py -3.11 -m engine.agents.ops_watchdog --verbose --dry-run

    # Production cron (Task Scheduler MacroAlphaPro_Watchdog daily 06:10 SGT)
    py -3.11 -m engine.agents.ops_watchdog --check

Exit codes (for Task Scheduler / CI):
    0 = run completed (regardless of severity; severity is logged, not exit code)
    1 = orchestrator raised an unexpected exception
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="engine.agents.ops_watchdog",
        description="Ops Watchdog Agent v1.0 (spec id=63 hash 9d050804).",
    )
    p.add_argument("--check", action="store_true",
                   help="Production cron mode (= no flags). Currently identical "
                        "to invoking with no flags; kept for spec compatibility.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip LLM ReAct (rules + triage only). $0 spend.")
    p.add_argument("--no-save-trace", action="store_true",
                   help="Skip writing data/ops_watchdog/YYYY-MM-DD_run.json.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Log INFO-level progress to stderr.")
    p.add_argument("--json", action="store_true",
                   help="Print full WatchdogRunResult as JSON to stdout (else compact summary).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
            stream=sys.stderr,
        )

    try:
        from engine.agents.ops_watchdog.agent import run_watchdog
        result = run_watchdog(
            dry_run    = args.dry_run,
            save_trace = not args.no_save_trace,
            verbose    = args.verbose,
        )
    except Exception:
        logging.exception("watchdog orchestrator failed")
        return 1

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), ensure_ascii=False,
                         indent=2, default=str))
    else:
        print(
            f"Watchdog {result.today_iso} "
            f"dry_run={result.dry_run} "
            f"audit_run_id={result.audit_run_id} "
            f"n_findings={result.n_findings} "
            f"severity={result.triage['severity']} "
            f"llm_used={result.llm_used} "
            f"llm_cost_usd={result.llm_cost_usd:.4f} "
            f"trace={result.trace_json_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
