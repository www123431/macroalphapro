"""
scripts/run_forensic_replay_anchor_events.py — Gap #3 forensic replay runner.

Runs replay_harness across 3 anchor events × 3 tickers (SPY/TLT/GLD):
  - Lehman 2008-09-15 (FF5 decomp will FAIL — QUAL/USMV pre-launch)
  - Christmas Eve 2018-12-24
  - COVID 2020-03-16

Total: 9 ReplayReport (each with devils_advocate + factor_decomp).
LLM cost: ~$0.008 (9 × ~$0.001 per dual-LLM call).

Outputs JSON to data/forensic_replay_results/<event_slug>_<run_id>.json
+ aggregate summary at data/forensic_replay_results/_summary_<run_id>.json.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import uuid
from pathlib import Path

# Allow `python scripts/run_forensic_replay_anchor_events.py` from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.forensic.replay_harness import run_anchor_event_replay  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


EVENT_TICKERS: dict[str, list[str]] = {
    "lehman_2008_09":        ["SPY", "TLT", "GLD"],
    "christmas_eve_2018_12": ["SPY", "TLT", "GLD"],
    "covid_2020_03":         ["SPY", "TLT", "GLD"],
}


def main() -> int:
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.datetime.utcnow().isoformat()
    print(f"=== Gap #3 Forensic Replay run_id={run_id} started_at={started_at} ===")

    # Output directory
    out_dir = ROOT / "data" / "forensic_replay_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_summary = {
        "run_id":     run_id,
        "started_at": started_at,
        "events":     {},
    }

    total_cost = 0.0
    n_devils_advocate_success = 0
    n_factor_decomp_success = 0
    n_total_invocations = 0

    for event_slug, tickers in EVENT_TICKERS.items():
        print(f"\n--- Event: {event_slug} (tickers: {tickers}) ---")
        reports = run_anchor_event_replay(
            event_slug=event_slug,
            tickers=tickers,
            realized_horizon_days=22,
            weight=0.10,
        )

        # Persist per-event JSON
        event_payload = {
            "run_id":        run_id,
            "event_slug":    event_slug,
            "reports":       [r.to_dict() for r in reports],
        }
        (out_dir / f"{event_slug}_{run_id}.json").write_text(
            json.dumps(event_payload, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        # Aggregate per-event summary
        event_summary = {"n_reports": len(reports), "per_ticker": {}}
        for r in reports:
            ticker_summary = {
                "realized_return":     r.context.realized_return,
                "n_anchors_injected":  r.context.n_anchors_injected,
                "agents": {},
            }
            for res in r.results:
                ticker_summary["agents"][res.agent_name] = {
                    "success":        res.success,
                    "output_summary": res.output_summary,
                    "notes":          res.notes,
                }
                n_total_invocations += 1
                if res.agent_name == "devils_advocate" and res.success:
                    n_devils_advocate_success += 1
                    cost = res.output_summary.get("total_cost_usd") or 0.0
                    total_cost += float(cost)
                    print(f"  [DA OK] {r.context.ticker} verdict={res.output_summary.get('primary_forensic_verdict')}/"
                          f"{res.output_summary.get('devil_forensic_verdict')} "
                          f"consistency={res.output_summary.get('consistency_score')} "
                          f"cost=${cost:.5f}")
                elif res.agent_name == "residual_attribution_factor_returns" and res.success:
                    n_factor_decomp_success += 1
                    print(f"  [FF5 OK] {r.context.ticker} Mkt={res.output_summary.get('Mkt')} "
                          f"SMB={res.output_summary.get('SMB')} HML={res.output_summary.get('HML')}")
                else:
                    print(f"  [{res.agent_name} FAIL] {r.context.ticker} reason={res.notes[:80]}")
            event_summary["per_ticker"][r.context.ticker] = ticker_summary
        aggregate_summary["events"][event_slug] = event_summary

    aggregate_summary["completed_at"] = datetime.datetime.utcnow().isoformat()
    aggregate_summary["total_invocations"]            = n_total_invocations
    aggregate_summary["n_devils_advocate_success"]    = n_devils_advocate_success
    aggregate_summary["n_factor_decomp_success"]      = n_factor_decomp_success
    aggregate_summary["total_devils_advocate_cost_usd"] = round(total_cost, 5)

    summary_path = out_dir / f"_summary_{run_id}.json"
    summary_path.write_text(json.dumps(aggregate_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Aggregate ===")
    print(f"total invocations:           {n_total_invocations}")
    print(f"devils_advocate success:     {n_devils_advocate_success} / 9")
    print(f"factor_decomp success:       {n_factor_decomp_success} / 9 (Lehman expected to FAIL → max 6)")
    print(f"total LLM cost:              ${total_cost:.5f}")
    print(f"summary written to:          {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
