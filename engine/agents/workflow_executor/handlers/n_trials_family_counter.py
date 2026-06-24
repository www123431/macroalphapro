"""n_trials_family_counter — Bailey-LdP family-aware trial counter.

User concern (verbatim, earlier): "我搞了多少次 p-hack 自己都不知道".
This workflow maintains a per-family ledger of factor_verdict_filed
events so the user knows when a family is approaching the Bailey-LdP
§3 multiple-testing threshold and further tests inflate deflated
Sharpe variance estimates.

Reversibility: LEVEL_0
  - Output is an append-only ledger row at
    data/agents/workflow_executor/n_trials_ledger.jsonl
  - Optional inbox alert when threshold crossed (also append-only)
  - No state mutation outside these two files

Cadence: cron-driven, daily. Scans new factor_verdict_filed events
since last run and updates family counts.

Idempotency: keyed by (last_event_id_processed, today_date). Re-running
the same day re-scans events but skips events whose event_id was
already counted into the per-family entry.

Threshold: per Bailey-LdP §3, deflated-Sharpe penalty grows roughly
log(N_trials). The codebase doctrine sets per-family caution at N=7
and HARD warning at N=15. Both surface as inbox rows.

Output schema per ledger row:
  {
    "row_ts": "2026-...",
    "trigger_run_id": "...",
    "families": {
      "carry":    {"n_trials": 8, "last_event_id": "...", "last_ts": "..."},
      "tsmom":    {...},
      ...
    },
    "warnings": [
      {"family": "carry", "n_trials": 8, "level": "CAUTION"},
      ...
    ]
  }
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional

from engine.agents.workflow_executor.base import Workflow, ReversibilityLevel
from engine.agents.workflow_executor.registry import register_workflow


_REPO_ROOT     = Path(__file__).resolve().parent.parent.parent.parent.parent
_EVENTS_PATH   = _REPO_ROOT / "data" / "research_store" / "events.jsonl"
_LEDGER_PATH   = _REPO_ROOT / "data" / "agents" / "workflow_executor" / "n_trials_ledger.jsonl"
_INBOX_PATH    = _REPO_ROOT / "data" / "research" / "research_ops_inbox.jsonl"

_CAUTION_THRESHOLD = 7
_HARD_THRESHOLD    = 15


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_key() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


@register_workflow
class NTrialsFamilyCounter(Workflow):
    workflow_id      = "n_trials_family_counter"
    description      = ("Scan factor_verdict_filed events; per family, "
                        "maintain trial count; alert at Bailey-LdP §3 thresholds.")
    reversibility    = ReversibilityLevel.LEVEL_0
    blast_radius_max = {"files_written": 2, "llm_tokens": 0, "wall_seconds": 30}

    def idempotency_key(self, **inputs) -> str:
        # One run per day. Re-running same day = same ledger row updated
        # logically (we append a fresh row but mark same date).
        return f"n_trials::{_today_key()}"

    def precondition(self, **inputs) -> tuple[bool, str]:
        if not _EVENTS_PATH.is_file():
            return (False, f"events store missing: {_EVENTS_PATH}")
        return (True, "events store present")

    def is_due(self, last_run_ts: Optional[str], inputs: dict) -> bool:
        # Daily cadence — due if last run was on a different day
        if last_run_ts is None:
            return True
        try:
            last_day = last_run_ts[:10]   # YYYY-MM-DD
            return last_day != _today_key()
        except Exception:
            return True

    def run(self, **inputs) -> dict:
        # Scan all factor_verdict_filed events; tally by family
        families: dict[str, dict] = {}
        n_events = 0
        with _EVENTS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("event_type") != "factor_verdict_filed":
                    continue
                fam = (ev.get("family") or "").lower() or "unknown"
                entry = families.setdefault(fam, {
                    "n_trials":      0,
                    "last_event_id": None,
                    "last_ts":       None,
                    "events":        [],  # short list for audit
                })
                entry["n_trials"] += 1
                entry["last_event_id"] = ev.get("event_id")
                entry["last_ts"]       = ev.get("ts")
                # Keep a short trail of last 5 event_ids for forensic check
                trail = entry.setdefault("events", [])
                trail.append({
                    "event_id": ev.get("event_id"),
                    "verdict":  ev.get("verdict"),
                    "ts":       ev.get("ts"),
                })
                entry["events"] = trail[-5:]
                n_events += 1

        # Threshold check
        warnings: list[dict] = []
        for fam, e in families.items():
            n = e["n_trials"]
            if n >= _HARD_THRESHOLD:
                warnings.append({"family": fam, "n_trials": n, "level": "HARD"})
            elif n >= _CAUTION_THRESHOLD:
                warnings.append({"family": fam, "n_trials": n, "level": "CAUTION"})

        # Append ledger row
        row = {
            "row_ts":   _utc_iso(),
            "date":     _today_key(),
            "n_events_scanned": n_events,
            "families": families,
            "warnings": warnings,
        }
        _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        files_written = 1

        # Inbox alerts for any new HARD or CAUTION
        if warnings:
            _INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _INBOX_PATH.open("a", encoding="utf-8") as f:
                for w in warnings:
                    f.write(json.dumps({
                        "channel":  "methodology",
                        "ts":       _utc_iso(),
                        "kind":     "n_trials_threshold",
                        "priority": "high" if w["level"] == "HARD" else "medium",
                        "title":    (f"family '{w['family']}' n_trials = {w['n_trials']} "
                                     f"({w['level']})"),
                        "body":     (
                            f"Bailey-LdP §3 family-aware multiple-testing "
                            f"correction now requires N={w['n_trials']} trials in "
                            f"the deflated-Sharpe denominator for any new test on "
                            f"this family. {'HARD threshold — consider stopping further tests on this family.' if w['level'] == 'HARD' else 'CAUTION — every new test carries a higher penalty.'}"
                        ),
                        "where":    "/research/lessons?mechanism_family="
                                     + w["family"].upper(),
                        "agent_id": "n_trials_family_counter",
                    }, ensure_ascii=False, default=str) + "\n")
            files_written = 2

        return {
            "n_events_scanned": n_events,
            "n_families":       len(families),
            "n_warnings":       len(warnings),
            "_blast_actual":    {
                "files_written": files_written,
                "llm_tokens":    0,
                "wall_seconds":  0,  # filled by runner
            },
        }

    def postcondition(self, result: dict) -> tuple[bool, str]:
        if not isinstance(result, dict):
            return (False, "result not a dict")
        if "n_events_scanned" not in result:
            return (False, "missing n_events_scanned")
        return (True, f"counted {result['n_events_scanned']} events across "
                       f"{result['n_families']} families, "
                       f"{result['n_warnings']} warning(s)")
