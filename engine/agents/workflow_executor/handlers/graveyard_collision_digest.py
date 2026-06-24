"""graveyard_collision_digest — weekly summary of graveyard collisions.

The graveyard_collision agent (Commit A) writes one warning row per
intent_filed event. Over a week, those rows accumulate. This workflow
digests them once per week and files a single concise inbox row so the
user notices RISK / WARN patterns without scrolling through individual
warnings.

Reversibility: LEVEL_0 — append-only inbox row + append-only digest
ledger.

Cadence: weekly. Idempotency keyed by ISO week. Re-running same week
overwrites the prior week's digest only logically (we append a new
ledger row, dedup happens at consumer side via week-key).
"""
from __future__ import annotations

import datetime as _dt
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from engine.agents.workflow_executor.base import Workflow, ReversibilityLevel
from engine.agents.workflow_executor.registry import register_workflow


_REPO_ROOT     = Path(__file__).resolve().parent.parent.parent.parent.parent
_WARNINGS_PATH = _REPO_ROOT / "data" / "graveyard_collision" / "warnings.jsonl"
_INBOX_PATH    = _REPO_ROOT / "data" / "research" / "research_ops_inbox.jsonl"
_LEDGER_PATH   = _REPO_ROOT / "data" / "agents" / "workflow_executor" / "graveyard_digests.jsonl"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_week_key() -> str:
    iso = _dt.datetime.utcnow().isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _parse_ts(ts: str) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.fromisoformat(ts.rstrip("Z")[:19])
    except Exception:
        return None


@register_workflow
class GraveyardCollisionDigest(Workflow):
    workflow_id      = "graveyard_collision_digest"
    description      = ("Weekly digest of graveyard_collision warnings. "
                        "Counts by verdict + family + top RISK candidates; "
                        "files one inbox row per week.")
    reversibility    = ReversibilityLevel.LEVEL_0
    blast_radius_max = {"files_written": 2, "llm_tokens": 0, "wall_seconds": 15}

    def idempotency_key(self, **inputs) -> str:
        return f"graveyard_digest::{_iso_week_key()}"

    def precondition(self, **inputs) -> tuple[bool, str]:
        # File may not exist if no intents have been filed yet — that's
        # still actionable (digest = "0 collisions this week").
        return (True, "ok")

    def is_due(self, last_run_ts: Optional[str], inputs: dict) -> bool:
        # Weekly: due if last run was in a different ISO week
        if last_run_ts is None:
            return True
        try:
            t = _parse_ts(last_run_ts)
            if not t:
                return True
            iso = t.isocalendar()
            last_key = f"{iso.year}-W{iso.week:02d}"
            return last_key != _iso_week_key()
        except Exception:
            return True

    def run(self, **inputs) -> dict:
        now = _dt.datetime.utcnow()
        cutoff = now - _dt.timedelta(days=7)

        rows: list[dict] = []
        if _WARNINGS_PATH.is_file():
            with _WARNINGS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    t = _parse_ts(str(r.get("checked_ts", "")))
                    if t and t >= cutoff:
                        rows.append(r)

        by_verdict = Counter(r.get("verdict", "?") for r in rows)
        by_family  = Counter(r.get("family", "?")  for r in rows)
        risk_rows  = [r for r in rows if r.get("verdict") == "RISK"]
        # Top-3 RISK by recency
        risk_rows.sort(key=lambda r: r.get("checked_ts", ""), reverse=True)
        top_risk = [{
            "warning_id":     r.get("warning_id"),
            "candidate_name": r.get("candidate_name"),
            "family":         r.get("family"),
            "reason":         r.get("reason"),
            "checked_ts":     r.get("checked_ts"),
        } for r in risk_rows[:3]]

        digest_row = {
            "row_ts":     _utc_iso(),
            "week":       _iso_week_key(),
            "n_total":    len(rows),
            "by_verdict": dict(by_verdict),
            "by_family":  dict(by_family),
            "top_risk":   top_risk,
        }
        _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(digest_row, ensure_ascii=False, default=str) + "\n")
        files_written = 1

        # Inbox alert — only if there's something to say
        if rows:
            n_risk = by_verdict.get("RISK", 0)
            n_warn = by_verdict.get("WARN", 0)
            priority = "high" if n_risk > 0 else "low"
            _INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _INBOX_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "channel":  "graveyard",
                    "ts":       _utc_iso(),
                    "kind":     "weekly_collision_digest",
                    "priority": priority,
                    "title":    (f"week {_iso_week_key()} — {len(rows)} graveyard checks, "
                                 f"{n_risk} RISK, {n_warn} WARN"),
                    "body":     (
                        f"Most-frequent families this week: "
                        f"{', '.join(f'{f}({n})' for f, n in by_family.most_common(3))}. "
                        + (f"Top RISK: {top_risk[0]['candidate_name']} "
                           f"(family={top_risk[0]['family']}). "
                           if top_risk else "")
                        + "Review /research/lessons?verdict=red if RISK trending up."
                    ),
                    "where":    "/research/lessons?verdict=red",
                    "agent_id": "graveyard_collision_digest",
                }, ensure_ascii=False, default=str) + "\n")
            files_written = 2

        return {
            "week":        _iso_week_key(),
            "n_total":     len(rows),
            "n_risk":      by_verdict.get("RISK", 0),
            "n_warn":      by_verdict.get("WARN", 0),
            "_blast_actual": {
                "files_written": files_written,
                "llm_tokens":    0,
                "wall_seconds":  0,
            },
        }

    def postcondition(self, result: dict) -> tuple[bool, str]:
        if not isinstance(result, dict) or "n_total" not in result:
            return (False, "missing n_total")
        return (True, f"digested {result['n_total']} warnings for {result['week']}")
