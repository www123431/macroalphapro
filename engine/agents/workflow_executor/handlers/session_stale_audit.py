"""session_stale_audit — flag research_new sessions that have been open
> 12h without emitting a verdict or capability evidence.

Per CLAUDE.md Session Protocol Doctrine, research_new sessions have
explicit exit conditions (≥1 factor_verdict_filed + ≥1
capability_evidence_filed). Sessions that sit open for hours signal
either (a) the user forgot to close them or (b) the work stalled.
Either way, the user should know.

Reversibility: LEVEL_0 — append-only inbox alert per stale session.

Cadence: cron-driven, daily.

Idempotency: keyed by (session_id, current_date). One alert per
session per day at most.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional

from engine.agents.workflow_executor.base import Workflow, ReversibilityLevel
from engine.agents.workflow_executor.registry import register_workflow


_REPO_ROOT      = Path(__file__).resolve().parent.parent.parent.parent.parent
_SESSIONS_PATH  = _REPO_ROOT / "data" / "sessions" / "sessions.jsonl"
_INBOX_PATH     = _REPO_ROOT / "data" / "research" / "research_ops_inbox.jsonl"
_LEDGER_PATH    = _REPO_ROOT / "data" / "agents" / "workflow_executor" / "session_stale_alerts.jsonl"

_STALE_HOURS = 12


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_key() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


def _parse_ts(ts: str) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.fromisoformat(ts.rstrip("Z")[:19])
    except Exception:
        return None


@register_workflow
class SessionStaleAudit(Workflow):
    workflow_id      = "session_stale_audit"
    description      = ("Scan in_flight research_new sessions; alert on any "
                        f"open > {_STALE_HOURS}h without a verdict event.")
    reversibility    = ReversibilityLevel.LEVEL_0
    blast_radius_max = {"files_written": 2, "llm_tokens": 0, "wall_seconds": 15}

    def idempotency_key(self, **inputs) -> str:
        return f"session_stale::{_today_key()}"

    def precondition(self, **inputs) -> tuple[bool, str]:
        if not _SESSIONS_PATH.is_file():
            return (False, f"sessions store missing: {_SESSIONS_PATH}")
        return (True, "ok")

    def is_due(self, last_run_ts: Optional[str], inputs: dict) -> bool:
        if last_run_ts is None:
            return True
        return last_run_ts[:10] != _today_key()

    def run(self, **inputs) -> dict:
        now = _dt.datetime.utcnow()
        cutoff = now - _dt.timedelta(hours=_STALE_HOURS)

        # Find the LATEST row per session_id (sessions.jsonl is append-only)
        sessions: dict[str, dict] = {}
        with _SESSIONS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                sid = r.get("session_id")
                if not sid:
                    continue
                sessions[sid] = r   # last-write-wins per session

        stale: list[dict] = []
        for sid, s in sessions.items():
            if s.get("session_type") != "research_new":
                continue
            phase = s.get("phase") or s.get("state") or ""
            if phase in ("closed", "abandoned", "completed"):
                continue
            opened_ts = s.get("opened_ts") or s.get("created_ts")
            t = _parse_ts(opened_ts or "") if opened_ts else None
            if not t or t > cutoff:
                continue
            age_hours = round((now - t).total_seconds() / 3600.0, 1)
            stale.append({
                "session_id":   sid,
                "title":        s.get("title", "(untitled)")[:120],
                "opened_ts":    opened_ts,
                "age_hours":    age_hours,
                "phase":        phase,
            })

        # Append ledger row even on zero hits (audit completeness)
        _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        ledger_row = {
            "row_ts":         _utc_iso(),
            "date":           _today_key(),
            "n_sessions_scanned": len(sessions),
            "n_stale":        len(stale),
            "stale":          stale,
            "stale_threshold_hours": _STALE_HOURS,
        }
        with _LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_row, ensure_ascii=False, default=str) + "\n")
        files_written = 1

        # Inbox alerts
        if stale:
            _INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _INBOX_PATH.open("a", encoding="utf-8") as f:
                for s in stale:
                    f.write(json.dumps({
                        "channel":  "session_protocol",
                        "ts":       _utc_iso(),
                        "kind":     "session_stale",
                        "priority": "medium",
                        "title":    f"session {s['session_id'][:8]} open {s['age_hours']}h — verify exit or abandon",
                        "body":     (
                            f"research_new session '{s['title']}' opened "
                            f"{s['age_hours']}h ago, currently in phase "
                            f"'{s['phase']}'. Per CLAUDE.md Session Protocol, "
                            f"close requires ≥1 factor_verdict_filed + ≥1 "
                            f"capability_evidence_filed; if work stalled, use "
                            f"Abandon with reason."
                        ),
                        "where":    "/lab/sessions?focus=" + s["session_id"],
                        "agent_id": "session_stale_audit",
                    }, ensure_ascii=False, default=str) + "\n")
            files_written = 2

        return {
            "n_sessions_scanned": len(sessions),
            "n_stale":            len(stale),
            "_blast_actual":      {
                "files_written": files_written,
                "llm_tokens":    0,
                "wall_seconds":  0,
            },
        }

    def postcondition(self, result: dict) -> tuple[bool, str]:
        if not isinstance(result, dict) or "n_stale" not in result:
            return (False, "missing n_stale")
        return (True, f"scanned {result['n_sessions_scanned']} sessions, "
                       f"{result['n_stale']} stale")
