"""engine.sessions.phase — derive an in-flight session's sub-phase.

A session's coarse state (pending_preflight / in_flight / closed) is
already in schema.SessionState. But during in_flight there are 4
meaningful sub-phases the UI should signal:

  awaiting_claude   — preflight done; Claude hasn't started yet
                       (no events emitted, no recent commits)
  claude_working    — recent emit / commit activity within window
                       (Claude is in the IDE doing work)
  awaiting_user     — Claude has open question (heuristic stub
                       for now; future: explicit flag)
  awaiting_close    — required exit artifacts already emitted but
                       session not yet closed (user can close now)

Why a separate module: schema/lifecycle stay stable; phase is a
DERIVED view that may evolve as we learn what signals matter. Putting
it in its own file means future tuning doesn't churn the schema.

Per Gap workflow narrative audit (2026-06-03). UI surfaces (/today,
ActiveSessionBanner) read the phase to show "next action" CTAs.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from engine.sessions.schema import SessionState, UserSession


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CLAUDE_ACTIVITY_WINDOW_MIN = 5   # if any event/commit in last N minutes → claude_working


class SessionPhase(str, Enum):
    pending_preflight = "pending_preflight"
    awaiting_claude   = "awaiting_claude"     # in_flight + zero activity
    claude_working    = "claude_working"      # in_flight + recent emit/commit
    awaiting_close    = "awaiting_close"      # in_flight + exit conditions satisfied
    closed            = "closed"
    abandoned         = "abandoned"


@dataclass(frozen=True)
class PhaseInfo:
    phase:            SessionPhase
    next_action_label:str      # human-readable next step
    next_action_kind: str      # 'copy_brief' / 'wait' / 'close' / 'none'
    last_activity_ts: Optional[str]   # iso ts of most recent emit OR commit
    n_events:         int      # events linked to this session
    n_commits:        int      # commits since session opened


def _utc_now() -> _dt.datetime:
    return _dt.datetime.utcnow()


def _parse_ts(ts: str) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _session_events(session_id: str) -> list:
    """Pull events linked to this session (tag-based)."""
    try:
        from engine.research_store import store as event_store
        return [e for e in event_store.all_events()
                if any(t == f"session:{session_id}" for t in e.tags)]
    except Exception:
        return []


def _session_commits(session: UserSession) -> list[str]:
    """Git short SHAs since the session opened."""
    try:
        out = subprocess.check_output(
            ["git", "log", "--since", session.opened_ts, "--pretty=%h"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return out.splitlines() if out else []
    except Exception:
        return []


def _exit_check_passes(session: UserSession, events: list, commits: list[str]) -> bool:
    """Run the protocol's exit_check WITHOUT raising — just returns bool."""
    try:
        from engine.sessions import protocols
        proto = protocols.for_type(session.session_type)
        satisfied, _missing = proto.exit_check(events, commits)
        return bool(satisfied)
    except Exception:
        return False


def derive_phase(session: UserSession) -> PhaseInfo:
    """Compute the sub-phase + next-action signage for a session.

    Mapping:
      state=pending_preflight   → phase=pending_preflight (UI: finish wizard)
      state=closed              → phase=closed
      state=abandoned           → phase=abandoned
      state=in_flight:
        exit_check satisfied    → awaiting_close   (UI: 'Close session now')
        recent activity (≤5min) → claude_working   (UI: 'Claude active')
        else                    → awaiting_claude  (UI: 'Open Claude · copy brief')
    """
    if session.state == SessionState.pending_preflight:
        return PhaseInfo(
            phase=SessionPhase.pending_preflight,
            next_action_label="Complete pre-flight wizard",
            next_action_kind="none",
            last_activity_ts=None, n_events=0, n_commits=0,
        )
    if session.state == SessionState.closed:
        return PhaseInfo(
            phase=SessionPhase.closed,
            next_action_label="Session closed",
            next_action_kind="none",
            last_activity_ts=session.closed_ts, n_events=0, n_commits=0,
        )
    if session.state == SessionState.abandoned:
        return PhaseInfo(
            phase=SessionPhase.abandoned,
            next_action_label="Session abandoned",
            next_action_kind="none",
            last_activity_ts=session.closed_ts, n_events=0, n_commits=0,
        )

    # In flight — derive sub-phase
    events = _session_events(session.session_id)
    commits = _session_commits(session)

    # Last activity = max(latest event ts, "now" for commits since we
    # don't have per-commit ts cheaply; approximate as "recent" if any
    # commit and we'd-have-to-shell-out git show for true ts).
    activity_dts: list[_dt.datetime] = []
    for e in events:
        dt = _parse_ts(e.ts)
        if dt: activity_dts.append(dt)
    last_activity_ts = max(activity_dts).strftime("%Y-%m-%dT%H:%M:%SZ") if activity_dts else None

    # Exit conditions met → ready to close
    if _exit_check_passes(session, events, commits):
        return PhaseInfo(
            phase=SessionPhase.awaiting_close,
            next_action_label="Exit conditions met — close session now",
            next_action_kind="close",
            last_activity_ts=last_activity_ts,
            n_events=len(events), n_commits=len(commits),
        )

    # Recent activity = Claude is working
    now = _utc_now()
    is_recent = any(
        (now - dt).total_seconds() < _CLAUDE_ACTIVITY_WINDOW_MIN * 60
        for dt in activity_dts
    )
    if is_recent or commits:
        return PhaseInfo(
            phase=SessionPhase.claude_working,
            next_action_label="Claude is working — wait for verdict emit",
            next_action_kind="wait",
            last_activity_ts=last_activity_ts,
            n_events=len(events), n_commits=len(commits),
        )

    # No activity yet → user needs to open Claude
    return PhaseInfo(
        phase=SessionPhase.awaiting_claude,
        next_action_label="Open Claude CLI · paste session brief",
        next_action_kind="copy_brief",
        last_activity_ts=last_activity_ts,
        n_events=len(events), n_commits=len(commits),
    )
