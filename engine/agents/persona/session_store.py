"""engine/agents/persona/session_store.py — Phase A.3 ε memory persistence.

Backs the Streamlit chat UI with a SQLite scratch pad so a browser
refresh or process restart does not wipe the conversation. Storage
is keyed only by agent_id (local single-user Streamlit; multi-user
would need a user_id dimension that we deliberately do not have).

Three public functions (the only API surface ui/chat_page.py should
touch):
  load_session(agent_id)         -> SessionSnapshot
  save_turn(agent_id, ...)       -> None   (replaces row)
  reset_session(agent_id)        -> None   (delete row, fresh start)

What this is NOT:
  - It is NOT an audit log. DecisionLog / RiskManagerAlert /
    DataQualityAlert handle audit history; their semantics are
    immutable-once-written. This table is mutable scratch.
  - It is NOT cross-agent. Each row is one agent's view; Pattern 5
    (autonomous agent debate) ban is enforced at the schema level.
  - It does not store the agent's SYSTEM PROMPT — that lives in the
    AgentPersona instance and is the spec contract, not session state.

Threading note: SQLite + Streamlit's "rerun on every interaction"
model means saves happen one-at-a-time per user input; we do not need
optimistic concurrency or row-level locking for the intended local
deployment. The whole-row REPLACE on save is the simplest correct
behavior given that st.session_state already holds the canonical
in-memory view between reruns.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SessionSnapshot:
    """One agent's persisted chat state.

    Fields mirror what ui.chat_page.render_chat_page stores in
    st.session_state for that agent: a message history, a per-turn
    tool-call log, and the cumulative cost / latency counters shown
    in the sidebar. ``last_visited_at`` drives the "N new alerts since
    last visit" badge (Phase A.4 2026-05-19); it is None until the
    page is opened for the first time. ``title`` is derived from the
    first user message at save_turn time (Phase A.7 Wave 3.2); shown
    in the sidebar as a human-readable session label.
    """
    history:         list[dict]
    tool_log:        list[list[dict]]
    cost_usd:        float
    latency_ms:      int
    last_visited_at: object = None     # datetime or None
    title:           str | None = None
    session_id:      str = "default"   # Phase A.7 Wave 4.1: multi-session


_EMPTY = SessionSnapshot(
    history=[], tool_log=[], cost_usd=0.0, latency_ms=0,
    last_visited_at=None, title=None, session_id="default",
)


DEFAULT_SESSION_ID = "default"


def new_session_id() -> str:
    """Generate a short fresh session_id like 's_3f4a9c'.

    Hex of a small int from time.time_ns — collision risk negligible at
    solo-PM scale (one user, sessions created seconds apart at most).
    """
    import time
    return f"s_{(time.time_ns() // 1000) & 0xFFFFFF:06x}"


_TITLE_MAX_CHARS = 80


def _derive_title(history: list[dict]) -> str | None:
    """Derive a session title from the first user message in history.

    Strategy: first ~80 chars of the first user turn's text, stripped of
    leading whitespace + trailing ellipsis if truncated. Returns None
    when history is empty or no user message has text yet.

    Cheap, deterministic, no LLM. Industry pattern (ChatGPT does LLM
    title generation; for a solo PM use case the marginal accuracy is
    not worth the per-turn $ cost).
    """
    for msg in history:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        # content may be a string OR Anthropic-format list of blocks
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    content = blk.get("text", "")
                    break
            else:
                content = ""
        if not isinstance(content, str):
            continue
        text = content.strip().split("\n", 1)[0]   # first line only
        if not text:
            continue
        if len(text) <= _TITLE_MAX_CHARS:
            return text
        return text[:_TITLE_MAX_CHARS].rstrip() + "..."
    return None


# Map persona agent_id → (alert ORM class, timestamp column name, source
# label). None entries mean the persona has no native alert table so the
# badge is hidden for that agent. Lazy-evaluated so the import doesn't
# fail if a table is missing during early dev.
def _alert_source_for(agent_id: str):
    try:
        if agent_id == "risk_manager":
            from engine.db_models import RiskManagerAlert
            return (RiskManagerAlert, "generated_at_utc", "Risk Manager alerts")
        if agent_id == "dq_inspector":
            from engine.db_models import DataQualityAlert
            return (DataQualityAlert, "generated_at_utc", "DQ alerts")
        if agent_id == "anomaly_sentinel":
            from engine.db_models import AnomalyFlag
            return (AnomalyFlag, "created_at", "Anomaly flags")
        if agent_id == "audit_recorder":
            from engine.auto_audit_models import AuditFinding
            return (AuditFinding, "detected_at", "Audit findings")
    except Exception as exc:
        logger.debug("session_store: alert source lookup for %s failed: %s",
                     agent_id, exc)
    # attribution_analyst / devils_advocate have no native alert source.
    return None


def count_new_alerts(agent_id: str, since) -> tuple[int, str | None]:
    """Count rows in the agent's native alert table created strictly AFTER
    ``since``. Returns (count, source_label). If the persona has no native
    alert source OR ``since`` is None (first visit), returns (0, None).

    Used by the chat sidebar to render a "N new alerts since last visit"
    badge. Never raises — DB errors degrade to (0, None) so the chat page
    keeps rendering.
    """
    if since is None:
        return (0, None)
    source = _alert_source_for(agent_id)
    if source is None:
        return (0, None)
    cls, ts_col, label = source
    try:
        from engine.db_models import SessionFactory
        with SessionFactory() as s:
            col = getattr(cls, ts_col)
            n = s.query(cls).filter(col > since).count()
        return (int(n), label)
    except Exception as exc:
        logger.warning("session_store.count_new_alerts(%s) failed: %s",
                       agent_id, exc)
        return (0, None)


def _get_row(s, agent_id: str, session_id: str):
    """Helper: fetch the ChatSession row keyed by (agent_id, session_id).
    Returns None if not yet created."""
    from engine.db_models import ChatSession
    return (s.query(ChatSession)
              .filter_by(agent_id=agent_id, session_id=session_id)
              .first())


def mark_visited(agent_id: str, session_id: str = DEFAULT_SESSION_ID) -> None:
    """Stamp ``last_visited_at`` on the ChatSession row so the next visit's
    badge is computed against this moment. Idempotent — creates the row
    with empty history if it doesn't exist yet (first ever visit).
    """
    try:
        import datetime
        from engine.db_models import ChatSession, SessionFactory
        now = datetime.datetime.utcnow()
        with SessionFactory() as s:
            row = _get_row(s, agent_id, session_id)
            if row is None:
                row = ChatSession(
                    agent_id        = agent_id,
                    session_id      = session_id,
                    history_json    = "[]",
                    tool_log_json   = "[]",
                    cost_usd        = 0.0,
                    latency_ms      = 0,
                    updated_at_utc  = now,
                    last_visited_at = now,
                )
                s.add(row)
            else:
                row.last_visited_at = now
            s.commit()
    except Exception as exc:
        logger.warning("session_store.mark_visited(%s/%s) failed: %s",
                       agent_id, session_id, exc)


def load_session(
    agent_id:   str,
    session_id: str = DEFAULT_SESSION_ID,
) -> SessionSnapshot:
    """Return the persisted snapshot for (agent_id, session_id) or _EMPTY.

    Never raises on DB error — instead logs and returns _EMPTY so a DB
    glitch degrades the chat to "fresh session" rather than crashing
    the Streamlit page on render.
    """
    try:
        from engine.db_models import SessionFactory
        with SessionFactory() as s:
            row = _get_row(s, agent_id, session_id)
            if row is None:
                return dataclasses.replace(_EMPTY, session_id=session_id)
            history  = json.loads(row.history_json  or "[]")
            tool_log = json.loads(row.tool_log_json or "[]")
            return SessionSnapshot(
                history         = history,
                tool_log        = tool_log,
                cost_usd        = float(row.cost_usd or 0.0),
                latency_ms      = int(row.latency_ms or 0),
                last_visited_at = row.last_visited_at,
                title           = row.title,
                session_id      = row.session_id,
            )
    except Exception as exc:
        logger.warning("session_store.load_session(%s/%s) failed: %s — "
                       "returning empty snapshot",
                       agent_id, session_id, exc)
        return dataclasses.replace(_EMPTY, session_id=session_id)


def save_turn(
    agent_id:   str,
    history:    list[dict],
    tool_log:   list[list[dict]],
    cost_usd:   float,
    latency_ms: int,
    session_id: str = DEFAULT_SESSION_ID,
) -> None:
    """Replace the row for (agent_id, session_id) with the given snapshot.

    Callers pass the FULL post-turn state (not a delta). The UI keeps
    the canonical view in st.session_state; this function just snaps
    it to disk so the next process / browser session can rehydrate.

    Logs and swallows on DB error — a failed persist must not stop
    the chat itself (the user already saw the answer on screen).
    """
    try:
        import datetime
        from engine.db_models import ChatSession, SessionFactory
        history_json  = json.dumps(history,  ensure_ascii=False)
        tool_log_json = json.dumps(tool_log, ensure_ascii=False)
        now = datetime.datetime.utcnow()
        # Phase A.7 Wave 3.2: derive title from first user message. Only
        # set if currently None — we don't overwrite a title the user
        # later edits manually (set_title API).
        derived_title = _derive_title(history)
        with SessionFactory() as s:
            row = _get_row(s, agent_id, session_id)
            if row is None:
                row = ChatSession(
                    agent_id       = agent_id,
                    session_id     = session_id,
                    history_json   = history_json,
                    tool_log_json  = tool_log_json,
                    cost_usd       = float(cost_usd),
                    latency_ms     = int(latency_ms),
                    updated_at_utc = now,
                    title          = derived_title,
                )
                s.add(row)
            else:
                row.history_json   = history_json
                row.tool_log_json  = tool_log_json
                row.cost_usd       = float(cost_usd)
                row.latency_ms     = int(latency_ms)
                row.updated_at_utc = now
                # Only set title if it's currently null — preserve any
                # user-edited title across subsequent saves.
                if not row.title and derived_title:
                    row.title = derived_title
            s.commit()

        # Phase A.7 Wave 4.3: Tier 2.5 cross-session memory. Embed any
        # new user→assistant turn pairs so future sessions can recall
        # them. Best-effort — failure here must not break the user-
        # facing chat (the answer is already on screen).
        try:
            from engine.agents.persona.turn_memory import embed_and_store_turn
            embed_and_store_turn(agent_id, history, session_id=session_id)
        except Exception as exc:
            logger.debug("save_turn: turn_memory embed failed: %s", exc)
    except Exception as exc:
        logger.warning("session_store.save_turn(%s/%s) failed: %s",
                       agent_id, session_id, exc)


def set_title(
    agent_id:   str,
    title:      str,
    session_id: str = DEFAULT_SESSION_ID,
) -> None:
    """Override the session title manually. Trimmed to TITLE_MAX_CHARS so
    the sidebar still fits. Empty / blank title resets to None so the
    next save_turn re-derives from the first message."""
    try:
        from engine.db_models import ChatSession, SessionFactory
        clean = (title or "").strip()
        clean = clean[:_TITLE_MAX_CHARS] if clean else None
        with SessionFactory() as s:
            row = _get_row(s, agent_id, session_id)
            if row is None:
                # No row to title yet — create a stub so the override
                # survives until the first save_turn writes content.
                row = ChatSession(
                    agent_id      = agent_id,
                    session_id    = session_id,
                    history_json  = "[]",
                    tool_log_json = "[]",
                    cost_usd      = 0.0,
                    latency_ms    = 0,
                    title         = clean,
                )
                s.add(row)
            else:
                row.title = clean
            s.commit()
    except Exception as exc:
        logger.warning("session_store.set_title(%s/%s) failed: %s",
                       agent_id, session_id, exc)


def reset_session(
    agent_id:   str,
    session_id: str = DEFAULT_SESSION_ID,
) -> None:
    """Delete the persisted row for (agent_id, session_id). Idempotent."""
    try:
        from engine.db_models import ChatSession, SessionFactory
        with SessionFactory() as s:
            row = _get_row(s, agent_id, session_id)
            if row is not None:
                s.delete(row)
                s.commit()
    except Exception as exc:
        logger.warning("session_store.reset_session(%s/%s) failed: %s",
                       agent_id, session_id, exc)


def list_sessions(agent_id: str) -> list[dict]:
    """Return all sessions for ``agent_id``, most-recently-updated first.

    Each entry: {session_id, title, updated_at_utc, n_turns,
    last_visited_at}. Used by the chat page sidebar to render the
    "switch session" dropdown. Empty list if no rows or DB error.
    """
    try:
        from engine.db_models import ChatSession, SessionFactory
        with SessionFactory() as s:
            rows = (s.query(ChatSession)
                      .filter(ChatSession.agent_id == agent_id)
                      .order_by(ChatSession.updated_at_utc.desc())
                      .all())
        out: list[dict] = []
        for r in rows:
            try:
                history = json.loads(r.history_json or "[]")
                n_user = sum(
                    1 for m in history
                    if m.get("role") == "user"
                    and isinstance(m.get("content"), str)
                    and m.get("content").strip()
                )
            except Exception:
                n_user = 0
            out.append({
                "session_id":      r.session_id,
                "title":           r.title,
                "updated_at_utc":  r.updated_at_utc,
                "last_visited_at": r.last_visited_at,
                "n_turns":         n_user,
            })
        return out
    except Exception as exc:
        logger.warning("session_store.list_sessions(%s) failed: %s",
                       agent_id, exc)
        return []
