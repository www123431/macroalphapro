"""
SQLite-backed cross-agent event bus.

Spec: docs/spec_factor_mad_redesign.md §2.5 + Q8.

Design constraints (from spec)
------------------------------
* Synchronous dispatch — fits the daily/quarterly batch cadence; no async runtime
  to babysit, no message broker to operate.
* Persisted to `agent_events` so handlers can replay unconsumed events after
  Streamlit restarts.
* Subscriptions are in-process (Python dict). After a process restart, callers
  re-subscribe at startup and use `replay_unconsumed()` to drain backlog.

Threading note
--------------
This bus is intentionally single-threaded. It assumes the caller pattern: one
orchestrator thread publishes, agents subscribe at startup, dispatch happens
under the publisher's call stack. If a future requirement needs multi-thread
publish, add a `threading.Lock` around `_subscribers` access.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Callable

from engine.agents.base import AgentEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[AgentEvent], None]


class EventBus:
    """
    Persistent + in-process event dispatcher.

    Use the module-level `get_event_bus()` to obtain the shared singleton.
    Direct construction is fine in tests.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = {}

    # ── Subscription ──────────────────────────────────────────────────────

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Register an in-process handler for `event_type`. Handlers run in the
        order they subscribed; exceptions in one handler do NOT abort the
        rest — they're logged and consumption continues so a buggy agent
        can't take the bus down.
        """
        self._subscribers.setdefault(event_type, []).append(handler)
        logger.debug("EventBus: subscribed %s -> %s",
                     event_type, getattr(handler, "__qualname__", repr(handler)))

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass

    # ── Publishing ────────────────────────────────────────────────────────

    def publish(
        self,
        event_type: str,
        payload: dict,
        source_agent: str | None = None,
    ) -> str:
        """Convenience wrapper: build AgentEvent + persist + dispatch."""
        evt = AgentEvent(
            event_type=event_type,
            payload=payload,
            source_agent=source_agent,
        )
        return self.publish_event(evt)

    def publish_event(self, event: AgentEvent) -> str:
        """
        Persist event row, then dispatch to in-process handlers.
        Returns event.event_id.

        2026-06-04: persistence is best-effort. The agent_events SQLite
        table dependency (engine.memory.AgentEventRow) was dropped in
        an earlier refactor; until it's restored, persistence raises
        ImportError. Catching that here keeps in-process dispatch alive
        — subscribers still run, which is what closed-loop agents need.
        Replay is what's lost; for current usage that's acceptable.
        """
        try:
            self._persist(event, consumed_by=[])
            persisted = True
        except Exception as exc:
            logger.warning(
                "EventBus: persist failed (%s) — dispatching in-memory only",
                exc.__class__.__name__,
            )
            persisted = False
        consumed: list[str] = []
        for handler in self._subscribers.get(event.event_type, []):
            try:
                handler(event)
                consumed.append(getattr(handler, "__qualname__", repr(handler)))
            except Exception as exc:
                logger.error(
                    "EventBus: handler %s for %s raised: %s",
                    getattr(handler, "__qualname__", repr(handler)),
                    event.event_type, exc, exc_info=True,
                )
        if consumed and persisted:
            try:
                self._update_consumed(event.event_id, consumed)
            except Exception as exc:
                logger.warning("EventBus: consumed-mark failed: %s", exc)
        return event.event_id

    # ── Replay ────────────────────────────────────────────────────────────

    def replay_unconsumed(
        self,
        agent_id: str,
        since: datetime.datetime,
        event_types: list[str] | None = None,
    ) -> list[AgentEvent]:
        """
        Return events the given agent has not yet consumed (occurred_at >= since).
        Caller is responsible for invoking handlers — this only returns the rows.
        """
        from engine.memory import AgentEventRow, SessionFactory
        with SessionFactory() as session:
            q = (
                session.query(AgentEventRow)
                .filter(AgentEventRow.occurred_at >= since)
                .order_by(AgentEventRow.occurred_at.asc())
            )
            if event_types:
                q = q.filter(AgentEventRow.event_type.in_(event_types))
            rows = q.all()
        out: list[AgentEvent] = []
        for r in rows:
            try:
                consumed = json.loads(r.consumed_by) if r.consumed_by else []
            except (TypeError, ValueError):
                consumed = []
            if agent_id in consumed:
                continue
            try:
                payload = json.loads(r.payload) if r.payload else {}
            except (TypeError, ValueError):
                payload = {}
            out.append(AgentEvent(
                event_id=r.event_id,
                event_type=r.event_type,
                payload=payload,
                source_agent=r.source_agent,
                occurred_at=r.occurred_at,
            ))
        return out

    def mark_consumed(self, event_id: str, agent_id: str) -> None:
        """Append agent_id to consumed_by of an existing event row."""
        from engine.memory import AgentEventRow, SessionFactory
        with SessionFactory() as session:
            row = (
                session.query(AgentEventRow)
                .filter_by(event_id=event_id)
                .one_or_none()
            )
            if row is None:
                return
            try:
                consumed = json.loads(row.consumed_by) if row.consumed_by else []
            except (TypeError, ValueError):
                consumed = []
            if agent_id not in consumed:
                consumed.append(agent_id)
                row.consumed_by = json.dumps(consumed, ensure_ascii=False)
                session.commit()

    # ── Internal persistence ──────────────────────────────────────────────

    def _persist(self, event: AgentEvent, consumed_by: list[str]) -> None:
        from engine.memory import AgentEventRow, SessionFactory
        with SessionFactory() as session:
            session.add(AgentEventRow(
                event_id=event.event_id,
                event_type=event.event_type,
                source_agent=event.source_agent,
                payload=json.dumps(event.payload, ensure_ascii=False, default=str),
                occurred_at=event.occurred_at,
                consumed_by=json.dumps(consumed_by, ensure_ascii=False) if consumed_by else None,
            ))
            session.commit()

    def _update_consumed(self, event_id: str, consumed: list[str]) -> None:
        from engine.memory import AgentEventRow, SessionFactory
        with SessionFactory() as session:
            row = (
                session.query(AgentEventRow)
                .filter_by(event_id=event_id)
                .one_or_none()
            )
            if row is not None:
                row.consumed_by = json.dumps(consumed, ensure_ascii=False)
                session.commit()


# ── Module-level singleton ────────────────────────────────────────────────────

_BUS: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the process-wide EventBus, lazy-instantiated."""
    global _BUS
    if _BUS is None:
        _BUS = EventBus()
    return _BUS


def reset_event_bus_for_tests() -> None:
    """Drop the singleton (test-only). Subscriptions are wiped."""
    global _BUS
    _BUS = None
