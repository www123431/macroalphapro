"""
Agent base class + dataclasses for triggers / results / events.

Spec: docs/spec_factor_mad_redesign.md §2.2.

Design notes
------------
* `Agent.run()` is the only mandatory subclass method. Subclasses call
  `_persist_run()` themselves before/after work; the base class supplies
  the helpers but does not wrap `run()` in a transaction (some agents
  may want multi-stage check-pointing).
* Events are first-class: cross-agent coordination flows through
  `engine.agents.event_bus.EventBus`. The base class exposes
  `_emit_event()` so subclasses don't need to import the bus directly.
* `_claim_lock()` uses SystemConfig as the lock store. We deliberately
  do NOT add a separate `agent_locks` table — SystemConfig already has
  upsert semantics and is read-light, lock churn here is < 1/min.
"""
from __future__ import annotations

import datetime
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Trigger:
    """How an agent run was kicked off."""
    type: str            # "scheduled" | "event" | "manual"
    source: str          # "quarterly_tick" | "regime.switch" | "supervisor:zhang"
    payload: dict = field(default_factory=dict)

    def label(self) -> str:
        """Compact string for AgentRun.triggered_by."""
        return f"{self.type}:{self.source}"


@dataclass
class AgentEvent:
    """Cross-agent event payload. Persisted in agent_events."""
    event_type: str                                  # "regime.switch" | "factor.decay" | …
    payload: dict = field(default_factory=dict)
    source_agent: str | None = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    occurred_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


@dataclass
class AgentResult:
    """Outcome of one Agent.run() invocation. Persisted in agent_runs."""
    run_id: str
    agent_id: str
    status: str                                       # succeeded|failed|interrupted|running
    started_at: datetime.datetime
    finished_at: datetime.datetime | None = None
    summary: dict = field(default_factory=dict)
    events_emitted: list[str] = field(default_factory=list)   # event_id list
    error: str | None = None
    triggered_by: str = ""
    parent_run_id: str | None = None
    state: str | None = None                          # e.g. "validating"
    input_params: dict = field(default_factory=dict)


# ── Base Agent ────────────────────────────────────────────────────────────────

class Agent(ABC):
    """Base class for all operations-architecture agents."""

    AGENT_ID: str = "abstract"   # subclasses must override

    # ── Public contract ───────────────────────────────────────────────────

    @abstractmethod
    def run(self, trigger: Trigger, as_of: datetime.date) -> AgentResult:
        """Execute one cycle of work. Subclass MUST persist its own AgentResult."""
        raise NotImplementedError

    def get_health(self, as_of: datetime.date) -> dict:
        """
        Default health summary: count of recent runs by status.
        Subclasses override for agent-specific KPIs (e.g. ICIR for FactorMAD).
        """
        from engine.memory import AgentRun, SessionFactory
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=30)
        with SessionFactory() as session:
            rows = (
                session.query(AgentRun.status, AgentRun.id)
                .filter(AgentRun.agent_id == self.AGENT_ID,
                        AgentRun.started_at >= cutoff)
                .all()
            )
        counts: dict[str, int] = {}
        for status, _ in rows:
            counts[status] = counts.get(status, 0) + 1
        return {"agent_id": self.AGENT_ID, "as_of": str(as_of), "last_30d": counts}

    # ── Helpers for subclasses ────────────────────────────────────────────

    def _new_run(
        self,
        trigger: Trigger,
        parent_run_id: str | None = None,
        input_params: dict | None = None,
    ) -> AgentResult:
        """Create an in-progress AgentResult (caller persists it)."""
        return AgentResult(
            run_id=uuid.uuid4().hex,
            agent_id=self.AGENT_ID,
            status="running",
            started_at=datetime.datetime.utcnow(),
            triggered_by=trigger.label(),
            parent_run_id=parent_run_id,
            input_params=input_params or {},
        )

    def _persist_run(self, run: AgentResult) -> None:
        """
        Upsert AgentRun row. Idempotent on run_id — call once at start
        (status='running') and again at end with the final status/summary.
        """
        from engine.memory import AgentRun, SessionFactory
        with SessionFactory() as session:
            row = (
                session.query(AgentRun)
                .filter_by(run_id=run.run_id)
                .one_or_none()
            )
            if row is None:
                row = AgentRun(
                    run_id=run.run_id,
                    agent_id=run.agent_id,
                    triggered_by=run.triggered_by or "manual:unknown",
                    status=run.status,
                    state=run.state,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    input_params=_to_json(run.input_params),
                    output_summary=_to_json(run.summary),
                    error=run.error,
                    parent_run_id=run.parent_run_id,
                )
                session.add(row)
            else:
                row.status = run.status
                row.state = run.state
                row.finished_at = run.finished_at
                row.output_summary = _to_json(run.summary)
                row.error = run.error
                if run.parent_run_id and not row.parent_run_id:
                    row.parent_run_id = run.parent_run_id
            session.commit()
        logger.debug("Persisted AgentRun run_id=%s agent=%s status=%s",
                     run.run_id, run.agent_id, run.status)

    def _emit_event(self, event: AgentEvent) -> str:
        """Publish an event onto the bus. Returns event_id."""
        from engine.agents.event_bus import get_event_bus
        bus = get_event_bus()
        if event.source_agent is None:
            event.source_agent = self.AGENT_ID
        return bus.publish_event(event)

    def _claim_lock(self, key: str, ttl_seconds: int = 3600) -> bool:
        """
        Best-effort exclusive lock via system_config. Returns True on success.
        TTL prevents stuck locks if the holder crashes.
        Lock key format: "agent_lock:<AGENT_ID>:<key>".
        """
        from sqlalchemy import text as _text
        from engine.memory import engine as _engine
        lock_key = f"agent_lock:{self.AGENT_ID}:{key}"
        now = datetime.datetime.utcnow()
        expires = now + datetime.timedelta(seconds=ttl_seconds)
        new_value = json.dumps({
            "holder": self.AGENT_ID,
            "claimed_at": now.isoformat(),
            "expires_at": expires.isoformat(),
        })
        with _engine.connect() as conn:
            row = conn.execute(
                _text("SELECT value FROM system_config WHERE key = :k"),
                {"k": lock_key},
            ).fetchone()
            if row is not None:
                try:
                    cur = json.loads(row[0]) if row[0] else {}
                    exp = datetime.datetime.fromisoformat(cur.get("expires_at", ""))
                    if exp > now:
                        # active lock held by someone (possibly self) → fail
                        return False
                except Exception:
                    pass  # corrupt lock → overwrite
                conn.execute(
                    _text("UPDATE system_config SET value = :v WHERE key = :k"),
                    {"v": new_value, "k": lock_key},
                )
            else:
                conn.execute(
                    _text("INSERT INTO system_config (key, value) VALUES (:k, :v)"),
                    {"k": lock_key, "v": new_value},
                )
            conn.commit()
        return True

    def _release_lock(self, key: str) -> None:
        from sqlalchemy import text as _text
        from engine.memory import engine as _engine
        lock_key = f"agent_lock:{self.AGENT_ID}:{key}"
        with _engine.connect() as conn:
            conn.execute(
                _text("DELETE FROM system_config WHERE key = :k"),
                {"k": lock_key},
            )
            conn.commit()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_json(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(asdict(obj) if hasattr(obj, "__dataclass_fields__")
                          else str(obj), ensure_ascii=False)
