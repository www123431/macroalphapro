"""
engine/agents/risk_manager/persist.py — Phase 4 persistence layer.

Writes RiskManagerAlert rows for every Breach produced by the gates
module. Deterministic UUID5 alert_id so re-running the orchestrator
for the same date with the same gate output produces idempotent UPSERT.

DOCTRINE compliance:
  - No LLM calls here (deterministic persistence only)
  - Narrative fields (narrative_text / narrative_cost_usd) start NULL;
    Phase 7 narrator updates them in a separate transaction
  - Composite PK (date, alert_id) guarantees no duplicate alerts
"""
from __future__ import annotations

import datetime
import json
import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.agents.risk_manager.gates import Breach

from engine.agents.risk_manager.gates import classify_severity

logger = logging.getLogger(__name__)


# Deterministic UUID namespace for Risk Manager alert IDs.
# Distinct from Sprint H trade ID namespace so the two ID spaces don't collide.
_RISK_MGR_NS = uuid.UUID("a4f9c2e1-bbbb-5555-8888-1337beef0420")


def _canonical_affected(affected: tuple[str, ...]) -> str:
    """Stable canonical string for hashing — sorted + joined."""
    return "|".join(sorted(affected))


def make_alert_id(date: datetime.date, mode_id: str, affected: tuple[str, ...]) -> str:
    """Deterministic alert_id from (date, mode_id, affected) tuple.

    Same breach on same date → same UUID → UPSERT cleanly.
    """
    key = f"{date.isoformat()}|{mode_id}|{_canonical_affected(affected)}"
    return str(uuid.uuid5(_RISK_MGR_NS, key))


def breach_to_alert_row(
    breach:        "Breach",
    date:          datetime.date,
    phase:         str,
    cb_severity:   str,
    halt_decision: bool,
) -> dict:
    """Convert a Breach dataclass into RiskManagerAlert row kwargs.

    Separated from persist for testability — pure function, no DB.
    """
    return {
        "date":             date,
        "alert_id":         make_alert_id(date, breach.mode_id, breach.affected),
        "mode_id":          breach.mode_id,
        "severity":         breach.severity,                  # HARD_HALT / SOFT_WARN
        "cb_severity":      cb_severity,                      # NONE/LIGHT/MEDIUM/SEVERE
        "halt_decision":    halt_decision,
        "phase":            phase,
        "rule_description": breach.rule_description,
        "observed_value":   (None if breach.observed_value is None
                             or (isinstance(breach.observed_value, float)
                                 and breach.observed_value != breach.observed_value)  # nan check
                             else float(breach.observed_value)),
        "threshold":        (None if breach.threshold is None
                             or (isinstance(breach.threshold, float)
                                 and breach.threshold != breach.threshold)
                             else float(breach.threshold)),
        "affected_json":    json.dumps(list(breach.affected), ensure_ascii=False),
        "extra_json":       json.dumps(breach.extra, ensure_ascii=False, default=str),
        "narrative_text":   None,        # Phase 7 narrator populates
        "narrative_cost_usd": None,
        "spec_anchor":      breach.spec_anchor,
        "generated_at_utc": datetime.datetime.utcnow(),
    }


def persist_breaches_to_db(
    breaches:      list["Breach"],
    date:          datetime.date,
    phase:         str,
    halt_decision: bool,
) -> list[str]:
    """SQLAlchemy UPSERT all breaches as RiskManagerAlert rows.

    Returns list of alert_ids written. Uses session.merge() for portable
    upsert (SQLite + PostgreSQL).

    cb_severity is computed ONCE per call from the full breach list, so
    every alert row from the same orchestrator cycle carries the same
    aggregate severity (audit consistency).
    """
    if not breaches:
        return []

    from engine.db_models import RiskManagerAlert
    from engine.memory import init_db, SessionFactory

    init_db()
    cb_sev = classify_severity(breaches)

    written_ids: list[str] = []
    sess = SessionFactory()
    try:
        for breach in breaches:
            row_kwargs = breach_to_alert_row(
                breach        = breach,
                date          = date,
                phase         = phase,
                cb_severity   = cb_sev,
                halt_decision = halt_decision,
            )
            obj = RiskManagerAlert(**row_kwargs)
            sess.merge(obj)
            written_ids.append(row_kwargs["alert_id"])
        sess.commit()
        logger.info(
            "risk_manager.persist: wrote %d alerts for %s phase=%s cb_severity=%s halt=%s",
            len(written_ids), date, phase, cb_sev, halt_decision,
        )
    except Exception:
        sess.rollback()
        logger.exception("risk_manager.persist: write failed; transaction rolled back")
        raise
    finally:
        sess.close()
    return written_ids


def update_narrative(
    date:           datetime.date,
    alert_id:       str,
    narrative_text: str,
    cost_usd:       float,
) -> bool:
    """Phase 7 narrator updates the prose fields on an existing row.

    Returns True if a row was found and updated; False if no matching row.
    Separated from persist_breaches_to_db so narration can run async after
    the daily orchestrator's main cycle.
    """
    from engine.db_models import RiskManagerAlert
    from engine.memory import SessionFactory

    sess = SessionFactory()
    try:
        row = sess.query(RiskManagerAlert).filter_by(
            date=date, alert_id=alert_id
        ).first()
        if row is None:
            return False
        row.narrative_text = narrative_text
        row.narrative_cost_usd = float(cost_usd)
        sess.commit()
        return True
    except Exception:
        sess.rollback()
        logger.exception("risk_manager.persist.update_narrative: failed")
        raise
    finally:
        sess.close()


def query_recent_alerts(
    days_back: int = 7,
    severity_min: str = "LIGHT",
) -> list[dict]:
    """Read tool used by other agents (cross-agent reference, Pattern 1).

    Returns list of alert dicts with severity >= threshold over the last
    N days. Default 7 days × LIGHT cap shows everything except clean days.

    Used by:
      - Anomaly Sentinel (correlation between anomalies + recent halts)
      - Audit Recorder (DD investigation context lookup)
      - Risk Console dashboard (alert feed)
    """
    from engine.db_models import RiskManagerAlert
    from engine.memory import SessionFactory

    severity_order = {"NONE": 0, "LIGHT": 1, "MEDIUM": 2, "SEVERE": 3}
    min_rank = severity_order.get(severity_min, 1)

    cutoff = datetime.date.today() - datetime.timedelta(days=days_back)
    sess = SessionFactory()
    try:
        rows = (
            sess.query(RiskManagerAlert)
            .filter(RiskManagerAlert.date >= cutoff)
            .order_by(RiskManagerAlert.date.desc(),
                      RiskManagerAlert.generated_at_utc.desc())
            .all()
        )
        out = []
        for r in rows:
            if severity_order.get(r.cb_severity, 0) < min_rank:
                continue
            out.append({
                "date":             r.date,
                "alert_id":         r.alert_id,
                "mode_id":          r.mode_id,
                "severity":         r.severity,
                "cb_severity":      r.cb_severity,
                "halt_decision":    r.halt_decision,
                "phase":            r.phase,
                "rule_description": r.rule_description,
                "observed_value":   r.observed_value,
                "threshold":        r.threshold,
                "affected":         json.loads(r.affected_json),
                "extra":            json.loads(r.extra_json),
                "narrative_text":   r.narrative_text,
                "spec_anchor":      r.spec_anchor,
                "generated_at_utc": r.generated_at_utc,
            })
        return out
    finally:
        sess.close()
