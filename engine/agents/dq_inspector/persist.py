"""
engine/agents/dq_inspector/persist.py — Phase 5 persistence layer.

Mirrors engine.agents.risk_manager.persist 1:1 plus source_id column
unique to DQ. Deterministic uuid5 alert_id keyed on (date, mode_id,
source_id, affected_canonical) for idempotent UPSERT.

No LLM calls. Narrator updates the narrative_text column AFTER the
deterministic detector path completes (Phase 7).
"""
from __future__ import annotations

import datetime
import json
import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.agents.dq_inspector.gates import Breach

from engine.agents.dq_inspector.gates import classify_severity

logger = logging.getLogger(__name__)


# DQ namespace UUID — distinct from RM's namespace so the two ID spaces
# never collide (RM = 1c3e7a4b… for trades + a4f9c2e1… for alerts;
# DQ gets its own).
_DQ_NS = uuid.UUID("d4a7f1b8-cccc-7777-bbbb-feedbeefcafe")


def _canonical_affected(affected: tuple[str, ...]) -> str:
    return "|".join(sorted(affected))


def make_alert_id(
    date:        datetime.date,
    mode_id:     str,
    source_id:   str,
    affected:    tuple[str, ...],
) -> str:
    """Deterministic alert_id: same inputs → same UUID."""
    key = f"{date.isoformat()}|{mode_id}|{source_id}|{_canonical_affected(affected)}"
    return str(uuid.uuid5(_DQ_NS, key))


def breach_to_alert_row(
    breach:        "Breach",
    date:          datetime.date,
    phase:         str,
    cb_severity:   str,
    halt_decision: bool,
) -> dict:
    """Pure function — Breach → kwargs dict for DataQualityAlert."""
    source_id = breach.extra.get("source_id", "unknown")
    return {
        "date":             date,
        "alert_id":         make_alert_id(date, breach.mode_id, source_id, breach.affected),
        "mode_id":          breach.mode_id,
        "severity":         breach.severity,
        "cb_severity":      cb_severity,
        "halt_decision":    halt_decision,
        "phase":            phase,
        "source_id":        source_id,
        "rule_description": breach.rule_description,
        "observed_value":   (None if breach.observed_value is None
                             or (isinstance(breach.observed_value, float)
                                 and breach.observed_value != breach.observed_value)
                             else float(breach.observed_value)),
        "threshold":        (None if breach.threshold is None
                             or (isinstance(breach.threshold, float)
                                 and breach.threshold != breach.threshold)
                             else float(breach.threshold)),
        "affected_json":    json.dumps(list(breach.affected), ensure_ascii=False),
        "extra_json":       json.dumps(breach.extra, ensure_ascii=False, default=str),
        "narrative_text":   None,
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
    """UPSERT all breaches as DataQualityAlert rows. Returns alert_ids written."""
    if not breaches:
        return []
    from engine.db_models import DataQualityAlert
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
            obj = DataQualityAlert(**row_kwargs)
            sess.merge(obj)
            written_ids.append(row_kwargs["alert_id"])
        sess.commit()
        logger.info(
            "dq_inspector.persist: wrote %d alerts for %s phase=%s cb_severity=%s halt=%s",
            len(written_ids), date, phase, cb_sev, halt_decision,
        )
    except Exception:
        sess.rollback()
        logger.exception("dq_inspector.persist: write failed; transaction rolled back")
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
    """Phase 7 narrator hook — populate prose fields on existing row."""
    from engine.db_models import DataQualityAlert
    from engine.memory import SessionFactory

    sess = SessionFactory()
    try:
        row = sess.query(DataQualityAlert).filter_by(
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
        logger.exception("dq_inspector.persist.update_narrative failed")
        raise
    finally:
        sess.close()


def query_recent_alerts(
    days_back:    int = 7,
    severity_min: str = "LIGHT",
) -> list[dict]:
    """Read tool — used by Risk Console UI panel + cross-agent reference."""
    from engine.db_models import DataQualityAlert
    from engine.memory import SessionFactory

    severity_order = {"NONE": 0, "LIGHT": 1, "MEDIUM": 2, "SEVERE": 3}
    min_rank = severity_order.get(severity_min, 1)

    cutoff = datetime.date.today() - datetime.timedelta(days=days_back)
    sess = SessionFactory()
    try:
        rows = (
            sess.query(DataQualityAlert)
            .filter(DataQualityAlert.date >= cutoff)
            .order_by(DataQualityAlert.date.desc(),
                      DataQualityAlert.generated_at_utc.desc())
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
                "source_id":        r.source_id,
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
