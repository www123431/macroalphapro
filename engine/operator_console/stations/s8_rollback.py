"""S8 — Rollback (safety station for deployed sleeve revert).

Mirrors S7's HUMAN-gated capital-decision pattern, but in the reverse
direction: deployed → previous. Per CLAUDE.md, capital changes stay
HUMAN even when initiated by safety signals (decay alerts, DQ
breaches, halt forensics, etc.).

S8 writes a pending rollback proposal; the principal hits APPROVE on
/approvals to actually mutate the deployed book.

Typical triggers (the conditions under which an operator opens S8):
  - decay_sentinel fires on a deployed sleeve
  - paper-trade NAV diverges materially from expected (>X bps)
  - external LLM audit flags a methodology problem on a deployed
    sleeve
  - halt forensic post-mortem concludes the sleeve is the cause

Design reference: docs/architecture/operator_console.md §5 (S8 spec).

Intentionally simple: 2 deterministic checks + 1 HUMAN gate. Matches
the doctrine that rollback is a careful, deliberate action — not
something to surface as a one-click button.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from pathlib import Path

from engine.operator_console.pipeline_station import (
    PipelineStation,
    SSEEmitter,
    Session,
)
from engine.operator_console.schema import (
    CancellationToken,
    CostEstimate,
    DataTier,
    NextStationHint,
    PreflightCheck,
    PreflightResult,
    PreflightStatus,
    SessionType,
    StationResult,
    StationSpec,
)
from engine.operator_console import emit as opcon_emit
from engine.operator_console import registry


logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROLLBACK_PROPOSALS_PATH = _REPO_ROOT / "data" / "operator_console" / "rollback_proposals.jsonl"


_VALID_TARGETS = {"full_remove", "revert_to_previous_config", "freeze_only"}


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _resolve_deployed_sleeve(sleeve_id: str) -> dict | None:
    """Lookup a sleeve in the active deployed config. Returns minimal
    info dict or None if not deployed."""
    try:
        from engine.portfolio.deployed_registry import load_active
        cfg = load_active()
        for s in cfg.sleeves:
            if s.name == sleeve_id:
                return {
                    "name":         s.name,
                    "role":         s.role,
                    "base_weight":  s.base_weight,
                    "target_vol":   s.target_vol,
                    "regime_modulated": s.regime_modulated,
                    "deploy_date":  cfg.deploy_date,
                    "config_id":    cfg.id,
                }
        return None
    except Exception as e:
        logger.exception("S8: could not load deployed_registry: %s", e)
        return None


def _list_deployed_sleeves() -> list[str]:
    """Return the names of all currently-deployed sleeves (for the
    'did you mean' hint when user types an unknown sleeve_id)."""
    try:
        from engine.portfolio.deployed_registry import load_active
        return list(load_active().sleeve_names)
    except Exception:
        return []


def _write_rollback_proposal(proposal: dict) -> str:
    _ROLLBACK_PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _ROLLBACK_PROPOSALS_PATH.open("a", encoding='utf-8') as f:
        f.write(json.dumps(proposal, ensure_ascii=False) + "\n")
    return proposal["proposal_id"]


# ── The station ──────────────────────────────────────────────────


class Rollback(PipelineStation):
    """S8 — Roll back a deployed sleeve. Safety-first; HUMAN-only."""

    STATION_SPEC = StationSpec(
        station_id              = "S8_rollback",
        title                   = "Rollback",
        description             = (
            "Roll back a deployed sleeve. Writes a pending rollback "
            "proposal and routes to /approvals — never auto-deploys "
            "the rollback. Use when decay_sentinel fires, paper-trade "
            "NAV diverges, audit flags methodology, or halt forensic "
            "concludes the sleeve is the cause."
        ),
        data_tier               = DataTier.SNAPSHOT_DATA,
        # audit + ops session types — the typical contexts for a
        # rollback decision. Not research_new (that's for new factor
        # investigation, not safety actions).
        requires_session_types  = {SessionType.AUDIT, SessionType.OPS},
        estimated_minutes       = 5,
        estimated_cost_usd      = 0.0,
        icon                    = "RotateCw",
        title_key               = "console.station.s8.title",
        description_key         = "console.station.s8.description",
        mutates_capital         = True,   # routes to /approvals via rollback_proposals.jsonl
    )

    def preflight(self, session: Session, config: dict) -> PreflightResult:
        checks: list[PreflightCheck] = []

        if not session or not getattr(session, "session_id", ""):
            checks.append(PreflightCheck("session_active", PreflightStatus.RED,
                                         "No active session."))
        else:
            checks.append(PreflightCheck("session_active", PreflightStatus.GREEN,
                                         f"Session {session.session_id} ready."))

        c = config or {}
        sleeve_id = str(c.get("sleeve_id", "")).strip()
        if not sleeve_id:
            deployed = _list_deployed_sleeves()
            checks.append(PreflightCheck(
                "sleeve_id_provided", PreflightStatus.RED,
                f"Provide sleeve_id. Currently deployed: {deployed}" if deployed else
                "Provide sleeve_id (could not load active deployed config to suggest options).",
            ))
        else:
            info = _resolve_deployed_sleeve(sleeve_id)
            if info is None:
                deployed = _list_deployed_sleeves()
                hint = f" Currently deployed: {deployed}" if deployed else ""
                checks.append(PreflightCheck(
                    "sleeve_resolvable", PreflightStatus.RED,
                    f"sleeve '{sleeve_id}' not in active deployed config.{hint}",
                ))
            else:
                checks.append(PreflightCheck(
                    "sleeve_resolvable", PreflightStatus.GREEN,
                    f"Found deployed sleeve '{sleeve_id}': role={info['role']}, weight={info['base_weight']}, deploy_date={info['deploy_date']}",
                ))

        target = str(c.get("rollback_target", "")).strip()
        if target not in _VALID_TARGETS:
            checks.append(PreflightCheck(
                "rollback_target_valid", PreflightStatus.RED,
                f"rollback_target must be one of: {sorted(_VALID_TARGETS)}; got '{target}'. full_remove=delete sleeve; revert_to_previous_config=use prior config snapshot; freeze_only=disable trading but keep position.",
            ))
        else:
            checks.append(PreflightCheck("rollback_target_valid", PreflightStatus.GREEN,
                                         f"Target='{target}' valid."))

        rationale = str(c.get("rationale", "")).strip()
        if len(rationale) < 30:
            checks.append(PreflightCheck(
                "rationale_substantive", PreflightStatus.RED,
                f"`rationale` MANDATORY ≥30 chars (got {len(rationale)}). Rollback proposals carry an audit trail; the reason is load-bearing for governance review.",
            ))
        else:
            checks.append(PreflightCheck("rationale_substantive", PreflightStatus.GREEN,
                                         f"Rationale length: {len(rationale)} chars."))

        trigger_source = str(c.get("trigger_source", "")).strip()
        if not trigger_source:
            checks.append(PreflightCheck(
                "trigger_source_provided", PreflightStatus.YELLOW,
                "trigger_source recommended (e.g. 'decay_alert:<event_id>' or 'halt_forensic:<id>' or 'audit_review:<verdict_id>') so the rollback links back to its precipitating signal.",
            ))
        else:
            checks.append(PreflightCheck("trigger_source_provided", PreflightStatus.GREEN,
                                         f"Triggered by: {trigger_source[:80]}"))

        # Final gate notice
        checks.append(PreflightCheck(
            "gate_human_approval", PreflightStatus.YELLOW,
            "Final gate = HUMAN approval (per CLAUDE.md capital-decision doctrine). S8 will write a proposal to /approvals and require you to click APPROVE there. Never auto-rolls-back.",
        ))

        return PreflightResult.from_checks(checks)

    def estimate_cost(self, config: dict) -> CostEstimate:
        return CostEstimate(llm_cost_usd_est=0.0, confidence="exact")

    def render_config_form(self) -> dict:
        deployed = _list_deployed_sleeves()
        return {
            "type": "object",
            "title": "Rollback proposal input",
            "description": (
                "Roll back a deployed sleeve. Writes a pending proposal "
                "to /approvals — never auto-rolls-back. "
                + (f"Currently deployed: {deployed}" if deployed else "")
            ),
            "properties": {
                "sleeve_id": {
                    "type": "string",
                    "title": "Sleeve to roll back",
                    "description": "Must match a currently-deployed sleeve name.",
                    "x-ui-widget": "text",
                    "x-ui-placeholder": "e.g. K1_BAB",
                },
                "rollback_target": {
                    "type": "string",
                    "title": "Rollback target",
                    "description": "full_remove: delete sleeve from book. revert_to_previous_config: snap to prior config. freeze_only: stop trading, keep current position.",
                    "enum": ["full_remove", "revert_to_previous_config", "freeze_only"],
                    "default": "freeze_only",
                    "x-ui-widget": "select",
                },
                "trigger_source": {
                    "type": "string",
                    "title": "Trigger source (recommended)",
                    "description": "Link to precipitating signal: decay_alert:<id>, halt_forensic:<id>, audit_review:<id>, etc. Goes to the rollback audit trail.",
                    "x-ui-widget": "text",
                    "default": "",
                },
                "rationale": {
                    "type": "string",
                    "title": "Rationale (mandatory; ≥30 chars)",
                    "description": "Why now? What changed? Why this target vs alternatives? Goes to /approvals for human review.",
                    "x-ui-widget": "text-area",
                    "x-ui-rows": 4,
                },
            },
            "required": ["sleeve_id", "rollback_target", "rationale"],
        }

    async def execute(
        self,
        session: Session,
        config: dict,
        emitter: SSEEmitter,
        cancellation: CancellationToken,
    ) -> StationResult:
        started_ts = _utc_iso()
        actor_id = getattr(session, "actor_id", "principal")
        session_id = getattr(session, "session_id", "")
        c = config or {}
        sleeve_id       = str(c.get("sleeve_id", "")).strip()
        rollback_target = str(c.get("rollback_target", "")).strip()
        rationale       = str(c.get("rationale", "")).strip()
        trigger_source  = str(c.get("trigger_source", "")).strip()

        # ── Stage 1: verify sleeve is in deployed config ─────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "verify_deployed")
        emitter.stage_started("verify_deployed", expected_seconds=1)
        sleeve_info = _resolve_deployed_sleeve(sleeve_id)
        if sleeve_info is None:
            deployed = _list_deployed_sleeves()
            emitter.stage_failed("verify_deployed",
                                   f"sleeve '{sleeve_id}' not deployed. Active: {deployed}")
            return self._refused(session, started_ts, "verify_deployed",
                                 f"sleeve '{sleeve_id}' not in active deployed config")
        emitter.stage_completed("verify_deployed", {
            "sleeve_id":   sleeve_id,
            "role":        sleeve_info["role"],
            "base_weight": sleeve_info["base_weight"],
            "deploy_date": sleeve_info["deploy_date"],
        })

        # ── Stage 2: write rollback proposal ─────────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "write_proposal")
        emitter.stage_started("write_proposal", expected_seconds=1)
        proposal_id = f"rollback_{uuid.uuid4().hex[:12]}"
        proposal = {
            "proposal_id":     proposal_id,
            "ts":              _utc_iso(),
            "sleeve_id":       sleeve_id,
            "sleeve_role":     sleeve_info["role"],
            "current_weight":  sleeve_info["base_weight"],
            "current_config":  sleeve_info["config_id"],
            "rollback_target": rollback_target,
            "trigger_source":  trigger_source,
            "rationale":       rationale[:2000],
            "session_id":      session_id,
            "actor_id":        actor_id,
            "state":           "pending_human_approval",
        }
        try:
            _write_rollback_proposal(proposal)
        except Exception as e:
            emitter.stage_failed("write_proposal", str(e)[:300])
            return self._failed(session, started_ts, "write_proposal", str(e)[:300])
        emitter.stage_completed("write_proposal", {
            "proposal_id":    proposal_id,
            "approvals_link": "/approvals",
        })

        # ── Stage 3: emit station_completed ──────────────────────
        emitter.stage_started("emit_event", expected_seconds=1)
        try:
            opcon_emit.station_completed(
                session_id      = session_id,
                actor_id        = actor_id,
                job_id          = "",
                station_id      = self.STATION_SPEC.station_id,
                cost_actual_usd = 0.0,
                artifacts       = {
                    "proposal_id":     proposal_id,
                    "sleeve_id":       sleeve_id,
                    "rollback_target": rollback_target,
                    "state":           "pending_human_approval",
                },
            )
        except Exception:
            logger.exception("operator_console: failed to emit station_completed")
        emitter.stage_completed("emit_event", {"proposal_id": proposal_id})

        return StationResult(
            job_id          = "",
            station_id      = self.STATION_SPEC.station_id,
            session_id      = session_id,
            actor_id        = actor_id,
            started_ts      = started_ts,
            completed_ts    = _utc_iso(),
            success         = True,
            artifacts       = {
                "outcome":         "AWAITING_HUMAN_APPROVAL",
                "proposal_id":     proposal_id,
                "sleeve_id":       sleeve_id,
                "rollback_target": rollback_target,
                "trigger_source":  trigger_source,
                "next_action":     "Visit /approvals and click APPROVE to execute rollback",
            },
            events_emitted  = [],
            next_stations   = [],   # terminal — human action next
            cost_actual_usd = 0.0,
        )

    def result_lineage(self, result: StationResult) -> list[NextStationHint]:
        return []

    def _refused(self, session: Session, started_ts: str, stage: str, reason: str) -> StationResult:
        return StationResult(
            job_id          = "",
            station_id      = self.STATION_SPEC.station_id,
            session_id      = getattr(session, "session_id", ""),
            actor_id        = getattr(session, "actor_id", "principal"),
            started_ts      = started_ts,
            completed_ts    = _utc_iso(),
            success         = True,
            artifacts       = {
                "outcome":        "REFUSED_AT_GATE",
                "refused_at":     stage,
                "refusal_reason": reason,
            },
            events_emitted  = [],
            next_stations   = [],
            cost_actual_usd = 0.0,
        )

    def _cancelled(self, session: Session, started_ts: str, stage: str) -> StationResult:
        return StationResult(
            job_id        = "",
            station_id    = self.STATION_SPEC.station_id,
            session_id    = getattr(session, "session_id", ""),
            actor_id      = getattr(session, "actor_id", "principal"),
            started_ts    = started_ts,
            completed_ts  = _utc_iso(),
            success       = False,
            error_message = f"Cancelled at stage '{stage}'.",
        )

    def _failed(self, session: Session, started_ts: str, stage: str, err: str) -> StationResult:
        return StationResult(
            job_id        = "",
            station_id    = self.STATION_SPEC.station_id,
            session_id    = getattr(session, "session_id", ""),
            actor_id      = getattr(session, "actor_id", "principal"),
            started_ts    = started_ts,
            completed_ts  = _utc_iso(),
            success       = False,
            error_message = f"Stage '{stage}' failed: {err}",
        )


registry.register(Rollback)
