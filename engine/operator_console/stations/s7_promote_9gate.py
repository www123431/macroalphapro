"""S7 — PROMOTE 9-gate (institutional capital-decision discipline).

The most-rigorous Pipeline Station: 9 sequential gates between a GREEN
FORWARD verdict and a deployed sleeve. Per CLAUDE.md doctrine, capital
decisions stay HUMAN-only — even when all 8 deterministic gates pass,
Gate 9 (human approval) NEVER auto-fires.

MVP scope (Phase 2 honest delivery):
  Gate 1  ✅ Verdict is GREEN          — deterministic; implemented
  Gate 2  ⏳ Cost-robust                — Phase 2 polish; YELLOW info
  Gate 3  ⏳ PIT clean                  — Phase 2 polish; YELLOW info
  Gate 4  ⏳ Replication (γ persona)    — Phase 2 polish; YELLOW info
  Gate 5  ⏳ Multi-period stability     — Phase 2 polish; YELLOW info
  Gate 6  ⏳ Anchor-residual            — Phase 2 polish; YELLOW info
  Gate 7  ⏳ Cross-sleeve correlation   — Phase 2 polish; YELLOW info
  Gate 8  ⏳ Capacity (Pastor-Stambaugh)— Phase 2 polish; YELLOW info
  Gate 9  ✅ HUMAN approval              — implemented; writes a
                                          promote_proposal row and
                                          routes to /approvals

When all 8 deterministic gates are implemented, only Gate 1 changes
from "deterministic verify" to "deterministic verify of upstream
verdict that itself ran gates 2-8 in a prior pipeline." S7's job is
the orchestration layer + the human handoff, not re-running each
statistical test.

Design reference: docs/architecture/operator_console.md §5 (S7 spec).
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
_EVENTS_PATH       = _REPO_ROOT / "data" / "research_store" / "events.jsonl"
_PROPOSALS_PATH    = _REPO_ROOT / "data" / "operator_console" / "promote_proposals.jsonl"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _find_verdict_event(verdict_event_id: str) -> dict | None:
    if not _EVENTS_PATH.is_file():
        return None
    for line in _EVENTS_PATH.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            continue
        if row.get("event_id") == verdict_event_id:
            return row
    return None


def _write_promote_proposal(proposal: dict) -> str:
    """Append a promote-proposal row + return proposal_id. Drives the
    /approvals UI surface for human review per the capital-decision
    doctrine."""
    _PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _PROPOSALS_PATH.open("a", encoding='utf-8') as f:
        f.write(json.dumps(proposal, ensure_ascii=False) + "\n")
    return proposal["proposal_id"]


# ── The station ──────────────────────────────────────────────────


class Promote9Gate(PipelineStation):
    """S7 — PROMOTE a GREEN verdict to deployment via 9-gate workflow.

    Capital decision lives at Gate 9. S7 prepares the proposal +
    routes to /approvals; the principal hits APPROVE there to actually
    deploy."""

    STATION_SPEC = StationSpec(
        station_id              = "S7_promote_9gate",
        title                   = "PROMOTE 9-gate",
        description             = (
            "Route a GREEN FORWARD verdict through the 9-gate promote "
            "workflow. Gate 1 (GREEN verify) + Gate 9 (HUMAN approval) "
            "are wired; Gates 2-8 (cost-robust / PIT / replication / "
            "multi-period / anchor-residual / cross-sleeve correlation / "
            "capacity) surface as deferred-info in MVP. Capital decision "
            "ALWAYS stays human — S7 never auto-deploys."
        ),
        data_tier               = DataTier.SNAPSHOT_DATA,
        requires_session_types  = {SessionType.RESEARCH_NEW},
        estimated_minutes       = 3,
        estimated_cost_usd      = 0.0,
        icon                    = "ShieldCheck",
        title_key               = "console.station.s7.title",
        description_key         = "console.station.s7.description",
        mutates_capital         = True,   # routes to /approvals via promote_proposals.jsonl
    )

    def preflight(self, session: Session, config: dict) -> PreflightResult:
        checks: list[PreflightCheck] = []

        if not session or not getattr(session, "session_id", ""):
            checks.append(PreflightCheck("session_active", PreflightStatus.RED,
                                         "No active session."))
        else:
            checks.append(PreflightCheck("session_active", PreflightStatus.GREEN,
                                         f"Session {session.session_id} ready."))

        verdict_event_id = str((config or {}).get("verdict_event_id", "")).strip()
        if not verdict_event_id:
            checks.append(PreflightCheck("verdict_event_id_provided", PreflightStatus.RED,
                                         "Provide verdict_event_id."))
            return PreflightResult.from_checks(checks)

        ev = _find_verdict_event(verdict_event_id)
        if ev is None:
            checks.append(PreflightCheck(
                "gate1_verdict_resolvable", PreflightStatus.RED,
                f"verdict_event_id '{verdict_event_id}' not in events.jsonl.",
            ))
            return PreflightResult.from_checks(checks)

        # Gate 1: must be a factor_verdict_filed event AND verdict=GREEN
        if ev.get("event_type") != "factor_verdict_filed":
            checks.append(PreflightCheck(
                "gate1_event_type", PreflightStatus.RED,
                f"Event type is '{ev.get('event_type')}', not factor_verdict_filed. PROMOTE only applies to FORWARD verdicts.",
            ))
        else:
            checks.append(PreflightCheck("gate1_event_type", PreflightStatus.GREEN,
                                         "Event is factor_verdict_filed."))

        payload = ev.get("payload", {}) or {}
        verdict = payload.get("verdict", "")
        if verdict == "GREEN":
            checks.append(PreflightCheck("gate1_verdict_is_green", PreflightStatus.GREEN,
                                         f"Verdict=GREEN (subject_id={ev.get('subject_id', '?')})."))
        elif verdict in ("MARGINAL", "RED"):
            checks.append(PreflightCheck(
                "gate1_verdict_is_green", PreflightStatus.RED,
                f"Verdict='{verdict}' — only GREEN verdicts can be PROMOTED.",
            ))
        else:
            # Legacy events have empty payload → can't determine verdict
            checks.append(PreflightCheck(
                "gate1_verdict_is_green", PreflightStatus.YELLOW,
                f"Verdict missing from payload (legacy event?). Proceed at your own risk; this likely won't promote correctly.",
            ))

        # Gates 2-8: deferred — show as YELLOW info so user sees the full doctrine
        deferred_gates = [
            ("gate2_cost_robust",         "Cost-robust (Almgren-Chriss optimal execution gap)"),
            ("gate3_pit_clean",           "PIT clean (look-ahead audit)"),
            ("gate4_replication",         "Replication (γ persona confirms paper)"),
            ("gate5_multi_period",        "Multi-period stability (Mann-Kendall across 5 sub-periods)"),
            ("gate6_anchor_residual",     "Anchor-residual (post-FF5+MOM residual Sharpe > threshold)"),
            ("gate7_cross_sleeve_corr",   "Cross-sleeve correlation (vs each deployed sleeve)"),
            ("gate8_capacity",            "Capacity (Pastor-Stambaugh / Berk-Green ceiling)"),
        ]
        for name, desc in deferred_gates:
            checks.append(PreflightCheck(
                name, PreflightStatus.YELLOW,
                f"DEFERRED (Phase 2 polish): {desc}. Will be SKIPPED in execute(); human reviewer should verify manually before approving at Gate 9.",
            ))

        # Gate 9: human approval — this is the GATE, not a check
        checks.append(PreflightCheck(
            "gate9_human_approval", PreflightStatus.YELLOW,
            "Gate 9 = HUMAN approval (per CLAUDE.md capital-decision doctrine). S7 will NEVER auto-promote — it writes a proposal to /approvals and requires you to click APPROVE there.",
        ))

        return PreflightResult.from_checks(checks)

    def estimate_cost(self, config: dict) -> CostEstimate:
        return CostEstimate(llm_cost_usd_est=0.0, confidence="exact")

    def render_config_form(self) -> dict:
        return {
            "type": "object",
            "title": "PROMOTE 9-gate input",
            "description": (
                "Promote a GREEN FORWARD verdict to deployment. The "
                "actual deploy happens only after human APPROVE on the "
                "/approvals page — S7 prepares the proposal."
            ),
            "properties": {
                "verdict_event_id": {
                    "type": "string",
                    "title": "verdict_event_id (must be GREEN factor_verdict_filed)",
                    "description": "Pick from S4 result artifacts or /research/lessons.",
                    "x-ui-widget": "text",
                    "x-ui-placeholder": "e.g. 86b4ebac-ef9d-...",
                },
                "target_weight": {
                    "type": "number",
                    "title": "Target weight in book (0.0 — 1.0)",
                    "description": "Suggested allocation; final weight set by human at /approvals.",
                    "default": 0.05,
                    "minimum": 0.0,
                    "maximum": 0.50,
                    "x-ui-widget": "text",
                },
                "role": {
                    "type": "string",
                    "title": "Sleeve role classification",
                    "description": "Per Markowitz/Frazzini-Pedersen baseline: insurance evaluated by crisis_payoff not Sharpe.",
                    "enum": ["alpha", "insurance", "regime_premium", "trend"],
                    "default": "alpha",
                    "x-ui-widget": "select",
                },
                "rationale": {
                    "type": "string",
                    "title": "Promotion rationale (mandatory; goes to /approvals)",
                    "description": "Why does the human reviewer benefit from approving? Lands in the proposal audit trail.",
                    "x-ui-widget": "text-area",
                    "x-ui-rows": 3,
                },
            },
            "required": ["verdict_event_id", "rationale"],
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
        verdict_event_id = str(c.get("verdict_event_id", "")).strip()
        target_weight    = float(c.get("target_weight", 0.05) or 0.05)
        role             = str(c.get("role", "alpha")).strip()
        rationale        = str(c.get("rationale", "")).strip()

        # ── Gate 1: deterministic GREEN verify ────────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "gate1_verify_green")
        emitter.stage_started("gate1_verify_green", expected_seconds=1)
        ev = _find_verdict_event(verdict_event_id)
        if ev is None:
            emitter.stage_failed("gate1_verify_green",
                                   f"verdict_event_id '{verdict_event_id}' not in events.jsonl")
            return self._failed(session, started_ts, "gate1_verify_green",
                                f"verdict not found")

        payload = ev.get("payload", {}) or {}
        verdict = payload.get("verdict", "")
        if verdict != "GREEN":
            emitter.stage_failed(
                "gate1_verify_green",
                f"Refused: verdict='{verdict or '(missing)'}', need GREEN.",
            )
            return self._refused(session, started_ts, "gate1_verify_green",
                                 f"verdict='{verdict}'; only GREEN can be promoted")
        emitter.stage_completed("gate1_verify_green", {
            "subject_id": ev.get("subject_id", ""),
            "verdict": "GREEN",
        })

        # ── Gates 2-8: stubbed deferred ──────────────────────────
        emitter.stage_started("gates_2_through_8_deferred", expected_seconds=1)
        deferred = [
            "gate2_cost_robust",
            "gate3_pit_clean",
            "gate4_replication",
            "gate5_multi_period",
            "gate6_anchor_residual",
            "gate7_cross_sleeve_corr",
            "gate8_capacity",
        ]
        emitter.stage_completed("gates_2_through_8_deferred", {
            "deferred_gates": deferred,
            "note": "Phase 2 polish — human reviewer should verify manually at /approvals before clicking APPROVE.",
        })

        # ── Gate 9: HUMAN approval — write proposal to /approvals ─
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "gate9_human_approval")
        emitter.stage_started("gate9_human_approval", expected_seconds=1)
        proposal_id = f"promote_{uuid.uuid4().hex[:12]}"
        proposal = {
            "proposal_id":      proposal_id,
            "ts":               _utc_iso(),
            "verdict_event_id": verdict_event_id,
            "subject_id":       ev.get("subject_id", ""),
            "target_weight":    target_weight,
            "role":             role,
            "rationale":        rationale[:1000],
            "session_id":       session_id,
            "actor_id":         actor_id,
            "state":            "pending_human_approval",
            "deferred_gates":   deferred,
        }
        try:
            _write_promote_proposal(proposal)
        except Exception as e:
            emitter.stage_failed("gate9_human_approval", str(e)[:300])
            return self._failed(session, started_ts, "gate9_human_approval", str(e)[:300])

        emitter.stage_completed("gate9_human_approval", {
            "proposal_id":     proposal_id,
            "state":           "pending_human_approval",
            "approvals_link":  "/approvals",
            "note":            "Capital decision stays HUMAN. Go to /approvals and click APPROVE to actually deploy.",
        })

        # ── Emit + return ─────────────────────────────────────────
        try:
            opcon_emit.station_completed(
                session_id      = session_id,
                actor_id        = actor_id,
                job_id          = "",
                station_id      = self.STATION_SPEC.station_id,
                cost_actual_usd = 0.0,
                artifacts       = {
                    "proposal_id":      proposal_id,
                    "verdict_event_id": verdict_event_id,
                    "state":            "pending_human_approval",
                },
            )
        except Exception:
            logger.exception("operator_console: failed to emit station_completed")

        return StationResult(
            job_id          = "",
            station_id      = self.STATION_SPEC.station_id,
            session_id      = session_id,
            actor_id        = actor_id,
            started_ts      = started_ts,
            completed_ts    = _utc_iso(),
            success         = True,
            artifacts       = {
                "outcome":          "AWAITING_HUMAN_APPROVAL",
                "proposal_id":      proposal_id,
                "verdict_event_id": verdict_event_id,
                "subject_id":       ev.get("subject_id", ""),
                "target_weight":    str(target_weight),
                "role":             role,
                "next_action":      "Visit /approvals and click APPROVE to deploy",
            },
            events_emitted  = [],
            next_stations   = [],   # human action next, not another station
            cost_actual_usd = 0.0,
        )

    def result_lineage(self, result: StationResult) -> list[NextStationHint]:
        return []

    def _refused(self, session: Session, started_ts: str, stage: str, reason: str) -> StationResult:
        """Refusal = successful execution that hit a gate. Not a failure."""
        return StationResult(
            job_id          = "",
            station_id      = self.STATION_SPEC.station_id,
            session_id      = getattr(session, "session_id", ""),
            actor_id        = getattr(session, "actor_id", "principal"),
            started_ts      = started_ts,
            completed_ts    = _utc_iso(),
            success         = True,
            artifacts       = {
                "outcome":         "REFUSED_AT_GATE",
                "refused_at":      stage,
                "refusal_reason":  reason,
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


registry.register(Promote9Gate)
