"""S4 — FORWARD Dispatch.

Wraps the existing dispatch_factor_spec() statistical-rigor pipeline
in a Pipeline Station shell so users can trigger it from the
Operator Console UI.

The underlying dispatcher (engine.agents.strengthener.factor_dispatcher
.dispatch_factor_spec) is synchronous and runs the full FORWARD chain
in-process:
  - pre_dispatch_check (gates: PIT whitelist, n_trials, family quota,
    template availability)
  - Belief Layer Phase 1: predict_and_log (AIR-GAPPED prediction
    written to data/research/predictions.jsonl)
  - Template registry dispatch (statistical rigor pipeline — Bootstrap
    CI / NW-t HAC / FF5+MOM spanning / Bailey-Lopez de Prado DSR /
    Hosmer-Lemeshow / ...; per-template; reads WRDS data)
  - OOS triple analysis (bt-flex-1)
  - Verdict synthesis + dispatch log + emit factor_verdict_filed

Because the existing dispatcher runs synchronously inside one Python
call, the station runs it in asyncio.to_thread() so the worker event
loop stays responsive. SSE stage emissions happen at the COARSE
boundaries we can observe from outside (preflight / belief_predict /
template_dispatch / verdict_persist). Per-statistical-test progress
(e.g. Bootstrap CI percentage) is NOT emitted in MVP — that would
require invasive callbacks into each template.

Design reference: docs/architecture/operator_console.md §5 (S4 spec).

WRDS_REQUIRED data tier: dispatch fails without WRDS for templates
that read CRSP / Compustat / OptionMetrics. Demo path (cached fixture
inputs) deferred to Phase 4 polish — for now S4 is the institutional-
data station and preflight surfaces that honestly.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from typing import Any

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


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ── Helpers ──────────────────────────────────────────────────────


def _wrds_env_present() -> bool:
    """Cheap check: do we have WRDS credentials in env / pgpass?
    Doesn't actually probe the connection (too slow for preflight);
    just confirms config exists so we fail fast on totally-missing
    setup."""
    import os
    if any(os.environ.get(k) for k in ("PGPASSWORD", "WRDS_USER", "WRDS_PASS")):
        return True
    # Windows .pgpass is in %APPDATA%/postgresql/pgpass.conf;
    # Linux/Mac is ~/.pgpass
    from pathlib import Path
    candidates = [
        Path.home() / ".pgpass",
        Path(os.environ.get("APPDATA", "")) / "postgresql" / "pgpass.conf",
    ]
    return any(p.is_file() for p in candidates)


def _load_spec_by_hash(spec_hash: str):
    """Load a FactorSpec by spec_hash from factor_specs.jsonl.
    Returns (FactorSpec, family_hint) tuple, or raises ValueError."""
    from engine.agents.strengthener.factor_spec_store import (
        _load_specs, _payload_to_factor_spec, _DEFAULT_SPECS_PATH,
    )
    specs = _load_specs(_DEFAULT_SPECS_PATH)
    if spec_hash not in specs:
        # Helpful "did you mean" hints: look for prefix matches
        prefix_hits = [h for h in specs.keys() if h.startswith(spec_hash[:8])]
        hint = f" Did you mean one of: {prefix_hits[:3]}?" if prefix_hits else ""
        raise ValueError(f"spec_hash '{spec_hash}' not in factor_specs.jsonl.{hint}")
    payload = specs[spec_hash]
    spec = _payload_to_factor_spec(payload)
    family_hint = (
        payload.get("family")
        or payload.get("family_hint")
        or getattr(spec, "family", "")
        or "unknown"
    )
    return spec, family_hint


# ── The station ──────────────────────────────────────────────────


class ForwardDispatch(PipelineStation):
    """S4 — Dispatch a FactorSpec through the FORWARD statistical
    rigor pipeline. Runs the same code path as
    `python scripts/burndown_run.py` for the single-spec case,
    just triggered from the UI within a session."""

    STATION_SPEC = StationSpec(
        station_id              = "S4_forward_dispatch",
        title                   = "FORWARD Dispatch",
        description             = (
            "Run a FactorSpec through the FORWARD statistical-rigor pipeline "
            "(FF5+MOM spanning · Newey-West HAC · Bootstrap CI · Bailey-Lopez "
            "de Prado DSR · Hosmer-Lemeshow). Belief Layer Phase 1 commits an "
            "air-gapped prediction first. Verdict: GREEN / MARGINAL / RED."
        ),
        data_tier               = DataTier.WRDS_REQUIRED,
        requires_session_types  = {SessionType.RESEARCH_NEW, SessionType.AUDIT},
        estimated_minutes       = 8,
        estimated_cost_usd      = 0.10,
        icon                    = "FlaskConical",
        title_key               = "console.station.s4.title",
        description_key         = "console.station.s4.description",
    )

    def preflight(self, session: Session, config: dict) -> PreflightResult:
        checks: list[PreflightCheck] = []

        if not session or not getattr(session, "session_id", ""):
            checks.append(PreflightCheck("session_active", PreflightStatus.RED,
                                         "No active session."))
        else:
            checks.append(PreflightCheck("session_active", PreflightStatus.GREEN,
                                         f"Session {session.session_id} ready."))

        spec_hash = str((config or {}).get("spec_hash", "")).strip()
        if not spec_hash:
            checks.append(PreflightCheck("spec_hash_provided", PreflightStatus.RED,
                                         "Provide spec_hash."))
        else:
            # Verify it loads (cheap disk read)
            try:
                _spec, family = _load_spec_by_hash(spec_hash)
                checks.append(PreflightCheck(
                    "spec_resolvable", PreflightStatus.GREEN,
                    f"Loaded spec; family_hint='{family}'.",
                ))
            except ValueError as e:
                checks.append(PreflightCheck(
                    "spec_resolvable", PreflightStatus.RED, str(e)[:150],
                ))
            except Exception as e:
                checks.append(PreflightCheck(
                    "spec_resolvable", PreflightStatus.RED,
                    f"Load error: {type(e).__name__}: {e}"[:150],
                ))

        # WRDS env presence (D3 — wrds_required tier check)
        if _wrds_env_present():
            checks.append(PreflightCheck(
                "wrds_env_present", PreflightStatus.GREEN,
                "WRDS credentials detected; templates can read CRSP/Compustat/IBES.",
            ))
        else:
            checks.append(PreflightCheck(
                "wrds_env_present", PreflightStatus.YELLOW,
                "No WRDS env detected. Most templates will fail at data_load. "
                "(Phase 4 demo-fixture path is not yet wired.)",
            ))

        return PreflightResult.from_checks(checks)

    def estimate_cost(self, config: dict) -> CostEstimate:
        # Most templates have small LLM use (extractor + light
        # narration). Per design doc S4 spec: $0.10 budget.
        return CostEstimate(llm_cost_usd_est=0.10, confidence="approximate")

    def render_config_form(self) -> dict:
        return {
            "type": "object",
            "title": "FORWARD Dispatch input",
            "description": (
                "Provide the spec_hash of a FactorSpec to dispatch. "
                "Find candidates in /research/forward (Forward vectors)."
            ),
            "properties": {
                "spec_hash": {
                    "type": "string",
                    "title": "FactorSpec hash",
                    "description": "From factor_specs.jsonl. Either full hash or unique prefix.",
                    "x-ui-widget": "text",
                    "x-ui-placeholder": "e.g. 24eedac6...",
                },
                "operator_note": {
                    "type": "string",
                    "title": "Operator note (optional)",
                    "description": "Why are you dispatching this spec now? Lands in the dispatch log.",
                    "x-ui-widget": "text-area",
                    "x-ui-rows": 2,
                    "default": "",
                },
            },
            "required": ["spec_hash"],
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
        spec_hash = str((config or {}).get("spec_hash", "")).strip()
        operator_note = str((config or {}).get("operator_note", "")).strip()

        # ── Stage 1: load + family resolution ─────────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "load_spec")
        emitter.stage_started("load_spec", expected_seconds=1)
        try:
            spec, family_hint = _load_spec_by_hash(spec_hash)
        except Exception as e:
            emitter.stage_failed("load_spec", str(e)[:300])
            return self._failed(session, started_ts, "load_spec", str(e)[:300])
        emitter.stage_completed("load_spec", {
            "family_hint":  family_hint,
            "signal_kind":  getattr(spec, "signal_kind", "?"),
            "hypothesis_id": getattr(spec, "hypothesis_id", "?"),
        })

        # ── Stage 2: run dispatcher (the big one — to_thread'd) ──
        # The underlying dispatch_factor_spec is sync + multi-minute.
        # Wrap in to_thread so worker event loop stays free.
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "dispatch")
        emitter.stage_started("dispatch", expected_seconds=self.STATION_SPEC.estimated_minutes * 60)
        emitter.log_line(
            "Dispatcher runs synchronously: preflight gates → Belief Phase-1 "
            "predict-commit → template registry → statistical rigor → "
            "OOS triple analysis → dispatch log + factor_verdict_filed event."
        )
        try:
            from engine.agents.strengthener.factor_dispatcher import dispatch_factor_spec
            result = await asyncio.to_thread(
                dispatch_factor_spec,
                spec,
                family_hint    = family_hint,
                spec_approved  = True,   # operator triggered through UI = approved
                dry_run        = False,
                human_override = (
                    f"operator_console S4 (session_id={session_id}, "
                    f"note={operator_note[:60]})" if operator_note else
                    f"operator_console S4 (session_id={session_id})"
                ),
                cron_run_id    = None,
                cron_source    = "operator_console_s4",
            )
        except Exception as e:
            emitter.stage_failed("dispatch", str(e)[:300])
            return self._failed(session, started_ts, "dispatch", str(e)[:300])

        # Surface refusal vs verdict clearly
        refusal = result.get("refusal")
        tpl_result = result.get("template_result") or {}
        dispatch_event_id = result.get("dispatch_event_id", "")

        if refusal:
            emitter.stage_completed("dispatch", {
                "outcome":      "REFUSED",
                "reason_code":  refusal.get("reason_code", ""),
                "detail":       str(refusal.get("detail", ""))[:200],
            })
            # Refusals are NOT failures — they're successful gate fires.
            # Return success=True so the UI doesn't show a red error box;
            # the verdict surface will explain.
            verdict = "REFUSED"
        else:
            verdict = tpl_result.get("verdict", "UNKNOWN")
            emitter.stage_completed("dispatch", {
                "outcome":          "VERDICT",
                "verdict":          verdict,
                "summary":          str(tpl_result.get("summary", ""))[:200],
                "dispatch_event_id": dispatch_event_id,
            })

        # ── Stage 3: persist + emit station_completed ─────────────
        emitter.stage_started("persist_event", expected_seconds=1)
        try:
            opcon_emit.station_completed(
                session_id      = session_id,
                actor_id        = actor_id,
                job_id          = "",
                station_id      = self.STATION_SPEC.station_id,
                cost_actual_usd = 0.0,   # actual cost tracked separately in llm_cost_ledger
                artifacts       = {
                    "dispatch_event_id":   dispatch_event_id or "",
                    "spec_hash":           result.get("spec_hash", spec_hash),
                    "verdict":             verdict,
                    "prediction_id":       result.get("prediction_id", ""),
                },
            )
        except Exception:
            logger.exception("operator_console: failed to emit station_completed")
        emitter.stage_completed("persist_event", {"verdict": verdict})

        # ── Result ────────────────────────────────────────────────
        completed_ts = _utc_iso()
        return StationResult(
            job_id           = "",
            station_id       = self.STATION_SPEC.station_id,
            session_id       = session_id,
            actor_id         = actor_id,
            started_ts       = started_ts,
            completed_ts     = completed_ts,
            success          = True,   # refusal still = successful execution
            artifacts        = {
                "verdict":             verdict,
                "spec_hash":           result.get("spec_hash", spec_hash),
                "dispatch_event_id":   dispatch_event_id or "",
                "prediction_id":       result.get("prediction_id", ""),
                "summary":             str(tpl_result.get("summary", ""))[:400],
                "refusal_reason_code": (refusal or {}).get("reason_code", ""),
            },
            events_emitted   = [],
            next_stations    = self._lineage(verdict, dispatch_event_id),
            cost_actual_usd  = 0.0,
        )

    def result_lineage(self, result: StationResult) -> list[NextStationHint]:
        verdict = result.artifacts.get("verdict", "")
        dispatch_event_id = result.artifacts.get("dispatch_event_id", "")
        return self._lineage(verdict, dispatch_event_id)

    @staticmethod
    def _lineage(verdict: str, dispatch_event_id: str) -> list[NextStationHint]:
        hints: list[NextStationHint] = []
        if verdict == "GREEN":
            hints.append(NextStationHint(
                station_id        = "S7_promote",
                label             = "PROMOTE GREEN verdict to deployment (9-gate)",
                suggested_config  = {"verdict_event_id": dispatch_event_id},
            ))
        if dispatch_event_id:
            hints.append(NextStationHint(
                station_id        = "S6_verdict_view",
                label             = "View verdict + autopsy + belief update",
                suggested_config  = {"verdict_event_id": dispatch_event_id},
            ))
        return hints

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


# Register at import time
registry.register(ForwardDispatch)
