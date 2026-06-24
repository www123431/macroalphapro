"""S3 — FactorSpec Extract.

Bridge between hypothesis-level claims and dispatch-ready FactorSpecs.
Takes a hypothesis_id, runs the LLM extractor over its prose claim,
persists the resulting FactorSpec to factor_specs.jsonl, and emits
its spec_hash for downstream S4 FORWARD Dispatch.

This is the missing link between paper ingestion (S1) and statistical
dispatch (S4). Without S3, the factor_specs.jsonl queue stays empty
and S4 has nothing to dispatch.

Design reference: docs/architecture/operator_console.md §5 (S3 spec).

MVP scope:
  - Single hypothesis_id input (no batch yet — batch is a Phase 4
    convenience, doesn't change architecture)
  - Reuses existing engine.agents.strengthener.factor_spec_store
    .extract_and_persist_pending() — the established production path
  - Idempotent: re-running on the same hypothesis returns the same
    spec_hash (the underlying function dedups by hash before append)
  - Emits station_completed with spec_hash in artifacts so S4 lineage
    can suggest itself
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


def _load_hypothesis_by_id(hypothesis_id: str):
    """Lookup a hypothesis by ID. Returns the Hypothesis dataclass or
    raises ValueError with a helpful 'did you mean' on miss."""
    from engine.research_store.hypothesis.store import load_hypotheses, find_by_id
    h = find_by_id(hypothesis_id)
    if h is not None:
        return h
    # Helpful hint: surface prefix matches
    all_hyps = load_hypotheses()
    prefix_hits = [hh.hypothesis_id for hh in all_hyps
                    if hh.hypothesis_id.startswith(hypothesis_id[:8])]
    hint = f" Did you mean one of: {prefix_hits[:3]}?" if prefix_hits else ""
    raise ValueError(f"hypothesis_id '{hypothesis_id}' not in hypotheses.jsonl.{hint}")


def _anthropic_key_present() -> bool:
    """Cheap check: do we have an Anthropic API key (for the Sonnet
    extractor)? Don't actually call the API."""
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    # Streamlit secrets file (project convention)
    from pathlib import Path
    secrets = Path(__file__).resolve().parents[3] / ".streamlit" / "secrets.toml"
    if secrets.is_file():
        try:
            txt = secrets.read_text(encoding='utf-8')
            return "ANTHROPIC_API_KEY" in txt or "anthropic_api_key" in txt.lower()
        except OSError:
            return False
    return False


# ── The station ──────────────────────────────────────────────────


class FactorSpecExtract(PipelineStation):
    """S3 — Extract a dispatch-ready FactorSpec from a hypothesis."""

    STATION_SPEC = StationSpec(
        station_id              = "S3_factorspec_extract",
        title                   = "FactorSpec Extract",
        description             = (
            "Run the LLM extractor over a hypothesis's prose claim and "
            "persist a hash-locked FactorSpec to factor_specs.jsonl. "
            "Idempotent: re-running on the same hypothesis returns the "
            "same spec_hash. Output feeds S4 FORWARD Dispatch."
        ),
        data_tier               = DataTier.USER_DATA,
        requires_session_types  = {SessionType.RESEARCH_NEW},
        estimated_minutes       = 2,
        estimated_cost_usd      = 0.05,
        icon                    = "Layers",
        title_key               = "console.station.s3.title",
        description_key         = "console.station.s3.description",
    )

    def preflight(self, session: Session, config: dict) -> PreflightResult:
        checks: list[PreflightCheck] = []

        if not session or not getattr(session, "session_id", ""):
            checks.append(PreflightCheck("session_active", PreflightStatus.RED,
                                         "No active session."))
        else:
            checks.append(PreflightCheck("session_active", PreflightStatus.GREEN,
                                         f"Session {session.session_id} ready."))

        hypothesis_id = str((config or {}).get("hypothesis_id", "")).strip()
        if not hypothesis_id:
            checks.append(PreflightCheck("hypothesis_id_provided", PreflightStatus.RED,
                                         "Provide hypothesis_id."))
        else:
            try:
                h = _load_hypothesis_by_id(hypothesis_id)
                fam = getattr(h, "mechanism_family", None)
                fam_val = getattr(fam, "value", str(fam)) if fam else "?"
                checks.append(PreflightCheck(
                    "hypothesis_resolvable", PreflightStatus.GREEN,
                    f"Loaded; mechanism_family={fam_val}",
                ))
            except ValueError as e:
                checks.append(PreflightCheck(
                    "hypothesis_resolvable", PreflightStatus.RED, str(e)[:150],
                ))
            except Exception as e:
                checks.append(PreflightCheck(
                    "hypothesis_resolvable", PreflightStatus.RED,
                    f"Load error: {type(e).__name__}: {e}"[:150],
                ))

        if _anthropic_key_present():
            checks.append(PreflightCheck(
                "llm_key_present", PreflightStatus.GREEN,
                "Anthropic key detected — extractor can run.",
            ))
        else:
            checks.append(PreflightCheck(
                "llm_key_present", PreflightStatus.RED,
                "No Anthropic key. Add ANTHROPIC_API_KEY to env or .streamlit/secrets.toml.",
            ))

        return PreflightResult.from_checks(checks)

    def estimate_cost(self, config: dict) -> CostEstimate:
        # Sonnet extractor call. Worst-case ~$0.05; usually less.
        return CostEstimate(llm_cost_usd_est=0.05, confidence="approximate")

    def render_config_form(self) -> dict:
        return {
            "type": "object",
            "title": "FactorSpec Extract input",
            "description": (
                "Provide the hypothesis_id of a hypothesis to extract a "
                "dispatch-ready FactorSpec from."
            ),
            "properties": {
                "hypothesis_id": {
                    "type": "string",
                    "title": "Hypothesis ID",
                    "description": "UUID from hypotheses.jsonl. Find candidates in /research/hypothesis or /research/forward.",
                    "x-ui-widget": "text",
                    "x-ui-placeholder": "e.g. 29d9cd0f-7643-4296-bdf7-b39da221f844",
                },
                "family_override": {
                    "type": "string",
                    "title": "Family hint override (optional)",
                    "description": "Defaults to hypothesis.mechanism_family. Override only if you have a stronger family classification.",
                    "x-ui-widget": "text",
                    "default": "",
                },
            },
            "required": ["hypothesis_id"],
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
        hypothesis_id = str((config or {}).get("hypothesis_id", "")).strip()
        family_override = str((config or {}).get("family_override", "")).strip()

        # ── Stage 1: load hypothesis ─────────────────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "load_hypothesis")
        emitter.stage_started("load_hypothesis", expected_seconds=1)
        try:
            h = _load_hypothesis_by_id(hypothesis_id)
        except Exception as e:
            emitter.stage_failed("load_hypothesis", str(e)[:300])
            return self._failed(session, started_ts, "load_hypothesis", str(e)[:300])
        fam = getattr(h, "mechanism_family", None)
        fam_default = getattr(fam, "value", str(fam)) if fam else "unknown"
        family_hint = family_override or fam_default
        emitter.stage_completed("load_hypothesis", {
            "mechanism_family": fam_default,
            "family_hint_used": family_hint,
            "claim":            (str(getattr(h, "claim", ""))[:120]),
        })

        # ── Stage 2: extract + persist (LLM call) ────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "extract_and_persist")
        emitter.stage_started("extract_and_persist", expected_seconds=60)
        emitter.log_line(
            "Calls Sonnet via factor_spec_extractor with structured tool-use schema. "
            "Idempotent — same hypothesis returns the same spec_hash."
        )
        try:
            from engine.agents.strengthener.factor_spec_store import extract_and_persist_pending
            spec_hash = await asyncio.to_thread(
                extract_and_persist_pending,
                h,
                family_hint,
            )
        except Exception as e:
            emitter.stage_failed("extract_and_persist", str(e)[:300])
            return self._failed(session, started_ts, "extract_and_persist", str(e)[:300])

        if spec_hash is None:
            # Extractor returned None — hypothesis didn't fit (procedural,
            # methodology, no provenance, etc.). Not a failure, but no
            # FactorSpec was created.
            emitter.stage_completed("extract_and_persist", {
                "outcome": "NOT_EXTRACTABLE",
                "reason": "Hypothesis didn't fit factor-spec extraction (procedural/methodology subtype OR extractor returned None).",
            })
            completed_ts = _utc_iso()
            return StationResult(
                job_id        = "",
                station_id    = self.STATION_SPEC.station_id,
                session_id    = session_id,
                actor_id      = actor_id,
                started_ts    = started_ts,
                completed_ts  = completed_ts,
                success       = True,   # not a failure; legitimate outcome
                artifacts     = {
                    "spec_hash":     "",
                    "hypothesis_id": hypothesis_id,
                    "outcome":       "NOT_EXTRACTABLE",
                    "family_hint":   family_hint,
                },
                events_emitted = [],
                next_stations  = [],
                cost_actual_usd = 0.0,
            )

        emitter.stage_completed("extract_and_persist", {
            "spec_hash":  spec_hash,
            "family_hint": family_hint,
        })

        # ── Stage 3: persist station_completed event ─────────────
        emitter.stage_started("persist_event", expected_seconds=1)
        try:
            opcon_emit.station_completed(
                session_id      = session_id,
                actor_id        = actor_id,
                job_id          = "",
                station_id      = self.STATION_SPEC.station_id,
                cost_actual_usd = 0.0,
                artifacts       = {
                    "spec_hash":     spec_hash,
                    "hypothesis_id": hypothesis_id,
                    "family_hint":   family_hint,
                    "registry_path": "data/strengthener/factor_specs.jsonl",
                },
            )
        except Exception:
            logger.exception("operator_console: failed to emit station_completed")
        emitter.stage_completed("persist_event", {"spec_hash": spec_hash})

        completed_ts = _utc_iso()
        return StationResult(
            job_id          = "",
            station_id      = self.STATION_SPEC.station_id,
            session_id      = session_id,
            actor_id        = actor_id,
            started_ts      = started_ts,
            completed_ts    = completed_ts,
            success         = True,
            artifacts       = {
                "spec_hash":     spec_hash,
                "hypothesis_id": hypothesis_id,
                "family_hint":   family_hint,
                "outcome":       "EXTRACTED",
            },
            events_emitted  = [],
            next_stations   = [NextStationHint(
                station_id        = "S4_forward_dispatch",
                label             = f"Dispatch this spec through FORWARD (~8 min, ~$0.10)",
                suggested_config  = {"spec_hash": spec_hash},
            )],
            cost_actual_usd = 0.05,
        )

    def result_lineage(self, result: StationResult) -> list[NextStationHint]:
        spec_hash = result.artifacts.get("spec_hash", "")
        if not spec_hash:
            return []
        return [NextStationHint(
            station_id        = "S4_forward_dispatch",
            label             = f"Dispatch this spec through FORWARD (~8 min, ~$0.10)",
            suggested_config  = {"spec_hash": spec_hash},
        )]

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


registry.register(FactorSpecExtract)
