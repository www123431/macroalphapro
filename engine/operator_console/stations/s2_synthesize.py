"""S2 — Hypothesis Synthesize.

Operator Console wrapper around `engine.agents.papers_curator.synthesis_runner.run_synthesis_pipeline`.

The synthesis pipeline already does the heavy lifting end-to-end:
  1. gather: build_synthesis_input() reads recent paper summaries,
     deployed sleeves, recent events, doctrine snippets, anchor
     library, belief layer summary
  2. synthesize: run_synthesis() — one Sonnet call with strict
     tool-use schema; returns 0-3 SynthesizedCandidate objects
     (Pattern 5 compliant: single call, no multi-agent debate)
  3. persist: write_synthesized_candidates() adapts candidates →
     Hypothesis rows → save_hypothesis(skip_cross_checks=True) →
     append to data/research_store/hypotheses.jsonl
  4. audit: emits a research_store event with provenance

S2's job is the session-bound, UI-triggerable shell around that:
  - preflight gates session type + papers_registry sanity
  - cost estimate flagged for the cost cap layer
  - 3-stage SSE progress wrapping the underlying call
  - lineage: each written hypothesis_id → NextStationHint(S3)

The synthesis call is LLM-bound (~$0.10 worst case). dry_run=True
preview mode is exposed so operators can sanity-check the input
snapshot before spending; it still calls the LLM (preview means
"don't persist"), so dry_run is for "did the synth produce something
plausible?" not "did the inputs look right?".

Design reference: docs/architecture/operator_console.md §5 (S2 spec).
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from pathlib import Path
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


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PAPERS_REGISTRY_PATH = _REPO_ROOT / "data" / "research_store" / "papers_registry.jsonl"

# Cost ceiling for the underlying papers_curator_synthesis workload.
# Sonnet 4.6 + ~6k max input + ~4k max output ≈ $0.08-0.10 worst case.
# We declare the upper bound so the cost-cap layer doesn't approve a
# trigger then halt mid-flight when actuals come in over.
_ESTIMATED_COST_USD = 0.10


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _papers_registry_size() -> int:
    """Cheap preflight check: papers_registry has at least N entries.
    The synthesis pipeline tolerates an empty registry (returns no
    candidates with a graceful errors entry) but operators rarely
    intend that — surface as a YELLOW warning, not a RED block."""
    if not _PAPERS_REGISTRY_PATH.is_file():
        return 0
    try:
        n = 0
        with _PAPERS_REGISTRY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    except OSError:
        return 0


class S2HypothesisSynthesize(PipelineStation):
    """Trigger a hypothesis-synthesis run from the operator console."""

    STATION_SPEC: Any = StationSpec(  # ClassVar typing matched in base class
        station_id              = "S2_synthesize",
        title                   = "Synthesize hypothesis",
        description             = (
            "Read the curator's recent paper summaries + deployed sleeves + "
            "recent events + doctrine snippets, send the snapshot to Sonnet "
            "with a strict synthesis schema, and persist 0-3 new hypothesis "
            "candidates to research_store/hypotheses.jsonl. Each written "
            "hypothesis can flow to S3 FactorSpec Extract → S4 FORWARD."
        ),
        data_tier               = DataTier.SNAPSHOT_DATA,
        requires_session_types  = {SessionType.RESEARCH_NEW},
        estimated_minutes       = 2,
        estimated_cost_usd      = _ESTIMATED_COST_USD,
        icon                    = "Sparkles",
        title_key               = "console.station.s2.title",
        description_key         = "console.station.s2.description",
        # mutates_capital=False — writes hypothesis rows, not deployed
        # config. Hypotheses must pass S3 → S4 → S7 → /approvals
        # before they can touch capital.
    )

    def preflight(self, session: Session, config: dict[str, Any]) -> PreflightResult:
        checks: list[PreflightCheck] = []

        # ── Session sanity
        if not session or not getattr(session, "session_id", ""):
            checks.append(PreflightCheck(
                "session_active", PreflightStatus.RED,
                "No active session. Open a research_new session first."))
            return PreflightResult.from_checks(checks)

        if getattr(session, "session_type", "") not in {"research_new"}:
            checks.append(PreflightCheck(
                "session_type", PreflightStatus.RED,
                f"S2 requires session_type=research_new; got "
                f"{getattr(session, 'session_type', '?')}."))
            return PreflightResult.from_checks(checks)
        checks.append(PreflightCheck(
            "session_type", PreflightStatus.GREEN,
            "research_new session active."))

        # ── Config sanity
        summaries_days = config.get("summaries_days", 14)
        events_days = config.get("events_days", 30)
        if not isinstance(summaries_days, int) or not 1 <= summaries_days <= 365:
            checks.append(PreflightCheck(
                "config", PreflightStatus.RED,
                f"summaries_days must be integer 1-365; got {summaries_days!r}."))
            return PreflightResult.from_checks(checks)
        if not isinstance(events_days, int) or not 1 <= events_days <= 365:
            checks.append(PreflightCheck(
                "config", PreflightStatus.RED,
                f"events_days must be integer 1-365; got {events_days!r}."))
            return PreflightResult.from_checks(checks)
        checks.append(PreflightCheck(
            "config", PreflightStatus.GREEN,
            f"window: summaries={summaries_days}d, events={events_days}d, "
            f"dry_run={bool(config.get('dry_run', False))}."))

        # ── Papers registry sanity (YELLOW, not RED — synthesis handles
        # empty input gracefully; surface as info)
        n_papers = _papers_registry_size()
        if n_papers == 0:
            checks.append(PreflightCheck(
                "papers_registry", PreflightStatus.YELLOW,
                "papers_registry.jsonl is empty or missing. "
                "Synthesis will run but likely produces no candidates."))
        else:
            checks.append(PreflightCheck(
                "papers_registry", PreflightStatus.GREEN,
                f"papers_registry has {n_papers} entries; "
                f"synthesis will pull from the recent window."))

        return PreflightResult.from_checks(checks)

    def estimate_cost(self, config: dict[str, Any]) -> CostEstimate:
        # Single Sonnet call — cost is dominated by the synthesis prompt.
        # Confidence "approximate" because actual depends on input size
        # (depends on recent paper count, doctrine snippet count, etc.).
        return CostEstimate(
            llm_cost_usd_est = _ESTIMATED_COST_USD,
            confidence       = "approximate",
        )

    def render_config_form(self) -> dict[str, Any]:
        return {
            "type":     "object",
            "title":    "Synthesize hypothesis configuration",
            "properties": {
                "summaries_days": {
                    "type":         "integer",
                    "title":        "Paper summary window (days)",
                    "description":  "Pull paper summaries from the last N days "
                                    "(default 14). Increase to widen the input "
                                    "if recent activity is sparse.",
                    "default":      14,
                    "minimum":      1,
                    "maximum":      365,
                    "x-ui-widget":  "number",
                },
                "events_days": {
                    "type":         "integer",
                    "title":        "Event window (days)",
                    "description":  "Pull research_store events from the last N "
                                    "days (default 30) for context.",
                    "default":      30,
                    "minimum":      1,
                    "maximum":      365,
                    "x-ui-widget":  "number",
                },
                "dry_run": {
                    "type":         "boolean",
                    "title":        "Dry-run (don't persist)",
                    "description":  "Still calls the LLM (preview mode), but "
                                    "does NOT write hypotheses to the store. "
                                    "Use to sanity-check the synthesizer's "
                                    "output before committing.",
                    "default":      False,
                    "x-ui-widget":  "checkbox",
                },
                "extra_tags": {
                    "type":         "string",
                    "title":        "Extra tags (comma-separated)",
                    "description":  "Optional tags appended to each persisted "
                                    "hypothesis (e.g. 'operator_triggered,"
                                    "bond_focus').",
                    "default":      "",
                    "x-ui-widget":  "text",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        session: Session,
        config: dict[str, Any],
        emitter: SSEEmitter,
        cancellation: CancellationToken,
    ) -> StationResult:
        started_ts = _utc_iso()
        session_id = getattr(session, "session_id", "")
        actor_id   = getattr(session, "actor_id", "principal")

        summaries_days = int(config.get("summaries_days", 14))
        events_days    = int(config.get("events_days", 30))
        dry_run        = bool(config.get("dry_run", False))
        extra_tags_raw = str(config.get("extra_tags", ""))
        extra_tags     = tuple(
            t.strip() for t in extra_tags_raw.split(",") if t.strip()
        )

        # ── Stage 1: prep + LLM call (single stage from S2's POV;
        # the underlying pipeline does gather + synthesize internally,
        # but they're not separable from outside — surface as one)
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "synthesize")
        emitter.stage_started("synthesize", expected_seconds=90)
        emitter.log_line(
            f"Calling run_synthesis_pipeline(summaries_days={summaries_days}, "
            f"events_days={events_days}, dry_run={dry_run}). Underlying "
            f"workload: papers_curator_synthesis (Sonnet, ~$0.10 worst case)."
        )

        try:
            from engine.agents.papers_curator.synthesis_runner import (
                run_synthesis_pipeline,
            )
            run_result = await asyncio.to_thread(
                run_synthesis_pipeline,
                dry_run        = dry_run,
                summaries_days = summaries_days,
                events_days    = events_days,
                created_by     = f"operator_console:{actor_id}",
                extra_tags     = extra_tags,
            )
        except Exception as e:
            emitter.stage_failed("synthesize", str(e)[:300])
            return self._failed(session, started_ts, "synthesize", str(e)[:300])

        # run_synthesis_pipeline is fail-safe (never raises); errors
        # show up in run_result["errors"]. Treat non-empty errors with
        # zero candidates as a soft failure.
        errors = run_result.get("errors", []) or []
        n_candidates = int(run_result.get("n_candidates", 0))
        n_written    = int(run_result.get("n_written", 0))
        hypothesis_ids = list(run_result.get("written_hypothesis_ids", []) or [])

        emitter.stage_completed("synthesize", {
            "n_candidates":    n_candidates,
            "n_written":       n_written,
            "dry_run":         dry_run,
            "errors":          errors[:3],   # cap for SSE payload
            "snapshot":        run_result.get("snapshot", {}),
            "event_id":        run_result.get("event_id"),
        })

        # ── Stage 2: persist station_completed event
        emitter.stage_started("persist_event", expected_seconds=1)
        try:
            opcon_emit.station_completed(
                session_id      = session_id,
                actor_id        = actor_id,
                job_id          = "",
                station_id      = self.STATION_SPEC.station_id,
                cost_actual_usd = _ESTIMATED_COST_USD,
                artifacts       = {
                    "n_candidates":     n_candidates,
                    "n_written":        n_written,
                    "dry_run":          dry_run,
                    "hypothesis_ids":   hypothesis_ids,
                    "synthesis_event":  run_result.get("event_id"),
                },
            )
        except Exception:
            logger.exception("operator_console: failed to emit station_completed")
        emitter.stage_completed("persist_event", {
            "n_written": n_written,
        })

        # Soft-fail criterion: pipeline raised errors AND produced
        # nothing usable. Both conditions required — synth that emitted
        # candidates but logged citation-check warnings is still success.
        soft_failed = bool(errors) and n_candidates == 0
        completed_ts = _utc_iso()

        return StationResult(
            job_id          = "",
            station_id      = self.STATION_SPEC.station_id,
            session_id      = session_id,
            actor_id        = actor_id,
            started_ts      = started_ts,
            completed_ts    = completed_ts,
            success         = not soft_failed,
            artifacts       = {
                "n_candidates":     n_candidates,
                "n_written":        n_written,
                "dry_run":          dry_run,
                "hypothesis_ids":   hypothesis_ids,
                "errors":           errors[:5],
                "snapshot":         run_result.get("snapshot", {}),
                "synthesis_event":  run_result.get("event_id", ""),
            },
            events_emitted  = (
                [run_result["event_id"]] if run_result.get("event_id") else []
            ),
            next_stations   = self._next_for(hypothesis_ids, dry_run),
            cost_actual_usd = _ESTIMATED_COST_USD,
            error_message   = "; ".join(errors[:2]) if soft_failed else "",
        )

    def result_lineage(self, result: StationResult) -> list[NextStationHint]:
        hypothesis_ids = list(result.artifacts.get("hypothesis_ids", []) or [])
        dry_run = bool(result.artifacts.get("dry_run", False))
        return self._next_for(hypothesis_ids, dry_run)

    def _next_for(self, hypothesis_ids: list[str], dry_run: bool) -> list[NextStationHint]:
        if dry_run or not hypothesis_ids:
            return []
        # One lineage hint per written hypothesis. UI may show first N.
        return [
            NextStationHint(
                station_id        = "S3_factorspec_extract",
                label             = f"Extract FactorSpec from hypothesis {hid[:8]}…",
                suggested_config  = {"hypothesis_id": hid},
            )
            for hid in hypothesis_ids
        ]

    def _cancelled(self, session: Session, started_ts: str, stage: str) -> StationResult:
        return StationResult(
            job_id        = "",
            station_id    = self.STATION_SPEC.station_id,
            session_id    = getattr(session, "session_id", ""),
            actor_id      = getattr(session, "actor_id", "principal"),
            started_ts    = started_ts,
            completed_ts  = _utc_iso(),
            success       = False,
            error_message = f"cancelled at stage={stage}",
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
            error_message = f"{stage}: {err}",
        )


registry.register(S2HypothesisSynthesize)
