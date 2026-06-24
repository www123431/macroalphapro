"""
engine/agents/sector_pipeline/agent.py — SectorPipelineAgent

Migrated from engine/sector_pipeline.run_sector_pipeline on 2026-05-03 as P0
step 2 of the agent-infra adoption sweep. The original `run_sector_pipeline`
function is now a thin back-compat wrapper that delegates here.

Persistence
-----------
* AgentRun rows track each run() invocation. `input_params` carries
  sector / vix / decision_source / parent_decision_id;
  `output_summary` carries saved_id / qc_flags_count / scaled_adj /
  pending_approval_id (NOT the full debate transcript, to keep agent_runs
  light — transcript still goes to DecisionLog.debate_transcript via
  save_decision()).
* status: "succeeded" when DecisionLog row written;
          "interrupted" when debate output was a quota/error marker
          (legitimate skip, not a failure);
          "failed" on uncaught exception.

Events emitted
--------------
* sector.decision_saved — when saved_id is non-None. Payload includes
  sector / saved_id / scaled_adj / qc_flags_count / pending_approval_id /
  decision_source. Downstream listeners: paper trading three-arm runner,
  supervisor approval queue, future reflection loop (decision Brier feedback).

Trigger payload contract
------------------------
Caller must put per-call params on trigger.payload:
    sector              : str  (required)
    vix                 : float (required)
    parent_decision_id  : int | None
    revision_reason     : str
    overwrite           : bool
    history_prefix      : str
trigger.source carries the decision_source label (preserved verbatim for
memory_curator / failure_attribution sample stratification).
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

from engine.agents.base import Agent, AgentEvent, AgentResult, Trigger

logger = logging.getLogger(__name__)


class SectorPipelineAgent(Agent):
    """Sector debate + decision-write agent (single source of truth)."""

    AGENT_ID = "sector_pipeline"

    def __init__(self, model: Any | None = None) -> None:
        self.model = model
        self._last_result: dict[str, Any] | None = None

    def run(self, trigger: Trigger, as_of: datetime.date) -> AgentResult:
        sector = trigger.payload.get("sector") or ""
        vix = float(trigger.payload.get("vix") or 20.0)

        run = self._new_run(
            trigger,
            input_params={
                "as_of": str(as_of),
                "sector": sector,
                "vix": vix,
                "decision_source": trigger.source,
                "parent_decision_id": trigger.payload.get("parent_decision_id"),
                "model_present": self.model is not None,
            },
        )
        run.state = "starting"
        self._persist_run(run)

        full_result: dict[str, Any] = {
            "saved_id":            None,
            "debate":              {},
            "qc_flags":            [],
            "scaled_adj":          0.0,
            "pending_approval_id": None,
            "inputs":              {},
        }

        try:
            run.state = "running_pipeline"
            self._persist_run(run)
            full_result = self._run_internal(
                sector_name=sector,
                t_day=as_of,
                vix=vix,
                decision_source=trigger.source,
                parent_decision_id=trigger.payload.get("parent_decision_id"),
                revision_reason=trigger.payload.get("revision_reason", ""),
                overwrite=bool(trigger.payload.get("overwrite", False)),
                history_prefix=trigger.payload.get("history_prefix", ""),
            )

            run.summary = {
                "saved_id":            full_result.get("saved_id"),
                "qc_flags_count":      len(full_result.get("qc_flags") or []),
                "scaled_adj":          full_result.get("scaled_adj"),
                "pending_approval_id": full_result.get("pending_approval_id"),
                "decision_source":     trigger.source,
                "sector":              sector,
            }

            if full_result.get("saved_id"):
                run.status = "succeeded"
            else:
                run.status = "interrupted"
                run.error = "decision skipped (quota/error output or empty result)"
        except Exception as exc:
            logger.exception("SectorPipelineAgent.run failed for sector=%s", sector)
            run.status = "failed"
            run.error = str(exc)
        finally:
            run.finished_at = datetime.datetime.utcnow()
            run.state = "done"
            self._persist_run(run)

        if run.status == "succeeded":
            try:
                ev_id = self._emit_event(AgentEvent(
                    event_type="sector.decision_saved",
                    payload={
                        "run_id":              run.run_id,
                        "as_of":               str(as_of),
                        "sector":              sector,
                        "saved_id":            full_result.get("saved_id"),
                        "scaled_adj":          full_result.get("scaled_adj"),
                        "qc_flags_count":      len(full_result.get("qc_flags") or []),
                        "pending_approval_id": full_result.get("pending_approval_id"),
                        "decision_source":     trigger.source,
                    },
                ))
                run.events_emitted.append(ev_id)
            except Exception as exc:
                logger.warning("sector.decision_saved publish failed: %s", exc)

        self._last_result = full_result
        return run

    # ── Internal: original run_sector_pipeline body (verbatim, refactored) ──

    def _run_internal(
        self,
        sector_name: str,
        t_day: datetime.date,
        vix: float,
        decision_source: str,
        parent_decision_id: int | None,
        revision_reason: str,
        overwrite: bool,
        history_prefix: str,
    ) -> dict[str, Any]:
        from engine.debate import run_sector_debate, run_quant_coherence_check
        from engine.memory import (
            SessionFactory, PendingApproval,
            save_decision, supersede_decision, extract_direction,
            update_reflections_injected,
        )
        from engine.sector_pipeline import (
            prepare_sector_inputs,
            _debate_output_is_error,
        )

        inputs = prepare_sector_inputs(sector_name, t_day, vix)

        # ── S2 Reflection retrieval (spec §5.1) ──────────────────────────────
        # Pull top-K relevant past reflections for THIS agent + prepend to
        # historical_context. Capability layer; failure = silent skip
        # (reflections must never block the decision path).
        reflection_ids: list[int] = []
        reflection_block: str = ""
        try:
            from engine.agents.reflection import (
                build_reflection_query,
                retrieve_relevant_reflections,
                format_reflections_for_prompt,
            )
            query_text = build_reflection_query(
                decision_summary={
                    "sector": sector_name,
                    "direction": "",
                    "rationale_excerpt": (
                        f"regime={inputs.get('regime_label','')}, vix={vix}"
                    ),
                },
                factor_context=None,
                extra_text=(inputs.get("news_context") or "")[:300],
            )
            reflections = retrieve_relevant_reflections(
                agent_id="sector_pipeline",
                query_text=query_text,
                k=5,
                as_of=t_day,
            )
            reflection_ids = [r.id for r in reflections]
            reflection_block = format_reflections_for_prompt(
                reflections, agent_id="sector_pipeline"
            )
            if reflection_ids:
                logger.info(
                    "SectorPipelineAgent: injected %d past reflections for %s on %s",
                    len(reflection_ids), sector_name, t_day,
                )
        except Exception as exc:
            logger.warning(
                "SectorPipelineAgent: reflection retrieval failed (%s); continuing without",
                exc,
            )

        historical_context = inputs["historical_context"]
        if reflection_block:
            historical_context = reflection_block + "\n\n" + historical_context
        if history_prefix:
            historical_context = history_prefix + historical_context

        # ── S3 spec_hash auto-inject (spec §三 Sprint 2) ─────────────────────
        # The sector pipeline is governed by docs/spec_sector_pipeline_unification.md
        # — pull its current git-blob hash so save_decision can record it on
        # the DecisionLog row. Failure is silent (None hash → R3 may flag
        # later, which is correct behaviour for un-registered specs).
        sector_spec_hash: str | None = None
        try:
            from engine.preregistration import (
                _compute_git_blob_hash, _resolve_to_abs,
            )
            sector_spec_hash = _compute_git_blob_hash(
                _resolve_to_abs("docs/spec_sector_pipeline_unification.md")
            )
        except Exception as exc:
            logger.warning(
                "SectorPipelineAgent: spec_hash injection failed (%s); proceeding "
                "with NULL spec_hash", exc,
            )

        debate = run_sector_debate(
            model              = self.model,
            sector_name        = sector_name,
            vix                = vix,
            macro_context      = inputs["macro_context"],
            news_context       = inputs["news_context"],
            historical_context = historical_context,
            valuation_context  = inputs["valuation_context"],
            quant_context      = inputs["quant_context"] or None,
            quant_gate         = inputs["quant_gate"] or None,
        )

        final_output = debate.get("final_output", "") or ""
        final_xai    = debate.get("final_xai", {}) or {}

        if _debate_output_is_error(final_output):
            logger.warning(
                "SectorPipelineAgent: skipping save for %s (source=%s) — quota/error output",
                sector_name, decision_source,
            )
            return {
                "saved_id":            None,
                "debate":              debate,
                "qc_flags":            [],
                "scaled_adj":          0.0,
                "pending_approval_id": None,
                "inputs":              inputs,
            }

        qc_direction = extract_direction(final_output)
        try:
            qc_flags = run_quant_coherence_check(
                final_xai, inputs["quant_context"], direction=qc_direction,
            )
        except Exception as e:
            logger.warning("quant_coherence_check failed for %s: %s", sector_name, e)
            qc_flags = []

        raw_adj  = debate.get("weight_adjustment_pct") or 0.0
        conf_val = final_xai.get("overall_confidence") or 50
        if conf_val < 40:
            conf_mult = 0.5
        elif conf_val > 70:
            conf_mult = 1.3
        else:
            conf_mult = 1.0
        scaled_adj = raw_adj * conf_mult

        signal_attribution = {
            "macro":     final_xai.get("macro_confidence"),
            "news":      final_xai.get("news_confidence"),
            "technical": final_xai.get("technical_confidence"),
            "drivers":   final_xai.get("signal_drivers", ""),
        }

        quant_ctx = inputs["quant_context"] or {}

        # H2 fix (Wave 2 2026-05-07 applied-focus reframe): populate DL-P0
        # attribution fields weight_before / weight_after.  Convention mirrors
        # canonical _BASE_WEIGHT map in engine/memory.py:1308 + clipping at
        # engine/memory.py:1317 (±0.20).  None for unrecognised directions so
        # downstream filters cleanly skip them.
        #
        # Note on the ±0.20 clip vs engine/config.py MAX_WEIGHT=0.25:  these
        # are deliberately different layers.  ±0.20 is the per-decision LLM-
        # adjusted attribution cap (single sector decision).  MAX_WEIGHT=0.25
        # is the portfolio-level hard policy cap applied later by
        # construct_portfolio.  If MAX_WEIGHT changes, this clip can stay
        # (or move in lockstep — supervisor decision).
        _DIRECTION_BASE_W = {
            "超配": 0.08, "标配": 0.0, "低配": -0.04,
            "中性": 0.0,
        }
        _wb = _DIRECTION_BASE_W.get(qc_direction)
        if _wb is not None:
            _wa = max(-0.20, min(0.20, _wb + (scaled_adj or 0.0) / 100.0))
        else:
            _wa = None

        saved_id = save_decision(
            tab_type="sector",
            ai_conclusion=final_output,
            vix_level=vix,
            sector_name=sector_name,
            ticker=inputs["ticker_for_news"],
            news_summary=inputs["news_context"][:500],
            overwrite=overwrite,
            macro_regime=inputs["regime_label"],
            horizon=final_xai.get("horizon", "季度(3个月)"),
            confidence_score=final_xai.get("overall_confidence"),
            macro_confidence=final_xai.get("macro_confidence"),
            news_confidence=final_xai.get("news_confidence"),
            technical_confidence=final_xai.get("technical_confidence"),
            signal_attribution=signal_attribution,
            invalidation_conditions=final_xai.get("invalidation_conditions", ""),
            decision_date=t_day,
            debate_transcript={
                "history":      debate.get("debate_history"),
                "arbitration":  debate.get("arbitration_notes"),
                "blue_output":  debate.get("blue_output"),
                "state_vector": inputs["state_vector"],
            },
            parent_decision_id=parent_decision_id,
            revision_reason=revision_reason,
            decision_source=decision_source,
            quant_p_noise=quant_ctx.get("p_noise"),
            quant_val_r2=quant_ctx.get("val_r2"),
            quant_test_r2=quant_ctx.get("test_r2"),
            quant_active=quant_ctx.get("active"),
            weight_adjustment_pct=scaled_adj,
            adjustment_reason=debate.get("final_data", {}).get("adjustment_reason"),
            signal_invalidation_risk=final_xai.get("signal_invalidation_risk"),
            spec_hash=sector_spec_hash,
            weight_before=_wb,
            weight_after=_wa,
        )

        if parent_decision_id and saved_id:
            try:
                supersede_decision(parent_decision_id, revision_reason)
            except Exception as e:
                logger.warning("supersede_decision failed parent=%s: %s",
                               parent_decision_id, e)

        # S2 audit: record retrieved-reflection ids on this DecisionLog row.
        if saved_id:
            update_reflections_injected(saved_id, reflection_ids)

        pending_approval_id: int | None = None
        if qc_flags and saved_id and not parent_decision_id:
            try:
                # Sector overlay retired (engine.approval_charter): the LLM QC
                # disagreement is recorded as a record-only routine_review trace,
                # not gated to the human inbox (discretionary organ decommissioned).
                from engine.approval_charter import retired_trace_fields
                with SessionFactory() as s:
                    existing = (
                        s.query(PendingApproval)
                        .filter(
                            PendingApproval.approval_type == "risk_control",
                            PendingApproval.sector == sector_name,
                            PendingApproval.triggered_date == t_day,
                        )
                        .first()
                    )
                    if not existing:
                        pa = PendingApproval(
                            approval_type="risk_control",
                            priority="high",
                            sector=sector_name,
                            ticker=inputs["ticker_for_news"],
                            triggered_condition="; ".join(qc_flags),
                            triggered_date=t_day,
                            contradicts_quant=True,
                            llm_confidence=final_xai.get("overall_confidence"),
                            **retired_trace_fields(),
                        )
                        s.add(pa)
                        s.commit()
                        pending_approval_id = pa.id
            except Exception as e:
                logger.warning("PendingApproval write failed for %s: %s", sector_name, e)

        return {
            "saved_id":            saved_id,
            "debate":              debate,
            "qc_flags":            qc_flags,
            "scaled_adj":          scaled_adj,
            "pending_approval_id": pending_approval_id,
            "inputs":              inputs,
        }
