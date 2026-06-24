"""engine.research.burndown_executor — burn-1b execution path.

Given a burndown plan (or just a list of ranked candidates), execute
each through the existing dispatch path:

  hypothesis_row → factor_spec_extractor.extract_factor_spec → FactorSpec
                  → dispatch_factor_spec(..., cron_run_id=<id>)
                  → ExecutionOutcome

Safety
======
- **Kill switch**: caller (the cron script) checks data/cron_burndown/_disabled
  BEFORE constructing the executor. Once execute_plan is running, the
  switch is not re-checked mid-loop (would create surprising
  half-completed runs); to halt, kill the process.
- **Cap re-check**: each candidate, we re-read burndown_caps.usage_last_7d()
  BEFORE dispatching. If the family or global cap has now bound (e.g. a
  previous candidate succeeded and bumped the count to cap), skip the
  candidate with reason CAP_HIT_MID_RUN.
- **Extraction failure → skip**: factor_spec_extractor calls Sonnet
  (~$0.03 + ~30s). When it returns None we DON'T burn a slot; the
  hypothesis stays in the queue for tomorrow.
- **Dispatch refusal → don't burn slot**: dispatcher's TIER_3/TIER_4 dead
  walls (TEMPLATE_NOT_CERTIFIED etc) get logged to capability_gaps via
  flex-3 and DON'T count against burndown_caps quota.
- **All outcomes recorded** to data/cron_burndown/outcomes/<plan_id>.json
  for burn-2 digest consumption.

NOT in this module (deferred)
=============================
- LLM call budget tracking — relies on existing llm_cost_ledger.
- Auto-retry of TIER_3 dead-walls (capability_gaps demand ledger has
  these; principal decides when to build the template).
- Auto-PROMOTE_TO_PAPER_TRADE on GREEN — capital decisions stay HUMAN
  per standing doctrine.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTCOME_DIR = _REPO_ROOT / "data" / "cron_burndown" / "outcomes"

# External adversarial audit (Phase 1.2, 2026-06-13) — every GREEN /
# MARGINAL / RED verdict emitted from the cron path gets reviewed by an
# independent LLM (default DeepSeek v4-flash). Mitigation #1 of the
# self-audit blind-spots doctrine.
EXTERNAL_AUDIT_BUDGET_PATH = (
    _REPO_ROOT / "data" / "cron_burndown" / "external_audit_budget.jsonl"
)
EXTERNAL_AUDIT_WEEKLY_BUDGET_USD = 1.50  # ~30-50 audits/wk at deepseek v4-flash


@_dc.dataclass(frozen=True)
class ExecutionOutcome:
    """One candidate's end-to-end run record."""
    hypothesis_id:     str
    family:            str
    cron_run_id:       str
    extraction_ok:     bool
    extraction_error:  Optional[str]
    spec_hash:         Optional[str]
    refusal_reason:    Optional[str]
    verdict:           Optional[str]    # GREEN / MARGINAL / RED / EXECUTION_ERROR
    decay_severity:    Optional[str]    # from bt-flex-1, if present
    dispatch_event_id: Optional[str]
    prediction_id:     Optional[str]
    ran_at:            str

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _week_iso_now() -> str:
    now = _dt.datetime.utcnow()
    iso_year, iso_week, _ = now.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _current_week_audit_spend(budget_path: Optional[Path] = None) -> float:
    """Sum cost_usd from all rows for the current ISO week. Returns 0.0
    when the budget file does not exist (= no prior audits this week)."""
    p = budget_path or EXTERNAL_AUDIT_BUDGET_PATH
    if not p.is_file():
        return 0.0
    wk = _week_iso_now()
    total = 0.0
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("week") == wk:
                total += float(row.get("cost_usd", 0.0))
    return total


def _record_audit_spend(
    cost: float, audit_id: str, severity: str,
    *, budget_path: Optional[Path] = None,
) -> None:
    """Append one row to the budget ledger. Non-fatal if write fails."""
    p = budget_path or EXTERNAL_AUDIT_BUDGET_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "week":      _week_iso_now(),
            "ts":        _utc_iso(),
            "audit_id":  audit_id,
            "cost_usd":  float(cost),
            "severity":  severity,
        }
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("burndown_executor: audit budget write failed: %s", exc)


def _maybe_run_post_green_rigor(
    outcome: "ExecutionOutcome",
    template_result: dict,
    spec,
    *,
    ledger_path: Optional[Path] = None,
    rigor_fn = None,
) -> None:
    """Phase 4.1 (2026-06-13): post-GREEN rigor pipeline.

    For every GREEN/MARGINAL verdict, fire two mechanical checks:
      1. post-publication out-of-sample re-run
      2. FF5+MOM spanning regression

    NEVER raises. Disabled via BURNDOWN_POST_GREEN_RIGOR_DISABLED=1.
    Critical findings (DEAD post-pub, SUBSUMED spanning) bubble up
    via WARNING logs.
    """
    import os
    if os.environ.get("BURNDOWN_POST_GREEN_RIGOR_DISABLED") == "1":
        return
    if outcome.verdict not in {"GREEN", "MARGINAL"}:
        return

    try:
        if rigor_fn is None:
            from engine.research.post_green_rigor import run_post_green_rigor
            rigor_fn = run_post_green_rigor
        # Need the actual template_result object (not dict) for artifact
        # extraction. Reconstruct a thin shim:
        class _ResultShim:
            pass
        shim = _ResultShim()
        shim.verdict          = outcome.verdict
        shim.metrics          = template_result.get("metrics") or {}
        shim.artifacts        = template_result.get("artifacts") or {}
        shim.template_version = template_result.get("template_version")
        shim.template_name    = template_result.get("template_name")

        # dispatch_fn for OOS: re-runs the template via the dispatcher.
        # We use the public dispatch_factor_spec wrapper to get a fresh
        # TemplateResult shape.
        def _oos_dispatch(oos_spec):
            from engine.agents.strengthener.factor_dispatcher import (
                TEMPLATE_REGISTRY,
            )
            tpl = TEMPLATE_REGISTRY.get(oos_spec.signal_kind)
            if tpl is None:
                return None
            return tpl(oos_spec)

        rigor_fn(
            spec              = spec,
            dispatch_fn       = _oos_dispatch,
            verdict           = outcome.verdict,
            hypothesis_id     = outcome.hypothesis_id,
            family            = outcome.family,
            template_result   = shim,
            verdict_event_id  = outcome.dispatch_event_id,
            ledger_path       = ledger_path,
        )
    except Exception as exc:
        logger.warning(
            "burndown_executor: post_green_rigor raised (suppressed): %s", exc,
        )


def _maybe_audit_verdict(
    outcome: "ExecutionOutcome",
    template_result: dict,
    *,
    budget_path: Optional[Path] = None,
    audit_fn = None,
) -> None:
    """External adversarial audit hook. Called for every GREEN/MARGINAL/RED
    verdict emitted by the cron path. NEVER raises.

    Skipped when:
      - verdict not in {GREEN, MARGINAL, RED} (no methodology to audit)
      - env var BURNDOWN_EXTERNAL_AUDIT_DISABLED=1 (tests)
      - weekly budget (EXTERNAL_AUDIT_WEEKLY_BUDGET_USD) already exhausted

    The audit record is appended to data/research/external_audits.jsonl by
    external_audit.audit_verdict_event. Critical/concern severities also
    surface in logs at WARNING level for cron-tail visibility.
    """
    import os
    if os.environ.get("BURNDOWN_EXTERNAL_AUDIT_DISABLED") == "1":
        return
    if outcome.verdict not in {"GREEN", "MARGINAL", "RED"}:
        return

    spent = _current_week_audit_spend(budget_path)
    if spent >= EXTERNAL_AUDIT_WEEKLY_BUDGET_USD:
        logger.info(
            "burndown_executor: external_audit skipped (weekly budget "
            "$%.2f exhausted; spent $%.2f)",
            EXTERNAL_AUDIT_WEEKLY_BUDGET_USD, spent,
        )
        return

    # Lazy-import the configured provider module so register_provider runs.
    # External_audit's _get_active_provider() then resolves the registered
    # adapter via EXTERNAL_AUDIT_PROVIDER env var.
    provider_name = os.environ.get("EXTERNAL_AUDIT_PROVIDER", "stub").lower()
    if provider_name == "deepseek":
        try:
            import engine.llm.providers.deepseek_external_audit_provider  # noqa: F401
        except Exception:
            logger.warning(
                "burndown_executor: DeepSeek provider import failed; "
                "audit will use stub", exc_info=True,
            )
    elif provider_name == "gemini":
        try:
            import engine.llm.providers.gemini_external_audit_provider  # noqa: F401
        except Exception:
            logger.warning(
                "burndown_executor: Gemini provider import failed; "
                "audit will use stub", exc_info=True,
            )

    try:
        if audit_fn is None:
            from engine.research.external_audit import audit_verdict_event
            audit_fn = audit_verdict_event
        event = {
            "event_id":   outcome.dispatch_event_id,
            "subject_id": outcome.hypothesis_id,
            "verdict":    outcome.verdict,
            "family":     outcome.family,
            "summary":    (template_result.get("summary") or "")[:300],
            "metrics":    template_result.get("metrics") or {},
        }
        record = audit_fn(event)
    except Exception as exc:
        logger.warning(
            "burndown_executor: external_audit raised (suppressed): %s", exc,
        )
        return

    # Record cost only if provider actually ran (skipped severity = no spend)
    if getattr(record, "severity", "") != "skipped":
        _record_audit_spend(
            getattr(record, "cost_estimate_usd", 0.0),
            getattr(record, "audit_id", ""),
            getattr(record, "severity", ""),
            budget_path=budget_path,
        )
    if getattr(record, "severity", "") in {"critical", "concern"}:
        logger.warning(
            "external_audit FLAG hypothesis=%s verdict=%s event=%s "
            "severity=%s categories=%s",
            outcome.hypothesis_id, outcome.verdict, outcome.dispatch_event_id,
            record.severity, getattr(record, "flagged_categories", []),
        )


def _load_hypothesis_by_id(hypothesis_id: str, *, hyp_path: Optional[Path] = None):
    """Read hypotheses.jsonl, locate the row, return a Hypothesis instance."""
    from engine.research.burndown_ranker import HYPOTHESES_PATH
    from engine.research_store.hypothesis.schema import Hypothesis

    path = hyp_path or HYPOTHESES_PATH
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("hypothesis_id") == hypothesis_id:
                try:
                    return Hypothesis.from_dict(row)
                except Exception as exc:
                    logger.warning(
                        "burndown_executor: Hypothesis.from_dict failed "
                        "for %s: %s", hypothesis_id, exc,
                    )
                    return None
    return None


class BurndownExecutor:
    """Dispatches a list of ranked candidates as ONE cron run.

    Construct with the cron_run_id (UUID4, generated by the cron script).
    Each candidate run tagged with this id so subsequent burndown_caps
    queries can aggregate by run.
    """

    def __init__(
        self,
        cron_run_id:      str,
        *,
        log_path:         Optional[Path] = None,
        spec_extractor_fn = None,   # injected for tests; default = real Sonnet call
        dispatcher_fn:    "object" = None,
        source:           str = "manual",   # burn-1c: "manual" | "auto"
        force_reason:     Optional[str] = None,   # burn-1c: audit string when --force used
    ):
        self.cron_run_id = cron_run_id
        self.log_path = log_path
        self._extract = spec_extractor_fn
        self._dispatch = dispatcher_fn
        self.source = source
        self.force_reason = force_reason

    def _resolve_extract(self):
        if self._extract is not None:
            return self._extract
        # Phase 2.1 (2026-06-13): switch to routing-aware extractor so cron
        # surfaces Stage-0 claim-shape refusals (NEEDS_NEW_TEMPLATE etc)
        # rather than the legacy generic EXTRACT_RETURNED_NONE. Returns
        # ExtractionResult; callers unwrap to spec.
        from engine.agents.strengthener.factor_spec_extractor import (
            extract_factor_spec_with_routing,
        )
        return extract_factor_spec_with_routing

    def _resolve_dispatch(self):
        if self._dispatch is not None:
            return self._dispatch
        from engine.agents.strengthener.factor_dispatcher import dispatch_factor_spec
        return dispatch_factor_spec

    def _log_extraction_failure(
        self, hypothesis_id: str, family: str, error_msg: str,
    ) -> Optional[str]:
        """burn-1c.1 — write a dispatch_log row when LLM extract fails so
        the next cron round dedups the hypothesis. Failure of the log
        write itself is non-fatal."""
        try:
            from engine.agents.strengthener.factor_dispatcher import (
                record_extraction_failure,
            )
            return record_extraction_failure(
                hypothesis_id = hypothesis_id,
                family_hint   = family.lower() if family else "",
                error_code    = (error_msg.split(":", 1)[0]
                                  if ":" in error_msg else error_msg)[:50],
                error_detail  = error_msg[:300],
                cron_run_id   = self.cron_run_id,
                cron_source   = self.source,
                log_path      = self.log_path,
            )
        except Exception as exc:
            logger.warning(
                "burndown_executor: record_extraction_failure failed "
                "for %s: %s", hypothesis_id, exc,
            )
            return None

    def execute_one(self, candidate) -> ExecutionOutcome:
        """Run ONE candidate end-to-end.

        `candidate` is a CandidateWithPrediction (or any obj with
        hypothesis_id + family attributes)."""
        hid = candidate.hypothesis_id
        fam = candidate.family
        ran_at = _utc_iso()

        # 1. Load the hypothesis row
        hyp = _load_hypothesis_by_id(hid)
        if hyp is None:
            return ExecutionOutcome(
                hypothesis_id     = hid,
                family            = fam,
                cron_run_id       = self.cron_run_id,
                extraction_ok     = False,
                extraction_error  = "HYPOTHESIS_NOT_FOUND",
                spec_hash         = None,
                refusal_reason    = None,
                verdict           = None,
                decay_severity    = None,
                dispatch_event_id = None,
                prediction_id     = None,
                ran_at            = ran_at,
            )

        # 2. Extract FactorSpec via LLM (Phase 2.1: now two-stage with
        # claim-shape router; legacy injected fn returns Optional[FactorSpec]
        # for backward compat — handled below).
        extract_fn = self._resolve_extract()
        try:
            extract_out = extract_fn(hyp)
        except Exception as exc:
            logger.exception("burndown_executor: extract raised for %s", hid)
            err_msg = f"EXTRACT_EXCEPTION: {type(exc).__name__}: {str(exc)[:120]}"
            ev_id = self._log_extraction_failure(hid, fam, err_msg)
            return ExecutionOutcome(
                hypothesis_id     = hid,
                family            = fam,
                cron_run_id       = self.cron_run_id,
                extraction_ok     = False,
                extraction_error  = err_msg,
                spec_hash         = None,
                refusal_reason    = None,
                verdict           = None,
                decay_severity    = None,
                dispatch_event_id = ev_id,
                prediction_id     = None,
                ran_at            = ran_at,
            )

        # Phase 2.1: unwrap ExtractionResult OR accept legacy Optional[FactorSpec]
        router_refusal: Optional[str] = None
        if hasattr(extract_out, "spec") and hasattr(extract_out, "router_verdict"):
            spec = extract_out.spec
            router_refusal = extract_out.refusal_reason
        else:
            spec = extract_out

        if spec is None:
            # Distinguish router refusal from generic extractor None.
            if router_refusal:
                err_msg = f"ROUTER_REFUSAL_{router_refusal}"
            else:
                err_msg = "EXTRACT_RETURNED_NONE"
            ev_id = self._log_extraction_failure(hid, fam, err_msg)
            return ExecutionOutcome(
                hypothesis_id     = hid,
                family            = fam,
                cron_run_id       = self.cron_run_id,
                extraction_ok     = False,
                extraction_error  = err_msg,  # ineligible / tool not called / etc
                spec_hash         = None,
                refusal_reason    = None,
                verdict           = None,
                decay_severity    = None,
                dispatch_event_id = ev_id,
                prediction_id     = None,
                ran_at            = ran_at,
            )

        # 3. Dispatch (with cron_run_id → bypasses WEEKLY_CAP)
        dispatch_fn = self._resolve_dispatch()
        dispatch_kwargs = dict(
            family_hint    = fam.lower(),
            spec_approved  = True,           # cron path = auto-approve at this gate
            cron_run_id    = self.cron_run_id,
            cron_source    = self.source,
            log_path       = self.log_path,
        )
        # burn-1c: --force path needs human_override (≥10 chars audit string)
        # since cron_run_id covers WEEKLY_CAP exemption but not all gates;
        # principal force-runs are equivalent to a manual override.
        if self.force_reason:
            dispatch_kwargs["human_override"] = (
                f"burndown_cron --force ({self.source}): {self.force_reason}"
            )
        try:
            result = dispatch_fn(spec, **dispatch_kwargs)
        except Exception as exc:
            logger.exception("burndown_executor: dispatch raised for %s", hid)
            return ExecutionOutcome(
                hypothesis_id     = hid,
                family            = fam,
                cron_run_id       = self.cron_run_id,
                extraction_ok     = True,
                extraction_error  = None,
                spec_hash         = None,
                refusal_reason    = f"DISPATCH_EXCEPTION: {type(exc).__name__}: {str(exc)[:120]}",
                verdict           = None,
                decay_severity    = None,
                dispatch_event_id = None,
                prediction_id     = None,
                ran_at            = ran_at,
            )

        refusal = result.get("refusal")
        tr = result.get("template_result") or {}
        verdict = tr.get("verdict")
        decay = None
        if isinstance(tr.get("metrics"), dict):
            oos = tr["metrics"].get("oos_triple")
            if isinstance(oos, dict):
                decay = oos.get("severity")

        outcome = ExecutionOutcome(
            hypothesis_id     = hid,
            family            = fam,
            cron_run_id       = self.cron_run_id,
            extraction_ok     = True,
            extraction_error  = None,
            spec_hash         = result.get("spec_hash"),
            refusal_reason    = refusal.get("reason_code") if refusal else None,
            verdict           = verdict,
            decay_severity    = decay,
            dispatch_event_id = result.get("dispatch_event_id"),
            prediction_id     = result.get("prediction_id"),
            ran_at            = ran_at,
        )
        # 2026-06-14 bug fix: prefer the in-memory template_result obj
        # (with DataFrame artifacts intact) over the jsonified dict for
        # rigor pipeline — _tr_to_jsonable strips pd.DataFrame so rigor
        # spanning sub-check was silently SKIPPING with "missing artifact"
        # note even when template DID expose pnl_series_df. The obj has
        # .artifacts dict containing the live DataFrame.
        tr_for_rigor = result.get("_template_result_obj")
        if tr_for_rigor is not None:
            # Build a shim-compatible dict view that rigor's
            # _maybe_run_post_green_rigor expects
            tr_rigor_dict = {
                "verdict":          getattr(tr_for_rigor, "verdict", verdict),
                "summary":          getattr(tr_for_rigor, "summary", ""),
                "metrics":          getattr(tr_for_rigor, "metrics", None) or {},
                "artifacts":        getattr(tr_for_rigor, "artifacts", None) or {},
                "template_version": getattr(tr_for_rigor, "template_version", None),
            }
        else:
            tr_rigor_dict = tr   # jsonified fallback (loses DataFrame)
        # Phase 1.2 (2026-06-13): external adversarial audit on every
        # GREEN/MARGINAL/RED cron verdict. Non-blocking; never raises.
        _maybe_audit_verdict(outcome, tr)
        # Phase 4.1 (2026-06-13): post-GREEN rigor pipeline (post-pub
        # OOS + FF5+MOM spanning). Fires on GREEN/MARGINAL only.
        # Non-blocking; never raises.
        _maybe_run_post_green_rigor(outcome, tr_rigor_dict, spec)
        return outcome

    def execute_plan(self, plan, *, respect_caps_mid_run: bool = True) -> list[ExecutionOutcome]:
        """Run every candidate in plan.candidates. Returns ordered outcomes.

        respect_caps_mid_run: re-check WeeklyUsage before each candidate;
        if family or global cap has bound, skip. Default True. False
        is for tests only.
        """
        from engine.research import burndown_caps

        outcomes: list[ExecutionOutcome] = []
        for c in plan.candidates:
            if respect_caps_mid_run:
                usage = burndown_caps.usage_last_7d(log_path=self.log_path)
                ok, reason = burndown_caps.can_dispatch(c.family, usage)
                if not ok:
                    logger.info(
                        "burndown_executor: skip %s family=%s (cap_hit_mid_run: %s)",
                        c.hypothesis_id, c.family, reason,
                    )
                    outcomes.append(ExecutionOutcome(
                        hypothesis_id     = c.hypothesis_id,
                        family            = c.family,
                        cron_run_id       = self.cron_run_id,
                        extraction_ok     = False,
                        extraction_error  = f"CAP_HIT_MID_RUN: {reason}",
                        spec_hash         = None,
                        refusal_reason    = None,
                        verdict           = None,
                        decay_severity    = None,
                        dispatch_event_id = None,
                        prediction_id     = None,
                        ran_at            = _utc_iso(),
                    ))
                    continue
            outcome = self.execute_one(c)
            outcomes.append(outcome)
        return outcomes


def write_outcomes(
    outcomes: list[ExecutionOutcome],
    plan_id: str,
    *,
    out_dir: Optional[Path] = None,
) -> Path:
    """Persist a run's outcomes to data/cron_burndown/outcomes/<date>_<id>.json."""
    d = out_dir or DEFAULT_OUTCOME_DIR
    d.mkdir(parents=True, exist_ok=True)
    date_str = _utc_iso()[:10]
    out_path = d / f"{date_str}_{plan_id[:8]}.json"
    out_path.write_text(
        json.dumps({
            "plan_id":   plan_id,
            "ran_at":    _utc_iso(),
            "outcomes":  [o.to_dict() for o in outcomes],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def summarize_outcomes(outcomes: list[ExecutionOutcome]) -> str:
    """Human-readable one-page run summary for stdout / Inbox digest."""
    lines: list[str] = []
    lines.append(f"=== burn-1b execution summary ({len(outcomes)} candidates) ===")
    verdict_counts: dict[str, int] = {}
    refusal_counts: dict[str, int] = {}
    extract_fail = 0
    for o in outcomes:
        if not o.extraction_ok:
            extract_fail += 1
            continue
        if o.refusal_reason:
            refusal_counts[o.refusal_reason] = refusal_counts.get(o.refusal_reason, 0) + 1
            continue
        v = o.verdict or "UNKNOWN"
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    lines.append(f"extraction failed:       {extract_fail}")
    lines.append("refusals (don't consume cap quota):")
    for k, v in sorted(refusal_counts.items()):
        lines.append(f"  {k}: {v}")
    lines.append("verdicts:")
    for k, v in sorted(verdict_counts.items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Per-candidate:")
    for i, o in enumerate(outcomes, 1):
        status = (
            f"verdict={o.verdict}" if o.verdict
            else f"refused={o.refusal_reason}" if o.refusal_reason
            else f"extract_err={o.extraction_error}"
        )
        decay = f" decay={o.decay_severity}" if o.decay_severity else ""
        lines.append(
            f"  {i}. {o.hypothesis_id[:8]}  [{o.family}]  {status}{decay}"
        )
    return "\n".join(lines)
