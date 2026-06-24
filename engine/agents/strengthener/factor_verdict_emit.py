"""engine.agents.strengthener.factor_verdict_emit — Tier C-2c.

Bridges the C-2b template result (GREEN/MARGINAL/RED Sharpe-based
verdict) into the research event store via emit.factor_verdict.

This is the second of three concerns the dispatcher pipeline cares
about:

  audit_log (C-2a):    raw dispatch record incl refusals; private
                         debugging surface (data/strengthener/
                         factor_dispatch_log.jsonl)
  event_store (C-2c):  typed factor_verdict_filed events; the
                         canonical research surface that n_trials
                         counter, decay sentinel, and UI consumers
                         already query
  capability evidence: per-test markdown stub (one per dispatch)
                         so emit.factor_verdict's artifacts.evidence_doc
                         contract is satisfied. Lightweight — just the
                         FactorSpec JSON + metrics in a md wrapper

Per CLAUDE.md Research Event Emission Doctrine (2026-06-02, STANDING):
  - subject_id MUST be in the registry → auto-registered idempotently
    here per (hypothesis_id, signal_kind) pair with deterministic
    naming so re-dispatches of the same spec_hash hit the same subject
  - artifacts.evidence_doc MUST exist on disk before emit → we write
    the markdown stub first, then emit
  - summary 1-2 sentences ≤ 400 chars
  - parent_event_ids filled when this verdict descends from a prior
    event (e.g. a B strengthener approval row)

Per A+B doctrine [[project-a-plus-b-substrate-first-roadmap-2026-06-05]]:
  Verdict emission does NOT auto-promote anything. The downstream
  paper-trade promoter + SLM check the auto_test_* tags in metrics
  and refuse to auto-act on a Tier C verdict — the human still owns
  capital decisions.

Only GREEN/MARGINAL/RED template verdicts emit factor_verdict_filed.
Dispatcher-internal verdicts (PENDING_TEMPLATE_BUILD, DATA_ERROR,
EXECUTION_ERROR, INSUFFICIENT_HISTORY, UNSUPPORTED_UNIVERSE,
CUSTOM_CODE_REQUIRED) stay in the dispatch log only — they aren't
research findings, they're system states.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

from engine.agents.strengthener.factor_spec_extractor import FactorSpec
from engine.agents.strengthener.factor_dispatcher import (
    DISPATCHER_VERSION, TemplateResult, _spec_hash,
)

logger = logging.getLogger(__name__)


# Verdicts that warrant emission. Dispatcher-internal states stay
# in the dispatch log; only template-level Sharpe-based verdicts
# become research events.
_EMITTABLE_VERDICTS = frozenset({"GREEN", "MARGINAL", "RED"})


# Where capability evidence stubs land. One markdown per dispatch.
_REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = (_REPO_ROOT / "docs" / "capability_evidence"
                  / "tier_c_auto")

# L2-4 prep (2026-06-08): per-dispatch monthly PnL series lands here
# as parquet, NOT in docs/ — binary regenerable data belongs in
# data/. One parquet per (spec_hash, verdict) pair. Substrate for
# L2-4 anchor orthogonality, L2-5 subsample stability, L2-6
# attribution.
PNL_DIR = (_REPO_ROOT / "data" / "research_store" / "tier_c_pnl")


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ────────────────────────────────────────────────────────────────────
# Subject identity — deterministic per (hypothesis, signal_kind)
# ────────────────────────────────────────────────────────────────────
def auto_subject_id(spec: FactorSpec) -> str:
    """Stable subject_id for a Tier C auto-test. Same hypothesis +
    same signal_kind always resolve to the same subject so
    re-dispatches accumulate verdict history under one identity
    (visible in /research surfaces).

    Format: tier_c_auto_<hypothesis_id_short>_<signal_kind>
    """
    short_hid = (spec.hypothesis_id or "unknown")[:8]
    return f"tier_c_auto_{short_hid}_{spec.signal_kind}"


def ensure_subject_registered(
    spec:        FactorSpec,
    family_hint: str,
) -> str:
    """Idempotent register the auto-subject. Returns subject_id.
    Per CLAUDE.md doctrine, subject_id MUST exist in registry
    before emit; this is where it gets there for Tier C auto-tests.
    """
    from engine.research_store import registry
    from engine.research_store.schema import SubjectType

    sid = auto_subject_id(spec)
    description = (
        f"Tier C auto-dispatched factor test. hypothesis_id="
        f"{spec.hypothesis_id} signal_kind={spec.signal_kind} "
        f"universe={spec.universe} extractor_model={spec.model}"
    )
    registry.register_subject(
        subject_id   = sid,
        subject_type = SubjectType.factor,
        family       = family_hint or "OTHER",
        description  = description,
        created_by   = "tier_c_auto_dispatcher",
    )
    return sid


# ────────────────────────────────────────────────────────────────────
# Capability evidence stub
# ────────────────────────────────────────────────────────────────────
def write_capability_evidence(
    spec:               FactorSpec,
    template_result:    TemplateResult,
    dispatch_event_id:  Optional[str],
    family_hint:        str,
) -> Path:
    """Write a markdown stub holding the SPEC + TemplateResult so
    emit.factor_verdict's artifacts.evidence_doc contract is met
    (file MUST exist on disk before emit). One file per dispatch.

    Path: docs/capability_evidence/tier_c_auto/<spec_hash>_<verdict>.md
    """
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    sh = _spec_hash(spec)
    path = EVIDENCE_DIR / f"{sh}_{template_result.verdict}.md"

    metrics_lines = []
    for k, v in sorted(template_result.metrics.items()):
        if isinstance(v, float):
            metrics_lines.append(f"| `{k}` | {v:.6g} |")
        else:
            metrics_lines.append(f"| `{k}` | {v} |")
    metrics_table = "\n".join(metrics_lines) if metrics_lines \
        else "_(no metrics)_"

    body = f"""# Tier C auto-dispatch — `{template_result.verdict}`

**Tags**: `tier_c_auto` `{spec.signal_kind}` `{family_hint}`
**Spec hash**: `{sh}`
**Dispatched**: {_utc_iso()}
**Dispatcher version**: `{DISPATCHER_VERSION}`
**Template version**: `{template_result.template_version}`
**Dispatch event id**: `{dispatch_event_id or '(dry_run / not persisted)'}`

This file is the capability_evidence stub for one Tier C
auto-dispatched factor backtest. Per A+B doctrine the GREEN/MARGINAL
verdict is RESEARCH data only — does not auto-promote to paper_trade.

## Source

| key | value |
|---|---|
| `hypothesis_id` | `{spec.hypothesis_id}` |
| `family_hint` | `{family_hint}` |
| `extractor_workload` | `strengthener_factor_spec` |
| `extractor_model` | `{spec.model}` |
| `extracted_ts` | `{spec.extracted_ts}` |

## SPEC (LLM-extracted, human-approved)

| key | value |
|---|---|
| `signal_kind` | `{spec.signal_kind}` |
| `universe` | `{spec.universe}` |
| `date_range` | `{spec.date_range}` |
| `rebal` | `{spec.rebal}` |
| `weighting` | `{spec.weighting}` |
| `expected_holding_period` | `{spec.expected_holding_period}` |
| `min_obs_months` | `{spec.min_obs_months}` |
| `pit_audits` | `{list(spec.pit_audits)}` |
| `cost_model` | `{spec.cost_model}` |
| `signal_inputs` | `{list(spec.signal_inputs)}` |

**Rationale (extractor)**: {spec.rationale}

## Verdict

**{template_result.verdict}** — {template_result.summary}

## Metrics

| metric | value |
|---|---|
{metrics_table}

## Provenance

This is an AUTO-DISPATCHED test. Lineage:
1. Hypothesis `{spec.hypothesis_id}` (from Employee A synthesis or extraction)
2. Employee B reviewed + approved → strengthener_approval
3. `strengthener_factor_spec` extracted SPEC (model={spec.model})
4. Human approved SPEC in /approvals
5. `factor_dispatcher.dispatch_factor_spec` ran gates + template
6. This evidence stub + emitted `factor_verdict_filed` event

Re-dispatching the same SPEC produces the same `spec_hash`; this
file is OVERWRITTEN on re-dispatch by design (most recent run is
the relevant evidence).
"""
    path.write_text(body, encoding="utf-8")
    return path


# ────────────────────────────────────────────────────────────────────
# L2-4 prep: PnL series parquet persistence
# ────────────────────────────────────────────────────────────────────
def write_pnl_parquet(
    spec:             FactorSpec,
    template_result:  TemplateResult,
) -> Optional[Path]:
    """Persist the template's monthly PnL DataFrame as parquet so
    L2-4 anchor orthogonality / L2-5 subsample / L2-6 attribution
    can re-read it without re-running the full backtest.

    Reads `template_result.artifacts["pnl_series_df"]`, which the
    template populates as a pandas DataFrame indexed by month-end
    with columns: pnl_gross / pnl_net_13bp / pnl_net_80bp / turnover.

    Returns the parquet path on success, None when:
      - template didn't emit pnl_series_df (older templates)
      - DataFrame is empty
      - pyarrow not installed
      - filesystem write failed

    Failures NEVER block verdict emission — PnL persistence is a
    follow-on capability, not the primary contract.

    Path: data/research_store/tier_c_pnl/<spec_hash>_<verdict>.parquet
    Overwrites on re-dispatch (most recent run is authoritative,
    same convention as the markdown evidence stub).
    """
    art = (template_result.artifacts or {}).get("pnl_series_df")
    if art is None:
        return None
    try:
        import pandas as pd_local
    except ImportError:
        logger.warning("pnl persist: pandas not available")
        return None
    if not isinstance(art, pd_local.DataFrame):
        logger.warning("pnl persist: pnl_series_df is %s not DataFrame",
                          type(art).__name__)
        return None
    if art.empty:
        return None
    try:
        PNL_DIR.mkdir(parents=True, exist_ok=True)
        sh = _spec_hash(spec)
        path = PNL_DIR / f"{sh}_{template_result.verdict}.parquet"
        # Reset index so the month-end timestamps survive as a
        # column ("date") — pyarrow handles plain index but
        # downstream consumers may not know to look at the index.
        out_df = art.copy()
        out_df.index.name = "date"
        out_df.reset_index(inplace=True)
        out_df.to_parquet(path, index=False)
        return path
    except ImportError:
        logger.warning("pnl persist: pyarrow not installed; "
                          "install pyarrow to enable L2-4/5/6 substrate")
        return None
    except (OSError, ValueError) as exc:
        logger.warning("pnl persist: write failed for %s: %s",
                          spec.hypothesis_id, exc)
        return None


# ────────────────────────────────────────────────────────────────────
# Main entry — emit the typed event
# ────────────────────────────────────────────────────────────────────
def emit_tier_c_verdict(
    spec:                  FactorSpec,
    family_hint:           str,
    template_result:       TemplateResult,
    dispatch_event_id:     Optional[str] = None,
    parent_event_ids:      tuple = (),
    self_doubt                = None,
    anchor_orthogonality:     Optional[dict] = None,
    subsample_stability:      Optional[dict] = None,
    industry_extension:       Optional[dict] = None,
    cross_asset_extension:    Optional[dict] = None,
    specification_robustness: Optional[dict] = None,
    pnl_diagnostics:          Optional[dict] = None,
    routing_decisions:        Optional[list] = None,
) -> Optional[str]:
    """Emit factor_verdict_filed for a Tier C auto-dispatched test.

    Returns the new event_id on success; None when verdict is
    dispatcher-internal (PENDING_TEMPLATE_BUILD, DATA_ERROR, etc.)
    or when emit raises.

    Args:
      spec: the FactorSpec the template ran on
      family_hint: mechanism_family from the source Hypothesis (the
                   subject's family + event's family field)
      template_result: the TemplateResult from dispatcher
      dispatch_event_id: matching row in factor_dispatch_log.jsonl
                          (recorded in metrics for JOIN, not for
                          parent_event_ids — that's reserved for
                          true event-store lineage)
      parent_event_ids: e.g. strengthener_approval event_id when
                       /approvals C-2d wires this in
      self_doubt: optional SelfDoubtAssessment from L3-2 (2026-06-08).
                   If provided, serialized into event metrics under
                   `self_doubt` key for downstream consumers
                   (/approvals UI, audit) to surface.
    """
    if template_result.verdict not in _EMITTABLE_VERDICTS:
        logger.debug("tier_c_emit: verdict=%s not emittable; "
                       "audit log only", template_result.verdict)
        return None

    # 1. Subject registration (idempotent)
    try:
        sid = ensure_subject_registered(spec, family_hint)
    except Exception as exc:
        logger.exception("tier_c_emit: subject registration failed")
        return None

    # 2. Capability evidence stub on disk (required before emit)
    try:
        evidence_path = write_capability_evidence(
            spec, template_result, dispatch_event_id, family_hint)
    except OSError as exc:
        logger.exception("tier_c_emit: evidence write failed")
        return None

    # 3. Build metrics — fold dispatcher provenance + template
    #    metrics into one dict; emit's contract is a single metrics
    sh = _spec_hash(spec)
    metrics = dict(template_result.metrics or {})
    metrics.update({
        "auto_test_spec_hash":  sh,
        "auto_test_llm_model":  spec.model,
        "extractor_workload":   "strengthener_factor_spec",
        "dispatcher_version":   DISPATCHER_VERSION,
        "template_version":     template_result.template_version,
        "source_hypothesis_id": spec.hypothesis_id,
        "dispatch_event_id":    dispatch_event_id,
        "tier_c_auto":          True,
    })

    # L3-2 (2026-06-08): include self_doubt assessment in event
    # metrics. Downstream consumers (UI, audit) can read it to
    # display confidence + caveats prominently.
    if self_doubt is not None:
        import dataclasses as _dc_local
        metrics["self_doubt"] = _dc_local.asdict(self_doubt)

    # L2-4 Commit 3 (2026-06-09): include anchor_orthogonality
    # residual-alpha regression result. Already JSON-safe (caller
    # uses compute_for_tier_c_pnl_series which strips pd.Series).
    if anchor_orthogonality is not None:
        metrics["anchor_orthogonality"] = dict(anchor_orthogonality)

    # L2-5 Commit 2 (2026-06-09): include subsample_stability
    # decomposition. Already JSON-safe.
    if subsample_stability is not None:
        metrics["subsample_stability"] = dict(subsample_stability)

    # L2-6 Commit 3 (2026-06-09, post-FWL-fix): include JOINT-model
    # industry extension. Already JSON-safe (caller uses
    # compute_for_tier_c_with_stage1_residual which strips pd.Series).
    if industry_extension is not None:
        metrics["industry_extension"] = dict(industry_extension)

    # Cross-asset Commit 4 (2026-06-09): include cross-asset macro
    # extension (joint FF5+MOM + Industry? + Macro). Already JSON-safe.
    if cross_asset_extension is not None:
        metrics["cross_asset_extension"] = dict(cross_asset_extension)

    # B (2026-06-09 senior施工建议): specification robustness ablation.
    # n_trials_increment=0 by contract — do not double-count when
    # downstream consumers (n_trials_family_counter) read this event.
    if specification_robustness is not None:
        metrics["specification_robustness"] = dict(specification_robustness)

    # N (2026-06-10): the 4 senior gates (DSR / rho1 / paper-OOS / power).
    if pnl_diagnostics is not None:
        metrics["pnl_diagnostics"] = dict(pnl_diagnostics)

    # Phase 1 Commit 3 (2026-06-09, role-routing): routing audit trail
    # (per docs/spec_role_aware_test_routing.md §15.A5). Records every
    # lens decision (executed / skipped / failed) for downstream UI +
    # L3-2 self_doubt transparency.
    if routing_decisions is not None:
        metrics["routing_decisions"] = list(routing_decisions)

    # L2-4 prep (2026-06-08): persist PnL series as parquet.
    # Failure does NOT block emission — persistence is a follow-on
    # capability layer. Path recorded in metrics + artifacts so L2-4/
    # L2-5/L2-6 consumers can locate the data.
    try:
        pnl_path = write_pnl_parquet(spec, template_result)
    except Exception:
        logger.exception("pnl persist: unexpected error for %s",
                          spec.hypothesis_id)
        pnl_path = None
    pnl_doc: Optional[str] = None
    if pnl_path is not None:
        try:
            pnl_doc = str(pnl_path.relative_to(_REPO_ROOT).as_posix())
        except ValueError:
            pnl_doc = str(pnl_path)
        metrics["pnl_series_parquet"] = pnl_doc

    # 4. Emit
    from engine.research_store import emit
    # Evidence path: relative-to-repo when under repo (production),
    # else absolute (tests using tmp_path outside repo). Both forms
    # satisfy emit's "file MUST exist on disk" contract.
    try:
        evidence_doc = str(evidence_path.relative_to(_REPO_ROOT)
                              .as_posix())
    except ValueError:
        evidence_doc = str(evidence_path)
    artifacts_payload = {"evidence_doc": evidence_doc}
    if pnl_doc is not None:
        artifacts_payload["pnl_series_parquet"] = pnl_doc
    # 2026-06-12 fix: event.family carries the canonical
    # strategy_family (Bailey-LdP denominator); claim_family
    # (mechanism_family from paper) is preserved as a tag for
    # paper-origin queries. See engine.research.strategy_family_classifier.
    try:
        from engine.research.strategy_family_classifier import (
            strategy_family_for_spec, canonical_strategy_family_tag,
            claim_family_tag,
        )
        strategy_family = strategy_family_for_spec(spec)
        sf_tag = canonical_strategy_family_tag(spec)
        cf_tag = claim_family_tag(family_hint)
    except Exception:
        logger.exception("tier_c_emit: strategy_family computation failed; "
                          "falling back to family_hint")
        strategy_family = family_hint
        sf_tag = f"strategy_family:{(family_hint or '').lower()}"
        cf_tag = f"claim_family:{(family_hint or '').upper()}"

    metrics["strategy_family"] = strategy_family
    metrics["claim_family"]    = family_hint

    try:
        event_id = emit.factor_verdict(
            subject_id       = sid,
            verdict          = template_result.verdict,
            metrics          = metrics,
            artifacts        = artifacts_payload,
            summary          = template_result.summary[:400],
            parent_event_ids = parent_event_ids,
            family           = strategy_family,
            tags             = ("tier_c_auto", spec.signal_kind,
                                  sf_tag, cf_tag),
            actor            = "engine.agents.strengthener.factor_dispatcher",
        )
        return event_id
    except Exception as exc:
        logger.exception("tier_c_emit: emit.factor_verdict failed for %s",
                          spec.hypothesis_id)
        return None
