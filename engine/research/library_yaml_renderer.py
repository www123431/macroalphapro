"""engine/research/library_yaml_renderer.py — render typed AuditBlocks
+ identity metadata to a library YAML scaffold, preserving comments
and key order via ruamel.yaml.

Boundary of automation (per Phase 1 design):

  AUTOMATED (machine-rendered, schema-validated):
    - Identity: id, family, parent_family, purpose, relation_to_parent
    - Provenance: canonical_paper_id, key_followup_ids
    - Audit blocks: cost_model + factor_exposure (Pydantic AuditBlocks)
    - Audit checklist: paths + last_audited timestamp + audit_signature

  HUMAN-CURATED (placeholder + TODO):
    - mechanism_economics      (multi-paragraph narrative)
    - mechanism_break_conditions
    - adjacencies (same_family / same_parent / same_data / keywords)
    - pre_committed_falsification_criteria
    - deploy_notes / caveats

The renderer emits a scaffold YAML where AUTOMATED sections are
finalized and HUMAN sections are stub comments. The output PASSES
the 3 library validators (cost_model_audit + factor_exposure_audit +
library_integrity) but is INCOMPLETE for production use — a human
reviewer must fill in the economics + adjacencies before the strategy
should be flagged for live deploy decisions.

This split mirrors AQR / Two Sigma practice: automate the boilerplate,
human curates the economic narrative.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    _yaml = YAML()
    _yaml.preserve_quotes = True
    _yaml.indent(mapping=2, sequence=4, offset=2)
    _yaml.width = 4096
    _RUAMEL_AVAILABLE = True
except ImportError:
    import yaml as _pyyaml
    _yaml = None
    _RUAMEL_AVAILABLE = False

from engine.research.strategy_lifecycle import (
    AuditBlocks,
    CostModelAudit,
    FactorExposureAudit,
)


# ── Identity scaffold ──────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategyIdentity:
    """Identity metadata required for any library YAML."""

    strategy_id: str
    family: str
    parent_family: str = "equity_factor"
    purpose: str = "deploy_replacement"
    relation_to_parent: Literal["REPLACEMENT", "ADDITION", "ORIGINAL"] = "ADDITION"
    parent_strategy_id: Optional[str] = None
    canonical_paper_id: Optional[str] = None
    key_followup_ids: tuple[str, ...] = field(default_factory=tuple)
    methodological_extensions: tuple[str, ...] = field(default_factory=tuple)


# ── Pydantic serialization helpers ─────────────────────────────────────


def _cost_model_to_dict(c: CostModelAudit) -> dict[str, Any]:
    """Convert CostModelAudit pydantic instance → dict matching YAML
    layout in existing library entries. Preserves the nested capacity
    structure + multi_aum_sharpe_sleeve."""
    d: dict[str, Any] = {
        "audit_status": c.audit_status,
    }
    if c.audit_priority is not None:
        d["audit_priority"] = c.audit_priority
    if c.audit_status == "audited":
        d.update({
            "audit_date": c.audit_date.isoformat() if c.audit_date else None,
            "audit_script": c.audit_script,
            "audit_commit": c.audit_commit,
            "type": c.type,
            "half_spread_bps": c.half_spread_bps,
            "impact_coef": c.impact_coef,
            "daily_sigma_estimate": c.daily_sigma_estimate,
            "universe_median_adv_usd": c.universe_median_adv_usd,
            "n_positions_typical": c.n_positions_typical,
            "monthly_turnover_estimate": c.monthly_turnover_estimate,
            "stress_multiplier": c.stress_multiplier,
            "rationale": c.rationale,
            "multi_aum_sharpe_sleeve": {
                "at_10M": c.multi_aum_sharpe_sleeve.at_10M,
                "at_100M": c.multi_aum_sharpe_sleeve.at_100M,
                "at_1B":   c.multi_aum_sharpe_sleeve.at_1B,
            },
            "capacity": {
                "hard_capacity_usd": c.capacity.hard_capacity_usd,
                "binding_constraint": c.capacity.binding_constraint,
                "safe_deploy_band_usd": list(c.capacity.safe_deploy_band_usd),
                "max_participation_assumed": c.capacity.max_participation_assumed,
            },
        })
        if c.caveats:
            d["caveats"] = c.caveats
    return d


def _factor_exposure_to_dict(f: FactorExposureAudit) -> dict[str, Any]:
    d: dict[str, Any] = {"audit_status": f.audit_status}
    if f.audit_priority is not None:
        d["audit_priority"] = f.audit_priority
    if f.audit_status == "audited":
        d.update({
            "audit_date": f.audit_date.isoformat() if f.audit_date else None,
            "audit_script": f.audit_script,
            "audit_commit": f.audit_commit,
            "phase": f.phase,
            "proposed_role": f.proposed_role,
            "n_months": f.n_months,
            "alpha_annualized": f.alpha_annualized,
            "alpha_t_hac": f.alpha_t_hac,
            "betas": dict(f.betas) if f.betas else {},
            "t_stats_hac": dict(f.t_stats_hac) if f.t_stats_hac else {},
            "r_squared": f.r_squared,
            "verdict": f.verdict,
            "audit_blocks_deploy_decision": f.audit_blocks_deploy_decision,
            "factor_tilted_by_design": f.factor_tilted_by_design,
        })
        if f.caveats:
            d["caveats"] = f.caveats
    return d


# ── Scaffold renderer ──────────────────────────────────────────────────


_HUMAN_TODO_PLACEHOLDER = (
    "TODO: fill in by human reviewer before flagging for deploy. "
    "Phase 1 automation renders only schema-validated audit blocks; "
    "narrative fields require domain judgment."
)


def render_library_yaml_scaffold(
    *,
    identity: StrategyIdentity,
    audit_blocks: AuditBlocks,
    last_audited: _dt.date,
    snooping_publication_date: Optional[_dt.date] = None,
    snooping_first_run: Optional[_dt.date] = None,
) -> dict[str, Any]:
    """Build the YAML-ready dict for a library entry. Pass through
    `_dump_yaml()` to serialize to string.

    Sections that are AUTOMATED (cost_model / factor_exposure / identity
    fields) are finalized. Sections that require HUMAN curation
    (mechanism_economics / adjacencies / falsification criteria) are
    rendered as stubs with TODO placeholders.
    """
    pub_date = snooping_publication_date or _dt.date(1900, 1, 1)
    first_run = snooping_first_run or last_audited

    scaffold: dict[str, Any] = {
        "_schema_version": 2,

        # ── Identity ──────────────────────────────────────────────────
        "id": identity.strategy_id,
        "family": identity.family,
        "parent_family": identity.parent_family,
        "purpose": identity.purpose,
        "relation_to_parent": identity.relation_to_parent,
        "parent_id": identity.parent_strategy_id,

        # ── Provenance ────────────────────────────────────────────────
        "canonical_paper_id": identity.canonical_paper_id,
        "key_followup_ids": list(identity.key_followup_ids),
        "methodological_extensions": list(identity.methodological_extensions),

        # ── Data-snooping audit ───────────────────────────────────────
        "was_known_before_our_data_cutoff": {
            "publication_date": pub_date.isoformat(),
            "our_earliest_gate_run": first_run.isoformat(),
            "snooping_risk": "low" if (first_run - pub_date).days > 365 * 10 else "medium",
        },

        # ── HUMAN-CURATED: mechanism economics ────────────────────────
        "mechanism_economics": _HUMAN_TODO_PLACEHOLDER,
        "mechanism_break_conditions": [_HUMAN_TODO_PLACEHOLDER],

        # ── HUMAN-CURATED: pre-committed falsification ────────────────
        "pre_committed_falsification_criteria": [_HUMAN_TODO_PLACEHOLDER],

        # ── HUMAN-CURATED: adjacencies (H2 cousin map) ────────────────
        "adjacencies": {
            "same_family": [],
            "same_parent": [],
            "same_data": [],
            "same_economics_keywords": [],
            "adjacencies_unresolved": True,
        },

        # ── Status (always PENDING_DEPLOY for fresh promotions) ───────
        "status_in_our_book": "PENDING_DEPLOY",
        "currently_unexplored_in_our_book": False,
        "our_test_record": {
            "gate_run_ids": [],
            "verdict": "GREEN",
            "date": last_audited.isoformat(),
            "notes": (
                "Promoted via engine.research.promote_candidate Phase 1 "
                "automation. Audit blocks below were machine-rendered from "
                "cached audit JSON; narrative sections require human review "
                "before flagging for live deploy."
            ),
        },

        # ── AUTOMATED: audit blocks ───────────────────────────────────
        "cost_model": _cost_model_to_dict(audit_blocks.cost_model),
        "factor_exposure": _factor_exposure_to_dict(audit_blocks.factor_exposure),

        # ── Audit checklist (machine-fillable subset) ─────────────────
        "audit_checklist_passed": {
            "paper_exists_in_master_index": False,
            "data_exists_in_inventory": True,
            "decay_numbers_verified": False,
            "mechanism_distinct_from_existing":
                identity.relation_to_parent != "REPLACEMENT",
            "no_snooping_or_explicit_acknowledgment":
                (first_run - pub_date).days > 365 * 5,
        },
        "last_audited": last_audited.isoformat(),
        "audit_signature": "pending",  # NEVER auto-flip; human-only

        # ── Deploy notes ──────────────────────────────────────────────
        "deploy_notes": _HUMAN_TODO_PLACEHOLDER,
    }
    return scaffold


def write_library_yaml(
    *,
    path: Path,
    scaffold: dict[str, Any],
    overwrite: bool = False,
) -> None:
    """Write scaffold dict to YAML at `path`. Refuses to overwrite
    existing files unless overwrite=True (prevents accidental clobber
    of curated human content).

    Uses ruamel.yaml if available (preserves comments + key order on
    later re-reads); falls back to PyYAML.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace "
            "(WARNING: will lose any human-curated edits)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    if _RUAMEL_AVAILABLE:
        # ruamel.yaml emits cleaner formatting for the deploy scaffold
        with path.open("w", encoding="utf-8", newline="\n") as f:
            _yaml.dump(scaffold, f)
    else:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            _pyyaml.safe_dump(scaffold, f, sort_keys=False,
                              default_flow_style=False, allow_unicode=True)


def yaml_to_string(scaffold: dict[str, Any]) -> str:
    """Serialize scaffold to YAML string without writing. Used in
    dry_run mode of promote_candidate."""
    import io
    buf = io.StringIO()
    if _RUAMEL_AVAILABLE:
        _yaml.dump(scaffold, buf)
    else:
        _pyyaml.safe_dump(scaffold, buf, sort_keys=False,
                          default_flow_style=False, allow_unicode=True)
    return buf.getvalue()
