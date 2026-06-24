"""engine.agents.strengthener.factor_spec_store — Tier C-2d backend.

Persistence + queue + resolution flow for Tier C factor SPECs
between extraction (C-1) and dispatch (C-2a+b). Mirrors the shape
of approval_view.py but operates on FactorSpec objects, not B
verdicts.

Flow (after B APPROVE_FOR_PIPELINE → human approves in /approvals):
  api/routes_strengthener.resolve_approval (existing)
   → _emit_forward_vector_created (existing)
   → factor_spec_store.extract_and_persist_pending  (NEW C-2d)
     → factor_spec_extractor.extract_factor_spec (Sonnet ~$0.03)
     → factor_specs.jsonl append
   → /approvals UI second card appears
   → human reviews SPEC + approves (or rejects)
   → factor_spec_store.resolve_factor_spec
     → on APPROVED: dispatch_factor_spec (gates + template + emit)
     → factor_spec_resolutions.jsonl append
     → /approvals card moves to resolved

Storage (append-only jsonl, same paradigm as the rest of Phase 2.0):
  data/strengthener/factor_specs.jsonl         — extracted SPECs
  data/strengthener/factor_spec_resolutions.jsonl — human decisions

Resolution model — keyed by spec_hash:
  spec_hash is the canonical id of a SPEC (stable across re-runs
  of the same controlled fields). Re-extracting the same hypothesis
  produces the same spec_hash → idempotent for the queue (no
  duplicate pending row).

Why spec_hash and not row-uuid:
  If the LLM extractor is rerun (e.g. prompt tuned) and emits a
  different SPEC for the same hypothesis, the spec_hash CHANGES.
  That should appear as a NEW pending row (the human should review
  the new SPEC, not auto-inherit the prior decision). Keying by
  spec_hash gets us that for free.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SPECS_PATH = (_REPO_ROOT / "data" / "strengthener"
                          / "factor_specs.jsonl")
_DEFAULT_RESOLUTIONS_PATH = (_REPO_ROOT / "data" / "strengthener"
                                / "factor_spec_resolutions.jsonl")

_DECISIONS = {"approved", "rejected", "deferred"}


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ────────────────────────────────────────────────────────────────────
# Persistence helpers
# ────────────────────────────────────────────────────────────────────
def _iter_jsonl(p: Path):
    if not p.is_file():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _append_jsonl(p: Path, record: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────────────────────────
# Persistence: pending SPEC + resolution
# ────────────────────────────────────────────────────────────────────
def _spec_to_payload(spec, family_hint: str,
                       source_hypothesis_id: str,
                       spec_hash: str) -> dict:
    """Serialize a FactorSpec + extra metadata into a jsonl row.
    Plain dict (no dataclass-asdict on tuple fields) for stable
    on-disk shape."""
    return {
        "spec_hash":            spec_hash,
        "source_hypothesis_id": source_hypothesis_id,
        "family_hint":          family_hint,
        "persisted_ts":         _utc_iso(),
        "spec": {
            "hypothesis_id":           spec.hypothesis_id,
            "signal_kind":             spec.signal_kind,
            "universe":                spec.universe,
            "date_range":              spec.date_range,
            "signal_inputs":           list(spec.signal_inputs),
            "rebal":                   spec.rebal,
            "weighting":               spec.weighting,
            "expected_holding_period": spec.expected_holding_period,
            "min_obs_months":          spec.min_obs_months,
            "pit_audits":              list(spec.pit_audits),
            "cost_model":              spec.cost_model,
            "rationale":               spec.rationale,
            "extracted_ts":            spec.extracted_ts,
            "model":                   spec.model,
            # L2-2 replication fields (optional, default None)
            "paper_original_window":   spec.paper_original_window,
            "paper_reported_t":        spec.paper_reported_t,
            # L2-1 Phase 2.6 B-class params (optional, default None)
            "universe_size":           spec.universe_size,
            "n_buckets":               spec.n_buckets,
            "signal_lookback_m":       spec.signal_lookback_m,
            "signal_skip_m":           spec.signal_skip_m,
            "vol_target_annual":       spec.vol_target_annual,
            "weighting_scheme_alt":    spec.weighting_scheme_alt,
        },
    }


def _payload_to_factor_spec(payload: dict):
    """Inverse of _spec_to_payload — build a FactorSpec from a
    jsonl row's 'spec' sub-dict. Used by resolve_factor_spec to
    invoke dispatcher on the original spec object."""
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    s = payload["spec"]
    return FactorSpec(
        hypothesis_id           = s["hypothesis_id"],
        signal_kind             = s["signal_kind"],
        universe                = s["universe"],
        date_range              = s["date_range"],
        signal_inputs           = tuple(s.get("signal_inputs") or ()),
        rebal                   = s["rebal"],
        weighting               = s["weighting"],
        expected_holding_period = s["expected_holding_period"],
        min_obs_months          = int(s["min_obs_months"]),
        pit_audits              = tuple(s.get("pit_audits") or ()),
        cost_model              = s["cost_model"],
        rationale               = s["rationale"],
        extracted_ts            = s["extracted_ts"],
        model                   = s["model"],
        # L2-2 replication fields (optional; .get for backward-compat
        # with pre-L2-2 persisted specs)
        paper_original_window   = s.get("paper_original_window"),
        paper_reported_t        = s.get("paper_reported_t"),
        # L2-1 Phase 2.6 B-class params (optional; .get for
        # backward-compat with pre-v2 persisted specs)
        universe_size           = s.get("universe_size"),
        n_buckets               = s.get("n_buckets"),
        signal_lookback_m       = s.get("signal_lookback_m"),
        signal_skip_m           = s.get("signal_skip_m"),
        vol_target_annual       = s.get("vol_target_annual"),
        weighting_scheme_alt    = s.get("weighting_scheme_alt"),
    )


@_dc.dataclass(frozen=True)
class FactorSpecResolution:
    spec_hash:           str
    decision:            str          # approved / rejected / deferred
    rationale:           str
    resolved_ts:         str
    resolved_by:         str
    dispatch_event_id:   Optional[str] = None   # populated on APPROVED
    verdict_event_id:    Optional[str] = None   # populated when emit fires


def _load_resolutions(path: Path) -> dict[str, FactorSpecResolution]:
    """Latest-wins per spec_hash. Append-only file; downstream takes
    the last row for each key."""
    latest: dict[str, FactorSpecResolution] = {}
    for r in _iter_jsonl(path):
        sh = r.get("spec_hash")
        if not sh:
            continue
        latest[sh] = FactorSpecResolution(
            spec_hash         = sh,
            decision          = r.get("decision", ""),
            rationale         = r.get("rationale", ""),
            resolved_ts       = r.get("resolved_ts", ""),
            resolved_by       = r.get("resolved_by", ""),
            dispatch_event_id = r.get("dispatch_event_id"),
            verdict_event_id  = r.get("verdict_event_id"),
        )
    return latest


def _load_specs(path: Path) -> dict[str, dict]:
    """First-wins per spec_hash (the first time a SPEC was extracted
    is the canonical row; subsequent re-extractions for the same
    hash are no-ops at queue level)."""
    out: dict[str, dict] = {}
    for r in _iter_jsonl(path):
        sh = r.get("spec_hash")
        if sh and sh not in out:
            out[sh] = r
    return out


# ────────────────────────────────────────────────────────────────────
# Public API — extract + persist
# ────────────────────────────────────────────────────────────────────
def extract_and_persist_pending(
    hypothesis,
    family_hint:       str,
    *,
    specs_path:        Optional[Path] = None,
) -> Optional[str]:
    """Run factor_spec_extractor on a B-approved hypothesis and
    persist the SPEC to factor_specs.jsonl as a pending approval.

    Idempotent: if the resulting spec_hash already exists in the
    queue, no duplicate row is written.

    Returns the spec_hash on success, or None when:
      - the hypothesis doesn't fit factor-spec extraction (procedural,
        methodology subtype, no provenance — see is_factor_hypothesis)
      - the LLM extractor returned None (timeout, tool not called,
        invalid enum)

    Args:
      hypothesis: a Hypothesis dataclass (loaded from hypotheses.jsonl).
                  Must have hypothesis_id, mechanism_family.value,
                  mechanism_subtype, predicted_direction, etc.
      family_hint: mechanism_family value used by dispatcher's
                   n_trials check + persisted alongside the SPEC
                   for downstream JOINs
      specs_path: tests override; prod uses _DEFAULT_SPECS_PATH
    """
    from engine.agents.strengthener.factor_spec_extractor import (
        extract_factor_spec,
    )
    from engine.agents.strengthener.factor_dispatcher import _spec_hash

    path = specs_path or _DEFAULT_SPECS_PATH

    spec = extract_factor_spec(hypothesis)
    if spec is None:
        logger.debug("extract_and_persist: extractor returned None for %s",
                       getattr(hypothesis, "hypothesis_id", "?"))
        return None

    sh = _spec_hash(spec)
    existing = _load_specs(path)
    if sh in existing:
        logger.debug("extract_and_persist: spec_hash %s already pending; "
                       "skip duplicate write", sh)
        return sh

    payload = _spec_to_payload(
        spec, family_hint=family_hint,
        source_hypothesis_id=hypothesis.hypothesis_id, spec_hash=sh,
    )
    _append_jsonl(path, payload)
    return sh


# ────────────────────────────────────────────────────────────────────
# Public API — queue view
# ────────────────────────────────────────────────────────────────────
def list_pending_factor_specs(
    *,
    specs_path:        Optional[Path] = None,
    resolutions_path:  Optional[Path] = None,
    include_resolved:  bool = False,
) -> dict:
    """Return the structured payload the /approvals UI consumes
    for SPEC approvals. Shape mirrors approval_view.list_pending_approvals.

    Returns:
      {
        "n_pending":   int,
        "n_resolved":  int,
        "rows": [
          {
            "spec_hash":            str,
            "source_hypothesis_id": str,
            "family_hint":          str,
            "persisted_ts":         str,
            "spec":                 {full spec dict},
            "resolved":             bool,
            "resolution":           {decision, rationale, ts,
                                      dispatch_event_id,
                                      verdict_event_id} | None,
          }, ...
        ],
      }
    """
    sp = specs_path or _DEFAULT_SPECS_PATH
    rp = resolutions_path or _DEFAULT_RESOLUTIONS_PATH

    specs = _load_specs(sp)
    resolutions = _load_resolutions(rp)

    pending: list[dict] = []
    resolved: list[dict] = []
    for sh, payload in specs.items():
        res = resolutions.get(sh)
        is_resolved = res is not None
        row = {
            "spec_hash":            sh,
            "source_hypothesis_id": payload.get("source_hypothesis_id",
                                                 ""),
            "family_hint":          payload.get("family_hint", ""),
            "persisted_ts":         payload.get("persisted_ts", ""),
            "spec":                 payload.get("spec", {}),
            "resolved":             is_resolved,
            "resolution":           (_dc.asdict(res) if is_resolved
                                       else None),
        }
        if is_resolved:
            resolved.append(row)
        else:
            pending.append(row)

    # Pending: oldest first (FIFO queue)
    pending.sort(key=lambda r: r["persisted_ts"])
    # Resolved: newest first
    resolved.sort(key=lambda r: r["persisted_ts"], reverse=True)

    rows = pending + (resolved if include_resolved else [])
    return {
        "n_pending":  len(pending),
        "n_resolved": len(resolved),
        "rows":       rows,
    }


# ────────────────────────────────────────────────────────────────────
# Public API — resolution + dispatch trigger
# ────────────────────────────────────────────────────────────────────
def resolve_factor_spec(
    spec_hash:         str,
    decision:          str,
    *,
    rationale:         str = "",
    resolved_by:       str = "user",
    specs_path:        Optional[Path] = None,
    resolutions_path:  Optional[Path] = None,
) -> dict:
    """Record the principal's decision on a pending SPEC. On
    decision='approved', synchronously invoke dispatch_factor_spec
    (which runs gates + template + emit).

    Returns:
      {
        spec_hash, decision, resolved_ts, resolved_by,
        dispatch_event_id, verdict_event_id, dispatch_result,
      }
      where dispatch_result is the full dispatcher out dict on
      APPROVED, None otherwise.

    Raises ValueError on:
      - unknown decision string
      - spec_hash not in factor_specs.jsonl
    """
    if decision not in _DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(_DECISIONS)}; got {decision!r}"
        )
    sp = specs_path or _DEFAULT_SPECS_PATH
    rp = resolutions_path or _DEFAULT_RESOLUTIONS_PATH

    specs = _load_specs(sp)
    if spec_hash not in specs:
        raise ValueError(
            f"spec_hash {spec_hash!r} not found in factor_specs.jsonl"
        )

    resolved_ts = _utc_iso()
    dispatch_event_id: Optional[str] = None
    verdict_event_id: Optional[str] = None
    dispatch_result: Optional[dict] = None

    if decision == "approved":
        # Synchronously invoke dispatcher — gates + template + emit.
        # Failure here MUST NOT block resolution row write (the
        # principal made a decision; record it regardless). Dispatch
        # log + event store handle their own persistence guarantees.
        try:
            from engine.agents.strengthener.factor_dispatcher import (
                dispatch_factor_spec,
            )
            payload = specs[spec_hash]
            spec_obj = _payload_to_factor_spec(payload)
            dispatch_result = dispatch_factor_spec(
                spec_obj,
                family_hint   = payload.get("family_hint", "OTHER"),
                spec_approved = True,
            )
            dispatch_event_id = dispatch_result.get("dispatch_event_id")
            verdict_event_id  = dispatch_result.get("verdict_event_id")
        except Exception as exc:
            logger.exception(
                "resolve_factor_spec: dispatch failed for %s",
                spec_hash,
            )

    # Persist resolution (regardless of dispatch outcome)
    _append_jsonl(rp, {
        "spec_hash":           spec_hash,
        "decision":            decision,
        "rationale":           rationale,
        "resolved_ts":         resolved_ts,
        "resolved_by":         resolved_by,
        "dispatch_event_id":   dispatch_event_id,
        "verdict_event_id":    verdict_event_id,
    })

    return {
        "spec_hash":         spec_hash,
        "decision":          decision,
        "resolved_ts":       resolved_ts,
        "resolved_by":       resolved_by,
        "dispatch_event_id": dispatch_event_id,
        "verdict_event_id":  verdict_event_id,
        "dispatch_result":   dispatch_result,
    }


# ────────────────────────────────────────────────────────────────────
# Public API — dispatcher gate helper (replaces caller bool)
# ────────────────────────────────────────────────────────────────────
def is_spec_approved(
    spec_hash:         str,
    *,
    resolutions_path:  Optional[Path] = None,
) -> bool:
    """True iff the latest resolution row for spec_hash is
    decision='approved'. Used by dispatcher gate #5 in
    contexts where the caller doesn't know the approval state
    (e.g. cron resuming a queue)."""
    rp = resolutions_path or _DEFAULT_RESOLUTIONS_PATH
    res = _load_resolutions(rp).get(spec_hash)
    return bool(res and res.decision == "approved")
