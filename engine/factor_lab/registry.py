"""
engine/factor_lab/registry.py — Factor candidate registry helpers.

Spec: docs/spec_factor_lab.md §2.3 (data persistence) + §2.2 (state machine)
Boundary invariant: zero LLM imports — pure DB query / mutation.

Public API
----------
  list_active_candidates()    — non-legacy, non-infrastructure rows in state machine
  list_legacy_specs()         — pre-LAB hypothesis-test specs (factor_kind=NULL)
  list_infrastructure_specs() — Lab self + future infrastructure
  get_candidate(spec_id)      — single row by SpecRegistry.id
  transition_state(spec_id, new_state, reason, *, n_trials_added=0)
                              — atomic state change + amendment_log entry
  state_counts()              — pipeline status board: {state: n_rows}

Schema reference (engine/db_models.py:952-989, after P-LAB migration):
  SpecRegistry.lab_state    : VARCHAR(32) — FactorState enum value or NULL
  SpecRegistry.factor_kind  : VARCHAR(32) — 'production_swap' | 'overlay' |
                              'shadow' | 'infrastructure_spec' | NULL (legacy)
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Optional

from engine.factor_lab.types import FactorState, assert_legal_transition


# Factor kinds eligible for the state machine pipeline (excludes legacy + infra)
_ACTIVE_FACTOR_KINDS = {"production_swap", "overlay", "shadow"}


def list_active_candidates() -> list[dict[str, Any]]:
    """Return rows that are part of the lab state machine.

    Excludes:
      - Legacy specs (factor_kind=NULL): pre-LAB hypothesis tests, displayed
        separately in the falsification chain timeline.
      - Infrastructure specs (factor_kind='infrastructure_spec'): the lab's
        own spec and similar housekeeping rows.
    """
    from engine.memory import SessionFactory, SpecRegistry

    out: list[dict[str, Any]] = []
    with SessionFactory() as s:
        rows = (
            s.query(SpecRegistry)
            .filter(SpecRegistry.factor_kind.in_(_ACTIVE_FACTOR_KINDS))
            .order_by(SpecRegistry.registered_at.desc())
            .all()
        )
        for r in rows:
            out.append(_row_to_dict(r))
    return out


def list_legacy_specs() -> list[dict[str, Any]]:
    """Return pre-LAB specs (factor_kind=NULL).

    These are the 8 historic hypothesis tests + supporting specs registered
    before P-LAB introduction (2026-05-08). Their verdicts live in
    docs/decisions/*.md and are displayed read-only in the falsification
    chain timeline.
    """
    from engine.memory import SessionFactory, SpecRegistry

    out: list[dict[str, Any]] = []
    with SessionFactory() as s:
        rows = (
            s.query(SpecRegistry)
            .filter(SpecRegistry.factor_kind.is_(None))
            .order_by(SpecRegistry.registered_at.asc())
            .all()
        )
        for r in rows:
            out.append(_row_to_dict(r))
    return out


def list_infrastructure_specs() -> list[dict[str, Any]]:
    """Return infrastructure_spec rows (e.g. spec_factor_lab.md itself).

    These are tracked for HARKing R1 silent-edit protection but exempt
    from EFFECTIVE_N_TRIALS accounting (per spec §6.1).
    """
    from engine.memory import SessionFactory, SpecRegistry

    out: list[dict[str, Any]] = []
    with SessionFactory() as s:
        rows = (
            s.query(SpecRegistry)
            .filter(SpecRegistry.factor_kind == "infrastructure_spec")
            .order_by(SpecRegistry.registered_at.asc())
            .all()
        )
        for r in rows:
            out.append(_row_to_dict(r))
    return out


def get_candidate(spec_id: int) -> Optional[dict[str, Any]]:
    """Single row by id. Returns None if not found."""
    from engine.memory import SessionFactory, SpecRegistry

    with SessionFactory() as s:
        r = s.query(SpecRegistry).filter_by(id=spec_id).first()
        return _row_to_dict(r) if r else None


def state_counts() -> dict[str, int]:
    """Pipeline counts grouped by FactorState value (active candidates only).

    Returns dict keyed by state name; states with 0 rows are included as 0.
    Useful for the Section 1 status board.
    """
    counts: dict[str, int] = {state.value: 0 for state in FactorState}
    counts["LEGACY"]         = 0   # pre-LAB hypothesis tests
    counts["INFRASTRUCTURE"] = 0   # lab spec self + similar

    from engine.memory import SessionFactory, SpecRegistry

    with SessionFactory() as s:
        for r in s.query(SpecRegistry).all():
            kind = r.factor_kind or ""
            if kind == "infrastructure_spec":
                counts["INFRASTRUCTURE"] += 1
            elif kind == "":
                counts["LEGACY"] += 1
            elif kind in _ACTIVE_FACTOR_KINDS:
                state = (r.lab_state or "DRAFT")
                counts[state] = counts.get(state, 0) + 1
    return counts


def transition_state(
    spec_id:         int,
    new_state:       FactorState,
    reason:          str,
    *,
    n_trials_added:  int = 0,
    actor:           str = "factor_lab.registry",
) -> dict[str, Any]:
    """Atomic state machine transition + amendment_log entry.

    Validates legality via types.assert_legal_transition; raises
    IllegalTransition (ValueError subclass) on violation. The amendment_log
    entry serves HARKing R1 silent-edit protection.

    Args:
        spec_id: SpecRegistry.id of the candidate row.
        new_state: target FactorState.
        reason: human-readable explanation written to amendment_log.
        n_trials_added: typically 0 for state transitions (not research
            amendments). Set >0 only when transition itself is a substantive
            hypothesis change (rare; usually use amend_spec instead).
        actor: caller identification for amendment_log.

    Returns:
        Updated row dict.

    Raises:
        IllegalTransition: source → target violates spec §2.2.
        LookupError: spec_id not found.
        ValueError: row's factor_kind makes it ineligible for state machine.
    """
    from engine.memory import SessionFactory, SpecRegistry

    with SessionFactory() as s:
        r = s.query(SpecRegistry).filter_by(id=spec_id).first()
        if r is None:
            raise LookupError(f"SpecRegistry id={spec_id} not found")
        if (r.factor_kind or "") not in _ACTIVE_FACTOR_KINDS:
            raise ValueError(
                f"spec_id={spec_id} factor_kind={r.factor_kind!r} is not "
                f"in active set {_ACTIVE_FACTOR_KINDS}; state machine "
                f"transitions only apply to active candidates."
            )

        # Validate legality
        src = FactorState(r.lab_state or FactorState.DRAFT.value)
        assert_legal_transition(src, new_state)

        # Append amendment_log entry
        try:
            log = json.loads(r.amendment_log or "[]")
        except Exception:
            log = []
        log.append({
            "at":             datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "kind":           "lab_state_transition",
            "actor":          actor,
            "from_state":     src.value,
            "to_state":       new_state.value,
            "reason":         reason,
            "n_trials_added": int(n_trials_added),
            # State transitions don't change the spec markdown content, so
            # new_hash == current spec hash. Required by detect_harking R1
            # which compares ledger[-1]["new_hash"] vs current_hash to detect
            # silent edits between amendments.
            "prev_hash":      r.current_hash,
            "new_hash":       r.current_hash,
        })

        # Apply
        r.lab_state             = new_state.value
        r.amendment_log         = json.dumps(log, ensure_ascii=False)
        r.last_validated_at     = datetime.datetime.utcnow()
        if n_trials_added:
            r.n_trials_contributed = int(r.n_trials_contributed or 0) + int(n_trials_added)
        s.commit()
        return _row_to_dict(r)


# ── Internals ────────────────────────────────────────────────────────────────

def _row_to_dict(r) -> dict[str, Any]:
    """SpecRegistry ORM row → plain dict for UI consumption."""
    return {
        "id":                   r.id,
        "spec_path":            r.spec_path,
        "current_hash":         r.current_hash,
        "git_blob_hash":        r.git_blob_hash,
        "registered_at":        r.registered_at,
        "last_validated_at":    r.last_validated_at,
        "lab_state":            r.lab_state,
        "factor_kind":          r.factor_kind,
        "status":               r.status,
        "retro_registered":     bool(r.retro_registered),
        "n_trials_contributed": int(r.n_trials_contributed or 0),
        "amendment_count":      _amendment_count(r.amendment_log),
    }


def _amendment_count(amendment_log_json: str | None) -> int:
    if not amendment_log_json:
        return 0
    try:
        log = json.loads(amendment_log_json)
        return len(log) if isinstance(log, list) else 0
    except Exception:
        return 0
