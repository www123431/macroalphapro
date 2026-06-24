"""
tests/test_factor_lab_registry.py — Registry / state machine integration tests.

Covers engine/factor_lab/registry.py against an isolated test DB.
Conventions: each test creates / mutates rows by spec_path so they don't
interfere with the session-scoped baseline DB.
"""
from __future__ import annotations

import json

import pytest

from engine.factor_lab import (
    FactorState,
    IllegalTransition,
    list_active_candidates,
    list_infrastructure_specs,
    list_legacy_specs,
    state_counts,
    transition_state,
)


def _create_candidate(spec_path: str, factor_kind: str = "production_swap",
                      lab_state: str | None = None) -> int:
    """Insert a SpecRegistry row directly for testing (bypasses register_spec
    so we can control factor_kind / lab_state cleanly)."""
    import datetime
    from engine.memory import SessionFactory, SpecRegistry

    with SessionFactory() as s:
        row = SpecRegistry(
            spec_path            = spec_path,
            git_blob_hash        = "0" * 40,
            current_hash         = "0" * 40,
            registered_at        = datetime.datetime.utcnow(),
            amendment_log        = "[]",
            status               = "active",
            retro_registered     = False,
            n_trials_contributed = 1,
            factor_kind          = factor_kind,
            lab_state            = lab_state,
        )
        s.add(row)
        s.commit()
        return row.id


class TestStateCounts:
    def test_state_counts_returns_full_dict(self):
        counts = state_counts()
        # All FactorState values present
        for state in FactorState:
            assert state.value in counts
        # Plus the two non-state buckets
        assert "LEGACY"         in counts
        assert "INFRASTRUCTURE" in counts

    def test_state_counts_sum_increases_after_insert(self):
        before = state_counts()
        sid = _create_candidate(
            spec_path="docs/spec_test_state_counts_increment.md",
            factor_kind="production_swap",
            lab_state="PROPOSED",
        )
        after = state_counts()
        assert after["PROPOSED"] == before["PROPOSED"] + 1


class TestListFunctions:
    def test_active_candidate_appears_in_active_list(self):
        sid = _create_candidate(
            spec_path="docs/spec_test_list_active.md",
            factor_kind="overlay",
            lab_state="REGISTERED",
        )
        rows = list_active_candidates()
        ids = {r["id"] for r in rows}
        assert sid in ids

    def test_legacy_spec_excluded_from_active_list(self):
        sid = _create_candidate(
            spec_path="docs/spec_test_legacy_exclusion.md",
            factor_kind=None,  # NULL → legacy
            lab_state=None,
        )
        active_ids = {r["id"] for r in list_active_candidates()}
        legacy_ids = {r["id"] for r in list_legacy_specs()}
        assert sid not in active_ids
        assert sid in legacy_ids

    def test_infrastructure_excluded_from_active_list(self):
        sid = _create_candidate(
            spec_path="docs/spec_test_infra_exclusion.md",
            factor_kind="infrastructure_spec",
            lab_state=None,
        )
        active_ids = {r["id"] for r in list_active_candidates()}
        infra_ids  = {r["id"] for r in list_infrastructure_specs()}
        assert sid not in active_ids
        assert sid in infra_ids


class TestTransitionState:
    def test_legal_proposed_to_registered(self):
        sid = _create_candidate(
            spec_path="docs/spec_test_legal_transition.md",
            factor_kind="production_swap",
            lab_state="PROPOSED",
        )
        result = transition_state(sid, FactorState.REGISTERED,
                                  reason="power check passed")
        assert result["lab_state"] == "REGISTERED"
        assert result["amendment_count"] == 1

    def test_legal_registered_to_testing(self):
        sid = _create_candidate(
            spec_path="docs/spec_test_registered_to_testing.md",
            factor_kind="production_swap",
            lab_state="REGISTERED",
        )
        transition_state(sid, FactorState.TESTING, reason="user click")
        # Verify stored state
        from engine.factor_lab import get_candidate
        row = get_candidate(sid)
        assert row["lab_state"] == "TESTING"

    def test_illegal_blocked_to_registered_raises(self):
        sid = _create_candidate(
            spec_path="docs/spec_test_blocked_terminal.md",
            factor_kind="production_swap",
            lab_state="BLOCKED_UNDERPOWERED",
        )
        with pytest.raises(IllegalTransition):
            transition_state(sid, FactorState.REGISTERED, reason="trying to bypass")

    def test_illegal_pass_to_testing_raises(self):
        sid = _create_candidate(
            spec_path="docs/spec_test_pass_terminal.md",
            factor_kind="production_swap",
            lab_state="PASS",
        )
        with pytest.raises(IllegalTransition):
            transition_state(sid, FactorState.TESTING, reason="re-test")

    def test_legacy_spec_rejects_transition(self):
        """factor_kind=NULL means not in state machine — must raise ValueError."""
        sid = _create_candidate(
            spec_path="docs/spec_test_legacy_reject_transition.md",
            factor_kind=None,
            lab_state=None,
        )
        with pytest.raises(ValueError, match="not in active set"):
            transition_state(sid, FactorState.PROPOSED,
                             reason="trying to add legacy to state machine")

    def test_infrastructure_spec_rejects_transition(self):
        """infrastructure_spec rows are tracked but not in state machine."""
        sid = _create_candidate(
            spec_path="docs/spec_test_infra_reject_transition.md",
            factor_kind="infrastructure_spec",
            lab_state=None,
        )
        with pytest.raises(ValueError, match="not in active set"):
            transition_state(sid, FactorState.PROPOSED,
                             reason="should be rejected")

    def test_missing_spec_id_raises_lookup(self):
        with pytest.raises(LookupError):
            transition_state(spec_id=999_999, new_state=FactorState.PROPOSED,
                             reason="non-existent")

    def test_amendment_log_records_transition(self):
        sid = _create_candidate(
            spec_path="docs/spec_test_amendment_log_records.md",
            factor_kind="production_swap",
            lab_state="DRAFT",
        )
        transition_state(sid, FactorState.PROPOSED, reason="spec drafted via P3.5")

        from engine.memory import SessionFactory, SpecRegistry
        with SessionFactory() as s:
            row = s.query(SpecRegistry).filter_by(id=sid).first()
            log = json.loads(row.amendment_log)
            assert len(log) == 1
            entry = log[0]
            assert entry["kind"]       == "lab_state_transition"
            assert entry["from_state"] == "DRAFT"
            assert entry["to_state"]   == "PROPOSED"
            assert "spec drafted via P3.5" in entry["reason"]


class TestRowDictSerialization:
    def test_active_candidate_dict_has_required_fields(self):
        sid = _create_candidate(
            spec_path="docs/spec_test_dict_fields.md",
            factor_kind="overlay",
            lab_state="REGISTERED",
        )
        from engine.factor_lab import get_candidate
        row = get_candidate(sid)
        for key in ("id", "spec_path", "current_hash", "lab_state",
                    "factor_kind", "amendment_count", "n_trials_contributed"):
            assert key in row, f"missing field: {key}"
        assert row["factor_kind"] == "overlay"
        assert row["lab_state"]   == "REGISTERED"

    def test_get_candidate_missing_returns_none(self):
        from engine.factor_lab import get_candidate
        assert get_candidate(999_999) is None
