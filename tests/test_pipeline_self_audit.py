"""Tests for engine.research.pipeline_self_audit (Phase 1 P1b)."""
from __future__ import annotations

import pytest

from engine.research import pipeline_self_audit as psa


# ── KNOWN_BASELINES structure validation ────────────────────────────────

def test_known_baselines_have_required_fields():
    """Each baseline must define returns_cache, proposal_name, role,
    expected_decision_in, expected_relation_in."""
    required = {
        "returns_cache", "proposal_name", "proposed_role",
        "expected_decision_in", "expected_relation_in",
    }
    for b in psa.KNOWN_BASELINES:
        missing = required - set(b.keys())
        assert not missing, f"baseline {b.get('proposal_name')} missing {missing}"


def test_known_baselines_at_least_three():
    """We expect to baseline-test at least 3 known sleeves."""
    assert len(psa.KNOWN_BASELINES) >= 3


def test_known_baselines_expected_decision_values_valid():
    """Expected decisions must be valid pipeline outcomes."""
    valid_decisions = {
        "PROMOTE_TO_GATE", "PROMOTE_AS_REPLACEMENT",
        "BORDERLINE_REVIEW", "SOFT_REJECT", "HARD_REJECT",
        "ROUTE_TO_HUMAN", "ROUTE_TO_REGIME_BACKTEST", "UNKNOWN_ROLE",
    }
    for b in psa.KNOWN_BASELINES:
        for d in b["expected_decision_in"]:
            assert d in valid_decisions, \
                f"baseline {b['proposal_name']} expects unknown decision {d}"


def test_known_baselines_expected_relation_values_valid():
    valid_relations = {"REPLACEMENT", "ADDITION", "UNKNOWN"}
    for b in psa.KNOWN_BASELINES:
        for r in b["expected_relation_in"]:
            assert r in valid_relations


# ── _check_baseline cache-missing graceful ──────────────────────────────

def test_check_baseline_skips_missing_cache():
    """Baseline with missing cache should SKIP, not crash."""
    fake_baseline = {
        "returns_cache":  "data/cache/nonexistent_xyz_123.parquet",
        "proposal_name":  "fake",
        "proposed_role":  "alpha_seeker",
        "expected_decision_in": {"PROMOTE_TO_GATE"},
        "expected_relation_in": {"ADDITION"},
    }
    result = psa._check_baseline(fake_baseline)
    assert result["status"] == "SKIP"
    assert "missing" in result["reason"].lower()


# ── run_self_audit summary structure ────────────────────────────────────

def test_run_self_audit_returns_summary_dict():
    """Run on real cached baselines; summary should have all required fields."""
    summary = psa.run_self_audit(phase=1)   # phase=1 is fastest
    assert "audit_date" in summary
    assert "n_total" in summary
    assert "n_pass" in summary
    assert "n_fail" in summary
    assert "n_error" in summary
    assert "n_skip" in summary
    assert "all_pass" in summary
    assert "results" in summary
    assert summary["n_total"] == len(psa.KNOWN_BASELINES)
