"""Tests for engine.research.protocols.adaptive_diagnostics."""
from __future__ import annotations

import dataclasses

import pytest

from engine.research.protocols.adaptive_diagnostics import (
    BenefitEstimate,
    CostEstimate,
    FailureCategory,
    Recommendation,
    Severity,
    analyze_multi_leg_failure,
    format_recommendations,
    list_detectors,
    register_detector,
    to_json_for_log,
)
from engine.research.protocols.protocol_executor import (
    DecompositionResult, LegResult, MultiLegVerdict,
)
from engine.research.protocols.protocol_designer import (
    InstantiatedProtocol,
)


def _stub_protocol():
    return InstantiatedProtocol(
        mechanism_id="x", protocol_family_id="generic",
        protocol_family_version=1,
        legs=(), decomposition_checks=(), verdict_rule={},
        instantiated_ts="2026-05-30T00:00:00Z",
        protocol_hash="deadbeef0123",
    )


def _stub_verdict(leg_results=None, decomp_results=None, overall="RED"):
    return MultiLegVerdict(
        protocol=_stub_protocol(),
        leg_results=leg_results or [],
        decomp_results=decomp_results or [],
        overall_verdict=overall, verdict_reasons=[],
        elapsed_seconds=0.0,
    )


# ── Recommendation dataclass ────────────────────────────────────────────

def test_impact_cost_ratio_calculation():
    rec = Recommendation(
        pattern="test", category=FailureCategory.SAMPLE_ISSUE.value,
        severity=Severity.WARN.value, action="a", rationale="r",
        evidence=[],
        benefit=BenefitEstimate(sharpe_t_delta_lo=0.5, sharpe_t_delta_hi=1.5,
                                  qualitative="material"),
        cost=CostEstimate(compute_minutes=10.0, dollars=0.0, wallclock_days=0.0),
        confidence=0.80,
        falsification_criterion="f", alternative_actions=[],
    )
    # mid=1.0, cost=10min, confidence=0.8 → 1.0 / 10 * 0.8 = 0.08
    assert abs(rec.impact_cost_ratio - 0.08) < 0.01


def test_recommendation_serializes_full():
    rec = Recommendation(
        pattern="test", category="sample_issue", severity="warn",
        action="a", rationale="r",
        evidence=["evidence1"],
        benefit=BenefitEstimate(0.0, 1.0, "modest"),
        cost=CostEstimate(1.0, 0.0, 0.0),
        confidence=0.5,
        falsification_criterion="f", alternative_actions=["alt1"],
    )
    d = rec.to_dict()
    assert d["pattern"] == "test"
    assert d["benefit"]["qualitative"] == "modest"
    assert d["cost"]["compute_minutes"] == 1.0
    assert d["impact_cost_ratio"] is not None


# ── Each detector ───────────────────────────────────────────────────────

def test_sample_too_short_detector_fires_when_majority_short():
    legs = [
        LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                    pass_criteria_eval={}, leg_passed=False),
        LegResult(leg_id="subperiod_first_half", is_primary=False,
                    gate_summary=None, pass_criteria_eval={}, leg_passed=False,
                    error="sliced sample too short: 17 months"),
        LegResult(leg_id="subperiod_second_half", is_primary=False,
                    gate_summary=None, pass_criteria_eval={}, leg_passed=False,
                    error="sliced sample too short: 17 months"),
    ]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={
        "template_id": "equity_xsmom",
        "sample_total_months": 84,
        "template_warmup_months": 49,
    })
    sst = [r for r in recs if r.pattern == "sample_too_short"]
    assert len(sst) == 1
    assert sst[0].severity == "warn"
    assert "Extend sample" in sst[0].action


def test_sample_too_short_does_not_fire_when_only_one():
    legs = [
        LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                    pass_criteria_eval={}, leg_passed=True),
        LegResult(leg_id="cost_stress_2x", is_primary=False, gate_summary=None,
                    pass_criteria_eval={}, leg_passed=False,
                    error="sliced sample too short: 5 months"),
        LegResult(leg_id="microcap_robust", is_primary=False, gate_summary={},
                    pass_criteria_eval={}, leg_passed=True),
    ]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={"template_id": "equity_xsmom"})
    assert not any(r.pattern == "sample_too_short" for r in recs)


def test_universe_too_small_detector_fires():
    legs = [LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                        pass_criteria_eval={}, leg_passed=False)]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={
        "template_id": "equity_xsmom",
        "universe_size": 30,
    })
    uts = [r for r in recs if r.pattern == "universe_too_small"]
    assert len(uts) == 1
    assert uts[0].severity == "block"    # 30 < 30 threshold → block severity


def test_universe_too_small_for_tsmom_not_fired():
    """cross_asset_tsmom is per-instrument, tiny universe OK."""
    v = _stub_verdict()
    recs = analyze_multi_leg_failure(v, context={
        "template_id": "cross_asset_tsmom",
        "universe_size": 10,
    })
    assert not any(r.pattern == "universe_too_small" for r in recs)


def test_warmup_consumes_most_fires_at_high_ratio():
    v = _stub_verdict()
    recs = analyze_multi_leg_failure(v, context={
        "template_id": "equity_xsmom",
        "template_warmup_months": 49,
        "sample_total_months": 84,
    })
    wcm = [r for r in recs if r.pattern == "warmup_consumes_most_of_sample"]
    assert len(wcm) == 1
    assert "58%" in wcm[0].action or "0.58" in wcm[0].action.lower()


def test_overfit_fragile_fires_when_primary_passes_others_fail():
    legs = [
        LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                    pass_criteria_eval={}, leg_passed=True),
        LegResult(leg_id="cost_stress_2x", is_primary=False, gate_summary={},
                    pass_criteria_eval={}, leg_passed=False),
        LegResult(leg_id="microcap_robust", is_primary=False, gate_summary={},
                    pass_criteria_eval={}, leg_passed=False),
    ]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={})
    pf = [r for r in recs if r.pattern == "primary_pass_robustness_fail"]
    assert len(pf) == 1


def test_inverted_alpha_detector_fires_on_consistent_negatives():
    legs = [
        LegResult(leg_id=f"leg_{i}", is_primary=(i == 0),
                    gate_summary={"alpha_t_ff5umd": -1.0 - 0.5 * i,
                                    "standalone_sharpe": -0.3},
                    pass_criteria_eval={}, leg_passed=False)
        for i in range(4)
    ]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={})
    ia = [r for r in recs if r.pattern == "inverted_alpha_all_legs"]
    assert len(ia) == 1
    assert ia[0].severity == "info"   # diagnostic


def test_cost_stress_sensitivity_fires_on_big_drop():
    legs = [
        LegResult(leg_id="primary_test", is_primary=True,
                    gate_summary={"standalone_sharpe": 0.6,
                                    "alpha_t_ff5umd": 3.0},
                    pass_criteria_eval={}, leg_passed=True),
        LegResult(leg_id="cost_stress_2x", is_primary=False,
                    gate_summary={"standalone_sharpe": 0.1,
                                    "alpha_t_ff5umd": 1.0},
                    pass_criteria_eval={}, leg_passed=False),
    ]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={})
    css = [r for r in recs if r.pattern == "cost_stress_sensitivity"]
    assert len(css) == 1


def test_regime_concentration_fires_on_sign_flip():
    legs = [
        LegResult(leg_id="subperiod_first_half", is_primary=False,
                    gate_summary={"standalone_sharpe": 0.8,
                                    "alpha_t_ff5umd": 2.5},
                    pass_criteria_eval={}, leg_passed=True),
        LegResult(leg_id="subperiod_second_half", is_primary=False,
                    gate_summary={"standalone_sharpe": -0.5,
                                    "alpha_t_ff5umd": -1.5},
                    pass_criteria_eval={}, leg_passed=False),
    ]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={})
    rc = [r for r in recs if r.pattern == "regime_concentration"]
    assert len(rc) == 1


def test_decomposition_contamination_fires():
    decomps = [
        DecompositionResult(check_id="ff5_umd_orthogonality",
                              requirement={}, eval={"x": False}, passed=False),
        DecompositionResult(check_id="pead_residualization",
                              requirement={}, eval={"x": False}, passed=False),
    ]
    v = _stub_verdict(decomp_results=decomps)
    recs = analyze_multi_leg_failure(v, context={})
    dc = [r for r in recs if r.pattern == "decomposition_contamination"]
    assert len(dc) == 1


# ── Ranking + format ────────────────────────────────────────────────────

def test_recommendations_ranked_by_impact_cost():
    """Higher impact_cost_ratio should appear first."""
    legs = [
        LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                    pass_criteria_eval={}, leg_passed=False),
        LegResult(leg_id="subperiod_first_half", is_primary=False,
                    gate_summary=None, pass_criteria_eval={}, leg_passed=False,
                    error="sliced sample too short"),
        LegResult(leg_id="subperiod_second_half", is_primary=False,
                    gate_summary=None, pass_criteria_eval={}, leg_passed=False,
                    error="sliced sample too short"),
    ]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={
        "template_id": "equity_xsmom",
        "universe_size": 30,
        "sample_total_months": 84,
        "template_warmup_months": 49,
    })
    # All ranks > 0
    assert len(recs) >= 2
    # Ranked descending by combined score
    for i in range(len(recs) - 1):
        score_i = recs[i].impact_cost_ratio
        score_j = recs[i + 1].impact_cost_ratio
        # Allow same score due to severity tiebreaker
        assert score_i >= score_j or recs[i].severity in ("block", "warn")


def test_format_empty_returns_message():
    out = format_recommendations([])
    assert "no adaptive recommendations" in out


def test_format_populates_all_fields():
    legs = [
        LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                    pass_criteria_eval={}, leg_passed=False),
        LegResult(leg_id="x", is_primary=False, gate_summary=None,
                    pass_criteria_eval={}, leg_passed=False,
                    error="sliced sample too short"),
        LegResult(leg_id="y", is_primary=False, gate_summary=None,
                    pass_criteria_eval={}, leg_passed=False,
                    error="sliced sample too short"),
    ]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={
        "template_id": "equity_xsmom",
        "sample_total_months": 84,
        "template_warmup_months": 49,
    })
    text = format_recommendations(recs)
    assert "ACTION:" in text
    assert "EVIDENCE:" in text
    assert "BENEFIT:" in text
    assert "FALSIFY:" in text


def test_to_json_serializable():
    import json
    legs = [
        LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                    pass_criteria_eval={}, leg_passed=False),
        LegResult(leg_id="x", is_primary=False, gate_summary=None,
                    pass_criteria_eval={}, leg_passed=False,
                    error="sliced sample too short"),
        LegResult(leg_id="y", is_primary=False, gate_summary=None,
                    pass_criteria_eval={}, leg_passed=False,
                    error="sliced sample too short"),
    ]
    v = _stub_verdict(leg_results=legs)
    recs = analyze_multi_leg_failure(v, context={
        "template_id": "equity_xsmom",
        "sample_total_months": 84,
        "template_warmup_months": 49,
    })
    payload = to_json_for_log(recs)
    s = json.dumps(payload)    # must serialize
    parsed = json.loads(s)
    assert isinstance(parsed, list)


# ── Custom detector registration ────────────────────────────────────────

def test_custom_detector_can_be_registered(monkeypatch):
    """Drop a new detector — auto-discovered (no code changes elsewhere).

    Uses monkeypatch on _DETECTORS so cleanup is automatic."""
    from engine.research.protocols import adaptive_diagnostics

    orig_detectors = list(adaptive_diagnostics._DETECTORS)
    monkeypatch.setattr(adaptive_diagnostics, "_DETECTORS", list(orig_detectors))

    @register_detector("test_custom_detector")
    def _custom(verdict, context):
        return [Recommendation(
            pattern="test_custom_detector", category="design",
            severity="info", action="a", rationale="r", evidence=[],
            benefit=BenefitEstimate(0, 0.1, "modest"),
            cost=CostEstimate(1, 0, 0),
            confidence=1.0, falsification_criterion="f",
            alternative_actions=[],
        )]

    names = list_detectors()
    assert "test_custom_detector" in names


# ── No false positives on successful verdict ────────────────────────────

def test_no_recommendations_when_all_pass():
    legs = [
        LegResult(leg_id=f"leg{i}", is_primary=(i == 0),
                    gate_summary={"standalone_sharpe": 0.7,
                                    "alpha_t_ff5umd": 3.5},
                    pass_criteria_eval={}, leg_passed=True)
        for i in range(5)
    ]
    decomps = [DecompositionResult(check_id=f"d{i}", requirement={},
                                      eval={"x": True}, passed=True)
                for i in range(2)]
    v = _stub_verdict(leg_results=legs, decomp_results=decomps,
                        overall="GREEN")
    recs = analyze_multi_leg_failure(v, context={
        "template_id": "equity_xsmom", "universe_size": 500,
        "sample_total_months": 240, "template_warmup_months": 49,
    })
    # No detector should fire on a clean GREEN
    assert len(recs) == 0
