"""Tests for engine.research.protocols.protocol_executor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research.protocols import (
    execute_protocol,
    instantiate_protocol,
    load_mechanism,
    MultiLegVerdict,
)
from engine.research.protocols.protocol_executor import (
    _aggregate_verdict,
    _eval_decomposition,
    _eval_pass_criteria,
    DecompositionResult,
    LegResult,
)
from engine.research.protocols.protocol_designer import DecompositionCheck


@pytest.fixture
def synth_prices():
    rng = np.random.RandomState(99)
    n_months, n_tickers = 120, 80
    dates = pd.date_range("2014-01-31", periods=n_months, freq="ME")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    drift = rng.uniform(-0.003, 0.010, n_tickers)
    rets = rng.randn(n_months, n_tickers) * 0.06 + drift
    return pd.DataFrame(
        np.cumprod(1 + rets, axis=0) * 100.0, index=dates, columns=tickers,
    )


@pytest.fixture
def equity_proposal():
    mech = load_mechanism("equity_xsmom_jt")
    return {
        "mechanism_id":      "equity_xsmom_jt",
        "execution_template": mech["execution_template"],
    }


@pytest.fixture
def equity_protocol():
    mech = load_mechanism("equity_xsmom_jt")
    return instantiate_protocol(
        mech,
        proposal_sample_start="2014-01-31",
        proposal_sample_end="2024-01-31",
    )


# ── Pass-criteria evaluation ─────────────────────────────────────────────

def test_eval_pass_criteria_sharpe_t_min_pass():
    criteria = {"sharpe_t_min": 2.0}
    gate = {"alpha_t_ff5umd": 3.5, "n_months": 100, "standalone_sharpe": 0.8}
    out = _eval_pass_criteria(criteria, gate)
    assert out["sharpe_t_min"] is True


def test_eval_pass_criteria_sharpe_t_min_fail():
    criteria = {"sharpe_t_min": 3.0}
    gate = {"alpha_t_ff5umd": 1.5, "n_months": 100, "standalone_sharpe": 0.2}
    out = _eval_pass_criteria(criteria, gate)
    assert out["sharpe_t_min"] is False


def test_eval_pass_criteria_deflated_sr_min():
    out = _eval_pass_criteria(
        {"deflated_sr_min": 0.9},
        {"deflated_sr": 0.95, "n_months": 100, "alpha_t_ff5umd": 4.0,
          "standalone_sharpe": 0.8},
    )
    assert out["deflated_sr_min"] is True


def test_eval_pass_criteria_book_corr_max_abs():
    out = _eval_pass_criteria(
        {"book_corr_max_abs": 0.5},
        {"corr_with_book": 0.3, "n_months": 100, "alpha_t_ff5umd": 4.0,
          "standalone_sharpe": 0.8},
    )
    assert out["book_corr_max_abs"] is True
    out2 = _eval_pass_criteria(
        {"book_corr_max_abs": 0.5},
        {"corr_with_book": 0.7, "n_months": 100, "alpha_t_ff5umd": 4.0,
          "standalone_sharpe": 0.8},
    )
    assert out2["book_corr_max_abs"] is False


def test_eval_pass_criteria_sign_required_positive():
    out_pos = _eval_pass_criteria(
        {"sign_required": "positive"},
        {"standalone_sharpe": 0.5, "n_months": 100, "alpha_t_ff5umd": 3.0},
    )
    assert out_pos["sign_required"] is True
    out_neg = _eval_pass_criteria(
        {"sign_required": "positive"},
        {"standalone_sharpe": -0.5, "n_months": 100, "alpha_t_ff5umd": -3.0},
    )
    assert out_neg["sign_required"] is False


def test_eval_pass_criteria_retain_ratio_vs_primary():
    criteria = {"retain_ratio_vs_primary": 0.5}
    primary = {"standalone_sharpe": 0.8}
    # Sharpe at 0.5 → ratio 0.625 > 0.5 OK
    out_pass = _eval_pass_criteria(criteria,
                                      {"standalone_sharpe": 0.5, "n_months": 100,
                                        "alpha_t_ff5umd": 2.0},
                                      primary_gate=primary)
    assert out_pass["retain_ratio_vs_primary"] is True
    # Sharpe at 0.2 → ratio 0.25 < 0.5 FAIL
    out_fail = _eval_pass_criteria(criteria,
                                      {"standalone_sharpe": 0.2, "n_months": 100,
                                        "alpha_t_ff5umd": 2.0},
                                      primary_gate=primary)
    assert out_fail["retain_ratio_vs_primary"] is False


def test_eval_pass_criteria_sign_consistency():
    criteria = {"sign_consistent_with_primary": True}
    primary = {"standalone_sharpe": 0.7}
    out_consistent = _eval_pass_criteria(criteria,
                                            {"standalone_sharpe": 0.3, "n_months": 100,
                                              "alpha_t_ff5umd": 2.0},
                                            primary_gate=primary)
    assert out_consistent["sign_consistent_with_primary"] is True
    out_inconsistent = _eval_pass_criteria(criteria,
                                              {"standalone_sharpe": -0.3, "n_months": 100,
                                                "alpha_t_ff5umd": -2.0},
                                              primary_gate=primary)
    assert out_inconsistent["sign_consistent_with_primary"] is False


# ── Decomposition evaluation ────────────────────────────────────────────

def test_eval_decomposition_ff5_orth_pass():
    check = DecompositionCheck(
        id="ff5_umd_orthogonality", description="",
        requirement={"ff5_umd_alpha_t_min_abs": 2.0,
                      "sign_must_match_primary": True},
    )
    primary = {"alpha_t_ff5umd": 3.5, "standalone_sharpe": 0.8}
    res = _eval_decomposition(check, primary)
    assert res.passed is True


def test_eval_decomposition_ff5_orth_fail():
    check = DecompositionCheck(
        id="ff5_umd_orthogonality", description="",
        requirement={"ff5_umd_alpha_t_min_abs": 2.0},
    )
    primary = {"alpha_t_ff5umd": 1.0, "standalone_sharpe": 0.4}
    res = _eval_decomposition(check, primary)
    assert res.passed is False


# ── Verdict aggregation ─────────────────────────────────────────────────

def test_aggregate_verdict_green():
    legs = [
        LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                    pass_criteria_eval={}, leg_passed=True),
    ] + [
        LegResult(leg_id=f"r{i}", is_primary=False, gate_summary={},
                    pass_criteria_eval={}, leg_passed=True)
        for i in range(4)
    ]
    decomp = [DecompositionResult(check_id="d1", requirement={}, eval={},
                                     passed=True)]
    rule = {
        "GREEN_requires_all_of": [
            {"primary_test_pass": True},
            {"n_robustness_pass_geq": 3},
            {"all_decomposition_pass": True},
        ],
        "YELLOW_requires_all_of": [
            {"primary_test_pass": True},
            {"n_robustness_pass_geq": 2},
        ],
        "RED_otherwise": True,
    }
    verdict, _ = _aggregate_verdict(legs, decomp, rule)
    assert verdict == "GREEN"


def test_aggregate_verdict_yellow():
    legs = [
        LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                    pass_criteria_eval={}, leg_passed=True),
        LegResult(leg_id="r1", is_primary=False, gate_summary={},
                    pass_criteria_eval={}, leg_passed=True),
        LegResult(leg_id="r2", is_primary=False, gate_summary={},
                    pass_criteria_eval={}, leg_passed=True),
        LegResult(leg_id="r3", is_primary=False, gate_summary={},
                    pass_criteria_eval={}, leg_passed=False),
        LegResult(leg_id="r4", is_primary=False, gate_summary={},
                    pass_criteria_eval={}, leg_passed=False),
    ]
    decomp = [DecompositionResult(check_id="d1", requirement={}, eval={},
                                     passed=True)]
    rule = {
        "GREEN_requires_all_of": [
            {"primary_test_pass": True},
            {"n_robustness_pass_geq": 3},
        ],
        "YELLOW_requires_all_of": [
            {"primary_test_pass": True},
            {"n_robustness_pass_geq": 2},
        ],
        "RED_otherwise": True,
    }
    verdict, _ = _aggregate_verdict(legs, decomp, rule)
    assert verdict == "YELLOW"


def test_aggregate_verdict_red():
    legs = [
        LegResult(leg_id="primary_test", is_primary=True, gate_summary={},
                    pass_criteria_eval={}, leg_passed=False),
    ]
    decomp = []
    rule = {
        "GREEN_requires_all_of": [{"primary_test_pass": True}],
        "YELLOW_requires_all_of": [{"primary_test_pass": True}],
        "RED_otherwise": True,
    }
    verdict, _ = _aggregate_verdict(legs, decomp, rule)
    assert verdict == "RED"


# ── End-to-end protocol execution ────────────────────────────────────────

def test_execute_protocol_end_to_end(equity_protocol, equity_proposal, synth_prices):
    """Full pipeline: 5-leg equity_factor_standard_v1 protocol on synth data."""
    result = execute_protocol(
        equity_protocol, equity_proposal,
        data_kwargs={"price_panel": synth_prices},
        pead_control=False,
    )
    assert isinstance(result, MultiLegVerdict)
    assert result.overall_verdict in ("GREEN", "YELLOW", "RED")
    assert len(result.leg_results) == 5    # primary + 4 robustness


def test_execute_protocol_all_legs_attempted(equity_protocol, equity_proposal, synth_prices):
    """Even on noisy synth data, ALL legs should at least attempt execution
    (no early-exit). Some may error out from short windows."""
    result = execute_protocol(
        equity_protocol, equity_proposal,
        data_kwargs={"price_panel": synth_prices}, pead_control=False,
    )
    leg_ids = {r.leg_id for r in result.leg_results}
    assert "primary_test" in leg_ids
    assert "cost_stress_2x" in leg_ids
    assert "microcap_robust" in leg_ids


def test_execute_protocol_handles_dsl_failure(equity_protocol, synth_prices):
    """Proposal without execution_template → DSL fails on every leg.
    Executor should aggregate to RED, not crash."""
    bad_proposal = {"mechanism_id": "ghost_v1"}    # no execution_template
    result = execute_protocol(
        equity_protocol, bad_proposal,
        data_kwargs={"price_panel": synth_prices}, pead_control=False,
    )
    # All legs error → no leg_passed → primary fails → RED
    assert result.overall_verdict == "RED"
    assert all(r.leg_passed is False for r in result.leg_results)


def test_execute_protocol_verdict_serializable(equity_protocol, equity_proposal, synth_prices):
    import json
    result = execute_protocol(
        equity_protocol, equity_proposal,
        data_kwargs={"price_panel": synth_prices}, pead_control=False,
    )
    d = result.to_dict()
    # Round-trips through JSON
    s = json.dumps(d, default=str)
    parsed = json.loads(s)
    assert parsed["overall_verdict"] in ("GREEN", "YELLOW", "RED")
