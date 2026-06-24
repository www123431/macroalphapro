"""Tests for engine.research.candidate_pipeline (Phase A enforced sequence)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research import candidate_pipeline as cp


@pytest.fixture
def synth_factor_panel(monkeypatch):
    """Set up synthetic factors + sleeves + regime for fast offline tests."""
    np.random.seed(7)
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    factors = pd.DataFrame({
        "MKT": np.random.randn(120) * 0.04,
        "SMB": np.random.randn(120) * 0.03,
        "MOM": np.random.randn(120) * 0.04,
    }, index=idx)
    regime = pd.Series(
        ["CALM"] * 40 + ["NORMAL"] * 40 + ["STRESS"] * 40,
        index=idx,
    )
    from engine.risk import barra_lite as bl
    monkeypatch.setattr(bl, "build_factor_returns", lambda phase=1: factors)

    # Mock book sleeves used by H9
    book_eq = (0.5 * factors["MOM"] + 0.01
               + np.random.RandomState(11).randn(120) * 0.005)
    book_cy = (0.1 * factors["MKT"]
               + np.random.RandomState(13).randn(120) * 0.003)
    book_ts = (0.3 * factors["MOM"]
               + np.random.RandomState(17).randn(120) * 0.01)
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_equity_book", lambda: book_eq
    )
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_carry_book", lambda: book_cy
    )
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_tsmom_book", lambda: book_ts
    )
    # Mock the hedge sleeves used by correlation_matrix + factor_budget steps
    crisis = pd.Series(np.random.RandomState(19).randn(120) * 0.02, index=idx)
    momh = pd.Series(np.random.RandomState(23).randn(120) * 0.015, index=idx)
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_crisis_hedge_book",
        lambda: crisis,
    )
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_mom_hedge_book",
        lambda: momh,
    )
    # Mock regime classifier
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_vix_regime_monthly",
        lambda: regime,
    )
    return factors, idx, regime


# ── PROMOTE_TO_GATE path ────────────────────────────────────────────────

def test_pipeline_alpha_seeker_runs_full_sequence(synth_factor_panel):
    """Strong alpha + role=alpha_seeker → goes through full sequence;
    final decision could be PROMOTE / BORDERLINE depending on number of
    WARN-class steps. We assert NOT HARD_REJECT."""
    factors, idx, _ = synth_factor_panel
    np.random.seed(101)
    candidate = pd.Series(0.018 + np.random.randn(120) * 0.003, index=idx)
    report = cp.run_candidate_pipeline(
        candidate, proposal_name="good_alpha",
        proposed_role="alpha_seeker", phase=1,
    )
    assert report.final_decision != "HARD_REJECT"
    assert report.role_used == "alpha_seeker"
    h10_step = next(s for s in report.step_results
                      if s.step_name == "H10_evaluate_candidate")
    assert h10_step.status == "PASS"
    # All new steps should be present in step_results
    step_names = {s.step_name for s in report.step_results}
    expected = {"H10_evaluate_candidate", "data_quality",
                "H2_cousin_check", "H6_post_pub_evidence",
                "H7_kill_this_proposal", "graveyard_check",
                "cost_model_check", "regime_stratified_BARRA",
                "factor_budget_delta", "multi_aum_cost",
                "sub_period_robustness", "correlation_matrix",
                "devils_advocate"}
    assert expected.issubset(step_names), \
        f"missing steps: {expected - step_names}"


# ── HARD_REJECT paths ───────────────────────────────────────────────────

def test_pipeline_insurance_blocked_by_regime_stratified(synth_factor_panel):
    """Insurance candidate whose STRESS alpha < NORMAL alpha → HARD_REJECT."""
    factors, idx, regime = synth_factor_panel
    # Construct insurance candidate: negative MOM beta + better in NORMAL
    # than STRESS (mimicking mom_hedge real-world finding)
    candidate = pd.Series(0.0, index=idx)
    # CALM: small negative drift
    candidate.loc[idx[:40]] = (-0.5 * factors["MOM"].iloc[:40] - 0.002
                                  + np.random.RandomState(11).randn(40) * 0.005)
    # NORMAL: medium negative drift
    candidate.loc[idx[40:80]] = (-0.5 * factors["MOM"].iloc[40:80] - 0.003
                                     + np.random.RandomState(13).randn(40) * 0.005)
    # STRESS: BIG negative drift (worse than NORMAL — this is the failure)
    candidate.loc[idx[80:]] = (-0.5 * factors["MOM"].iloc[80:] - 0.015
                                   + np.random.RandomState(17).randn(40) * 0.005)
    report = cp.run_candidate_pipeline(
        candidate, proposal_name="bad_insurance",
        proposed_role="insurance", phase=1,
    )
    assert report.final_decision == "HARD_REJECT"
    regime_step = next(s for s in report.step_results
                          if s.step_name == "regime_stratified_BARRA")
    assert regime_step.status == "FAIL"
    assert "HYPOTHESIS REJECTED" in regime_step.verdict


def test_pipeline_h10_failing_blocks_subsequent_steps(synth_factor_panel):
    """If H10 fails, downstream steps shouldn't run."""
    factors, idx, _ = synth_factor_panel
    np.random.seed(103)
    candidate = pd.Series(np.random.randn(120) * 0.01, index=idx)
    report = cp.run_candidate_pipeline(
        candidate, proposal_name="noise",
        proposed_role="alpha_seeker", phase=1,
    )
    assert report.final_decision == "HARD_REJECT"
    step_names_after_h10 = [s.step_name for s in report.step_results
                                if s.step_name != "H10_evaluate_candidate"]
    assert len(step_names_after_h10) == 0


# ── New P0/P1/P2 step-specific tests ─────────────────────────────────────

def test_data_quality_detects_outliers():
    """Series with extreme outliers → WARN."""
    np.random.seed(5)
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    vals = np.random.randn(120) * 0.02
    vals[10] = 1.0  # extreme outlier
    vals[20] = -0.9
    vals[30] = 0.8
    s = pd.Series(vals, index=idx)
    step = cp._run_data_quality_check(s)
    assert step.status == "WARN"
    assert "outliers" in step.verdict.lower() or "extreme" in step.verdict.lower()


def test_data_quality_passes_clean_series():
    np.random.seed(7)
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    s = pd.Series(np.random.randn(120) * 0.02, index=idx)
    step = cp._run_data_quality_check(s)
    assert step.status == "PASS"


def test_data_quality_flags_zero_variance():
    s = pd.Series([0.01] * 30,
                       index=pd.date_range("2020-01-31", periods=30, freq="ME"))
    step = cp._run_data_quality_check(s)
    assert step.status == "FAIL"


def test_h7_skips_when_no_proposal_dict():
    """Pipeline calls H7 with empty dict → SKIP, not crash."""
    step = cp._run_h7({})
    assert step.status == "SKIP"


def test_meta_decision_promotes_when_clean(synth_factor_panel):
    """No FAIL + h10_accept=True + few warns → PROMOTE."""
    steps = [
        cp.StepResult("step_a", "PASS", {}, "ok"),
        cp.StepResult("step_b", "PASS", {}, "ok"),
        cp.StepResult("data_quality", "PASS", {}, "ok"),
    ]
    decision, _ = cp._compute_meta_decision(steps, h10_accept=True)
    assert decision == "PROMOTE_TO_GATE"


def test_meta_decision_borderline_review_on_critical_warn():
    """1 critical WARN → BORDERLINE_REVIEW."""
    steps = [
        cp.StepResult("H10_evaluate_candidate", "PASS", {}, "ok"),
        cp.StepResult("regime_stratified_BARRA", "WARN", {}, "regime warn"),
        cp.StepResult("step_c", "PASS", {}, "ok"),
    ]
    decision, _ = cp._compute_meta_decision(steps, h10_accept=True)
    assert decision == "BORDERLINE_REVIEW"


def test_meta_decision_soft_rejects_on_many_critical_warns():
    """3+ critical WARN → SOFT_REJECT."""
    steps = [
        cp.StepResult("H10_evaluate_candidate", "PASS", {}, "ok"),
        cp.StepResult("regime_stratified_BARRA", "WARN", {}, "regime warn"),
        cp.StepResult("factor_budget_delta", "WARN", {}, "piles on"),
        cp.StepResult("devils_advocate", "WARN", {}, "DA concerns"),
        cp.StepResult("data_quality", "WARN", {}, "outliers"),
    ]
    decision, _ = cp._compute_meta_decision(steps, h10_accept=True)
    assert decision == "SOFT_REJECT"


def test_meta_decision_hard_rejects_on_any_fail():
    steps = [
        cp.StepResult("regime_stratified_BARRA", "FAIL", {}, "bad"),
    ]
    decision, _ = cp._compute_meta_decision(steps, h10_accept=True)
    assert decision == "HARD_REJECT"


def test_meta_decision_soft_rejects_when_h10_not_accept():
    steps = [
        cp.StepResult("H10_evaluate_candidate", "WARN", {}, "borderline"),
    ]
    decision, _ = cp._compute_meta_decision(steps, h10_accept=False)
    assert decision == "SOFT_REJECT"


# ── ROLE INFERENCE ──────────────────────────────────────────────────────

def test_pipeline_role_inferred_when_not_provided(synth_factor_panel):
    factors, idx, _ = synth_factor_panel
    np.random.seed(107)
    candidate = pd.Series(0.018 + np.random.randn(120) * 0.003, index=idx)
    report = cp.run_candidate_pipeline(
        candidate, proposal_name="auto_role", phase=1,
    )
    assert report.role_was_inferred is True
    assert report.role_used is not None


# ── SKIPPED steps ───────────────────────────────────────────────────────

def test_pipeline_h2_h6_skipped_when_no_mechanism_id(synth_factor_panel):
    """Without mechanism_id, H2 and H6 should SKIP, not FAIL."""
    factors, idx, _ = synth_factor_panel
    np.random.seed(109)
    candidate = pd.Series(0.018 + np.random.randn(120) * 0.003, index=idx)
    report = cp.run_candidate_pipeline(
        candidate, proposal_name="no_id", proposed_role="alpha_seeker",
        phase=1,
    )
    h2_step = next(s for s in report.step_results
                      if s.step_name == "H2_cousin_check")
    h6_step = next(s for s in report.step_results
                      if s.step_name == "H6_post_pub_evidence")
    assert h2_step.status == "SKIP"
    assert h6_step.status == "SKIP"


# ── Devil's Advocate WARN (always present, placeholder) ─────────────────

def test_pipeline_devils_advocate_always_warns(synth_factor_panel):
    factors, idx, _ = synth_factor_panel
    np.random.seed(113)
    candidate = pd.Series(0.018 + np.random.randn(120) * 0.003, index=idx)
    report = cp.run_candidate_pipeline(
        candidate, proposal_name="da_test",
        proposed_role="alpha_seeker", phase=1,
    )
    da_step = next(s for s in report.step_results
                      if s.step_name == "devils_advocate")
    # Placeholder always WARN until LLM persona wired
    assert da_step.status == "WARN"


# ── Report serializable ─────────────────────────────────────────────────

def test_pipeline_report_to_dict(synth_factor_panel):
    factors, idx, _ = synth_factor_panel
    np.random.seed(119)
    candidate = pd.Series(0.018 + np.random.randn(120) * 0.003, index=idx)
    report = cp.run_candidate_pipeline(
        candidate, proposal_name="ser",
        proposed_role="alpha_seeker", phase=1,
    )
    d = report.to_dict()
    assert "proposal_name" in d
    assert "role_used" in d
    assert "step_results" in d
    assert "final_decision" in d
    # step_results should be list of dicts
    assert isinstance(d["step_results"], list)
    for s in d["step_results"]:
        assert "step_name" in s
        assert "status" in s
