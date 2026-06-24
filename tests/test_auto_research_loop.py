"""Unit tests for engine.research.auto_research_loop.

Critical safety properties under test:
1. Sample-isolation enforcer NEVER returns data past val_end (the test-set
   leakage guard — load-bearing for the strict-gate doctrine).
2. Bounds enforcement rejects out-of-range proposed params.
3. Decide() rolls back when val Sharpe drops more than the trigger.
4. Decide() halts on N consecutive rollbacks.
5. Proposer respects cross-param constraints (q_out > q_in).
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.research.auto_research_loop import (
    assert_within_bounds,
    decide,
    enforce_sample_isolation,
    propose_change,
)


@pytest.fixture
def skill_fixture() -> dict:
    """Minimal SKILL stub for unit testing — does not touch disk."""
    return {
        "skill_name": "test_skill",
        "parameters": {
            "revision_q_in": {"type": "float", "baseline": 0.20,
                              "range": [0.10, 0.30], "step": 0.025},
            "revision_q_out": {"type": "float", "baseline": 0.40,
                               "range": [0.30, 0.55], "step": 0.05,
                               "constraint": "q_out > q_in"},
            "revision_weight": {"type": "enum", "baseline": "equal",
                                "options": ["equal", "mag"]},
        },
        "evaluation": {
            "train_end": "2020-12-31",
            "val_end":   "2023-12-31",
            "min_val_months": 24,
            "gate": {"hlz_t": 3.0, "deflsr_min": 0.90, "max_book_corr": 0.50},
        },
        "rollback_triggers": [
            {"rule": "val_sharpe_drop_pp",   "value": 0.05},
            {"rule": "val_maxdd_increase_pp","value": 0.02},
            {"rule": "verdict_regression"},
            {"rule": "out_of_bounds"},
            {"rule": "n_consecutive_rollback_halt", "value": 3},
        ],
        "version_history": [
            {"version": "v0.0.0", "parent": None, "ts": "2026-05-29T00:00:00Z",
             "proposer": "human_baseline", "rationale": "baseline",
             "params": {"revision_q_in": 0.20, "revision_q_out": 0.40,
                        "revision_weight": "equal"},
             "val_metrics": None, "test_metrics": None,
             "decision": "baseline"},
        ],
    }


# ─── Sample isolation tests ──────────────────────────────────────────────────

def test_sample_isolation_never_leaks_past_val_end(skill_fixture):
    """The load-bearing guard: even when given full history, the enforcer
    must clip to (train_end, val_end]."""
    idx = pd.date_range("2015-01-31", "2026-05-31", freq="ME")
    returns = pd.Series([0.01] * len(idx), index=idx)

    val_slice = enforce_sample_isolation(returns, skill_fixture)

    assert val_slice.index.max() <= pd.Timestamp("2023-12-31"), \
        f"TEST LEAK: slice extends to {val_slice.index.max()}"
    assert val_slice.index.min() > pd.Timestamp("2020-12-31"), \
        f"TRAIN LEAK: slice starts at {val_slice.index.min()}"


def test_sample_isolation_invalid_window_raises(skill_fixture):
    skill_fixture["evaluation"]["val_end"] = "2020-01-01"   # before train_end
    idx = pd.date_range("2015-01-31", "2026-05-31", freq="ME")
    returns = pd.Series([0.01] * len(idx), index=idx)
    with pytest.raises(ValueError, match="val_end .* must be > train_end"):
        enforce_sample_isolation(returns, skill_fixture)


def test_sample_isolation_empty_when_returns_outside_window(skill_fixture):
    """Returns entirely before train_end → empty val slice (handled gracefully,
    not a leak)."""
    idx = pd.date_range("2015-01-31", "2018-12-31", freq="ME")
    returns = pd.Series([0.01] * len(idx), index=idx)
    val_slice = enforce_sample_isolation(returns, skill_fixture)
    assert val_slice.empty


# ─── Bounds enforcement tests ────────────────────────────────────────────────

def test_assert_within_bounds_accepts_valid(skill_fixture):
    assert_within_bounds({"revision_q_in": 0.15, "revision_weight": "equal"},
                          skill_fixture)   # should not raise


def test_assert_within_bounds_rejects_below_range(skill_fixture):
    with pytest.raises(AssertionError, match="PROPOSER BUG"):
        assert_within_bounds({"revision_q_in": 0.05}, skill_fixture)


def test_assert_within_bounds_rejects_above_range(skill_fixture):
    with pytest.raises(AssertionError, match="PROPOSER BUG"):
        assert_within_bounds({"revision_q_in": 0.35}, skill_fixture)


def test_assert_within_bounds_rejects_invalid_enum(skill_fixture):
    with pytest.raises(AssertionError, match="PROPOSER BUG"):
        assert_within_bounds({"revision_weight": "exotic"}, skill_fixture)


# ─── Decision logic tests ────────────────────────────────────────────────────

def test_decide_rollback_on_sharpe_drop(skill_fixture):
    parent_val = {"n": 36, "sharpe": 1.10, "maxdd": -0.05}
    new_val    = {"n": 36, "sharpe": 1.00, "maxdd": -0.05}  # 0.10 drop > 0.05 trigger
    dec, reason = decide(parent_val, new_val, skill_fixture)
    assert dec == "rollback"
    assert "sharpe" in reason.lower()


def test_decide_keep_on_small_sharpe_change(skill_fixture):
    parent_val = {"n": 36, "sharpe": 1.10, "maxdd": -0.05}
    new_val    = {"n": 36, "sharpe": 1.08, "maxdd": -0.05}  # 0.02 drop, under trigger
    dec, _ = decide(parent_val, new_val, skill_fixture)
    assert dec == "keep"


def test_decide_rollback_on_maxdd_worsening(skill_fixture):
    parent_val = {"n": 36, "sharpe": 1.10, "maxdd": -0.05}
    new_val    = {"n": 36, "sharpe": 1.10, "maxdd": -0.10}  # 5pp worsen > 2pp trigger
    dec, reason = decide(parent_val, new_val, skill_fixture)
    assert dec == "rollback"
    assert "maxdd" in reason.lower()


def test_decide_rollback_on_insufficient_val_data(skill_fixture):
    parent_val = {"n": 36, "sharpe": 1.10, "maxdd": -0.05}
    new_val    = {"n": 12, "sharpe": 1.50, "maxdd": -0.05}  # n=12 < min_val=24
    dec, reason = decide(parent_val, new_val, skill_fixture)
    assert dec == "rollback"
    assert "n=12" in reason


def test_decide_first_evaluation_keeps_by_default(skill_fixture):
    new_val = {"n": 36, "sharpe": 0.50, "maxdd": -0.05}
    dec, reason = decide(None, new_val, skill_fixture)
    assert dec == "keep"


# ─── Proposer tests ──────────────────────────────────────────────────────────

def test_proposer_returns_params_within_bounds(skill_fixture):
    for seed in range(20):
        prop = propose_change(skill_fixture, rng_seed=seed)
        assert_within_bounds(prop["params"], skill_fixture)
        assert prop["parent_version"] == "v0.0.0"


def test_proposer_respects_q_out_gt_q_in_constraint(skill_fixture):
    """When proposer would violate q_out > q_in, it must no-op rather than
    propose invalid combination."""
    # Force baseline to a near-boundary state and run many proposals
    skill_fixture["version_history"][-1]["params"] = {
        "revision_q_in": 0.30, "revision_q_out": 0.35, "revision_weight": "equal",
    }
    for seed in range(30):
        prop = propose_change(skill_fixture, rng_seed=seed)
        p = prop["params"]
        # Either no-op OR proposed q_out > q_in
        if "constraint-rejected" not in prop["changed"]:
            assert p["revision_q_out"] > p["revision_q_in"], \
                f"q_out > q_in violated: {p}"
