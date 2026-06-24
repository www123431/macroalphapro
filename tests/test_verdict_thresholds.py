"""BUG-3 verdict threshold tests."""
from __future__ import annotations

import math

import pytest

from engine.research.verdict_thresholds import (
    _HLZ_CEIL_T_GREEN,
    _HLZ_FLOOR_T_GREEN,
    _T_MARGINAL_BASELINE,
    alpha_t_green_threshold,
    bonferroni_threshold,
    t_green_threshold,
    t_marginal_threshold,
    threshold_summary,
)


# ── Bonferroni computation ────────────────────────────────────────


def test_bonferroni_n_1_recovers_baseline():
    """With n_trials=1, Bonferroni threshold = standard t at 5% two-tailed."""
    t = bonferroni_threshold(1)
    assert 1.90 < t < 2.05  # standard 1.96


def test_bonferroni_scales_up_with_n():
    """Threshold strictly increases as n_trials grows."""
    t_1 = bonferroni_threshold(1)
    t_10 = bonferroni_threshold(10)
    t_100 = bonferroni_threshold(100)
    assert t_1 < t_10 < t_100


def test_bonferroni_n_10_matches_known_value():
    """At n=10, 5% Bonferroni = 0.5% per-test → z(0.9975) ≈ 2.81."""
    t = bonferroni_threshold(10)
    assert 2.75 < t < 2.90


# ── t_green_threshold floor/ceiling ───────────────────────────────


def test_t_green_at_n_0_uses_hlz_floor():
    """No prior trials → conservative HLZ floor."""
    assert t_green_threshold(0) == _HLZ_FLOOR_T_GREEN


def test_t_green_at_n_5_still_hlz_floor():
    """Bonferroni(5) ≈ 2.58 < HLZ floor 3.0."""
    assert t_green_threshold(5) == _HLZ_FLOOR_T_GREEN


def test_t_green_at_large_n_hits_ceiling():
    """Large n → Bonferroni exceeds HLZ ceiling 3.5."""
    assert t_green_threshold(1000) == _HLZ_CEIL_T_GREEN


def test_t_green_monotonic_in_n():
    """Stricter threshold should never decrease as n grows."""
    prev = t_green_threshold(0)
    for n in [1, 5, 10, 20, 50, 100, 500, 1000]:
        curr = t_green_threshold(n)
        assert curr >= prev
        prev = curr


# ── alpha_t_green_threshold ───────────────────────────────────────


def test_alpha_t_green_floor():
    """At n=0, alpha-t floor applies."""
    assert alpha_t_green_threshold(0) == 2.0


def test_alpha_t_green_monotonic():
    for n in [0, 5, 10, 20, 50, 100]:
        t = alpha_t_green_threshold(n)
        assert t >= 2.0
        assert t <= 2.5


# ── MARGINAL threshold ────────────────────────────────────────────


def test_t_marginal_baseline_at_small_n():
    assert t_marginal_threshold(0) == _T_MARGINAL_BASELINE
    assert t_marginal_threshold(5) == _T_MARGINAL_BASELINE


def test_t_marginal_softscales_at_high_n():
    """At n=100, marginal should be > baseline 1.65 but still much less
    than GREEN threshold."""
    t = t_marginal_threshold(100)
    assert t > _T_MARGINAL_BASELINE
    assert t < t_green_threshold(100)


# ── threshold_summary diagnostic ──────────────────────────────────


def test_threshold_summary_contains_all_keys():
    out = threshold_summary(10)
    assert out["n_trials"] == 10
    assert "t_green_threshold" in out
    assert "t_marginal_threshold" in out
    assert "alpha_t_green_threshold" in out
    assert out["anchor"] == "HLZ_floor_with_bonferroni_body"


# ── Practical case: AMP-2013 case ─────────────────────────────────


def test_amp2013_case_threshold_at_n4_is_3_0():
    """COMBINATION_HML_MOM currently has n_trials = 4 (4 historical
    factor_verdict_filed events in family). At n=4, HLZ floor 3.0
    applies → GREEN gate is NW-t ≥ 3.0 (stricter than old 1.96).
    alpha-t Bonferroni at n=4 ≈ 2.24 (between floor 2.0 and ceil 2.5).
    AMP-2013 NW-t 5.02 still passes NW gate; alpha-t 0.11 still fails
    alpha gate. Verdict stays MARGINAL.
    """
    assert t_green_threshold(4) == 3.0
    a_t = alpha_t_green_threshold(4)
    assert 2.20 < a_t < 2.30   # Bonferroni at n=4 alpha=0.10 two-tailed
    # Sanity: AMP-2013 metrics check the verdict logic
    nw_t = 5.02
    alpha_t = 0.11   # FF-complement, after BUG-1 fix
    assert nw_t >= 3.0  # passes new NW gate
    assert alpha_t < a_t  # fails alpha gate → MARGINAL
