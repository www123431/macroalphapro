"""Tests for engine.research.forward_decay_prediction (Phase 1 P1a)."""
from __future__ import annotations

import math
from datetime import date

import pytest

from engine.research import forward_decay_prediction as fdp


# ── _decay_at_year ──────────────────────────────────────────────────────

def test_decay_at_zero_is_baseline():
    assert fdp._decay_at_year(0.05, 0.20, 0) == pytest.approx(0.05)


def test_decay_at_t_years():
    # baseline=0.05, λ=0.20, t=5 → 0.05 × exp(-1.0) ≈ 0.01839
    result = fdp._decay_at_year(0.05, 0.20, 5)
    expected = 0.05 * math.exp(-1.0)
    assert result == pytest.approx(expected, abs=1e-6)


def test_decay_at_t_negative_baseline():
    """Insurance candidates with negative drift: decay model applies same."""
    r = fdp._decay_at_year(-0.02, 0.05, 10)
    assert r < 0
    assert abs(r) < 0.02


# ── _decay_curve ────────────────────────────────────────────────────────

def test_decay_curve_decreasing():
    curve = fdp._decay_curve(0.05, 0.20, max_years=5)
    values = [curve[t] for t in range(6)]
    assert values[0] > values[1] > values[2] > values[3] > values[4] > values[5]


def test_decay_curve_has_correct_year_keys():
    curve = fdp._decay_curve(0.05, 0.20, max_years=10)
    assert set(curve.keys()) == set(range(11))


# ── predict_decay ───────────────────────────────────────────────────────

def test_predict_real_library_entry_post_earnings_drift():
    """post_earnings_drift is in library with publication 1989."""
    p = fdp.predict_decay("post_earnings_drift")
    assert p.mechanism_id == "post_earnings_drift"
    assert p.publication_year == 1989
    # 37+ years since pub → expected alpha now near zero
    assert abs(p.expected_alpha_now) < 0.01
    assert p.half_life_years > 0
    assert p.recommended_review_date


def test_predict_real_library_entry_crisis_hedge():
    """crisis_hedge_tlt_gld has cross_asset_hedge family (low decay)."""
    p = fdp.predict_decay("crisis_hedge_tlt_gld")
    assert p.family == "cross_asset_hedge"
    # Low decay rate → still significant alpha remaining
    assert abs(p.expected_alpha_now) > 0.001
    # 8.7-year half-life
    assert 5 < p.half_life_years < 15


def test_predict_unknown_mechanism_raises():
    with pytest.raises(FileNotFoundError):
        fdp.predict_decay("nonexistent_mechanism_xyz")


def test_predict_with_explicit_baseline():
    """Caller-supplied baseline overrides library value."""
    p = fdp.predict_decay("post_earnings_drift", baseline_alpha=0.10)
    assert p.baseline_alpha == pytest.approx(0.10)
    # With 10% baseline + 37 yrs at λ=0.20, current ~10% × exp(-7.4) ≈ very small
    assert abs(p.expected_alpha_now) < 0.001


def test_predict_to_dict_serializable():
    import json
    p = fdp.predict_decay("post_earnings_drift")
    d = p.to_dict()
    assert "mechanism_id" in d
    assert "expected_alpha_now" in d
    assert "half_life_years" in d
    # Should serialize
    json.dumps(d, default=str)


def test_predict_confidence_band_present():
    p = fdp.predict_decay("post_earnings_drift")
    assert "mp_2016_main" in p.confidence_band
    assert "lr_2018_lower" in p.confidence_band


# ── predict_all_audited ─────────────────────────────────────────────────

def test_predict_all_audited_returns_list():
    predictions = fdp.predict_all_audited()
    assert isinstance(predictions, list)
    # We have at least 3 audited entries (D_PEAD, carry, TSMOM)
    assert len(predictions) >= 3


def test_predict_all_audited_includes_known_audited():
    predictions = fdp.predict_all_audited()
    ids = {p.mechanism_id for p in predictions}
    assert "post_earnings_drift" in ids


# ── Family decay parameter mapping ──────────────────────────────────────

def test_family_decay_params_has_default():
    assert "_default" in fdp.FAMILY_DECAY_PARAMS
    assert fdp.FAMILY_DECAY_PARAMS["_default"]["lambda"] > 0


def test_factor_hedge_has_low_decay():
    """Insurance / hedge families should have lower λ than alpha families."""
    factor_hedge_lambda = fdp.FAMILY_DECAY_PARAMS["factor_hedge"]["lambda"]
    momentum_lambda = fdp.FAMILY_DECAY_PARAMS["momentum"]["lambda"]
    assert factor_hedge_lambda < momentum_lambda


def test_pure_equity_families_have_higher_decay():
    """Pure equity / cross-section factors should decay faster than
    cross-asset / structural."""
    momentum = fdp.FAMILY_DECAY_PARAMS["momentum"]["lambda"]
    carry = fdp.FAMILY_DECAY_PARAMS["carry"]["lambda"]
    assert momentum > carry
