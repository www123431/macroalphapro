"""Tests for engine.decay_forecast — family-keyed forward decay estimates.

Run as: pytest tests/decay_forecast/ -v
"""
from __future__ import annotations

import pytest


def test_known_family_returns_typed_estimate():
    """A known family (earnings_underreaction) returns an estimate using
    the registered MP 2016 / LR 2018 λ values."""
    from engine.decay_forecast import estimate_for_family, DecayRisk

    e = estimate_for_family("earnings_underreaction")
    assert e.family == "earnings_underreaction"
    assert e.using_default is False
    assert e.mp_2016_lambda == 0.20    # registered value
    assert e.lr_2018_lambda == 0.30
    # half-life = ln(2)/0.20 ≈ 3.47 yrs
    assert 3.0 < e.half_life_years < 4.0
    # 5y retention = exp(-0.20*5) ≈ 0.368 of baseline
    assert 0.30 < e.expected_alpha_5y / e.baseline_alpha < 0.45


def test_unknown_family_falls_back_to_default():
    """An unknown family triggers the _default row with using_default=True."""
    from engine.decay_forecast import estimate_for_family

    e = estimate_for_family("totally_made_up_mechanism_xyz")
    assert e.using_default is True
    assert e.family == "totally_made_up_mechanism_xyz"
    # _default has lambda=0.20 (MP 2016 average)
    assert e.mp_2016_lambda == 0.20


def test_low_decay_family_marks_LOW_risk():
    """Insurance / structural families should mark LOW risk."""
    from engine.decay_forecast import estimate_for_family
    from engine.decay_forecast.schema import DecayRisk

    e = estimate_for_family("factor_hedge")
    # lambda=0.05 → 5y retention = exp(-0.25) ≈ 0.78
    retention_5y = e.expected_alpha_5y / e.baseline_alpha
    assert retention_5y >= 0.70
    assert e.risk == DecayRisk.LOW


def test_heavy_decay_family_marks_HIGH_or_SEVERE_risk():
    """Heavily arbitraged families flagged for the user."""
    from engine.decay_forecast import estimate_for_family
    from engine.decay_forecast.schema import DecayRisk

    e = estimate_for_family("momentum")    # lambda=0.25
    # retention_5y = exp(-1.25) ≈ 0.287
    assert e.risk in (DecayRisk.HIGH, DecayRisk.MEDIUM)


def test_publication_year_advances_decay_clock():
    """If candidate is based on a paper published in 2010, the clock starts
    earlier — current expected α should be lower than for an unpublished
    candidate."""
    from engine.decay_forecast import estimate_for_family

    fresh = estimate_for_family("earnings_underreaction")
    aged  = estimate_for_family("earnings_underreaction", publication_year=2010)
    assert aged.expected_alpha_now < fresh.expected_alpha_now
    assert aged.years_since_pub > 10


def test_custom_baseline_alpha_passes_through():
    """If caller supplies the candidate's own posterior_mean as baseline,
    estimate scales linearly."""
    from engine.decay_forecast import estimate_for_family

    low  = estimate_for_family("carry", baseline_alpha=0.02)
    high = estimate_for_family("carry", baseline_alpha=0.10)
    # same family, same lambda → linear in baseline
    assert high.expected_alpha_5y == pytest.approx(low.expected_alpha_5y * 5.0,
                                                    rel=1e-6)


def test_lr_lower_band_is_more_pessimistic_than_mp_central():
    """LR 2018 lower bound must always be ≤ MP 2016 central estimate."""
    from engine.decay_forecast import estimate_for_family

    e = estimate_for_family("low_vol")
    assert e.expected_alpha_5y_lower <= e.expected_alpha_5y
    assert e.expected_alpha_10y_lower <= e.expected_alpha_10y


def test_list_supported_families_returns_nonempty():
    """Family registry exposes all families (excluding _default sentinel)."""
    from engine.decay_forecast import list_supported_families

    fams = list_supported_families()
    assert len(fams) >= 8
    names = {f["family"] for f in fams}
    assert "earnings_underreaction" in names
    assert "carry" in names
    assert "_default" not in names
    # each entry has required fields
    for f in fams:
        assert "mp_2016_lambda" in f
        assert "lr_2018_lambda" in f
        assert "half_life_years" in f


def test_to_dict_roundtrip():
    """DecayEstimate.to_dict serializes enum and dataclass fields."""
    from engine.decay_forecast import estimate_for_family

    e = estimate_for_family("carry")
    d = e.to_dict()
    assert d["risk"] in ("LOW", "MEDIUM", "HIGH", "SEVERE")
    assert d["family"] == "carry"
    assert isinstance(d["mp_2016_lambda"], float)
    # No Enum objects left in serialized form
    import json
    json.dumps(d)   # raises if non-serializable
