"""Tests for engine.capacity — family-keyed capacity sub-MVP."""
from __future__ import annotations

import pytest


def test_known_family_returns_typed_estimate():
    from engine.capacity import estimate_for_family
    from engine.capacity.schema import CapacityClass

    e = estimate_for_family("carry")
    assert e.family == "carry"
    assert e.using_default is False
    assert e.capacity_class == CapacityClass.VERY_HIGH
    assert e.estimated_capacity_usd >= 1_000_000_000   # carry is at least $1B


def test_unknown_family_falls_back_to_default():
    from engine.capacity import estimate_for_family

    e = estimate_for_family("totally_made_up_xyz")
    assert e.using_default is True
    assert e.family == "totally_made_up_xyz"


def test_carry_higher_capacity_than_equity_singlename():
    """Sanity: cross-asset carry should have far higher capacity than
    single-name equity event-driven."""
    from engine.capacity import estimate_for_family

    carry = estimate_for_family("carry")
    pead  = estimate_for_family("earnings_underreaction")
    assert carry.estimated_capacity_usd > pead.estimated_capacity_usd * 5


def test_thresholds_ordered_correctly():
    """comfortable_aum < estimated_capacity (you don't want to be at the
    half-Sharpe-haircut AUM); minimum_aum < comfortable_aum."""
    from engine.capacity import estimate_for_family

    for fam in ["carry", "momentum", "earnings_underreaction", "factor_hedge"]:
        e = estimate_for_family(fam)
        assert e.minimum_aum_usd < e.comfortable_aum_usd
        assert e.comfortable_aum_usd < e.estimated_capacity_usd


def test_all_classes_represented_in_registry():
    """Registry should cover at least 4 of 5 capacity classes (sanity)."""
    from engine.capacity import list_supported_families
    from engine.capacity.schema import CapacityClass

    fams = list_supported_families()
    classes = {f["capacity_class"] for f in fams}
    assert CapacityClass.VERY_HIGH.value in classes
    assert CapacityClass.HIGH.value in classes
    assert CapacityClass.MEDIUM.value in classes


def test_list_supported_families_returns_nonempty():
    from engine.capacity import list_supported_families

    fams = list_supported_families()
    assert len(fams) >= 8
    for f in fams:
        assert "family" in f
        assert "capacity_class" in f
        assert "estimated_capacity_usd" in f


def test_to_dict_serialization():
    from engine.capacity import estimate_for_family
    import json

    e = estimate_for_family("tsmom")
    d = e.to_dict()
    assert d["capacity_class"] in ("VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW")
    json.dumps(d)   # raises if non-serializable
