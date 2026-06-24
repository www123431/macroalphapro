"""Tests for engine.research.protocols.protocol_designer."""
from __future__ import annotations

import yaml as _yaml

import pytest

from engine.research.protocols import protocol_designer as PD


@pytest.fixture
def equity_xsmom_jt_mechanism():
    return PD.load_mechanism("equity_xsmom_jt")


@pytest.fixture
def low_vol_bab_mechanism():
    return PD.load_mechanism("low_vol_bab")


def test_list_protocol_families_includes_equity_and_generic():
    families = PD.list_protocol_families()
    assert "equity_factor_standard_v1" in families
    assert "generic_v1" in families


def test_load_equity_family_required_fields():
    fam = PD.load_protocol_family("equity_factor_standard_v1")
    assert fam["protocol_family_id"] == "equity_factor_standard"
    assert fam["version"] == 1
    assert any(leg["id"] == "primary_test" for leg in fam["legs"])
    assert "verdict_rule" in fam


def test_select_family_equity_mechanism(equity_xsmom_jt_mechanism):
    """A momentum mechanism should map to equity_factor_standard_v1."""
    fam_id = PD.select_family_for_mechanism(equity_xsmom_jt_mechanism)
    assert fam_id == "equity_factor_standard_v1"


def test_select_family_unknown_falls_back_to_generic():
    fake_mech = {"family": "alien_factor", "parent_family": "unknown_parent"}
    fam_id = PD.select_family_for_mechanism(fake_mech)
    assert fam_id == "generic_v1"


def test_select_family_preferred_overrides(equity_xsmom_jt_mechanism):
    fam_id = PD.select_family_for_mechanism(
        equity_xsmom_jt_mechanism, preferred_family="generic_v1"
    )
    assert fam_id == "generic_v1"


# ── instantiate_protocol returns frozen + hashed ─────────────────────────

def test_instantiate_protocol_basic(equity_xsmom_jt_mechanism):
    proto = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    assert isinstance(proto, PD.InstantiatedProtocol)
    assert proto.mechanism_id == "equity_xsmom_jt"
    assert proto.protocol_family_id == "equity_factor_standard"
    assert len(proto.legs) == 5    # primary + 4 robustness
    assert any(leg.is_primary for leg in proto.legs)
    assert proto.protocol_hash != ""


def test_instantiate_protocol_is_frozen(equity_xsmom_jt_mechanism):
    import dataclasses
    proto = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        proto.legs = ()    # type: ignore


def test_instantiate_protocol_hash_stable(equity_xsmom_jt_mechanism):
    """Same inputs → same protocol_hash (across re-instantiations)."""
    p1 = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    p2 = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    assert p1.protocol_hash == p2.protocol_hash


def test_instantiate_protocol_hash_changes_with_sample(equity_xsmom_jt_mechanism):
    """Different sample window → potentially different hash if sample
    influences leg resolution."""
    p1 = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    p2 = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="1980-01-01",
        proposal_sample_end="2024-12-31",
    )
    # Different sample range → different first/second-half dates → different hash
    assert p1.protocol_hash != p2.protocol_hash


def test_primary_leg_resolves_full_sample(equity_xsmom_jt_mechanism):
    proto = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    primary = next(leg for leg in proto.legs if leg.is_primary)
    assert primary.sample_start == "1965-01-01"
    assert primary.sample_end == "2024-12-31"


def test_split_first_half_resolves(equity_xsmom_jt_mechanism):
    """Post Phase 6c warmup fix: split happens on EFFECTIVE range, not raw.
    equity_xsmom warmup ~49 months (12 lookback + 1 lag + 36 vol_target).
    20-year sample → ~16yr effective → midpoint ~8yr in.
    """
    import datetime
    proto = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="2000-01-01",
        proposal_sample_end="2020-01-01",
    )
    fh = next(leg for leg in proto.legs if leg.id == "subperiod_first_half")
    # Effective start = 2000-01-01 + ~49 months ≈ early 2004
    eff_start = datetime.date.fromisoformat(fh.sample_start)
    assert datetime.date(2003, 12, 1) <= eff_start <= datetime.date(2004, 6, 1)
    # End = midpoint between effective_start and 2020-01-01 ≈ 2012
    mid_date = datetime.date.fromisoformat(fh.sample_end)
    assert datetime.date(2011, 1, 1) <= mid_date <= datetime.date(2013, 1, 1)


def test_cost_stress_binding_override(equity_xsmom_jt_mechanism):
    proto = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="2000-01-01",
        proposal_sample_end="2024-12-31",
    )
    cs = next(leg for leg in proto.legs if leg.id == "cost_stress_2x")
    assert cs.binding["cost_bps_per_side"] == 24.0
    # Other binding fields inherited from canonical
    assert cs.binding["lookback_months"] == 12


def test_microcap_override_doesnt_affect_other_legs(equity_xsmom_jt_mechanism):
    proto = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="2000-01-01",
        proposal_sample_end="2024-12-31",
    )
    primary = next(leg for leg in proto.legs if leg.is_primary)
    micro = next(leg for leg in proto.legs if leg.id == "microcap_robust")
    assert primary.binding["microcap_price_threshold"] == 5.0
    assert micro.binding["microcap_price_threshold"] == 10.0


def test_decomposition_checks_present(equity_xsmom_jt_mechanism):
    proto = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    ids = {d.id for d in proto.decomposition_checks}
    assert "ff5_umd_orthogonality" in ids
    assert "pead_residualization" in ids


def test_low_vol_bab_routes_to_equity_family(low_vol_bab_mechanism):
    """low_vol_bab has family=quality which is in equity_factor_standard_v1.applies_to."""
    fam_id = PD.select_family_for_mechanism(low_vol_bab_mechanism)
    assert fam_id == "equity_factor_standard_v1"


def test_generic_fallback_for_unknown_family():
    fake_mech = {
        "id": "alien_v1",
        "family": "nonexistent",
        "parent_family": "nonexistent",
        "execution_template": {"binding": {}},
    }
    proto = PD.instantiate_protocol(
        fake_mech,
        proposal_sample_start="2000-01-01",
        proposal_sample_end="2024-12-31",
    )
    assert proto.protocol_family_id == "generic"


def test_yaml_serialization_roundtrip(equity_xsmom_jt_mechanism):
    proto = PD.instantiate_protocol(
        equity_xsmom_jt_mechanism,
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    yaml_str = proto.to_yaml()
    assert "protocol_hash" in yaml_str
    parsed = _yaml.safe_load(yaml_str)
    assert parsed["protocol_family_id"] == "equity_factor_standard"
    assert len(parsed["legs"]) == 5
