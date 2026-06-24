"""Phase 5 cognitive adaptation tests — A (fidelity_level) + G (negative_evidence)."""
from __future__ import annotations

import pytest
import yaml

from engine.research.protocols import instantiate_protocol, load_mechanism
from engine.research.protocols.protocol_designer import (
    _apply_fidelity_to_legs,
    _FIDELITY_BAR_ADJUSTMENT,
    ResolvedLeg,
)
from engine.research.hygiene_tools import h2_cousin_check_multilevel


# ── Phase 5 A: fidelity_level → bar adjustment ──────────────────────────

def test_fidelity_adjustment_dict_complete():
    """All 3 fidelity levels must have bar adjustment specified."""
    assert "literal" in _FIDELITY_BAR_ADJUSTMENT
    assert "adapted" in _FIDELITY_BAR_ADJUSTMENT
    assert "inspired" in _FIDELITY_BAR_ADJUSTMENT


def test_fidelity_literal_no_bump():
    """literal → unchanged bars."""
    bump, oos = _FIDELITY_BAR_ADJUSTMENT["literal"]
    assert bump == 0.0
    assert oos == 0


def test_fidelity_adapted_adds_05_to_sharpe_bar():
    """adapted → +0.5 to sharpe_t_min."""
    legs = [ResolvedLeg(
        id="primary_test", description="", sample_start="2000-01-01",
        sample_end="2024-12-31", binding={}, is_primary=True,
        pass_criteria={"sharpe_t_min": 3.0, "deflated_sr_min": 0.9},
    )]
    out = _apply_fidelity_to_legs(legs, "adapted")
    assert out[0].pass_criteria["sharpe_t_min"] == 3.5
    # Other criteria unchanged
    assert out[0].pass_criteria["deflated_sr_min"] == 0.9


def test_fidelity_inspired_adds_10_to_sharpe_bar():
    legs = [ResolvedLeg(
        id="primary_test", description="", sample_start="2000-01-01",
        sample_end="2024-12-31", binding={}, is_primary=True,
        pass_criteria={"sharpe_t_min": 3.0},
    )]
    out = _apply_fidelity_to_legs(legs, "inspired")
    assert out[0].pass_criteria["sharpe_t_min"] == 4.0


def test_fidelity_unknown_level_no_bump():
    legs = [ResolvedLeg(
        id="primary_test", description="", sample_start="2000-01-01",
        sample_end="2024-12-31", binding={}, is_primary=True,
        pass_criteria={"sharpe_t_min": 3.0},
    )]
    out = _apply_fidelity_to_legs(legs, "unknown_level")
    assert out[0].pass_criteria["sharpe_t_min"] == 3.0


def test_instantiate_protocol_with_adapted_fidelity():
    """A mechanism with fidelity_level=adapted gets bumped bars."""
    mech = load_mechanism("equity_xsmom_jt")
    mech_adapted = dict(mech)
    mech_adapted["fidelity_level"] = "adapted"
    proto_lit = instantiate_protocol(
        mech, proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    proto_adapted = instantiate_protocol(
        mech_adapted, proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    primary_lit = next(leg for leg in proto_lit.legs if leg.is_primary)
    primary_adapted = next(leg for leg in proto_adapted.legs if leg.is_primary)
    assert primary_adapted.pass_criteria["sharpe_t_min"] == (
        primary_lit.pass_criteria["sharpe_t_min"] + 0.5
    )


def test_instantiate_protocol_notes_records_fidelity():
    """Protocol notes field records the fidelity adjustment for audit."""
    mech = load_mechanism("equity_xsmom_jt")
    mech_inspired = dict(mech)
    mech_inspired["fidelity_level"] = "inspired"
    proto = instantiate_protocol(
        mech_inspired, proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    assert "inspired" in proto.notes
    assert "+1.0" in proto.notes


def test_fidelity_hash_differs_across_levels():
    """Different fidelity → different bars → different protocol_hash."""
    mech = load_mechanism("equity_xsmom_jt")
    proto_lit = instantiate_protocol(
        dict(mech, fidelity_level="literal"),
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    proto_adapted = instantiate_protocol(
        dict(mech, fidelity_level="adapted"),
        proposal_sample_start="1965-01-01",
        proposal_sample_end="2024-12-31",
    )
    assert proto_lit.protocol_hash != proto_adapted.protocol_hash


# ── Phase 5 G: negative_evidence cousin-check anchor ────────────────────

def test_h2_hard_rejects_same_family_as_negative_evidence(tmp_path, monkeypatch):
    """If a library entry has purpose=negative_evidence and matches family,
    H2 cousin check should hard-reject."""
    # Stand up isolated library with target + negative_evidence anchor
    lib_dir = tmp_path / "mechanism_library"
    lib_dir.mkdir()
    for entry_id, data in [
        ("target_mech", {
            "id": "target_mech", "family": "test_family", "parent_family": "test_parent",
            "audit_signature": "human-confirmed", "purpose": "candidate",
            "required_data": ["a"], "mechanism_economics": "x" * 200,
            "currently_unexplored_in_our_book": True,
            "post_pub_decay": {"post_2020_replications": []},
        }),
        ("dead_cousin", {
            "id": "dead_cousin", "family": "test_family", "parent_family": "test_parent",
            "audit_signature": "human-confirmed", "purpose": "negative_evidence",
            "required_data": ["a"], "mechanism_economics": "y" * 200,
            "status_in_our_book": "UNTESTED",
        }),
    ]:
        (lib_dir / f"{entry_id}.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
    monkeypatch.setattr(
        "engine.research.hygiene_tools.LIBRARY_DIR", lib_dir
    )
    result = h2_cousin_check_multilevel("target_mech")
    assert result.success
    assert result.payload["verdict"] == "hard_reject"
    assert any("negative_evidence" in r for r in result.payload["hard_reject_reasons"])


def test_h2_no_hard_reject_without_negative_evidence(tmp_path, monkeypatch):
    """Same-family cousin_anchor with UNTESTED status doesn't hard-reject."""
    lib_dir = tmp_path / "mechanism_library"
    lib_dir.mkdir()
    for entry_id, data in [
        ("target_mech", {
            "id": "target_mech", "family": "test_family", "parent_family": "test_parent",
            "audit_signature": "human-confirmed", "purpose": "candidate",
            "required_data": ["a"], "mechanism_economics": "x" * 200,
            "currently_unexplored_in_our_book": True,
        }),
        ("benign_cousin", {
            "id": "benign_cousin", "family": "test_family", "parent_family": "test_parent",
            "audit_signature": "human-confirmed", "purpose": "cousin_anchor",
            "required_data": ["a"], "mechanism_economics": "z" * 200,
            "status_in_our_book": "UNTESTED",
        }),
    ]:
        (lib_dir / f"{entry_id}.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
    monkeypatch.setattr(
        "engine.research.hygiene_tools.LIBRARY_DIR", lib_dir
    )
    result = h2_cousin_check_multilevel("target_mech")
    assert result.success
    # cousin_anchor + UNTESTED = no hard reject
    # (status not in {RED, DEPLOYED} AND purpose != negative_evidence)
    assert result.payload["verdict"] != "hard_reject"
