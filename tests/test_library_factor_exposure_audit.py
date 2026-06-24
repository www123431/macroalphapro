"""Tests for engine.research.library_factor_exposure_audit."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.research import library_factor_exposure_audit as lfe


# -- _check_one ------------------------------------------------------------

def test_check_missing_block():
    res = lfe._check_one(Path("/tmp/x.yaml"), {"id": "x"})
    assert res["status"] == "MISSING_BLOCK"
    assert not res["pass"]
    assert not res["weak_alpha_warn"]


def test_check_pending_minimal_ok():
    entry = {"factor_exposure": {"audit_status": "pending", "audit_priority": "high"}}
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert res["pass"]
    assert res["audit_status"] == "pending"


def test_check_pending_missing_priority():
    entry = {"factor_exposure": {"audit_status": "pending"}}
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert not res["pass"]
    assert "audit_priority" in res["missing"]


def test_check_audited_complete():
    entry = {
        "factor_exposure": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "phase": 1,
            "n_months": 100,
            "alpha_annualized": 0.05,
            "alpha_t_hac": 2.5,
            "betas": {"MKT": 0.1, "SMB": 0.05, "MOM": 0.2},
            "t_stats_hac": {"alpha": 2.5, "MKT": 0.5, "SMB": 0.3, "MOM": 1.5},
            "r_squared": 0.4,
            "verdict": "x" * 60,
            "audit_blocks_deploy_decision": False,
            "factor_tilted_by_design": False,
            "proposed_role": "alpha_seeker",
        }
    }
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert res["pass"]


def test_check_audited_missing_factor_beta():
    """Phase 1 audited must have all 3 betas (MKT/SMB/MOM)."""
    entry = {
        "factor_exposure": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "phase": 1,
            "n_months": 100,
            "alpha_annualized": 0.05,
            "alpha_t_hac": 2.5,
            "betas": {"MKT": 0.1},   # missing SMB + MOM
            "t_stats_hac": {"alpha": 2.5},
            "r_squared": 0.4,
            "verdict": "x" * 60,
            "audit_blocks_deploy_decision": False,
            "factor_tilted_by_design": False,
        }
    }
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert not res["pass"]
    assert "betas.SMB" in res["missing"]
    assert "betas.MOM" in res["missing"]


def test_weak_alpha_warn_on_deployed():
    """DEPLOYED sleeve with alpha_t < 2 and not tilted → warn flag."""
    entry = {
        "status_in_our_book": "DEPLOYED",
        "factor_exposure": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "phase": 1,
            "n_months": 100,
            "alpha_annualized": 0.01,
            "alpha_t_hac": 0.8,   # < 2.0 threshold
            "betas": {"MKT": 0.1, "SMB": 0.05, "MOM": 0.0},
            "t_stats_hac": {"alpha": 0.8, "MKT": 2.0, "SMB": 1.0, "MOM": 0.0},
            "r_squared": 0.15,
            "verdict": "x" * 60,
            "audit_blocks_deploy_decision": False,
            "factor_tilted_by_design": False,
            "proposed_role": "alpha_seeker",   # role that gets weak_alpha warnings
        }
    }
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert res["pass"]
    assert res["weak_alpha_warn"] is True


def test_weak_alpha_NOT_warned_when_tilted_by_design():
    """Same as above but factor_tilted_by_design=True → no warn."""
    entry = {
        "status_in_our_book": "DEPLOYED",
        "factor_exposure": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "phase": 1,
            "n_months": 100,
            "alpha_annualized": 0.01,
            "alpha_t_hac": 0.8,
            "betas": {"MKT": 0.1, "SMB": 0.05, "MOM": 0.5},
            "t_stats_hac": {"alpha": 0.8, "MKT": 2.0, "SMB": 1.0, "MOM": 4.0},
            "r_squared": 0.35,
            "verdict": "x" * 60,
            "audit_blocks_deploy_decision": False,
            "factor_tilted_by_design": True,   # explicitly intentional
            "proposed_role": "alpha_seeker",
        }
    }
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert res["pass"]
    assert res["weak_alpha_warn"] is False


def test_weak_alpha_NOT_warned_when_role_is_insurance():
    """FLAW 2 FIX: insurance role exempt from weak-alpha warning."""
    entry = {
        "status_in_our_book": "DEPLOYED",
        "factor_exposure": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "phase": 1,
            "n_months": 100,
            "alpha_annualized": -0.02,
            "alpha_t_hac": -1.5,   # negative alpha — expected for insurance
            "betas": {"MKT": -1.0, "SMB": 0.1, "MOM": -0.5},
            "t_stats_hac": {"alpha": -1.5, "MKT": -10.0, "SMB": 1.0, "MOM": -5.0},
            "r_squared": 0.85,
            "verdict": "Insurance role — negative drift is the premium paid for "
                       "MOM crash protection. " * 2,
            "audit_blocks_deploy_decision": False,
            "factor_tilted_by_design": False,
            "proposed_role": "insurance",
        }
    }
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert res["pass"]
    assert res["weak_alpha_warn"] is False, "insurance role exempt from weak-alpha"


def test_weak_alpha_NOT_warned_when_role_is_diversifier():
    """FLAW 2 FIX: diversifier role exempt from weak-alpha warning."""
    entry = {
        "status_in_our_book": "DEPLOYED",
        "factor_exposure": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "phase": 1,
            "n_months": 100,
            "alpha_annualized": 0.02,
            "alpha_t_hac": 1.0,   # weak alpha — but OK for diversifier
            "betas": {"MKT": 0.1, "SMB": 0.0, "MOM": -0.05},
            "t_stats_hac": {"alpha": 1.0, "MKT": 1.5, "SMB": 0.0, "MOM": -0.5},
            "r_squared": 0.4,
            "verdict": "Diversifier role — alpha not the gating criterion; "
                       "negative correlation with book is. " * 2,
            "audit_blocks_deploy_decision": False,
            "factor_tilted_by_design": False,
            "proposed_role": "diversifier",
        }
    }
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert res["pass"]
    assert res["weak_alpha_warn"] is False


def test_check_audited_rejects_unknown_role():
    """FLAW 3 FIX: proposed_role must be in VALID_ROLES set."""
    entry = {
        "status_in_our_book": "DEPLOYED",
        "factor_exposure": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "phase": 1,
            "n_months": 100,
            "alpha_annualized": 0.05,
            "alpha_t_hac": 2.5,
            "betas": {"MKT": 0.1, "SMB": 0.05, "MOM": 0.2},
            "t_stats_hac": {"alpha": 2.5},
            "r_squared": 0.4,
            "verdict": "x" * 60,
            "audit_blocks_deploy_decision": False,
            "factor_tilted_by_design": False,
            "proposed_role": "made_up_role",   # not in VALID_ROLES
        }
    }
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert res["pass"] is False
    assert any("proposed_role" in m for m in res["missing"])


def test_weak_alpha_NOT_warned_when_not_deployed():
    """RED / UNTESTED sleeves don't get weak-alpha warnings."""
    entry = {
        "status_in_our_book": "RED",
        "factor_exposure": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "phase": 1,
            "n_months": 100,
            "alpha_annualized": 0.0,
            "alpha_t_hac": 0.1,
            "betas": {"MKT": 0.0, "SMB": 0.0, "MOM": 0.0},
            "t_stats_hac": {"alpha": 0.1, "MKT": 0.0, "SMB": 0.0, "MOM": 0.0},
            "r_squared": 0.05,
            "verdict": "RED — does not work, included for completeness " * 2,
            "audit_blocks_deploy_decision": False,
            "factor_tilted_by_design": False,
            "proposed_role": "alpha_seeker",
        }
    }
    res = lfe._check_one(Path("/tmp/x.yaml"), entry)
    assert res["pass"]
    assert res["weak_alpha_warn"] is False


# -- audit_library on real library ---------------------------------------

def test_audit_library_no_missing_blocks():
    summary = lfe.audit_library()
    assert summary["total"] >= 9
    assert summary["missing_block"] == 0


def test_audit_library_3_audited_phase_1():
    summary = lfe.audit_library()
    by_name = {r["name"]: r for r in summary["results"]}
    audited = [r for r in summary["results"] if r["audit_status"] == "audited"]
    assert len(audited) >= 3
    expected_audited = {"post_earnings_drift", "cross_asset_carry", "time_series_momentum"}
    actual_audited_names = {r["name"] for r in audited}
    assert expected_audited.issubset(actual_audited_names)


def test_carry_sleeve_NOT_warned_after_role_assigned():
    """FLAW 2 FIX: carry was flagged in prior versions, but its
    proposed_role=risk_premium_harvester now exempts it correctly."""
    summary = lfe.audit_library()
    by_name = {r["name"]: r for r in summary["results"]}
    assert "cross_asset_carry" in by_name
    if by_name["cross_asset_carry"]["audit_status"] == "audited":
        assert by_name["cross_asset_carry"]["weak_alpha_warn"] is False, \
            "risk_premium_harvester role exempt from weak_alpha; FLAW 2 fix"


def test_tsmom_NOT_warned_because_tilted_by_design():
    """TSMOM has alpha_t=1.19 (< 2.0) but is factor-tilted-by-design.
    Should NOT trigger the weak-alpha warning."""
    summary = lfe.audit_library()
    by_name = {r["name"]: r for r in summary["results"]}
    if "time_series_momentum" in by_name and by_name["time_series_momentum"]["audit_status"] == "audited":
        assert by_name["time_series_momentum"]["weak_alpha_warn"] is False
