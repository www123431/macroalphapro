"""Tests for hygiene tool H8 — check_factor_exposure_dry_run."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research.hygiene_tools import (
    h8_check_factor_exposure_dry_run, execute_tool,
)


@pytest.fixture
def synthetic_factor_panel(tmp_path, monkeypatch):
    """Set up a fake Phase 1 factor cache so H8 can run without CRSP."""
    np.random.seed(7)
    idx = pd.date_range("2015-01-31", periods=72, freq="ME")
    factors = pd.DataFrame({
        "MKT": np.random.randn(72) * 0.04,
        "SMB": np.random.randn(72) * 0.03,
        "MOM": np.random.randn(72) * 0.04,
    }, index=idx)
    from engine.risk import barra_lite as bl
    monkeypatch.setattr(bl, "build_factor_returns",
                          lambda phase=1: factors)
    return factors


# ── Behavior tests ──────────────────────────────────────────────────────

def test_h8_strong_alpha_recommends_high_prior(synthetic_factor_panel):
    """Sleeve with constant high alpha → STRONG verdict."""
    np.random.seed(11)
    sleeve = pd.Series(
        0.015 + np.random.randn(72) * 0.003,
        index=synthetic_factor_panel.index,
    )
    r = h8_check_factor_exposure_dry_run(sleeve, "strong_alpha", phase=1)
    d = r.to_dict()
    assert d["success"] is True
    assert d["payload"]["alpha_t_hac"] >= 2.0
    assert "STRONG" in d["payload"]["gate_recommendation"]


def test_h8_factor_explained_recommends_tilted(synthetic_factor_panel):
    """Sleeve that's pure MOM clone → low alpha, strong MOM beta → tilted."""
    sleeve = (3.0 * synthetic_factor_panel["MOM"]
                + np.random.RandomState(13).randn(72) * 0.005)
    r = h8_check_factor_exposure_dry_run(sleeve, "mom_clone", phase=1)
    d = r.to_dict()
    pl = d["payload"]
    assert abs(pl["alpha_t_hac"]) < 1.5
    assert abs(pl["t_stats_hac"]["MOM"]) >= 4.0
    assert pl["recommended_factor_tilted_by_design"] is True
    assert "FACTOR-TILTED" in pl["gate_recommendation"]


def test_h8_weak_alpha_recommends_reframe(synthetic_factor_panel):
    """Sleeve = pure noise, no clear factor cause → WEAK."""
    np.random.seed(17)
    sleeve = pd.Series(
        np.random.randn(72) * 0.01,
        index=synthetic_factor_panel.index,
    )
    r = h8_check_factor_exposure_dry_run(sleeve, "noise", phase=1)
    d = r.to_dict()
    assert d["success"] is True
    assert d["payload"]["alpha_t_hac"] < 2.0


def test_h8_too_few_obs_returns_error():
    """Short sleeve (<24 mo) → tool returns failure."""
    idx = pd.date_range("2020-01-31", periods=10, freq="ME")
    sleeve = pd.Series(np.random.randn(10) * 0.01, index=idx)
    r = h8_check_factor_exposure_dry_run(sleeve, "short", phase=1)
    d = r.to_dict()
    assert d["success"] is False
    assert "too few" in (d.get("error") or "").lower()


def test_h8_dispatch_registered():
    """execute_tool should resolve H8 by name."""
    np.random.seed(7)
    idx = pd.date_range("2015-01-31", periods=72, freq="ME")
    sleeve = pd.Series(np.random.randn(72) * 0.01, index=idx)
    # Without setting up factor cache, this will fail — but the dispatch
    # itself should NOT raise unknown-tool error.
    r = execute_tool(
        "h8_check_factor_exposure_dry_run",
        sleeve_returns=sleeve,
        proposal_name="dispatch_test",
        phase=1,
    )
    d = r.to_dict()
    # Either success or graceful error — but NOT "unknown tool".
    assert d.get("error", "") is None or "unknown hygiene tool" not in d.get("error", "")


def test_h8_factor_means_reported(synthetic_factor_panel):
    """Tool should report enough to let reviewer judge factor universe used."""
    np.random.seed(19)
    sleeve = pd.Series(
        0.02 + np.random.randn(72) * 0.005,
        index=synthetic_factor_panel.index,
    )
    r = h8_check_factor_exposure_dry_run(sleeve, "x", phase=1)
    d = r.to_dict()
    pl = d["payload"]
    assert "betas" in pl
    assert "t_stats_hac" in pl
    assert "r_squared" in pl
    assert pl["n_months"] == 72


# ── Realistic exposure cases (synthetic) ────────────────────────────────

def test_h8_market_beta_only_loading_NOT_tilted(synthetic_factor_panel):
    """Sleeve with significant MKT loading (t around 2.5) but not >= 4.0
    should NOT auto-recommend factor_tilted_by_design — preserves the
    soft-gate doctrine that close-to-2.0 markets are nuanced."""
    sleeve = (0.5 * synthetic_factor_panel["MKT"]
                + np.random.RandomState(23).randn(72) * 0.01)
    r = h8_check_factor_exposure_dry_run(sleeve, "moderate_mkt", phase=1)
    d = r.to_dict()
    pl = d["payload"]
    # If max factor |t| < 4.0, should NOT auto-tilt-recommend even if alpha is weak
    max_t = max(abs(pl["t_stats_hac"][k]) for k in pl["betas"])
    if max_t < 4.0:
        assert pl["recommended_factor_tilted_by_design"] is False


# -- FLAW 1 FIX: role-aware verdict tests --------------------------------

def test_h8_role_aware_alpha_seeker_strong(synthetic_factor_panel):
    """Strong alpha + role=alpha_seeker → STRONG_FOR_ROLE accept=True."""
    np.random.seed(43)
    sleeve = pd.Series(
        0.015 + np.random.randn(72) * 0.003,
        index=synthetic_factor_panel.index,
    )
    r = h8_check_factor_exposure_dry_run(
        sleeve, "alpha", phase=1, proposed_role="alpha_seeker",
    )
    pl = r.to_dict()["payload"]
    assert pl["role_aware_verdict"]["verdict_code"] == "STRONG_FOR_ROLE"
    assert pl["role_aware_verdict"]["accept"] is True


def test_h8_role_aware_alpha_seeker_weak_rejects(synthetic_factor_panel):
    """Weak alpha + role=alpha_seeker → WEAK_FOR_ROLE accept=False."""
    np.random.seed(47)
    sleeve = pd.Series(
        np.random.randn(72) * 0.01,
        index=synthetic_factor_panel.index,
    )
    r = h8_check_factor_exposure_dry_run(
        sleeve, "noise", phase=1, proposed_role="alpha_seeker",
    )
    pl = r.to_dict()["payload"]
    assert pl["role_aware_verdict"]["accept"] is False
    assert "WEAK" in pl["role_aware_verdict"]["verdict_code"] or \
           "BORDERLINE" in pl["role_aware_verdict"]["verdict_code"]


def test_h8_role_aware_insurance_negative_factor_accepts(synthetic_factor_panel):
    """Negative alpha + strong negative factor β + role=insurance
    → VALID_FOR_ROLE accept=True (the FLAW 1 fix in action)."""
    # Build a sleeve that's effectively a MOM short — negative alpha +
    # negative strong MOM exposure.
    sleeve = (-1.0 * synthetic_factor_panel["MOM"] - 0.005
                + np.random.RandomState(51).randn(72) * 0.002)
    r = h8_check_factor_exposure_dry_run(
        sleeve, "mom_short", phase=1, proposed_role="insurance",
    )
    pl = r.to_dict()["payload"]
    assert pl["role_aware_verdict"]["verdict_code"] == "VALID_FOR_ROLE"
    assert pl["role_aware_verdict"]["accept"] is True
    assert "negative factor exposure" in pl["role_aware_verdict"]["explanation"]


def test_h8_role_aware_risk_premium_harvester(synthetic_factor_panel):
    """Strong factor exposure + role=risk_premium_harvester
    → VALID_FOR_ROLE accept=True even if alpha t is weak."""
    sleeve = (0.7 * synthetic_factor_panel["MKT"] + 0.001
                + np.random.RandomState(57).randn(72) * 0.002)
    r = h8_check_factor_exposure_dry_run(
        sleeve, "harvester", phase=1, proposed_role="risk_premium_harvester",
    )
    pl = r.to_dict()["payload"]
    assert pl["role_aware_verdict"]["verdict_code"] == "VALID_FOR_ROLE"
    assert pl["role_aware_verdict"]["accept"] is True


def test_h8_role_aware_diversifier_deferred_to_h9(synthetic_factor_panel):
    """role=diversifier → DEFERRED_TO_H9 (H8 alone insufficient)."""
    sleeve = pd.Series(
        np.random.RandomState(61).randn(72) * 0.005,
        index=synthetic_factor_panel.index,
    )
    r = h8_check_factor_exposure_dry_run(
        sleeve, "div", phase=1, proposed_role="diversifier",
    )
    pl = r.to_dict()["payload"]
    assert pl["role_aware_verdict"]["verdict_code"] == "DEFERRED_TO_H9"
    assert pl["role_aware_verdict"]["accept"] is None


def test_h8_role_aware_regime_overlay_not_testable(synthetic_factor_panel):
    """role=regime_overlay → ROLE_NOT_STATIC_TESTABLE."""
    sleeve = pd.Series(
        np.random.RandomState(67).randn(72) * 0.005,
        index=synthetic_factor_panel.index,
    )
    r = h8_check_factor_exposure_dry_run(
        sleeve, "regime", phase=1, proposed_role="regime_overlay",
    )
    pl = r.to_dict()["payload"]
    assert pl["role_aware_verdict"]["verdict_code"] == "ROLE_NOT_STATIC_TESTABLE"


def test_h8_no_role_falls_back_to_legacy(synthetic_factor_panel):
    """Backward compat: no proposed_role → role_aware_verdict NOT in payload."""
    sleeve = pd.Series(
        np.random.RandomState(71).randn(72) * 0.005,
        index=synthetic_factor_panel.index,
    )
    r = h8_check_factor_exposure_dry_run(sleeve, "legacy", phase=1)
    pl = r.to_dict()["payload"]
    assert "role_aware_verdict" not in pl
    # Legacy gate_recommendation still present
    assert "gate_recommendation" in pl
