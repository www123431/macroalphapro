"""Tests for hygiene tool H10 — evaluate_candidate.

H10 is the unified L3 -> L4 evaluator. It composes role-inference + H8
(factor exposure, role-aware) + H9 (orthogonality vs book) into a single
agent-facing call.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research.hygiene_tools import (
    _h10_combine_to_final,
    _h10_infer_role_from_h8,
    h10_evaluate_candidate,
    execute_tool,
)
from engine.research import hygiene_tools as ht


# ── _h10_infer_role_from_h8 — pure-function tests ────────────────────────

def test_infer_role_insurance_when_negative_alpha_plus_negative_factor():
    pl = {
        "alpha_t_hac": -1.5,
        "r_squared": 0.85,
        "strong_factor_loadings": {"MOM": -0.4, "MKT": -1.0},
    }
    role, rationale = _h10_infer_role_from_h8(pl)
    assert role == "insurance"
    assert "negative" in rationale.lower()


def test_infer_role_alpha_seeker_when_strong_alpha_moderate_r2():
    pl = {
        "alpha_t_hac": 3.5,
        "r_squared": 0.55,
        "strong_factor_loadings": {"MOM": 0.5},
    }
    role, _ = _h10_infer_role_from_h8(pl)
    assert role == "alpha_seeker"


def test_infer_role_risk_premium_harvester_when_low_alpha_strong_factor():
    pl = {
        "alpha_t_hac": 0.5,
        "r_squared": 0.4,
        "strong_factor_loadings": {"MKT": 0.7},
    }
    role, _ = _h10_infer_role_from_h8(pl)
    assert role == "risk_premium_harvester"


def test_infer_role_diversifier_as_default_for_unclear():
    pl = {
        "alpha_t_hac": 0.8,
        "r_squared": 0.1,
        "strong_factor_loadings": {},
    }
    role, _ = _h10_infer_role_from_h8(pl)
    assert role == "diversifier"


# ── _h10_combine_to_final — pure-function tests ─────────────────────────

def _h8_payload(alpha_t: float, accept: bool, code: str,
                  betas: dict | None = None) -> dict:
    return {
        "alpha_t_hac":             alpha_t,
        "alpha_annualized":        alpha_t * 0.01,
        "r_squared":               0.4,
        "strong_factor_loadings":  betas or {},
        "role_aware_verdict": {
            "verdict_code": code,
            "accept":       accept,
            "explanation":  "test",
        },
    }


def test_combine_alpha_seeker_accepts_when_h8_accepts():
    h8 = _h8_payload(2.5, True, "STRONG_FOR_ROLE")
    h9 = {"cosine_to_book_risk": 0.1}
    final = _h10_combine_to_final("alpha_seeker", h8, h9)
    assert final["verdict_code"] == "ACCEPT_FOR_DEPLOY"
    assert final["accept"] is True


def test_combine_alpha_seeker_rejects_when_h8_rejects():
    h8 = _h8_payload(0.3, False, "WEAK_FOR_ROLE")
    final = _h10_combine_to_final("alpha_seeker", h8, {"cosine_to_book_risk": -0.5})
    assert final["accept"] is False


def test_combine_insurance_accepts_when_h8_accepts_and_h9_not_positive():
    h8 = _h8_payload(-1.5, True, "VALID_FOR_ROLE", betas={"MOM": -0.4})
    final = _h10_combine_to_final("insurance", h8, {"cosine_to_book_risk": -0.1})
    assert final["verdict_code"] == "ACCEPT_FOR_DEPLOY"
    assert final["accept"] is True


def test_combine_insurance_rejects_when_h9_pile_on():
    """Insurance with H8 valid + H9 cosine > 0.5 → REJECT (insurance
    piles onto book — this is the STR/BAB failure mode)."""
    h8 = _h8_payload(-1.5, True, "VALID_FOR_ROLE", betas={"MOM": -0.4})
    final = _h10_combine_to_final("insurance", h8, {"cosine_to_book_risk": +0.7})
    assert final["accept"] is False
    assert "PILE" in final["rationale"] or "pile" in final["rationale"]


def test_combine_diversifier_accepts_strong_negative_cosine():
    h8 = _h8_payload(1.0, None, "DEFERRED_TO_H9")
    final = _h10_combine_to_final("diversifier", h8, {"cosine_to_book_risk": -0.30})
    assert final["verdict_code"] == "ACCEPT_FOR_DEPLOY"
    assert final["accept"] is True


def test_combine_diversifier_rejects_positive_cosine():
    h8 = _h8_payload(1.0, None, "DEFERRED_TO_H9")
    final = _h10_combine_to_final("diversifier", h8, {"cosine_to_book_risk": +0.30})
    assert final["accept"] is False


def test_combine_diversifier_borderline_mildly_negative():
    h8 = _h8_payload(1.0, None, "DEFERRED_TO_H9")
    final = _h10_combine_to_final("diversifier", h8, {"cosine_to_book_risk": -0.10})
    assert final["verdict_code"] == "BORDERLINE_FOR_DEPLOY"


def test_combine_regime_overlay_routes_to_regime_backtest():
    h8 = _h8_payload(0.0, None, "ROLE_NOT_STATIC_TESTABLE")
    final = _h10_combine_to_final("regime_overlay", h8, None)
    assert final["verdict_code"] == "ROUTE_TO_REGIME_BACKTEST"
    assert final["accept"] is None


# ── End-to-end test (synthetic, monkeypatched) ───────────────────────────

@pytest.fixture
def synthetic_factor_panel(monkeypatch):
    np.random.seed(7)
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    factors = pd.DataFrame({
        "MKT": np.random.randn(120) * 0.04,
        "SMB": np.random.randn(120) * 0.03,
        "MOM": np.random.randn(120) * 0.04,
    }, index=idx)
    from engine.risk import barra_lite as bl
    monkeypatch.setattr(bl, "build_factor_returns", lambda phase=1: factors)
    # Mock the deployed book sleeves used by H9
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
    return factors, idx


def test_h10_full_pipeline_alpha_seeker(synthetic_factor_panel):
    """End-to-end: strong-alpha candidate + role=alpha_seeker → ACCEPT."""
    factors, idx = synthetic_factor_panel
    np.random.seed(101)
    candidate = pd.Series(0.018 + np.random.randn(120) * 0.003, index=idx)
    r = h10_evaluate_candidate(candidate, "good_alpha", phase=1,
                                  proposed_role="alpha_seeker")
    d = r.to_dict()
    assert d["success"]
    pl = d["payload"]
    assert pl["role_used"] == "alpha_seeker"
    assert pl["role_was_inferred"] is False
    assert pl["final"]["verdict_code"] == "ACCEPT_FOR_DEPLOY"


def test_h10_role_inference_works(synthetic_factor_panel):
    """End-to-end without proposed_role → role is inferred + ok."""
    factors, idx = synthetic_factor_panel
    np.random.seed(103)
    candidate = pd.Series(0.018 + np.random.randn(120) * 0.003, index=idx)
    r = h10_evaluate_candidate(candidate, "no_role_given", phase=1)
    d = r.to_dict()
    pl = d["payload"]
    assert pl["role_was_inferred"] is True
    assert pl["role_used"] in {
        "alpha_seeker", "risk_premium_harvester",
        "insurance", "diversifier",
    }


def test_h10_insurance_candidate_with_negative_factor(synthetic_factor_panel):
    """A candidate with anti-MOM construction + role=insurance → ACCEPT."""
    factors, idx = synthetic_factor_panel
    np.random.seed(107)
    candidate = (-1.5 * factors["MOM"] - 0.005
                 + np.random.RandomState(109).randn(120) * 0.003)
    r = h10_evaluate_candidate(candidate, "antimom", phase=1,
                                  proposed_role="insurance")
    pl = r.to_dict()["payload"]
    assert pl["final"]["accept"] is True


def test_h10_dispatch_registered(synthetic_factor_panel):
    factors, idx = synthetic_factor_panel
    np.random.seed(113)
    candidate = pd.Series(np.random.randn(120) * 0.01, index=idx)
    r = execute_tool(
        "h10_evaluate_candidate",
        candidate_sleeve_returns=candidate,
        proposal_name="dispatch",
        phase=1,
    )
    d = r.to_dict()
    assert d.get("error", None) is None or \
           "unknown hygiene tool" not in d.get("error", "")
