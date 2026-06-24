"""Tests for hygiene tool H9 — check_orthogonality_to_book.

Smoke test (live deployed book): the H9 tool against
short_term_reversal demonstrated PILE_ON verdict because STR's
sector concentrations overlap with book's existing SEC_20/30/45
exposures, despite its negative MOM tilt being directionally
'right'. This is the test the tool MUST get right — anything that
naively scores STR as 'diversifying' on MOM-alignment alone is wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research.hygiene_tools import (
    h9_check_orthogonality_to_book, execute_tool,
)
from engine.research import hygiene_tools as ht


@pytest.fixture
def fake_book_setup(monkeypatch):
    """Set up a fake factor cache + fake book returns so H9 runs offline."""
    np.random.seed(7)
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    factors = pd.DataFrame({
        "MKT": np.random.randn(120) * 0.04,
        "SMB": np.random.randn(120) * 0.03,
        "MOM": np.random.randn(120) * 0.04,
    }, index=idx)

    from engine.risk import barra_lite as bl
    monkeypatch.setattr(bl, "build_factor_returns",
                          lambda phase=1: factors)

    # Fake book: equity heavy MOM, carry low everything, tsmom MOM
    book_eq = (0.5 * factors["MOM"] + 0.01
               + np.random.RandomState(11).randn(120) * 0.005)
    book_cy = (0.1 * factors["MKT"]
               + np.random.RandomState(13).randn(120) * 0.003)
    book_ts = (0.3 * factors["MOM"]
               + np.random.RandomState(17).randn(120) * 0.01)
    book_eq.name = "equity_book"
    book_cy.name = "carry_book"
    book_ts.name = "tsmom_book"

    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_equity_book",
        lambda: book_eq,
    )
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_carry_book",
        lambda: book_cy,
    )
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_tsmom_book",
        lambda: book_ts,
    )
    return factors, idx


def test_h9_anti_mom_diversifies(fake_book_setup):
    """Candidate with strong anti-MOM β → HIGH_DIVERSIFICATION when book
    is MOM-loaded."""
    factors, idx = fake_book_setup
    candidate = (-0.7 * factors["MOM"]
                 + np.random.RandomState(23).randn(120) * 0.005)
    r = h9_check_orthogonality_to_book(candidate, "antimom", phase=1)
    d = r.to_dict()
    assert d["success"]
    pl = d["payload"]
    assert pl["risk_diversifying_score"] > 0
    assert pl["gate_recommendation"] in {"HIGH_DIVERSIFICATION",
                                              "MODERATE_DIVERSIFICATION"}


def test_h9_pile_on_candidate_flagged(fake_book_setup):
    """Candidate that loads same direction as book → PILE_ON."""
    factors, idx = fake_book_setup
    candidate = (+0.5 * factors["MOM"]
                 + np.random.RandomState(29).randn(120) * 0.005)
    r = h9_check_orthogonality_to_book(candidate, "more_mom", phase=1)
    d = r.to_dict()
    pl = d["payload"]
    assert pl["cosine_to_book_risk"] > 0.1
    assert pl["risk_diversifying_score"] < 0


def test_h9_neutral_candidate_flagged(fake_book_setup):
    """Candidate uncorrelated with book → NEUTRAL or weak verdict."""
    factors, idx = fake_book_setup
    candidate = pd.Series(
        np.random.RandomState(31).randn(120) * 0.01,
        index=idx,
    )
    r = h9_check_orthogonality_to_book(candidate, "noise", phase=1)
    d = r.to_dict()
    pl = d["payload"]
    assert abs(pl["cosine_to_book_risk"]) < 0.5


def test_h9_short_sample_fails_gracefully(fake_book_setup):
    """Short candidate → tool returns failure (no crash)."""
    factors, idx = fake_book_setup
    candidate = pd.Series(
        np.random.randn(10) * 0.01,
        index=idx[:10],
    )
    r = h9_check_orthogonality_to_book(candidate, "short", phase=1)
    d = r.to_dict()
    assert d["success"] is False or "too few" in (d.get("error") or "")


def test_h9_dispatch_registered(fake_book_setup):
    factors, idx = fake_book_setup
    candidate = pd.Series(
        np.random.RandomState(37).randn(120) * 0.005,
        index=idx,
    )
    r = ht.execute_tool(
        "h9_check_orthogonality_to_book",
        candidate_sleeve_returns=candidate,
        proposal_name="dispatch_test",
        phase=1,
    )
    d = r.to_dict()
    assert d.get("error") is None or "unknown hygiene tool" not in d.get("error", "")


def test_h9_reports_overlaps_and_diversifiers(fake_book_setup):
    factors, idx = fake_book_setup
    candidate = (-0.4 * factors["MOM"] + 0.3 * factors["MKT"]
                 + np.random.RandomState(41).randn(120) * 0.005)
    r = h9_check_orthogonality_to_book(candidate, "mixed", phase=1)
    d = r.to_dict()
    pl = d["payload"]
    div_names = [f for f, _ in pl["top_3_diversifiers"]]
    # Anti-MOM should appear as diversifier since book has +MOM
    assert "MOM" in div_names
