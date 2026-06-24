"""Tests for the two-tier confidence routing (A) + family-aware
threshold (C). Per [[project-e2e-smoke-v3-funnel-findings-2026-05-30]].

Verifies the Bitcoin ETF carry scenario: LLM identifies family but
abstract markers are thin → bumped to borderline, not skipped.
"""
from __future__ import annotations

from unittest import mock

import pytest

from engine.research.discovery import family_thresholds as ft
from engine.research.discovery.family_thresholds import (
    BORDERLINE_THRESHOLD, DEFAULT_THRESHOLD, FAMILY_BONUS_FLOOR,
    FAMILY_THRESHOLDS,
    adjust_confidence_for_llm_family, classify_confidence,
    explain_routing, threshold_for_family,
)


# ── threshold_for_family ──────────────────────────────────────────────────

def test_threshold_unknown_returns_default():
    assert threshold_for_family(None) == DEFAULT_THRESHOLD
    assert threshold_for_family("") == DEFAULT_THRESHOLD
    assert threshold_for_family("unknown") == DEFAULT_THRESHOLD


def test_threshold_strict_families_get_default():
    """value/momentum/quality literature uses explicit numeric markers
    → strict 0.50 threshold."""
    for fam in ("value", "momentum", "quality", "low_vol"):
        assert threshold_for_family(fam) == 0.50


def test_threshold_carry_family_is_looser():
    """Carry literature describes mechanism, leaves numerics to body →
    0.40 threshold."""
    assert threshold_for_family("carry") == 0.40
    assert threshold_for_family("vol_carry") == 0.40
    assert threshold_for_family("lead_lag") == 0.40


def test_threshold_event_driven_is_mid():
    """PEAD / behavioral / merger-arb sit at 0.45."""
    assert threshold_for_family("pead") == 0.45
    assert threshold_for_family("merger_arb") == 0.45


def test_threshold_unrecognized_family_returns_default():
    """Custom family not in the table → fall back to DEFAULT."""
    assert threshold_for_family("some_made_up_family") == DEFAULT_THRESHOLD


# ── adjust_confidence_for_llm_family (the bonus) ─────────────────────────

def test_no_bonus_when_family_unknown():
    adj, bumped = adjust_confidence_for_llm_family(0.10, "unknown")
    assert adj == 0.10
    assert not bumped


def test_bonus_bumps_low_confidence_to_floor_when_family_known():
    """The Bitcoin ETF case: LLM says family=carry but conf=0.10."""
    adj, bumped = adjust_confidence_for_llm_family(0.10, "carry")
    assert adj == FAMILY_BONUS_FLOOR
    assert bumped


def test_bonus_does_not_lower_already_high_confidence():
    """A 0.60 conf with known family stays 0.60 — bonus only bumps,
    never reduces."""
    adj, bumped = adjust_confidence_for_llm_family(0.60, "carry")
    assert adj == 0.60
    assert not bumped


def test_bonus_at_exact_floor_does_not_bump():
    """conf == FAMILY_BONUS_FLOOR → no bump needed."""
    adj, bumped = adjust_confidence_for_llm_family(FAMILY_BONUS_FLOOR, "carry")
    assert adj == FAMILY_BONUS_FLOOR
    assert not bumped


# ── classify_confidence (the tier decision) ──────────────────────────────

def test_high_confidence_known_family_routes_to_review():
    assert classify_confidence(0.60, "carry") == "review"
    assert classify_confidence(0.50, "carry") == "review"


def test_borderline_routes_to_borderline():
    """0.30 ≤ conf < family_threshold → borderline."""
    assert classify_confidence(0.35, "carry") == "borderline"     # 0.35 < 0.40
    assert classify_confidence(0.45, "momentum") == "borderline"   # 0.45 < 0.50


def test_below_floor_skips():
    """conf < BORDERLINE_THRESHOLD → skip."""
    assert classify_confidence(0.25, "carry") == "skip"
    assert classify_confidence(0.05, "value") == "skip"
    assert classify_confidence(0.0, "unknown") == "skip"


def test_unknown_family_strict_threshold():
    """Unknown family uses DEFAULT (0.50) threshold."""
    assert classify_confidence(0.45, "unknown") == "borderline"
    assert classify_confidence(0.50, "unknown") == "review"


# ── explain_routing (the audit dict) ─────────────────────────────────────

def test_explain_routing_bitcoin_etf_case():
    """LLM said family=carry, raw conf=0.10 → bonus bumps to 0.30,
    which is below carry threshold 0.40 → borderline tier (not skip)."""
    info = explain_routing(0.10, "carry")
    assert info["base_confidence"] == 0.10
    assert info["family"] == "carry"
    assert info["family_threshold"] == 0.40
    assert info["family_bonus_applied"] is True
    assert info["adjusted_confidence"] == 0.30
    assert info["routing"] == "borderline"


def test_explain_routing_unknown_low_conf_skips():
    """conf=0.10 family=unknown → no bonus → skip."""
    info = explain_routing(0.10, "unknown")
    assert info["family_bonus_applied"] is False
    assert info["adjusted_confidence"] == 0.10
    assert info["routing"] == "skip"


def test_explain_routing_high_conf_review():
    """conf=0.65 family=carry → review."""
    info = explain_routing(0.65, "carry")
    assert info["family_bonus_applied"] is False
    assert info["adjusted_confidence"] == 0.65
    assert info["routing"] == "review"


def test_explain_routing_all_keys_present():
    info = explain_routing(0.30, "carry")
    for k in ("base_confidence", "family", "family_threshold",
                "family_bonus_applied", "adjusted_confidence",
                "borderline_floor", "routing"):
        assert k in info


# ── End-to-end discovery_pipeline integration ───────────────────────────

def test_bitcoin_etf_carry_routes_to_borderline(monkeypatch, tmp_path):
    """E2E: simulate the Bitcoin ETF carry paper outcome — LLM family=carry,
    confidence=0.10 → routed to borderline_review."""
    from engine.research.discovery import discovery_pipeline as dp
    from engine.research.discovery import paper_extractor

    def _mock_extract(arxiv_id, title, abstract, *, use_llm=True):
        return paper_extractor.PaperExtraction(
            arxiv_id=arxiv_id, title=title,
            mechanism_proposal="Exploit ETF carry-rate wedges",
            family_guess="carry", parent_family_guess="cross_asset",
            required_data_tokens=["fx_futures"],   # token registered in DATA_INVENTORY
            economic_intuition="Carry segmentation creates wedges.",
            decay_resilience_claim="not addressed",
            novelty_assessment="extension", confidence=0.10,
            cost_usd=0.0005, mode="llm",
        )
    monkeypatch.setattr(paper_extractor, "extract_from_paper", _mock_extract)
    # Stub data inventory to accept the token so the test isolates the
    # routing-tier decision, not the data-presence check.
    import engine.research.hygiene_tools as ht
    monkeypatch.setattr(ht, "DATA_INVENTORY", {"fx_futures": {}})

    paper = {
        "arxiv_id": "test_btc_carry", "source_id": "test_btc_carry",
        "title": "Implied ETF Carry Rates and the Limits of Arbitrage",
        "abstract": "We study Bitcoin ETF carry rates in segmented markets.",
        "submitted_date": "2024-05-01",
        "venue": "arxiv",
    }
    # Pass empty library_families to bypass the cousin check (which
    # would otherwise fire on the existing "carry" family entry).
    # We're testing the new TWO-TIER routing logic in isolation.
    out = dp.process_paper(paper, use_llm=True, confidence_threshold=0.5,
                              library_titles=set(),
                              library_families={})

    # Should NOT skip with low_confidence (which old logic did)
    assert out["verdict"] != "skip", \
        f"Bitcoin ETF case should not skip; got verdict={out['verdict']}"
    # Should route to borderline_review (new behavior)
    assert out["verdict"] == "borderline_review"
    assert out["routing"]["routing"] == "borderline"
    assert out["routing"]["family_bonus_applied"] is True


def test_truly_low_confidence_unknown_family_still_skips():
    """Sanity: unknown family + raw conf 0.05 → skip (no bonus)."""
    from engine.research.discovery import discovery_pipeline as dp
    from engine.research.discovery import paper_extractor

    with mock.patch.object(
        paper_extractor, "extract_from_paper",
        return_value=paper_extractor.PaperExtraction(
            arxiv_id="x", title="x",
            mechanism_proposal="x", family_guess="unknown",
            parent_family_guess="unknown",
            required_data_tokens=[],
            economic_intuition="", decay_resilience_claim="",
            novelty_assessment="", confidence=0.05,
            cost_usd=0.0, mode="llm",
        ),
    ):
        paper = {"arxiv_id": "x", "source_id": "x",
                  "title": "Some random paper",
                  "abstract": "no factor markers here at all",
                  "submitted_date": "2024-01-01"}
        out = dp.process_paper(paper, use_llm=True, confidence_threshold=0.5)
        assert out["verdict"] == "skip"
        assert out["stage"] == "low_confidence"
