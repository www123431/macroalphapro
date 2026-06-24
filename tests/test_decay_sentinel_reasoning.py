"""Unit tests for engine.agents.decay_sentinel.reasoning.

Critical safety properties under test:
1. Deterministic narrator NEVER changes statuses from the input report
2. All narratives cite specific numbers from the report (no vibes)
3. recommended_action is None for HEALTHY, non-None for WATCH and ACTION
4. Overall counts aggregate per-mechanism statuses correctly
5. Role-specific judging respected (trend/insurance not flagged on calm Sharpe)
6. WATCH triggered when decay_ratio < 0.5 but not structural
7. ACTION triggered when structural_decay is True
8. No banned hedge words in narratives (BlackRock-Slack tone)
"""
from __future__ import annotations

import re

import pytest

from engine.agents.decay_sentinel.reasoning import (
    narrate_deterministic,
    narrate_mechanism,
    narrate_overall,
)

BANNED_PHRASES = (
    r"\bmaybe\b", r"\bperhaps\b", r"\bcould be\b", r"\bmight be\b",
    r"\bprobably\b", r"\bpossibly\b", r"\blikely\b", r"\bI think\b",
    r"\bI feel\b", r"\bseems? to\b", r"\bappears? to\b",
)
_BANNED_RE = re.compile("|".join(BANNED_PHRASES), re.IGNORECASE)


def _no_banned(text: str) -> bool:
    m = _BANNED_RE.search(text)
    return m is None


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_alpha() -> dict:
    return {
        "role": "alpha",
        "weight": 0.30,
        "full_sharpe": 1.05,
        "rolling_sharpe": 1.10,
        "rolling_t": 1.90,
        "decay_ratio": 1.05,
        "mkt_beta": -0.15,
        "stress_beta": -0.10,
        "crisis_payoff": 0.02,
        "structural_decay": False,
        "signal_ic": 0.04,
        "decay_reason": "old rule output",
    }


@pytest.fixture
def watch_alpha() -> dict:
    return {
        "role": "alpha",
        "weight": 0.20,
        "full_sharpe": 1.00,
        "rolling_sharpe": 0.30,
        "rolling_t": 0.50,
        "decay_ratio": 0.30,       # < 0.5 triggers WATCH
        "mkt_beta": 0.10,
        "stress_beta": 0.20,
        "crisis_payoff": 0.01,
        "structural_decay": False, # not structural yet
        "signal_ic": 0.015,
    }


@pytest.fixture
def action_alpha() -> dict:
    return {
        "role": "alpha",
        "weight": 0.15,
        "full_sharpe": 0.80,
        "rolling_sharpe": -0.05,
        "rolling_t": -0.10,
        "decay_ratio": -0.06,
        "mkt_beta": 0.05,
        "stress_beta": 0.30,
        "crisis_payoff": 0.005,
        "structural_decay": True,      # structural flag = ACTION
        "signal_ic": 0.002,            # below 0.005 fade threshold
    }


@pytest.fixture
def trend_hedge() -> dict:
    """A trend sleeve with weak calm Sharpe — should still be HEALTHY (role doctrine)."""
    return {
        "role": "trend",
        "weight": 0.10,
        "full_sharpe": 0.45,
        "rolling_sharpe": 0.05,
        "rolling_t": 0.10,
        "decay_ratio": 0.11,
        "mkt_beta": -0.12,
        "stress_beta": -0.45,
        "crisis_payoff": 0.012,
        "structural_decay": False,
        "signal_ic": None,
    }


@pytest.fixture
def insurance_hedge() -> dict:
    """Insurance sleeve with negative calm Sharpe — still HEALTHY by role doctrine."""
    return {
        "role": "insurance",
        "weight": 0.10,
        "full_sharpe": 0.53,
        "rolling_sharpe": -0.02,
        "rolling_t": -0.03,
        "decay_ratio": -0.04,
        "mkt_beta": 0.03,
        "stress_beta": -0.15,
        "crisis_payoff": 0.002,
        "structural_decay": False,
        "signal_ic": None,
    }


# ── Per-mechanism narrative correctness ──────────────────────────────────────

def test_healthy_alpha_status_and_action(healthy_alpha):
    out = narrate_mechanism("D_PEAD", healthy_alpha)
    assert out["status"] == "HEALTHY"
    assert out["recommended_action"] is None
    assert "1.10" in out["narrative"] or "1.1" in out["narrative"]   # rolling Sharpe cited
    assert _no_banned(out["narrative"])


def test_watch_alpha_status_and_action(watch_alpha):
    out = narrate_mechanism("SomeAlpha", watch_alpha)
    assert out["status"] == "WATCH"
    assert out["recommended_action"] is not None
    assert "WATCH" in out["narrative"] or "watch" in out["narrative"].lower()
    assert _no_banned(out["narrative"])


def test_action_alpha_status_and_action(action_alpha):
    out = narrate_mechanism("DecayedAlpha", action_alpha)
    assert out["status"] == "ACTION"
    assert out["recommended_action"] is not None
    assert "re-allocate" in out["narrative"].lower() or "structural decay" in out["narrative"].lower()
    assert _no_banned(out["narrative"])


def test_trend_role_not_flagged_on_calm_sharpe(trend_hedge):
    """Doctrine: trend hedges judged on crisis payoff, NOT calm Sharpe."""
    out = narrate_mechanism("CTA", trend_hedge)
    assert out["status"] == "HEALTHY"        # even with rolling Sharpe 0.05
    assert out["recommended_action"] is None
    assert "convex hedge" in out["narrative"] or "by design" in out["narrative"]


def test_insurance_role_not_flagged_on_negative_sharpe(insurance_hedge):
    out = narrate_mechanism("TLT_GLD", insurance_hedge)
    assert out["status"] == "HEALTHY"
    assert out["recommended_action"] is None


# ── Evidence-cited claim verification ────────────────────────────────────────

def test_evidence_contains_full_sharpe(healthy_alpha):
    out = narrate_mechanism("X", healthy_alpha)
    metrics = {ev["metric"] for ev in out["evidence"]}
    assert "full_sample_sharpe" in metrics
    assert "rolling_36m_sharpe" in metrics
    assert "rolling_36m_signal_ic" in metrics


def test_evidence_signal_ic_threshold(action_alpha):
    out = narrate_mechanism("X", action_alpha)
    ic_ev = next(ev for ev in out["evidence"] if ev["metric"] == "rolling_36m_signal_ic")
    assert ic_ev["verdict"] == "FADED"
    assert ic_ev["fade_threshold"] == 0.005


# ── Top-level orchestrator (narrate_deterministic) ───────────────────────────

def _make_report(mechanisms: dict, overall: str = "WATCH",
                 realloc: bool = False) -> dict:
    return {
        "as_of": "2026-05-29",
        "overall": overall,
        "realloc_action": realloc,
        "mechanisms": mechanisms,
    }


def test_overall_all_healthy(healthy_alpha):
    rpt = _make_report({"A": healthy_alpha, "B": dict(healthy_alpha)},
                       overall="HEALTHY")
    out = narrate_deterministic(rpt)
    assert out["mode"] == "deterministic"
    assert out["overall"]["counts"] == {"action": 0, "watch": 0, "healthy": 2}
    assert out["overall"]["recommended_action"] is None
    assert "no action" in out["overall"]["narrative"].lower()


def test_overall_with_action(action_alpha, healthy_alpha):
    rpt = _make_report({"bad": action_alpha, "good": healthy_alpha},
                       overall="ACTION", realloc=True)
    out = narrate_deterministic(rpt)
    assert out["overall"]["counts"] == {"action": 1, "watch": 0, "healthy": 1}
    assert "bad" in out["overall"]["narrative"]   # action target named
    assert "recommend_allocation" in (out["overall"]["recommended_action"] or "")


def test_overall_with_watch_only(watch_alpha, healthy_alpha):
    rpt = _make_report({"w": watch_alpha, "h": healthy_alpha}, overall="WATCH")
    out = narrate_deterministic(rpt)
    assert out["overall"]["counts"] == {"action": 0, "watch": 1, "healthy": 1}
    assert "monitor" in out["overall"]["narrative"].lower()


# ── Tone discipline (no banned phrases) ──────────────────────────────────────

@pytest.mark.parametrize("fixture_name", [
    "healthy_alpha", "watch_alpha", "action_alpha", "trend_hedge", "insurance_hedge",
])
def test_no_banned_phrases_in_narrative(request, fixture_name):
    m = request.getfixturevalue(fixture_name)
    out = narrate_mechanism("X", m)
    assert _no_banned(out["narrative"]), f"Banned phrase found in: {out['narrative']!r}"


# ── Verdict immutability — narrator NEVER changes deterministic statuses ─────

def test_narrator_preserves_input_overall():
    """If report says overall=ACTION, the narrator preserves it verbatim."""
    rpt = _make_report({}, overall="ACTION", realloc=True)
    out = narrate_deterministic(rpt)
    assert out["overall"]["book_health"] == "ACTION"


def test_narrator_preserves_per_mechanism_status(action_alpha):
    """Even if narrator THINKS a mech is healthy, structural_decay=True forces ACTION."""
    rpt = _make_report({"x": action_alpha})
    out = narrate_deterministic(rpt)
    assert out["per_mechanism"]["x"]["status"] == "ACTION"


def test_empty_mechanisms_dict():
    rpt = _make_report({}, overall="HEALTHY")
    out = narrate_deterministic(rpt)
    assert out["per_mechanism"] == {}
    assert out["overall"]["counts"] == {"action": 0, "watch": 0, "healthy": 0}
