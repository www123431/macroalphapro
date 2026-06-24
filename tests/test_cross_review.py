"""Unit tests for engine.agents.cross_review — Pattern 6 cross-agent DD v1.

Critical safety properties:
1. Stance parsing handles all 3 values + default to neutral on malformed
2. Theme extraction picks up the 7 standard themes
3. Deterministic handlers produce the correct stance for prototypical inputs
   (clear RED → concerned; clear GREEN → supportive; ambiguous → neutral)
4. Consensus aggregation correctly identifies majority / split / no_consensus
5. run_cross_review (deterministic) end-to-end yields a well-formed packet
6. Ledger append is content-preserving
7. Banned-phrase stripping does not produce garbled output
"""
from __future__ import annotations

import json

import pytest

from engine.agents.cross_review import (
    CandidateContext,
    PersonaReview,
    PERSONAS_V1,
    _aggregate_consensus,
    _deterministic_attribution,
    _deterministic_devils_advocate,
    _deterministic_risk_manager,
    _extract_themes,
    _parse_stance,
    _strip_banned,
    read_ledger,
    run_cross_review,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def green_candidate() -> CandidateContext:
    """A clean GREEN candidate similar to our TSMOM 5-leg case."""
    return CandidateContext(
        name      = "FakeGreenSleeve",
        mechanism = "Test mechanism",
        gate_result = {
            "verdict":            "GREEN",
            "standalone_sharpe":  0.62,
            "alpha_t_ff5umd":     3.12,
            "alpha_ann_ff5umd":   0.04,
            "deflated_sr":        0.91,
            "oos_sharpe":         0.35,
            "corr_with_book":     0.37,
            "n_trials":           5,
        },
        returns_summary = {
            "n_months": 305, "range": "2001-2026",
            "ann_ret": 0.062, "ann_vol": 0.10, "sharpe": 0.62, "maxdd": -0.15,
            "hit_rate": 0.58,
        },
    )


@pytest.fixture
def red_candidate() -> CandidateContext:
    """A clear RED similar to our Quality POC (Sharpe -0.67, α-t -5.39)."""
    return CandidateContext(
        name      = "FakeRedSleeve",
        mechanism = "Test reverse-direction mechanism",
        gate_result = {
            "verdict":            "RED",
            "standalone_sharpe":  -0.67,
            "alpha_t_ff5umd":     -5.39,
            "alpha_ann_ff5umd":   -0.10,
            "deflated_sr":        0.0,
            "oos_sharpe":         -0.70,
            "corr_with_book":     0.28,
            "n_trials":           20,
        },
        returns_summary = {
            "n_months": 129, "range": "2013-2024",
            "ann_ret": -0.053, "ann_vol": 0.08, "sharpe": -0.67, "maxdd": -0.46,
            "hit_rate": 0.43,
        },
    )


@pytest.fixture
def yellow_candidate() -> CandidateContext:
    """A borderline YELLOW similar to our VIX carry case."""
    return CandidateContext(
        name      = "FakeYellowSleeve",
        mechanism = "Test marginal mechanism",
        gate_result = {
            "verdict":            "RED",
            "standalone_sharpe":  0.225,
            "alpha_t_ff5umd":     -0.83,
            "alpha_ann_ff5umd":   -0.027,
            "deflated_sr":        0.114,
            "oos_sharpe":         0.576,
            "corr_with_book":     -0.18,
            "n_trials":           19,
        },
        returns_summary = {
            "n_months": 101, "range": "2018-2026",
            "ann_ret": 0.023, "ann_vol": 0.10, "sharpe": 0.225, "maxdd": -0.16,
            "hit_rate": 0.55,
        },
    )


# ─── Stance parsing ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Some text\nSTANCE: concerned",  "concerned"),
    ("Some text\nSTANCE: supportive", "supportive"),
    ("Some text\nSTANCE: neutral",    "neutral"),
    ("Some text\nstance: concerned",  "concerned"),
    ("Mixed CASE\nSTANCE: SUPPORTIVE", "supportive"),
    ("No stance line at all", "neutral"),
    ("STANCE: invalid_value",  "neutral"),
])
def test_parse_stance(text, expected):
    assert _parse_stance(text) == expected


# ─── Theme extraction ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,theme", [
    ("There is publication bias here",   "overfitting_risk"),
    ("Signal has decayed post-2018",     "decay_risk"),
    ("Correlation with the book is high", "correlation_risk"),
    ("Costs would eat the alpha",        "cost_risk"),
    ("Capacity is limited",              "capacity_risk"),
    ("Sample window is too narrow",      "sample_risk"),
    ("Orthogonal to existing factors",   "diversification_benefit"),
])
def test_extract_themes(text, theme):
    themes = _extract_themes(text)
    assert theme in themes


def test_extract_themes_multiple():
    text = "Sample window is short and turnover-cost would dominate"
    themes = _extract_themes(text)
    assert "sample_risk" in themes
    assert "cost_risk" in themes


# ─── Deterministic handlers ──────────────────────────────────────────────────

def test_devils_advocate_concerned_on_red(red_candidate):
    text = _deterministic_devils_advocate(red_candidate)
    assert _parse_stance(text) == "concerned"
    assert "kill-shot" in text.lower() or "reject" in text.lower()
    # Must cite at least one number
    assert any(s in text for s in ("-0.67", "-5.39", "-0.70", "0.0"))


def test_devils_advocate_supportive_on_green(green_candidate):
    text = _deterministic_devils_advocate(green_candidate)
    assert _parse_stance(text) == "supportive"
    assert "deploy" in text.lower() or "no kill" in text.lower()


def test_devils_advocate_concerned_on_yellow_marginal(yellow_candidate):
    text = _deterministic_devils_advocate(yellow_candidate)
    # standalone Sharpe 0.225 < 0.4 + alpha-t -0.83 not significant + deflSR 0.114 → concerned
    assert _parse_stance(text) == "concerned"


def test_attribution_concerned_on_negative_alpha(red_candidate):
    text = _deterministic_attribution(red_candidate)
    assert _parse_stance(text) == "concerned"
    assert "alpha" in text.lower()


def test_attribution_supportive_on_green_orthogonal(green_candidate):
    # alpha-t 3.12 > 2 AND book corr 0.37 < 0.5 → supportive
    text = _deterministic_attribution(green_candidate)
    assert _parse_stance(text) == "supportive"


def test_risk_manager_concerned_on_huge_maxdd(red_candidate):
    text = _deterministic_risk_manager(red_candidate)
    # MaxDD -46% breaches -20% guard → concerned
    assert _parse_stance(text) == "concerned"


def test_risk_manager_neutral_on_low_sharpe_no_kill_factor(yellow_candidate):
    text = _deterministic_risk_manager(yellow_candidate)
    # Sharpe 0.225 < 0.5, MaxDD -16% not breaching, no kill factor → neutral
    assert _parse_stance(text) == "neutral"


# ─── Consensus aggregation ───────────────────────────────────────────────────

def _make_review(persona: str, stance: str, themes: list[str] = None) -> PersonaReview:
    return PersonaReview(
        persona=persona, workload="x", agent_id="x", mode="deterministic",
        text=f"...\nSTANCE: {stance}", stance=stance, themes=themes or [],
    )


def test_consensus_majority_concerned():
    reviews = [
        _make_review("a", "concerned"),
        _make_review("b", "concerned"),
        _make_review("c", "supportive"),
    ]
    c = _aggregate_consensus(reviews)
    assert c["summary"] == "majority_concerned"
    assert c["n_concerned"] == 2
    assert c["n_supportive"] == 1


def test_consensus_majority_supportive():
    reviews = [
        _make_review("a", "supportive"),
        _make_review("b", "supportive"),
        _make_review("c", "neutral"),
    ]
    c = _aggregate_consensus(reviews)
    assert c["summary"] == "majority_supportive"


def test_consensus_split():
    reviews = [
        _make_review("a", "concerned"),
        _make_review("b", "supportive"),
        _make_review("c", "neutral"),
    ]
    c = _aggregate_consensus(reviews)
    # 1 concerned vs 1 supportive — split (since both > 0 and equal)
    assert c["summary"] == "split"


def test_consensus_themes_union():
    reviews = [
        _make_review("a", "concerned", ["overfitting_risk", "decay_risk"]),
        _make_review("b", "concerned", ["decay_risk", "cost_risk"]),
    ]
    c = _aggregate_consensus(reviews)
    assert set(c["themes"]) == {"overfitting_risk", "decay_risk", "cost_risk"}


# ─── End-to-end deterministic run_cross_review ───────────────────────────────

def test_run_cross_review_red_candidate(red_candidate, tmp_path, monkeypatch):
    # Redirect ledger to a tmp file to avoid polluting production data
    import engine.agents.cross_review as m
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(m, "LEDGER_PATH", ledger)

    packet = run_cross_review(red_candidate, use_llm=False, log=True)

    assert packet.candidate == "FakeRedSleeve"
    assert packet.gate_verdict == "RED"
    assert packet.n_reviews == 3
    # All 3 personas should be concerned for this RED candidate
    assert packet.consensus["n_concerned"] == 3
    assert packet.consensus["summary"] == "majority_concerned"
    # Ledger written
    assert ledger.exists()
    raw = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1
    parsed = json.loads(raw[0])
    assert parsed["candidate"] == "FakeRedSleeve"
    assert parsed["gate_verdict"] == "RED"
    assert len(parsed["reviews"]) == 3


def test_run_cross_review_green_candidate(green_candidate, tmp_path, monkeypatch):
    import engine.agents.cross_review as m
    monkeypatch.setattr(m, "LEDGER_PATH", tmp_path / "ledger.jsonl")

    packet = run_cross_review(green_candidate, use_llm=False, log=True)

    assert packet.consensus["n_supportive"] >= 2
    assert packet.consensus["summary"] in ("majority_supportive", "split")


def test_run_cross_review_no_logging(red_candidate, tmp_path, monkeypatch):
    import engine.agents.cross_review as m
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(m, "LEDGER_PATH", ledger)

    packet = run_cross_review(red_candidate, use_llm=False, log=False)
    assert not ledger.exists()
    assert packet.candidate == "FakeRedSleeve"


# ─── Banned-phrase stripping ─────────────────────────────────────────────────

def test_strip_banned_removes_phrases():
    text = "This is maybe a strong signal, probably worth looking at."
    out = _strip_banned(text)
    assert "maybe" not in out.lower()
    assert "probably" not in out.lower()


def test_strip_banned_preserves_other_text():
    text = "Sharpe is 1.10 and OOS is 0.83."
    assert _strip_banned(text) == text


# ─── Persona registry sanity ─────────────────────────────────────────────────

def test_v1_personas_have_three_distinct_workloads():
    workloads = {p.workload for p in PERSONAS_V1}
    assert len(workloads) == 3
    assert workloads == {"devils_advocate", "attribution_analyst", "rm_agent"}


def test_persona_specs_have_required_fields():
    for p in PERSONAS_V1:
        assert p.display_name
        assert p.workload
        assert p.agent_id
        assert p.system_prompt
        assert "BANNED vocabulary" in p.system_prompt
        assert "STANCE:" in p.system_prompt
