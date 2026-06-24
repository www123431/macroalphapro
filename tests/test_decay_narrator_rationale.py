"""tests/test_decay_narrator_rationale.py — the junior-analyst rationale layer.

Pins that the deterministic narrator now EXPLAINS its verdict (the Man-Group lesson): the
decay-vs-drawdown logic + per-mechanism evidence chain, composed from the report's own numbers,
0-LLM and banned-phrase-clean.
"""
from __future__ import annotations

from engine.agents.decay_sentinel.narrator import contains_banned_phrase, narrate_report


def _report(overall, mechs, decay, roles, weights, crisis=None, alarms=None):
    return {"overall": overall, "window": 36, "mechanisms": mechs, "decay": decay,
            "roles": roles, "base_weights": weights, "crisis": crisis or {},
            "alarms": alarms or [], "realloc_action": overall == "ACTION",
            "recommended_weights": {}}


def test_rationale_explains_structural_decay_action():
    mechs = {"X": {"rolling_sharpe": -0.2, "full_sharpe": 0.8, "decay_ratio": -0.25}}
    decay = {"X": {"signal_ic": -0.05, "structural_decay": True}}
    r = narrate_report(_report("ACTION", mechs, decay, {"X": "alpha"}, {"X": 0.3})).text
    assert "Why ACTION" in r and "structural decay" in r.lower()
    assert "halve" in r.lower() and "Evidence:" in r
    assert contains_banned_phrase(r) is None


def test_rationale_distinguishes_drawdown_from_decay():
    # soft return BUT signal-IC intact → drawdown, hold (not decay)
    mechs = {"X": {"rolling_sharpe": 0.1, "full_sharpe": 0.9, "decay_ratio": 0.11}}
    decay = {"X": {"signal_ic": 0.04, "structural_decay": False}}
    r = narrate_report(_report("WATCH", mechs, decay, {"X": "alpha"}, {"X": 0.3})).text
    assert "Why WATCH" in r and "drawdown" in r.lower() and "hold" in r.lower()
    assert contains_banned_phrase(r) is None


def test_rationale_watch_names_risk_flag_when_no_soft_mechanism():
    # all mechanisms healthy but a stress-corr WARN → name the actual driver, not "a flagged item"
    mechs = {"X": {"rolling_sharpe": 0.9, "full_sharpe": 0.8, "decay_ratio": 1.1}}
    decay = {"X": {"signal_ic": 0.05, "structural_decay": False}}
    alarms = [("WARN", "(X,Y) DOWNSIDE corr +0.46 > 0.4 — co-loss risk")]
    r = narrate_report(_report("WATCH", mechs, decay, {"X": "alpha"}, {"X": 0.5}, alarms=alarms)).text
    assert "RISK flag" in r and "co-loss" in r
    assert contains_banned_phrase(r) is None


def test_rationale_healthy_walks_evidence():
    mechs = {"A": {"rolling_sharpe": 1.2, "full_sharpe": 1.0, "decay_ratio": 1.2},
             "H": {"rolling_sharpe": float("nan"), "full_sharpe": float("nan")}}
    decay = {"A": {"signal_ic": 0.05, "structural_decay": False}, "H": {}}
    r = narrate_report(_report("HEALTHY", mechs, decay, {"A": "alpha", "H": "insurance"},
                               {"A": 0.5, "H": 0.1}, crisis={"H": 0.002})).text
    assert "Why HEALTHY" in r and "Evidence:" in r
    assert "premium for the hedge" in r            # insurance explained on crisis-payoff logic
    assert contains_banned_phrase(r) is None
