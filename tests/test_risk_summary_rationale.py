"""tests/test_risk_summary_rationale.py — Risk Manager run-level rationale (junior-analyst layer).

The per-breach templates only fire on a breach; this pins the run-level synthesis that explains a
CLEAR run too (binding constraint + headroom) and names the breach on HALT — deterministic,
banned-phrase clean.
"""
from __future__ import annotations

from engine.agents.risk_manager.narrator import contains_banned_phrase, narrate_risk_summary


def test_clear_run_reports_binding_and_headroom():
    utils = [
        {"mode": "gross leverage", "observed_txt": "1.08×", "limit_txt": "2.50×", "util": 0.43},
        {"mode": "single-name cap", "observed_txt": "16%", "limit_txt": "25%", "util": 0.64},
        {"mode": "HHI", "observed_txt": "0.05", "limit_txt": "0.20", "util": 0.25},
    ]
    r = narrate_risk_summary(utils, "PASS", halt=False)
    assert "CLEAR" in r
    assert "single-name cap" in r           # binding = highest utilization (0.64)
    assert "headroom" in r
    assert "Next-closest" in r and "gross leverage" in r
    assert contains_banned_phrase(r) is None


def test_halt_run_names_the_breach():
    utils = [
        {"mode": "gross leverage", "observed_txt": "2.70×", "limit_txt": "2.50×", "util": 1.08},
        {"mode": "HHI", "observed_txt": "0.05", "limit_txt": "0.20", "util": 0.25},
    ]
    r = narrate_risk_summary(utils, "HARD_HALT", halt=True)
    assert "HALT" in r and "gross leverage" in r
    assert "re-submission" in r
    assert contains_banned_phrase(r) is None


def test_no_live_modes_safe():
    r = narrate_risk_summary([{"mode": "x", "observed_txt": "", "limit_txt": "", "util": float("nan")}],
                             "PASS", halt=False)
    assert "CLEAR" in r and contains_banned_phrase(r) is None
