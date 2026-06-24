"""tests/test_dq_summary_rationale.py — DQ Inspector run-level rationale (junior-analyst layer, 3/3).

Pins the run-level data-quality verdict: explains CLEAN (all freshness checks pass) as well as
WARN/HALT, names the driving check, banned-phrase clean, 0-LLM.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from engine.agents.dq_inspector.narrator import contains_banned_phrase, narrate_dq_summary


def _b(mode, sev, rule, affected=None):
    return NS(mode_id=mode, severity=sev, rule_description=rule, affected=affected or [])


def test_dq_clean_explains_pass():
    r = narrate_dq_summary([])
    assert "CLEAN" in r and "freshness" in r
    assert contains_banned_phrase(r) is None


def test_dq_warn_names_source_and_proceeds():
    r = narrate_dq_summary([_b("1", "SOFT_WARN", "FRED series stale 3d", ["DGS10"])])
    assert "WARN" in r and "DGS10" in r and "proceeds" in r
    assert contains_banned_phrase(r) is None


def test_dq_halt_blocks_batch():
    r = narrate_dq_summary([_b("3", "HARD_HALT", "PEAD panel rdq coverage 0")])
    assert "HARD HALT" in r and "blocked" in r
    assert contains_banned_phrase(r) is None
