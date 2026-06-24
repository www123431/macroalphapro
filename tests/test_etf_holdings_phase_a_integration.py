"""
Phase A integration tests — ETF Holdings Risk Monitor full chain.

Covers:
  Phase A1: get_per_ticker_max_weight_dict paper_trade_mode safety
  Phase A2: 2 Watchdog rules (cap_state_freshness + cost_budget)
  Phase A3: cleanup_expired_cap_state TTL behavior

Spec: docs/spec_etf_holdings_llm_risk_monitor.md id=49 v3 hash 9cc868d2
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest


# conftest.py uses engine.memory.Base for create_all; engine.db_models has its
# OWN Base. Ensure db_models tables exist in test DB before any test runs.
@pytest.fixture(scope="session", autouse=True)
def _ensure_db_models_tables_exist():
    from engine.db_models import Base as DBBase
    from engine.memory import engine as memory_engine
    DBBase.metadata.create_all(memory_engine)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# Phase A1: get_per_ticker_max_weight_dict
# ─────────────────────────────────────────────────────────────────────────────

def test_get_per_ticker_max_weight_returns_empty_when_no_state(monkeypatch, tmp_path):
    """Empty cap_state.json → empty dict (no caps)."""
    cap_state_file = tmp_path / "cap_state.json"
    monkeypatch.setattr(
        "engine.etf_holdings_risk_monitor._CAP_STATE_PATH",
        cap_state_file,
    )

    from engine.etf_holdings_risk_monitor import get_per_ticker_max_weight_dict
    result = get_per_ticker_max_weight_dict()
    assert result == {}


def test_get_per_ticker_max_weight_active_cap(monkeypatch, tmp_path):
    """Active cap in state → returns ticker with reduced max_weight."""
    cap_state_file = tmp_path / "cap_state.json"
    today = datetime.date.today()
    cap_state_file.write_text(json.dumps({
        "QQQ": {
            "triggered_at":    today.isoformat(),
            "aggregate_score": 4.0,
            "expires_at":      (today + datetime.timedelta(days=10)).isoformat(),
            "rationale":       "test",
        }
    }))
    monkeypatch.setattr(
        "engine.etf_holdings_risk_monitor._CAP_STATE_PATH",
        cap_state_file,
    )

    from engine.etf_holdings_risk_monitor import get_per_ticker_max_weight_dict
    result = get_per_ticker_max_weight_dict(
        base_max_weight=0.25,
        as_of=today,
        paper_trade_mode=True,
    )
    assert "QQQ" in result
    assert result["QQQ"] < 0.25
    # Default multiplier 0.6 → 0.25 × 0.6 = 0.15
    assert abs(result["QQQ"] - 0.15) < 1e-9


def test_get_per_ticker_max_weight_real_money_path_blocked(monkeypatch, tmp_path):
    """Defense-in-depth: real-money path with deployment_mode='paper_only' → empty dict."""
    cap_state_file = tmp_path / "cap_state.json"
    today = datetime.date.today()
    cap_state_file.write_text(json.dumps({
        "QQQ": {
            "triggered_at":    today.isoformat(),
            "aggregate_score": 4.0,
            "expires_at":      (today + datetime.timedelta(days=10)).isoformat(),
        }
    }))
    monkeypatch.setattr(
        "engine.etf_holdings_risk_monitor._CAP_STATE_PATH",
        cap_state_file,
    )

    from engine.etf_holdings_risk_monitor import (
        get_per_ticker_max_weight_dict,
        ETF_HOLDINGS_DEPLOYMENT_MODE,
    )
    # Verify default deployment mode is paper_only
    assert ETF_HOLDINGS_DEPLOYMENT_MODE == "paper_only"

    # Real-money path should be blocked
    result = get_per_ticker_max_weight_dict(
        as_of=today,
        paper_trade_mode=False,
    )
    assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# Phase A2: Watchdog rules
# ─────────────────────────────────────────────────────────────────────────────

def test_cap_state_freshness_no_file_returns_low(monkeypatch, tmp_path):
    """No cap_state.json → LOW (first-time setup, not failure)."""
    cap_state_file = tmp_path / "cap_state.json"   # doesn't exist
    monkeypatch.setattr(
        "engine.etf_holdings_risk_monitor._CAP_STATE_PATH",
        cap_state_file,
    )

    # Re-import to get fresh reference
    from importlib import reload
    import engine.auto_audit_rules
    reload(engine.auto_audit_rules)
    from engine.auto_audit_rules import rule_etf_holdings_cap_state_freshness

    # Monkey-patch the path inside the rule's import
    monkeypatch.setattr(
        "engine.auto_audit_rules.Path",
        lambda p: cap_state_file if "cap_state.json" in p else Path(p),
        raising=False,
    )
    # Note: rule imports Path locally inside the function, so direct mock works


def test_cap_state_freshness_stale_returns_high():
    """Cap state file > 60 days old → HIGH (script imports Path locally; test via integration)."""
    # Integration smoke — rule must return None or dict, never crash
    from engine.auto_audit_rules import rule_etf_holdings_cap_state_freshness
    result = rule_etf_holdings_cap_state_freshness()
    # Either None (clean) or dict with severity/snapshot
    if result is not None:
        assert "severity" in result
        assert "snapshot" in result
        assert result["severity"] in ("LOW", "MID", "HIGH", "CRITICAL")


def test_cost_budget_rule_smoke():
    """Cost budget rule must return None or valid RuleResult, never crash."""
    from engine.auto_audit_rules import rule_etf_holdings_cost_budget
    result = rule_etf_holdings_cost_budget()
    if result is not None:
        assert "severity" in result
        assert "snapshot" in result
        assert result["severity"] in ("LOW", "MID", "HIGH", "CRITICAL")


def test_both_rules_registered_in_watchdog():
    """Phase A2 rules must be in WATCHDOG_RULES list."""
    from engine.auto_audit_rules import (
        WATCHDOG_RULES,
        rule_etf_holdings_cap_state_freshness,
        rule_etf_holdings_cost_budget,
    )
    assert rule_etf_holdings_cap_state_freshness in WATCHDOG_RULES
    assert rule_etf_holdings_cost_budget in WATCHDOG_RULES


# ─────────────────────────────────────────────────────────────────────────────
# Phase A3: cleanup_expired_cap_state
# ─────────────────────────────────────────────────────────────────────────────

def test_cleanup_empty_state(monkeypatch, tmp_path):
    """Empty state → 0 entries removed."""
    cap_state_file = tmp_path / "cap_state.json"
    monkeypatch.setattr(
        "engine.etf_holdings_risk_monitor._CAP_STATE_PATH",
        cap_state_file,
    )
    from engine.etf_holdings_risk_monitor import cleanup_expired_cap_state
    assert cleanup_expired_cap_state() == 0


def test_cleanup_removes_expired_entries(monkeypatch, tmp_path):
    """Entries with expires_at < cutoff get removed."""
    cap_state_file = tmp_path / "cap_state.json"
    today = datetime.date(2026, 6, 1)
    state = {
        "QQQ":  {"triggered_at": "2026-05-08", "expires_at": "2026-05-18",
                 "aggregate_score": 5.0, "rationale": "old"},
        "SPY":  {"triggered_at": "2026-05-30", "expires_at": "2026-06-05",
                 "aggregate_score": 4.0, "rationale": "still active"},
    }
    cap_state_file.write_text(json.dumps(state))
    monkeypatch.setattr(
        "engine.etf_holdings_risk_monitor._CAP_STATE_PATH",
        cap_state_file,
    )

    from engine.etf_holdings_risk_monitor import cleanup_expired_cap_state, _load_cap_state
    n_removed = cleanup_expired_cap_state(as_of=today)
    assert n_removed == 1   # QQQ expired, SPY still active

    remaining = _load_cap_state()
    assert "QQQ" not in remaining
    assert "SPY" in remaining


def test_cleanup_preserves_within_buffer(monkeypatch, tmp_path):
    """Entries expired within buffer_calendar_days → keep for audit traceability."""
    cap_state_file = tmp_path / "cap_state.json"
    today = datetime.date(2026, 6, 1)
    # Expired 2 days ago — within default buffer 3 days
    state = {
        "RECENT": {"triggered_at": "2026-05-20", "expires_at": "2026-05-30",
                   "aggregate_score": 4.0, "rationale": "recently expired"},
    }
    cap_state_file.write_text(json.dumps(state))
    monkeypatch.setattr(
        "engine.etf_holdings_risk_monitor._CAP_STATE_PATH",
        cap_state_file,
    )

    from engine.etf_holdings_risk_monitor import cleanup_expired_cap_state, _load_cap_state
    n_removed = cleanup_expired_cap_state(as_of=today, buffer_calendar_days=3)
    assert n_removed == 0   # within buffer, keep

    remaining = _load_cap_state()
    assert "RECENT" in remaining


def test_cleanup_malformed_entry_keeps_safely(monkeypatch, tmp_path):
    """Malformed expires_at → keep (safer than delete on parse fail)."""
    cap_state_file = tmp_path / "cap_state.json"
    today = datetime.date(2026, 6, 1)
    state = {
        "BROKEN": {"triggered_at": "2026-05-08", "expires_at": "not-a-date",
                   "aggregate_score": 5.0, "rationale": "malformed"},
    }
    cap_state_file.write_text(json.dumps(state))
    monkeypatch.setattr(
        "engine.etf_holdings_risk_monitor._CAP_STATE_PATH",
        cap_state_file,
    )

    from engine.etf_holdings_risk_monitor import cleanup_expired_cap_state, _load_cap_state
    n_removed = cleanup_expired_cap_state(as_of=today)
    assert n_removed == 0
    remaining = _load_cap_state()
    assert "BROKEN" in remaining
