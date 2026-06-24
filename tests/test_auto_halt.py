"""Unit tests for engine.agents.anomaly_sentinel.auto_halt.

Critical safety properties:
1. Each trigger fires correctly on threshold breach
2. Each trigger does NOT fire on under-threshold input
3. Fail-safe behavior: missing data triggers (or defaults to safe state)
4. Halt flag is written ONLY on halt=True
5. Halt acknowledgement preserves audit trail
6. is_halt_active() correctly reads + interprets the flag
7. Insufficient history → trigger silently no-fire (avoid spurious halts at startup)
"""
from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from engine.agents.anomaly_sentinel.auto_halt import (
    TRIGGER_SHARPE_60D,
    TRIGGER_TICKER_MAXDD_30D,
    TRIGGER_VOL_MULTIPLE,
    TRIGGER_VOL_TARGET,
    TRIGGER_WEIGHT_ABS,
    acknowledge_halt,
    clear_halt_flag,
    evaluate,
    is_halt_active,
    read_halt_flag,
    write_halt_flag,
    _t1_book_sharpe_60d,
    _t2_book_vol_21d,
    _t3_ticker_maxdd_30d,
    _t4_position_concentration,
    _t5_nav_stale,
)


# ── Trigger T1 — book Sharpe 60d ──────────────────────────────────────────────

def test_t1_fires_on_negative_sharpe():
    """100 days of -2% daily returns → very negative Sharpe → T1 fires."""
    nav = pd.Series(np.cumprod(1 - 0.02 * np.ones(100)) * 100)
    nav.index = pd.date_range("2026-01-01", periods=100, freq="B")
    result = _t1_book_sharpe_60d(nav)
    assert result.fired
    assert result.metric < TRIGGER_SHARPE_60D


def test_t1_no_fire_on_positive_sharpe():
    """100 days of +0.5% daily returns → strongly positive Sharpe → no fire."""
    rng = np.random.default_rng(seed=42)
    daily = 0.005 + 0.001 * rng.standard_normal(100)
    nav = pd.Series(np.cumprod(1 + daily) * 100)
    nav.index = pd.date_range("2026-01-01", periods=100, freq="B")
    result = _t1_book_sharpe_60d(nav)
    assert not result.fired


def test_t1_silent_no_fire_on_insufficient_history():
    """30 days NAV → not enough for 60d window → no fire, not error."""
    nav = pd.Series([100.0] * 30, index=pd.date_range("2026-01-01", periods=30, freq="B"))
    result = _t1_book_sharpe_60d(nav)
    assert not result.fired
    assert result.metric is None


def test_t1_handles_none_nav():
    result = _t1_book_sharpe_60d(None)
    assert not result.fired


# ── Trigger T2 — book vol 21d ─────────────────────────────────────────────────

def test_t2_fires_on_excess_vol():
    """Highly volatile NAV → 21d realized vol exceeds 1.5× target → fires."""
    rng = np.random.default_rng(seed=7)
    # daily vol ~3% → annualized ~47% → way above 1.5×10%=15%
    daily = 0.03 * rng.standard_normal(30)
    nav = pd.Series(np.cumprod(1 + daily) * 100)
    nav.index = pd.date_range("2026-01-01", periods=30, freq="B")
    result = _t2_book_vol_21d(nav)
    assert result.fired
    assert result.metric > TRIGGER_VOL_MULTIPLE * TRIGGER_VOL_TARGET


def test_t2_no_fire_on_calm_vol():
    """Low-vol NAV → no fire."""
    rng = np.random.default_rng(seed=7)
    daily = 0.005 * rng.standard_normal(30)
    nav = pd.Series(np.cumprod(1 + daily) * 100)
    nav.index = pd.date_range("2026-01-01", periods=30, freq="B")
    result = _t2_book_vol_21d(nav)
    assert not result.fired


# ── Trigger T3 — ticker MaxDD 30d ─────────────────────────────────────────────

def test_t3_fires_on_deep_drawdown():
    """One ticker drops 40% in 30 days → T3 fires."""
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    p = pd.DataFrame({
        "OK":   [100.0] * 30,
        "TANK": np.linspace(100, 60, 30),     # -40% drawdown
    }, index=idx)
    result = _t3_ticker_maxdd_30d(p)
    assert result.fired
    assert result.evidence["worst_ticker"] == "TANK"


def test_t3_no_fire_on_shallow_drawdown():
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    p = pd.DataFrame({
        "OK":    [100.0] * 30,
        "DIP":   np.linspace(100, 85, 30),    # -15% drawdown (above -25% bar)
    }, index=idx)
    result = _t3_ticker_maxdd_30d(p)
    assert not result.fired


def test_t3_no_fire_on_empty_panel():
    result = _t3_ticker_maxdd_30d(pd.DataFrame())
    assert not result.fired


# ── Trigger T4 — position concentration ───────────────────────────────────────

def test_t4_fires_on_oversized_position():
    weights = {"A": 0.40, "B": 0.20, "C": -0.10}    # A at 40% > 30% bar
    result = _t4_position_concentration(weights)
    assert result.fired
    assert result.evidence["max_ticker"] == "A"
    assert "A" in result.evidence["breached"]


def test_t4_fires_on_short_position_too_large():
    weights = {"A": 0.10, "B": -0.35}    # |B|=0.35 > 0.30 bar
    result = _t4_position_concentration(weights)
    assert result.fired
    assert result.evidence["max_ticker"] == "B"


def test_t4_no_fire_on_diversified():
    weights = {"A": 0.20, "B": 0.20, "C": 0.20, "D": -0.15}
    result = _t4_position_concentration(weights)
    assert not result.fired


def test_t4_no_fire_on_empty():
    result = _t4_position_concentration({})
    assert not result.fired
    result = _t4_position_concentration(None)
    assert not result.fired


# ── Trigger T5 — NAV stale ────────────────────────────────────────────────────

def test_t5_fires_on_stale_file():
    now = datetime.datetime(2026, 5, 29, 12, 0, 0).timestamp()
    stale_mtime = now - 72 * 3600    # 72h ago > 48h threshold
    result = _t5_nav_stale(stale_mtime, now)
    assert result.fired
    assert result.metric > 48.0


def test_t5_no_fire_on_recent_file():
    now = datetime.datetime(2026, 5, 29, 12, 0, 0).timestamp()
    fresh_mtime = now - 6 * 3600    # 6h ago
    result = _t5_nav_stale(fresh_mtime, now)
    assert not result.fired


def test_t5_fires_on_missing_file_failsafe():
    """Fail-safe: missing file → HALT, not silently allow."""
    now = datetime.datetime(2026, 5, 29).timestamp()
    result = _t5_nav_stale(None, now)
    assert result.fired      # fail-safe = halt


# ── Orchestrator evaluate() ──────────────────────────────────────────────────

def test_evaluate_no_halt_when_all_clear(tmp_path, monkeypatch):
    """Healthy book → halt=False, triggers_fired empty."""
    rng = np.random.default_rng(seed=42)
    daily = 0.0005 + 0.005 * rng.standard_normal(100)
    nav = pd.Series(np.cumprod(1 + daily) * 100)
    nav.index = pd.date_range("2026-01-01", periods=100, freq="B")
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    prices = pd.DataFrame({"A": [100] * 30, "B": [100] * 30}, index=idx)
    weights = {"A": 0.10, "B": 0.10}
    now = datetime.datetime(2026, 5, 29).timestamp()
    nav_mtime = now - 3600    # 1 hour ago
    decision = evaluate(nav, prices, weights, nav_mtime, now)
    assert decision.halt is False
    assert decision.triggers_fired == []


def test_evaluate_halts_when_trigger_fires(tmp_path, monkeypatch):
    """Any single trigger → halt=True."""
    nav = pd.Series(np.cumprod(1 - 0.02 * np.ones(100)) * 100)  # -2%/day → T1 fires
    nav.index = pd.date_range("2026-01-01", periods=100, freq="B")
    now = datetime.datetime(2026, 5, 29).timestamp()
    decision = evaluate(nav, None, {}, now - 3600, now)
    assert decision.halt
    assert "T1_book_sharpe_60d" in decision.triggers_fired


# ── Halt flag I/O ────────────────────────────────────────────────────────────

def test_write_halt_flag_only_on_halt(tmp_path, monkeypatch):
    import engine.agents.anomaly_sentinel.auto_halt as m
    monkeypatch.setattr(m, "HALT_FLAG_PATH", tmp_path / "halt.json")

    # No-halt decision → no file written
    from engine.agents.anomaly_sentinel.auto_halt import HaltDecision
    dec_ok = HaltDecision(halt=False, triggers_fired=[], triggers_all=[],
                           as_of="2026-05-29T00:00:00Z",
                           suggested_action="all clear")
    write_halt_flag(dec_ok)
    assert not (tmp_path / "halt.json").exists()

    # Halt decision → file written
    dec_halt = HaltDecision(halt=True, triggers_fired=["T1"], triggers_all=[],
                             as_of="2026-05-29T00:00:00Z",
                             suggested_action="halt now")
    write_halt_flag(dec_halt)
    assert (tmp_path / "halt.json").exists()
    payload = json.loads((tmp_path / "halt.json").read_text(encoding="utf-8"))
    assert payload["halt"] is True
    assert "T1" in payload["triggers_fired"]


def test_is_halt_active_no_file(tmp_path, monkeypatch):
    import engine.agents.anomaly_sentinel.auto_halt as m
    monkeypatch.setattr(m, "HALT_FLAG_PATH", tmp_path / "halt.json")
    active, payload = is_halt_active()
    assert active is False
    assert payload is None


def test_is_halt_active_with_active_halt(tmp_path, monkeypatch):
    import engine.agents.anomaly_sentinel.auto_halt as m
    halt_path = tmp_path / "halt.json"
    monkeypatch.setattr(m, "HALT_FLAG_PATH", halt_path)
    halt_path.write_text(json.dumps({"halt": True, "triggers_fired": ["T1"]}),
                         encoding="utf-8")
    active, payload = is_halt_active()
    assert active is True
    assert payload["triggers_fired"] == ["T1"]


def test_is_halt_active_after_acknowledgement(tmp_path, monkeypatch):
    import engine.agents.anomaly_sentinel.auto_halt as m
    halt_path = tmp_path / "halt.json"
    monkeypatch.setattr(m, "HALT_FLAG_PATH", halt_path)
    halt_path.write_text(json.dumps({
        "halt": True, "triggers_fired": ["T1"],
        "acknowledged_by_human_at": "2026-05-29T12:00:00Z"
    }), encoding="utf-8")
    active, payload = is_halt_active()
    assert active is False    # acknowledged → not active
    assert payload["acknowledged_by_human_at"] == "2026-05-29T12:00:00Z"


def test_acknowledge_halt_preserves_audit_trail(tmp_path, monkeypatch):
    import engine.agents.anomaly_sentinel.auto_halt as m
    halt_path = tmp_path / "halt.json"
    monkeypatch.setattr(m, "HALT_FLAG_PATH", halt_path)
    halt_path.write_text(json.dumps({"halt": True, "triggers_fired": ["T1"]}),
                         encoding="utf-8")
    payload = acknowledge_halt(by_user="alice", note="investigated, restart OK")
    assert payload["acknowledged_by"] == "alice"
    assert "acknowledged_by_human_at" in payload
    assert payload["acknowledgement_note"] == "investigated, restart OK"
    # File persisted
    on_disk = json.loads(halt_path.read_text(encoding="utf-8"))
    assert on_disk["acknowledged_by"] == "alice"


def test_acknowledge_halt_raises_when_no_active_halt(tmp_path, monkeypatch):
    import engine.agents.anomaly_sentinel.auto_halt as m
    monkeypatch.setattr(m, "HALT_FLAG_PATH", tmp_path / "halt.json")
    with pytest.raises(RuntimeError, match="No active halt"):
        acknowledge_halt()


def test_fail_safe_on_unreadable_flag(tmp_path, monkeypatch):
    """Corrupted halt_flag.json → treated as ACTIVE halt, not silently allow."""
    import engine.agents.anomaly_sentinel.auto_halt as m
    halt_path = tmp_path / "halt.json"
    monkeypatch.setattr(m, "HALT_FLAG_PATH", halt_path)
    halt_path.write_text("{ invalid json", encoding="utf-8")
    payload = read_halt_flag()
    assert payload["halt"] is True       # fail-safe
    assert "error" in payload


# ── End-to-end shape ─────────────────────────────────────────────────────────

def test_evaluate_returns_well_formed_decision():
    decision = evaluate(None, None, None, None,
                         datetime.datetime(2026, 5, 29).timestamp())
    obj = decision.to_jsonable()
    assert "halt" in obj
    assert "triggers_fired" in obj
    assert "triggers_all" in obj
    assert "as_of" in obj
    assert "suggested_action" in obj
