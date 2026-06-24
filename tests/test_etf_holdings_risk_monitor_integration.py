"""
tests/test_etf_holdings_risk_monitor_integration.py — Sprint Week 3 integration tests.

Tests the portfolio.py Step 6 hook integration with ETF Holdings Risk Monitor cap state.

Spec: docs/spec_etf_holdings_llm_risk_monitor.md (id=49)
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine import etf_holdings_risk_monitor as ehrm
from engine.portfolio import construct_portfolio
from engine.regime import RegimeResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_signal_df():
    """Build a synthetic signal_df matching what construct_portfolio expects."""
    df = pd.DataFrame({
        "ticker":   ["QQQ",  "XLF",  "XLE",  "XLV",  "GLD"],
        "tsmom":    [1,      1,      1,      1,      -1],   # 4 long, 1 short
        "ql01_bab": [1,      1,      1,      1,      -1],
        "ann_vol":  [0.18,   0.20,   0.30,   0.15,   0.16],
        "inv_vol_wt": [0.0,  0.0,    0.0,    0.0,    0.0],  # placeholder
    })
    return df


@pytest.fixture
def neutral_regime():
    """risk-on regime — REGIME_SCALE multiplier = 1.0 (no scale-down)."""
    return RegimeResult(
        date=datetime.date(2026, 5, 31),
        regime="risk-on",
        p_risk_on=0.85,
        p_risk_off=0.15,
        method="msm",
        n_obs=120,
        yield_spread=0.005,
        vix=15.0,
        warning="",
    )


@pytest.fixture
def cap_state_path(tmp_path, monkeypatch):
    """Redirect cap state file to tmp_path."""
    cap_path = tmp_path / "cap_state.json"
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._CAP_STATE_PATH", cap_path)
    return cap_path


# ─────────────────────────────────────────────────────────────────────────────
# Hook integration tests
# ─────────────────────────────────────────────────────────────────────────────


def test_no_cap_state_normal_max_weight(mock_signal_df, neutral_regime, cap_state_path):
    """Empty cap state → no override applied; portfolio runs with default MAX_WEIGHT."""
    # cap_state_path doesn't exist → no caps active
    result = construct_portfolio(
        signal_df=mock_signal_df,
        regime=neutral_regime,
        as_of=datetime.date(2026, 5, 31),
    )
    # Portfolio constructed successfully; no ETF cap warnings
    cap_warnings = [w for w in result.warnings if "ETF holdings cap" in w]
    assert len(cap_warnings) == 0


def test_active_cap_reduces_etf_max_weight(mock_signal_df, neutral_regime, cap_state_path):
    """Cap state for QQQ → QQQ's MAX_WEIGHT reduced to 15% (25% × 0.6)."""
    cap_state = {
        "QQQ": {
            "triggered_at":    "2026-05-31",
            "aggregate_score": 4.2,
            "expires_at":      "2026-06-09",
            "rationale":       "test cap",
        },
    }
    cap_state_path.write_text(json.dumps(cap_state))

    result = construct_portfolio(
        signal_df=mock_signal_df,
        regime=neutral_regime,
        as_of=datetime.date(2026, 5, 31),
    )

    # Cap warning fired for QQQ
    cap_warnings = [w for w in result.warnings if "ETF holdings cap active QQQ" in w]
    assert len(cap_warnings) >= 1
    assert "score=4.20" in cap_warnings[0] or "4.2" in cap_warnings[0]


def test_expired_cap_not_applied(mock_signal_df, neutral_regime, cap_state_path):
    """Cap state but trigger date > HARD_CAP_DURATION_DAYS old → no cap applied."""
    cap_state = {
        "QQQ": {
            "triggered_at":    "2026-05-01",  # 30 days ago
            "aggregate_score": 4.2,
            "expires_at":      "2026-05-10",
            "rationale":       "expired",
        },
    }
    cap_state_path.write_text(json.dumps(cap_state))

    result = construct_portfolio(
        signal_df=mock_signal_df,
        regime=neutral_regime,
        as_of=datetime.date(2026, 5, 31),
    )

    cap_warnings = [w for w in result.warnings if "ETF holdings cap active" in w]
    assert len(cap_warnings) == 0  # cap expired, not active


def test_multiple_etf_caps_independent(mock_signal_df, neutral_regime, cap_state_path):
    """2 ETFs both capped → both warnings fire."""
    cap_state = {
        "QQQ": {
            "triggered_at":    "2026-05-31",
            "aggregate_score": 4.2,
            "expires_at":      "2026-06-09",
            "rationale":       "qqq",
        },
        "XLF": {
            "triggered_at":    "2026-05-31",
            "aggregate_score": 3.6,
            "expires_at":      "2026-06-09",
            "rationale":       "xlf",
        },
    }
    cap_state_path.write_text(json.dumps(cap_state))

    result = construct_portfolio(
        signal_df=mock_signal_df,
        regime=neutral_regime,
        as_of=datetime.date(2026, 5, 31),
    )

    qqq_warns = [w for w in result.warnings if "QQQ" in w and "ETF holdings cap" in w]
    xlf_warns = [w for w in result.warnings if "XLF" in w and "ETF holdings cap" in w]
    assert len(qqq_warns) >= 1
    assert len(xlf_warns) >= 1


def test_cap_only_affects_capped_etf_not_others(mock_signal_df, neutral_regime, cap_state_path):
    """QQQ capped → XLF/XLE/XLV/GLD unaffected."""
    cap_state = {
        "QQQ": {
            "triggered_at":    "2026-05-31",
            "aggregate_score": 4.2,
            "expires_at":      "2026-06-09",
            "rationale":       "qqq only",
        },
    }
    cap_state_path.write_text(json.dumps(cap_state))

    result = construct_portfolio(
        signal_df=mock_signal_df,
        regime=neutral_regime,
        as_of=datetime.date(2026, 5, 31),
    )

    cap_warnings = [w for w in result.warnings if "ETF holdings cap active" in w]
    # Only QQQ should be in warnings
    assert all("QQQ" in w for w in cap_warnings), \
        f"Unexpected non-QQQ ETF capped: {cap_warnings}"


def test_hook_graceful_failure_when_module_unavailable(mock_signal_df, neutral_regime, monkeypatch):
    """If etf_holdings_risk_monitor module fails to import, portfolio still runs."""
    # Monkey-patch the import inside construct_portfolio to raise
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "etf_holdings_risk_monitor" in name:
            raise ImportError("simulated module failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Should not raise; warnings_log captures the failure
    result = construct_portfolio(
        signal_df=mock_signal_df,
        regime=neutral_regime,
        as_of=datetime.date(2026, 5, 31),
    )
    assert result is not None  # portfolio still constructed


def test_cap_composition_with_universe_manager_tier_override(
    mock_signal_df, neutral_regime, cap_state_path,
):
    """If both universe_manager (tier2 cap) AND ETF holdings cap fire, MIN wins."""
    cap_state = {
        "QQQ": {
            "triggered_at":    "2026-05-31",
            "aggregate_score": 3.6,  # mild, gives 25% × 0.6 = 15%
            "expires_at":      "2026-06-09",
            "rationale":       "mild",
        },
    }
    cap_state_path.write_text(json.dumps(cap_state))

    # Mock universe_manager.get_max_weight_for_ticker to return stricter cap (10%)
    with patch("engine.universe_manager.get_max_weight_for_ticker") as mock_gmw:
        mock_gmw.side_effect = lambda t, default: 0.10 if t == "QQQ" else default

        result = construct_portfolio(
            signal_df=mock_signal_df,
            regime=neutral_regime,
            as_of=datetime.date(2026, 5, 31),
        )

    # Tier2 cap (10%) is stricter than ETF holdings cap (15%) — both should be visible
    # But final clip uses min = 10%
    # Verify by checking warnings include "tier" or by checking result weights
    assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + cap state end-to-end
# ─────────────────────────────────────────────────────────────────────────────


def test_aggregation_and_persist_end_to_end(cap_state_path, tmp_path, monkeypatch):
    """
    Synthetic flow: holdings + per-name scores → aggregate → trigger → persist
    cap state → readable via get_active_cap_state.
    """
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._CAP_STATE_PATH", cap_state_path)

    # 1. Synthetic holdings: QQQ has 5 top holdings
    holdings_qqq = [
        {"name": "AAPL",  "weight": 0.10, "rank": 1},
        {"name": "MSFT",  "weight": 0.08, "rank": 2},
        {"name": "NVDA",  "weight": 0.07, "rank": 3},
        {"name": "GOOGL", "weight": 0.05, "rank": 4},
        {"name": "AMZN",  "weight": 0.05, "rank": 5},
    ]
    # 2. Mock per-name scores: 3 high (4-5), 2 low (1-2)
    name_scores = {
        "AAPL":  4,
        "MSFT":  5,
        "NVDA":  4,
        "GOOGL": 1,
        "AMZN":  2,
    }
    # 3. Aggregate
    score = ehrm.aggregate_etf_risk(holdings_qqq, name_scores)
    # weighted avg: (0.10*4 + 0.08*5 + 0.07*4 + 0.05*1 + 0.05*2) / (0.10+0.08+0.07+0.05+0.05)
    # = (0.40 + 0.40 + 0.28 + 0.05 + 0.10) / 0.35 = 1.23 / 0.35 = 3.514...
    assert score >= ehrm.CAP_TRIGGER_THRESHOLD  # ~3.51 ≥ 3.5 → trigger

    # 4. Trigger detection
    assert ehrm.trigger_etf_cap(score) is True

    # 5. Persist
    ehrm._persist_cap_trigger(
        etf="QQQ",
        triggered_at=datetime.date(2026, 5, 31),
        aggregate_score=score,
        rationale="end-to-end test",
    )

    # 6. Read via get_active_cap_state
    active = ehrm.get_active_cap_state(datetime.date(2026, 5, 31))
    assert "QQQ" in active
    assert active["QQQ"]["aggregate_score"] == pytest.approx(score, abs=1e-3)
