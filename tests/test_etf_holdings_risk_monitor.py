"""
tests/test_etf_holdings_risk_monitor.py — Sprint Week 2 main module tests.

Spec: docs/spec_etf_holdings_llm_risk_monitor.md (id=49)
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from engine import etf_holdings_risk_monitor as ehrm
from engine.etf_holdings_risk_monitor import (
    CAP_TRIGGER_THRESHOLD,
    HARD_CAP_MULTIPLIER,
    HARD_CAP_DURATION_DAYS,
    HARD_CAP_DURATION_CAP,
    HARD_CAP_FLOOR,
    HARD_CAP_UPPER,
    LLM_MODEL_VERSION,
    LLM_TEMPERATURE,
    ANNUAL_BUDGET_USD,
    PER_CALL_BUDGET_USD,
    LLM_OUTPUT_SCHEMA,
    BudgetExceeded,
    aggregate_etf_risk,
    trigger_etf_cap,
    apply_cap_to_max_weight,
    validate_and_classify_screening,
    _check_and_record_cost,
    _trailing_365d_total,
    _persist_cap_trigger,
    get_active_cap_state,
    get_cost_status,
    screen_name,
    build_prompt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants lock (spec §2.3, §2.7)
# ─────────────────────────────────────────────────────────────────────────────


def test_constants_locked_to_spec():
    """All locked constants must match spec §2.3 + §2.7."""
    assert CAP_TRIGGER_THRESHOLD == 3.5
    assert HARD_CAP_MULTIPLIER == 0.6
    assert HARD_CAP_DURATION_DAYS == 5
    assert HARD_CAP_DURATION_CAP == 10
    assert HARD_CAP_FLOOR == 0.5
    assert HARD_CAP_UPPER == 1.0
    assert LLM_MODEL_VERSION == "gemini-2.5-flash"
    assert LLM_TEMPERATURE == 0.0
    assert ANNUAL_BUDGET_USD == 120.0
    assert PER_CALL_BUDGET_USD == 0.10


def test_hard_cap_duration_within_cap():
    """Spec §六 forbidden: HARD_CAP_DURATION_DAYS ≤ HARD_CAP_DURATION_CAP."""
    assert HARD_CAP_DURATION_DAYS <= HARD_CAP_DURATION_CAP


def test_hard_cap_multiplier_one_way_defensive():
    """Spec §六 forbidden: multiplier ∈ [HARD_CAP_FLOOR, HARD_CAP_UPPER] = [0.5, 1.0]."""
    assert HARD_CAP_FLOOR <= HARD_CAP_MULTIPLIER <= HARD_CAP_UPPER


def test_llm_output_schema_required_fields():
    required = LLM_OUTPUT_SCHEMA["required"]
    assert "name" in required
    assert "risk_score" in required
    assert "event_class" in required
    assert "rationale" in required
    assert "as_of_date" in required


# ─────────────────────────────────────────────────────────────────────────────
# Validation + neutral fallback
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_valid_output():
    out = {
        "name": "AAPL",
        "risk_score": 3,
        "event_class": "earnings_warning",
        "rationale": "Q3 guidance reduced 8% on iPhone demand softness",
        "evidence_refs": ["8-K 2026-05-15"],
        "as_of_date": "2026-05-31",
    }
    result = validate_and_classify_screening(out)
    assert result["risk_score"] == 3
    assert result["event_class"] == "earnings_warning"
    assert result["fallback"] is False


def test_validate_invalid_risk_score_falls_back():
    out = {"name": "AAPL", "risk_score": 7, "event_class": "no_signal",
           "rationale": "x", "as_of_date": "2026-05-31"}
    result = validate_and_classify_screening(out)
    assert result["risk_score"] == 1
    assert result["fallback"] is True


def test_validate_invalid_event_class_falls_back():
    out = {"name": "AAPL", "risk_score": 3, "event_class": "made_up_class",
           "rationale": "x", "as_of_date": "2026-05-31"}
    result = validate_and_classify_screening(out)
    assert result["risk_score"] == 1
    assert result["fallback"] is True


def test_validate_non_dict_falls_back():
    assert validate_and_classify_screening("not a dict")["fallback"] is True
    assert validate_and_classify_screening(None)["fallback"] is True
    assert validate_and_classify_screening([1, 2, 3])["fallback"] is True


def test_validate_truncates_rationale_to_200():
    long_rationale = "x" * 500
    out = {"name": "AAPL", "risk_score": 2, "event_class": "no_signal",
           "rationale": long_rationale, "as_of_date": "2026-05-31"}
    result = validate_and_classify_screening(out)
    assert len(result["rationale"]) <= 200


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation truth table (spec §2.6)
# ─────────────────────────────────────────────────────────────────────────────


def test_aggregate_empty_holdings_returns_1():
    assert aggregate_etf_risk([], {}) == 1.0


def test_aggregate_zero_weights_returns_1():
    """Pathological: all zero weights → fallback to 1.0."""
    holdings = [{"name": "AAPL", "weight": 0.0, "rank": 1}]
    assert aggregate_etf_risk(holdings, {"AAPL": 5}) == 1.0


def test_aggregate_single_holding_score_3():
    holdings = [{"name": "AAPL", "weight": 0.5, "rank": 1}]
    score = aggregate_etf_risk(holdings, {"AAPL": 3})
    assert score == pytest.approx(3.0, abs=1e-9)


def test_aggregate_two_holdings_weighted():
    """Score = (0.6 × 4 + 0.4 × 2) / (0.6 + 0.4) = 3.2."""
    holdings = [
        {"name": "AAPL", "weight": 0.6, "rank": 1},
        {"name": "MSFT", "weight": 0.4, "rank": 2},
    ]
    score = aggregate_etf_risk(holdings, {"AAPL": 4, "MSFT": 2})
    assert score == pytest.approx(3.2, abs=1e-9)


def test_aggregate_uses_normalized_weights():
    """Non-normalized weights (sum != 1) → normalize within top 10."""
    holdings = [
        {"name": "AAPL", "weight": 0.30, "rank": 1},  # sum = 0.50
        {"name": "MSFT", "weight": 0.20, "rank": 2},
    ]
    score = aggregate_etf_risk(holdings, {"AAPL": 4, "MSFT": 2})
    # normalized: AAPL 0.6, MSFT 0.4 → score = 0.6*4 + 0.4*2 = 3.2
    assert score == pytest.approx(3.2, abs=1e-9)


def test_aggregate_missing_name_score_defaults_to_1():
    """LLM didn't screen this name → default risk_score = 1."""
    holdings = [
        {"name": "AAPL", "weight": 0.5, "rank": 1},
        {"name": "UNSEEN", "weight": 0.5, "rank": 2},
    ]
    score = aggregate_etf_risk(holdings, {"AAPL": 5})  # UNSEEN not in dict
    # AAPL 5 × 0.5 + UNSEEN 1 × 0.5 = 3.0
    assert score == pytest.approx(3.0, abs=1e-9)


def test_aggregate_clamps_to_5():
    """All score 5 → result = 5.0 (clamped)."""
    holdings = [
        {"name": "A", "weight": 0.5, "rank": 1},
        {"name": "B", "weight": 0.5, "rank": 2},
    ]
    score = aggregate_etf_risk(holdings, {"A": 5, "B": 5})
    assert score == 5.0


def test_aggregate_uppercase_normalization():
    """name keys are case-insensitive (uppercased before lookup)."""
    holdings = [{"name": "aapl", "weight": 1.0, "rank": 1}]
    score = aggregate_etf_risk(holdings, {"AAPL": 4})  # uppercase key in dict
    assert score == pytest.approx(4.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Trigger logic (spec §2.7)
# ─────────────────────────────────────────────────────────────────────────────


def test_trigger_below_threshold_false():
    assert trigger_etf_cap(3.49) is False


def test_trigger_at_threshold_true():
    assert trigger_etf_cap(3.5) is True


def test_trigger_above_threshold_true():
    assert trigger_etf_cap(4.2) is True


def test_trigger_max_score_true():
    assert trigger_etf_cap(5.0) is True


# ─────────────────────────────────────────────────────────────────────────────
# v2 amendment 2026-05-09 — max-of fallback trigger (hypothesis_amend +3 trials)
# ─────────────────────────────────────────────────────────────────────────────


def test_trigger_max_of_fallback_fires_on_severe_single_name():
    """Spec §2.7 v2 — single holding score=5 weight=10% → fallback fires."""
    holdings = [
        {"name": "AAPL", "weight": 0.10, "rank": 1},
        {"name": "MSFT", "weight": 0.05, "rank": 2},
        # 8 more with score 1, large total weight
        *[
            {"name": f"FILLER{i}", "weight": 0.05, "rank": i + 3}
            for i in range(8)
        ],
    ]
    name_scores = {"AAPL": 5, "MSFT": 1}  # AAPL severe but small relative weight
    # weighted-avg: (5×0.10 + 1×(0.05×9)) / 0.55 ≈ (0.5 + 0.45) / 0.55 ≈ 1.73 < 3.5
    # max-of fallback: AAPL score=5 ≥ 4.5 AND normalized weight 0.10/0.55 ≈ 0.18 ≥ 0.05 → fire
    weighted_avg = 1.73  # mock — would be computed by aggregate_etf_risk
    assert trigger_etf_cap(weighted_avg, holdings=holdings, name_scores=name_scores) is True


def test_trigger_max_of_fallback_skips_below_score_floor():
    """Single holding score=4 (below 4.5 SEVERE) → fallback does NOT fire."""
    holdings = [
        {"name": "AAPL", "weight": 0.10, "rank": 1},
        *[
            {"name": f"F{i}", "weight": 0.05, "rank": i + 2}
            for i in range(9)
        ],
    ]
    name_scores = {"AAPL": 4}  # below SEVERE threshold 4.5
    assert trigger_etf_cap(2.0, holdings=holdings, name_scores=name_scores) is False


def test_trigger_max_of_fallback_skips_below_weight_floor():
    """Single holding score=5 weight=2% (below 5% floor) → fallback does NOT fire."""
    holdings = [
        {"name": "AAPL", "weight": 0.02, "rank": 1},  # tiny weight
        *[
            {"name": f"F{i}", "weight": 0.05, "rank": i + 2}
            for i in range(9)
        ],
    ]
    name_scores = {"AAPL": 5}  # severe but trivial weight
    # Normalized AAPL weight = 0.02 / 0.47 ≈ 0.043 < 0.05 floor
    assert trigger_etf_cap(1.5, holdings=holdings, name_scores=name_scores) is False


def test_trigger_max_of_fallback_at_threshold_floor():
    """At exactly 5% normalized weight + score 4.5 → fires (boundary)."""
    holdings = [
        {"name": "X", "weight": 0.05, "rank": 1},
        *[
            {"name": f"F{i}", "weight": 0.10555, "rank": i + 2}  # 0.05 + 9×0.10555 = 1.0
            for i in range(9)
        ],
    ]
    name_scores = {"X": 5}  # 5 ≥ SEVERE_SINGLE_NAME_SCORE (4.5)
    # normalized weight 0.05 / 1.0 = 0.05 == floor → fire
    assert trigger_etf_cap(1.0, holdings=holdings, name_scores=name_scores) is True


def test_trigger_primary_takes_precedence_when_aggregate_high():
    """Aggregate ≥ 3.5 → primary fires regardless of fallback."""
    holdings = [{"name": "X", "weight": 1.0, "rank": 1}]
    name_scores = {"X": 1}  # low single score
    # primary: weighted_avg = 1.0 × 4.0 = 4.0 (mock value passed) ≥ 3.5 → fire
    assert trigger_etf_cap(4.0, holdings=holdings, name_scores=name_scores) is True


def test_trigger_backward_compat_no_holdings_args():
    """trigger_etf_cap(score) without holdings args — primary check only."""
    assert trigger_etf_cap(3.5) is True   # primary fires
    assert trigger_etf_cap(2.0) is False  # primary doesn't fire, no fallback args


def test_trigger_max_of_no_severe_names_no_fire():
    """All holdings score ≤ 3 → max-of cannot fire."""
    holdings = [{"name": f"X{i}", "weight": 0.10, "rank": i} for i in range(10)]
    name_scores = {f"X{i}": min(3, i + 1) for i in range(10)}
    assert trigger_etf_cap(2.0, holdings=holdings, name_scores=name_scores) is False


def test_trigger_max_of_severe_constant_lock():
    """Spec §2.7 v2 — SEVERE_SINGLE_NAME_SCORE locked at 4.5; WEIGHT_FLOOR at 0.05."""
    from engine.etf_holdings_risk_monitor import (
        SEVERE_SINGLE_NAME_SCORE,
        SEVERE_SINGLE_NAME_WEIGHT_FLOOR,
    )
    assert SEVERE_SINGLE_NAME_SCORE == 4.5
    assert SEVERE_SINGLE_NAME_WEIGHT_FLOOR == 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Cap application (spec §2.7)
# ─────────────────────────────────────────────────────────────────────────────


def test_apply_cap_inactive_returns_base():
    assert apply_cap_to_max_weight(0.25, cap_active=False, days_since_trigger=0) == 0.25
    assert apply_cap_to_max_weight(0.25, cap_active=False, days_since_trigger=10) == 0.25


def test_apply_cap_active_day_0():
    """Day 0 of cap → multiplier × base."""
    assert apply_cap_to_max_weight(0.25, cap_active=True, days_since_trigger=0) == pytest.approx(
        0.25 * HARD_CAP_MULTIPLIER  # 0.25 × 0.6 = 0.15
    )


def test_apply_cap_active_within_window():
    for d in range(HARD_CAP_DURATION_DAYS):
        result = apply_cap_to_max_weight(0.25, cap_active=True, days_since_trigger=d)
        assert result == pytest.approx(0.25 * HARD_CAP_MULTIPLIER)


def test_apply_cap_expired_returns_base():
    """Day ≥ duration → cap expired, return base."""
    assert apply_cap_to_max_weight(
        0.25, cap_active=True, days_since_trigger=HARD_CAP_DURATION_DAYS,
    ) == 0.25
    assert apply_cap_to_max_weight(
        0.25, cap_active=True, days_since_trigger=HARD_CAP_DURATION_DAYS + 5,
    ) == 0.25


def test_apply_cap_negative_days_returns_base():
    """Negative days_since (future trigger) → return base."""
    assert apply_cap_to_max_weight(
        0.25, cap_active=True, days_since_trigger=-1,
    ) == 0.25


def test_apply_cap_floor_clamp():
    """Defense-in-depth: even if multiplier somehow exceeds bounds, clamp to floor."""
    # Synthetic scenario: try to manually compute with extreme multiplier
    # apply_cap should never let result < HARD_CAP_FLOOR × base
    result = apply_cap_to_max_weight(0.25, cap_active=True, days_since_trigger=0)
    assert result >= HARD_CAP_FLOOR * 0.25


def test_apply_cap_upper_clamp():
    """Defense-in-depth: result never > HARD_CAP_UPPER × base."""
    result = apply_cap_to_max_weight(0.25, cap_active=True, days_since_trigger=0)
    assert result <= HARD_CAP_UPPER * 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Cost ledger
# ─────────────────────────────────────────────────────────────────────────────


def test_cost_ledger_per_call_cap_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._COST_LEDGER_PATH", tmp_path / "cost.json")
    with pytest.raises(BudgetExceeded, match="per_call"):
        _check_and_record_cost(0.15, as_of=datetime.date(2026, 5, 31))  # > 0.10 cap


def test_cost_ledger_annual_cap_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._COST_LEDGER_PATH", tmp_path / "cost.json")
    # Pre-load ledger with $119.99
    ledger = [{"date": "2026-05-31", "cost_usd": 119.99}]
    (tmp_path / "cost.json").write_text(json.dumps(ledger))
    with pytest.raises(BudgetExceeded, match="annual"):
        _check_and_record_cost(0.05, as_of=datetime.date(2026, 5, 31))  # 119.99 + 0.05 > 120


def test_cost_ledger_within_caps_records(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._COST_LEDGER_PATH", tmp_path / "cost.json")
    _check_and_record_cost(0.05, as_of=datetime.date(2026, 5, 31))
    _check_and_record_cost(0.05, as_of=datetime.date(2026, 5, 31))
    ledger = json.loads((tmp_path / "cost.json").read_text())
    assert len(ledger) == 2
    assert sum(e["cost_usd"] for e in ledger) == pytest.approx(0.10)


def test_trailing_365d_excludes_old_entries():
    ledger = [
        {"date": "2024-01-01", "cost_usd": 50.0},  # > 365d ago (from 2026-05-31)
        {"date": "2025-06-01", "cost_usd": 30.0},  # within 365d
        {"date": "2026-05-31", "cost_usd": 10.0},
    ]
    total = _trailing_365d_total(datetime.date(2026, 5, 31), ledger)
    # 2024-01-01 is 882 days before 2026-05-31, excluded
    # 2025-06-01 is 364 days before — borderline, included
    assert total == pytest.approx(40.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cap state severity-priority (spec §2.8)
# ─────────────────────────────────────────────────────────────────────────────


def test_cap_state_first_trigger_persists(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._CAP_STATE_PATH", tmp_path / "state.json")
    _persist_cap_trigger(
        etf="QQQ",
        triggered_at=datetime.date(2026, 5, 31),
        aggregate_score=4.2,
        rationale="test",
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert "QQQ" in state
    assert state["QQQ"]["aggregate_score"] == pytest.approx(4.2)


def test_cap_state_severity_priority_replaces_higher(tmp_path, monkeypatch):
    """新 score > 现 active → 替换."""
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._CAP_STATE_PATH", tmp_path / "state.json")
    _persist_cap_trigger("QQQ", datetime.date(2026, 5, 31), 3.6, "first")
    _persist_cap_trigger("QQQ", datetime.date(2026, 6, 2), 4.8, "second_higher")
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["QQQ"]["aggregate_score"] == pytest.approx(4.8)
    assert state["QQQ"]["triggered_at"] == "2026-06-02"


def test_cap_state_severity_priority_ignores_lower_when_active(tmp_path, monkeypatch):
    """新 score ≤ 现 active (still within window) → 忽略."""
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._CAP_STATE_PATH", tmp_path / "state.json")
    _persist_cap_trigger("QQQ", datetime.date(2026, 5, 31), 4.5, "first")
    # 2 trading days later, lower score
    _persist_cap_trigger("QQQ", datetime.date(2026, 6, 2), 3.6, "second_lower")
    state = json.loads((tmp_path / "state.json").read_text())
    # Original retained
    assert state["QQQ"]["aggregate_score"] == pytest.approx(4.5)
    assert state["QQQ"]["triggered_at"] == "2026-05-31"


def test_get_active_cap_state_filters_expired(tmp_path, monkeypatch):
    """Spec §2.8 — return only entries within HARD_CAP_DURATION_DAYS window."""
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._CAP_STATE_PATH", tmp_path / "state.json")
    state = {
        "QQQ": {  # 1 day ago, active
            "triggered_at":    "2026-05-31",
            "aggregate_score": 4.0,
            "expires_at":      "2026-06-09",
            "rationale":       "active",
        },
        "XLF": {  # 30 days ago, expired
            "triggered_at":    "2026-05-01",
            "aggregate_score": 3.7,
            "expires_at":      "2026-05-10",
            "rationale":       "expired",
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    active = get_active_cap_state(datetime.date(2026, 6, 1))
    assert "QQQ" in active
    assert "XLF" not in active


# ─────────────────────────────────────────────────────────────────────────────
# screen_name with skip_llm_call (deterministic test injection)
# ─────────────────────────────────────────────────────────────────────────────


def test_screen_name_skip_llm_requires_inject():
    with pytest.raises(ValueError, match="skip_llm_call=True requires inject_classification"):
        screen_name("AAPL", datetime.date(2026, 5, 31), skip_llm_call=True)


def test_screen_name_skip_llm_uses_inject():
    inject = {
        "name":       "AAPL",
        "risk_score": 4,
        "event_class": "earnings_warning",
        "rationale": "guidance reduction",
        "evidence_refs": ["8-K"],
        "as_of_date": "2026-05-31",
    }
    result = screen_name(
        "AAPL", datetime.date(2026, 5, 31),
        skip_llm_call=True, inject_classification=inject,
    )
    assert result["risk_score"] == 4
    assert result["event_class"] == "earnings_warning"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction (deterministic — same inputs, same prompt)
# ─────────────────────────────────────────────────────────────────────────────


def test_build_prompt_deterministic():
    args = {
        "name":               "AAPL",
        "as_of":              datetime.date(2026, 5, 31),
        "sector":             "Technology",
        "recent_8k_filings":  [{"date": "2026-05-15", "item": "2.02", "summary": "earnings"}],
        "recent_news":        [{"publish_date": "2026-05-20", "source": "Reuters", "title": "news"}],
        "price_30d_return":   -0.05,
        "next_earnings_date": datetime.date(2026, 7, 25),
    }
    p1 = build_prompt(**args)
    p2 = build_prompt(**args)
    assert p1 == p2  # deterministic


def test_build_prompt_includes_all_inputs():
    p = build_prompt(
        name="AAPL", as_of=datetime.date(2026, 5, 31),
        sector="Technology",
        recent_8k_filings=[],
        recent_news=[],
        price_30d_return=-0.05,
        next_earnings_date=None,
    )
    assert "AAPL" in p
    assert "Technology" in p
    assert "-5.00%" in p
    assert "RISK SCORE" in p  # system instructions present


def test_build_prompt_handles_none_optional_inputs():
    p = build_prompt(
        name="XYZ", as_of=datetime.date(2026, 5, 31),
        sector=None, recent_8k_filings=[], recent_news=[],
        price_30d_return=None, next_earnings_date=None,
    )
    assert "XYZ" in p
    assert "n/a" in p  # null sector + null earnings + null price → "n/a"


# ─────────────────────────────────────────────────────────────────────────────
# Cost status reporting
# ─────────────────────────────────────────────────────────────────────────────


def test_get_cost_status_empty_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._COST_LEDGER_PATH", tmp_path / "cost.json")
    status = get_cost_status(datetime.date(2026, 5, 31))
    assert status["trailing_365d_total_usd"] == 0.0
    assert status["annual_cap_usd"] == ANNUAL_BUDGET_USD
    assert status["fraction_of_annual_cap"] == 0.0


def test_get_cost_status_partial_burn(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.etf_holdings_risk_monitor._COST_LEDGER_PATH", tmp_path / "cost.json")
    ledger = [{"date": "2026-05-31", "cost_usd": 30.0}]
    (tmp_path / "cost.json").write_text(json.dumps(ledger))
    status = get_cost_status(datetime.date(2026, 5, 31))
    assert status["trailing_365d_total_usd"] == pytest.approx(30.0)
    assert status["fraction_of_annual_cap"] == pytest.approx(0.25, abs=1e-3)
