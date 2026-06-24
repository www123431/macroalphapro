"""
tests/test_fomc_surprise_override.py — Unit tests for engine.fomc_surprise_override
(spec id=48, hash 036b2805f0d6).

Coverage scope (per spec §五 validation gates):
  - is_fomc_day calendar truth
  - validate_and_classify schema enforcement + fallback
  - trigger_emergency_override AND-gate truth table (3 labels × 4 regimes = 12)
  - apply_override_to_regime_scale clamp + duration window
  - get_active_override_state disk read + expiry
  - _trading_days_elapsed weekend handling

Out of scope (deferred to integration tests):
  - Real LLM call (would charge $)
  - End-to-end process_fomc_day (touches FRED + WRDS-style fetchers)
  - portfolio.py integration (covered by existing portfolio tests + hook audit)
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from engine import fomc_surprise_override as fso


# ── is_fomc_day ─────────────────────────────────────────────────────────────
def test_is_fomc_day_returns_true_for_known_date() -> None:
    """At least one known FOMC date must round-trip True."""
    dates = fso._get_fomc_dates()
    assert len(dates) > 0, "FOMC calendar empty — fix engine.decision_context._FOMC_DATES_2024_2026"
    sample = dates[0]
    assert fso.is_fomc_day(sample) is True


def test_is_fomc_day_returns_false_for_non_fomc() -> None:
    """A random weekday clearly not in calendar must be False."""
    assert fso.is_fomc_day(datetime.date(2024, 7, 4)) is False  # July 4 holiday


def test_is_fomc_day_rejects_non_date_input() -> None:
    with pytest.raises(TypeError):
        fso.is_fomc_day("2024-01-31")  # type: ignore[arg-type]


# ── validate_and_classify ───────────────────────────────────────────────────
def test_validate_classify_accepts_well_formed_normal() -> None:
    out = fso.validate_and_classify({
        "surprise_label": "NORMAL",
        "direction":      "neutral",
        "rationale":      "rate decision aligned with prior guidance",
        "confidence":     3,
    })
    assert out["surprise_label"] == "NORMAL"
    assert out["fallback"] is False


def test_validate_classify_accepts_extreme_surprise() -> None:
    out = fso.validate_and_classify({
        "surprise_label": "EXTREME_SURPRISE",
        "direction":      "hawkish",
        "rationale":      "75bp emergency hike outside scheduled meeting",
        "confidence":     5,
    })
    assert out["surprise_label"] == "EXTREME_SURPRISE"
    assert out["fallback"] is False


def test_validate_classify_fallback_on_invalid_label() -> None:
    out = fso.validate_and_classify({
        "surprise_label": "PANIC",  # not in enum
        "direction":      "neutral",
        "rationale":      "x",
        "confidence":     3,
    })
    assert out["surprise_label"] == "NORMAL"
    assert out["fallback"] is True


def test_validate_classify_fallback_on_missing_field() -> None:
    out = fso.validate_and_classify({
        "surprise_label": "MILD_SURPRISE",
        # direction missing
        "rationale":      "x",
        "confidence":     2,
    })
    assert out["fallback"] is True


def test_validate_classify_fallback_on_non_dict() -> None:
    out = fso.validate_and_classify("not a dict")  # type: ignore[arg-type]
    assert out["fallback"] is True
    assert out["surprise_label"] == "NORMAL"


def test_validate_classify_rationale_truncated_to_300() -> None:
    long_rat = "x" * 500
    out = fso.validate_and_classify({
        "surprise_label": "NORMAL",
        "direction":      "neutral",
        "rationale":      long_rat,
        "confidence":     3,
    })
    assert len(out["rationale"]) == 300


# ── trigger_emergency_override truth table (12 cells) ───────────────────────
@pytest.mark.parametrize("label,regime,expected", [
    # EXTREME_SURPRISE × 4 regimes
    ("EXTREME_SURPRISE", "risk-off",   True),   # AND-gate fires
    ("EXTREME_SURPRISE", "transition", True),   # AND-gate fires
    ("EXTREME_SURPRISE", "risk-on",    False),  # quant veto
    ("EXTREME_SURPRISE", "unknown",    False),  # safety: unknown regime → no override
    # MILD_SURPRISE × 4 regimes (none fire)
    ("MILD_SURPRISE",    "risk-off",   False),
    ("MILD_SURPRISE",    "transition", False),
    ("MILD_SURPRISE",    "risk-on",    False),
    ("MILD_SURPRISE",    "unknown",    False),
    # NORMAL × 4 regimes (none fire)
    ("NORMAL",           "risk-off",   False),
    ("NORMAL",           "transition", False),
    ("NORMAL",           "risk-on",    False),
    ("NORMAL",           "unknown",    False),
])
def test_trigger_truth_table(label: str, regime: str, expected: bool) -> None:
    assert fso.trigger_emergency_override(label, regime) is expected


# ── apply_override_to_regime_scale ──────────────────────────────────────────
def test_apply_override_no_trigger_returns_base() -> None:
    """triggered_at=None → passthrough."""
    out = fso.apply_override_to_regime_scale(
        base_scale=0.6, triggered_at=None, as_of=datetime.date(2024, 1, 31)
    )
    assert out == 0.6


def test_apply_override_active_window_applies_multiplier() -> None:
    """Day 0 (trigger day): multiplier applied. 0.6 × 0.5 = 0.3."""
    out = fso.apply_override_to_regime_scale(
        base_scale=0.6,
        triggered_at=datetime.date(2024, 1, 31),  # Wed
        as_of=datetime.date(2024, 1, 31),
    )
    assert out == pytest.approx(0.3)


def test_apply_override_day_4_still_active() -> None:
    """Days 1-4 trading days after trigger: still active (HARD_DURATION_DAYS=5)."""
    out = fso.apply_override_to_regime_scale(
        base_scale=0.6,
        triggered_at=datetime.date(2024, 1, 31),  # Wed
        as_of=datetime.date(2024, 2, 6),          # Tue (4 trading days later)
    )
    assert out == pytest.approx(0.3)


def test_apply_override_day_5_expired() -> None:
    """Day 5 trading day: override expired, base returned."""
    out = fso.apply_override_to_regime_scale(
        base_scale=0.6,
        triggered_at=datetime.date(2024, 1, 31),  # Wed
        as_of=datetime.date(2024, 2, 7),          # Wed next week (5 trading days)
    )
    assert out == 0.6


def test_apply_override_clamp_lower_bound() -> None:
    """Multiplier × base must not drop below HARD_MULTIPLIER_LOWER × base."""
    # HARD_OVERRIDE_MULTIPLIER=0.5 = HARD_MULTIPLIER_LOWER so clamp is no-op here,
    # but verify clamp lower bound math holds.
    out = fso.apply_override_to_regime_scale(
        base_scale=1.0,
        triggered_at=datetime.date(2024, 1, 31),
        as_of=datetime.date(2024, 1, 31),
    )
    assert out >= fso.HARD_MULTIPLIER_LOWER * 1.0


# ── _trading_days_elapsed weekend handling ─────────────────────────────────
def test_trading_days_elapsed_skips_weekend() -> None:
    """Fri → Mon should be 1 trading day, not 3 calendar days."""
    fri = datetime.date(2024, 2, 2)   # Friday
    mon = datetime.date(2024, 2, 5)   # Monday
    assert fso._trading_days_elapsed(fri, mon) == 1


def test_trading_days_elapsed_same_day_zero() -> None:
    d = datetime.date(2024, 1, 31)
    assert fso._trading_days_elapsed(d, d) == 0


def test_trading_days_elapsed_start_after_current_returns_zero() -> None:
    """Defensive: if start > current, return 0 (not negative)."""
    assert fso._trading_days_elapsed(
        datetime.date(2024, 2, 5),
        datetime.date(2024, 1, 31),
    ) == 0


# ── get_active_override_state ───────────────────────────────────────────────
def test_get_active_state_returns_inactive_when_no_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No override_state.json on disk → triggered_at=None."""
    monkeypatch.setattr(fso, "_OVERRIDE_STATE_PATH", tmp_path / "override_state.json")
    state = fso.get_active_override_state(datetime.date(2024, 1, 31))
    assert state.triggered_at is None
    assert state.days_remaining == 0


def test_get_active_state_returns_active_within_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Override triggered yesterday → active, days_remaining > 0."""
    path = tmp_path / "override_state.json"
    monkeypatch.setattr(fso, "_OVERRIDE_STATE_PATH", path)
    triggered = datetime.date(2024, 1, 31)  # Wed
    path.write_text(json.dumps({
        "triggered_at":   triggered.isoformat(),
        "fomc_date":      triggered.isoformat(),
        "surprise_label": "EXTREME_SURPRISE",
        "direction":      "hawkish",
    }), encoding="utf-8")
    state = fso.get_active_override_state(datetime.date(2024, 2, 1))  # Thu, day 1
    assert state.triggered_at == triggered
    assert state.surprise_label == "EXTREME_SURPRISE"
    assert state.days_remaining > 0
    assert state.multiplier_applied == fso.HARD_OVERRIDE_MULTIPLIER


def test_get_active_state_returns_inactive_when_expired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Override triggered 6 trading days ago → expired, triggered_at=None."""
    path = tmp_path / "override_state.json"
    monkeypatch.setattr(fso, "_OVERRIDE_STATE_PATH", path)
    triggered = datetime.date(2024, 1, 31)  # Wed
    path.write_text(json.dumps({
        "triggered_at":   triggered.isoformat(),
        "surprise_label": "EXTREME_SURPRISE",
    }), encoding="utf-8")
    # 6 trading days later: Wed Jan 31 → Thu Feb 8
    state = fso.get_active_override_state(datetime.date(2024, 2, 8))
    assert state.triggered_at is None
    assert state.days_remaining == 0


def test_get_active_state_handles_corrupt_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Malformed JSON → defaults to inactive (defensive)."""
    path = tmp_path / "override_state.json"
    monkeypatch.setattr(fso, "_OVERRIDE_STATE_PATH", path)
    path.write_text("not json", encoding="utf-8")
    state = fso.get_active_override_state(datetime.date(2024, 1, 31))
    assert state.triggered_at is None


# ── Hard bounds constants (spec §2.7 defense-in-depth) ──────────────────────
def test_spec_constants_within_bounds() -> None:
    """Spec §2.7 hard bounds — code must match spec lock values."""
    assert fso.HARD_OVERRIDE_MULTIPLIER == 0.5
    assert fso.HARD_DURATION_DAYS == 5
    assert fso.HARD_DURATION_CAP == 10
    assert fso.HARD_MULTIPLIER_LOWER == 0.5
    assert fso.HARD_MULTIPLIER_UPPER == 1.0
    # Defense-in-depth: duration ≤ cap
    assert fso.HARD_DURATION_DAYS <= fso.HARD_DURATION_CAP
    # Multiplier within bounds
    assert fso.HARD_MULTIPLIER_LOWER <= fso.HARD_OVERRIDE_MULTIPLIER <= fso.HARD_MULTIPLIER_UPPER


def test_annual_budget_caps_set() -> None:
    """Spec §2.3 cost caps — both annual and per-call positive + small."""
    assert 0 < fso.ANNUAL_BUDGET_USD <= 10.0
    assert 0 < fso.PER_CALL_BUDGET_USD <= 1.0
    assert fso.PER_CALL_BUDGET_USD < fso.ANNUAL_BUDGET_USD


# ── process_fomc_day noop on non-FOMC day (no LLM call) ─────────────────────
def test_process_fomc_day_noop_on_non_fomc_day() -> None:
    """Non-FOMC day → immediate return without LLM call (calendar lock)."""
    # Pick a date guaranteed not in FOMC list
    out = fso.process_fomc_day(datetime.date(2024, 7, 4))  # July 4 holiday
    assert out["action"] == "noop_not_fomc_day"


def test_process_fomc_day_rejects_non_date_input() -> None:
    with pytest.raises(TypeError):
        fso.process_fomc_day("2024-01-31")  # type: ignore[arg-type]
