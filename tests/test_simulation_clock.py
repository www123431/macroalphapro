"""tests/test_simulation_clock.py — Tier C L2-1 Phase 2.

Unit tests for engine.data.pit_warehouse.simulation_clock.SimClock.

The clock is the architectural primitive that makes PIT enforcement
possible (any data access where as_of > clock.now is rejected). Bugs
here would undermine the whole L2-1 PIT safety guarantee, so the
tests check edge cases aggressively.
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_clock_initializes_now_to_start():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start=pd.Timestamp("2020-01-01"),
                  end=pd.Timestamp("2024-12-31"))
    assert c.now == pd.Timestamp("2020-01-01")


def test_clock_string_dates_get_coerced():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    assert c.now == pd.Timestamp("2020-01-01")


def test_clock_raises_on_inverted_bounds():
    from engine.data.pit_warehouse import SimClock
    with pytest.raises(ValueError, match="start.*>"):
        SimClock(start="2025-01-01", end="2020-01-01")


def test_advance_by_offset_string():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance("1ME")
    assert c.now == pd.Timestamp("2020-01-31")


def test_advance_by_timedelta():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance(pd.Timedelta(days=10))
    assert c.now == pd.Timestamp("2020-01-11")


def test_advance_to_absolute_timestamp():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance(pd.Timestamp("2022-06-15"))
    assert c.now == pd.Timestamp("2022-06-15")


def test_advance_rejects_rewinding():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance("1ME")
    with pytest.raises(ValueError, match="rewind"):
        c.advance(pd.Timestamp("2019-12-31"))


def test_advance_clamps_at_end():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2020-06-30")
    c.advance("5y")
    assert c.now == c.end


def test_knows_about_true_for_past_and_present():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance(pd.Timestamp("2022-06-15"))
    assert c.knows_about(pd.Timestamp("2022-06-15")) is True
    assert c.knows_about(pd.Timestamp("2022-06-14")) is True
    assert c.knows_about(pd.Timestamp("2020-01-01")) is True


def test_knows_about_false_for_future():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance(pd.Timestamp("2022-06-15"))
    assert c.knows_about(pd.Timestamp("2022-06-16")) is False
    assert c.knows_about(pd.Timestamp("2024-01-01")) is False


def test_knows_about_none_returns_false():
    """Defensive: None as_of → never knowable (no look-ahead)."""
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    assert c.knows_about(None) is False


def test_knows_about_accepts_strings():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance(pd.Timestamp("2022-06-15"))
    assert c.knows_about("2022-06-01") is True
    assert c.knows_about("2022-07-01") is False


def test_reset_returns_to_start():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance(pd.Timestamp("2022-06-15"))
    c.reset()
    assert c.now == c.start


def test_iter_rebal_dates_monthly():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2020-06-30")
    dates = list(c.iter_rebal_dates("ME"))
    assert len(dates) == 6   # Jan-Jun
    assert dates[0] == pd.Timestamp("2020-01-31")
    assert dates[-1] == pd.Timestamp("2020-06-30")


def test_iter_rebal_dates_does_not_advance_clock():
    """Pure iterator; caller must advance() if they want state."""
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2020-06-30")
    _ = list(c.iter_rebal_dates("ME"))
    assert c.now == c.start   # Unchanged


def test_repr_is_human_readable():
    from engine.data.pit_warehouse import SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    s = repr(c)
    assert "2020-01-01" in s
    assert "2024-12-31" in s
