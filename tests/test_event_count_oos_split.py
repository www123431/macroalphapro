"""Tests for run_gate event-count OOS split (deferred from gate redesign,
finished 2026-05-30).

Sparse-event templates (event_study, PEAD-like) need OOS = "second half
of events" not "second half of time". With ~250 events over 300 months,
time-bisect can leave 90% of events in one half, gutting OOS power.
Event-count split puts ~125 events in each half.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── _event_count_split_point ──────────────────────────────────────────────

def test_event_count_split_uniform_density_equals_time_bisect():
    """Uniform event_density → split at n//2 (same as time-bisect)."""
    from engine.research.pipeline import _event_count_split_point
    idx = pd.date_range("2010-01-01", periods=100, freq="ME")
    returns = pd.Series(np.random.randn(100), index=idx)
    events = pd.Series([1] * 100, index=idx)
    split = _event_count_split_point(returns, events)
    # Cumsum hits 50 at index 49 (0-based, 50th element); accept within ±1
    assert abs(split - 50) <= 1


def test_event_count_split_front_loaded_events_pushes_oos_late():
    """All events in first 30% → OOS starts ~25% in (where cumsum=half)."""
    from engine.research.pipeline import _event_count_split_point
    n = 100
    idx = pd.date_range("2010-01-01", periods=n, freq="ME")
    returns = pd.Series(np.random.randn(n), index=idx)
    events = pd.Series([0] * n, index=idx)
    events.iloc[:30] = 1     # 30 events all in first 30 months
    split = _event_count_split_point(returns, events)
    # Cumsum=15 hits at idx ~14
    assert 10 <= split <= 20


def test_event_count_split_back_loaded_events_pushes_oos_early():
    """All events in last 30% → OOS starts ~85% in but capped."""
    from engine.research.pipeline import _event_count_split_point
    n = 100
    idx = pd.date_range("2010-01-01", periods=n, freq="ME")
    returns = pd.Series(np.random.randn(n), index=idx)
    events = pd.Series([0] * n, index=idx)
    events.iloc[70:] = 1
    split = _event_count_split_point(returns, events)
    # Cumsum=15 hits at idx ~84
    assert 80 <= split <= 90


def test_event_count_split_zero_total_falls_back_to_time_bisect():
    """All-zero event_density → degenerate; fall back to n//2."""
    from engine.research.pipeline import _event_count_split_point
    n = 100
    idx = pd.date_range("2010-01-01", periods=n, freq="ME")
    returns = pd.Series(np.random.randn(n), index=idx)
    events = pd.Series([0] * n, index=idx)
    split = _event_count_split_point(returns, events)
    assert split == n // 2


def test_event_count_split_handles_missing_index_overlap():
    """event_density covers a subset of returns index → reindex+fillna."""
    from engine.research.pipeline import _event_count_split_point
    idx = pd.date_range("2010-01-01", periods=100, freq="ME")
    returns = pd.Series(np.random.randn(100), index=idx)
    # Events only on every 4th month, total = 25
    events_idx = idx[::4]
    events = pd.Series([1] * len(events_idx), index=events_idx)
    split = _event_count_split_point(returns, events)
    # Cumsum=12.5 hits at index ~48 (every 4th, so 12 events in [0,48])
    assert 40 <= split <= 56


# ── run_gate integration with oos_split kwarg ────────────────────────────

def test_run_gate_oos_split_time_default():
    """Default oos_split='time' → ledger records 'time'."""
    from engine.research.pipeline import run_gate
    np.random.seed(101)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    verdict = run_gate(series, name="default_time", pead_control=False, log=False)
    assert verdict["oos_split"] == "time"
    assert verdict["oos_split_pos"] == 60     # n // 2 for 120 months


def test_run_gate_oos_split_event_count_with_density():
    """oos_split='event_count' + event_density → uses event-count split."""
    from engine.research.pipeline import run_gate
    np.random.seed(102)
    n = 120
    idx = pd.date_range("2014-01-01", periods=n, freq="ME")
    series = pd.Series(np.random.randn(n) * 0.05, index=idx)
    # Front-loaded events
    events = pd.Series([0] * n, index=idx)
    events.iloc[:40] = 1     # 40 events in first 40 months
    verdict = run_gate(series, name="event_split",
                          oos_split="event_count",
                          event_density=events,
                          pead_control=False, log=False)
    assert verdict["oos_split"] == "event_count"
    # Split should be far earlier than 60 (time-bisect) because events
    # are front-loaded
    assert verdict["oos_split_pos"] < 40


def test_run_gate_event_count_without_density_falls_back_with_warning(caplog):
    """Asking for event_count without density → log warning + time-bisect."""
    from engine.research.pipeline import run_gate
    np.random.seed(103)
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    import logging
    with caplog.at_level(logging.WARNING):
        verdict = run_gate(series, name="missing_density",
                              oos_split="event_count",
                              event_density=None,
                              pead_control=False, log=False)
    assert verdict["oos_split"] == "time"
    assert verdict["oos_split_pos"] == 60
    assert any("event_count" in r.message and "fallback" in r.message.lower()
                  for r in caplog.records) or \
              any("event_density is None" in r.message
                  for r in caplog.records)


def test_run_gate_invalid_oos_split_raises():
    from engine.research.pipeline import run_gate
    series = pd.Series(np.random.randn(120) * 0.05,
                        index=pd.date_range("2014-01-01", periods=120, freq="ME"))
    with pytest.raises(ValueError):
        run_gate(series, name="bad", oos_split="banana",
                    pead_control=False, log=False)


def test_event_count_oos_changes_oos_sharpe_when_uneven_events():
    """Verify oos_sharpe DIFFERS between time and event_count splits when
    events are unevenly distributed (the actual reason event-count exists)."""
    from engine.research.pipeline import run_gate
    np.random.seed(104)
    n = 120
    idx = pd.date_range("2014-01-01", periods=n, freq="ME")
    # Returns with a regime shift around month 60
    base = np.random.randn(n) * 0.05
    base[60:] += 0.02     # second half = better
    series = pd.Series(base, index=idx)
    # Front-loaded events: 90% in first 30 months
    events = pd.Series([0] * n, index=idx)
    events.iloc[:30] = 9
    events.iloc[30:60] = 1

    v_time = run_gate(series, name="t_v_e_time",
                          oos_split="time",
                          pead_control=False, log=False)
    v_event = run_gate(series, name="t_v_e_event",
                          oos_split="event_count",
                          event_density=events,
                          pead_control=False, log=False)
    # With time-bisect, OOS = months 60-119 (the regime-shift "good" half)
    # With event_count, OOS starts ~month 17 (where cumsum=half-of-events)
    # → covers MUCH more sample → different OOS Sharpe
    assert v_time["oos_split_pos"] != v_event["oos_split_pos"]
    assert v_time["oos_sharpe"] != v_event["oos_sharpe"]


# ── event_study GATE_PROFILE integration ────────────────────────────────

def test_event_study_profile_carries_event_count_split():
    """event_study template must declare oos_split='event_count' so callers
    auto-pick it up when passing profile=template.GATE_PROFILE."""
    from engine.research.templates import event_study
    assert event_study.GATE_PROFILE.get("oos_split") == "event_count"


def test_event_study_profile_applied_via_run_gate():
    """End-to-end: pass event_study.GATE_PROFILE + event_density → gate
    uses event-count split automatically."""
    from engine.research.pipeline import run_gate
    from engine.research.templates import event_study
    np.random.seed(105)
    n = 120
    idx = pd.date_range("2014-01-01", periods=n, freq="ME")
    series = pd.Series(np.random.randn(n) * 0.05, index=idx)
    # Front-loaded events
    events = pd.Series([0] * n, index=idx)
    events.iloc[:40] = 2
    verdict = run_gate(series, name="event_study_profile",
                          profile=event_study.GATE_PROFILE,
                          event_density=events,
                          log=False)
    assert verdict["oos_split"] == "event_count"
    # Profile also brings HAC=12 + pead_control=True + n_trials_base=8
    assert verdict["hac_lags"] == 12
    assert verdict["n_trials"] == 8


def test_event_density_from_panel_helper():
    """event_density_from_panel sums per-month event count from event_panel."""
    from engine.research.templates.event_study import event_density_from_panel
    idx = pd.date_range("2020-01-01", periods=5, freq="ME")
    tickers = ["A", "B", "C"]
    ep = pd.DataFrame([
        [True,  False, False],   # 1 event
        [True,  True,  False],   # 2 events
        [False, False, False],   # 0 events
        [True,  True,  True ],   # 3 events
        [False, True,  False],   # 1 event
    ], index=idx, columns=tickers)
    density = event_density_from_panel(ep)
    assert list(density.values) == [1, 2, 0, 3, 1]
