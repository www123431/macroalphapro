"""Tests for P1a (broker_reconciliation) + P1b (nav_anomaly) liveness
modules. Both are best-effort — failures must NEVER raise, only return
a verdict dict the heartbeat can embed.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import pytest


# ── broker_reconciliation ────────────────────────────────────────


@pytest.fixture
def tmp_repo_root(tmp_path, monkeypatch):
    from engine.research import broker_reconciliation as B
    monkeypatch.setattr(B, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    yield tmp_path


def test_reconcile_no_submit_artifact_returns_status(tmp_repo_root):
    from engine.research import broker_reconciliation as B
    verdict = B.reconcile(_dt.date(2026, 6, 2))
    assert verdict["status"] == B.STATUS_NO_SUBMIT_ARTIFACT
    assert verdict["as_of"] == "2026-06-02"
    assert verdict["explanation"].startswith("No data/_paper_submit_")


def test_reconcile_with_submit_and_full_fills_returns_ok_or_unreachable(tmp_repo_root, monkeypatch):
    """When a submit artifact has fills == orders and broker is alpaca_paper
    but live API is unreachable, status = broker_unreachable (honest)."""
    from engine.research import broker_reconciliation as B
    as_of = _dt.date(2026, 6, 2)
    (tmp_repo_root / "data" / f"_paper_submit_{as_of.isoformat()}.json").write_text(
        json.dumps({
            "as_of":         as_of.isoformat(),
            "n_tickers":     174,
            "gross_weight":  0.79,
            "report": {
                "broker":        "alpaca_paper",
                "equity_before": 100046.19,
                "n_orders":      114,
                "n_fills":       114,
                "warnings":      [],
            },
        }), encoding="utf-8")
    # Force live alpaca probe to return None (no key in test env)
    monkeypatch.setattr(B, "_try_live_alpaca", lambda: None)
    verdict = B.reconcile(as_of)
    assert verdict["status"] == B.STATUS_BROKER_UNREACHABLE
    assert verdict["n_orders_submitted"] == 114
    assert verdict["n_fills"] == 114
    assert verdict["fill_rate"] == 1.0


def test_reconcile_fill_shortfall_when_fills_below_orders(tmp_repo_root, monkeypatch):
    from engine.research import broker_reconciliation as B
    as_of = _dt.date(2026, 6, 2)
    (tmp_repo_root / "data" / f"_paper_submit_{as_of.isoformat()}.json").write_text(
        json.dumps({
            "as_of": as_of.isoformat(), "n_tickers": 100,
            "report": {"broker": "alpaca_paper", "equity_before": 100000.0,
                       "n_orders": 100, "n_fills": 70, "warnings": []},
        }), encoding="utf-8")
    monkeypatch.setattr(B, "_try_live_alpaca", lambda: {
        "equity": 100200.0, "cash": 12000.0, "buying_power": 12000.0,
        "n_positions": 95, "gross_exposure": 79123.45, "position_tickers": ["AAPL"],
    })
    verdict = B.reconcile(as_of)
    assert verdict["status"] == B.STATUS_FILL_SHORTFALL
    assert verdict["fill_rate"] == 0.7


def test_reconcile_status_ok_when_everything_aligned(tmp_repo_root, monkeypatch):
    from engine.research import broker_reconciliation as B
    as_of = _dt.date(2026, 6, 2)
    (tmp_repo_root / "data" / f"_paper_submit_{as_of.isoformat()}.json").write_text(
        json.dumps({
            "as_of": as_of.isoformat(), "n_tickers": 100,
            "report": {"broker": "alpaca_paper", "equity_before": 100000.0,
                       "n_orders": 100, "n_fills": 100, "warnings": []},
        }), encoding="utf-8")
    monkeypatch.setattr(B, "_try_live_alpaca", lambda: {
        "equity": 100200.0, "cash": 12000.0, "buying_power": 12000.0,
        "n_positions": 100, "gross_exposure": 79123.45, "position_tickers": [],
    })
    verdict = B.reconcile(as_of)
    assert verdict["status"] == B.STATUS_OK
    assert "Reconciled" in verdict["explanation"]


# ── nav_anomaly ─────────────────────────────────────────────────


@pytest.fixture
def tmp_nav_ledger(tmp_path, monkeypatch):
    from engine.research import nav_anomaly as N
    p = tmp_path / "nav_history.jsonl"
    monkeypatch.setattr(N, "NAV_LEDGER", p)
    yield p


def test_record_nav_no_prior_status(tmp_nav_ledger):
    from engine.research import nav_anomaly as N
    v = N.record_nav(as_of=_dt.date(2026, 5, 1), equity=100000.0)
    assert v["status"] == N.STATUS_NO_PRIOR
    assert v["log_return"] is None


def test_record_nav_normal_after_2_prior_records_2026_06_18(tmp_nav_ledger):
    """Post-2026-06-18 fix: z-score uses deployed vol target as baseline,
    so we don't need a rolling-window warmup. After 1+ prior NAVs, status
    is OK (or anomaly if z>3 vs deployed vol target). The old
    INSUFFICIENT-after-2 semantic is gone — it was the source of the
    early-deployment false-positive cascade flagged by the 2026-06-18
    NAV monitoring audit.
    """
    from engine.research import nav_anomaly as N
    N.record_nav(as_of=_dt.date(2026, 5, 1), equity=100000.0)
    N.record_nav(as_of=_dt.date(2026, 5, 2), equity=100050.0)
    v = N.record_nav(as_of=_dt.date(2026, 5, 3), equity=100100.0)
    # +0.05% move at 10% vol target (0.63% daily) → z ≈ +0.08σ → OK
    assert v["status"] == N.STATUS_OK
    assert v["z_score"] is not None
    assert abs(v["z_score"]) < 1.0


def test_record_nav_normal_walk_returns_ok(tmp_nav_ledger):
    """Realistic noisy walk → STATUS_OK with finite z near 0. Constant
    daily returns would zero variance → INSUFFICIENT, so we use noise."""
    from engine.research import nav_anomaly as N
    eq = 100000.0
    d = _dt.date(2026, 5, 1)
    # 12 noisy daily returns around +20bp mean, ~30bp stdev
    noisy = [0.0030, -0.0015, 0.0010, -0.0007, 0.0025, -0.0020,
             0.0012, -0.0011, 0.0018, -0.0009, 0.0014, -0.0006]
    for i, r in enumerate(noisy):
        eq *= (1 + r)
        N.record_nav(as_of=d + _dt.timedelta(days=i), equity=eq)
    eq *= (1 + 0.0005)
    v = N.record_nav(as_of=d + _dt.timedelta(days=12), equity=eq)
    assert v["status"] == N.STATUS_OK
    assert v["z_score"] is not None
    assert abs(v["z_score"]) < 2.0


def test_record_nav_3sigma_move_flagged_anomaly(tmp_nav_ledger):
    from engine.research import nav_anomaly as N
    # Seed with ~10 small-noise log returns, then one 5% drop
    d = _dt.date(2026, 5, 1)
    eq = 100000.0
    N.record_nav(as_of=d, equity=eq)
    # Build mild zig-zag for variance
    seq = [0.001, -0.001, 0.0015, -0.0008, 0.0012, -0.0005, 0.0009, -0.0011,
           0.0007, 0.0003, 0.0005, -0.0004]
    for i, r in enumerate(seq, start=1):
        eq *= (1 + r)
        N.record_nav(as_of=d + _dt.timedelta(days=i), equity=eq)
    # Now slam a -5% move — must register as anomaly
    eq *= (1 - 0.05)
    v = N.record_nav(as_of=d + _dt.timedelta(days=len(seq) + 1), equity=eq)
    assert v["status"] == N.STATUS_ANOMALY
    assert v["z_score"] < -3


def test_record_nav_is_idempotent_on_same_equity(tmp_nav_ledger):
    from engine.research import nav_anomaly as N
    d = _dt.date(2026, 5, 1)
    N.record_nav(as_of=d, equity=100000.0)
    # Call twice — second call should NOT append a duplicate row
    N.record_nav(as_of=d, equity=100000.0)
    text = tmp_nav_ledger.read_text(encoding="utf-8").strip()
    rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    assert len(rows) == 1
