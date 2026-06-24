"""
tests/test_auto_audit_rules_watchdog_phase1.py — 10 Watchdog rules unit tests

Phase 1 of Ops Watchdog Agent v1.0 (spec id=63 hash 512c918f).
Each rule has 5 tests: clean / primary-detect / secondary-detect / edge / boundary.
All tests use an isolated tmp sqlite DB to avoid polluting the real engine.
"""
from __future__ import annotations

import datetime
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ── isolated DB fixture ─────────────────────────────────────────────────────
@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    """Fresh sqlite DB + monkeypatch engine.memory / engine.db_models bindings."""
    from engine import memory as _memory
    from engine import db_models as _db_models
    from engine.db_models import Base

    db_path = tmp_path / "test_watchdog_phase1.db"
    test_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(test_engine)

    # UniverseETF uses its own declarative_base() (engine.universe_manager._Base).
    # Create those tables too so rule_universe_data_freshness_per_ticker works.
    from engine import universe_manager as _um
    _um._Base.metadata.create_all(test_engine)

    TestSession = sessionmaker(bind=test_engine, expire_on_commit=False)

    monkeypatch.setattr(_memory,    "engine",        test_engine)
    monkeypatch.setattr(_memory,    "SessionFactory", TestSession)
    monkeypatch.setattr(_db_models, "engine",        test_engine)
    monkeypatch.setattr(_db_models, "SessionFactory", TestSession)
    # universe_manager binds SessionFactory at import-time; patch its local too.
    monkeypatch.setattr(_um,        "SessionFactory", TestSession)

    return TestSession


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — rule_cycle_state_completion
# ─────────────────────────────────────────────────────────────────────────────
class TestCycleStateCompletion:
    def test_clean_no_cycles_returns_none(self, isolated_db):
        from engine.auto_audit_rules import rule_cycle_state_completion
        assert rule_cycle_state_completion() is None

    def test_detects_failed_cycle(self, isolated_db):
        from engine.auto_audit_rules import rule_cycle_state_completion
        from engine.db_models import CycleState
        with isolated_db() as s:
            s.add(CycleState(cycle_type="daily", as_of_date=datetime.date.today(),
                             status="failed", started_at=datetime.datetime.utcnow(),
                             error_log="batch_step_3_crashed"))
            s.commit()
        result = rule_cycle_state_completion()
        assert result is not None and result["severity"] == "HIGH"
        kinds = [i["kind"] for i in result["snapshot"]["issues"]]
        assert "cycle_failed" in kinds

    def test_detects_stuck_running(self, isolated_db):
        from engine.auto_audit_rules import rule_cycle_state_completion
        from engine.db_models import CycleState
        old = datetime.datetime.utcnow() - datetime.timedelta(hours=14)
        with isolated_db() as s:
            s.add(CycleState(cycle_type="daily", as_of_date=datetime.date.today(),
                             status="running", started_at=old))
            s.commit()
        result = rule_cycle_state_completion()
        assert result is not None and result["severity"] == "HIGH"
        kinds = [i["kind"] for i in result["snapshot"]["issues"]]
        assert "cycle_stuck_running" in kinds

    def test_detects_no_recent_in_36h(self, isolated_db):
        from engine.auto_audit_rules import rule_cycle_state_completion
        from engine.db_models import CycleState
        old = datetime.datetime.utcnow() - datetime.timedelta(hours=48)
        with isolated_db() as s:
            s.add(CycleState(cycle_type="daily", as_of_date=datetime.date.today(),
                             status="completed", started_at=old, finished_at=old))
            s.commit()
        result = rule_cycle_state_completion()
        assert result is not None and result["severity"] == "MID"
        kinds = [i["kind"] for i in result["snapshot"]["issues"]]
        assert "no_recent_cycle_in_36h" in kinds

    def test_healthy_recent_completed_returns_none(self, isolated_db):
        from engine.auto_audit_rules import rule_cycle_state_completion
        from engine.db_models import CycleState
        recent = datetime.datetime.utcnow() - datetime.timedelta(hours=2)
        with isolated_db() as s:
            s.add(CycleState(cycle_type="daily", as_of_date=datetime.date.today(),
                             status="completed", started_at=recent, finished_at=recent))
            s.commit()
        assert rule_cycle_state_completion() is None


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — rule_universe_data_freshness_per_ticker
# ─────────────────────────────────────────────────────────────────────────────
def _seed_universe(session, tickers):
    """Seed UniverseETF with active rows. Returns the inserted tickers."""
    from engine.universe_manager import UniverseETF
    for i, t in enumerate(tickers):
        session.add(UniverseETF(
            ticker=t, sector=f"sector_{t}", asset_class="equity_sector",
            batch=1, active=True,
        ))
    session.commit()
    return tickers


def _seed_signal_record(session, ticker, sector, date):
    from engine.db_models import SignalRecord
    session.add(SignalRecord(date=date, ticker=ticker, sector=sector,
                             tsmom_signal=1, composite_score=75.0,
                             gate_status="passed"))


class TestUniverseFreshness:
    def test_clean_all_fresh(self, isolated_db):
        from engine.auto_audit_rules import rule_universe_data_freshness_per_ticker
        today = datetime.date.today()
        with isolated_db() as s:
            tickers = _seed_universe(s, ["AAA", "BBB", "CCC"])
            for t in tickers:
                _seed_signal_record(s, t, f"sector_{t}", today)
            s.commit()
        assert rule_universe_data_freshness_per_ticker() is None

    def test_detects_missing_ticker(self, isolated_db):
        from engine.auto_audit_rules import rule_universe_data_freshness_per_ticker
        today = datetime.date.today()
        with isolated_db() as s:
            tickers = _seed_universe(s, ["AAA", "BBB", "CCC"])
            # Only AAA and BBB have SignalRecord; CCC missing
            _seed_signal_record(s, "AAA", "sector_AAA", today)
            _seed_signal_record(s, "BBB", "sector_BBB", today)
            s.commit()
        result = rule_universe_data_freshness_per_ticker()
        assert result is not None and result["severity"] == "HIGH"
        assert result["snapshot"]["n_missing"] >= 1
        assert any(m["ticker"] == "CCC" for m in result["snapshot"]["missing_sample"])

    def test_detects_stale_ticker(self, isolated_db):
        """1 stale ticker out of 10 = 10% pct_bad → MID (below 20% HIGH threshold)."""
        from engine.auto_audit_rules import rule_universe_data_freshness_per_ticker
        today = datetime.date.today()
        old = today - datetime.timedelta(days=10)
        with isolated_db() as s:
            tickers = _seed_universe(s, [f"X{i:02d}" for i in range(10)])
            for t in tickers[:-1]:
                _seed_signal_record(s, t, f"sector_{t}", today)
            _seed_signal_record(s, tickers[-1], f"sector_{tickers[-1]}", old)
            s.commit()
        result = rule_universe_data_freshness_per_ticker()
        assert result is not None
        assert result["snapshot"]["n_stale"] == 1
        assert result["severity"] == "MID"

    def test_handles_empty_universe(self, isolated_db, monkeypatch):
        from engine.auto_audit_rules import rule_universe_data_freshness_per_ticker
        from engine import universe_manager
        monkeypatch.setattr(universe_manager, "get_active_universe",
                            lambda asset_classes=None: {})
        assert rule_universe_data_freshness_per_ticker() is None

    def test_detects_broad_outage(self, isolated_db):
        from engine.auto_audit_rules import rule_universe_data_freshness_per_ticker
        today = datetime.date.today()
        old = today - datetime.timedelta(days=12)
        with isolated_db() as s:
            tickers = _seed_universe(s, ["AAA", "BBB", "CCC", "DDD", "EEE"])
            _seed_signal_record(s, "AAA", "sector_AAA", today)
            for t in tickers[1:]:
                _seed_signal_record(s, t, f"sector_{t}", old)
            s.commit()
        result = rule_universe_data_freshness_per_ticker()
        assert result is not None and result["severity"] == "HIGH"
        assert result["snapshot"]["pct_bad"] >= 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Mode 5 — rule_weight_delta_p99_unexplained
# ─────────────────────────────────────────────────────────────────────────────
def _seed_trade(session, days_ago, ticker, weight_delta, trigger_reason=None):
    from engine.db_models import SimulatedTrade
    d = datetime.date.today() - datetime.timedelta(days=days_ago)
    session.add(SimulatedTrade(
        trade_date=d, sector=f"sector_{ticker}", ticker=ticker,
        action="BUY" if weight_delta > 0 else "SELL",
        weight_before=0.0, weight_after=weight_delta, weight_delta=weight_delta,
        trigger_reason=trigger_reason,
    ))


class TestWeightDeltaP99:
    def test_clean_no_spikes(self, isolated_db):
        from engine.auto_audit_rules import rule_weight_delta_p99_unexplained
        with isolated_db() as s:
            for i in range(30):
                _seed_trade(s, days_ago=20 + i % 30, ticker=f"T{i}", weight_delta=0.01)
            _seed_trade(s, days_ago=2, ticker="RECENT", weight_delta=0.015)
            s.commit()
        assert rule_weight_delta_p99_unexplained() is None

    def test_detects_spike_no_trigger(self, isolated_db):
        from engine.auto_audit_rules import rule_weight_delta_p99_unexplained
        with isolated_db() as s:
            for i in range(30):
                _seed_trade(s, days_ago=20 + i % 30, ticker=f"T{i}", weight_delta=0.01)
            _seed_trade(s, days_ago=1, ticker="SPIKE", weight_delta=0.5,
                        trigger_reason="threshold")
            s.commit()
        result = rule_weight_delta_p99_unexplained()
        assert result is not None and result["severity"] == "HIGH"
        spike_tickers = [sp["ticker"] for sp in result["snapshot"]["spikes"]]
        assert "SPIKE" in spike_tickers

    def test_ignores_spike_with_signal_flip(self, isolated_db):
        from engine.auto_audit_rules import rule_weight_delta_p99_unexplained
        with isolated_db() as s:
            for i in range(30):
                _seed_trade(s, days_ago=20 + i % 30, ticker=f"T{i}", weight_delta=0.01)
            _seed_trade(s, days_ago=1, ticker="LEGITSPIKE", weight_delta=0.5,
                        trigger_reason="signal_flip")
            s.commit()
        assert rule_weight_delta_p99_unexplained() is None

    def test_handles_insufficient_baseline(self, isolated_db):
        from engine.auto_audit_rules import rule_weight_delta_p99_unexplained
        with isolated_db() as s:
            for i in range(5):
                _seed_trade(s, days_ago=15, ticker=f"T{i}", weight_delta=0.01)
            _seed_trade(s, days_ago=1, ticker="BIG", weight_delta=0.5)
            s.commit()
        assert rule_weight_delta_p99_unexplained() is None

    def test_handles_zero_p99(self, isolated_db):
        from engine.auto_audit_rules import rule_weight_delta_p99_unexplained
        with isolated_db() as s:
            for i in range(30):
                _seed_trade(s, days_ago=20 + i % 30, ticker=f"T{i}", weight_delta=0.0)
            _seed_trade(s, days_ago=1, ticker="BIG", weight_delta=0.5)
            s.commit()
        assert rule_weight_delta_p99_unexplained() is None


# ─────────────────────────────────────────────────────────────────────────────
# Mode 6 — rule_signal_trade_referential_integrity
# ─────────────────────────────────────────────────────────────────────────────
class TestSignalTradeIntegrity:
    def test_clean_signal_with_trade(self, isolated_db):
        from engine.auto_audit_rules import rule_signal_trade_referential_integrity
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_signal_record(s, "ZZZ", "sec_ZZZ", today)
            _seed_trade(s, days_ago=0, ticker="ZZZ", weight_delta=0.05,
                        trigger_reason="signal_flip")
            s.commit()
        assert rule_signal_trade_referential_integrity() is None

    def test_detects_orphan_signal(self, isolated_db):
        from engine.auto_audit_rules import rule_signal_trade_referential_integrity
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_signal_record(s, "ORPHAN", "sec_ORPHAN", today)
            s.commit()
        result = rule_signal_trade_referential_integrity()
        assert result is not None and result["severity"] == "MID"
        orphan_tickers = [o["ticker"] for o in result["snapshot"]["orphans"]]
        assert "ORPHAN" in orphan_tickers

    def test_ignores_low_score_signal(self, isolated_db):
        from engine.auto_audit_rules import rule_signal_trade_referential_integrity
        from engine.db_models import SignalRecord
        today = datetime.date.today()
        with isolated_db() as s:
            s.add(SignalRecord(date=today, ticker="LOWS", sector="sec_LOWS",
                               composite_score=60.0, gate_status="passed"))
            s.commit()
        assert rule_signal_trade_referential_integrity() is None

    def test_ignores_blocked_signal(self, isolated_db):
        from engine.auto_audit_rules import rule_signal_trade_referential_integrity
        from engine.db_models import SignalRecord
        today = datetime.date.today()
        with isolated_db() as s:
            s.add(SignalRecord(date=today, ticker="BLK", sector="sec_BLK",
                               composite_score=85.0, gate_status="blocked"))
            s.commit()
        assert rule_signal_trade_referential_integrity() is None

    def test_clean_signal_with_offset_trade(self, isolated_db):
        """Trade ±2 days from signal date still counts as matched."""
        from engine.auto_audit_rules import rule_signal_trade_referential_integrity
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_signal_record(s, "OFFSET", "sec_OFFSET", today)
            _seed_trade(s, days_ago=2, ticker="OFFSET", weight_delta=0.05)
            s.commit()
        assert rule_signal_trade_referential_integrity() is None


# ─────────────────────────────────────────────────────────────────────────────
# Mode 7 — rule_nav_move_vs_rebalance_audit
# ─────────────────────────────────────────────────────────────────────────────
def _seed_nav(session, date, nav_after_flow, nav_close, daily_return):
    from engine.db_models import PortfolioNavSnapshot
    session.add(PortfolioNavSnapshot(
        snapshot_date=date, nav_open=nav_after_flow, external_flow=0.0,
        nav_after_flow=nav_after_flow, nav_close=nav_close,
        gross_pnl=nav_close - nav_after_flow,
        daily_modified_dietz=daily_return,
    ))


class TestNavMoveAudit:
    def test_clean_normal_move(self, isolated_db):
        from engine.auto_audit_rules import rule_nav_move_vs_rebalance_audit
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(60):
                d = today - datetime.timedelta(days=60 - i)
                _seed_nav(s, d, 100.0, 100.1, 0.001)
            _seed_nav(s, today, 100.0, 100.15, 0.0015)   # ~0.5σ off mean
            s.commit()
        assert rule_nav_move_vs_rebalance_audit() is None

    def test_detects_3sigma_no_trades(self, isolated_db):
        from engine.auto_audit_rules import rule_nav_move_vs_rebalance_audit
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(60):
                d = today - datetime.timedelta(days=60 - i)
                # baseline daily_modified_dietz ≈ 0.001 ± 0.002
                ret = 0.001 + (0.002 if i % 2 == 0 else -0.002)
                _seed_nav(s, d, 100.0, 100.0 * (1 + ret), ret)
            # Today: 5% move, NO trades
            _seed_nav(s, today, 100.0, 105.0, 0.05)
            s.commit()
        result = rule_nav_move_vs_rebalance_audit()
        assert result is not None and result["severity"] == "HIGH"
        assert abs(result["snapshot"]["z_vs_60d_baseline"]) > 3.0

    def test_ignores_3sigma_with_trades(self, isolated_db):
        from engine.auto_audit_rules import rule_nav_move_vs_rebalance_audit
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(60):
                d = today - datetime.timedelta(days=60 - i)
                ret = 0.001 + (0.002 if i % 2 == 0 else -0.002)
                _seed_nav(s, d, 100.0, 100.0 * (1 + ret), ret)
            _seed_nav(s, today, 100.0, 105.0, 0.05)
            _seed_trade(s, days_ago=0, ticker="X", weight_delta=0.05)
            s.commit()
        assert rule_nav_move_vs_rebalance_audit() is None

    def test_handles_insufficient_baseline(self, isolated_db):
        from engine.auto_audit_rules import rule_nav_move_vs_rebalance_audit
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(5):
                d = today - datetime.timedelta(days=10 - i)
                _seed_nav(s, d, 100.0, 100.1, 0.001)
            _seed_nav(s, today, 100.0, 105.0, 0.05)
            s.commit()
        assert rule_nav_move_vs_rebalance_audit() is None

    def test_handles_zero_sigma(self, isolated_db):
        from engine.auto_audit_rules import rule_nav_move_vs_rebalance_audit
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(60):
                d = today - datetime.timedelta(days=60 - i)
                _seed_nav(s, d, 100.0, 100.5, 0.005)   # constant
            _seed_nav(s, today, 100.0, 105.0, 0.05)
            s.commit()
        assert rule_nav_move_vs_rebalance_audit() is None


# ─────────────────────────────────────────────────────────────────────────────
# Mode 8 — rule_signal_panel_nan_scan
# ─────────────────────────────────────────────────────────────────────────────
class TestSignalNanScan:
    def test_clean_no_nan(self, isolated_db):
        from engine.auto_audit_rules import rule_signal_panel_nan_scan
        from engine.db_models import SignalRecord
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(20):
                s.add(SignalRecord(date=today, ticker=f"T{i}", sector=f"sec_{i}",
                                   tsmom_signal=1, composite_score=70.0,
                                   gate_status="passed"))
            s.commit()
        assert rule_signal_panel_nan_scan() is None

    def test_detects_high_nan_composite(self, isolated_db):
        from engine.auto_audit_rules import rule_signal_panel_nan_scan
        from engine.db_models import SignalRecord
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(20):
                s.add(SignalRecord(date=today, ticker=f"T{i}", sector=f"sec_{i}",
                                   tsmom_signal=1,
                                   composite_score=None if i < 5 else 70.0,
                                   gate_status="passed"))
            s.commit()
        result = rule_signal_panel_nan_scan()
        assert result is not None and result["severity"] == "HIGH"
        assert result["snapshot"]["pct_nan"] >= 0.20

    def test_handles_low_nan(self, isolated_db):
        from engine.auto_audit_rules import rule_signal_panel_nan_scan
        from engine.db_models import SignalRecord
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(20):
                s.add(SignalRecord(date=today, ticker=f"T{i}", sector=f"sec_{i}",
                                   tsmom_signal=1,
                                   composite_score=None if i == 0 else 70.0,
                                   gate_status="passed"))
            s.commit()
        assert rule_signal_panel_nan_scan() is None

    def test_handles_insufficient_total(self, isolated_db):
        from engine.auto_audit_rules import rule_signal_panel_nan_scan
        from engine.db_models import SignalRecord
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(3):
                s.add(SignalRecord(date=today, ticker=f"T{i}", sector=f"sec_{i}",
                                   composite_score=None))
            s.commit()
        assert rule_signal_panel_nan_scan() is None

    def test_detects_nan_on_tsmom_signal(self, isolated_db):
        from engine.auto_audit_rules import rule_signal_panel_nan_scan
        from engine.db_models import SignalRecord
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(20):
                s.add(SignalRecord(date=today, ticker=f"T{i}", sector=f"sec_{i}",
                                   tsmom_signal=None if i < 3 else 1,
                                   composite_score=70.0, gate_status="passed"))
            s.commit()
        result = rule_signal_panel_nan_scan()
        assert result is not None and result["severity"] == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# Mode 9 — rule_realized_tc_vs_spec_rate (amendment 2: per-spec TC compare)
#
# Pre-amendment-2 (Phase 1 v1) used cost_bps / |weight_delta| ratio vs hardcoded
# 2× LIVE_FLAT_COST_BPS threshold. That metric was conceptually wrong (false-
# positives on every strategy since it's just inverse weight_delta scaling).
# Replaced 2026-05-12 per spec id=63 hash c0a5f989 to: median realized cost_bps
# per sleeve compared to spec_metadata.SPEC_TC_BPS_PER_EVENT for the sleeve's
# PRIMARY strategy, flagging if |deviation| / locked > 0.5.
# ─────────────────────────────────────────────────────────────────────────────
def _seed_trade_with_cost(session, days_ago, weight_delta, cost_bps,
                          sleeve_id="etf_l1"):
    from engine.db_models import SimulatedTrade
    d = datetime.date.today() - datetime.timedelta(days=days_ago)
    session.add(SimulatedTrade(
        trade_date=d, sector="sec_X",
        ticker=f"T{days_ago}_{int((weight_delta or 0)*1000)}_{sleeve_id}",
        action="BUY", weight_before=0.0, weight_after=weight_delta,
        weight_delta=weight_delta, cost_bps=cost_bps,
        sleeve_id=sleeve_id,
    ))


class TestRealizedTcVsSpec:
    # Unit conversion: stored cost_bps = |weight_delta| × half_spread_bps (per
    # engine.cost_model.compute_cost_bps line 61 contract). Rule computes
    # realized_half_spread = cost_bps / |weight_delta| and compares to
    # spec-locked tc_bps_per_event. Test seeds set cost_bps = wd × target_ratio
    # so the recovered half_spread matches the intended test scenario.

    def test_clean_at_spec_tc(self, isolated_db):
        """etf_l1 primary spec_id=44 locked tc=8.0 bp; seed trades with
        cost_bps = wd × 8.0 → recovered ratio = 8.0 = spec → None."""
        from engine.auto_audit_rules import rule_realized_tc_vs_spec_rate
        wd = 0.05
        with isolated_db() as s:
            for i in range(15):
                _seed_trade_with_cost(s, days_ago=i, weight_delta=wd,
                                       cost_bps=wd * 8.0, sleeve_id="etf_l1")
            s.commit()
        assert rule_realized_tc_vs_spec_rate() is None

    def test_detects_drift_high(self, isolated_db):
        """Seed cost_bps = wd × 20 → recovered ratio = 20 vs locked 8.0 →
        deviation 1.5 > 0.5 → HIGH."""
        from engine.auto_audit_rules import rule_realized_tc_vs_spec_rate
        wd = 0.05
        with isolated_db() as s:
            for i in range(15):
                _seed_trade_with_cost(s, days_ago=i, weight_delta=wd,
                                       cost_bps=wd * 20.0, sleeve_id="etf_l1")
            s.commit()
        result = rule_realized_tc_vs_spec_rate()
        assert result is not None and result["severity"] == "HIGH"
        etf_v = next(v for v in result["snapshot"]["violations"]
                     if v["sleeve_id"] == "etf_l1")
        assert etf_v["locked_tc_bps"] == 8.0
        assert etf_v["median_realized_bps"] == 20.0
        assert etf_v["deviation"] > 0.5

    def test_detects_drift_low(self, isolated_db):
        """Seed cost_bps = wd × 2 → recovered ratio = 2 vs locked 8.0 →
        deviation 0.75 > 0.5 → HIGH. Below-spec realized is ALSO a finding
        (TC computation may be silently dropping fees / units bug)."""
        from engine.auto_audit_rules import rule_realized_tc_vs_spec_rate
        wd = 0.05
        with isolated_db() as s:
            for i in range(15):
                _seed_trade_with_cost(s, days_ago=i, weight_delta=wd,
                                       cost_bps=wd * 2.0, sleeve_id="etf_l1")
            s.commit()
        result = rule_realized_tc_vs_spec_rate()
        assert result is not None and result["severity"] == "HIGH"
        etf_v = next(v for v in result["snapshot"]["violations"]
                     if v["sleeve_id"] == "etf_l1")
        assert etf_v["median_realized_bps"] == 2.0
        assert etf_v["deviation"] == 0.75

    def test_handles_insufficient_data(self, isolated_db):
        """<MODE_9_MIN_TRADES_FOR_CHECK trades → status='insufficient_data',
        no violation regardless of ratio."""
        from engine.auto_audit_rules import rule_realized_tc_vs_spec_rate
        wd = 0.05
        with isolated_db() as s:
            for i in range(5):
                _seed_trade_with_cost(s, days_ago=i, weight_delta=wd,
                                       cost_bps=wd * 20.0, sleeve_id="etf_l1")
            s.commit()
        assert rule_realized_tc_vs_spec_rate() is None

    def test_handles_unknown_sleeve(self, isolated_db):
        """Trades in a sleeve NOT in SPEC_TC_BPS_PER_EVENT registry → skipped."""
        from engine.auto_audit_rules import rule_realized_tc_vs_spec_rate
        wd = 0.05
        with isolated_db() as s:
            for i in range(15):
                _seed_trade_with_cost(s, days_ago=i, weight_delta=wd,
                                       cost_bps=wd * 99.0,
                                       sleeve_id="not_a_sleeve")
            s.commit()
        # No registered TC metadata for "not_a_sleeve" → rule returns None
        assert rule_realized_tc_vs_spec_rate() is None


# ─────────────────────────────────────────────────────────────────────────────
# Mode 10 — rule_max_position_weight_vs_cap
# ─────────────────────────────────────────────────────────────────────────────
def _seed_position(session, date, sector, ticker, actual_weight, sleeve_id="etf_l1"):
    from engine.db_models import SimulatedPosition
    session.add(SimulatedPosition(
        snapshot_date=date, sector=sector, ticker=ticker,
        target_weight=actual_weight, actual_weight=actual_weight,
        sleeve_id=sleeve_id,
    ))


class TestMaxPositionWeight:
    def test_clean_within_cap(self, isolated_db):
        from engine.auto_audit_rules import rule_max_position_weight_vs_cap
        today = datetime.date.today()
        with isolated_db() as s:
            for i in range(5):
                _seed_position(s, today, f"sec_{i}", f"T{i}", 0.20)
            s.commit()
        assert rule_max_position_weight_vs_cap() is None

    def test_detects_over_cap(self, isolated_db):
        from engine.auto_audit_rules import rule_max_position_weight_vs_cap
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_position(s, today, "sec_BIG", "BIG", 0.30)
            s.commit()
        result = rule_max_position_weight_vs_cap()
        assert result is not None and result["severity"] == "MID"
        tickers = [v["ticker"] for v in result["snapshot"]["violations"]]
        assert "BIG" in tickers

    def test_boundary_at_tolerance(self, isolated_db):
        """0.25 + 0.005 tolerance = 0.255; exactly at boundary → no finding."""
        from engine.auto_audit_rules import rule_max_position_weight_vs_cap
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_position(s, today, "sec_EDGE", "EDGE", 0.255)
            s.commit()
        assert rule_max_position_weight_vs_cap() is None

    def test_handles_no_positions(self, isolated_db):
        from engine.auto_audit_rules import rule_max_position_weight_vs_cap
        assert rule_max_position_weight_vs_cap() is None

    def test_handles_negative_weight(self, isolated_db):
        """Negative (short) weight beyond cap |w| also flagged via abs()."""
        from engine.auto_audit_rules import rule_max_position_weight_vs_cap
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_position(s, today, "sec_SHORT", "SHORT", -0.30)
            s.commit()
        result = rule_max_position_weight_vs_cap()
        assert result is not None and result["severity"] == "MID"


# ─────────────────────────────────────────────────────────────────────────────
# Mode 11 — rule_rebalance_frequency_audit
# ─────────────────────────────────────────────────────────────────────────────
def _seed_bulk_trades(session, year, month, day, n_trades, ticker_prefix="X"):
    from engine.db_models import SimulatedTrade
    d = datetime.date(year, month, day)
    for i in range(n_trades):
        session.add(SimulatedTrade(
            trade_date=d, sector=f"sec_{i}", ticker=f"{ticker_prefix}{i}",
            action="BUY", weight_before=0.0, weight_after=0.05,
            weight_delta=0.05,
        ))


def _last_n_completed_months(n):
    """Return list of (year, month) for last n completed months."""
    today = datetime.date.today()
    first_of_cur = today.replace(day=1)
    months = []
    cur = first_of_cur
    for _ in range(n):
        prev_last = cur - datetime.timedelta(days=1)
        cur = prev_last.replace(day=1)
        months.append((cur.year, cur.month))
    return list(reversed(months))


class TestRebalanceFrequencyAudit:
    def test_clean_one_rebalance_per_month(self, isolated_db):
        from engine.auto_audit_rules import rule_rebalance_frequency_audit
        months = _last_n_completed_months(6)
        with isolated_db() as s:
            for (y, m) in months:
                _seed_bulk_trades(s, y, m, 15, n_trades=5)
            s.commit()
        assert rule_rebalance_frequency_audit() is None

    def test_detects_missing_rebalance_month(self, isolated_db):
        from engine.auto_audit_rules import rule_rebalance_frequency_audit
        months = _last_n_completed_months(6)
        with isolated_db() as s:
            # Skip month index 2 (no trades)
            for idx, (y, m) in enumerate(months):
                if idx == 2:
                    continue
                _seed_bulk_trades(s, y, m, 15, n_trades=5)
            s.commit()
        result = rule_rebalance_frequency_audit()
        assert result is not None and result["severity"] == "HIGH"
        kinds = [v["kind"] for v in result["snapshot"]["violations"]]
        assert "no_rebalance" in kinds

    def test_detects_excess_rebalance_days(self, isolated_db):
        from engine.auto_audit_rules import rule_rebalance_frequency_audit
        months = _last_n_completed_months(6)
        with isolated_db() as s:
            for idx, (y, m) in enumerate(months):
                _seed_bulk_trades(s, y, m, 15, n_trades=5)
            # Add 3 extra bulk days into the most recent completed month
            (y, m) = months[-1]
            for day in (5, 10, 20):
                _seed_bulk_trades(s, y, m, day, n_trades=5, ticker_prefix=f"E{day}_")
            s.commit()
        result = rule_rebalance_frequency_audit()
        assert result is not None
        kinds = [v["kind"] for v in result["snapshot"]["violations"]]
        assert "excess_rebalance_days" in kinds

    def test_ignores_ad_hoc_singletons(self, isolated_db):
        from engine.auto_audit_rules import rule_rebalance_frequency_audit
        months = _last_n_completed_months(6)
        with isolated_db() as s:
            for (y, m) in months:
                _seed_bulk_trades(s, y, m, 15, n_trades=5)
            # Add singleton ad-hoc fills (1 trade each, NOT bulk)
            (y, m) = months[-1]
            for day in (3, 8, 22):
                _seed_bulk_trades(s, y, m, day, n_trades=1, ticker_prefix=f"AD{day}_")
            s.commit()
        assert rule_rebalance_frequency_audit() is None

    def test_handles_only_current_month(self, isolated_db):
        """Current partial month is excluded; only completed months audited."""
        from engine.auto_audit_rules import rule_rebalance_frequency_audit
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_bulk_trades(s, today.year, today.month,
                              min(today.day, 28), n_trades=5)
            s.commit()
        # All 6 completed months will be no_rebalance → finding fires
        result = rule_rebalance_frequency_audit()
        assert result is not None
        # Current-month trades NOT visible in violations
        for v in result["snapshot"]["violations"]:
            ym_v = v["month"]
            ym_cur = f"{today.year:04d}-{today.month:02d}"
            assert ym_v != ym_cur


# ─────────────────────────────────────────────────────────────────────────────
# Mode 12 — rule_regime_scale_vs_exposure_audit
# ─────────────────────────────────────────────────────────────────────────────
def _seed_regime(session, regime_label, date=None):
    from engine.db_models import RegimeSnapshot
    d = date or datetime.date.today()
    session.add(RegimeSnapshot(
        as_of_date=d, train_end=d, regime=regime_label,
        p_risk_on=0.3 if regime_label != "risk-on" else 0.7,
        p_risk_off=0.7 if regime_label == "risk-off" else 0.3,
        method="msm", n_obs=120,
    ))


class TestRegimeScaleExposure:
    def test_clean_overlay_disabled(self, isolated_db, monkeypatch):
        from engine import config
        monkeypatch.setattr(config, "REGIME_SCALE", 1.0)
        from engine.auto_audit_rules import rule_regime_scale_vs_exposure_audit
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_regime(s, "risk-off", today)
            _seed_position(s, today, "sec_X", "X", 0.30)
            _seed_position(s, today, "sec_Y", "Y", 0.20)   # gross_long=0.5
            s.commit()
        assert rule_regime_scale_vs_exposure_audit() is None

    def test_detects_overscaled_long(self, isolated_db, monkeypatch):
        from engine import config
        monkeypatch.setattr(config, "REGIME_SCALE", 0.6)
        from engine.auto_audit_rules import rule_regime_scale_vs_exposure_audit
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_regime(s, "risk-off", today)
            _seed_position(s, today, "sec_X", "X", 0.30)
            _seed_position(s, today, "sec_Y", "Y", 0.25)   # gross_long=0.55 > 0.4
            s.commit()
        result = rule_regime_scale_vs_exposure_audit()
        assert result is not None and result["severity"] == "MID"
        assert result["snapshot"]["gross_long"] > result["snapshot"]["max_net"]

    def test_ignores_risk_on(self, isolated_db, monkeypatch):
        from engine import config
        monkeypatch.setattr(config, "REGIME_SCALE", 0.6)
        from engine.auto_audit_rules import rule_regime_scale_vs_exposure_audit
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_regime(s, "risk-on", today)
            _seed_position(s, today, "sec_X", "X", 0.30)
            _seed_position(s, today, "sec_Y", "Y", 0.30)
            s.commit()
        assert rule_regime_scale_vs_exposure_audit() is None

    def test_handles_no_regime_data(self, isolated_db, monkeypatch):
        from engine import config
        monkeypatch.setattr(config, "REGIME_SCALE", 0.6)
        from engine.auto_audit_rules import rule_regime_scale_vs_exposure_audit
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_position(s, today, "sec_X", "X", 0.30)
            s.commit()
        assert rule_regime_scale_vs_exposure_audit() is None

    def test_ignores_low_gross_long(self, isolated_db, monkeypatch):
        from engine import config
        monkeypatch.setattr(config, "REGIME_SCALE", 0.6)
        from engine.auto_audit_rules import rule_regime_scale_vs_exposure_audit
        today = datetime.date.today()
        with isolated_db() as s:
            _seed_regime(s, "risk-off", today)
            _seed_position(s, today, "sec_X", "X", 0.15)
            _seed_position(s, today, "sec_Y", "Y", 0.10)   # gross_long=0.25 < 0.4
            s.commit()
        assert rule_regime_scale_vs_exposure_audit() is None


# ─────────────────────────────────────────────────────────────────────────────
# Mode 13 — rule_watchdog_daily_cost_budget (amendment 1, spec hash 9d050804)
# ─────────────────────────────────────────────────────────────────────────────
def _seed_ledger(tmp_path, monkeypatch, entries):
    """Point engine.llm_cost_ledger at a tmp file and seed entries."""
    import json
    from engine import llm_cost_ledger as _ledger
    ledger_path = tmp_path / "test_cost_ledger.jsonl"
    monkeypatch.setattr(_ledger, "_LEDGER_PATH", ledger_path)
    with open(ledger_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return ledger_path


class TestWatchdogDailyCostBudget:
    def test_clean_no_ledger_entries(self, tmp_path, monkeypatch):
        from engine.auto_audit_rules import rule_watchdog_daily_cost_budget
        _seed_ledger(tmp_path, monkeypatch, [])
        assert rule_watchdog_daily_cost_budget() is None

    def test_clean_under_budget(self, tmp_path, monkeypatch):
        from engine.auto_audit_rules import rule_watchdog_daily_cost_budget
        today_ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        _seed_ledger(tmp_path, monkeypatch, [
            {"ts": today_ts, "agent_id": "ops_watchdog", "provider": "gemini",
             "model": "gemini-2.5-flash", "prompt_tokens": 100,
             "completion_tokens": 50, "cost_usd": 0.15, "latency_ms": 800,
             "scope": "react_step", "extra": {}},
        ])
        assert rule_watchdog_daily_cost_budget() is None

    def test_detects_over_budget(self, tmp_path, monkeypatch):
        from engine.auto_audit_rules import rule_watchdog_daily_cost_budget
        today_ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        # 4 × $0.20 = $0.80 > $0.50 daily budget
        entries = [
            {"ts": today_ts, "agent_id": "ops_watchdog", "provider": "gemini",
             "model": "gemini-2.5-flash", "prompt_tokens": 500,
             "completion_tokens": 200, "cost_usd": 0.20, "latency_ms": 1200,
             "scope": "react_step", "extra": {}}
            for _ in range(4)
        ]
        _seed_ledger(tmp_path, monkeypatch, entries)
        result = rule_watchdog_daily_cost_budget()
        assert result is not None and result["severity"] == "HIGH"
        assert result["snapshot"]["cost_today_usd"] > 0.50
        assert result["snapshot"]["agent_id"] == "ops_watchdog"
        assert result["snapshot"]["ratio_vs_budget"] > 1.0

    def test_ignores_other_agents(self, tmp_path, monkeypatch):
        """Charges to a different agent_id must not count against Watchdog budget."""
        from engine.auto_audit_rules import rule_watchdog_daily_cost_budget
        today_ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        _seed_ledger(tmp_path, monkeypatch, [
            {"ts": today_ts, "agent_id": "r_audit", "provider": "gemini",
             "model": "gemini-2.5-flash", "prompt_tokens": 500,
             "completion_tokens": 200, "cost_usd": 5.0, "latency_ms": 1200,
             "scope": "react_step", "extra": {}},
        ])
        assert rule_watchdog_daily_cost_budget() is None

    def test_ignores_prior_day_costs(self, tmp_path, monkeypatch):
        """Prior-day costs must not contribute to today's bucket."""
        from engine.auto_audit_rules import rule_watchdog_daily_cost_budget
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        prior_ts = datetime.datetime.combine(yesterday, datetime.time(12, 0)) \
                                   .isoformat() + "Z"
        _seed_ledger(tmp_path, monkeypatch, [
            {"ts": prior_ts, "agent_id": "ops_watchdog", "provider": "gemini",
             "model": "gemini-2.5-flash", "prompt_tokens": 500,
             "completion_tokens": 200, "cost_usd": 5.0, "latency_ms": 1200,
             "scope": "react_step", "extra": {}},
        ])
        assert rule_watchdog_daily_cost_budget() is None


# ─────────────────────────────────────────────────────────────────────────────
# Meta — registration sanity
# ─────────────────────────────────────────────────────────────────────────────
def test_watchdog_rules_registered():
    from engine.auto_audit_rules import WATCHDOG_RULES
    names = {r.__name__ for r in WATCHDOG_RULES}
    expected = {
        "rule_cycle_state_completion",
        "rule_universe_data_freshness_per_ticker",
        "rule_weight_delta_p99_unexplained",
        "rule_signal_trade_referential_integrity",
        "rule_nav_move_vs_rebalance_audit",
        "rule_signal_panel_nan_scan",
        "rule_realized_tc_vs_spec_rate",
        "rule_max_position_weight_vs_cap",
        "rule_rebalance_frequency_audit",
        "rule_regime_scale_vs_exposure_audit",
        # Amendment 1, 2026-05-12 (spec hash 9d050804): mode 13 meta-monitoring
        "rule_watchdog_daily_cost_budget",
    }
    assert names == expected, f"missing/extra rules: {expected ^ names}"
    assert len(WATCHDOG_RULES) == 11
