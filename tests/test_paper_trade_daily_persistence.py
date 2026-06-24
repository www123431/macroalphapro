"""
tests/test_paper_trade_daily_persistence.py — Sprint D-2 persistence + idempotency.

Tests for persist_run_to_db() bridge: PaperTradeRunResult → PaperTradeStrategyLog.
"""
from __future__ import annotations

import datetime
import json

import pandas as pd
import pytest


def _build_synthetic_run_result(as_of: datetime.date) -> "PaperTradeRunResult":
    """Synthetic PaperTradeRunResult with 4 strategies for testing persistence."""
    from engine.portfolio.paper_trade_combined import (
        PaperTradeRunResult, StrategySignal, PAPER_TRADE_SLEEVE_ALLOCATION,
    )

    signals = [
        StrategySignal(
            strategy_name="K1_BAB", sleeve_id="etf_l1", intra_sleeve_weight=1.0,
            weights=pd.Series({"SPY": 0.5, "QQQ": -0.5}),
            n_positions=2, status="OK", notes="synthetic K1",
        ),
        StrategySignal(
            strategy_name="D_PEAD", sleeve_id="ss_sp500", intra_sleeve_weight=0.5,
            weights=pd.Series({"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}),
            n_positions=3, status="OK", notes="synthetic D-PEAD",
        ),
        StrategySignal(
            strategy_name="PATH_N", sleeve_id="ss_sp500", intra_sleeve_weight=0.5,
            weights=pd.Series(dtype=float),
            n_positions=0, status="NO_SIGNAL", notes="no pending events",
        ),
        StrategySignal(
            strategy_name="CTA_PQTIX", sleeve_id="cta_defensive", intra_sleeve_weight=1.0,
            weights=pd.Series({"PQTIX": 1.0}),
            n_positions=1, status="OK", notes="passive hold",
        ),
    ]
    return PaperTradeRunResult(
        as_of=as_of,
        signals=signals,
        combined_portfolio=pd.Series(dtype=float),
        sleeve_attribution={},
        run_timestamp_utc=datetime.datetime.utcnow(),
        errors=[],
        intended_allocation=dict(PAPER_TRADE_SLEEVE_ALLOCATION),
    )


def test_persist_run_to_db_inserts_then_updates():
    """First call inserts; second call for same date updates in place."""
    from engine.portfolio.paper_trade_combined import persist_run_to_db
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeStrategyLog

    test_date = datetime.date(2099, 6, 15)  # far-future to avoid collision
    init_db()
    sess = SessionFactory()
    # Cleanup any prior test rows
    sess.query(PaperTradeStrategyLog).filter_by(date=test_date).delete()
    sess.commit()
    sess.close()

    # First run: 4 inserts
    result = _build_synthetic_run_result(test_date)
    counts1 = persist_run_to_db(result)
    assert counts1["inserted"] == 4
    assert counts1["updated"] == 0
    assert counts1["errors"] == 0

    # Read back: 4 rows, statuses correct
    sess = SessionFactory()
    rows = sess.query(PaperTradeStrategyLog).filter_by(date=test_date).order_by(
        PaperTradeStrategyLog.strategy_name).all()
    assert len(rows) == 4
    by_name = {r.strategy_name: r for r in rows}
    assert by_name["K1_BAB"].status == "OK"
    assert by_name["K1_BAB"].n_positions == 2
    assert by_name["PATH_N"].status == "NO_SIGNAL"
    assert by_name["PATH_N"].n_positions == 0
    assert by_name["CTA_PQTIX"].n_positions == 1
    # positions_json must be valid JSON
    pos = json.loads(by_name["K1_BAB"].positions_json)
    assert "SPY" in pos and abs(pos["SPY"] - 0.5) < 1e-9
    sess.close()

    # Second run for same date: 4 updates (no inserts)
    result2 = _build_synthetic_run_result(test_date)
    counts2 = persist_run_to_db(result2)
    assert counts2["inserted"] == 0
    assert counts2["updated"] == 4
    assert counts2["errors"] == 0

    # Cleanup
    sess = SessionFactory()
    sess.query(PaperTradeStrategyLog).filter_by(date=test_date).delete()
    sess.commit()
    sess.close()


def test_persist_run_to_db_handles_empty_weights():
    """Strategy with empty weights persists with empty positions_json."""
    from engine.portfolio.paper_trade_combined import persist_run_to_db
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeStrategyLog

    test_date = datetime.date(2099, 7, 1)
    init_db()

    result = _build_synthetic_run_result(test_date)
    persist_run_to_db(result)

    sess = SessionFactory()
    try:
        path_n = sess.query(PaperTradeStrategyLog).filter_by(
            date=test_date, strategy_name="PATH_N",
        ).first()
        assert path_n is not None
        assert path_n.status == "NO_SIGNAL"
        assert path_n.n_positions == 0
        # Empty weights → empty JSON dict {}
        pos = json.loads(path_n.positions_json)
        assert pos == {}

        # Cleanup
        sess.query(PaperTradeStrategyLog).filter_by(date=test_date).delete()
        sess.commit()
    finally:
        sess.close()


def test_persist_run_to_db_rebalance_day_flag_computed():
    """persist_run_to_db should compute and write is_rebalance_day per strategy."""
    from engine.portfolio.paper_trade_combined import persist_run_to_db
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeStrategyLog

    # CTA Dec 31 is rebalance day; K1 mid-month is not
    test_date = datetime.date(2099, 12, 31)
    init_db()
    result = _build_synthetic_run_result(test_date)
    persist_run_to_db(result)

    sess = SessionFactory()
    try:
        cta = sess.query(PaperTradeStrategyLog).filter_by(
            date=test_date, strategy_name="CTA_PQTIX",
        ).first()
        # CTA Dec 31 → is_rebalance_day = True
        assert cta.is_rebalance_day is True

        k1 = sess.query(PaperTradeStrategyLog).filter_by(
            date=test_date, strategy_name="K1_BAB",
        ).first()
        # K1 EOM = True for Dec 31
        assert k1.is_rebalance_day is True

        # Cleanup
        sess.query(PaperTradeStrategyLog).filter_by(date=test_date).delete()
        sess.commit()
    finally:
        sess.close()


def test_daily_runner_script_importable():
    """scripts/run_paper_trade_daily.py is importable as a module."""
    import importlib.util
    from pathlib import Path
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_paper_trade_daily.py"
    assert script_path.exists()

    spec = importlib.util.spec_from_file_location("run_paper_trade_daily", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Verify expected functions exist
    assert hasattr(mod, "step_refresh_sp500_feed")
    assert hasattr(mod, "step_run_orchestrator")
    assert hasattr(mod, "step_persist_to_db")
    assert hasattr(mod, "main")
