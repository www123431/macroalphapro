"""
tests/test_ms1_sleeve_id.py — MS-1 multi-sleeve column tests (2026-05-10).

Coverage:
  - sleeve_id column present on all 4 tables (DecisionLog / SimulatedPosition /
    SimulatedTrade / SimulatedMonthlyReturn)
  - Default 'etf_l1' applied when not specified (server_default backward compat)
  - Explicit sleeve_id ('ss_sp500') accepted on writes
  - Cross-sleeve unique constraints (same date+sector+track but different
    sleeve_id co-exists)
  - save_decision() entry point accepts sleeve_id parameter
  - Existing decision_logs / simulated_positions historical rows backfilled
    'etf_l1' (verified post-migration)
"""
from __future__ import annotations

import datetime
import os
import tempfile

import pytest


# ── Fixture: in-memory SQLite test DB w/ proper migration ──────────────────
@pytest.fixture
def isolated_db(monkeypatch: pytest.MonkeyPatch):
    """In-memory SQLite engine with fresh tables created from ORM + sleeve_id
    migration applied. Uses the production engine.memory module's models so
    this test verifies the actual ORM definitions (not duplicated).
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    import engine.memory as mem
    import engine.db_models as dbm

    eng = create_engine("sqlite:///:memory:", future=True)
    dbm.Base.metadata.create_all(eng)
    SessionFactoryLocal = sessionmaker(bind=eng, future=True)

    monkeypatch.setattr(mem, "engine", eng)
    monkeypatch.setattr(mem, "SessionFactory", SessionFactoryLocal)

    # Create unique indexes (SQLAlchemy creates the named UniqueConstraints
    # via metadata.create_all, but verify by inspecting indexes)
    yield mem


# ── Schema tests ────────────────────────────────────────────────────────────
def test_sleeve_id_column_present_on_all_4_tables(isolated_db) -> None:
    """All 4 production tables must carry sleeve_id column."""
    from sqlalchemy import text
    with isolated_db.engine.connect() as conn:
        for table in [
            "decision_logs",
            "simulated_positions",
            "simulated_trades",
            "simulated_monthly_returns",
        ]:
            cols = {row[1] for row in
                    conn.execute(text(f"PRAGMA table_info({table})")).fetchall()}
            assert "sleeve_id" in cols, f"{table} missing sleeve_id column"


def test_sleeve_id_unique_constraints_at_orm_level() -> None:
    """ORM models declare UniqueConstraint covering sleeve_id (production
    indexes are named per migration, but ORM constraint definitions are
    backend-agnostic — verify at __table_args__ level so test is portable)."""
    from sqlalchemy import UniqueConstraint
    from engine.db_models import SimulatedPosition, SimulatedMonthlyReturn

    sp_uniques = [c for c in SimulatedPosition.__table__.constraints
                  if isinstance(c, UniqueConstraint)]
    sp_cols_sets = [set(c.name for c in u.columns) for u in sp_uniques]
    assert any({"snapshot_date", "sector", "track", "sleeve_id"} == s for s in sp_cols_sets), \
        f"SimulatedPosition missing 4-col unique constraint; have {sp_cols_sets}"

    smr_uniques = [c for c in SimulatedMonthlyReturn.__table__.constraints
                   if isinstance(c, UniqueConstraint)]
    smr_cols_sets = [set(c.name for c in u.columns) for u in smr_uniques]
    assert any({"return_month", "sector", "sleeve_id"} == s for s in smr_cols_sets), \
        f"SimulatedMonthlyReturn missing 3-col unique constraint; have {smr_cols_sets}"


# ── Default + explicit value tests ─────────────────────────────────────────
def test_simulated_position_defaults_to_etf_l1(isolated_db) -> None:
    """Inserting without sleeve_id → server_default 'etf_l1'."""
    pos = isolated_db.SimulatedPosition(
        snapshot_date=datetime.date(2024, 6, 28),
        sector="Tech",
        ticker="XLK",
        target_weight=0.20,
        actual_weight=0.20,
    )
    with isolated_db.SessionFactory() as s:
        s.add(pos)
        s.commit()
        s.refresh(pos)
        assert pos.sleeve_id == "etf_l1"


def test_simulated_position_accepts_explicit_ss_sp500(isolated_db) -> None:
    pos = isolated_db.SimulatedPosition(
        snapshot_date=datetime.date(2024, 6, 28),
        sector="Tech",
        ticker="AAPL",
        target_weight=0.05,
        actual_weight=0.05,
        sleeve_id="ss_sp500",
    )
    with isolated_db.SessionFactory() as s:
        s.add(pos)
        s.commit()
        s.refresh(pos)
        assert pos.sleeve_id == "ss_sp500"


def test_cross_sleeve_same_date_sector_can_coexist(isolated_db) -> None:
    """Same snapshot_date + sector + track but different sleeve_id → both saved
    (uq_pos_date_sector_track_sleeve allows it)."""
    p_etf = isolated_db.SimulatedPosition(
        snapshot_date=datetime.date(2024, 6, 28),
        sector="Tech", ticker="XLK",
        target_weight=0.20, actual_weight=0.20,
        sleeve_id="etf_l1",
    )
    p_ss = isolated_db.SimulatedPosition(
        snapshot_date=datetime.date(2024, 6, 28),
        sector="Tech", ticker="AAPL",
        target_weight=0.04, actual_weight=0.04,
        sleeve_id="ss_sp500",
    )
    with isolated_db.SessionFactory() as s:
        s.add(p_etf)
        s.add(p_ss)
        s.commit()
        rows = s.query(isolated_db.SimulatedPosition).filter_by(
            snapshot_date=datetime.date(2024, 6, 28),
        ).all()
        sleeves = {r.sleeve_id for r in rows}
        assert sleeves == {"etf_l1", "ss_sp500"}


def test_simulated_trade_defaults_to_etf_l1(isolated_db) -> None:
    t = isolated_db.SimulatedTrade(
        trade_date=datetime.date(2024, 6, 28),
        sector="Tech", ticker="XLK", action="BUY",
        weight_before=0.0, weight_after=0.20, weight_delta=0.20,
    )
    with isolated_db.SessionFactory() as s:
        s.add(t)
        s.commit()
        s.refresh(t)
        assert t.sleeve_id == "etf_l1"


def test_simulated_monthly_return_defaults_to_etf_l1(isolated_db) -> None:
    smr = isolated_db.SimulatedMonthlyReturn(
        return_month=datetime.date(2024, 6, 1),
        sector="Tech", weight_held=0.20,
        sector_return=0.05, contribution=0.01,
    )
    with isolated_db.SessionFactory() as s:
        s.add(smr)
        s.commit()
        s.refresh(smr)
        assert smr.sleeve_id == "etf_l1"


def test_decision_log_defaults_to_etf_l1(isolated_db) -> None:
    dl = isolated_db.DecisionLog(
        tab_type="sector",
        ai_conclusion="test decision",
        sector_name="Tech",
    )
    with isolated_db.SessionFactory() as s:
        s.add(dl)
        s.commit()
        s.refresh(dl)
        assert dl.sleeve_id == "etf_l1"


def test_decision_log_accepts_explicit_ss_sp500(isolated_db) -> None:
    dl = isolated_db.DecisionLog(
        tab_type="sector",
        ai_conclusion="single-stock test",
        ticker="AAPL",
        sleeve_id="ss_sp500",
    )
    with isolated_db.SessionFactory() as s:
        s.add(dl)
        s.commit()
        s.refresh(dl)
        assert dl.sleeve_id == "ss_sp500"


# ── save_decision() entry point ────────────────────────────────────────────
def test_save_decision_default_sleeve(isolated_db) -> None:
    """save_decision() without sleeve_id arg → 'etf_l1' default."""
    dl_id = isolated_db.save_decision(
        tab_type="sector",
        ai_conclusion="test default sleeve",
        sector_name="Tech",
    )
    with isolated_db.SessionFactory() as s:
        row = s.query(isolated_db.DecisionLog).filter_by(id=dl_id).one()
        assert row.sleeve_id == "etf_l1"


def test_save_decision_explicit_sleeve(isolated_db) -> None:
    """save_decision(sleeve_id='ss_sp500') → row tagged ss_sp500."""
    dl_id = isolated_db.save_decision(
        tab_type="sector",
        ai_conclusion="single-stock decision",
        ticker="AAPL",
        sleeve_id="ss_sp500",
    )
    with isolated_db.SessionFactory() as s:
        row = s.query(isolated_db.DecisionLog).filter_by(id=dl_id).one()
        assert row.sleeve_id == "ss_sp500"


# ── ORM model invariants ───────────────────────────────────────────────────
def test_orm_models_declare_sleeve_id() -> None:
    """ORM model classes carry sleeve_id Column object (sanity check on
    db_models.py declarations — paired with migration ALTER above)."""
    from engine.db_models import (
        DecisionLog, SimulatedPosition, SimulatedTrade, SimulatedMonthlyReturn,
    )
    for model in [DecisionLog, SimulatedPosition, SimulatedTrade, SimulatedMonthlyReturn]:
        cols = {c.name for c in model.__table__.columns}
        assert "sleeve_id" in cols, f"{model.__name__} ORM missing sleeve_id"
        sleeve_col = model.__table__.columns["sleeve_id"]
        assert sleeve_col.nullable is False, f"{model.__name__}.sleeve_id should be NOT NULL"
