"""
tests/test_rule_sleeve_id_integrity.py — MS-7 Tier R rule unit tests.

Coverage:
  - Rule passes when DB is clean (all rows valid sleeve_id)
  - Rule catches NULL sleeve_id rows
  - Rule catches unknown sleeve_id values (typo silent sharding)
  - Rule catches rows in zero-capital-allocation sleeves
  - Rule registered in CRITICAL_RULES
"""
from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


# ── In-memory DB fixture ────────────────────────────────────────────────────
@pytest.fixture
def fresh_db(monkeypatch: pytest.MonkeyPatch):
    import engine.memory as mem
    import engine.db_models as dbm

    eng = create_engine("sqlite:///:memory:", future=True)
    dbm.Base.metadata.create_all(eng)
    SF = sessionmaker(bind=eng, future=True)
    monkeypatch.setattr(mem, "engine", eng)
    monkeypatch.setattr(mem, "SessionFactory", SF)
    return eng, SF


# ── Helper ─────────────────────────────────────────────────────────────────
def _seed_position(SF, sleeve_id="etf_l1", date=None, sector="Tech", ticker="XLK"):
    from engine.memory import SimulatedPosition
    if date is None:
        date = datetime.date(2024, 6, 28)
    with SF() as s:
        s.add(SimulatedPosition(
            snapshot_date=date, sector=sector, ticker=ticker,
            target_weight=0.10, actual_weight=0.10,
            sleeve_id=sleeve_id,
        ))
        s.commit()


# ── Rule passes on clean DB ─────────────────────────────────────────────────
def test_rule_passes_on_clean_db(fresh_db) -> None:
    _eng, SF = fresh_db
    _seed_position(SF, sleeve_id="etf_l1")
    from engine.auto_audit_rules import rule_sleeve_id_integrity
    assert rule_sleeve_id_integrity() is None


def test_rule_passes_on_empty_db(fresh_db) -> None:
    """Empty tables → no issues (vacuously true)."""
    from engine.auto_audit_rules import rule_sleeve_id_integrity
    assert rule_sleeve_id_integrity() is None


# ── Rule detects unknown sleeve_id values ──────────────────────────────────
def test_rule_catches_unknown_sleeve_id(fresh_db) -> None:
    """Bypass NOT NULL via raw SQL to inject typo'd sleeve_id."""
    eng, SF = fresh_db
    # Insert valid row first
    _seed_position(SF, sleeve_id="etf_l1")
    # Inject typo via raw SQL (bypasses ORM validation)
    with eng.connect() as conn:
        conn.execute(text(
            "INSERT INTO simulated_positions "
            "(snapshot_date, sector, ticker, target_weight, sleeve_id, track) "
            "VALUES ('2024-07-01', 'Tech', 'XLK', 0.1, 'etfl1_typo', 'main')"
        ))
        conn.commit()

    from engine.auto_audit_rules import rule_sleeve_id_integrity
    result = rule_sleeve_id_integrity()
    assert result is not None
    assert result["severity"] == "HIGH"

    issues = result["snapshot"]["issues"]
    unknown_kinds = [i for i in issues if i.get("kind") == "unknown_sleeve_id_values"]
    assert len(unknown_kinds) > 0
    assert "etfl1_typo" in unknown_kinds[0]["unknown"]


# ── Rule detects rows in zero-capital sleeve ───────────────────────────────
def test_rule_catches_zero_capital_sleeve_writes(
    fresh_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial config: ss_sp500 = 0%. Writing rows there → integrity flag."""
    _eng, SF = fresh_db
    _seed_position(SF, sleeve_id="ss_sp500", ticker="AAPL")

    from engine.auto_audit_rules import rule_sleeve_id_integrity
    result = rule_sleeve_id_integrity()
    assert result is not None
    issues = result["snapshot"]["issues"]
    zero_cap = [i for i in issues if i.get("kind") == "rows_in_zero_capital_sleeve"]
    assert len(zero_cap) > 0
    assert zero_cap[0]["sleeve_id"] == "ss_sp500"


def test_rule_skips_zero_capital_check_when_active_config_unavailable(
    fresh_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If get_active_config raises, skip the check rather than crash."""
    _eng, SF = fresh_db
    _seed_position(SF, sleeve_id="etf_l1")

    def _raise(*a, **kw):
        raise RuntimeError("simulated SystemConfig unreachable")
    monkeypatch.setattr("engine.portfolio_sleeves.get_active_config", _raise)

    from engine.auto_audit_rules import rule_sleeve_id_integrity
    # Should not raise; clean DB → returns None despite get_active_config failure
    assert rule_sleeve_id_integrity() is None


# ── Registration ────────────────────────────────────────────────────────────
def test_rule_registered_in_critical_rules() -> None:
    from engine.auto_audit_rules import rule_sleeve_id_integrity, CRITICAL_RULES
    assert rule_sleeve_id_integrity in CRITICAL_RULES
