"""
Sprint H tests — engine/portfolio/attribution_logger.py + 4 strategy hook integration.

Spec: docs/spec_per_strategy_attribution_logger_v1.md
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from engine.portfolio.attribution_logger import (
    STRATEGY_SPEC_MAP,
    TradeAttribution,
    TradeLogRow,
    attributions_from_result,
    make_trade_id,
    persist_attribution_to_db,
    persist_attribution_to_jsonl,
    query_trade_log,
)
from engine.portfolio.paper_trade_combined import (
    PaperTradeRunResult,
    StrategySignal,
    get_cta_pqtix_signal,
)


def _make_fake_result(as_of: datetime.date) -> PaperTradeRunResult:
    """Synthetic PaperTradeRunResult for deterministic testing."""
    sig_k1 = StrategySignal(
        strategy_name       = "K1_BAB",
        sleeve_id           = "etf_l1",
        intra_sleeve_weight = 1.0,
        weights             = pd.Series({"SPY": 0.5, "QQQ": -0.5}),
        n_positions         = 2,
        status              = "OK",
        notes               = "test",
        trade_attributions  = (
            TradeAttribution("SPY", "long",  0.5, 1.2, as_of.isoformat(), 30, '{}'),
            TradeAttribution("QQQ", "short",-0.5,-0.8, as_of.isoformat(), 30, '{}'),
        ),
    )
    sig_d = StrategySignal(
        strategy_name       = "D_PEAD",
        sleeve_id           = "ss_sp500",
        intra_sleeve_weight = 0.5,
        weights             = pd.Series({"NVDA": 1.0}),
        n_positions         = 1,
        status              = "OK",
        notes               = "test",
        trade_attributions  = (
            TradeAttribution("NVDA", "long", 1.0, 2.31, "2026-05-10", 60,
                             '{"sue": 2.31}'),
        ),
    )
    sig_n = StrategySignal(
        strategy_name       = "PATH_N",
        sleeve_id           = "ss_sp500",
        intra_sleeve_weight = 0.5,
        weights             = pd.Series(dtype=float),
        n_positions         = 0,
        status              = "NO_SIGNAL",
        notes               = "no events",
        trade_attributions  = (),
    )
    sig_cta = StrategySignal(
        strategy_name       = "CTA_PQTIX",
        sleeve_id           = "cta_defensive",
        intra_sleeve_weight = 1.0,
        weights             = pd.Series({"PQTIX": 1.0}),
        n_positions         = 1,
        status              = "OK",
        notes               = "saa",
        trade_attributions  = (
            TradeAttribution("PQTIX", "long", 1.0, None,
                             "annual_or_2pct_drift_rebal", 0, '{}'),
        ),
    )
    return PaperTradeRunResult(
        as_of                = as_of,
        signals              = [sig_k1, sig_d, sig_n, sig_cta],
        combined_portfolio   = pd.Series(dtype=float),
        sleeve_attribution   = {},
        run_timestamp_utc    = datetime.datetime.utcnow(),
        errors               = [],
        intended_allocation  = {},
    )


# ───────────────────────────────────────────────────────────────────────────
# Test 1 — make_trade_id deterministic
# ───────────────────────────────────────────────────────────────────────────

def test_make_trade_id_deterministic():
    """Same (date, strategy, ticker) → same UUID. Different inputs → different UUIDs."""
    d = datetime.date(2026, 5, 13)
    id1 = make_trade_id(d, "D_PEAD", "NVDA")
    id2 = make_trade_id(d, "D_PEAD", "NVDA")
    assert id1 == id2, "Deterministic UUID violated"

    id3 = make_trade_id(d, "D_PEAD", "META")
    assert id1 != id3, "Different ticker should produce different UUID"

    id4 = make_trade_id(datetime.date(2026, 5, 14), "D_PEAD", "NVDA")
    assert id1 != id4, "Different date should produce different UUID"

    # Format: UUID v5
    assert len(id1) == 36 and id1.count("-") == 4


# ───────────────────────────────────────────────────────────────────────────
# Test 2 — STRATEGY_SPEC_MAP coverage
# ───────────────────────────────────────────────────────────────────────────

def test_strategy_spec_map_complete():
    """All production strategies must be in spec map.

    Updated 2026-05-18 to include AC_TLT_GLD (Tier 3 approved 2026-05-15).
    Source of truth is the registry; if a new strategy is added there,
    this test will fail until the assertion is extended.
    """
    required = {"K1_BAB", "D_PEAD", "PATH_N", "CTA_PQTIX", "AC_TLT_GLD"}
    assert set(STRATEGY_SPEC_MAP.keys()) == required

    # Validate spec IDs match known LOCKED registry.
    # 2026-05-18 evening: PATH_N/CTA/AC re-registered (was 70/73/77 doc refs,
    # now 71/72/73 DB ids) after DQ Inspector spec collision discovery.
    # See STRATEGY_HASH_GOVERNANCE_LOG in tests/test_strategy_meta_locked.py.
    assert STRATEGY_SPEC_MAP["K1_BAB"][0]     == 61
    assert STRATEGY_SPEC_MAP["D_PEAD"][0]     == 62
    assert STRATEGY_SPEC_MAP["PATH_N"][0]     == 71
    assert STRATEGY_SPEC_MAP["CTA_PQTIX"][0]  == 72
    assert STRATEGY_SPEC_MAP["AC_TLT_GLD"][0] == 73

    # Validate expected_horizon defaults
    assert STRATEGY_SPEC_MAP["K1_BAB"][2]    == 30
    assert STRATEGY_SPEC_MAP["D_PEAD"][2]    == 60
    assert STRATEGY_SPEC_MAP["PATH_N"][2]    == 5
    assert STRATEGY_SPEC_MAP["CTA_PQTIX"][2] == 0


# ───────────────────────────────────────────────────────────────────────────
# Test 3 — attributions_from_result flatten
# ───────────────────────────────────────────────────────────────────────────

def test_attributions_from_result_flatten():
    """Synthetic result → 4 rows (2 K1 + 1 D-PEAD + 0 Path N + 1 CTA)."""
    as_of = datetime.date(2026, 5, 13)
    result = _make_fake_result(as_of)
    is_rebal = {"K1_BAB": True, "D_PEAD": False, "PATH_N": False, "CTA_PQTIX": True}
    rows = attributions_from_result(result, is_rebal)

    assert len(rows) == 4, f"Expected 4 rows, got {len(rows)}"

    by_strat = {r.strategy_name: [r2 for r2 in rows if r2.strategy_name == r.strategy_name]
                for r in rows}
    assert len(by_strat["K1_BAB"])    == 2
    assert len(by_strat["D_PEAD"])    == 1
    assert len(by_strat["CTA_PQTIX"]) == 1
    assert "PATH_N" not in by_strat  # NO_SIGNAL → no rows

    # Verify is_rebalance_day propagated
    assert all(r.is_rebalance_day for r in by_strat["K1_BAB"])
    assert all(not r.is_rebalance_day for r in by_strat["D_PEAD"])
    assert all(r.is_rebalance_day for r in by_strat["CTA_PQTIX"])

    # CTA signal_value = None (passive)
    assert by_strat["CTA_PQTIX"][0].signal_value is None
    # D-PEAD signal = SUE
    assert by_strat["D_PEAD"][0].signal_value == 2.31


# ───────────────────────────────────────────────────────────────────────────
# Test 4 — persist DB idempotent
# ───────────────────────────────────────────────────────────────────────────

def test_persist_to_db_idempotent(tmp_path, monkeypatch):
    """Re-running persist for same date does NOT create duplicate rows."""
    # Isolate DB to tmp_path
    db_file = tmp_path / "test_sprint_h.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    # Reload db_models with new URL
    import importlib
    import engine.db_models as db_models
    importlib.reload(db_models)
    db_models.Base.metadata.create_all(db_models.engine)

    # Also reload attribution_logger to pick up new SessionFactory
    import engine.portfolio.attribution_logger as attr_log
    importlib.reload(attr_log)

    as_of = datetime.date(2026, 5, 13)
    result = _make_fake_result(as_of)
    is_rebal = {"K1_BAB": False, "D_PEAD": False, "PATH_N": False, "CTA_PQTIX": False}
    rows = attr_log.attributions_from_result(result, is_rebal)

    n1 = attr_log.persist_attribution_to_db(rows)
    n2 = attr_log.persist_attribution_to_db(rows)  # re-run

    assert n1 == 4
    assert n2 == 4   # merge counts the operations

    # Verify DB count is 4 (not 8)
    s = db_models.SessionFactory()
    count = s.query(db_models.PaperTradeTradeLog).filter_by(date=as_of).count()
    s.close()
    assert count == 4, f"Idempotency failed: expected 4 rows in DB, got {count}"


# ───────────────────────────────────────────────────────────────────────────
# Test 5 — JSONL append-only
# ───────────────────────────────────────────────────────────────────────────

def test_persist_to_jsonl_append(tmp_path):
    """JSONL append doesn't rewrite prior rows."""
    jsonl_path = tmp_path / "attribution_log.jsonl"

    as_of_1 = datetime.date(2026, 5, 13)
    result_1 = _make_fake_result(as_of_1)
    is_rebal = {"K1_BAB": True, "D_PEAD": False, "PATH_N": False, "CTA_PQTIX": True}
    rows_1 = attributions_from_result(result_1, is_rebal)
    n1 = persist_attribution_to_jsonl(rows_1, path=jsonl_path)
    assert n1 == 4

    # Second day
    as_of_2 = datetime.date(2026, 5, 14)
    result_2 = _make_fake_result(as_of_2)
    rows_2 = attributions_from_result(result_2, is_rebal)
    n2 = persist_attribution_to_jsonl(rows_2, path=jsonl_path)
    assert n2 == 4

    # File has both days' rows
    lines = jsonl_path.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 8

    # Validate JSON parseable
    first = json.loads(lines[0])
    last  = json.loads(lines[-1])
    assert first["date"] == "2026-05-13"
    assert last["date"]  == "2026-05-14"


# ───────────────────────────────────────────────────────────────────────────
# Test 6 — query_trade_log filters
# ───────────────────────────────────────────────────────────────────────────

def test_query_trade_log_filters(tmp_path, monkeypatch):
    """Query API filters by date / strategy / ticker / spec_id."""
    db_file = tmp_path / "test_query.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    import importlib
    import engine.db_models as db_models
    importlib.reload(db_models)
    db_models.Base.metadata.create_all(db_models.engine)
    import engine.portfolio.attribution_logger as attr_log
    importlib.reload(attr_log)

    # Persist 2 days of data
    is_rebal = {"K1_BAB": True, "D_PEAD": False, "PATH_N": False, "CTA_PQTIX": True}
    for d in [datetime.date(2026, 5, 13), datetime.date(2026, 5, 14)]:
        result = _make_fake_result(d)
        rows = attr_log.attributions_from_result(result, is_rebal)
        attr_log.persist_attribution_to_db(rows)

    df_all = attr_log.query_trade_log()
    assert len(df_all) == 8

    df_dpead = attr_log.query_trade_log(strategy_name="D_PEAD")
    assert len(df_dpead) == 2
    assert all(df_dpead["strategy_name"] == "D_PEAD")

    df_nvda = attr_log.query_trade_log(ticker="NVDA")
    assert len(df_nvda) == 2

    df_spec62 = attr_log.query_trade_log(spec_id=62)
    assert len(df_spec62) == 2

    df_may13 = attr_log.query_trade_log(
        date_start=datetime.date(2026, 5, 13),
        date_end=datetime.date(2026, 5, 13),
    )
    assert len(df_may13) == 4


# ───────────────────────────────────────────────────────────────────────────
# Test 7 — smoke end-to-end with real CTA hook
# ───────────────────────────────────────────────────────────────────────────

def test_smoke_real_cta_attribution_round_trip(tmp_path, monkeypatch):
    """Real get_cta_pqtix_signal() → attribution → DB persist → query."""
    db_file = tmp_path / "test_smoke.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    import importlib
    import engine.db_models as db_models
    importlib.reload(db_models)
    db_models.Base.metadata.create_all(db_models.engine)
    import engine.portfolio.attribution_logger as attr_log
    importlib.reload(attr_log)

    as_of = datetime.date(2026, 5, 13)
    cta_sig = get_cta_pqtix_signal(as_of)
    assert cta_sig.status == "OK"
    assert len(cta_sig.trade_attributions) == 1
    assert cta_sig.trade_attributions[0].ticker == "PQTIX"
    assert cta_sig.trade_attributions[0].signal_value is None

    # Wrap in minimal PaperTradeRunResult
    fake_result = PaperTradeRunResult(
        as_of                = as_of,
        signals              = [cta_sig],
        combined_portfolio   = pd.Series(dtype=float),
        sleeve_attribution   = {},
        run_timestamp_utc    = datetime.datetime.utcnow(),
        errors               = [],
        intended_allocation  = {},
    )
    rows = attr_log.attributions_from_result(fake_result, {"CTA_PQTIX": True})
    assert len(rows) == 1
    n = attr_log.persist_attribution_to_db(rows)
    assert n == 1

    df = attr_log.query_trade_log(strategy_name="CTA_PQTIX")
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "PQTIX"
    assert df.iloc[0]["spec_id"] == 72   # re-registered 2026-05-18 (was 73 doc ref)
    assert df.iloc[0]["spec_hash_short"] == "9630c2bb"
    assert df.iloc[0]["expected_horizon_days"] == 0
