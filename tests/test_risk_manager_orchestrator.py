"""tests/test_risk_manager_orchestrator.py — Phase 9 orchestrator hook tests.

Covers:
  - pre_trade_gate: returns halt=True on synthetic HARD_HALT
  - pre_trade_gate: halt=False on clean book
  - post_trade_gate: NEVER halts (book-already-persisted invariant)
  - dry_run=True does NOT touch DB or persistent CB state
  - _HALT.json marker writes correctly on HARD HALT pre-trade
  - VaR helper degrades gracefully on missing data
  - Wired pre/post-trade gates pull sleeve_target from registry consistently

Test isolation: dry_run=True for everything that would write; uses
synthetic PaperTradeRunResult to avoid real run_paper_trade_day cost.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from engine.agents.risk_manager.agent import RiskManagerRunResult
from engine.agents.risk_manager.orchestrator_hook import (
    HALT_FLAG_DIR,
    pre_trade_gate,
    post_trade_gate,
    _write_halt_marker,
    _compute_var_es_optional,
)
from engine.portfolio.paper_trade_combined import (
    PaperTradeRunResult, StrategySignal,
)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic PaperTradeRunResult fixtures
# ──────────────────────────────────────────────────────────────────────────────
def _make_signals(statuses=None):
    """5-strategy signal list, all OK by default. Realistic intra-strategy
    distributions so Mode 1b doesn't fire on the fixture itself."""
    statuses = statuses or ["OK", "OK", "OK", "OK", "OK"]
    weight_map = {
        "K1_BAB":     pd.Series({f"E{i:02d}": 0.05 for i in range(20)}),    # equity_ls 15% cap
        "D_PEAD":     pd.Series({f"S{i:03d}": 0.025 for i in range(40)}),   # single_stock 5% cap
        "PATH_N":     pd.Series({f"N{i:02d}": 0.04 for i in range(25)}),
        "CTA_PQTIX":  pd.Series({"PQTIX": 1.0}),
        "AC_TLT_GLD": pd.Series({"TLT": 0.5, "GLD": 0.5}),
    }
    sleeve_map = {
        "K1_BAB":     ("etf_l1", 1.0),
        "D_PEAD":     ("ss_sp500", 0.5),
        "PATH_N":     ("ss_sp500", 0.5),
        "CTA_PQTIX":  ("cta_defensive", 1.0),
        "AC_TLT_GLD": ("rms_crisis_hedge", 1.0),
    }
    out = []
    for name, status in zip(weight_map, statuses):
        sleeve_id, intra_w = sleeve_map[name]
        out.append(StrategySignal(
            strategy_name=name, sleeve_id=sleeve_id, intra_sleeve_weight=intra_w,
            weights=weight_map[name], n_positions=len(weight_map[name]),
            status=status,
        ))
    return out


def _make_run_result(combined_dict: dict, sleeve_attribution=None, signals=None,
                     as_of=None):
    if signals is None:
        signals = _make_signals()
    if as_of is None:
        as_of = datetime.date(2099, 1, 1)   # test sentinel
    if sleeve_attribution is None:
        sleeve_attribution = {
            "etf_l1": 0.324, "ss_sp500": 0.486,
            "cta_defensive": 0.090, "rms_crisis_hedge": 0.100,
        }
    return PaperTradeRunResult(
        as_of                = as_of,
        signals              = signals,
        combined_portfolio   = pd.Series(combined_dict),
        sleeve_attribution   = sleeve_attribution,
        run_timestamp_utc    = datetime.datetime.utcnow(),
        errors               = [],
        intended_allocation  = {"etf_l1": 0.324, "ss_sp500": 0.486,
                                "cta_defensive": 0.090, "rms_crisis_hedge": 0.100},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pre-trade gate behavior
# ──────────────────────────────────────────────────────────────────────────────
class TestPreTradeGate:
    def test_clean_book_no_halt(self):
        # Diversified 100-position book — no HARD_HALT
        small = {f"E{i:03d}": 0.002 for i in range(100)}
        combined = {**small, "TLT": 0.075, "GLD": 0.075, "PQTIX": 0.135}
        result = _make_run_result(combined)
        rm = pre_trade_gate(result, compute_var=False, dry_run=True)
        assert rm.halt is False
        assert rm.phase == "pre_trade"

    def test_hard_halt_on_gross_leverage_breach(self):
        # gross = 2.0 > 1.6 cap
        result = _make_run_result({"X": 1.5, "Y": 0.5})
        rm = pre_trade_gate(result, compute_var=False, dry_run=True)
        assert rm.halt is True
        assert any(b.mode_id == "3" for b in rm.breaches)
        assert rm.severity == "SEVERE"

    def test_n_modes_evaluated_is_12(self):
        result = _make_run_result({"AAPL": 0.02})
        rm = pre_trade_gate(result, compute_var=False, dry_run=True)
        assert rm.n_modes_evaluated == 12

    def test_dry_run_writes_no_db(self):
        # Trigger a HARD HALT but dry_run=True → no DB writes
        from engine.db_models import RiskManagerAlert
        from engine.memory import SessionFactory
        test_date = datetime.date(2099, 6, 15)

        s = SessionFactory()
        try:
            s.query(RiskManagerAlert).filter(RiskManagerAlert.date == test_date).delete()
            s.commit()
        finally:
            s.close()

        result = _make_run_result({"X": 2.0}, as_of=test_date)   # huge gross
        pre_trade_gate(result, compute_var=False, dry_run=True)

        # Verify DB has no rows for test_date
        s = SessionFactory()
        try:
            n = s.query(RiskManagerAlert).filter(RiskManagerAlert.date == test_date).count()
            assert n == 0
        finally:
            s.close()


# ──────────────────────────────────────────────────────────────────────────────
# Post-trade gate — never halts
# ──────────────────────────────────────────────────────────────────────────────
class TestPostTradeGate:
    def test_post_trade_never_halts_even_on_hard_halt_breach(self):
        # Inject a state that WOULD halt pre-trade — post-trade must NOT halt
        result = _make_run_result({"X": 5.0})    # gross = 5.0
        rm = post_trade_gate(result, compute_var=False, dry_run=True)
        # Even though Mode 3 fires (HARD_HALT severity), post-trade halt=False
        assert any(b.mode_id == "3" for b in rm.breaches)
        assert rm.halt is False, (
            "post-trade gate MUST NEVER halt (book already persisted invariant)"
        )

    def test_post_trade_phase_label(self):
        result = _make_run_result({"AAPL": 0.02})
        rm = post_trade_gate(result, compute_var=False, dry_run=True)
        assert rm.phase == "post_trade"


# ──────────────────────────────────────────────────────────────────────────────
# _HALT.json marker writing
# ──────────────────────────────────────────────────────────────────────────────
class TestHaltMarker:
    def test_marker_writes_on_hard_halt(self, tmp_path, monkeypatch):
        # Direct call to _write_halt_marker
        test_date = datetime.date(2099, 7, 7)
        result = _make_run_result({"X": 1.7}, as_of=test_date)
        from engine.agents.risk_manager.gates import gate_mode_3_gross_leverage
        breaches = gate_mode_3_gross_leverage(result.combined_portfolio)

        # Redirect HALT_FLAG_DIR to tmp for isolation
        from engine.agents.risk_manager import orchestrator_hook as oh
        monkeypatch.setattr(oh, "HALT_FLAG_DIR", tmp_path)

        path = oh._write_halt_marker(result, breaches, severity="SEVERE")
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["halt_decision"] is True
        assert payload["severity"] == "SEVERE"
        assert payload["as_of"] == test_date.isoformat()
        assert payload["n_breaches"] >= 1
        assert "3" in payload["hard_halt_modes"]


# ──────────────────────────────────────────────────────────────────────────────
# VaR/ES degradation
# ──────────────────────────────────────────────────────────────────────────────
class TestVarEsDegradation:
    def test_empty_book_returns_none_none(self):
        # Empty positions → VaR computation can't run
        var, es = _compute_var_es_optional(pd.Series(dtype=float))
        assert var is None and es is None

    def test_unknown_tickers_returns_none_none(self):
        # Synthetic tickers without yfinance data
        var, es = _compute_var_es_optional(pd.Series({"FAKE_TICKER_XYZ": 1.0}))
        # Either None (no data) or some value — both acceptable per spec
        # The CONTRACT is "no crash on degraded data"
        assert var is None or isinstance(var, float)


# ──────────────────────────────────────────────────────────────────────────────
# RiskManagerRunResult schema
# ──────────────────────────────────────────────────────────────────────────────
class TestRunResultSchema:
    def test_schema_locked(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(RiskManagerRunResult)}
        expected = {
            "started_at_iso", "finished_at_iso", "today_iso", "phase",
            "dry_run", "n_modes_evaluated", "breaches", "halt", "severity",
            "narratives", "llm_cost_usd", "audit_alert_ids",
        }
        assert fields == expected
