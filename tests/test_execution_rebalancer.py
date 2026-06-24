"""tests/test_execution_rebalancer.py — broker-agnostic execution layer (Phase B).

Pins: (1) compute_orders arithmetic (targets vs current → correct deltas, anti-churn min, liquidate
untargeted), (2) SimAdapter round-trip reaches the target book, (3) PAPER-ONLY guards refuse live
trading, (4) the execution layer imports NO LLM (0-LLM-in-DECISION: execution is plumbing).
"""
from __future__ import annotations

import pathlib

import pytest

from engine.execution.broker import Account, ExecutionAdapter, Fill, Order, Position
from engine.execution.rebalancer import compute_orders, rebalance
from engine.execution.sim_adapter import SimAdapter


def test_compute_orders_basic_from_flat():
    acct = Account(equity=100_000.0, cash=100_000.0)
    orders, skipped, warns = compute_orders(
        {"AAPL": 0.5, "MSFT": 0.5}, acct, {}, {"AAPL": 100.0, "MSFT": 200.0})
    by = {o.ticker: o.qty for o in orders}
    assert by["AAPL"] == pytest.approx(500.0)    # 50k / 100
    assert by["MSFT"] == pytest.approx(250.0)     # 50k / 200
    assert not skipped


def test_compute_orders_min_threshold_skips_churn():
    acct = Account(equity=100_000.0)
    # holding exactly target; a $10 drift is below the $50 min → skipped, no order
    pos = {"AAPL": Position("AAPL", qty=500.1)}
    orders, skipped, _ = compute_orders(
        {"AAPL": 0.5}, acct, pos, {"AAPL": 100.0}, min_order_usd=50.0)
    assert orders == []
    assert "AAPL" in skipped


def test_compute_orders_liquidates_untargeted_holding():
    acct = Account(equity=100_000.0)
    pos = {"OLD": Position("OLD", qty=100.0)}
    orders, _, _ = compute_orders({"NEW": 1.0}, acct, pos, {"OLD": 50.0, "NEW": 10.0})
    by = {o.ticker: o.qty for o in orders}
    assert by["OLD"] == pytest.approx(-100.0)     # fully liquidated
    assert by["NEW"] > 0


def test_sim_roundtrip_reaches_target(tmp_path):
    sim = SimAdapter(starting_cash=1_000_000.0, state_path=str(tmp_path / "s.json"), reset=True)
    sim.set_prices({"AAPL": 100.0, "MSFT": 200.0})
    rep = rebalance(sim, {"AAPL": 0.3, "MSFT": 0.2})
    assert rep.paper is True
    pos = sim.get_positions()
    acct = sim.get_account()
    # 30% of ~1M into AAPL @100, 20% into MSFT @200
    assert pos["AAPL"].qty == pytest.approx(3000.0, rel=1e-3)
    assert pos["MSFT"].qty == pytest.approx(1000.0, rel=1e-3)
    # equity conserved (no slippage): cash + MV ≈ starting
    assert acct.equity == pytest.approx(1_000_000.0, rel=1e-6)


def test_sim_nav_track_persists(tmp_path):
    p = str(tmp_path / "s.json")
    sim = SimAdapter(starting_cash=500_000.0, state_path=p, reset=True)
    sim.set_prices({"SPY": 400.0})
    rebalance(sim, {"SPY": 1.0})
    sim.mark_nav("2026-05-27")
    # reload from disk → state survived
    sim2 = SimAdapter(state_path=p)
    assert sim2.get_positions()["SPY"].qty == pytest.approx(1250.0, rel=1e-3)
    assert sim2.nav_history()[-1]["date"] == "2026-05-27"


class _LiveStub(ExecutionAdapter):
    name = "live_stub"

    @property
    def is_paper(self) -> bool:
        return False

    def get_account(self): return Account(equity=1.0)
    def get_positions(self): return {}
    def get_prices(self, tickers): return {}
    def submit_order(self, order): return Fill(order.ticker, order.qty, 1.0)


def test_rebalancer_refuses_non_paper_adapter():
    with pytest.raises(RuntimeError, match="NOT a paper"):
        rebalance(_LiveStub(), {"AAPL": 1.0})
    # dry_run is allowed (computes but never submits)
    rep = rebalance(_LiveStub(), {"AAPL": 1.0}, dry_run=True)
    assert any("dry_run" in w for w in rep.warnings)


def test_alpaca_refuses_live_endpoint():
    from engine.execution.alpaca_adapter import AlpacaAdapter, AlpacaConfigError
    with pytest.raises(AlpacaConfigError, match="non-paper"):
        AlpacaAdapter(base_url="https://api.alpaca.markets", key="k", secret="s")


def test_reconcile_on_target_after_rebalance(tmp_path):
    from engine.execution.reconcile import reconcile
    sim = SimAdapter(starting_cash=1_000_000.0, state_path=str(tmp_path / "s.json"), reset=True)
    sim.set_prices({"AAPL": 100.0, "MSFT": 200.0})
    rebalance(sim, {"AAPL": 0.3, "MSFT": 0.2})
    rec = reconcile(sim, {"AAPL": 0.3, "MSFT": 0.2})
    assert rec["tracking_error"] == pytest.approx(0.0, abs=1e-4)
    assert rec["gross_actual"] == pytest.approx(0.5, rel=1e-3)
    assert rec["breaks"]["targeted_not_held"] == []
    assert rec["breaks"]["held_not_targeted"] == []
    assert rec["n_on_target"] == 2


def test_reconcile_surfaces_breaks(tmp_path):
    """A target we never bought → targeted_not_held; a holding no longer targeted → held_not_targeted."""
    from engine.execution.reconcile import reconcile
    sim = SimAdapter(starting_cash=1_000_000.0, state_path=str(tmp_path / "s.json"), reset=True)
    sim.set_prices({"AAPL": 100.0, "OLD": 50.0, "WANT": 10.0})
    sim._state.qty["OLD"] = 100.0           # holding an untargeted name
    # target WANT but give it no chance to fill (reconcile only, no rebalance)
    rec = reconcile(sim, {"AAPL": 0.3, "WANT": 0.2})
    assert "WANT" in rec["breaks"]["targeted_not_held"]   # wanted, not held
    assert "AAPL" in rec["breaks"]["targeted_not_held"]
    assert "OLD" in rec["breaks"]["held_not_targeted"]     # held, not wanted
    assert rec["tracking_error"] > 0


def test_multi_venue_consolidation_and_short_residual(tmp_path):
    """Two venues (equity sim + futures sim) → one book NAV; an unborrowable SHORT target that no
    venue holds is flagged as a short-borrow residual (the honest Alpaca-vs-IB gap, measured)."""
    from engine.execution.multi_venue import consolidate, reconcile_multi
    eqv = SimAdapter(starting_cash=600_000.0, state_path=str(tmp_path / "eq.json"), reset=True)
    eqv.set_prices({"AAPL": 100.0, "BADSHORT": 20.0})
    rebalance(eqv, {"AAPL": 0.5})                       # long AAPL; BADSHORT (a short) NOT placed
    futv = SimAdapter(starting_cash=400_000.0, state_path=str(tmp_path / "fut.json"), reset=True)
    futv.set_prices({"CL": 75.0})
    rebalance(futv, {"CL": 0.25})

    adapters = {"alpaca": eqv, "futures_sim": futv}
    con = consolidate(adapters)
    assert con["total_equity"] == pytest.approx(1_000_000.0, rel=1e-4)   # cash summed across venues
    assert con["position_venue"]["CL"] == "futures_sim"

    # full model wants AAPL long, CL long, and a BADSHORT short that couldn't be borrowed
    target = {"AAPL": 0.5, "CL": 0.25, "BADSHORT": -0.05}
    rec = reconcile_multi(adapters, target)
    assert "BADSHORT" in rec["breaks"]["short_borrow_residual"]
    assert rec["breaks"]["short_borrow_residual_weight"] == pytest.approx(0.05, abs=1e-6)
    assert rec["per_venue"]["alpaca"]["paper"] is True


def test_futures_sim_whole_contracts_and_pnl_accounting(tmp_path):
    """Futures sim: positions are WHOLE contracts at real notionals; futures accounting (buying a
    future costs NO notional, only slippage; NAV moves by contracts×notional×return)."""
    from engine.execution.futures_sim_adapter import FuturesSimAdapter
    fs = FuturesSimAdapter(starting_equity=10_000_000.0, use_micro=False, slippage_bps=1.0,
                           state_path=str(tmp_path / "f.json"), reset=True)
    fs.seed_notionals_from_specs(["CL_WTI", "GC_Gold", "EUR"])
    rebalance(fs, {"CL_WTI": 0.2, "GC_Gold": 0.1, "EUR": 0.1})
    pos = fs.get_positions()
    # whole contracts only
    for p in pos.values():
        assert abs(p.qty - round(p.qty)) < 1e-9, f"{p.ticker} not whole: {p.qty}"
    # CL target 0.2*10M / 75k ≈ 26.7 → 27 contracts
    assert pos["CL_WTI"].qty == pytest.approx(27.0)
    # futures accounting: equity ≈ 10M (only slippage debited, NOT the ~$3.5M notional)
    assert fs.get_account().equity > 9_990_000.0
    # mark a +5% CL move → equity += 27 × 75k × 0.05
    eq0 = fs.get_account().equity
    fs.mark({"CL_WTI": 0.05}, date="2026-05-31")
    assert fs.get_account().equity == pytest.approx(eq0 + 27 * 75_000 * 0.05, rel=1e-6)


def test_futures_sim_scale_fixes_lumpiness(tmp_path):
    """The whole point of scaling: at $10M the contract rounding tracking error is small."""
    from engine.execution.futures_sim_adapter import FuturesSimAdapter
    syms = ["CL_WTI", "GC_Gold", "EUR", "UST10"]
    fs = FuturesSimAdapter(starting_equity=10_000_000.0, use_micro=True,
                           state_path=str(tmp_path / "f.json"), reset=True)
    fs.seed_notionals_from_specs(syms)
    tgt = {s: 0.15 for s in syms}
    rebalance(fs, tgt)
    eq = fs.get_account().equity
    pos = fs.get_positions()
    for s in syms:
        actual_w = pos[s].market_value / eq if s in pos else 0.0
        assert abs(actual_w - 0.15) < 0.03, f"{s} weight {actual_w} far from 0.15 target"


@pytest.mark.skipif(
    not __import__("os").path.exists("data/cache/_cmdty_settle.parquet"),
    reason="futures cache absent")
def test_futures_sim_replay_reproduces_B_at_scale():
    """The realism layer (whole contracts + futures accounting + slippage) must NOT break the strategy:
    at $10M+micro the FuturesSimAdapter replay of sleeve B tracks the frictionless cta_trend tightly."""
    from engine.execution.futures_book import replay_b
    r = replay_b(book_equity=10_000_000.0, use_micro=True)
    assert r["corr_sim_vs_frictionless"] is not None and r["corr_sim_vs_frictionless"] > 0.95
    assert r["tracking_error_ann_pct"] < 2.0
    assert abs(r["sim_sharpe"] - r["frictionless_sharpe"]) < 0.1


@pytest.mark.skipif(
    not __import__("os").path.exists("data/cache/_cmdty_settle.parquet"),
    reason="futures cache absent")
def test_futures_sim_replay_reproduces_combined_carry_plus_B():
    """The FULL futures sleeve (carry ⊕ B) runs through the realistic sim at $10M and tracks the
    frictionless combined tightly — diversification uplift preserved (combined Sharpe > either leg)."""
    from engine.execution.futures_book import replay_combined
    r = replay_combined(book_equity=10_000_000.0, use_micro=True)
    assert r["corr_sim_vs_frictionless"] > 0.95
    assert r["tracking_error_ann_pct"] < 2.0
    assert r["frictionless_sharpe"] > 0.7        # carry⊕B diversification beats either alone
    assert abs(r["sim_sharpe"] - r["frictionless_sharpe"]) < 0.1


def test_book_risk_weights_read_from_source_not_hardcoded():
    """The equity/carry split must come from combined_book.DEFAULT_CARRY_RISK_WEIGHT (single source),
    not a hardcoded guess; equity = 1 - carry, sums to 1."""
    from engine.execution.multi_venue import book_risk_weights
    from engine.portfolio.combined_book import DEFAULT_CARRY_RISK_WEIGHT
    rw = book_risk_weights()
    assert rw["carry"] == pytest.approx(DEFAULT_CARRY_RISK_WEIGHT)
    assert rw["equity"] == pytest.approx(1.0 - DEFAULT_CARRY_RISK_WEIGHT)
    assert rw["equity"] + rw["carry"] == pytest.approx(1.0)
    assert "combined_book" in rw["source"]


def test_risk_weighted_book_return_is_scale_invariant():
    """Blending RETURNS at the risk split is unaffected by each venue's dollar scale (the $100k-vs-
    $10M mismatch can't distort the book) — and uses the code-sourced weights."""
    import pandas as pd
    from engine.execution.multi_venue import risk_weighted_book_return, book_risk_weights
    idx = pd.date_range("2026-01-31", periods=6, freq="ME")
    eq = pd.Series([0.01, -0.02, 0.03, 0.0, 0.015, -0.01], index=idx)
    cy = pd.Series([0.005, 0.01, -0.01, 0.02, -0.005, 0.0], index=idx)
    rw = book_risk_weights()
    book = risk_weighted_book_return(eq, cy)
    expected = rw["equity"] * eq + rw["carry"] * cy
    assert (book - expected).abs().max() < 1e-12
    # scale-invariance: a venue's $-scale lives in its NAV, not its return → same return ⇒ same book
    assert (risk_weighted_book_return(eq, cy) - book).abs().max() < 1e-12


def test_execution_layer_imports_no_llm():
    """0-LLM-in-DECISION: the execution plumbing must not import any LLM/agent machinery."""
    root = pathlib.Path(__file__).resolve().parents[1] / "engine" / "execution"
    forbidden = ("anthropic", "openai", "deepseek", "chat_turn", "persona", "agents.persona",
                 "google.genai", "litellm")
    for py in root.glob("*.py"):
        src = py.read_text(encoding="utf-8").lower()
        for tok in forbidden:
            assert tok.lower() not in src, f"{py.name} references LLM token '{tok}'"
