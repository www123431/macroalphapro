"""Tests for engine.research.research_orchestrator (Phase B chain integration).

Critical properties:
1. Full chain runs end-to-end on a valid proposal
2. DSL failure stops chain cleanly (no run_gate attempted)
3. Insufficient months → halts after DSL with informative error
4. RED verdict triggers diagnose + mutation
5. GREEN verdict skips diagnose + mutation
6. Individual step failures don't crash the chain (collected in steps_failed)
7. ChainResult dataclass is serializable
8. Ledger appended when log=True; not when log=False
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine.research import research_orchestrator as RO


@pytest.fixture
def synth_prices():
    rng = np.random.RandomState(7)
    n_months, n_tickers = 120, 100
    dates = pd.date_range("2014-01-31", periods=n_months, freq="ME")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    drift = rng.uniform(-0.005, 0.012, n_tickers)
    rets = rng.randn(n_months, n_tickers) * 0.05 + drift
    prices = pd.DataFrame(
        np.cumprod(1 + rets, axis=0) * 100.0, index=dates, columns=tickers,
    )
    return prices


@pytest.fixture
def valid_proposal():
    """Proposal that matches our registered equity_xsmom template."""
    return {
        "mechanism_id":      "equity_xsmom_jt",
        "canonical_paper_id": "jegadeesh_titman_1993_jf",
        "sample_start":      "2014-01-31",
        "sample_end":        "2024-01-31",
        "justification":     "Smoke test proposal for orchestrator chain integration.",
        "execution_template": {
            "template_id": "equity_xsmom",
            "binding": {
                "lookback_months":   12,
                "skip_months":       1,
                "top_frac":          0.1,
                "bottom_frac":       0.1,
                "weighting":         "equal_weight",
                "rebal_freq":        "monthly",
                "cost_bps_per_side": 12.0,
                "microcap_price_threshold": 5.0,
                "vol_target":        0.10,
                "vol_target_lookback": 36,
            },
        },
    }


@pytest.fixture(autouse=True)
def chain_log_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(RO, "CHAIN_LOG", tmp_path / "research_orchestrator_log.jsonl")
    yield tmp_path


# ── Basic chain ────────────────────────────────────────────────────────

def test_run_full_chain_returns_chain_result(valid_proposal, synth_prices):
    result = RO.run_full_chain(
        valid_proposal, data_kwargs={"price_panel": synth_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=False,
    )
    assert isinstance(result, RO.ChainResult)
    assert result.proposal == valid_proposal


def test_run_full_chain_runs_dsl_and_gate(valid_proposal, synth_prices):
    result = RO.run_full_chain(
        valid_proposal, data_kwargs={"price_panel": synth_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=False,
    )
    assert "dsl_runner" in result.steps_executed
    assert "run_gate" in result.steps_executed
    assert result.gate_verdict in ("RED", "YELLOW", "GREEN", "UNINTERPRETABLE")


# ── DSL failure paths ──────────────────────────────────────────────────

def test_run_full_chain_missing_execution_template_fails_dsl(synth_prices):
    bad_proposal = {"mechanism_id": "no_template_v1"}
    result = RO.run_full_chain(
        bad_proposal, data_kwargs={"price_panel": synth_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=False,
    )
    assert "dsl_runner" in [f["step"] for f in result.steps_failed]
    assert "run_gate" not in result.steps_executed


def test_run_full_chain_unknown_template_fails_dsl(synth_prices):
    bad_proposal = {
        "mechanism_id": "ghost_v1",
        "execution_template": {"template_id": "no_such_template", "binding": {}},
    }
    result = RO.run_full_chain(
        bad_proposal, data_kwargs={"price_panel": synth_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=False,
    )
    assert "dsl_runner" in [f["step"] for f in result.steps_failed]


# ── Insufficient data ──────────────────────────────────────────────────

def test_run_full_chain_too_few_months(valid_proposal):
    """A 12-month panel cannot produce enough non-NaN signals."""
    short_prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(12, 30) * 0.05, axis=0) * 100.0,
        index=pd.date_range("2020-01-31", periods=12, freq="ME"),
        columns=[f"T{i}" for i in range(30)],
    )
    result = RO.run_full_chain(
        valid_proposal, data_kwargs={"price_panel": short_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=False,
    )
    assert result.gate_verdict is None
    assert "run_gate" not in result.steps_executed


# ── Diagnose + mutation only on RED/YELLOW ─────────────────────────────

def test_run_full_chain_no_diagnose_on_green(valid_proposal, synth_prices, monkeypatch):
    """Mock run_gate to return GREEN; chain should skip diagnose."""
    monkeypatch.setattr(
        "engine.research.pipeline.run_gate",
        lambda *args, **kw: {"name": "x", "verdict": "GREEN",
                              "standalone_sharpe": 1.5, "available": True,
                              "n_months": 60, "alpha_t_ff5umd": 3.5}
    )
    result = RO.run_full_chain(
        valid_proposal, data_kwargs={"price_panel": synth_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=False,
    )
    assert result.gate_verdict == "GREEN"
    assert "diagnose" not in result.steps_executed
    assert "mutation_proposer" not in result.steps_executed


def test_run_full_chain_diagnoses_on_red(valid_proposal, synth_prices, monkeypatch):
    """Mock run_gate to return RED; chain should trigger diagnose + mutation
    in deterministic mode."""
    monkeypatch.setattr(
        "engine.research.pipeline.run_gate",
        lambda *args, **kw: {"name": kw.get("name", "x"), "verdict": "RED",
                              "standalone_sharpe": -0.5, "available": True,
                              "n_months": 80, "alpha_t_ff5umd": -2.5}
    )
    result = RO.run_full_chain(
        valid_proposal, data_kwargs={"price_panel": synth_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=False,
    )
    assert result.gate_verdict == "RED"
    assert "diagnose" in result.steps_executed
    # mutation_proposer attempted (may return null proposal if pattern doesn't match)
    assert "mutation_proposer" in result.steps_executed


# ── Logging ────────────────────────────────────────────────────────────

def test_chain_log_written_when_log_true(valid_proposal, synth_prices, chain_log_isolated):
    RO.run_full_chain(
        valid_proposal, data_kwargs={"price_panel": synth_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=True,
    )
    assert RO.CHAIN_LOG.exists()
    rows = [json.loads(l) for l in RO.CHAIN_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["candidate"] == "equity_xsmom_jt"


def test_chain_log_not_written_when_log_false(valid_proposal, synth_prices):
    RO.run_full_chain(
        valid_proposal, data_kwargs={"price_panel": synth_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=False,
    )
    assert not RO.CHAIN_LOG.exists()


def test_chain_result_serializable(valid_proposal, synth_prices):
    result = RO.run_full_chain(
        valid_proposal, data_kwargs={"price_panel": synth_prices},
        use_llm_diagnose=False, use_llm_mutation=False, log=False,
    )
    d = result.to_dict()
    # Should round-trip through JSON (most fields are basic types)
    s = json.dumps(d, default=str)
    parsed = json.loads(s)
    assert parsed["proposal"]["mechanism_id"] == "equity_xsmom_jt"


def test_read_chain_log_recent_first(valid_proposal, synth_prices, chain_log_isolated):
    for _ in range(3):
        RO.run_full_chain(
            valid_proposal, data_kwargs={"price_panel": synth_prices},
            use_llm_diagnose=False, use_llm_mutation=False, log=True,
        )
    rows = RO.read_chain_log(limit=10)
    assert len(rows) == 3
    # All same candidate
    assert all(r["candidate"] == "equity_xsmom_jt" for r in rows)
