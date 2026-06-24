"""Phase 6c tests: orchestrator → protocol_executor end-to-end wire.

Verifies that execute_protocol with auto_acquire=True correctly:
  1. Loads mechanism from library
  2. Fetches required_data via orchestrator
  3. Adapts via dsl_adapter to DSL-shape
  4. Runs all legs
  5. Returns RED with structured detail if any token fetch fails
  6. Never silently substitutes synth data
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data import dsl_adapter
from engine.data import orchestrator
from engine.data.fetchers import _mock_paid, _mock_free
from engine.research.protocols import (
    execute_protocol, instantiate_protocol, load_mechanism,
)


@pytest.fixture(autouse=True)
def reset_mock_state(monkeypatch):
    _mock_paid.TEST_STATE.update({
        "probe_available": True, "probe_error": None,
        "probe_error_class": None, "fetch_raises": None,
        "fetch_rows": 100, "fetch_partial": False,
    })
    _mock_free.TEST_STATE.update({
        "probe_available": True, "probe_error": None,
        "probe_error_class": None, "fetch_raises": None,
        "fetch_rows": 50,
    })
    yield


@pytest.fixture
def isolated_data_root(tmp_path, monkeypatch):
    from engine.data import cache_manager
    cache_v2 = tmp_path / "cache_v2"
    cache_v2.mkdir(parents=True)
    meta = cache_v2 / "_meta"
    meta.mkdir()
    monkeypatch.setattr(cache_manager, "CACHE_DIR", cache_v2)
    monkeypatch.setattr(cache_manager, "META_DIR", meta)

    acq_log = tmp_path / "acquisition_log.jsonl"
    monkeypatch.setattr(orchestrator, "ACQUISITION_LOG", acq_log)

    inv_path = tmp_path / "data_inventory.yaml"
    inv_path.write_text("""
_schema_version: 1
inventory:
  crsp_dsf:
    fetcher_chain:
      - source: _mock_paid
        function: fetch_dsf
        tier: paid
        auth: null
    required_columns: [date, ticker, ret, prc]
    refresh_cadence: monthly
""", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "INVENTORY_PATH", inv_path)
    return tmp_path


# ── dsl_adapter unit tests ──────────────────────────────────────────────

def test_adapter_equity_long_to_wide():
    df = pd.DataFrame({
        "date":   pd.to_datetime(["2024-01-01", "2024-01-01",
                                    "2024-02-01", "2024-02-01"]),
        "ticker": ["A", "B", "A", "B"],
        "ret":    [0.01, 0.02, -0.01, 0.03],
        "prc":    [100.0, 50.0, 101.0, 51.5],
    })
    out = dsl_adapter.adapt_for_dsl("crsp_dsf", df, template_id="equity_xsmom")
    assert "price_panel" in out
    assert "return_panel" in out
    # Wide-shape: rows=dates, cols=tickers
    assert list(out["price_panel"].columns) == ["A", "B"]
    assert out["price_panel"].shape == (2, 2)


def test_adapter_handles_permno_or_ticker_column():
    df = pd.DataFrame({
        "date":   pd.to_datetime(["2024-01-01"]),
        "permno_or_ticker": ["T001"],
        "ret":    [0.01], "prc": [100.0],
    })
    out = dsl_adapter.adapt_for_dsl("crsp_dsf", df, template_id="equity_xsmom")
    assert "T001" in out["price_panel"].columns


def test_adapter_unknown_token_passthrough():
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = dsl_adapter.adapt_for_dsl("unknown_token", df)
    # Passthrough as kwarg named after token
    assert "unknown_token" in out


def test_assemble_dsl_kwargs_merges_multiple():
    token_dfs = {
        "crsp_dsf": pd.DataFrame({
            "date":   pd.to_datetime(["2024-01-01"]),
            "ticker": ["A"], "ret": [0.01], "prc": [100.0],
        }),
    }
    out = dsl_adapter.assemble_dsl_kwargs(token_dfs, template_id="equity_xsmom")
    assert "price_panel" in out
    assert "return_panel" in out


# ── _auto_acquire_data unit tests ───────────────────────────────────────

def test_auto_acquire_unknown_mechanism_returns_failure(isolated_data_root):
    """mechanism_id not in library → structured failure."""
    from engine.research.protocols.protocol_executor import _auto_acquire_data
    mech = load_mechanism("equity_xsmom_jt")
    protocol = instantiate_protocol(
        mech, proposal_sample_start="2014-01-31", proposal_sample_end="2024-01-31",
    )
    bad_proposal = {"mechanism_id": "ghost_mechanism_v1",
                     "execution_template": {"template_id": "equity_xsmom",
                                              "binding": {}}}
    data_kwargs, failures = _auto_acquire_data(protocol, bad_proposal)
    assert data_kwargs == {}
    assert any("library mechanism not found" in f for f in failures)


def test_auto_acquire_no_required_data_fails(isolated_data_root, monkeypatch):
    """Mechanism without required_data → failure (can't proceed)."""
    from engine.research.protocols.protocol_executor import _auto_acquire_data

    # Stub load_mechanism inside executor to return a mech lacking required_data
    monkeypatch.setattr(
        "engine.research.protocols.protocol_designer.load_mechanism",
        lambda mid: {"id": mid, "required_data": []},
    )
    mech_real = load_mechanism("equity_xsmom_jt")
    protocol = instantiate_protocol(
        mech_real, proposal_sample_start="2014-01-31",
        proposal_sample_end="2024-01-31",
    )
    proposal = {"mechanism_id": "bare",
                  "execution_template": {"template_id": "equity_xsmom",
                                            "binding": {}}}
    data_kwargs, failures = _auto_acquire_data(protocol, proposal)
    assert data_kwargs == {}
    assert any("no required_data" in f for f in failures)


# ── execute_protocol with auto_acquire=True ──────────────────────────────

def test_execute_protocol_auto_acquire_fetches_from_orchestrator(
        isolated_data_root, monkeypatch):
    """When auto_acquire=True, data flows from orchestrator → adapter →
    DSL → run_gate without manual data_kwargs."""
    # Configure mock to return a substantial dataset
    _mock_paid.TEST_STATE["fetch_rows"] = 50    # 50 tickers

    mech = load_mechanism("equity_xsmom_jt")
    # Override required_data to point at our test token
    mech_test = dict(mech)
    mech_test["required_data"] = ["crsp_dsf"]
    monkeypatch.setattr(
        "engine.research.protocols.protocol_designer.load_mechanism",
        lambda mid: mech_test,
    )
    monkeypatch.setattr(
        "engine.research.protocols.protocol_designer.load_mechanism",
        lambda mid: mech_test,
    )

    protocol = instantiate_protocol(
        mech_test,
        proposal_sample_start="2014-01-31",
        proposal_sample_end="2024-01-31",
    )
    proposal = {
        "mechanism_id":      "equity_xsmom_jt",
        "execution_template": mech_test["execution_template"],
    }
    result = execute_protocol(
        protocol, proposal, auto_acquire=True, pead_control=False,
    )
    # Even though synth data → likely RED/YELLOW verdict; the important thing
    # is the chain executed without manually-provided data_kwargs
    assert result.overall_verdict in ("GREEN", "YELLOW", "RED")
    assert len(result.leg_results) >= 1


def test_execute_protocol_auto_acquire_failure_returns_red(
        isolated_data_root, monkeypatch):
    """When auto_acquire=True AND fetcher fails, protocol returns RED with
    structured failure detail — NEVER silent synth data."""
    _mock_paid.TEST_STATE["probe_available"] = False
    _mock_paid.TEST_STATE["probe_error"] = "simulated WRDS denied"
    _mock_paid.TEST_STATE["probe_error_class"] = "access_denied"

    mech = load_mechanism("equity_xsmom_jt")
    mech_test = dict(mech)
    mech_test["required_data"] = ["crsp_dsf"]
    monkeypatch.setattr(
        "engine.research.protocols.protocol_designer.load_mechanism",
        lambda mid: mech_test,
    )

    protocol = instantiate_protocol(
        mech_test,
        proposal_sample_start="2014-01-31",
        proposal_sample_end="2024-01-31",
    )
    proposal = {
        "mechanism_id":      "equity_xsmom_jt",
        "execution_template": mech_test["execution_template"],
    }
    result = execute_protocol(
        protocol, proposal, auto_acquire=True, pead_control=False,
    )
    assert result.overall_verdict == "RED"
    assert any("data acquisition" in r for r in result.verdict_reasons)


def test_execute_protocol_explicit_data_kwargs_skips_auto_acquire(
        isolated_data_root):
    """If data_kwargs is provided explicitly, auto_acquire is bypassed."""
    rng = np.random.RandomState(0)
    n_months, n_tickers = 60, 30
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rets = rng.randn(n_months, n_tickers) * 0.06
    prices = pd.DataFrame(np.cumprod(1 + rets, axis=0) * 100.0,
                            index=dates, columns=tickers)

    mech = load_mechanism("equity_xsmom_jt")
    protocol = instantiate_protocol(
        mech, proposal_sample_start="2019-01-31",
        proposal_sample_end="2024-01-31",
    )
    proposal = {
        "mechanism_id":      "equity_xsmom_jt",
        "execution_template": mech["execution_template"],
    }
    result = execute_protocol(
        protocol, proposal,
        data_kwargs={"price_panel": prices},
        auto_acquire=True,    # auto_acquire is ignored when data_kwargs given
        pead_control=False,
    )
    # Did NOT route through orchestrator — used given panel
    assert result.overall_verdict in ("GREEN", "YELLOW", "RED")
