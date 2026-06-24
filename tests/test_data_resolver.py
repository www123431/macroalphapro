"""Tests for engine.research.discovery.data_resolver."""
from __future__ import annotations

import datetime
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from engine.research.discovery import data_resolver as dr


# ── can_resolve ──────────────────────────────────────────────────────────

def test_can_resolve_missing_template():
    ok, reason = dr.can_resolve({"required_data": ["crsp_dsf"]})
    assert ok is False
    assert "template_id" in reason


def test_can_resolve_empty_required_data():
    yaml_doc = {
        "execution_template": {"template_id": "equity_xsmom"},
        "required_data": [],
    }
    ok, reason = dr.can_resolve(yaml_doc)
    assert ok is False
    assert "required_data" in reason


def test_can_resolve_unwired_template_returns_false():
    yaml_doc = {
        "execution_template": {"template_id": "event_study"},
        "required_data": ["crsp_dsf"],
    }
    ok, reason = dr.can_resolve(yaml_doc)
    assert ok is False
    assert "no real-data path yet" in reason


def test_can_resolve_blocked_tokens():
    yaml_doc = {
        "execution_template": {"template_id": "equity_xsmom"},
        "required_data": ["crsp_dsf", "ibes_summary"],   # ibes_summary not yet fetchable
    }
    ok, reason = dr.can_resolve(yaml_doc)
    assert ok is False
    assert "ibes_summary" in reason


def test_can_resolve_unknown_token():
    yaml_doc = {
        "execution_template": {"template_id": "equity_xsmom"},
        "required_data": ["totally_unknown_token"],
    }
    ok, reason = dr.can_resolve(yaml_doc)
    assert ok is False
    assert "unknown" in reason


def test_can_resolve_happy_path():
    yaml_doc = {
        "execution_template": {"template_id": "equity_xsmom"},
        "required_data": ["crsp_msf", "vix_index"],
    }
    ok, reason = dr.can_resolve(yaml_doc)
    assert ok is True
    assert reason is None


# ── fetch_token ──────────────────────────────────────────────────────────

def test_fetch_token_unwired_raises():
    with pytest.raises(NotImplementedError):
        dr.fetch_token("ibes_detail", start="2024-01-01", end="2024-06-30")


def test_fetch_token_unknown_raises():
    with pytest.raises(KeyError):
        dr.fetch_token("garbage_token", start="2024-01-01", end="2024-06-30")


def test_fetch_token_uses_cache(monkeypatch, tmp_path):
    """Second call hits cache without re-running fetcher."""
    monkeypatch.setattr(dr, "CACHE_DIR", tmp_path)

    call_count = {"n": 0}
    def _mock_fetcher(s, e, u):
        call_count["n"] += 1
        return pd.DataFrame({"date": ["2024-01-01"], "ticker": ["X"],
                                "prc": [100.0], "ret": [0.01]})

    monkeypatch.setitem(dr._TOKEN_FETCHERS, "crsp_dsf", _mock_fetcher)
    df1 = dr.fetch_token("crsp_dsf", start="2024-01-01", end="2024-06-30",
                           universe=["X"])
    df2 = dr.fetch_token("crsp_dsf", start="2024-01-01", end="2024-06-30",
                           universe=["X"])
    assert call_count["n"] == 1     # 2nd call hit cache
    assert df1.equals(df2)


def test_fetch_token_cache_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(dr, "CACHE_DIR", tmp_path)
    call_count = {"n": 0}
    monkeypatch.setitem(
        dr._TOKEN_FETCHERS, "crsp_dsf",
        lambda s, e, u: (call_count.update(n=call_count["n"]+1)
                            or pd.DataFrame({"date": ["2024-01-01"],
                                                "ticker": ["X"], "prc": [100],
                                                "ret": [0.01]})),
    )
    dr.fetch_token("crsp_dsf", start="2024-01-01", end="2024-06-30",
                      universe=["X"], use_cache=False)
    dr.fetch_token("crsp_dsf", start="2024-01-01", end="2024-06-30",
                      universe=["X"], use_cache=False)
    assert call_count["n"] == 2


# ── Cache helpers ────────────────────────────────────────────────────────

def test_cache_key_stable_for_same_inputs():
    p1 = dr._cache_key("crsp_dsf", "2024-01-01", "2024-06-30", ["A", "B"])
    p2 = dr._cache_key("crsp_dsf", "2024-01-01", "2024-06-30", ["A", "B"])
    assert p1 == p2


def test_cache_key_differs_for_different_universe():
    p1 = dr._cache_key("crsp_dsf", "2024-01-01", "2024-06-30", ["A"])
    p2 = dr._cache_key("crsp_dsf", "2024-01-01", "2024-06-30", ["B"])
    assert p1 != p2


def test_load_cached_stale_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(dr, "CACHE_DIR", tmp_path)
    path = tmp_path / "stale.parquet"
    df = pd.DataFrame({"x": [1]})
    df.to_parquet(path)
    # Backdate mtime to 30 days ago
    old = datetime.datetime.utcnow().timestamp() - 30 * 86400
    import os
    os.utime(path, (old, old))
    assert dr._load_cached(path) is None


# ── Panel construction ──────────────────────────────────────────────────

def test_long_to_wide_prc():
    df = pd.DataFrame({
        "date": ["2024-01-31", "2024-02-29", "2024-01-31", "2024-02-29"],
        "ticker": ["A", "A", "B", "B"],
        "prc": [100, 110, 50, 55],
    })
    wide = dr._long_to_wide_prc(df)
    assert wide.shape == (2, 2)
    assert list(wide.columns) == ["A", "B"]


def test_long_to_wide_handles_empty():
    df = pd.DataFrame()
    assert dr._long_to_wide_prc(df).empty


# ── resolve_panels_for_template (mocked fetchers) ────────────────────────

def test_resolve_panels_equity_xsmom_path(monkeypatch):
    """Mock equity data; verify panels build for equity_xsmom."""
    mock_data = pd.DataFrame({
        "date":   pd.date_range("2020-01-31", "2024-12-31", freq="ME").tolist() * 3,
        "ticker": ["A"] * 60 + ["B"] * 60 + ["C"] * 60,
        "prc":    [100 + i * 0.5 for i in range(180)],
        "ret":    [0.01] * 180,
    })
    monkeypatch.setitem(dr._TOKEN_FETCHERS, "crsp_msf",
                          lambda s, e, u: mock_data)

    yaml_doc = {
        "execution_template": {"template_id": "equity_xsmom",
                                "binding": {"top_frac": 0.1}},
        "required_data": ["crsp_msf"],
    }
    panels = dr.resolve_panels_for_template(yaml_doc, sample_years=5)
    assert "price_panel" in panels
    assert "return_panel" in panels
    assert panels["price_panel"].shape[1] == 3   # 3 tickers


def test_resolve_panels_event_study_not_implemented():
    yaml_doc = {
        "execution_template": {"template_id": "event_study", "binding": {}},
        "required_data": ["crsp_dsf"],
    }
    with pytest.raises(NotImplementedError) as exc_info:
        dr.resolve_panels_for_template(yaml_doc)
    assert "event_study" in str(exc_info.value)


def test_resolve_panels_no_template_raises():
    yaml_doc = {"required_data": ["crsp_dsf"]}
    with pytest.raises(ValueError):
        dr.resolve_panels_for_template(yaml_doc)


def test_resolve_panels_no_required_data_raises():
    yaml_doc = {
        "execution_template": {"template_id": "equity_xsmom"},
    }
    with pytest.raises(ValueError):
        dr.resolve_panels_for_template(yaml_doc)


# ── factor_quartile + factor_panel synthesis ─────────────────────────────

def test_resolve_panels_factor_quartile_synthesizes_factor_panel(monkeypatch):
    """For factor_quartile, the resolver must synthesize a factor_panel
    from returns (since the YAML didn't supply a real factor source)."""
    mock_data = pd.DataFrame({
        "date":   pd.date_range("2020-01-31", "2024-12-31", freq="ME").tolist() * 3,
        "ticker": ["A"] * 60 + ["B"] * 60 + ["C"] * 60,
        "prc":    [100 + i * 0.5 for i in range(180)],
        "ret":    [0.01] * 180,
    })
    monkeypatch.setitem(dr._TOKEN_FETCHERS, "crsp_msf",
                          lambda s, e, u: mock_data)

    yaml_doc = {
        "execution_template": {"template_id": "factor_quartile",
                                "binding": {"top_frac": 0.1}},
        "required_data": ["crsp_msf"],
    }
    panels = dr.resolve_panels_for_template(yaml_doc, sample_years=5)
    assert "factor_panel" in panels
    assert "price_panel" in panels
    assert "return_panel" in panels
