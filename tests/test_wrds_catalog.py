"""Tests for engine.data.fetchers.wrds_catalog."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from engine.data.fetchers import wrds_catalog as wc


# ── DATA_INVENTORY_TO_WRDS mapping ───────────────────────────────────────

def test_token_map_covers_critical_tokens():
    """The mapping should at least cover the most-used DATA_INVENTORY tokens."""
    critical = {
        "crsp_dsf", "crsp_msf",
        "compustat_quarterly", "compustat_annual",
        "ibes_summary", "ibes_detail", "ibes_guidance",
        "optionm_iv_surface", "optionm_skew",
    }
    mapped = set(wc.DATA_INVENTORY_TO_WRDS.keys())
    assert critical.issubset(mapped), \
        f"missing critical tokens: {critical - mapped}"


def test_token_map_schemas_in_target_schemas():
    """Every mapping's schema should be in TARGET_SCHEMAS (so probe
    actually covers it)."""
    for token, m in wc.DATA_INVENTORY_TO_WRDS.items():
        schema = m["schema"]
        if m["table"] is None:
            continue   # FRED / non-WRDS tokens
        assert schema in wc.TARGET_SCHEMAS, \
            f"token {token!r} maps to schema {schema!r} not in TARGET_SCHEMAS"


# ── save / load roundtrip ────────────────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "CATALOG_PATH", tmp_path / "catalog.json")
    catalog = {
        "probed_at": "2026-01-01T00:00:00Z",
        "account": "${WRDS_USER_1}",
        "schemas": {"crsp": {"tables": ["dsf", "msf"]}},
        "errors": [],
    }
    saved = wc.save_catalog(catalog)
    assert saved.exists()
    loaded = wc.load_catalog()
    assert loaded == catalog


def test_load_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "CATALOG_PATH", tmp_path / "missing.json")
    assert wc.load_catalog() is None


def test_load_returns_none_on_corrupt_file(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr(wc, "CATALOG_PATH", bad)
    assert wc.load_catalog() is None


# ── is_token_available_in_catalog ────────────────────────────────────────

def test_token_available_happy_path():
    catalog = {
        "schemas": {"crsp": {"tables": ["dsf", "msf"]}},
    }
    ok, reason = wc.is_token_available_in_catalog("crsp_dsf", catalog)
    assert ok is True
    assert reason is None


def test_token_available_no_catalog():
    ok, reason = wc.is_token_available_in_catalog("crsp_dsf", catalog={})
    assert ok is False
    # When catalog={}, schemas dict is empty → "schema not probed"
    assert "schema" in reason.lower() or "probed" in reason.lower()


def test_token_available_unmapped_token():
    ok, reason = wc.is_token_available_in_catalog(
        "made_up_token", catalog={"schemas": {}},
    )
    assert ok is False
    assert "no DATA_INVENTORY_TO_WRDS mapping" in reason


def test_token_available_schema_not_probed():
    catalog = {"schemas": {"crsp": {"tables": ["dsf"]}}}    # missing ibes
    ok, reason = wc.is_token_available_in_catalog("ibes_detail", catalog)
    assert ok is False
    assert "ibes" in reason


def test_token_available_table_not_present():
    catalog = {"schemas": {"crsp": {"tables": ["dsf"]}}}     # missing msf
    ok, reason = wc.is_token_available_in_catalog("crsp_msf", catalog)
    assert ok is False
    assert "crsp.msf" in reason


def test_token_available_fred_skipped():
    """FRED token has table=None — should report 'doesn't come from WRDS'."""
    catalog = {"schemas": {"frb": {"tables": []}}}
    ok, reason = wc.is_token_available_in_catalog("fred_macro", catalog)
    assert ok is False
    assert "doesn't come from WRDS" in reason


# ── summarize_catalog ────────────────────────────────────────────────────

def test_summarize_empty_catalog():
    s = wc.summarize_catalog({})
    # If we passed an actual empty dict, summarize still has access to it
    assert s["probed"] is True
    assert s["n_schemas_probed"] == 0


def test_summarize_no_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "CATALOG_PATH", tmp_path / "no.json")
    s = wc.summarize_catalog()
    assert s == {"probed": False}


def test_summarize_with_data():
    catalog = {
        "probed_at": "2026-01-01T00:00:00Z",
        "schemas": {
            "crsp": {"tables": ["dsf", "msf", "names"]},
            "ibes": {"tables": ["statsum_epsus"]},
        },
        "errors": [{"stage": "x"}],
    }
    s = wc.summarize_catalog(catalog)
    assert s["n_schemas_probed"] == 2
    assert s["n_tables_total"] == 4
    assert s["errors"] == 1


# ── probe_wrds (mocked connection) ───────────────────────────────────────

def test_probe_handles_connect_failure(monkeypatch):
    """Connection failure → error captured, returns partial catalog."""
    def _fail(*a, **kw):
        raise ConnectionError("simulated connect fail")
    monkeypatch.setattr("engine.line_c.wrds_direct.connect", _fail)

    catalog = wc.probe_wrds(account="${WRDS_USER_1}", target_schemas=["crsp"])
    assert catalog["errors"]
    assert "simulated" in catalog["errors"][0]["error"]
    assert catalog["schemas"] == {}


def test_probe_handles_partial_schema_failure(monkeypatch):
    """One schema queries OK, another fails → both states recorded."""
    class FakeCursor:
        def __init__(self):
            self._next = []
        def execute(self, sql, params=None):
            if params and params[0] == "crsp":
                self._next = [("dsf",), ("msf",)]
            elif params and params[0] == "ibes":
                raise RuntimeError("permission denied")
            else:
                self._next = []
        def fetchall(self):
            return self._next
        def fetchone(self):
            return self._next[0] if self._next else None
    class FakeConn:
        def cursor(self): return FakeCursor()
        def close(self): pass
    monkeypatch.setattr("engine.line_c.wrds_direct.connect",
                          lambda *a, **kw: FakeConn())

    catalog = wc.probe_wrds(
        account="${WRDS_USER_1}", target_schemas=["crsp", "ibes"],
        include_row_counts=False,
    )
    assert "crsp" in catalog["schemas"]
    assert catalog["schemas"]["crsp"]["tables"] == ["dsf", "msf"]
    assert "ibes" not in catalog["schemas"]
    # Error recorded for ibes
    assert any(e.get("schema") == "ibes" for e in catalog["errors"])
