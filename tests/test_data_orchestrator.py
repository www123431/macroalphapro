"""Phase 6a tests for data orchestrator + cache + probe pattern.

Covers all 5 scenarios from project_data_acquisition_engine_design:
  1. Cache hit (return cache, no fetch)
  2. Cache miss → primary fetch works
  3. Primary unavailable → fallback fetch works (with quality_caveat)
  4. ALL fetchers fail → success=False with attempt detail
  5. Partial date coverage → flagged

Plus probe-first behavior per WRDS care doctrine:
  - probe failure skips fetch (no quota burn)
  - auth-missing skips probe + fetch immediately
  - probe success allows fetch
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.data import cache_manager
from engine.data import orchestrator
from engine.data.fetchers import _mock_paid, _mock_free


# ── Test isolation: isolate cache, logs, inventory ──────────────────────

@pytest.fixture
def isolated_data_root(tmp_path, monkeypatch):
    """Stand up isolated data/cache_v2 + acquisition log + inventory."""
    cache_v2 = tmp_path / "data" / "cache_v2"
    cache_v2.mkdir(parents=True)
    meta_dir = cache_v2 / "_meta"
    meta_dir.mkdir()
    monkeypatch.setattr(cache_manager, "CACHE_DIR", cache_v2)
    monkeypatch.setattr(cache_manager, "META_DIR", meta_dir)

    acq_log = tmp_path / "acquisition_log.jsonl"
    monkeypatch.setattr(orchestrator, "ACQUISITION_LOG", acq_log)

    # Write a test inventory with both _mock_paid and _mock_free
    inv_path = tmp_path / "data_inventory.yaml"
    inv_path.write_text("""
_schema_version: 1

inventory:
  crsp_dsf:
    fetcher_chain:
      - source:    _mock_paid
        function:  fetch_dsf
        tier:      paid
        auth:      mock_paid_credentials
      - source:    _mock_free
        function:  fetch_equity_daily
        tier:      free
        auth:      null
        quality_caveat: "mock free source; lower fidelity"
    required_columns: [date, permno_or_ticker, ret, prc]
    refresh_cadence: monthly
""", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "INVENTORY_PATH", inv_path)
    return tmp_path


@pytest.fixture
def reset_mock_state(monkeypatch):
    """Reset mock fetcher TEST_STATE between tests."""
    _mock_paid.TEST_STATE.update({
        "probe_available":   True, "probe_error": None,
        "probe_error_class": None, "fetch_raises": None,
        "fetch_rows":        100,  "fetch_partial": False,
    })
    _mock_free.TEST_STATE.update({
        "probe_available":   True, "probe_error": None,
        "probe_error_class": None, "fetch_raises": None,
        "fetch_rows":        50,
    })
    # By default, mock paid auth is available
    monkeypatch.setattr(orchestrator, "_auth_available",
                          lambda auth: True if not auth else True)
    yield


# ── Scenario 1: cache hit ───────────────────────────────────────────────

def test_cache_hit_returns_cached_data(isolated_data_root, reset_mock_state):
    # First fetch (writes cache)
    r1 = orchestrator.fetch_token("crsp_dsf", start="2020-01-01", end="2020-12-31",
                                     log=False)
    assert r1.success
    assert r1.cache_hit is False

    # Second fetch (should hit cache, no new fetcher call)
    r2 = orchestrator.fetch_token("crsp_dsf", start="2020-01-01", end="2020-12-31",
                                     log=False)
    assert r2.success
    assert r2.cache_hit is True
    assert len(r2.attempts) == 0    # no fetcher attempted


# ── Scenario 2: cache miss → primary works ──────────────────────────────

def test_primary_fetcher_serves_first(isolated_data_root, reset_mock_state):
    r = orchestrator.fetch_token("crsp_dsf",
                                    start="2020-01-01", end="2020-12-31",
                                    log=False)
    assert r.success
    assert r.source_used == "_mock_paid"
    assert r.source_tier == "paid"
    assert r.quality_caveats == []    # paid → no caveat


# ── Scenario 3: primary unavailable → fallback ──────────────────────────

def test_fallback_to_free_when_paid_unavailable(isolated_data_root,
                                                  reset_mock_state):
    _mock_paid.TEST_STATE["probe_available"] = False
    _mock_paid.TEST_STATE["probe_error"] = "simulated access_denied"
    _mock_paid.TEST_STATE["probe_error_class"] = "access_denied"

    r = orchestrator.fetch_token("crsp_dsf",
                                    start="2020-01-01", end="2020-12-31",
                                    log=False)
    assert r.success
    assert r.source_used == "_mock_free"
    assert r.source_tier == "free"
    # Quality caveat propagated from inventory
    assert any("mock free source" in c for c in r.quality_caveats)
    # AND orchestrator adds downgrade caveat
    assert any("paid source(s) unavailable" in c for c in r.quality_caveats)


def test_attempt_records_probe_failure(isolated_data_root, reset_mock_state):
    _mock_paid.TEST_STATE["probe_available"] = False
    _mock_paid.TEST_STATE["probe_error"] = "auth expired"
    _mock_paid.TEST_STATE["probe_error_class"] = "auth_missing"
    r = orchestrator.fetch_token("crsp_dsf",
                                    start="2020-01-01", end="2020-12-31",
                                    log=False)
    assert r.success
    # First attempt should be probe-only failure
    assert r.attempts[0].source == "_mock_paid"
    assert r.attempts[0].probe_only is True
    assert r.attempts[0].error_class == "auth_missing"
    # Second attempt is the actual fetch via fallback
    assert r.attempts[1].success is True


# ── Scenario 4: ALL fetchers fail → structured failure ──────────────────

def test_all_fetchers_fail_returns_structured_failure(isolated_data_root,
                                                       reset_mock_state):
    _mock_paid.TEST_STATE["probe_available"] = False
    _mock_paid.TEST_STATE["probe_error"] = "wrds denied"
    _mock_paid.TEST_STATE["probe_error_class"] = "access_denied"
    _mock_free.TEST_STATE["probe_available"] = False
    _mock_free.TEST_STATE["probe_error"] = "network timeout"
    _mock_free.TEST_STATE["probe_error_class"] = "network"

    r = orchestrator.fetch_token("crsp_dsf",
                                    start="2020-01-01", end="2020-12-31",
                                    log=False)
    assert r.success is False
    assert r.df is None
    assert r.source_used is None
    assert len(r.attempts) == 2
    assert all(a.success is False for a in r.attempts)
    assert "all 2 fetchers failed" in r.quality_caveats[0]


def test_failure_does_not_silently_substitute(isolated_data_root,
                                                 reset_mock_state):
    """The orchestrator MUST NEVER fall back to synth/empty data silently."""
    _mock_paid.TEST_STATE["probe_available"] = False
    _mock_free.TEST_STATE["probe_available"] = False
    r = orchestrator.fetch_token("crsp_dsf",
                                    start="2020-01-01", end="2020-12-31",
                                    log=False)
    assert r.success is False
    # df should be None, not an empty DataFrame masquerading
    assert r.df is None


# ── Scenario 5: partial coverage ────────────────────────────────────────

def test_partial_coverage_flagged(isolated_data_root, reset_mock_state):
    _mock_paid.TEST_STATE["fetch_partial"] = True
    r = orchestrator.fetch_token("crsp_dsf",
                                    start="2020-01-01", end="2020-12-31",
                                    log=False)
    assert r.success
    assert r.partial_coverage is True
    assert r.coverage_end < "2020-12-31"


# ── Probe-first specific tests ──────────────────────────────────────────

def test_probe_failure_skips_fetch(isolated_data_root, reset_mock_state):
    """Probe failure should NOT call fetch() — saves WRDS quota."""
    _mock_paid.TEST_STATE["probe_available"] = False
    _mock_paid.TEST_STATE["probe_error"] = "access denied"
    _mock_paid.TEST_STATE["fetch_raises"] = AssertionError    # should NEVER be called

    r = orchestrator.fetch_token("crsp_dsf",
                                    start="2020-01-01", end="2020-12-31",
                                    log=False)
    # Probe-only failure recorded; fetch_raises NEVER triggered
    assert r.attempts[0].probe_only is True


# ── Acquisition log ─────────────────────────────────────────────────────

def test_acquisition_log_written(isolated_data_root, reset_mock_state):
    orchestrator.fetch_token("crsp_dsf", start="2020-01-01", end="2020-12-31",
                               log=True)
    assert orchestrator.ACQUISITION_LOG.exists()
    rows = [json.loads(l) for l in orchestrator.ACQUISITION_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["token"] == "crsp_dsf"
    assert rows[0]["source_used"] == "_mock_paid"


def test_acquisition_log_records_failure(isolated_data_root, reset_mock_state):
    _mock_paid.TEST_STATE["probe_available"] = False
    _mock_free.TEST_STATE["probe_available"] = False
    orchestrator.fetch_token("crsp_dsf", start="2020-01-01", end="2020-12-31",
                               log=True)
    rows = [json.loads(l) for l in orchestrator.ACQUISITION_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["success"] is False


# ── Unknown token ───────────────────────────────────────────────────────

def test_unknown_token_fails_cleanly(isolated_data_root, reset_mock_state):
    r = orchestrator.fetch_token("nonexistent_token",
                                    start="2020-01-01", end="2020-12-31",
                                    log=False)
    assert r.success is False
    assert any("not in data_inventory" in c for c in r.quality_caveats)


# ── assemble_data_kwargs ────────────────────────────────────────────────

def test_assemble_returns_dict_and_results(isolated_data_root, reset_mock_state):
    kwargs, results = orchestrator.assemble_data_kwargs(
        ["crsp_dsf"], start="2020-01-01", end="2020-12-31", log=False,
    )
    assert "crsp_dsf" in kwargs
    assert isinstance(kwargs["crsp_dsf"], pd.DataFrame)
    assert len(results) == 1


def test_assemble_with_partial_failure(isolated_data_root, reset_mock_state):
    """Some tokens succeed, others fail — both reflected in results."""
    kwargs, results = orchestrator.assemble_data_kwargs(
        ["crsp_dsf", "nonexistent_token"],
        start="2020-01-01", end="2020-12-31", log=False,
    )
    assert "crsp_dsf" in kwargs
    assert "nonexistent_token" not in kwargs
    assert len(results) == 2
    assert results[0].success and not results[1].success


# ── Cache invalidation ──────────────────────────────────────────────────

def test_cache_invalidate(isolated_data_root, reset_mock_state):
    orchestrator.fetch_token("crsp_dsf", start="2020-01-01", end="2020-12-31",
                               log=False)
    n_removed = cache_manager.invalidate("crsp_dsf")
    assert n_removed >= 1
    # Next fetch should be cache miss
    r = orchestrator.fetch_token("crsp_dsf", start="2020-01-01", end="2020-12-31",
                                    log=False)
    assert r.cache_hit is False
