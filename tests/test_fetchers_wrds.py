"""Tests for engine.data.fetchers.wrds_crsp.

CAREFUL: Real WRDS connections are NEVER attempted by these tests. Every
WRDS call is mocked to avoid quota burn / triggering access-denied
escalation per WRDS-care doctrine.
"""
from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from engine.data.fetchers import wrds_crsp


# ── Probe tests ─────────────────────────────────────────────────────────

def test_probe_returns_unavailable_when_wrds_direct_missing(monkeypatch):
    """If wrds_direct can't be imported, probe should fail cleanly."""
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: None)
    result = wrds_crsp.probe("2020-01-01", "2020-12-31")
    assert result.available is False
    assert result.error_class == "schema_unknown"


def test_probe_classifies_access_denied(monkeypatch):
    fake = mock.MagicMock()
    fake.raw_sql.side_effect = Exception("permission denied for table crsp.dsf")
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: fake)
    result = wrds_crsp.probe("2020-01-01", "2020-12-31")
    assert result.available is False
    assert result.error_class == "access_denied"


def test_probe_classifies_auth_missing(monkeypatch):
    fake = mock.MagicMock()
    fake.raw_sql.side_effect = Exception("FileNotFoundError: pgpass.conf")
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: fake)
    result = wrds_crsp.probe("2020-01-01", "2020-12-31")
    assert result.available is False
    assert result.error_class == "auth_missing"


def test_probe_classifies_network(monkeypatch):
    fake = mock.MagicMock()
    fake.raw_sql.side_effect = Exception("connection refused")
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: fake)
    result = wrds_crsp.probe("2020-01-01", "2020-12-31")
    assert result.available is False
    assert result.error_class == "network"


def test_probe_succeeds_with_non_empty_result(monkeypatch):
    fake = mock.MagicMock()
    fake.raw_sql.return_value = pd.DataFrame({"ok": [1]})
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: fake)
    result = wrds_crsp.probe("2020-01-01", "2020-12-31")
    assert result.available is True


# ── _safe_raw_sql doctrine: no retry on permission denied ───────────────

def test_safe_raw_sql_returns_none_on_denied_no_retry(monkeypatch):
    """Permission-denied MUST NOT retry (would trigger abuse detection)."""
    call_count = {"n": 0}

    fake = mock.MagicMock()
    def raise_denied(*args, **kw):
        call_count["n"] += 1
        raise Exception("permission denied")
    fake.raw_sql.side_effect = raise_denied
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: fake)

    result = wrds_crsp._safe_raw_sql("SELECT 1")
    assert result is None
    assert call_count["n"] == 1    # NO retry


def test_safe_raw_sql_returns_none_when_connector_missing(monkeypatch):
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: None)
    result = wrds_crsp._safe_raw_sql("SELECT 1")
    assert result is None


# ── fetch_dsf canonical schema ──────────────────────────────────────────

def test_fetch_dsf_returns_canonical_columns(monkeypatch):
    fake = mock.MagicMock()
    fake.raw_sql.return_value = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "permno": [10001, 10001],
        "ticker": ["AAPL", "AAPL"],
        "ret":    [0.01, 0.02],
        "prc":    [100.0, 101.0],
        "vol":    [1000, 1100],
        "shrout": [10000, 10000],
    })
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: fake)
    df = wrds_crsp.fetch_dsf("2020-01-01", "2020-12-31", merge_delisting=False)
    assert set(df.columns) >= {"date", "permno", "ticker", "ret", "prc"}


def test_fetch_dsf_returns_empty_on_error(monkeypatch):
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: None)
    df = wrds_crsp.fetch_dsf("2020-01-01", "2020-12-31")
    assert df.empty
    assert set(df.columns) >= {"date", "permno", "ticker", "ret"}


def test_fetch_dsf_merges_delisting_returns(monkeypatch):
    """Delisting return should compound into ret column when present."""
    call_count = {"n": 0}
    fake = mock.MagicMock()
    def mock_sql(sql, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:    # first call: DSF
            return pd.DataFrame({
                "date": [pd.Timestamp("2020-01-15")],
                "permno": [10001], "ticker": ["X"],
                "ret": [0.05], "prc": [50.0],
                "vol": [1000], "shrout": [10000],
            })
        # second call: dsedelist
        return pd.DataFrame({
            "date": [pd.Timestamp("2020-01-15")],
            "permno": [10001],
            "dlret": [-0.20],
        })
    fake.raw_sql.side_effect = mock_sql
    monkeypatch.setattr(wrds_crsp, "_get_connector", lambda: fake)
    df = wrds_crsp.fetch_dsf("2020-01-15", "2020-01-15",
                                merge_delisting=True)
    # Compounded: (1+0.05)(1-0.20) - 1 = -0.16
    assert abs(df.loc[0, "ret"] - (-0.16)) < 1e-6
    assert "dlret" not in df.columns    # dropped after merge
