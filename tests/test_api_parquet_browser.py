"""Tests for the parquet inventory endpoint (/api/research/parquets)
and the underlying scanner (engine.research.parquet_browser).

Phase Lab-Step-A — surfaces cached return-series in the Lab UI so
senior can validate a freshly-fetched parquet WITHOUT switching to a
shell. The closing of the step 1 ("data exists?") gap in the
factor-exploration workflow.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


# ── engine.research.parquet_browser.scan_parquets ────────────────────────

def test_scan_parquets_runs_on_real_cache():
    """Scanner discovers SOMETHING in the actual repo cache.

    Doesn't pin to a specific file — tests against whatever WRDS
    fetchers happen to have populated. Just asserts the contract.
    """
    from engine.research.parquet_browser import scan_parquets
    out = scan_parquets(include_internal=True, limit=500)
    assert "n" in out
    assert "parquets" in out
    assert "cache_dir" in out
    assert isinstance(out["parquets"], list)
    if out["n"] > 0:
        first = out["parquets"][0]
        for key in ("filename", "relpath", "size_bytes", "mtime",
                    "n_rows", "n_cols", "columns",
                    "date_start", "date_end", "is_internal", "error"):
            assert key in first, f"missing key: {key}"


def test_scan_parquets_include_internal_toggle():
    """include_internal=False filters underscore-prefix entries."""
    from engine.research.parquet_browser import scan_parquets
    with_int = scan_parquets(include_internal=True,  limit=500)
    no_int   = scan_parquets(include_internal=False, limit=500)
    assert no_int["n"] <= with_int["n"]
    for entry in no_int["parquets"]:
        assert not entry["filename"].startswith("_"), \
            f"underscore-prefix leaked through: {entry['filename']}"


def test_scan_parquets_limit_caps_results():
    from engine.research.parquet_browser import scan_parquets
    out = scan_parquets(include_internal=True, limit=2)
    assert out["n"] <= 2


def test_scan_parquets_sort_order_curated_first():
    """Non-underscore files surface before underscore-prefix files."""
    from engine.research.parquet_browser import scan_parquets
    out = scan_parquets(include_internal=True, limit=500)
    seen_internal = False
    for entry in out["parquets"]:
        if entry["is_internal"]:
            seen_internal = True
        elif seen_internal:
            pytest.fail(
                f"curated file '{entry['filename']}' appeared AFTER "
                "an internal file — sort order violated"
            )


# ── REST: GET /api/research/parquets ─────────────────────────────────────

def test_parquets_endpoint_200(client):
    r = client.get("/api/research/parquets")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "parquets" in body
    assert "n" in body
    assert "cache_dir" in body


def test_parquets_endpoint_respects_query(client):
    r = client.get("/api/research/parquets",
                   params={"include_internal": "false", "limit": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] <= 5
    for entry in body["parquets"]:
        assert not entry["filename"].startswith("_")


def test_parquets_endpoint_rejects_bad_limit(client):
    r = client.get("/api/research/parquets", params={"limit": 99999})
    # Query(le=1000) → 422 from FastAPI validator
    assert r.status_code == 422


# ── Integration: scanner handles synthetic parquet ───────────────────────

def test_scan_parquets_reads_date_range_from_index(tmp_path, monkeypatch):
    """Scanner must extract date_start/date_end from a DatetimeIndex."""
    import engine.research.parquet_browser as pb

    fake = tmp_path / "cache"
    fake.mkdir()
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(
            ["2020-01-01", "2020-06-30", "2021-12-31"], name="date",
        ),
    )
    df.to_parquet(fake / "fake_series.parquet")

    monkeypatch.setattr(pb, "CACHE_DIR", fake)
    out = pb.scan_parquets(include_internal=True, limit=10)
    assert out["n"] == 1
    entry = out["parquets"][0]
    assert entry["filename"] == "fake_series.parquet"
    assert entry["n_rows"] == 3
    assert entry["date_start"] == "2020-01-01"
    assert entry["date_end"] == "2021-12-31"
    assert entry["is_internal"] is False
    assert entry["error"] is None
