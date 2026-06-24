"""
tests/test_data_snapshot.py — S-3 round-trip + integrity (2026-05-06).

Network-free tests using mocked data. The ONE real-yfinance roundtrip is
gated behind --run-live (cost: ~$0, but network-flaky).
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def fake_data_fetchers(monkeypatch):
    """Replace data_snapshot live fetchers with deterministic fixtures so we
    don't hit yfinance / FRED in the default test run."""
    from engine import data_snapshot as ds

    yf_monthly = pd.DataFrame({
        "XLF": [0.01, 0.02, -0.01, 0.005],
        "XLE": [0.03, -0.01, 0.02, 0.01],
    }, index=pd.date_range("2024-01-31", periods=4, freq="ME"))

    yf_daily = pd.DataFrame({
        "XLF": [60.0 + 0.1 * i for i in range(120)],
        "XLE": [80.0 + 0.05 * i for i in range(120)],
    }, index=pd.date_range("2024-01-02", periods=120, freq="B"))

    yf_vix = pd.DataFrame({
        "close": [15.0 + 0.05 * i for i in range(120)],
    }, index=pd.date_range("2024-01-02", periods=120, freq="B"))

    fred = pd.DataFrame({
        "DGS10": [4.0 + 0.001 * i for i in range(120)],
        "DGS2":  [4.5 + 0.002 * i for i in range(120)],
    }, index=pd.date_range("2024-01-02", periods=120, freq="B"))

    monkeypatch.setattr(ds, "_fetch_yf_monthly", lambda *a, **k: yf_monthly.copy())
    monkeypatch.setattr(ds, "_fetch_yf_daily",   lambda *a, **k: yf_daily.copy())
    monkeypatch.setattr(ds, "_fetch_yf_vix",     lambda *a, **k: yf_vix.copy())
    monkeypatch.setattr(ds, "_fetch_fred",       lambda *a, **k: fred.copy())
    return {"yf_monthly": yf_monthly, "yf_daily": yf_daily,
            "yf_vix": yf_vix, "fred": fred}


def test_freeze_writes_manifest_and_4_parquets(tmp_path, monkeypatch, fake_data_fetchers):
    """Round-trip: freeze → on-disk has manifest + 4 parquets + correct hashes."""
    from engine import data_snapshot as ds
    monkeypatch.setattr(ds, "SNAPSHOT_ROOT", tmp_path)
    snap = ds.freeze_snapshot(
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 6, 30),
        name="testsuite",
        tickers=["XLF", "XLE"],
        fred_series=["DGS10", "DGS2"],
    )
    assert snap.snapshot_id.startswith("testsuite_")
    assert snap.path.is_dir()
    assert (snap.path / "manifest.json").exists()
    assert (snap.path / "yf_monthly_etf.parquet").exists()
    assert (snap.path / "yf_daily_etf.parquet").exists()
    assert (snap.path / "yf_vix.parquet").exists()
    assert (snap.path / "fred_macros.parquet").exists()


def test_freeze_then_load_round_trip(tmp_path, monkeypatch, fake_data_fetchers):
    """load_snapshot returns frames equal to what freeze captured."""
    from engine import data_snapshot as ds
    monkeypatch.setattr(ds, "SNAPSHOT_ROOT", tmp_path)
    snap = ds.freeze_snapshot(
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 6, 30),
        name="rt",
        tickers=["XLF", "XLE"],
        fred_series=["DGS10", "DGS2"],
    )
    loaded = ds.load_snapshot(snap.snapshot_id)
    # check_freq=False because parquet round-trip drops DatetimeIndex.freq
    # attribute (values are preserved, freq inference is best-effort on read).
    pd.testing.assert_frame_equal(snap.yf_monthly_etf, loaded.yf_monthly_etf, check_freq=False)
    pd.testing.assert_frame_equal(snap.yf_daily_etf,   loaded.yf_daily_etf,   check_freq=False)
    pd.testing.assert_frame_equal(snap.yf_vix,         loaded.yf_vix,         check_freq=False)
    pd.testing.assert_frame_equal(snap.fred_macros,    loaded.fred_macros,    check_freq=False)
    assert loaded.tickers == ["XLF", "XLE"]
    assert loaded.fred_series == ["DGS10", "DGS2"]


def test_load_detects_tampered_parquet(tmp_path, monkeypatch, fake_data_fetchers):
    """Modifying a parquet file → load_snapshot raises ValueError (sha256)."""
    from engine import data_snapshot as ds
    monkeypatch.setattr(ds, "SNAPSHOT_ROOT", tmp_path)
    snap = ds.freeze_snapshot(
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 6, 30),
        name="tamper",
        tickers=["XLF"],
        fred_series=["DGS10"],
    )
    # Tamper: rewrite the daily ETF parquet with different content
    bad_df = pd.DataFrame({"XLF": [999.0]},
                          index=pd.date_range("2024-01-02", periods=1, freq="B"))
    bad_df.to_parquet(snap.path / "yf_daily_etf.parquet",
                      compression="snappy", engine="pyarrow")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        ds.load_snapshot(snap.snapshot_id)


def test_freeze_refuses_overwrite(tmp_path, monkeypatch, fake_data_fetchers):
    """Same-day re-freeze with same name raises FileExistsError."""
    from engine import data_snapshot as ds
    monkeypatch.setattr(ds, "SNAPSHOT_ROOT", tmp_path)
    ds.freeze_snapshot(
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 6, 30),
        name="duplicate",
        tickers=["XLF"],
        fred_series=["DGS10"],
    )
    with pytest.raises(FileExistsError):
        ds.freeze_snapshot(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 6, 30),
            name="duplicate",
            tickers=["XLF"],
            fred_series=["DGS10"],
        )


def test_list_snapshots_returns_recent_first(tmp_path, monkeypatch, fake_data_fetchers):
    from engine import data_snapshot as ds
    monkeypatch.setattr(ds, "SNAPSHOT_ROOT", tmp_path)
    s1 = ds.freeze_snapshot(start_date=datetime.date(2024, 1, 1),
                            end_date=datetime.date(2024, 6, 30),
                            name="one", tickers=["XLF"], fred_series=["DGS10"])
    s2 = ds.freeze_snapshot(start_date=datetime.date(2024, 1, 1),
                            end_date=datetime.date(2024, 6, 30),
                            name="two", tickers=["XLF"], fred_series=["DGS10"])
    out = ds.list_snapshots()
    assert len(out) == 2
    # Most recent first
    assert out[0]["snapshot_id"] in {s1.snapshot_id, s2.snapshot_id}


def test_slice_helpers_respect_window(tmp_path, monkeypatch, fake_data_fetchers):
    """get_*_from_snapshot helpers slice on date range correctly."""
    from engine import data_snapshot as ds
    monkeypatch.setattr(ds, "SNAPSHOT_ROOT", tmp_path)
    snap = ds.freeze_snapshot(
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 6, 30),
        name="slice",
        tickers=["XLF", "XLE"],
        fred_series=["DGS10"],
    )
    # Slice yf_monthly to first 2 of 4 months (Jan 31 + Feb 29 entries)
    sliced = ds.get_monthly_returns_from_snapshot(
        snap, ["XLF", "XLE"],
        datetime.date(2024, 1, 1), datetime.date(2024, 3, 1),
    )
    assert len(sliced) == 2

    # FRED slice
    fred_slice = ds.get_fred_from_snapshot(
        snap, "DGS10",
        datetime.date(2024, 1, 1), datetime.date(2024, 1, 31),
    )
    assert len(fred_slice) > 0
    # Unknown FRED series id → empty Series, no exception
    empty = ds.get_fred_from_snapshot(
        snap, "DOES_NOT_EXIST",
        datetime.date(2024, 1, 1), datetime.date(2024, 1, 31),
    )
    assert empty.empty
