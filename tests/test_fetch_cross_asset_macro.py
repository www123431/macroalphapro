"""tests/test_fetch_cross_asset_macro.py — cross-asset macro fetcher.

Offline tests for the FRED-based cross-asset macro anchor fetcher.
Network calls themselves are NOT tested here (integration only).
The unit-testable pieces are:
  - _to_month_end_close (daily → monthly resampling)
  - _ensure_fred_api_key_in_env (secrets.toml loading)
  - integration shape check on the real parquet when cached
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import fetch_cross_asset_macro_anchors as fcm  # noqa: E402


def test_to_month_end_close_takes_last_valid_per_month():
    """Daily series with holidays / NaN should resample to month-end
    using the LAST valid observation in each month."""
    raw = pd.DataFrame({
        "series_id": "VIXCLS",
        "date":  pd.to_datetime([
            "2024-01-02", "2024-01-15", "2024-01-31",  # Jan
            "2024-02-01", "2024-02-29",                  # Feb
        ]),
        "value": [12.0, 14.0, 13.5, 13.6, 15.0],
    })
    out = fcm._to_month_end_close(raw, "VIXCLS")
    assert len(out) == 2
    assert out.loc["2024-01-31"] == 13.5  # last Jan value
    assert out.loc["2024-02-29"] == 15.0  # last Feb value


def test_to_month_end_close_ignores_other_series():
    raw = pd.DataFrame({
        "series_id": ["VIXCLS", "DTWEXBGS"],
        "date":  pd.to_datetime(["2024-01-15", "2024-01-15"]),
        "value": [14.0, 100.0],
    })
    out = fcm._to_month_end_close(raw, "VIXCLS")
    assert len(out) == 1
    assert out.iloc[0] == 14.0


def test_to_month_end_close_returns_empty_on_unknown_series():
    raw = pd.DataFrame({"series_id": ["X"], "date": [pd.Timestamp("2024-01-15")],
                          "value": [1.0]})
    assert fcm._to_month_end_close(raw, "VIXCLS").empty


def test_fred_series_dict_has_expected_keys():
    """5 series, all FRED-resolvable, all map to documented column
    names. Anti-regression for accidental rename / drop."""
    assert set(fcm.FRED_SERIES.keys()) == {
        "VIXCLS", "DTWEXBGS", "BAA10Y", "T10Y3M", "T10YIE",
    }
    # alias values are referenced in fetch_cross_asset_macro_monthly;
    # if these change, the derive-columns step breaks silently
    aliases = set(fcm.FRED_SERIES.values())
    assert aliases >= {"VIX_level", "_dxy_level", "_baa_spread_level",
                          "_term_spread_level", "_breakeven_level"}


def test_ensure_fred_api_key_loads_from_secrets_when_env_absent(tmp_path, monkeypatch):
    """In standalone scripts streamlit context is absent; we read
    secrets.toml directly and inject into os.environ."""
    import os
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    fake_secrets = tmp_path / "secrets.toml"
    fake_secrets.write_text('FRED_API_KEY = "test-key-12345"\n',
                               encoding="utf-8")
    fake_repo = tmp_path
    (fake_repo / ".streamlit").mkdir()
    (fake_repo / ".streamlit" / "secrets.toml").write_text(
        'FRED_API_KEY = "test-key-12345"\n', encoding="utf-8")
    monkeypatch.setattr(fcm, "REPO_ROOT", fake_repo)
    fcm._ensure_fred_api_key_in_env()
    assert os.environ["FRED_API_KEY"] == "test-key-12345"


def test_ensure_fred_api_key_noop_when_env_set(monkeypatch):
    import os
    monkeypatch.setenv("FRED_API_KEY", "already-set")
    fcm._ensure_fred_api_key_in_env()  # should not raise
    assert os.environ["FRED_API_KEY"] == "already-set"


# ────────────────────────────────────────────────────────────────────
# Integration: real cached parquet if present
# ────────────────────────────────────────────────────────────────────
def test_real_cross_asset_parquet_shape_if_present():
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "cross_asset_macro_monthly.parquet")
    if not p.exists():
        pytest.skip("cross-asset parquet not cached — run fetcher")
    df = pd.read_parquet(p)
    expected = {"date", "VIX_level", "VIX_change", "DXY_return",
                  "BAA_spread_change", "T10Y3M_change", "T10YIE_change"}
    assert set(df.columns) == expected
    # 2006-2026 inner join (DTWEXBGS is the floor at 2006)
    assert len(df) > 200
    # VIX_level should be in normal range (9-90)
    assert 9 < df["VIX_level"].mean() < 50
    assert df["VIX_level"].max() < 100   # COVID/GFC max was ~80
    # DXY return is small (sub-percent monthly typical)
    assert df["DXY_return"].abs().mean() < 0.05
    # All columns no NaN (inner join handled)
    for c in df.columns:
        if c == "date":
            continue
        assert df[c].notna().all(), f"{c} has NaN"
