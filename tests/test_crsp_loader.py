"""
tests/test_crsp_loader.py — Unit tests for engine.universe_singlename.crsp_loader
(W-B-1 prep, 2026-05-10).

Coverage scope:
  - Mock-mode panel generation (deterministic, reproducible)
  - is_wrds_available() feature flag behavior
  - Public API shape mirrors panel_fetcher.bulk_fetch_singlestock_panel
  - Real-path stub raises with actionable error pre-WRDS-activation

Real WRDS integration tests are NOT included here — they are deferred to a
separate `test_crsp_loader_integration.py` once WRDS account is configured
(skip-by-default if `is_wrds_available()` returns False).
"""
from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from engine.universe_singlename import crsp_loader
from engine.universe_singlename.crsp_loader import (
    bulk_fetch_crsp_daily_panel,
    is_wrds_available,
)


# ── is_wrds_available ───────────────────────────────────────────────────────
def test_is_wrds_available_returns_false_when_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `wrds` lib not installed, feature flag must return False."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "wrds":
            raise ImportError("simulated: wrds not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert is_wrds_available() is False


def test_is_wrds_available_returns_false_when_no_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `wrds` is importable but no pgpass + no env + no secrets, feature
    flag must return False (avoid false positive).

    Post-2026-05-11: is_wrds_available() additionally checks Windows
    pgpass location (%APPDATA%/postgresql/pgpass.conf) and project
    secrets.toml [WRDS] section. All credential sources must be cleared.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("WRDS_USER", raising=False)
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    # Windows pgpass at APPDATA — redirect to tmp so file isn't found
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    # Block project secrets.toml read (return None for username)
    monkeypatch.setattr(
        "engine.universe_singlename.crsp_loader._get_wrds_username",
        lambda: None,
    )
    # Stub wrds module into sys.modules so the import succeeds
    import sys, types
    fake_wrds = types.ModuleType("wrds")
    monkeypatch.setitem(sys.modules, "wrds", fake_wrds)
    assert is_wrds_available() is False


def test_is_wrds_available_returns_true_with_pgpass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".pgpass").write_text("placeholder", encoding="utf-8")
    import sys, types
    fake_wrds = types.ModuleType("wrds")
    monkeypatch.setitem(sys.modules, "wrds", fake_wrds)
    assert is_wrds_available() is True


def test_is_wrds_available_returns_true_with_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("WRDS_USER", "test_user")
    import sys, types
    fake_wrds = types.ModuleType("wrds")
    monkeypatch.setitem(sys.modules, "wrds", fake_wrds)
    assert is_wrds_available() is True


# ── Mock-mode panel ─────────────────────────────────────────────────────────
def test_mock_panel_returns_correct_shape() -> None:
    """Mock panel dimensions must match (B-days × tickers)."""
    panel = bulk_fetch_crsp_daily_panel(
        tickers=["AAPL", "MSFT", "GOOG"],
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 31),
        mock_mode=True,
    )
    assert isinstance(panel, pd.DataFrame)
    assert set(panel.columns) == {"AAPL", "GOOG", "MSFT"}
    # Jan 2024 has ~22 business days
    assert 20 <= len(panel) <= 23


def test_mock_panel_is_deterministic() -> None:
    """Same ticker + range must produce same prices across runs (seed by hash)."""
    args = dict(
        tickers=["AAPL"],
        start_date=datetime.date(2023, 6, 1),
        end_date=datetime.date(2023, 6, 30),
        mock_mode=True,
    )
    p1 = bulk_fetch_crsp_daily_panel(**args)
    p2 = bulk_fetch_crsp_daily_panel(**args)
    pd.testing.assert_frame_equal(p1, p2)


def test_mock_panel_different_tickers_have_different_seeds() -> None:
    """Two distinct tickers must have different price series (independent seeds)."""
    panel = bulk_fetch_crsp_daily_panel(
        tickers=["AAPL", "MSFT"],
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 3, 31),
        mock_mode=True,
    )
    # Should not be exactly equal
    assert not panel["AAPL"].equals(panel["MSFT"])


def test_mock_panel_starts_at_100() -> None:
    """First row of mock panel = 100.0 * exp(first ret) ≈ near 100."""
    panel = bulk_fetch_crsp_daily_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 10),
        mock_mode=True,
    )
    # First price should be in a sensible range around 100 (1-day GBM step)
    first_price = panel["AAPL"].iloc[0]
    assert 90.0 < first_price < 110.0, f"first price {first_price} out of GBM range"


def test_mock_panel_no_negative_prices() -> None:
    """GBM via exp() must never produce negative prices."""
    panel = bulk_fetch_crsp_daily_panel(
        tickers=["AAPL", "MSFT", "TSLA", "NVDA"],
        start_date=datetime.date(2010, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        mock_mode=True,
    )
    assert (panel > 0).all().all(), "mock panel should have all positive prices"


def test_mock_panel_handles_empty_date_range() -> None:
    """Start == end + 1 (no business days in range) → empty DataFrame."""
    panel = bulk_fetch_crsp_daily_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2024, 1, 6),   # Saturday
        end_date=datetime.date(2024, 1, 7),     # Sunday
        mock_mode=True,
    )
    assert panel.empty


def test_mock_panel_dedups_tickers() -> None:
    """Caller passing duplicate tickers should still get unique columns."""
    panel = bulk_fetch_crsp_daily_panel(
        tickers=["AAPL", "AAPL", "MSFT"],
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 15),
        mock_mode=True,
    )
    assert set(panel.columns) == {"AAPL", "MSFT"}


# ── Auto mock_mode detection ────────────────────────────────────────────────
def test_auto_mode_falls_back_to_mock_when_wrds_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mock_mode=None + no WRDS → auto-routes to mock (no error)."""
    monkeypatch.setattr(crsp_loader, "is_wrds_available", lambda: False)
    panel = bulk_fetch_crsp_daily_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 15),
        # mock_mode unspecified — should auto-fallback
    )
    assert not panel.empty
    assert "AAPL" in panel.columns


# ── Real-mode stub error path ───────────────────────────────────────────────
def test_real_mode_with_no_wrds_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mock_mode=False + no WRDS → must raise informative RuntimeError.

    Note: bypass cache (use_cache=False) so the real-path guard actually fires;
    a cached panel from prior real runs would short-circuit ahead of the check.
    """
    monkeypatch.setattr(crsp_loader, "is_wrds_available", lambda: False)
    with pytest.raises(RuntimeError, match="WRDS not configured"):
        bulk_fetch_crsp_daily_panel(
            tickers=["AAPL"],
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
            mock_mode=False,
            use_cache=False,
        )


def test_real_mode_no_longer_stubbed_post_activation() -> None:
    """Post-2026-05-11 (Wave B activation): real CRSP path is implemented.
    This test guards against the stub regressing back to NotImplementedError.

    Live WRDS verification (with actual data flowing) is covered via
    integration smoke runs, not unit tests, to keep the test suite fast
    and offline. We only assert the function is callable / does NOT raise
    NotImplementedError when invoked with WRDS reachable.
    """
    from engine.universe_singlename.crsp_loader import _real_crsp_panel
    # Cannot raise NotImplementedError by virtue of being implemented;
    # we test by inspecting the function source — if "NotImplementedError"
    # appears, the stub has regressed.
    import inspect
    src = inspect.getsource(_real_crsp_panel)
    assert "NotImplementedError" not in src, (
        "_real_crsp_panel regressed to stub state. Post-Wave-B-activation, "
        "this function must be implemented."
    )


# ── API parity with panel_fetcher (Wave A) ──────────────────────────────────
def test_api_signature_parity_with_wave_a_panel_fetcher() -> None:
    """Both loaders must accept the same core arg shape so Wave B walk-forward
    can swap call site with single-line change."""
    import inspect
    from engine.factor_ensemble_singlename.panel_fetcher import bulk_fetch_singlestock_panel

    wave_a_sig = inspect.signature(bulk_fetch_singlestock_panel)
    wave_b_sig = inspect.signature(bulk_fetch_crsp_daily_panel)

    # Core positional/keyword args MUST match (tickers / start_date / end_date / use_cache)
    common = {"tickers", "start_date", "end_date", "use_cache"}
    assert common.issubset(set(wave_a_sig.parameters)), wave_a_sig
    assert common.issubset(set(wave_b_sig.parameters)), wave_b_sig
