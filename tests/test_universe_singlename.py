"""
tests/test_universe_singlename.py — Stage 2 Wave A SP500 universe loader tests.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.1
"""
from __future__ import annotations

import datetime
from unittest import mock

import pandas as pd
import pytest

from engine.universe_singlename import (
    UNIVERSE_SOURCES_LOCKED,
    load_sp500_constituents_at_date,
    SP500ConstituentsResult,
)


def test_universe_sources_locked():
    """Spec §2.1 amendments:
       - 2026-05-09: GitHub dropped → 2 sources (wikipedia + mktcap proxy)
       - 2026-05-10 (W-B-2): crsp_vintage added → 3 sources (Wave B skeleton)
       - 2026-05-12 (Path J): russell2000_proxy added → 4 sources (generic CRSP
         market-cap-rank loader; Path J uses rank 1001-3000, Path D uses rank 1-1500)
    """
    assert UNIVERSE_SOURCES_LOCKED == (
        "wikipedia_archive",
        "mktcap_top500_proxy",
        "crsp_vintage",
        "russell2000_proxy",
    )


def test_load_rejects_unknown_source():
    with pytest.raises(ValueError, match="not in"):
        load_sp500_constituents_at_date(
            as_of=datetime.date(2020, 6, 30),
            source="bogus_source",
        )


def test_load_rejects_non_date():
    with pytest.raises(TypeError):
        load_sp500_constituents_at_date(as_of="2020-06-30", source="wikipedia_archive")


def test_wikipedia_reconstruction_logic():
    """Mock Wikipedia data → verify reconstruction logic.

    Real Wikipedia structure: today's universe table + history of changes.
    A stock that was added 2015 AND is still in today's universe appears in BOTH:
      - 'current' (it's in today's set)
      - 'added' at 2015 (event of addition)

    Setup mocking real structure:
      Today's universe: {AAPL, MSFT, X}  (X added 2015 still in today)
      History:
        2015-01-01 added X (event)
        2018-06-01 removed Y (Y was historical member, not in today)
    """
    fake_wiki = pd.DataFrame([
        # current today's set (mock real Wikipedia structure)
        {"ticker": "AAPL", "action": "current", "date": pd.Timestamp("1990-01-01")},
        {"ticker": "MSFT", "action": "current", "date": pd.Timestamp("1990-01-01")},
        {"ticker": "X",    "action": "current", "date": pd.Timestamp("1990-01-01")},
        # historical change events
        {"ticker": "X",    "action": "added",   "date": pd.Timestamp("2015-01-01")},
        {"ticker": "Y",    "action": "removed", "date": pd.Timestamp("2018-06-01")},
    ])
    with mock.patch(
        "engine.universe_singlename.constituents_loader._fetch_wikipedia_sp500_history",
        return_value=fake_wiki,
    ):
        # At 2014-06-30: X added AFTER → remove from today_set;
        #                Y removed AFTER 2014-06 → add back (was in at 2014-06)
        # Expected: {AAPL, MSFT, Y}
        result_2014 = load_sp500_constituents_at_date(
            as_of=datetime.date(2014, 6, 30),
            source="wikipedia_archive",
        )
        assert "X" not in result_2014.tickers, "X added 2015-01 should NOT be in 2014-06 universe"
        assert "Y" in result_2014.tickers, "Y removed 2018-06 (after 2014-06) SHOULD be in 2014-06 universe"
        assert "AAPL" in result_2014.tickers
        assert "MSFT" in result_2014.tickers

        # At 2020-06-30: X added BEFORE 2020-06 → keep in today_set (no change);
        #                Y removed BEFORE 2020-06 → not in today_set, not added back
        # Expected: {AAPL, MSFT, X}
        result_2020 = load_sp500_constituents_at_date(
            as_of=datetime.date(2020, 6, 30),
            source="wikipedia_archive",
        )
        assert "X" in result_2020.tickers, "X added 2015-01 SHOULD be in 2020-06 universe"
        assert "Y" not in result_2020.tickers, "Y removed 2018-06 should NOT be in 2020-06 universe"
        assert "AAPL" in result_2020.tickers
        assert "MSFT" in result_2020.tickers


def test_proxy_returns_today_universe(tmp_path, monkeypatch):
    """mktcap_top500_proxy returns same set regardless of as_of (pure survivorship).

    Test isolation fix 2026-05-09: redirect cache paths to tmp_path to avoid
    polluting real disk cache (previous bug: real cache got overwritten with
    mock {AAPL, MSFT}, breaking subsequent Wave A run).
    """
    fake_wiki = pd.DataFrame([
        {"ticker": "AAPL", "action": "current", "date": pd.Timestamp("1990-01-01")},
        {"ticker": "MSFT", "action": "current", "date": pd.Timestamp("1990-01-01")},
    ])
    # Redirect cache paths to tmp_path so real caches stay untouched
    monkeypatch.setattr(
        "engine.universe_singlename.constituents_loader._WIKIPEDIA_CACHE",
        tmp_path / "_wikipedia.parquet",
    )
    monkeypatch.setattr(
        "engine.universe_singlename.constituents_loader._PROXY_CACHE",
        tmp_path / "_proxy.parquet",
    )
    with mock.patch(
        "engine.universe_singlename.constituents_loader._fetch_wikipedia_sp500_history",
        return_value=fake_wiki,
    ):
        r1 = load_sp500_constituents_at_date(
            as_of=datetime.date(2000, 6, 30),
            source="mktcap_top500_proxy",
        )
        r2 = load_sp500_constituents_at_date(
            as_of=datetime.date(2024, 6, 30),
            source="mktcap_top500_proxy",
        )
        assert r1.tickers == r2.tickers, "proxy should be invariant under as_of"
        assert "PURE SURVIVORSHIP" in r1.metadata.get("warning", "")


def test_result_dataclass_immutable():
    """SP500ConstituentsResult is frozen dataclass."""
    result = SP500ConstituentsResult(
        as_of=datetime.date(2020, 6, 30), source="wikipedia_archive",
        tickers=["A"], n_constituents=1, metadata={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError if False else Exception):
        result.tickers = ["B"]  # type: ignore


import dataclasses


# ── W-B-2 (Wave B CRSP vintage) tests ──────────────────────────────────────
def test_crsp_vintage_in_locked_sources():
    """crsp_vintage MUST be in UNIVERSE_SOURCES_LOCKED (Wave B activation gate)."""
    assert "crsp_vintage" in UNIVERSE_SOURCES_LOCKED


def test_crsp_vintage_mock_mode_returns_proxy_with_fallback_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When WRDS not configured, crsp_vintage MUST fall back to mktcap proxy
    AND tag metadata so downstream knows it's skeleton, not real Wave B data."""
    from engine.universe_singlename import crsp_loader as cl
    monkeypatch.setattr(cl, "is_wrds_available", lambda: False)

    result = load_sp500_constituents_at_date(
        as_of=datetime.date(2020, 6, 30),
        source="crsp_vintage",
    )
    assert result.source == "crsp_vintage"
    assert isinstance(result.tickers, list)
    assert len(result.tickers) > 0  # proxy fallback returns real ticker list
    assert result.metadata.get("WAVE_B_FALLBACK") is True
    assert "fallback_warning" in result.metadata
    assert "NOT YET ACTIVE" in result.metadata["fallback_warning"]


def test_crsp_vintage_real_path_no_longer_stubbed_post_activation() -> None:
    """Post-2026-05-11 (Wave B activation): real CRSP vintage constituents
    path is implemented. Guards against stub regression."""
    from engine.universe_singlename import constituents_loader as constit
    import inspect
    src = inspect.getsource(constit._fetch_crsp_vintage_constituents)
    assert "NotImplementedError" not in src, (
        "_fetch_crsp_vintage_constituents regressed to stub state."
    )


def test_crsp_vintage_force_mock_mode_works_independently_of_wrds_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mock_mode=True forces fallback even if WRDS is available — useful for
    deterministic skeleton testing in CI."""
    from engine.universe_singlename import crsp_loader as cl
    from engine.universe_singlename import constituents_loader as constit
    monkeypatch.setattr(cl, "is_wrds_available", lambda: True)  # WRDS available
    tickers, meta = constit._fetch_crsp_vintage_constituents(
        as_of=datetime.date(2020, 6, 30),
        mock_mode=True,  # but caller forces mock
    )
    assert len(tickers) > 0
    assert meta.get("WAVE_B_FALLBACK") is True


def test_crsp_vintage_real_path_no_wrds_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mock_mode=False + no WRDS → must raise informative RuntimeError."""
    from engine.universe_singlename import crsp_loader as cl
    from engine.universe_singlename import constituents_loader as constit
    monkeypatch.setattr(cl, "is_wrds_available", lambda: False)
    with pytest.raises(RuntimeError, match="WRDS not configured"):
        constit._fetch_crsp_vintage_constituents(
            as_of=datetime.date(2020, 6, 30),
            mock_mode=False,
        )


def test_load_sp500_at_date_dispatches_crsp_vintage_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public API load_sp500_constituents_at_date(source='crsp_vintage') must
    route to the new branch and return SP500ConstituentsResult."""
    from engine.universe_singlename import crsp_loader as cl
    monkeypatch.setattr(cl, "is_wrds_available", lambda: False)

    result = load_sp500_constituents_at_date(
        as_of=datetime.date(2024, 1, 31),
        source="crsp_vintage",
    )
    assert isinstance(result, SP500ConstituentsResult)
    assert result.as_of == datetime.date(2024, 1, 31)
    assert result.source == "crsp_vintage"
    assert result.n_constituents == len(result.tickers)
    assert result.tickers == sorted(result.tickers)  # sorted invariant
