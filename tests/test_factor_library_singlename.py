"""
tests/test_factor_library_singlename.py — unit tests for Tier 1 mining content layer.

Coverage:
  - FactorSpecSinglename validation (asset_class, expected_sign)
  - register_factor idempotency + integrity
  - get_factor / list_factors lazy import behavior
  - Type alias contract (signal_fn signature)
"""
from __future__ import annotations

import datetime

import pandas as pd
import pytest

from engine import factor_library_singlename as flsn
from engine.factor_library_singlename import (
    FACTOR_REGISTRY_SINGLENAME,
    FactorSpecSinglename,
    SignalFnSinglename,
    get_factor,
    list_factors,
    register_factor,
)


# ── Fixture: isolate registry state, restore after each test ────────────────
@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test sees an empty registry; restore prior state after teardown
    so subsequent test files (test_ivol, test_strev, test_mining_runner)
    still see the IVOL/STREV registrations from module-load.

    Using `.clear()` + `.update(saved)` preserves the same dict object so
    monkeypatched references stay valid; saving + restoring prevents
    cross-file pollution.
    """
    saved = dict(flsn.FACTOR_REGISTRY_SINGLENAME)
    flsn.FACTOR_REGISTRY_SINGLENAME.clear()
    monkeypatch.setattr(flsn, "_registry_populated", False)
    monkeypatch.setattr(flsn, "_KNOWN_FACTOR_MODULES", ())
    yield
    flsn.FACTOR_REGISTRY_SINGLENAME.clear()
    flsn.FACTOR_REGISTRY_SINGLENAME.update(saved)


# ── Helper: minimal valid spec for tests ────────────────────────────────────
def _trivial_signal_fn(
    as_of:    datetime.date,
    universe: list[str],
    panel:    pd.DataFrame,
) -> pd.Series:
    """Returns 0 z-score for every ticker in universe — placeholder for tests."""
    return pd.Series(0.0, index=universe, dtype=float)


def _make_spec(factor_id: str = "test_factor", expected_sign: int = -1) -> FactorSpecSinglename:
    return FactorSpecSinglename(
        factor_id       = factor_id,
        citation        = "Test (2026) Test Journal 1(1):1-10",
        asset_class     = "equity_singlename",
        formula_summary = "test factor for unit tests",
        signal_fn       = _trivial_signal_fn,
        expected_sign   = expected_sign,
    )


# ── FactorSpecSinglename frozen dataclass invariants ────────────────────────
def test_factor_spec_is_frozen() -> None:
    spec = _make_spec()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        spec.factor_id = "modified"  # type: ignore


def test_factor_spec_carries_all_metadata() -> None:
    spec = _make_spec(factor_id="ivol_test", expected_sign=-1)
    assert spec.factor_id == "ivol_test"
    assert spec.citation.startswith("Test")
    assert spec.asset_class == "equity_singlename"
    assert spec.formula_summary
    assert callable(spec.signal_fn)
    assert spec.expected_sign == -1


# ── register_factor validation ──────────────────────────────────────────────
def test_register_factor_basic() -> None:
    spec = _make_spec(factor_id="basic_factor")
    register_factor(spec)
    assert "basic_factor" in FACTOR_REGISTRY_SINGLENAME
    assert FACTOR_REGISTRY_SINGLENAME["basic_factor"] is spec


def test_register_factor_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="expected FactorSpecSinglename"):
        register_factor("not a spec")  # type: ignore


def test_register_factor_rejects_wrong_asset_class() -> None:
    bad = FactorSpecSinglename(
        factor_id="bad", citation="x", asset_class="equity_etf",
        formula_summary="x", signal_fn=_trivial_signal_fn, expected_sign=1,
    )
    with pytest.raises(ValueError, match="equity_singlename"):
        register_factor(bad)


def test_register_factor_rejects_invalid_expected_sign() -> None:
    bad = FactorSpecSinglename(
        factor_id="bad", citation="x", asset_class="equity_singlename",
        formula_summary="x", signal_fn=_trivial_signal_fn, expected_sign=0,
    )
    with pytest.raises(ValueError, match="expected_sign must be"):
        register_factor(bad)


def test_register_factor_idempotent_for_same_spec() -> None:
    spec = _make_spec(factor_id="dup_factor")
    register_factor(spec)
    register_factor(spec)  # second call: silent no-op
    assert len(FACTOR_REGISTRY_SINGLENAME) == 1


def test_register_factor_rejects_different_spec_with_same_id() -> None:
    spec_a = _make_spec(factor_id="conflict", expected_sign=-1)
    spec_b = _make_spec(factor_id="conflict", expected_sign=+1)  # different sign
    register_factor(spec_a)
    with pytest.raises(ValueError, match="already registered with different spec"):
        register_factor(spec_b)


# ── get_factor / list_factors lookup ────────────────────────────────────────
def test_get_factor_returns_registered() -> None:
    spec = _make_spec(factor_id="lookup_test")
    register_factor(spec)
    got = get_factor("lookup_test")
    assert got is spec


def test_get_factor_raises_for_unknown() -> None:
    with pytest.raises(KeyError, match="not in FACTOR_REGISTRY_SINGLENAME"):
        get_factor("nonexistent_factor")


def test_list_factors_returns_sorted_ids() -> None:
    register_factor(_make_spec(factor_id="zebra"))
    register_factor(_make_spec(factor_id="alpha"))
    register_factor(_make_spec(factor_id="mike"))
    assert list_factors() == ["alpha", "mike", "zebra"]


def test_list_factors_empty_registry() -> None:
    assert list_factors() == []


# ── Lazy import behavior ────────────────────────────────────────────────────
def test_lazy_import_triggers_on_get_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_factor should trigger _ensure_registry_populated()."""
    called = {"flag": False}

    def fake_ensure() -> None:
        called["flag"] = True

    monkeypatch.setattr(flsn, "_ensure_registry_populated", fake_ensure)
    with pytest.raises(KeyError):
        get_factor("anything")
    assert called["flag"] is True


def test_lazy_import_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ensure_registry_populated() should not re-import after first call."""
    monkeypatch.setattr(flsn, "_KNOWN_FACTOR_MODULES", ())
    flsn._ensure_registry_populated()
    assert flsn._registry_populated is True
    # Second call: should be no-op (state unchanged)
    flsn._ensure_registry_populated()
    assert flsn._registry_populated is True


def test_lazy_import_failure_is_warned_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    caplog:      pytest.LogCaptureFixture,
) -> None:
    """If a known factor module fails to import, log warning, don't raise."""
    monkeypatch.setattr(
        flsn, "_KNOWN_FACTOR_MODULES",
        ("engine.factors_singlename.nonexistent_module",),
    )
    monkeypatch.setattr(flsn, "_registry_populated", False)
    import logging
    with caplog.at_level(logging.WARNING):
        flsn._ensure_registry_populated()  # should not raise
    assert any("lazy import" in r.message for r in caplog.records)


# ── Signal_fn signature contract ────────────────────────────────────────────
def test_signal_fn_returns_pd_series_indexed_by_ticker() -> None:
    """SignalFnSinglename contract: returns pd.Series indexed by ticker."""
    sig = _trivial_signal_fn(
        as_of=datetime.date(2024, 6, 28),
        universe=["AAPL", "MSFT", "GOOG"],
        panel=pd.DataFrame(),
    )
    assert isinstance(sig, pd.Series)
    assert set(sig.index) == {"AAPL", "MSFT", "GOOG"}


# ── Boundary invariant (no LLM imports) ─────────────────────────────────────
def test_module_has_no_llm_imports() -> None:
    """Per spec_factor_lab.md §6 + memory: zero LLM imports in content layer."""
    src = open(
        flsn.__file__, encoding="utf-8",
    ).read()
    forbidden = ["google.generativeai", "google.genai",
                 "from engine.deepseek_client", "from engine.key_pool"]
    for pattern in forbidden:
        assert pattern not in src, (
            f"factor_library_singlename violates 0-LLM-imports invariant: "
            f"found {pattern!r}"
        )
