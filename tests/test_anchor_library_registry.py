"""tests/test_anchor_library_registry.py — Option A.1 registry tests.

Locks the AnchorLibrary registration contract + the generic loader's
behavior across the 4 registered libraries (FF5+MOM, LRV FX, FF12,
US macro).

Unit conversion is the highest-stakes contract — if a future library
forgets to declare units="percent" the residual α regression silently
runs with 100× anchor magnitudes and produces garbage. These tests
lock it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────────
# Registry shape — what's registered
# ────────────────────────────────────────────────────────────────────
def test_libraries_registered():
    """B.3 (2026-06-10): msss_gfx_vol joined — 5 libraries."""
    from engine.research.anchor_library_registry import ANCHOR_LIBRARIES
    expected = {"ken_french_ff5_mom", "lrv_fx_carry",
                  "ff12_us_industry", "macro_us", "msss_gfx_vol"}
    assert set(ANCHOR_LIBRARIES.keys()) == expected


def test_each_library_has_required_metadata():
    from engine.research.anchor_library_registry import ANCHOR_LIBRARIES
    for name, lib in ANCHOR_LIBRARIES.items():
        assert lib.name == name
        assert lib.display_name
        assert lib.parquet_filename.endswith(".parquet")
        assert lib.anchor_columns
        assert lib.applicable_asset_classes
        assert lib.units in ("decimal", "percent")
        assert lib.paper_citation
        assert lib.fetch_script_hint


def test_lrv_fx_declares_percent_units():
    """Critical: LRV parquet stores monthly returns as % values
    (HML_FX mean ~0.18 / std ~2.1). Loader MUST divide by 100 to
    align with decimal-convention factor PnL. Test locks this."""
    from engine.research.anchor_library_registry import get_library
    lib = get_library("lrv_fx_carry")
    assert lib.units == "percent"


def test_other_libraries_declare_decimal_units():
    """FF5+MOM, FF12, US macro all stored as decimals — anchors
    align directly with template factor PnL units."""
    from engine.research.anchor_library_registry import get_library
    for name in ("ken_french_ff5_mom", "ff12_us_industry", "macro_us"):
        assert get_library(name).units == "decimal", (
            f"{name} should be decimal units"
        )


def test_unknown_library_returns_none():
    from engine.research.anchor_library_registry import (
        get_library, load_library, library_sha,
    )
    assert get_library("nope_does_not_exist") is None
    assert load_library("nope_does_not_exist") is None
    assert library_sha("nope_does_not_exist") == ""


# ────────────────────────────────────────────────────────────────────
# Applicability filter — matches FactorSpec.asset_class
# ────────────────────────────────────────────────────────────────────
def test_fx_asset_class_picks_lrv_and_macro():
    from engine.research.anchor_library_registry import (
        libraries_for_asset_class,
    )
    libs = {l.name for l in libraries_for_asset_class("fx")}
    assert "lrv_fx_carry" in libs
    assert "macro_us" in libs
    # FF5+MOM (equity-only) MUST NOT apply to FX
    assert "ken_french_ff5_mom" not in libs
    assert "ff12_us_industry" not in libs


def test_equity_asset_class_picks_ff5_industry_macro():
    from engine.research.anchor_library_registry import (
        libraries_for_asset_class,
    )
    libs = {l.name for l in libraries_for_asset_class("equity")}
    assert "ken_french_ff5_mom" in libs
    assert "ff12_us_industry" in libs
    assert "macro_us" in libs
    # LRV FX carry (fx-only) MUST NOT apply to equity
    assert "lrv_fx_carry" not in libs


# ────────────────────────────────────────────────────────────────────
# Generic loader — fixture parquet with unit conversion
# ────────────────────────────────────────────────────────────────────
def test_load_library_applies_percent_unit_conversion(tmp_path):
    """Build a tiny fixture LRV-shape parquet with values in % and
    confirm loader divides by 100."""
    from engine.research.anchor_library_registry import load_library

    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    df_pct = pd.DataFrame({
        "date":   idx,
        "HML_FX": np.full(24, 2.5),   # 2.5% per month
        "DOL":    np.full(24, 0.5),
        # spurious columns get dropped by the registered column subset
        "P_HIGH": np.full(24, 3.0),
    })
    parquet_path = tmp_path / "fixture.parquet"
    df_pct.to_parquet(parquet_path, index=False)

    out = load_library("lrv_fx_carry", explicit_path=str(parquet_path))
    assert out is not None
    assert "HML_FX" in out.columns
    assert "DOL" in out.columns
    # Spurious column dropped
    assert "P_HIGH" not in out.columns
    # Unit converted to decimal
    np.testing.assert_array_almost_equal(out["HML_FX"].values,
                                            np.full(24, 0.025))
    np.testing.assert_array_almost_equal(out["DOL"].values,
                                            np.full(24, 0.005))


def test_load_library_skips_unit_conversion_for_decimal(tmp_path):
    """Decimal-library parquet values pass through unchanged."""
    from engine.research.anchor_library_registry import load_library

    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    df = pd.DataFrame({
        "date":   idx,
        "MKT_RF": np.full(24, 0.008),
        "SMB":    np.full(24, 0.002),
        "HML":    np.full(24, 0.003),
        "RMW":    np.full(24, 0.003),
        "CMA":    np.full(24, 0.002),
        "MOM":    np.full(24, 0.005),
    })
    parquet_path = tmp_path / "ff5mom_fixture.parquet"
    df.to_parquet(parquet_path, index=False)

    out = load_library("ken_french_ff5_mom",
                          explicit_path=str(parquet_path))
    assert out is not None
    np.testing.assert_array_almost_equal(out["MKT_RF"].values,
                                            np.full(24, 0.008))


def test_load_library_returns_none_when_parquet_missing(tmp_path):
    from engine.research.anchor_library_registry import load_library
    nonexistent = str(tmp_path / "missing.parquet")
    assert load_library("ken_french_ff5_mom",
                          explicit_path=nonexistent) is None


def test_load_library_handles_no_matching_columns(tmp_path):
    """Parquet exists but lacks ALL declared anchor columns →
    warns + returns None (signal to caller something is wrong)."""
    from engine.research.anchor_library_registry import load_library
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    df = pd.DataFrame({
        "date":         idx,
        "WRONG_COL_1":  np.zeros(24),
        "WRONG_COL_2":  np.zeros(24),
    })
    parquet_path = tmp_path / "bad.parquet"
    df.to_parquet(parquet_path, index=False)
    assert load_library("ken_french_ff5_mom",
                          explicit_path=str(parquet_path)) is None


def test_load_library_subset_present_columns(tmp_path):
    """Parquet has SOME declared columns + extras — loader picks
    just the present declared ones."""
    from engine.research.anchor_library_registry import load_library
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    df = pd.DataFrame({
        "date":   idx,
        "MKT_RF": np.full(24, 0.008),
        "SMB":    np.full(24, 0.002),
        # missing HML/RMW/CMA/MOM
    })
    parquet_path = tmp_path / "partial.parquet"
    df.to_parquet(parquet_path, index=False)
    out = load_library("ken_french_ff5_mom",
                          explicit_path=str(parquet_path))
    assert out is not None
    assert set(out.columns) == {"MKT_RF", "SMB"}


# ────────────────────────────────────────────────────────────────────
# SHA — file_sha256 wrapping
# ────────────────────────────────────────────────────────────────────
def test_library_sha_returns_empty_when_missing(tmp_path):
    from engine.research.anchor_library_registry import library_sha
    assert library_sha("ken_french_ff5_mom",
                          explicit_path=str(tmp_path / "no.parquet")) == ""


def test_library_sha_is_64_hex_when_present(tmp_path):
    from engine.research.anchor_library_registry import library_sha
    p = tmp_path / "fixture.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(p, index=False)
    sha = library_sha("ken_french_ff5_mom", explicit_path=str(p))
    assert len(sha) == 64
    # Hex chars only
    int(sha, 16)


# ────────────────────────────────────────────────────────────────────
# Real cached parquets — integration (auto-skip if absent)
# ────────────────────────────────────────────────────────────────────
def _real_cached(library_name: str) -> bool:
    from engine.research.anchor_library_registry import get_library
    lib = get_library(library_name)
    return lib is not None and lib.parquet_path.is_file()


@pytest.mark.skipif(not _real_cached("lrv_fx_carry"),
                     reason="LRV parquet not cached")
def test_integration_lrv_loaded_in_decimal():
    """Real LRV cached parquet: after loader unit conversion, HML_FX
    decimal values must be in [-0.10, +0.10] range (monthly returns
    typically ~0.5-2% in decimal). If they came back > 0.5 the /100
    conversion failed silently."""
    from engine.research.anchor_library_registry import load_library
    df = load_library("lrv_fx_carry")
    assert df is not None
    # Monthly FX carry returns in decimal units stay below ~15%
    # even in crisis months (Aug 2008 ~11% would be a typical worst
    # observation). Without the /100 conversion the sentinel is
    # ~10+ vs 0.10 — bar of 0.15 still catches missing conversion.
    assert df["HML_FX"].abs().max() < 0.15, (
        f"HML_FX max abs = {df['HML_FX'].abs().max()} — "
        "unit conversion to decimal looks wrong"
    )
    assert df["DOL"].abs().max() < 0.20


@pytest.mark.skipif(not _real_cached("ken_french_ff5_mom"),
                     reason="FF5+MOM parquet not cached")
def test_integration_ff5mom_loaded_in_decimal():
    from engine.research.anchor_library_registry import load_library
    df = load_library("ken_french_ff5_mom")
    assert df is not None
    # FF5+MOM monthly factor returns: similar range in decimal
    assert df["MKT_RF"].abs().max() < 0.30


@pytest.mark.skipif(not _real_cached("ken_french_ff5_mom"),
                     reason="FF5+MOM parquet not cached")
def test_integration_library_sha_stable_across_calls():
    """SHA must be deterministic — same parquet → same hash."""
    from engine.research.anchor_library_registry import library_sha
    sha1 = library_sha("ken_french_ff5_mom")
    sha2 = library_sha("ken_french_ff5_mom")
    assert sha1 == sha2
    assert len(sha1) == 64
