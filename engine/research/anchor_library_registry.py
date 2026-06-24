"""engine.research.anchor_library_registry — declarative anchor library catalog.

Created 2026-06-09 as Option A of the senior施工建议 (see
[[project-tier-c-senior-construction-plan-2026-06-09]]).

REPLACES the per-module anchor-loading pattern (`load_famafrench_anchors`
in anchor_regression, `load_fx_carry_anchors` in fx_carry_anchor_regression,
`load_industry_anchors` in industry_attribution, `load_macro_anchors` +
`load_lrv_fx_anchors` in cross_asset_attribution) with a SINGLE
declarative registry. Each library is a frozen `AnchorLibrary`
registration carrying:

  - parquet path (relative to repo data/anchor_library)
  - canonical anchor column names
  - applicable asset_class set (matches FactorSpec.asset_class)
  - units ("decimal" or "percent" — auto-converted on load)
  - paper citation (surfaced in self_doubt prompt)
  - fetcher script hint (shown in error message when parquet missing)

ADDING A NEW ANCHOR FAMILY
=========================
Before this registry: copy ~300 lines of a sibling anchor module,
change constants, hope you didn't forget the `/100` unit conversion
or the SHA helper.

After this registry: add ONE entry below + ensure the parquet is
buildable via the named fetcher script. The factory pattern in
Commit A.2 (planned next) will then auto-generate the lens
declaration; until then sibling lens modules just call
load_library(name).

DESIGN NOTES
============
- We considered shipping `loader` as a Callable on each registration
  (max flexibility) but doing so reintroduces per-module functions
  with subtle inconsistencies. Forcing a single generic loader keeps
  unit conversion and missing-file handling identical across libraries.
- `applicable_asset_classes` matches the convention used in
  LensDeclaration.applicable_to so future Commit A.2 can directly
  reuse it without re-mapping.
- `paper_citation` is set up for self_doubt prompt enrichment — when
  the prompt cites the anchor library, the principal sees WHICH paper's
  factors the residual α stripped (currently we render only the library
  short name).
"""
from __future__ import annotations

import dataclasses as _dc
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.research.lens_helpers import file_sha256

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANCHOR_DIR = _REPO_ROOT / "data" / "anchor_library"


@_dc.dataclass(frozen=True)
class AnchorLibrary:
    """Declarative anchor-panel metadata. Frozen — registrations
    are immutable; updates require new commit + (eventually)
    bumping a SHA pin in events for old verdicts."""

    name:                       str            # internal key, e.g. "ken_french_ff5_mom"
    display_name:               str            # human label, e.g. "Ken French FF5+MOM"
    parquet_filename:           str            # under data/anchor_library/
    anchor_columns:             tuple[str, ...]
    applicable_asset_classes:   tuple[str, ...]   # matches FactorSpec.asset_class
    units:                      str            # "decimal" | "percent"
    paper_citation:             str            # for self_doubt prompt
    fetch_script_hint:          str            # error-message guidance

    @property
    def parquet_path(self) -> Path:
        return _ANCHOR_DIR / self.parquet_filename


# ────────────────────────────────────────────────────────────────────
# THE REGISTRY — single source of truth for all anchor libraries
# ────────────────────────────────────────────────────────────────────
ANCHOR_LIBRARIES: dict[str, AnchorLibrary] = {

    "ken_french_ff5_mom": AnchorLibrary(
        name                     = "ken_french_ff5_mom",
        display_name             = "Ken French FF5+MOM",
        parquet_filename         = "famafrench_monthly.parquet",
        anchor_columns           = ("MKT_RF", "SMB", "HML",
                                      "RMW", "CMA", "MOM"),
        # Matches original anchor_regression LENS_DECLARATION before
        # A.1 — cross_asset included because FF5+MOM is still
        # informative as a partial-attribution baseline even when
        # not the primary anchor family for that asset class.
        applicable_asset_classes = ("equity", "multi_asset", "cross_asset"),
        units                    = "decimal",
        paper_citation           = ("Fama-French 2015 (FF5) + "
                                      "Carhart 1997 (MOM)"),
        fetch_script_hint        = "scripts/fetch_anchor_library.py",
    ),

    "lrv_fx_carry": AnchorLibrary(
        name                     = "lrv_fx_carry",
        display_name             = "LRV HML_FX + DOL",
        parquet_filename         = "lrv_fx_carry_anchors_monthly.parquet",
        anchor_columns           = ("HML_FX", "DOL"),
        applicable_asset_classes = ("fx",),
        units                    = "percent",   # divided by 100 on load
        paper_citation           = ("Lustig-Roussanov-Verdelhan 2011 "
                                      "(HML_FX cross-sectional carry)"),
        fetch_script_hint        = ("python -c 'from engine.research."
                                      "fx_carry_anchors import "
                                      "build_and_cache_carry_anchors; "
                                      "build_and_cache_carry_anchors()'"),
    ),

    "ff12_us_industry": AnchorLibrary(
        name                     = "ff12_us_industry",
        display_name             = "Fama-French 12-Industry US",
        parquet_filename         = "industries_12_monthly.parquet",
        anchor_columns           = (
            "NoDur", "Durbl", "Manuf", "Enrgy", "Chems",
            "BusEq", "Telcm", "Utils", "Shops", "Hlth",
            "Money", "Other",
        ),
        applicable_asset_classes = ("equity", "multi_asset"),
        units                    = "decimal",
        paper_citation           = ("Fama-French 12-Industry "
                                      "(Ken French data library)"),
        fetch_script_hint        = "scripts/fetch_anchor_library.py",
    ),

    # B.3 (2026-06-10): MSSS 2012 global FX volatility. The
    # textbook-correct vol regressor for FX carry attribution —
    # VIX_change (macro_us) correlates but is US-equity vol;
    # MSSS Table 3 shows carry's loading on Δσ_FX explains the
    # HML_FX cross-sectional spread.
    "msss_gfx_vol": AnchorLibrary(
        name                     = "msss_gfx_vol",
        display_name             = "MSSS global FX volatility",
        parquet_filename         = "gfx_vol_monthly.parquet",
        anchor_columns           = ("GFX_VOL_change",),
        applicable_asset_classes = ("fx", "cross_asset", "multi_asset"),
        units                    = "decimal",
        paper_citation           = ("Menkhoff-Sarno-Schmeling-Schrimpf "
                                      "2012 (sigma_FX innovation)"),
        fetch_script_hint        = "scripts/fetch_gfx_vol_msss.py",
    ),

    "macro_us": AnchorLibrary(
        name                     = "macro_us",
        display_name             = "US macro panel (VIX/DXY/BAA/T10Y)",
        parquet_filename         = "cross_asset_macro_monthly.parquet",
        anchor_columns           = (
            "VIX_change", "DXY_return", "BAA_spread_change",
            "T10Y3M_change", "T10YIE_change",
        ),
        applicable_asset_classes = ("equity", "fx", "multi_asset",
                                      "cross_asset"),
        units                    = "decimal",
        paper_citation           = ("FRED VIX/DTWEXBGS/BAA/T10Y3M/T10YIE "
                                      "monthly changes"),
        fetch_script_hint        = "scripts/fetch_cross_asset_macro.py",
    ),
}


# ────────────────────────────────────────────────────────────────────
# Lookup + generic loader
# ────────────────────────────────────────────────────────────────────
def get_library(name: str) -> Optional[AnchorLibrary]:
    """Return the AnchorLibrary registration by name, or None."""
    return ANCHOR_LIBRARIES.get(name)


def libraries_for_asset_class(asset_class: str) -> list[AnchorLibrary]:
    """All AnchorLibrary registrations applicable to the given
    FactorSpec.asset_class. Order = ANCHOR_LIBRARIES dict order
    (insertion / canonical precedence)."""
    return [lib for lib in ANCHOR_LIBRARIES.values()
            if asset_class in lib.applicable_asset_classes]


def load_library(
    name: str,
    *,
    explicit_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Generic loader for any registered anchor library.

    Args:
      name: registry key (e.g. "ken_french_ff5_mom").
      explicit_path: override the registered parquet path (for tests
                     that want to point at a fixture).

    Returns:
      DataFrame indexed by month-end DatetimeIndex, columns =
      library.anchor_columns (only those actually present in the
      parquet). Units auto-converted to DECIMAL via library.units.
      None when the library is unknown or the parquet missing.
    """
    lib = get_library(name)
    if lib is None:
        logger.warning("anchor_library_registry: unknown library %r "
                          "(known: %s)", name,
                          sorted(ANCHOR_LIBRARIES.keys()))
        return None

    p = Path(explicit_path) if explicit_path else lib.parquet_path
    if not p.is_file():
        logger.info("anchor_library_registry: %s parquet missing at "
                      "%s (build via: %s)",
                      lib.name, p, lib.fetch_script_hint)
        return None

    df = pd.read_parquet(p)
    # Normalize index — some parquets store date as a column
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)

    # Select declared anchor columns; subset that actually present
    present = [c for c in lib.anchor_columns if c in df.columns]
    if not present:
        logger.warning("anchor_library_registry: %s parquet lacks "
                          "ANY of expected anchor columns %s; got %s",
                          lib.name, lib.anchor_columns,
                          list(df.columns))
        return None

    out = df[present].astype(float)

    # Unit conversion — anchors must be DECIMAL/month for the
    # downstream OLS contract. "percent" libraries (LRV: monthly
    # returns stored as %) get divided by 100.
    if lib.units == "percent":
        out = out / 100.0
    elif lib.units != "decimal":
        logger.warning("anchor_library_registry: unknown units %r "
                          "for %s; assuming decimal", lib.units, lib.name)

    out = out.sort_index()
    return out


def library_sha(
    name: str,
    *,
    explicit_path: Optional[str] = None,
) -> str:
    """SHA-256 of the library's cached parquet. Returns "" if absent."""
    lib = get_library(name)
    if lib is None:
        return ""
    p = Path(explicit_path) if explicit_path else lib.parquet_path
    return file_sha256(p)
