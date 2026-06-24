"""engine.research.fx_carry_anchor_regression — FX-carry anchor lens (B.1).

A.2 (2026-06-09): collapsed to a thin shim over
`engine.research.residual_alpha_lens.make_residual_alpha_lens`.

The actual residual-α math lives ONCE in
`engine.research.anchor_regression.compute_residual_alpha`; this module
just exposes the `LENS_DECLARATION` and a backwards-compat
`compute_for_tier_c_pnl_series` wrapper so consumers that imported
these names directly (sibling lenses, dispatcher, tests) keep working
without changes.

WHY THIS LENS EXISTS
====================
When auditing a CARRY sleeve through Tier C, the FF5+MOM anchor
regression correctly `skipped_inapplicable` because asset_class !=
"equity". Without an FX-asset-class counterpart, carry candidates
have NO anchor-orthogonality check — a thinly-disguised rephrasing
of LRV's HML_FX could pass the Sharpe gate.

This lens fills the gap with the LRV 2011 HML_FX + DOL panel:
  r_carry(t) = α + β_HML_FX · HML_FX(t) + β_DOL · DOL(t) + ε(t)

β_HML_FX ≈ 1 + R² ≈ 1 → degenerate restatement of HML_FX (sentinel
behavior — see [[project-tier-c-senior-construction-plan-2026-06-09]]).

A.2 SHIM CONVERSION
===================
The lens's Tier C wiring helper + LENS_DECLARATION are now produced
by the factory pattern. Adding a new single-stage anchor lens after
A.2 is one AnchorLibrary registration + a 4-line module like this
one — no copy-paste of regression logic.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from engine.research.anchor_library_registry import (
    get_library as _get_library,
    library_sha as _registry_library_sha,
    load_library as _registry_load_library,
)
from engine.research.residual_alpha_lens import (
    compute_for_tier_c_pnl_series as _factory_compute,
    make_residual_alpha_lens,
)

logger = logging.getLogger(__name__)


MIN_OVERLAP_MONTHS_DEFAULT = 24
ANCHOR_COLUMNS = _get_library("lrv_fx_carry").anchor_columns
_FX_CARRY_PARQUET = _get_library("lrv_fx_carry").parquet_path


# ────────────────────────────────────────────────────────────────────
# Backwards-compat loader/SHA shims (sibling modules + tests import
# these names directly)
# ────────────────────────────────────────────────────────────────────
def load_fx_carry_anchors(
    path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Routes through anchor_library_registry. Registered
    units="percent" applies the /100 unit conversion."""
    return _registry_load_library("lrv_fx_carry", explicit_path=path)


def _fx_carry_parquet_sha256(path: Optional[str] = None) -> str:
    """Routes through anchor_library_registry SHA helper."""
    return _registry_library_sha("lrv_fx_carry", explicit_path=path)


# ────────────────────────────────────────────────────────────────────
# Backwards-compat math wrapper (B.1 contract — `pnl, anchors → dict`)
# ────────────────────────────────────────────────────────────────────
def compute_fx_carry_residual_alpha(
    factor_pnl:    pd.Series,
    anchor_pnls:   pd.DataFrame,
    *,
    nw_lag:        Optional[int] = None,
    min_overlap:   int           = MIN_OVERLAP_MONTHS_DEFAULT,
    periods_per_year: int        = 12,
) -> Optional[dict]:
    """Delegating wrapper around anchor_regression.compute_residual_alpha
    — kept for direct callers (tests). The factory-built lens runner
    calls compute_residual_alpha directly via residual_alpha_lens."""
    from engine.research.anchor_regression import compute_residual_alpha
    return compute_residual_alpha(
        factor_pnl, anchor_pnls,
        nw_lag           = nw_lag,
        min_overlap      = min_overlap,
        periods_per_year = periods_per_year,
    )


# ────────────────────────────────────────────────────────────────────
# Tier C wiring helper — thin shim binding library_name
# ────────────────────────────────────────────────────────────────────
def compute_for_tier_c_pnl_series(
    pnl_series: pd.Series | pd.DataFrame,
    *,
    anchors:   Optional[pd.DataFrame] = None,
    artifacts: Optional[dict]         = None,
) -> Optional[dict]:
    """Binds library_name="lrv_fx_carry" and delegates to the generic
    factory helper. Kept for backwards compat — direct callers
    (sibling lenses, tests) import this name."""
    return _factory_compute(
        "lrv_fx_carry",
        pnl_series,
        anchors=anchors,
        artifacts=artifacts,
    )


# ────────────────────────────────────────────────────────────────────
# Lens registry declaration — auto-discovered by Phase 1 registry.
# Built by the factory; equivalent to the pre-A.2 inline declaration.
# ────────────────────────────────────────────────────────────────────
import sys as _sys

LENS_DECLARATION = make_residual_alpha_lens(
    "lrv_fx_carry",
    lens_name     = "fx_carry_anchor_regression",
    # See anchor_regression.py for the wiring_module rationale —
    # preserves patchability of the module-level
    # compute_for_tier_c_pnl_series helper.
    wiring_module = _sys.modules[__name__],
)
