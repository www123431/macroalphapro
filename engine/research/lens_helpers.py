"""engine.research.lens_helpers — shared utilities for Tier C lenses.

Created 2026-06-09 in response to user feedback "你做这些好像又回到
硬编码了？" — the column-naming guess pattern was being copy-pasted
across lens modules (fx_carry_anchor_regression had its own
`_pick_net_column`, subsample_stability was about to grow another).

PRINCIPLE
=========
Templates DECLARE their column conventions; lenses READ those
declarations. No more digit-extraction guessing.

Templates declare via the `artifacts` dict on TemplateResult:
  artifacts = {
      "pnl_series_df":   <DataFrame>,
      "pnl_default_col": "pnl_net_13bp",   # equity convention
      # or
      "pnl_default_col": "pnl_net_8bp",    # FX-carry convention
      "pnl_gross_col":   "pnl_gross",      # optional; defaults to "pnl_gross"
  }

Lenses call resolve_default_net_col(artifacts) which:
  1. Returns artifacts["pnl_default_col"] if present — NO GUESSING.
  2. Otherwise falls back to legacy heuristic (lowest `pnl_net_<N>bp`
     column) for backwards compat with templates not yet migrated.
  3. Returns None if neither path resolves a column.

Migration target: every template under
engine.agents.strengthener.templates.* declares the keys; legacy
fallback is then deletable.
"""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Newey-West HAC lag rule (Newey-West 1987 + Stock-Watson textbook)
# Replaces 3 identical copies across anchor_regression /
# industry_attribution / cross_asset_attribution. Moved here so
# any new lens uses the same lag formula.
# ────────────────────────────────────────────────────────────────────
def nw_lag_rule_of_thumb(n: int) -> int:
    """Newey-West HAC lag length: floor(4 · (N/100)^(2/9))."""
    if n <= 0:
        return 0
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


# ────────────────────────────────────────────────────────────────────
# Anchor parquet SHA256 — provenance pinning for reproducibility.
# Replaces 4 identical implementations across the anchor modules.
# ────────────────────────────────────────────────────────────────────
def file_sha256(path) -> str:
    """SHA-256 hex digest of a file. Returns "" if file is missing
    (callers use this as the "no provenance to pin" sentinel)."""
    p = Path(path) if not isinstance(path, Path) else path
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ────────────────────────────────────────────────────────────────────
# Canonical artifact keys (controlled vocabulary)
# ────────────────────────────────────────────────────────────────────
ARTIFACT_KEY_PNL_DF          = "pnl_series_df"
ARTIFACT_KEY_PNL_DEFAULT_COL = "pnl_default_col"
ARTIFACT_KEY_PNL_GROSS_COL   = "pnl_gross_col"

DEFAULT_GROSS_COL_NAME = "pnl_gross"


def _legacy_pick_net_col(df: pd.DataFrame) -> Optional[str]:
    """LEGACY fallback — pick lowest-bp `pnl_net_<N>bp` column.

    DEPRECATED but kept for backwards compat. Once all templates
    under engine.agents.strengthener.templates.* declare
    `pnl_default_col` explicitly, this helper can be removed and
    resolve_default_net_col raises on missing declaration.
    """
    candidates = [c for c in df.columns
                    if isinstance(c, str) and c.startswith("pnl_net_")]
    if not candidates:
        return None
    def _bp(col: str) -> int:
        try:
            return int("".join(ch for ch in col if ch.isdigit()))
        except ValueError:
            return 10_000
    return min(candidates, key=_bp)


def resolve_default_net_col(
    artifacts: Optional[dict],
) -> Optional[str]:
    """Return the column name lenses should treat as the
    default-cost net PnL series.

    Resolution order:
      1. artifacts["pnl_default_col"] — explicit declaration (preferred)
      2. legacy fallback: lowest-bp `pnl_net_<N>bp` column on the
         template's pnl_series_df
      3. None — caller decides (typically falls back to gross or
         returns None as a lens output)

    Args:
      artifacts: template_result.artifacts dict (may be None / missing
                 keys). Tolerant of None to keep lens-runner call
                 sites simple.

    Returns:
      Column name string or None.
    """
    if not artifacts:
        return None

    # Path 1: explicit declaration
    declared = artifacts.get(ARTIFACT_KEY_PNL_DEFAULT_COL)
    if isinstance(declared, str) and declared:
        # Sanity: declared column must actually exist on the df
        pnl_df = artifacts.get(ARTIFACT_KEY_PNL_DF)
        if (isinstance(pnl_df, pd.DataFrame)
                and declared in pnl_df.columns):
            return declared
        logger.warning(
            "lens_helpers: artifacts declared pnl_default_col=%r "
            "but column missing from pnl_series_df; falling back "
            "to legacy heuristic", declared,
        )

    # Path 2: legacy fallback
    pnl_df = artifacts.get(ARTIFACT_KEY_PNL_DF)
    if isinstance(pnl_df, pd.DataFrame) and not pnl_df.empty:
        return _legacy_pick_net_col(pnl_df)

    return None


def resolve_gross_col(
    artifacts: Optional[dict],
) -> Optional[str]:
    """Return the column name lenses should treat as the gross PnL
    series.

    Resolution order:
      1. artifacts["pnl_gross_col"] — explicit declaration
      2. "pnl_gross" — universal convention (all current templates use it)
      3. None — no gross series available (rare)
    """
    if not artifacts:
        return None

    declared = artifacts.get(ARTIFACT_KEY_PNL_GROSS_COL)
    if isinstance(declared, str) and declared:
        pnl_df = artifacts.get(ARTIFACT_KEY_PNL_DF)
        if (isinstance(pnl_df, pd.DataFrame)
                and declared in pnl_df.columns):
            return declared

    pnl_df = artifacts.get(ARTIFACT_KEY_PNL_DF)
    if (isinstance(pnl_df, pd.DataFrame)
            and DEFAULT_GROSS_COL_NAME in pnl_df.columns):
        return DEFAULT_GROSS_COL_NAME
    return None


def slice_pnl_net_and_gross(
    artifacts: Optional[dict],
) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    """One-shot helper: resolve both columns + return the Series.
    Returns (net_series, gross_series). Either may be None.

    Convenience for lens runners that need both — single source of
    truth for the artifact contract.
    """
    if not artifacts:
        return None, None
    pnl_df = artifacts.get(ARTIFACT_KEY_PNL_DF)
    if not isinstance(pnl_df, pd.DataFrame) or pnl_df.empty:
        return None, None
    net_col   = resolve_default_net_col(artifacts)
    gross_col = resolve_gross_col(artifacts)
    net_series   = (pnl_df[net_col].dropna()
                       if net_col and net_col in pnl_df.columns else None)
    gross_series = (pnl_df[gross_col].dropna()
                       if gross_col and gross_col in pnl_df.columns else None)
    return net_series, gross_series
