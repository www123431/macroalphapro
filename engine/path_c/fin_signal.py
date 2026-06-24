"""
engine/path_c/fin_signal.py — Path D FIN factor cross-section composite + decile.

Pre-registration: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62) §2.4 step 3 + §2.6

Consumes raw FIN panel from engine.path_c.fin_signal_panel and produces:
  - z_nsi:  cross-section z-norm of nsi within each quarter
  - z_acc:  cross-section z-norm of acc_scaled within each quarter
  - fin:    composite = (z_nsi + z_acc) / 2   (NaN-safe average of available components)
  - fin_for_long_rank: -fin (high → bullish/long; per spec §2.5 negate convention)
  - fin_rank_pct: percentile rank within quarter on fin_for_long_rank
  - leg:    "long" (top decile) / "short" (bottom decile) / "flat"

DHS 2020 + Daniel-Titman 2006 + Sloan 1996 composite. 0 LLM (spec invariant).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.path_c.sue_signal import (
    rank_within_quarter,
    assign_decile_legs,
)

logger = logging.getLogger(__name__)


# Locked thresholds (per spec §2.6 + §六, identical to id=57/58/59/60)
DECILE_LONG_THRESHOLD:  float = 0.9
DECILE_SHORT_THRESHOLD: float = 0.1


# ── Cross-section z-norm ───────────────────────────────────────────────────
def _z_within_quarter(
    panel:       pd.DataFrame,
    value_col:   str,
    quarter_col: str = "fiscal_yearq",
) -> pd.Series:
    """Compute cross-section z-norm of `value_col` within each quarter.

    Returns a Series aligned with panel index. NaN values pass through as NaN
    (they don't contribute to the within-quarter mean/std).
    """
    if panel.empty:
        return pd.Series(dtype=float, index=panel.index)

    def _zscore(s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if len(s_clean) < 2:
            return pd.Series(np.nan, index=s.index)
        mu = s_clean.mean()
        sd = s_clean.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    return panel.groupby(quarter_col, group_keys=False)[value_col].apply(_zscore)


def compute_fin_composite(panel: pd.DataFrame) -> pd.DataFrame:
    """Add z_nsi + z_acc + fin composite columns.

    Composite per spec §2.4 step 3:
      fin = (z_nsi + z_acc) / 2

    NaN-handling: if NSI raw missing, use z_acc alone (and vice versa).
    If both missing, fin = NaN.

    Does not mutate input; returns copy.
    """
    if panel.empty:
        out = panel.copy()
        for col in ("z_nsi", "z_acc", "fin"):
            out[col] = pd.Series(dtype=float)
        return out

    required = {"nsi", "acc_scaled", "fiscal_yearq"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"compute_fin_composite: panel missing columns {missing}")

    out = panel.copy()
    out["z_nsi"] = _z_within_quarter(out, "nsi")
    out["z_acc"] = _z_within_quarter(out, "acc_scaled")

    # Composite: average of available components
    # If both present: (z_nsi + z_acc) / 2
    # If only one present: that component
    # If neither: NaN
    znsi = out["z_nsi"]
    zacc = out["z_acc"]
    both = znsi.notna() & zacc.notna()
    only_nsi = znsi.notna() & zacc.isna()
    only_acc = znsi.isna() & zacc.notna()

    fin = pd.Series(np.nan, index=out.index)
    fin.loc[both]     = (znsi.loc[both] + zacc.loc[both]) / 2.0
    fin.loc[only_nsi] = znsi.loc[only_nsi]
    fin.loc[only_acc] = zacc.loc[only_acc]
    out["fin"] = fin
    # Negate for long-leg ranking convention per spec §2.5 step 2
    out["fin_for_long_rank"] = -fin
    return out


def assign_fin_decile_legs(
    panel: pd.DataFrame,
    *,
    long_threshold:  float = DECILE_LONG_THRESHOLD,
    short_threshold: float = DECILE_SHORT_THRESHOLD,
) -> pd.DataFrame:
    """End-to-end: compute fin composite + rank within quarter + assign decile legs.

    Pipeline:
      1. compute_fin_composite (adds z_nsi, z_acc, fin, fin_for_long_rank)
      2. rank_within_quarter on fin_for_long_rank (top decile = bullish / low FIN)
      3. assign_decile_legs at thresholds (long/short/flat)

    Tie-break: ascending ticker (per spec §2.4 deterministic).
    """
    if panel.empty:
        out = panel.copy()
        for col in ("z_nsi", "z_acc", "fin", "fin_for_long_rank", "fin_rank_pct", "leg"):
            out[col] = pd.Series(dtype=float if col != "leg" else object)
        return out

    out = compute_fin_composite(panel)
    # Use rank_within_quarter helper (parameterized: sue_col → fin_for_long_rank)
    out = rank_within_quarter(
        out,
        sue_col="fin_for_long_rank",
        tie_break_col="ticker",
    )
    # The helper writes sue_rank_pct; rename to fin_rank_pct for clarity
    out = out.rename(columns={"sue_rank_pct": "fin_rank_pct"})
    # Apply decile assignment
    out = assign_decile_legs(
        out,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        rank_col="fin_rank_pct",
    )
    return out
