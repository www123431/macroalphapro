"""
engine/path_c/sue_signal.py — Path C #1 PEAD SUE signal + decile-leg assignment.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57) §2.3 + §2.4

Consumes the firm-quarter panel from engine.path_c.earnings_panel and adds:
  - `sue` column: SUE = (actual_eps - consensus_median) / consensus_dispersion
    (Livnat & Mendenhall 2006 standard form)
  - `sue_rank_pct` column: cross-sectional percentile within each quarter
    (tie-break: ascending ticker_ibes per spec §N6 deterministic)
  - `leg` column: "long" (≥ 90th pct) / "short" (≤ 10th pct) / "flat" (middle)

This module is SIGNAL LAYER. Portfolio formation lives in
engine/path_c/pead_backtest.py (Sprint 4). Verdict lives in
engine/path_c/verdict.py (Sprint 5).

All operations deterministic, 0 LLM (spec invariant).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.path_c import (
    DECILE_LONG_THRESHOLD,
    DECILE_SHORT_THRESHOLD,
    MIN_ANALYSTS_REQUIRED,
)

logger = logging.getLogger(__name__)


# ── SUE formula ─────────────────────────────────────────────────────────────
def compute_sue(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `sue` column = (actual_eps - consensus_median) / consensus_dispersion.

    Per Livnat & Mendenhall 2006: dispersion = cross-analyst std of forecasts
    in the 90d pre-rdq window (computed in earnings_panel Sprint 2).

    Defensive (Sprint 2 already filters most invalid rows):
      - dispersion == 0 → SUE = NaN (later excluded by rank step)
      - any input NaN → SUE = NaN
      - n_analysts < MIN_ANALYSTS_REQUIRED → SUE = NaN

    Returns a copy with `sue` column appended; does not mutate input.
    """
    if panel.empty:
        return panel.copy()

    required = {"actual_eps", "consensus_median", "consensus_dispersion", "n_analysts"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"compute_sue: panel missing columns {missing}")

    out = panel.copy()
    # Safe division: where dispersion ≤ 0 or NaN, SUE = NaN
    dispersion = out["consensus_dispersion"].astype(float)
    actual = out["actual_eps"].astype(float)
    consensus = out["consensus_median"].astype(float)
    sue_numer = actual - consensus
    with np.errstate(divide="ignore", invalid="ignore"):
        sue_values = np.where(
            (dispersion > 0) & np.isfinite(dispersion),
            sue_numer / dispersion,
            np.nan,
        )
    # Additional defensive filter on n_analysts
    insufficient_analysts = out["n_analysts"].fillna(0).astype(int) < MIN_ANALYSTS_REQUIRED
    sue_values = np.where(insufficient_analysts, np.nan, sue_values)

    out["sue"] = sue_values
    return out


# ── Cross-sectional rank within quarter ────────────────────────────────────
def rank_within_quarter(
    panel: pd.DataFrame,
    *,
    quarter_col: str = "fiscal_yearq",
    sue_col:     str = "sue",
    tie_break_col: str = "ticker_ibes",
) -> pd.DataFrame:
    """Compute cross-sectional SUE percentile within each quarter.

    Adds `sue_rank_pct` ∈ [0.0, 1.0] (the firm's empirical-CDF rank among
    that quarter's firms with valid SUE).

    Tie-break rule per spec §N6: ascending `ticker_ibes` (deterministic;
    avoids decile-boundary ambiguity).

    Rows with SUE=NaN get sue_rank_pct=NaN and are excluded from the
    denominator (won't influence other firms' ranks).
    """
    if panel.empty:
        out = panel.copy()
        out["sue_rank_pct"] = pd.Series(dtype=float)
        return out

    if sue_col not in panel.columns:
        raise ValueError(f"rank_within_quarter: '{sue_col}' column required (run compute_sue first)")
    if quarter_col not in panel.columns:
        raise ValueError(f"rank_within_quarter: '{quarter_col}' column required")
    if tie_break_col not in panel.columns:
        raise ValueError(f"rank_within_quarter: '{tie_break_col}' column required")

    out = panel.copy().reset_index(drop=True)

    rank_pcts = pd.Series(np.nan, index=out.index, dtype=float)

    for quarter, group_idx in out.groupby(quarter_col).groups.items():
        group = out.loc[group_idx]
        valid_mask = group[sue_col].notna()
        valid_group = group[valid_mask]
        n_valid = len(valid_group)
        if n_valid < 2:
            # Too few firms to rank meaningfully — leave NaN
            continue
        # Sort by (SUE asc, tie_break asc) → assign rank 1..n
        sorted_group = valid_group.sort_values(
            by=[sue_col, tie_break_col], ascending=[True, True], kind="mergesort"
        )
        ranks = np.arange(1, n_valid + 1)
        # Mid-rank empirical CDF: pct = (rank - 0.5) / n_valid.
        # Standard convention (Livnat-Mendenhall 2006, AFP 2019): symmetric
        # decile counts for n divisible by 10. For n=100: bottom-10 gets pct in
        # (0.005..0.095) → all ≤ 0.10 short; top-10 gets (0.905..0.995) →
        # all ≥ 0.90 long. Spec §N6 boundary inclusiveness preserved at the
        # threshold definition (≥0.90 / ≤0.10), but the discrete grid avoids
        # spurious rank-100/n=1.0 asymmetry.
        pct = (ranks - 0.5) / n_valid
        rank_pcts.loc[sorted_group.index] = pct

    out["sue_rank_pct"] = rank_pcts
    return out


# ── Decile leg assignment ───────────────────────────────────────────────────
def assign_decile_legs(
    panel: pd.DataFrame,
    *,
    long_threshold:  float = DECILE_LONG_THRESHOLD,
    short_threshold: float = DECILE_SHORT_THRESHOLD,
    rank_col:        str   = "sue_rank_pct",
) -> pd.DataFrame:
    """Assign each firm-quarter a `leg` label: "long" / "short" / "flat".

    Per spec §2.4:
      - leg = "long"  iff sue_rank_pct ≥ long_threshold  (default 0.90)
      - leg = "short" iff sue_rank_pct ≤ short_threshold (default 0.10)
      - else leg = "flat"
      - NaN rank → leg = "excluded"

    Boundary INCLUSIVE per spec §N6: firms exactly at 10/90 are included
    in their respective legs (tie-break already deterministic via ticker).
    """
    if panel.empty:
        out = panel.copy()
        out["leg"] = pd.Series(dtype="object")
        return out

    if rank_col not in panel.columns:
        raise ValueError(f"assign_decile_legs: '{rank_col}' column required (run rank_within_quarter first)")
    if not (0.0 < short_threshold < long_threshold < 1.0):
        raise ValueError(
            f"assign_decile_legs: thresholds must satisfy 0 < short ({short_threshold}) "
            f"< long ({long_threshold}) < 1"
        )

    out = panel.copy()
    rank = out[rank_col]
    leg = pd.Series("flat", index=out.index, dtype="object")
    leg[rank >= long_threshold]  = "long"
    leg[rank <= short_threshold] = "short"
    leg[rank.isna()] = "excluded"
    out["leg"] = leg
    return out


# ── One-shot convenience wrapper ────────────────────────────────────────────
def build_sue_signal_panel(
    panel: pd.DataFrame,
    *,
    long_threshold:  float = DECILE_LONG_THRESHOLD,
    short_threshold: float = DECILE_SHORT_THRESHOLD,
) -> pd.DataFrame:
    """Pipeline: compute_sue → rank_within_quarter → assign_decile_legs.

    Convenience for Sprint 4 walk-forward. Returns the input panel augmented
    with `sue`, `sue_rank_pct`, `leg` columns.
    """
    p1 = compute_sue(panel)
    p2 = rank_within_quarter(p1)
    p3 = assign_decile_legs(
        p2,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
    )
    if not p3.empty:
        logger.info(
            "build_sue_signal_panel: %d firm-quarters → long=%d short=%d flat=%d excluded=%d",
            len(p3),
            int((p3["leg"] == "long").sum()),
            int((p3["leg"] == "short").sum()),
            int((p3["leg"] == "flat").sum()),
            int((p3["leg"] == "excluded").sum()),
        )
    return p3
