"""
engine/path_c/dhs_combined_signal.py — Path D DHS COMBINED signal (PEAD + FIN).

Pre-registration: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62) §2.5

Combines per-firm-quarter SUE (PEAD time-series) + FIN composite into a
single rank-average score, then assigns decile legs.

Spec §2.5 formula:
  1. r_PEAD_iq = rank_within_quarter(SUE_iq)            # high SUE = bullish/long
  2. r_FIN_iq  = rank_within_quarter(fin_for_long_rank) # low FIN  = bullish/long (already negated)
  3. COMBINED_iq = (r_PEAD_iq + r_FIN_iq) / 2
  4. Excluded if EITHER component excluded (intersection)
  5. Decile rank on COMBINED → leg

0 LLM (spec invariant). Deterministic.
"""
from __future__ import annotations

import logging

import pandas as pd

from engine.path_c.sue_signal import (
    rank_within_quarter,
    assign_decile_legs,
)
from engine.path_c.fin_signal import (
    compute_fin_composite,
    DECILE_LONG_THRESHOLD,
    DECILE_SHORT_THRESHOLD,
)

logger = logging.getLogger(__name__)


def build_combined_panel(
    pead_panel: pd.DataFrame,
    fin_panel:  pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join PEAD-TS panel + FIN panel on (gvkey, fiscal_yearq).

    Output rows = firm-quarters where BOTH SUE and FIN-composite are computable.
    Carries: permno, ticker, gvkey, fiscal_yearq, rdq, sue, nsi, acc_scaled,
             market_cap_at_q (from PEAD-TS panel for universe-filter consistency).
    """
    if pead_panel.empty or fin_panel.empty:
        return pd.DataFrame()

    # Inner-join: both must exist
    join_keys = ["gvkey", "fiscal_yearq"]
    pead_cols = ["permno", "ticker", "gvkey", "fiscal_yearq", "rdq",
                 "sue", "market_cap_at_q"]
    fin_cols  = ["gvkey", "fiscal_yearq", "nsi", "acc_scaled"]

    pead_subset = pead_panel[[c for c in pead_cols if c in pead_panel.columns]].copy()
    fin_subset  = fin_panel[[c for c in fin_cols if c in fin_panel.columns]].copy()

    # Cast gvkey to string in both panels to avoid int64-vs-object merge mismatch
    # (CLI runner may pre-convert fin_panel.gvkey to str for universe filtering)
    pead_subset["gvkey"] = pead_subset["gvkey"].astype(str)
    fin_subset["gvkey"]  = fin_subset["gvkey"].astype(str)

    # Defensive dedupe (Compustat fundq can have restated rows for same gvkey-fiscal_yearq;
    # spec assumes one_to_one but real data has rare violations — keep first row deterministically).
    pead_subset = pead_subset.sort_values(join_keys + ["rdq"]).drop_duplicates(
        subset=join_keys, keep="first"
    )
    fin_subset = fin_subset.sort_values(join_keys).drop_duplicates(
        subset=join_keys, keep="first"
    )

    joined = pead_subset.merge(
        fin_subset,
        on=join_keys,
        how="inner",
    )
    return joined


def assign_combined_decile_legs(
    pead_panel: pd.DataFrame,
    fin_panel:  pd.DataFrame,
    *,
    long_threshold:  float = DECILE_LONG_THRESHOLD,
    short_threshold: float = DECILE_SHORT_THRESHOLD,
) -> pd.DataFrame:
    """End-to-end COMBINED signal: PEAD rank + FIN rank average → decile leg.

    Pipeline (spec §2.5):
      1. Inner-join PEAD + FIN panels
      2. Compute FIN composite (z_nsi + z_acc + fin + fin_for_long_rank)
      3. Cross-section rank SUE within quarter → r_pead
      4. Cross-section rank fin_for_long_rank within quarter → r_fin
      5. COMBINED = (r_pead + r_fin) / 2
      6. Cross-section rank COMBINED → combined_rank_pct
      7. assign_decile_legs at thresholds (long/short/flat)
    """
    joined = build_combined_panel(pead_panel, fin_panel)
    if joined.empty:
        # Empty output schema
        cols = ["permno", "ticker", "gvkey", "fiscal_yearq", "rdq",
                "sue", "nsi", "acc_scaled", "z_nsi", "z_acc", "fin",
                "fin_for_long_rank", "r_pead", "r_fin", "combined",
                "combined_rank_pct", "leg", "market_cap_at_q"]
        out = pd.DataFrame({c: pd.Series(dtype=float if c not in ("leg", "ticker", "fiscal_yearq")
                                          else object)
                            for c in cols})
        return out

    # FIN composite + fin_for_long_rank
    joined = compute_fin_composite(joined)

    # Rank SUE within quarter
    pead_ranked = rank_within_quarter(
        joined,
        sue_col="sue",
        tie_break_col="ticker",
    ).rename(columns={"sue_rank_pct": "r_pead"})

    # Rank fin_for_long_rank within quarter
    both_ranked = rank_within_quarter(
        pead_ranked,
        sue_col="fin_for_long_rank",
        tie_break_col="ticker",
    ).rename(columns={"sue_rank_pct": "r_fin"})

    # Composite rank average
    both_ranked["combined"] = (both_ranked["r_pead"] + both_ranked["r_fin"]) / 2.0

    # Rank COMBINED within quarter
    final = rank_within_quarter(
        both_ranked,
        sue_col="combined",
        tie_break_col="ticker",
    ).rename(columns={"sue_rank_pct": "combined_rank_pct"})

    # Decile leg
    final = assign_decile_legs(
        final,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        rank_col="combined_rank_pct",
    )
    return final
