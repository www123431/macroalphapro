"""
engine/path_c/k1_universe.py — Path K1 Size-Expanded B++ universe definition.

Pre-registration: docs/spec_path_k1_size_expanded_b_plus_v1.md (id=61) §2.1

K1 Universe = Tier-1 (33 sector/asset/region ETFs from B++ original) +
              10 size/style ETFs (verified full 10y 2014-2023 coverage)
= 43 ETFs total

All ticker → sector_label mapping locked at register time. Sector labels
arbitrary (just unique strings); used by B++ pipeline as ETF identifiers.
"""
from __future__ import annotations


# 10 size/style ETFs locked per spec §2.1 (verified 2026-05-12 inception coverage)
K1_SIZE_STYLE_ETFS: dict[str, str] = {
    # Russell 2000 small-cap family (3 ETFs)
    "size_russell2000_blend":   "IWM",
    "size_russell2000_growth":  "IWO",
    "size_russell2000_value":   "IWN",
    # Russell 1000 large-cap broader + style tilts (3 ETFs)
    "size_russell1000_blend":   "IWB",
    "size_russell1000_value":   "IWD",
    "size_russell1000_growth":  "IWF",
    # Mid-cap (1 ETF)
    "size_sp_midcap":           "IJH",
    # Style factor (large-cap momentum)
    "style_us_momentum":        "MTUM",
    # Vanguard size/value alternates (2 ETFs — capacity diversifier)
    "size_vanguard_smallvalue": "VBR",
    "size_vanguard_largevalue": "VTV",
}


def get_k1_universe() -> dict[str, str]:
    """Returns K1 universe = Tier-1 (33 ETFs) + 10 size/style ETFs = 43 ETFs.

    Tier-1 fetched via engine.b_plus_search.get_universe_tier(1) — single source
    of truth for B++ Tier-1 list. Adds 10 size/style ETFs per spec §2.1.

    Returns: dict {sector_label: ticker} suitable for B++ run_single_strategy_weekly.
    """
    from engine.b_plus_search import get_universe_tier

    tier1 = get_universe_tier(1)
    expanded = {**tier1, **K1_SIZE_STYLE_ETFS}

    # Sanity: no ticker collisions (Tier-1 shouldn't already have these)
    tier1_tickers = set(tier1.values())
    k1_added_tickers = set(K1_SIZE_STYLE_ETFS.values())
    overlap = tier1_tickers & k1_added_tickers
    if overlap:
        raise ValueError(
            f"k1_universe: ticker overlap with Tier-1 not allowed. Overlap: {overlap}"
        )

    return expanded
