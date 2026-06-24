"""engine/validation/capacity.py — D_PEAD family book capacity study.

Spec id=76 flagged "small-cap capacity ceiling" as a key pre-deployment
unknown: a real fund must know how much AUM the strategy holds before market
impact erodes the alpha. This quantifies it.

Method (transparent, order-of-magnitude):
  - holdings = top/bottom SUE-decile names each quarter (the D_PEAD long/short
    book), from the 2014-2023 panel + their market_cap_at_q.
  - ADV proxy = 0.5%/day of market cap (WRDS volume pull was blocked by a
    transient connectivity issue; the precise dollar-volume version is a
    refinement — the proxy is adequate for an order-of-magnitude ceiling).
  - market impact (square-root / Almgren): one-way cost ≈ C·σ·sqrt(position/ADV),
    C=0.4, σ_daily=2.5%; annual drag = 2·oneway·turnover (turnover≈5x/yr).
  - capacity ceiling ≈ AUM where annual impact erodes the ~10%/yr gross alpha.

Result (2026-05-20): the DEPLOYED top-1500 book holds median-$5.5B names, so it
scales to a comfortable ~$1-2B and a ~$5-10B ceiling (net alpha still positive
but participation heavy). The A.2 small-cap-tilted (alpha-dense) version is
capacity-limited to ~$0.5-1B. This is the explicit alpha-density vs capacity
trade-off — institutional mid-tier scale, not a toy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"

ADV_TURNOVER_PROXY = 0.005   # daily ADV ≈ 0.5% of market cap
SIGMA_DAILY = 0.025
IMPACT_C = 0.4
ANNUAL_TURNOVER = 5.0
GROSS_ALPHA = 0.10


def capacity_curve(small_cap_tilt: bool = False) -> pd.DataFrame:
    """AUM → (participation, annual impact, net alpha) for the D_PEAD book.
    small_cap_tilt restricts to the bottom market-cap tertile (A.2 alpha-dense)."""
    panel = pd.read_parquet(_PANEL).dropna(subset=["sue", "market_cap_at_q"])
    panel = panel[panel["market_cap_at_q"] > 0]
    legs = []
    for _, g in panel.groupby("fiscal_yearq"):
        if small_cap_tilt:
            g = g[g["market_cap_at_q"] <= g["market_cap_at_q"].quantile(1 / 3)]
        if len(g) < 50:
            continue
        hi = g[g["sue"] >= g["sue"].quantile(0.9)]
        lo = g[g["sue"] <= g["sue"].quantile(0.1)]
        legs.append(pd.concat([hi, lo])["market_cap_at_q"])
    m = pd.concat(legs)
    n_pos = int(np.median([len(x) for x in legs])) * 2
    adv = m * 1e6 * ADV_TURNOVER_PROXY               # $ ADV per name (mcap in $M)

    rows = []
    for aum in [1e8, 5e8, 1e9, 5e9, 1e10, 5e10]:
        pos = aum / n_pos
        part = pos / adv
        oneway = IMPACT_C * SIGMA_DAILY * np.sqrt(part)
        ann_cost = 2 * oneway * ANNUAL_TURNOVER
        rows.append({
            "aum_bn": aum / 1e9,
            "pos_per_name_m": pos / 1e6,
            "median_participation_pct": float(part.median() * 100),
            "ann_impact_median_pct": float(np.median(ann_cost) * 100),
            "ann_impact_p90_pct": float(np.percentile(ann_cost, 90) * 100),
            "net_alpha_pct": float((GROSS_ALPHA - np.median(ann_cost)) * 100),
        })
    out = pd.DataFrame(rows)
    out.attrs["n_positions"] = n_pos
    out.attrs["adv_median_m"] = float(adv.median() / 1e6)
    return out
