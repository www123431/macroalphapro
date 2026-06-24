"""engine.research.ablation.portfolio — L/S decile portfolio + sector
neutralization + capacity constraints + vol-targeting.

Per Phase A v3 rigor items #5, #6, #8:
  - Sector-neutralize via DGTW 1997 method: long/short legs balanced
    across GICS gsectors (top-decile within sector vs bottom-decile
    within sector)
  - Market-cap floor: drop micro-caps (mcap < $500M) to ensure
    deployability + reduce noise
  - ADV cap (simulated): assume 5% × ADV per position cap; here
    approximated by cap on max single-name weight (1/N_leg × 2.0)
    since we don't have intraday ADV — full ADV gate would require
    CRSP daily volume data join
  - Vol-target each leg to 10% ex-ante vol using realized σ_idio:
    weights → weights × (target_vol / portfolio_vol_estimate)

The L/S structure is FIXED across all weighting variants — only the
internal WEIGHT assignment within a leg differs. This ensures variants
are apples-to-apples (per the v1 self-audit lesson).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CACHE     = _REPO_ROOT / "data" / "cache"


# ── Constants ──────────────────────────────────────────────────────


DECILE_BOTTOM   = 0.10
DECILE_TOP      = 0.90
MIN_PER_DECILE  = 3
MIN_PER_MONTH   = 30
# market_cap_at_q in Compustat is in millions USD; floor = 500M
MCAP_FLOOR_USD  = 500.0
SECTOR_NEUTRAL_DEFAULT = True
ADV_CAP_FRACTION = 0.05    # max position size = 5% × ADV (not enforced; cap on weight)
MAX_SINGLE_WEIGHT = 0.10   # cap any single position at 10% of leg
TARGET_VOL_ANN   = 0.10    # target annualized vol per leg


# ── Decile assignment (sector-neutral) ─────────────────────────────


def assign_deciles_sector_neutral(
    panel: pd.DataFrame,
    signal_col: str,
    gics_map: Optional[pd.DataFrame] = None,
    sector_neutral: bool = SECTOR_NEUTRAL_DEFAULT,
) -> pd.DataFrame:
    """Assign each event to long/short/neutral leg per month.

    If sector_neutral=True:
      WITHIN each (month, gsector) bucket, take top-DECILE_TOP and
      bottom-DECILE_BOTTOM by signal. This ensures long and short legs
      have matched sector exposure (DGTW 1997).

    Else: simple month-level deciles.
    """
    df = panel.copy()
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)

    if sector_neutral and gics_map is not None and not gics_map.empty:
        df = df.merge(gics_map[["gvkey", "gsector"]], on="gvkey", how="left")
        df["gsector"] = df["gsector"].fillna("UNK")
        group_cols = ["month", "gsector"]
    else:
        df["gsector"] = "ALL"
        group_cols = ["month"]

    df["leg"] = "neutral"
    for keys, g in df.groupby(group_cols):
        if len(g) < 4:
            # Too few in sector — assign by simple median
            med = g[signal_col].median()
            mask_long = g[signal_col] > med
            df.loc[g.index[mask_long], "leg"] = "long"
            df.loc[g.index[~mask_long], "leg"] = "short"
            continue
        lo = g[signal_col].quantile(DECILE_BOTTOM)
        hi = g[signal_col].quantile(DECILE_TOP)
        df.loc[g.index[g[signal_col] >= hi], "leg"] = "long"
        df.loc[g.index[g[signal_col] <= lo], "leg"] = "short"

    return df[df["leg"] != "neutral"].copy()


# ── Capacity / ADV cap ────────────────────────────────────────────


def apply_mcap_floor(panel: pd.DataFrame, floor_usd: float = MCAP_FLOOR_USD) -> pd.DataFrame:
    """Drop events whose market_cap_at_q is below the floor.

    The Compustat panel has market_cap_at_q as the quarter-end mcap.
    NaN/missing rows are kept (no information to filter).
    """
    if "market_cap_at_q" not in panel.columns:
        return panel
    mask = panel["market_cap_at_q"].isna() | (panel["market_cap_at_q"] >= floor_usd)
    return panel[mask].copy()


def cap_single_weights(weights: pd.Series, cap: float = MAX_SINGLE_WEIGHT) -> pd.Series:
    """Iteratively cap any weight above `cap` and redistribute excess."""
    w = weights.copy()
    for _ in range(8):   # 8 iterations is more than enough for our N
        over = w[w > cap]
        if len(over) == 0:
            return w / w.sum() if w.sum() > 0 else w
        excess = (over - cap).sum()
        w[over.index] = cap
        under = w[w < cap]
        if under.sum() > 0:
            w[under.index] += excess * (under / under.sum())
    return w / w.sum() if w.sum() > 0 else w


# ── Vol targeting ──────────────────────────────────────────────────


def vol_target_leg(weights: pd.Series,
                   sigma_idio: pd.Series,
                   target_vol_ann: float = TARGET_VOL_ANN,
                   ) -> pd.Series:
    """Scale weights so the portfolio's ex-ante daily vol ≈ target_vol_ann/√252.

    Under zero-correlation assumption: σ_port² = Σ(w_i × σ_i)²
    Scale factor = (target_daily / sqrt(Σ(w_i × σ_i)²)).
    """
    target_daily = target_vol_ann / np.sqrt(252)
    common = weights.index.intersection(sigma_idio.index)
    if len(common) < 2:
        return weights
    w = weights.loc[common]
    s = sigma_idio.loc[common]
    port_vol = float(np.sqrt(((w * s) ** 2).sum()))
    if port_vol <= 0:
        return weights
    scale = target_daily / port_vol
    return (weights * scale).clip(lower=-1.0, upper=1.0)


# ── L/S monthly return builder ────────────────────────────────────


def build_ls_monthly_returns(
    panel: pd.DataFrame,
    signal_col: str,
    weighting_fn: Callable[[pd.DataFrame, str], pd.Series],
    gics_map: Optional[pd.DataFrame] = None,
    sector_neutral: bool = SECTOR_NEUTRAL_DEFAULT,
    mcap_floor: float = MCAP_FLOOR_USD,
    vol_target_each_leg: bool = True,
) -> dict:
    """Build the long/short portfolio's monthly net returns + diagnostics.

    Returns dict with:
      monthly_net:      pd.Series of monthly net returns
      mean_turnover:    average month-over-month name turnover
      long_count_avg:   mean # names per long leg
      short_count_avg:  mean # names per short leg
      n_months:         number of months in the series
    """
    panel = apply_mcap_floor(panel, floor_usd=mcap_floor)
    panel_ls = assign_deciles_sector_neutral(
        panel, signal_col, gics_map=gics_map, sector_neutral=sector_neutral,
    )

    monthly: dict = {}
    turnovers: list[float] = []
    long_counts: list[int] = []
    short_counts: list[int] = []
    prev_long, prev_short = set(), set()

    for month, g in panel_ls.groupby("month"):
        longs  = g[g["leg"] == "long"]
        shorts = g[g["leg"] == "short"]
        if len(longs) < MIN_PER_DECILE or len(shorts) < MIN_PER_DECILE:
            continue
        wL = weighting_fn(longs, signal_col)
        wS = weighting_fn(shorts, signal_col)

        if vol_target_each_leg:
            wL = vol_target_leg(wL, longs.set_index(longs.index)["sigma_idio"])
            wS = vol_target_leg(wS, shorts.set_index(shorts.index)["sigma_idio"])

        wL = cap_single_weights(wL)
        wS = cap_single_weights(wS)

        long_ret  = float((wL * longs["fwd_ret_log"]).sum())
        short_ret = float((wS * shorts["fwd_ret_log"]).sum())
        monthly[month.to_timestamp()] = long_ret - short_ret

        # Turnover (Jaccard-style: fraction of NEW names)
        cur_long  = set(longs["permno"])
        cur_short = set(shorts["permno"])
        if prev_long or prev_short:
            new_l = len(cur_long  - prev_long)
            new_s = len(cur_short - prev_short)
            denom = max(1, len(cur_long) + len(cur_short))
            turnovers.append((new_l + new_s) / denom)
        prev_long, prev_short = cur_long, cur_short
        long_counts.append(len(longs))
        short_counts.append(len(shorts))

    ser = pd.Series(monthly).sort_index() if monthly else pd.Series(dtype=float)
    return {
        "monthly_net":     ser,
        "mean_turnover":   float(np.mean(turnovers)) if turnovers else 1.0,
        "long_count_avg":  float(np.mean(long_counts)) if long_counts else 0,
        "short_count_avg": float(np.mean(short_counts)) if short_counts else 0,
        "n_months":        len(ser),
    }


# ── Transaction costs ─────────────────────────────────────────────


RT_EQ_BPS = 30.0


def apply_costs(monthly_ret: pd.Series, mean_turnover: float,
                rt_bps: float = RT_EQ_BPS,
                ) -> pd.Series:
    """Apply per-month transaction cost.

    Cost = RT bps / 10000 × turnover × 2 (both long + short legs)
    """
    cost_per_month = (rt_bps / 10000.0) * mean_turnover * 2.0
    return monthly_ret - cost_per_month
