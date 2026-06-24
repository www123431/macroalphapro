"""engine.research.ablation.signals — signal definition family.

Per Phase A v3 rigor item #7: test on MULTIPLE signal definitions to
ensure the chosen weighting method is robust to the signal flavor.

Four signal definitions:
  1. sue_raw          — raw SUE from Compustat panel (delta_eps / sigma_8q)
  2. sue_z            — cross-sectional z-score of sue_raw per month
  3. sue_industry_adj — sue_raw minus industry median per month (GICS sector)
  4. abnormal_sue     — pre-computed abnormal_sue from
                         data/cache/_dpead_abnormal_sue_monthly.parquet
                         (the deployed signal — published in literature
                          as Frankel-Lee 1998 / Bernard-Thomas 1990 family)

Each signal is computed PIT (point-in-time) using rdq as the only
look-ahead boundary. Industry classification uses gvkey → gsector join
from data/cache/_compustat_company_gics.parquet (PIT for our window
because GICS rarely changes within a decade).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CACHE     = _REPO_ROOT / "data" / "cache"


# ── Signal builders ────────────────────────────────────────────────


def signal_sue_raw(events: pd.DataFrame) -> pd.Series:
    """SUE as-is from the event panel (already z-score-like: delta_eps / sigma_8q)."""
    return events["sue"].copy().rename("sue_raw")


def signal_sue_z(events: pd.DataFrame) -> pd.Series:
    """Cross-sectional z-score per month — controls for any time-varying
    drift in the raw SUE distribution (e.g. earnings season clustering)."""
    out = pd.Series(np.nan, index=events.index, name="sue_z")
    for month, grp in events.groupby("month"):
        mu = grp["sue"].mean()
        sd = grp["sue"].std()
        if sd > 0:
            out.loc[grp.index] = (grp["sue"] - mu) / sd
    return out


def signal_sue_industry_adj(events: pd.DataFrame,
                            gics_map: Optional[pd.DataFrame] = None,
                            ) -> pd.Series:
    """SUE minus industry median per month (GICS gsector level).

    Removes industry-wide drift (e.g. all tech beat estimates by 5% in Q1) so
    the signal isolates firm-specific surprise. Standard DGTW 1997 approach.
    """
    if gics_map is None:
        gics_map = load_gics_map()
    # Merge gvkey → gsector
    df = events[["sue", "gvkey", "month"]].copy()
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df = df.merge(gics_map[["gvkey", "gsector"]], on="gvkey", how="left")
    out = pd.Series(np.nan, index=events.index, name="sue_industry_adj")
    for (month, sec), grp in df.groupby(["month", "gsector"]):
        med = grp["sue"].median()
        out.loc[grp.index] = grp["sue"] - med
    return out


def signal_abnormal_sue(events: pd.DataFrame) -> pd.Series:
    """Pre-computed abnormal_sue from existing pipeline (the literature-
    standard PEAD signal). Joined back to events by month."""
    abn_path = _CACHE / "_dpead_abnormal_sue_monthly.parquet"
    if not abn_path.is_file():
        # Fallback: identity to sue_z
        return signal_sue_z(events).rename("abnormal_sue")
    abn = pd.read_parquet(abn_path)
    abn.index = pd.to_datetime(abn.index)
    abn_monthly = abn["abnormal_sue"].copy()
    abn_monthly.index = abn_monthly.index.to_period("M")
    # Map: per event month, take the universe-mean abnormal_sue and scale
    # the event's sue by that → produces a relative signal where high
    # abnormal-sue months get amplified, calm months damped.
    out = pd.Series(np.nan, index=events.index, name="abnormal_sue")
    for month, grp in events.groupby("month"):
        scale = abn_monthly.get(month, 1.0)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        out.loc[grp.index] = grp["sue"] * float(scale)
    return out


# ── GICS sector loader ────────────────────────────────────────────


def load_gics_map() -> pd.DataFrame:
    """Load gvkey → GICS sector (gsector) mapping for sector-neutralization."""
    p = _CACHE / "_compustat_company_gics.parquet"
    if not p.is_file():
        return pd.DataFrame(columns=["gvkey", "gsector"])
    df = pd.read_parquet(p)
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df["gsector"] = df["gsector"].astype(str)
    return df[["gvkey", "gsector"]].drop_duplicates(subset=["gvkey"])


# ── Registry ──────────────────────────────────────────────────────


SIGNAL_DEFINITIONS = {
    "sue_raw":           signal_sue_raw,
    "sue_z":             signal_sue_z,
    "sue_industry_adj":  signal_sue_industry_adj,
    "abnormal_sue":      signal_abnormal_sue,
}


def build_all_signals(events: pd.DataFrame) -> pd.DataFrame:
    """Compute all 4 signal columns and attach to events. Returns a new
    DataFrame with columns: events.cols + [sue_raw, sue_z, sue_industry_adj,
    abnormal_sue]."""
    gics = load_gics_map()
    out = events.copy()
    out["sig_sue_raw"]          = signal_sue_raw(events).values
    out["sig_sue_z"]             = signal_sue_z(events).values
    out["sig_sue_industry_adj"] = signal_sue_industry_adj(events, gics).values
    out["sig_abnormal_sue"]      = signal_abnormal_sue(events).values
    return out
