"""engine/feature_store/primitives.py — Signal computation primitives.

The atomic operations from which signal recipes are composed. Each
primitive is a pure pandas function: takes a wide-format panel
(date × asset) and parameters, returns a wide-format panel.

DESIGN RULE: every primitive must be:
  - PURE (no I/O, no global state, no random)
  - VECTORIZED (operates on the whole panel, not row-by-row)
  - NaN-AWARE (preserve NaN where input is NaN; do not silently fill)
  - SHAPE-PRESERVING for unary ops, SHAPE-COMBINING for binary ops

Adding a primitive: bump _SCHEMA_VERSION in the recipe schema, add the
function here, add a test. Never overload existing primitives — give
new behavior a new name. This keeps existing recipes stable.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bump when primitive semantics change in a backwards-incompatible way.
PRIMITIVES_SCHEMA_VERSION = 1


# ── Temporal primitives ──────────────────────────────────────────────


def rolling_return(panel: pd.DataFrame, *, months: int) -> pd.DataFrame:
    """Cumulative return over a rolling N-month window.

    Assumes monthly-frequency input. For daily input, use rolling_return_days.
    Returns NaN for rows where the full window isn't available.
    """
    if months < 1:
        raise ValueError(f"months must be >= 1, got {months}")
    # (1+r).rolling.product - 1
    return (1.0 + panel).rolling(months, min_periods=months).apply(
        lambda x: x.prod(), raw=True,
    ) - 1.0


def rolling_return_days(panel: pd.DataFrame, *, days: int) -> pd.DataFrame:
    """Cumulative return over rolling N-day window (for daily-frequency input)."""
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    return (1.0 + panel).rolling(days, min_periods=days).apply(
        lambda x: x.prod(), raw=True,
    ) - 1.0


def shift(panel: pd.DataFrame, *, periods: int) -> pd.DataFrame:
    """Shift by N periods (positive = into the past, equivalent to lag)."""
    return panel.shift(periods)


def skip(panel: pd.DataFrame, *, periods: int) -> pd.DataFrame:
    """Alias for shift — semantically: "skip the most recent N periods".

    Used in 12-1 momentum: rolling_return(12).skip(1) reads as
    "12-month return, skipping the most recent month"."""
    return shift(panel, periods=periods)


def diff(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Period-over-period absolute change."""
    return panel.diff(periods)


# ── Normalization primitives ─────────────────────────────────────────


def ts_zscore(panel: pd.DataFrame, *, window: int,
               min_window: int | None = None) -> pd.DataFrame:
    """Time-series z-score per asset over a rolling window."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    mw = min_window if min_window is not None else window
    mean = panel.rolling(window, min_periods=mw).mean()
    std = panel.rolling(window, min_periods=mw).std()
    return (panel - mean) / std


def xs_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: subtract row mean, divide by row std.

    NaN entries are skipped from the row statistics."""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1)
    # Use broadcasting; .sub/.div align by index
    return panel.sub(mean, axis=0).div(std, axis=0)


def xs_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in [0, 1]. NaN preserved."""
    return panel.rank(axis=1, pct=True, method="average")


def vol_scale(panel: pd.DataFrame, *, window: int, target_vol: float = 0.10,
               freq_per_year: int = 12, cap: float = 2.0) -> pd.DataFrame:
    """Inverse-vol position scaling per asset.

    multiplier = min(target_vol / realized_vol, cap)
    Returns the panel multiplied by per-asset multiplier (broadcast).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    realized = panel.rolling(window, min_periods=window).std() * (freq_per_year ** 0.5)
    mult = (target_vol / realized).clip(upper=cap)
    return panel * mult


def sign(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-cell sign (+1, 0, -1). NaN preserved."""
    return np.sign(panel)


# ── Primitive registry ──────────────────────────────────────────────


_REGISTRY: dict[str, Callable] = {
    "rolling_return":       rolling_return,
    "rolling_return_days":  rolling_return_days,
    "shift":                shift,
    "skip":                 skip,
    "diff":                 diff,
    "ts_zscore":            ts_zscore,
    "xs_zscore":            xs_zscore,
    "xs_rank":              xs_rank,
    "vol_scale":            vol_scale,
    "sign":                 sign,
}


def list_primitives() -> list[str]:
    return sorted(_REGISTRY.keys())


def apply_primitive(
    name: str, panel: pd.DataFrame, **kwargs,
) -> pd.DataFrame:
    """Dispatch by name with arg validation.

    Raises KeyError on unknown primitive (NOT silent skip — recipes
    that reference missing primitives should fail loud at materialize
    time)."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown primitive {name!r}; available: {list_primitives()}"
        )
    return _REGISTRY[name](panel, **kwargs)
