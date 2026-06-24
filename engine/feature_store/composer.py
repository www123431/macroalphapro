"""engine/feature_store/composer.py — 4-axis factor composition.

A compose-spec describes a factor as a tuple:
    (universe, signal, weighting, rebalance)

Each axis is independently defined in its own YAML directory:
    data/feature_store/_universes/<name>.yaml
    data/feature_store/_signal_recipes/<name>.yaml
    data/feature_store/_weightings/<name>.yaml
    (rebalance is inline in the compose-spec: monthly / weekly / daily)

This module:
  1. Loads + validates each axis component
  2. Runs the signal recipe pipeline against the universe panel
  3. Applies the weighting scheme to convert signal → returns
  4. Returns a pd.Series matching the function-wrapper spec output shape

The materializer auto-routes between v1 function-wrapper specs and v2
compose-specs based on YAML structure (compose-spec has a `compose:`
block instead of `materialize:`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from engine.feature_store.primitives import apply_primitive
from engine.feature_store.registry import REPO_ROOT, _safe_relpath

logger = logging.getLogger(__name__)

UNIVERSES_DIR    = REPO_ROOT / "data" / "feature_store" / "_universes"
SIGNAL_RECIPES_DIR = REPO_ROOT / "data" / "feature_store" / "_signal_recipes"
WEIGHTINGS_DIR    = REPO_ROOT / "data" / "feature_store" / "_weightings"


# ── Axis definitions ──────────────────────────────────────────────────


@dataclass
class Universe:
    """One asset universe + the input panel that defines it.

    The 'input' is a path to a parquet that gives a wide-format panel
    (date × asset returns). The composer feeds this panel into the
    signal recipe.
    """
    name:        str
    description: str
    input_path:  str            # data/cache/foo.parquet
    input_kind:  str            # "wide_returns_monthly" | "wide_returns_daily" |
                                # "long_panel_pead" (custom shapes)
    date_column: Optional[str] = None  # for long_panel inputs
    value_column: Optional[str] = None  # for long_panel inputs
    asset_column: Optional[str] = None
    universe_filter: dict = field(default_factory=dict)  # e.g. {"top_n_by_mcap": 1500}


@dataclass
class SignalRecipe:
    """A declarative pipeline of primitive operations.

    Each step is {primitive: name, args: {...}}. Steps are applied
    sequentially; the output of step N is the input to step N+1.
    """
    name:        str
    description: str
    steps:       list[dict]    # [{"primitive": "rolling_return", "args": {"months": 12}}, ...]


@dataclass
class Weighting:
    """Signal → positions scheme.

    kind ∈ {
      "decile_long_short" — args: q (quantile)
      "rank_weighted"     — args: (none)
      "sign_then_vol_target" — args: target_vol, vol_window, freq_per_year
    }
    """
    name:        str
    description: str
    kind:        str
    args:        dict = field(default_factory=dict)


@dataclass
class ComposeSpec:
    """Top-level compose-spec — the 4 axes glued together."""
    spec_id:           str
    version:           int
    description:       str
    universe:          Universe
    signal:            SignalRecipe
    weighting:         Weighting
    rebalance:         str               # "monthly" | "daily" — output frequency
    output_kind:       str               # "monthly_returns" | "daily_returns"
    output_start:      str
    output_end_min:    str
    output_n_rows_range: list[int]
    sanity:            dict
    inputs:            list[str]
    source_module_files: list[str]
    mechanism_library_id: Optional[str] = None
    audit:             dict = field(default_factory=dict)
    spec_path:         Optional[Path] = None


# ── Loaders ──────────────────────────────────────────────────────────


def _load_yaml(p: Path) -> dict:
    if not p.is_file():
        raise FileNotFoundError(f"axis component missing: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_universe(name: str, *, base_dir: Path = UNIVERSES_DIR) -> Universe:
    raw = _load_yaml(base_dir / f"{name}.yaml")
    required = {"name", "description", "input_path", "input_kind"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"universe {name}: missing {sorted(missing)}")
    return Universe(
        name=raw["name"],
        description=raw["description"],
        input_path=raw["input_path"],
        input_kind=raw["input_kind"],
        date_column=raw.get("date_column"),
        value_column=raw.get("value_column"),
        asset_column=raw.get("asset_column"),
        universe_filter=dict(raw.get("universe_filter") or {}),
    )


def load_signal_recipe(name: str, *, base_dir: Path = SIGNAL_RECIPES_DIR) -> SignalRecipe:
    raw = _load_yaml(base_dir / f"{name}.yaml")
    required = {"name", "description", "steps"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"signal recipe {name}: missing {sorted(missing)}")
    steps = raw["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"signal recipe {name}: steps must be a non-empty list")
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "primitive" not in step:
            raise ValueError(
                f"signal recipe {name}: step {i} missing 'primitive' key"
            )
    return SignalRecipe(
        name=raw["name"],
        description=raw["description"],
        steps=steps,
    )


def load_weighting(name: str, *, base_dir: Path = WEIGHTINGS_DIR) -> Weighting:
    raw = _load_yaml(base_dir / f"{name}.yaml")
    required = {"name", "description", "kind"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"weighting {name}: missing {sorted(missing)}")
    valid_kinds = {"decile_long_short", "rank_weighted", "sign_then_vol_target"}
    if raw["kind"] not in valid_kinds:
        raise ValueError(
            f"weighting {name}: kind must be one of {sorted(valid_kinds)}; "
            f"got {raw['kind']!r}"
        )
    return Weighting(
        name=raw["name"],
        description=raw["description"],
        kind=raw["kind"],
        args=dict(raw.get("args") or {}),
    )


def load_compose_spec(spec_path: Path | str) -> ComposeSpec:
    p = Path(spec_path) if not isinstance(spec_path, Path) else spec_path
    if not p.is_file():
        raise FileNotFoundError(f"compose spec not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    required = {"spec_id", "version", "description", "compose",
                "output", "inputs", "source_module_files"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(
            f"compose spec {p.name}: missing top-level {sorted(missing)}"
        )

    compose = raw["compose"]
    for k in ("universe", "signal", "weighting", "rebalance"):
        if k not in compose:
            raise ValueError(f"compose spec {p.name}: compose.{k} required")

    universe   = load_universe(compose["universe"]["ref"]
                                  if isinstance(compose["universe"], dict)
                                  else compose["universe"])
    signal     = load_signal_recipe(compose["signal"]["ref"]
                                       if isinstance(compose["signal"], dict)
                                       else compose["signal"])
    weighting  = load_weighting(compose["weighting"]["ref"]
                                    if isinstance(compose["weighting"], dict)
                                    else compose["weighting"])
    # Allow weighting args override at compose-spec level
    if isinstance(compose["weighting"], dict) and "args" in compose["weighting"]:
        weighting.args.update(compose["weighting"]["args"])

    rebalance = compose["rebalance"]
    if isinstance(rebalance, dict):
        rebalance = rebalance.get("freq", "monthly")

    out = raw["output"]
    return ComposeSpec(
        spec_id=str(raw["spec_id"]),
        version=int(raw["version"]),
        description=str(raw["description"]),
        universe=universe,
        signal=signal,
        weighting=weighting,
        rebalance=rebalance,
        output_kind=str(out["kind"]),
        output_start=str(out["expected_date_range"]["start"]),
        output_end_min=str(out["expected_date_range"]["end_min"]),
        output_n_rows_range=list(out["expected_shape"]["n_rows"]),
        sanity=dict(out["sanity"]),
        inputs=[str(i["cache_path"]) for i in raw["inputs"]],
        source_module_files=list(raw["source_module_files"]),
        mechanism_library_id=raw.get("mechanism_library_id"),
        audit=dict(raw.get("audit") or {}),
        spec_path=p,
    )


# ── Universe loader: parquet → wide panel ────────────────────────────


def load_universe_panel(universe: Universe) -> pd.DataFrame:
    """Read the universe's input parquet and return a wide-format
    DataFrame (date × asset returns).

    Handles two input kinds:
      - wide_returns_monthly / wide_returns_daily: parquet IS already wide
      - long_panel_returns: parquet is (date, asset, value) long-format;
        pivot to wide
    """
    p = REPO_ROOT / universe.input_path
    if not p.is_file():
        raise FileNotFoundError(
            f"universe {universe.name}: input parquet not found at {p}"
        )

    df = pd.read_parquet(p)

    if universe.input_kind in ("wide_returns_monthly", "wide_returns_daily"):
        # Already wide. Expect DatetimeIndex.
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try parsing first column as index
            if df.columns[0] in ("date", "Date", "DATE"):
                df = df.set_index(df.columns[0])
            df.index = pd.to_datetime(df.index)
        return df

    if universe.input_kind == "long_panel_returns":
        dc = universe.date_column  or "date"
        ac = universe.asset_column or "asset"
        vc = universe.value_column or "ret"
        for col in (dc, ac, vc):
            if col not in df.columns:
                raise ValueError(
                    f"universe {universe.name}: long_panel_returns expected "
                    f"column {col!r} in parquet"
                )
        df[dc] = pd.to_datetime(df[dc])
        # Optional top-N filter by some grouping column
        topn = universe.universe_filter.get("top_n_by_value_column")
        if topn:
            grouper = universe.universe_filter.get("top_n_within", dc)
            # Keep top-N assets per period by value (e.g. top 1500 by market cap)
            keep = (df.groupby(grouper)
                       .apply(lambda g: g.nlargest(int(topn), vc))
                       .reset_index(drop=True))
            df = keep
        wide = df.pivot_table(index=dc, columns=ac, values=vc, aggfunc="first")
        return wide

    raise ValueError(
        f"universe {universe.name}: unknown input_kind {universe.input_kind!r}"
    )


# ── Signal pipeline executor ─────────────────────────────────────────


def run_signal_recipe(panel: pd.DataFrame, recipe: SignalRecipe) -> pd.DataFrame:
    """Execute the recipe's steps sequentially.

    Returns a panel of the SAME shape as input, holding the SIGNAL
    (not yet weighted). Downstream weighting layer converts to positions.
    """
    out = panel
    for i, step in enumerate(recipe.steps):
        prim = step["primitive"]
        args = step.get("args") or {}
        try:
            out = apply_primitive(prim, out, **args)
        except Exception as exc:
            raise RuntimeError(
                f"signal recipe {recipe.name}: step {i} ({prim}) failed: {exc}"
            )
    return out


# ── Weighting: signal panel → return series ──────────────────────────


def apply_weighting(
    signal_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    weighting: Weighting,
) -> pd.Series:
    """Convert signal + returns into a long-short return time series.

    The return_panel is the SAME universe panel used as input to the
    signal — needed because positions taken at t earn returns at t+1
    (handled here via lag-1 alignment).
    """
    if weighting.kind == "decile_long_short":
        q = float(weighting.args.get("q", 0.1))
        if not (0 < q < 0.5):
            raise ValueError(f"decile_long_short: q must be in (0, 0.5), got {q}")
        # Cross-sectional rank of signal at each date
        ranks = signal_panel.rank(axis=1, pct=True)
        longs  = (ranks >= 1.0 - q).astype(float)
        shorts = (ranks <= q).astype(float)
        # Normalize within each leg
        long_w  = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0)
        short_w = shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0)
        # Positions formed at t, return realized at t+1 (lag-1 alignment)
        pos = long_w - short_w
        pos_lag = pos.shift(1)
        return (pos_lag * return_panel).sum(axis=1)

    if weighting.kind == "rank_weighted":
        # Centered rank in [-1, +1], normalized to unit gross
        ranks = signal_panel.rank(axis=1, pct=True) * 2 - 1
        gross = ranks.abs().sum(axis=1).replace(0, np.nan)
        w = ranks.div(gross, axis=0)
        return (w.shift(1) * return_panel).sum(axis=1)

    if weighting.kind == "sign_then_vol_target":
        # Position = sign(signal) * min(target_vol / realized_vol, cap)
        # Used for TSMOM-style time-series momentum
        target_vol = float(weighting.args.get("target_vol", 0.10))
        vol_window = int(weighting.args.get("vol_window", 36))
        cap        = float(weighting.args.get("cap", 2.0))
        freq_per_year = int(weighting.args.get("freq_per_year", 12))
        pos = np.sign(signal_panel)
        realized = return_panel.rolling(vol_window, min_periods=vol_window).std() \
                                  * (freq_per_year ** 0.5)
        mult = (target_vol / realized).clip(upper=cap)
        sized = pos * mult
        # Average across instruments (equal-weight TSMOM combine)
        # Lag positions to align with next-period returns
        return (sized.shift(1) * return_panel).mean(axis=1)

    raise ValueError(f"unknown weighting kind {weighting.kind!r}")


# ── Top-level composer ───────────────────────────────────────────────


def compose_factor(spec: ComposeSpec) -> pd.Series:
    """Execute a compose-spec end-to-end → return series.

    Steps:
      1. Load universe panel (wide format)
      2. Run signal recipe on universe panel → signal panel
      3. Apply weighting using signal + universe panel → return series
      4. Resample to rebalance frequency if needed
    """
    universe_panel = load_universe_panel(spec.universe)
    signal_panel = run_signal_recipe(universe_panel, spec.signal)
    returns = apply_weighting(signal_panel, universe_panel, spec.weighting)

    # Frequency handling — for now, output frequency matches input
    # universe frequency. Resample only if explicitly different.
    if spec.rebalance == "monthly" and spec.universe.input_kind.endswith("_daily"):
        # Daily input, monthly output — compound to month-end
        returns = (1 + returns).resample("ME").prod() - 1

    returns.name = spec.spec_id
    return returns
