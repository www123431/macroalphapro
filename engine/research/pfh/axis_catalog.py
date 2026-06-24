"""engine/research/pfh/axis_catalog.py — Axis component discovery.

Reads the feature_store axis YAMLs to learn what universes / signals /
weightings exist. PFH constrained mode uses this to enumerate ONLY
candidates that can actually materialize, rather than emitting
PLACEHOLDER refs that need human follow-up.

Also reads existing compose-specs to know what tuples have ALREADY
been tested — PFH-constrained mode excludes these so suggestions are
genuinely fresh.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
UNIVERSES_DIR     = REPO_ROOT / "data" / "feature_store" / "_universes"
SIGNAL_RECIPES_DIR = REPO_ROOT / "data" / "feature_store" / "_signal_recipes"
WEIGHTINGS_DIR    = REPO_ROOT / "data" / "feature_store" / "_weightings"
COMPOSE_SPECS_DIR = REPO_ROOT / "data" / "feature_store" / "_specs"


@dataclass
class AxisCatalog:
    """Snapshot of what axis components exist + which tuples are tested."""
    universes:    list[str]
    signals:      list[str]
    weightings:   list[str]
    tested_tuples: set[tuple[str, str, str]]  # (universe, signal, weighting)

    @property
    def n_possible(self) -> int:
        return len(self.universes) * len(self.signals) * len(self.weightings)

    @property
    def n_untested(self) -> int:
        return self.n_possible - len(self.tested_tuples)


def _list_yaml_names(dir_path: Path) -> list[str]:
    """Return sorted file stems for *.yaml files (excluding underscore-prefix)."""
    if not dir_path.is_dir():
        return []
    return sorted(
        p.stem for p in dir_path.glob("*.yaml")
        if not p.name.startswith("_")
    )


def _read_compose_spec_tuple(spec_path: Path) -> Optional[tuple[str, str, str]]:
    """Read a compose-spec YAML and return (universe, signal, weighting)
    refs, or None if this is a v1 function-wrapper spec (no compose: block)."""
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    compose = raw.get("compose")
    if not isinstance(compose, dict):
        return None

    def _ref(entry):
        if isinstance(entry, dict):
            return str(entry.get("ref") or "")
        return str(entry)

    u = _ref(compose.get("universe"))
    s = _ref(compose.get("signal"))
    w = _ref(compose.get("weighting"))
    if not (u and s and w):
        return None
    return (u, s, w)


def load_axis_catalog() -> AxisCatalog:
    """Snapshot all axis components + already-tested compose tuples."""
    universes  = _list_yaml_names(UNIVERSES_DIR)
    signals    = _list_yaml_names(SIGNAL_RECIPES_DIR)
    weightings = _list_yaml_names(WEIGHTINGS_DIR)

    tested: set[tuple[str, str, str]] = set()
    if COMPOSE_SPECS_DIR.is_dir():
        for p in COMPOSE_SPECS_DIR.glob("*.yaml"):
            if p.name.startswith("_"):
                continue
            t = _read_compose_spec_tuple(p)
            if t is not None:
                tested.add(t)

    return AxisCatalog(
        universes=universes,
        signals=signals,
        weightings=weightings,
        tested_tuples=tested,
    )


def enumerate_untested_tuples(
    catalog: Optional[AxisCatalog] = None,
) -> list[tuple[str, str, str]]:
    """Cartesian product (universes × signals × weightings) minus tested."""
    if catalog is None:
        catalog = load_axis_catalog()
    out: list[tuple[str, str, str]] = []
    for u in catalog.universes:
        for s in catalog.signals:
            for w in catalog.weightings:
                t = (u, s, w)
                if t not in catalog.tested_tuples:
                    out.append(t)
    return out
