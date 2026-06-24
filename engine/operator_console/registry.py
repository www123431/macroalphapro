"""Station registry — single source of truth for what stations exist.

Stations register themselves at import time. The API uses get_all()
to enumerate; UI launchpad uses the same call.

When a station is added (S1 PaperIngest, S4 FORWARD Dispatch, etc.):
    1. Implement subclass of PipelineStation in
       engine/operator_console/stations/sX_name.py
    2. Call `registry.register(SubclassName)` at the bottom of the
       module (manual registration; no __init_subclass__ magic)
    3. Import the module from
       engine/operator_console/stations/__init__.py so the import-time
       register() side effect fires when the package is loaded

## Capital-decision doctrine enforcement

`register()` lints the station's source module at import time per the
CLAUDE.md capital-decision doctrine:

  - `mutates_capital=False`: source must NOT contain writes to the
    deployed-config YAML files.
  - `mutates_capital=True`: source MUST reference `_proposals.jsonl`
    as positive evidence the station routes through /approvals
    (human-in-the-loop) instead of mutating capital directly.

Violations raise `CapitalDoctrineViolation` — loud, at import time,
not a runtime surprise after the station has already shipped.

A future polish item is to move registration into an explicit
init_stations() entry point so module imports stop having side
effects (today this makes unit tests order-sensitive — see Agent 2
code review, 2026-06-24).
"""
from __future__ import annotations

import inspect
import re

from engine.operator_console.pipeline_station import PipelineStation


_REGISTRY: dict[str, type[PipelineStation]] = {}


class CapitalDoctrineViolation(Exception):
    """Raised at register time when a station's source contradicts its
    declared mutates_capital intent. Bug; not user-recoverable."""


# Patterns that indicate a station is writing the live capital-allocation
# state directly instead of routing through /approvals. These match the
# files / module paths that downstream consumers treat as source-of-truth
# for "what is currently deployed."
#
# Add patterns here as new capital-state surfaces are introduced. Better
# to false-positive (forcing an explicit allowlist comment) than to miss
# a real violation.
_FORBIDDEN_CAPITAL_WRITE_PATTERNS = (
    re.compile(r"\bdeployed_registry\.(save|write|persist|update|set)_"),
    re.compile(r"open\([^)]*config[/\\]deployed_[^)]*['\"][^)]*['\"]w"),
    re.compile(r"open\([^)]*config[/\\]deployed_[^)]*,\s*['\"]w"),
    re.compile(r"\.write_text\(\s*[^)]*deployed_sleeves"),
)

# Stations that legitimately mutate capital must route through this
# proposal queue (consumed by /approvals UI for human review).
_REQUIRED_APPROVAL_ROUTING_PATTERN = re.compile(r"_proposals\.jsonl")


def _lint_capital_doctrine(station_cls: type[PipelineStation]) -> None:
    """Verify the station's source matches its declared mutates_capital
    intent. Raises CapitalDoctrineViolation on mismatch."""
    spec = station_cls.STATION_SPEC
    try:
        source = inspect.getsource(inspect.getmodule(station_cls))
    except (TypeError, OSError):
        # Built-in / dynamically generated class — can't lint; skip.
        # In practice all real stations come from .py files.
        return

    forbidden_hits = [p.pattern for p in _FORBIDDEN_CAPITAL_WRITE_PATTERNS
                      if p.search(source)]

    if spec.mutates_capital:
        if not _REQUIRED_APPROVAL_ROUTING_PATTERN.search(source):
            raise CapitalDoctrineViolation(
                f"Station '{spec.station_id}' declares mutates_capital=True "
                f"but its source does not reference '_proposals.jsonl'. "
                f"Per CLAUDE.md, capital-mutating stations MUST route through "
                f"/approvals (write to data/operator_console/*_proposals.jsonl), "
                f"not mutate deployed config directly.")
        # When mutates_capital=True, forbidden writes are still forbidden:
        # the doctrine is "write a proposal, let human apply it" — never
        # both.
        if forbidden_hits:
            raise CapitalDoctrineViolation(
                f"Station '{spec.station_id}' declares mutates_capital=True "
                f"AND its source matches forbidden direct-mutation patterns: "
                f"{forbidden_hits}. The doctrine is 'propose, don't apply'.")
    else:
        if forbidden_hits:
            raise CapitalDoctrineViolation(
                f"Station '{spec.station_id}' declares mutates_capital=False "
                f"but its source matches forbidden capital-mutation patterns: "
                f"{forbidden_hits}. Either route through /approvals (write "
                f"_proposals.jsonl) or set mutates_capital=True (which still "
                f"forbids direct writes).")


def register(station_cls: type[PipelineStation]) -> None:
    """Add a station class to the registry. Idempotent — re-registering
    the same station_id silently replaces (useful during hot-reload).

    Raises CapitalDoctrineViolation if the station's source violates the
    declared mutates_capital intent — see module docstring for the rule."""
    spec = station_cls.STATION_SPEC
    _lint_capital_doctrine(station_cls)
    _REGISTRY[spec.station_id] = station_cls


def get(station_id: str) -> type[PipelineStation] | None:
    """Lookup a station class by id."""
    return _REGISTRY.get(station_id)


def get_all() -> list[type[PipelineStation]]:
    """Return all registered station classes (no particular order)."""
    return list(_REGISTRY.values())


def all_specs() -> list[dict]:
    """Serialize all StationSpec instances as dicts for API surface.
    The frontend renders the launchpad from this list."""
    from dataclasses import asdict
    specs = []
    for cls in _REGISTRY.values():
        s = cls.STATION_SPEC
        d = asdict(s)
        # Convert set + enums to JSON-friendly values
        d["data_tier"] = s.data_tier.value
        d["requires_session_types"] = sorted(t.value for t in s.requires_session_types)
        specs.append(d)
    specs.sort(key=lambda d: d["station_id"])
    return specs
