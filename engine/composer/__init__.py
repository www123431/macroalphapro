"""engine.composer — spec → atomic components → returns series.

The architecture insight (user 2026-06-05): the project's CARRY
sub-builders showed the wrong shape. N family × M subtype = N*M
hand-written builders, each duplicating 80% of the construction.

The right shape (institutional pattern from AQR / Two Sigma /
RenTech):

  HypothesisSpec
       ↓
   Composer (single ~300-line function, NEVER hand-edited per family)
       ↓
   atomic components (~30 total in scope, each ~50 lines):
       - SignalPanel:  produces signal values per (asset, date)
       - Universe:     produces asset-set membership per date
       - Weighting:    produces position weights from signals
       - Rebalance:    produces rebalance dates
       - Filter:       produces overlay multipliers (vol-target, TC filter)
       ↓
   returns: pd.Series
       ↓
   cache key = spec_hash (LdP §2 reproducibility)

Each component has:
  - A typed contract (subclass + abstract methods)
  - A registration key (string the spec maps to)
  - Coverage metadata (which spec field values it accepts)
  - Self-documenting docstring (what it builds, with what assumptions)

The composer reads the spec, looks up the right component per role
(signal / universe / weighting / rebalance / filter), passes them
the spec, gets atomic outputs, assembles, returns Series.

NO subtype enum. NO if/elif chain on family. Pure spec → composer →
components.
"""
from engine.composer.contract import (
    Component, ComponentRole, ComponentResult, COMPONENT_REGISTRY,
    register_component, get_component, list_components,
    iter_role_components, is_spec_covered, coverage_summary,
    ComponentNotFound,
)
from engine.composer.composer import compose, cache_path, cached_for

__all__ = [
    "Component",
    "ComponentRole",
    "ComponentResult",
    "COMPONENT_REGISTRY",
    "register_component",
    "get_component",
    "list_components",
    "iter_role_components",
    "is_spec_covered",
    "coverage_summary",
]
