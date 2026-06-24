"""composer.components — atomic component implementations.

Each module side-effect-registers one or more Component subclasses
via @register_component(role, key) decorators. Importing this package
triggers them all.

Roles → files:
  signals/      SIGNAL components, one per signal_type
  universes/    UNIVERSE components, one per (asset_class, subset)
  weightings/   WEIGHTING components, one per weighting scheme
  rebalances/   REBALANCE components, one per cadence
  risk_filters/ RISK_FILTER components, one per filter
"""

# Side-effect imports — order matters only for self-consistency checks
from engine.composer.components import universes      # noqa: F401
from engine.composer.components import signals        # noqa: F401
from engine.composer.components import weightings     # noqa: F401
from engine.composer.components import rebalances     # noqa: F401
from engine.composer.components import risk_filters   # noqa: F401
