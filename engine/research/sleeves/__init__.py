"""engine/research/sleeves/ — concrete Sleeve implementations registered
via @register_sleeve decorator.

Importing this package triggers registration of all in-tree sleeves
(side-effect of class-level decorators in each module). Downstream
callers should:

    import engine.research.sleeves  # noqa: F401  (triggers registration)
    from engine.research.sleeve_registry import get_sleeve
    sleeve = get_sleeve("post_earnings_drift_pit_sn")
"""
from __future__ import annotations

# Side-effect imports — order matters for deterministic registration
from engine.research.sleeves import post_earnings_drift_pit_sn  # noqa: F401
from engine.research.sleeves import post_earnings_drift_parent  # noqa: F401
from engine.research.sleeves import tail_hedge_put_spread_sleeve  # noqa: F401
