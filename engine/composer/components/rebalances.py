"""composer.components.rebalances — REBALANCE atomic components.

Each component produces a DatetimeIndex of rebalance dates. Composer
uses this to forward-fill positions between rebalances.
"""
from __future__ import annotations

import pandas as pd

from engine.composer.contract import (
    Component, ComponentRole, ComponentResult, register_component,
)


@register_component(ComponentRole.REBALANCE, "MONTHLY")
class RebalanceMonthly(Component):
    """Month-end rebalance. The canonical cadence for XS factor work."""
    description = "Month-end rebalance"

    def build(self, spec, context: dict) -> ComponentResult:
        weights = context.get("weights")
        if weights is None:
            raise ValueError("REBALANCE needs weights in context")
        idx = weights.data.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx)
            except Exception:
                raise ValueError("REBALANCE got non-datetime weights index")
        # The weights panel is already at monthly granularity in our
        # carry/etc panels, so all dates are rebalance dates.
        return ComponentResult(
            data=idx,
            metadata={
                "rebalance":     "MONTHLY",
                "n_dates":       len(idx),
                "first":         str(idx.min())[:10],
                "last":          str(idx.max())[:10],
            },
        )


@register_component(ComponentRole.REBALANCE, "WEEKLY")
class RebalanceWeekly(Component):
    """Weekly rebalance (typically Friday close). Coarser than the carry
    panels we currently have — issues a warning if the underlying data
    is monthly."""
    description = "Friday-weekly rebalance"

    def build(self, spec, context: dict) -> ComponentResult:
        weights = context.get("weights")
        if weights is None:
            raise ValueError("REBALANCE needs weights in context")
        idx = weights.data.index
        if isinstance(idx, pd.DatetimeIndex) and idx.freq is not None and "M" in str(idx.freq):
            # Monthly data being asked for weekly rebalance — warn but proceed
            return ComponentResult(
                data=idx,
                metadata={
                    "rebalance":  "WEEKLY",
                    "warning":    "underlying weights panel is monthly; weekly cadence approximated",
                    "n_dates":    len(idx),
                },
            )
        return ComponentResult(
            data=idx,
            metadata={"rebalance": "WEEKLY", "n_dates": len(idx)},
        )


@register_component(ComponentRole.REBALANCE, "QUARTERLY")
class RebalanceQuarterly(Component):
    """Quarter-end rebalance: select Mar/Jun/Sep/Dec month-ends."""
    description = "Quarter-end rebalance (Mar/Jun/Sep/Dec)"

    def build(self, spec, context: dict) -> ComponentResult:
        weights = context.get("weights")
        if weights is None:
            raise ValueError("REBALANCE needs weights in context")
        idx = weights.data.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        q_idx = idx[idx.month.isin([3, 6, 9, 12])]
        return ComponentResult(
            data=q_idx,
            metadata={"rebalance": "QUARTERLY", "n_dates": len(q_idx)},
        )
