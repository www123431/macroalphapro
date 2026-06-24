"""engine.capacity.schema — typed family-keyed capacity estimate.

Mirror of engine.decay_forecast — same axis-keyed pre-flight pattern.
For DETAILED capacity analysis at specific AUM levels, use
engine.portfolio.capacity_simulator (Pastor-Stambaugh / Berk-Green
framework with ADV-based TC model).
"""
from __future__ import annotations

import dataclasses as _dc
from enum import Enum
from typing import Any


class CapacityClass(str, Enum):
    """Coarse capacity band based on estimated $ capacity at 50%-Sharpe haircut.

    Industry convention (AQR / Two Sigma / Bridgewater capacity disclosures):
      VERY_HIGH  — futures / cross-asset risk premia ($5B+)
      HIGH       — large-cap equity factors ($1B-$5B)
      MEDIUM     — mid-cap or event-driven equity ($200M-$1B)
      LOW        — small-cap / illiquid ($20M-$200M)
      VERY_LOW   — micro-cap / niche ($<20M)
    """
    VERY_HIGH = "VERY_HIGH"
    HIGH      = "HIGH"
    MEDIUM    = "MEDIUM"
    LOW       = "LOW"
    VERY_LOW  = "VERY_LOW"


@_dc.dataclass(frozen=True)
class CapacityEstimate:
    """Family-typical capacity estimate for a candidate (no library entry).

    Used by /lab/roadmap to render the capacity badge inline with the
    decay badge — together they tell the user "is this axis worth
    pursuing at our AUM target."
    """
    family:                       str
    using_default:                bool       # True if family unknown → _default
    capacity_class:               CapacityClass
    estimated_capacity_usd:       float      # AUM where Sharpe halves
    comfortable_aum_usd:          float      # AUM with ~80% Sharpe retained
    minimum_aum_usd:              float      # Below this, fixed costs eat α
    notes:                        str        # 1-sentence note
    methodology:                  str        # what the estimate is based on
    parent_family:                str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = _dc.asdict(self)
        d["capacity_class"] = self.capacity_class.value
        return d
