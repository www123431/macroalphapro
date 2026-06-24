"""engine.decay_forecast.schema — typed estimate output."""
from __future__ import annotations

import dataclasses as _dc
from enum import Enum
from typing import Any


class DecayRisk(str, Enum):
    """Coarse risk band derived from expected_alpha_5y_ahead.
    UI badge uses this for tone (green / warn / alert)."""
    LOW    = "LOW"        # >=70% of baseline retained at 5y
    MEDIUM = "MEDIUM"     # 40-70%
    HIGH   = "HIGH"       # 15-40%
    SEVERE = "SEVERE"     # <15% (effectively dead)


@_dc.dataclass(frozen=True)
class DecayEstimate:
    """Family-keyed forward decay estimate for a candidate.

    Constructed for a CANDIDATE that doesn't yet exist in library —
    parameters come from FAMILY_DECAY_PARAMS table (MP 2016 / LR 2018
    family-typical λ).
    """
    family:                str          # canonical family label
    parent_family:         str | None
    using_default:         bool         # True if family not in registry → fell back to _default
    baseline_alpha:        float        # assumed starting α/yr (default 0.05)
    publication_year:      int | None
    years_since_pub:       float

    # Decay rates (annual exponential λ)
    mp_2016_lambda:        float        # central estimate (McLean-Pontiff 2016)
    lr_2018_lambda:        float        # upper-bound (Linnainmaa-Roberts 2018)
    half_life_years:       float        # ln(2) / mp_lambda

    # Forecasts
    expected_alpha_now:    float        # α today (after years_since_pub decay)
    expected_alpha_5y:     float        # α 5 years forward
    expected_alpha_10y:    float        # α 10 years forward

    # Confidence band (lower = LR upper-bound = more pessimistic)
    expected_alpha_5y_lower:  float
    expected_alpha_10y_lower: float

    # Coarse risk + actionable note
    risk:                  DecayRisk
    note:                  str          # 1-sentence human reading

    def to_dict(self) -> dict[str, Any]:
        d = _dc.asdict(self)
        d["risk"] = self.risk.value
        return d
