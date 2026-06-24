"""engine.decay_forecast — forward alpha-mortality estimates for SESSION pre-flight.

Wraps engine.research.forward_decay_prediction's FAMILY_DECAY_PARAMS into
a candidate-keyed query (vs. the library-keyed predict_decay() upstream).

Use case: when user opens a research_new session for a candidate that
doesn't yet exist in /lab/library, pre-flight wizard needs the
3-number "alpha mortality" badge:
  - empirical decay rate (MP 2016 family-typical λ)
  - theoretical upper bound (LR 2018 family-typical lr_lambda)
  - expected α at 5y forward

This module provides that without requiring a library YAML entry.

Per Gap B施工 (2026-06-03 session protocol gaps audit). Industry-
standard "alpha mortality table" per Two Sigma factor proposal pattern.
"""
from engine.decay_forecast import api, schema
from engine.decay_forecast.api import (
    estimate_for_family,
    list_supported_families,
)
from engine.decay_forecast.schema import (
    DecayEstimate, DecayRisk,
)

__all__ = [
    "api", "schema",
    "estimate_for_family", "list_supported_families",
    "DecayEstimate", "DecayRisk",
]
