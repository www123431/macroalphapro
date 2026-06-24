"""engine.capacity — family-keyed capacity sub-MVP for /lab/roadmap.

For detailed AUM-level capacity simulation, use
engine.portfolio.capacity_simulator (Pastor-Stambaugh / Berk-Green
framework).

This module is a fast lookup for PRE-FLIGHT / roadmap badges where
a coarse family-typical estimate is sufficient.
"""
from engine.capacity import api, schema
from engine.capacity.api import (
    estimate_for_family,
    list_supported_families,
)
from engine.capacity.schema import CapacityEstimate, CapacityClass

__all__ = [
    "api", "schema",
    "estimate_for_family", "list_supported_families",
    "CapacityEstimate", "CapacityClass",
]
