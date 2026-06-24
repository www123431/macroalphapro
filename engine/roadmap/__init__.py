"""engine.roadmap — typed research-direction roadmap.

Public API:
    schema.ResearchAxis / AxisState / AxisTier / AxisOutcome
    store.list_axes / get_axis / upsert_axis / delete_axis

See `data/roadmap/axes.yaml` for the registry. Mutations should flow
through doctrine sessions (so a memory_doctrine_locked event accompanies
the roadmap change) — but direct upsert is allowed for one-off curation.
"""
from engine.roadmap import schema, store
from engine.roadmap.schema import (
    AxisOutcome, AxisState, AxisTier, ResearchAxis,
)

__all__ = [
    "schema", "store",
    "ResearchAxis", "AxisState", "AxisTier", "AxisOutcome",
]
