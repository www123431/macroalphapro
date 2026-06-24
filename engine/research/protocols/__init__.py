"""engine.research.protocols — Protocol Designer + Executor.

Wraps run_gate into a multi-leg pre-committed test protocol that mirrors how
a senior factor researcher tests a mechanism (primary + 4 robustness +
decomposition + aggregation). Designer is pure deterministic; Executor runs
all legs independently and aggregates verdict.

Public API:
  instantiate_protocol(mechanism, ...) → InstantiatedProtocol (frozen)
  execute_protocol(protocol, proposal, **data) → MultiLegVerdict
"""
from __future__ import annotations

from engine.research.protocols.protocol_designer import (
    DecompositionCheck,
    GENERIC_FAMILY_ID,
    InstantiatedProtocol,
    ResolvedLeg,
    compute_template_warmup,
    instantiate_protocol,
    list_protocol_families,
    load_mechanism,
    load_protocol_family,
    select_family_for_mechanism,
)
from engine.research.protocols.protocol_executor import (
    DecompositionResult,
    LegResult,
    MultiLegVerdict,
    execute_protocol,
)

__all__ = [
    "DecompositionCheck", "DecompositionResult",
    "GENERIC_FAMILY_ID", "InstantiatedProtocol",
    "LegResult", "MultiLegVerdict", "ResolvedLeg",
    "compute_template_warmup",
    "execute_protocol", "instantiate_protocol",
    "list_protocol_families", "load_mechanism",
    "load_protocol_family", "select_family_for_mechanism",
]
