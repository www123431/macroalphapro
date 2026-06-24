"""engine/research/pfh/constrained_generator.py — Catalog-bounded
candidate enumeration.

The "closed loop" mode of PFH: enumerate only (universe × signal ×
weighting) tuples that:
  - reference EXISTING axis components in the feature_store catalog
  - have NOT yet been tested (i.e. no compose-spec already exists)

Each candidate is a CandidateProposal compatible with the same
scoring/diversification/output pipeline as the unconstrained generator
in generator.py. The difference is candidates carry NO needs_new_axes
warnings — they're immediately materialize-able.

DESIGN PRINCIPLE: this generator does NOT extrapolate beyond what
EXISTS. If a user wants PFH to suggest factors needing new axes, they
use the unconstrained generator. This module is the strictly-honest
"engine suggests something it can ACTUALLY run" path.

Family inference: the generator infers a family for each tuple from
the signal recipe name (heuristic, since signal YAMLs don't yet carry
explicit family tags). This is a known approximation; signal YAMLs
will gain explicit `family:` fields in Week 5+ when we add more
sophisticated cousin reasoning.
"""
from __future__ import annotations

import logging
from typing import Optional

from engine.research.pfh.axis_catalog import (
    AxisCatalog, enumerate_untested_tuples, load_axis_catalog,
)
from engine.research.pfh.generator import CandidateProposal

logger = logging.getLogger(__name__)


# Signal-name → family heuristic. Keep this table small and explicit
# until we add explicit `family:` fields to signal recipes (Week 5+).
_SIGNAL_FAMILY_MAP: dict[str, str] = {
    "momentum_12_1":            "momentum",
    "momentum_3_1":             "momentum",
    "momentum_12_1_vol_scaled": "momentum",
    "reversal_1m":              "reversal",
    "zscore_36mo_residual":     "ts_anomaly",
}


def _infer_family_from_signal(signal_name: str) -> str:
    """Map signal recipe name → family. Falls back to the raw signal
    name (lowercased) when no explicit mapping exists."""
    if signal_name in _SIGNAL_FAMILY_MAP:
        return _SIGNAL_FAMILY_MAP[signal_name]
    # Fallback: use the signal name as its own family
    return signal_name.lower()


def generate_constrained_candidates(
    catalog: Optional[AxisCatalog] = None,
) -> list[CandidateProposal]:
    """Enumerate ALL (universe × signal × weighting) tuples not yet tested.

    Returns CandidateProposal objects ready for the same scoring +
    diversification pipeline as the unconstrained generator. Each
    candidate's needs_new_axes is EMPTY by construction.
    """
    if catalog is None:
        catalog = load_axis_catalog()

    out: list[CandidateProposal] = []
    for (u, s, w) in enumerate_untested_tuples(catalog):
        family = _infer_family_from_signal(s)
        cid = f"pfh_constrained_{u}__{s}__{w}"
        # Truncate id if it gets too long for filesystem
        if len(cid) > 100:
            cid = cid[:97] + "..."
        out.append(CandidateProposal(
            candidate_id=cid,
            proposal_kind="constrained",
            family_normalized=family,
            universe=u,
            signal_recipe=s,
            weighting=w,
            rebalance="monthly",
            derived_from=[],
            cousin_warnings=[],
            needs_new_axes=[],
            rationale_seeds=[
                f"untested combination on existing axes: "
                f"{u} × {s} × {w}",
            ],
        ))
    return out
