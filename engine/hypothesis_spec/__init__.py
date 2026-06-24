"""engine.hypothesis_spec — typed structured hypothesis specification.

The project's epistemic backbone (commit thread starting 2026-06-05).
User-stated soul: "把 B+C 当成项目灵魂做".

Why this exists
---------------
A research hypothesis stored as free-text claim ("BAB on the rate curve
carry") is reproducible to a human researcher BUT NOT to:
  - a Composer that builds returns series
  - an Auditor 5 years later asking "exactly what was tested?"
  - a regulator asking "what data vintages + components went in?"

HypothesisSpec is the structured form. Every hypothesis MUST be
extractable into a HypothesisSpec, and every spec is identified by a
deterministic spec_hash so re-running the same spec always produces
the same series → the same verdict.

This is the missing middle layer of the PAPER → HYPOTHESIS → ??? →
TEST → VERDICT chain.

Components
----------
  schema.py    typed dataclass + enums
  hash.py      deterministic spec_hash
  store.py     load/save jsonl (data/research_store/hypothesis_specs.jsonl)
  extractor.py LLM claim-text → spec
  enums.py     all controlled vocabularies

Reproducibility contract (LdP §2):
  spec_hash(spec_A) == spec_hash(spec_B)  ⇒  same returns series  ⇒  same verdict

Audit contract (Bailey-LdP §3 + SR-11-7):
  every Composer call appends to provenance.jsonl
  with {spec_hash, components_used, data_vintages, git_sha, build_ts}
"""
from engine.hypothesis_spec.schema import (
    HypothesisSpec,
    SignalLeg,
    Universe,
    PortfolioConstruction,
    RiskManagement,
    PredictedOutcome,
)
from engine.hypothesis_spec.enums import (
    AssetClass,
    SignalType,
    Sign,
    Weighting,
    Rebalance,
    UniverseSubset,
    FamilyV2,
)
from engine.hypothesis_spec.hash import spec_hash

__all__ = [
    "HypothesisSpec",
    "SignalLeg",
    "Universe",
    "PortfolioConstruction",
    "RiskManagement",
    "PredictedOutcome",
    "AssetClass",
    "SignalType",
    "Sign",
    "Weighting",
    "Rebalance",
    "UniverseSubset",
    "FamilyV2",
    "spec_hash",
]
