"""engine.research_store.hypothesis — Hypothesis store.

Hypothesis is the LOAD-BEARING new artifact in the
PAPER → HYPOTHESIS → TEST → VERDICT chain (locked 2026-06-04 per
[[feedback-paper-driven-research-chain-locked-2026-06-04]]).

A Hypothesis is a structured testable claim extracted from a SPECIFIC
ingested paper. It is the contract between:
  - papers_registry (source paper, with verbatim chunk references)
  - red_lessons (a paper_grounded lesson cites which hypothesis it
    tested)

This package is peer to red_lessons / papers (not a sub-module of
either), because Hypothesis is its own first-class entity in the chain.

Public API:

    from engine.research_store.hypothesis import (
        Hypothesis, VerbatimQuote, HypothesisDirection,
        load_hypotheses, save_hypothesis, find_by_id, latest_per_paper,
        HYPOTHESES_PATH,
    )
"""
from engine.research_store.hypothesis.schema import (
    Hypothesis,
    HypothesisDirection,
    VerbatimQuote,
    HYPOTHESIS_SCHEMA_VERSION,
)
from engine.research_store.hypothesis.store import (
    HYPOTHESES_PATH,
    find_by_id,
    latest_per_paper,
    load_hypotheses,
    save_hypothesis,
)

__all__ = [
    "Hypothesis", "VerbatimQuote", "HypothesisDirection",
    "HYPOTHESIS_SCHEMA_VERSION",
    "HYPOTHESES_PATH",
    "find_by_id", "latest_per_paper", "load_hypotheses", "save_hypothesis",
]
