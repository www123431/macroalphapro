"""engine.research_store.papers.shelves — controlled Shelf taxonomy.

A `Shelf` is a logical category a paper occupies. Papers are MULTI-LABEL:
one paper can be on several shelves simultaneously. Example:

  Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere"
    → shelves = {green_motivation,         # we deploy momentum
                 green_critique,           # the AMP critique of pure value
                 doctrine_method,          # canonical regime-conditional
                                            #   correlation reference
                 dormant_revisit}          # CN A-share retesting candidate

This is the partitioning axis: when a candidate's status flips
GREEN→YELLOW, we update the shelves on its paper, not the paper's
embedding or text.

Why a taxonomy and not free labels:
  - Stable contract for the RAG briefing layer (P4): "give me all
    `green_critique` papers in the MOMENTUM family" must mean the same
    thing every session.
  - Cross-paper consistency: if 50 papers carry `green_motivation`,
    they ALL must have the same semantic meaning (motivated a deployed
    factor).
  - Adding a new shelf requires deliberation. If you can't fit a
    paper into the existing 8 shelves, that's an insight, not noise.
"""
from __future__ import annotations

from enum import Enum


class Shelf(str, Enum):
    """8-value controlled vocabulary. Do not widen by reflex."""

    # Papers behind deployed factors / sleeves
    GREEN_MOTIVATION    = "green_motivation"     # paper motivated a DEPLOYED factor
    GREEN_CRITIQUE      = "green_critique"       # paper critiques / refines a deployed factor

    # Papers behind marginal-but-promising candidates
    YELLOW_MOTIVATION   = "yellow_motivation"    # paper motivated a YELLOW (iterating)

    # Papers behind failed candidates
    RED_MOTIVATION      = "red_motivation"       # paper motivated a RED candidate
    RED_CRITIQUE        = "red_critique"         # paper explains WHY a RED failed

    # Methodology + doctrine
    DOCTRINE_METHOD     = "doctrine_method"      # framework / methodology paper used
                                                  #   across judgments (HLZ / LR / MP /
                                                  #   Bailey-LdP / Frazzini-Israel-
                                                  #   Moskowitz / FF / HXZ)

    # Future
    DORMANT_REVISIT     = "dormant_revisit"      # currently not actionable; would
                                                  #   reactivate if new data / market /
                                                  #   tool arrives

    # Escape hatch — must come with rationale
    OTHER               = "other"


SHELF_DOCS: dict[Shelf, dict[str, str]] = {
    Shelf.GREEN_MOTIVATION: {
        "definition":
            "Paper that motivated a strategy now DEPLOYED in production. "
            "These are the load-bearing references — when defending the "
            "book, reviewers will ask 'show me the paper for this sleeve'.",
        "queryability_hint":
            "Use as primary citation when explaining a deployed sleeve.",
    },
    Shelf.GREEN_CRITIQUE: {
        "definition":
            "Paper that critiques / refines / extends a now-DEPLOYED factor. "
            "Must be visible in any sleeve-decay or sleeve-review session.",
        "queryability_hint":
            "Surface during decay audits; this is the 'have we been "
            "explicitly disproven by recent work?' signal.",
    },
    Shelf.YELLOW_MOTIVATION: {
        "definition":
            "Paper behind a YELLOW (marginal, iterating, not yet deployed) "
            "candidate. These are 'maybe later' tracks.",
        "queryability_hint":
            "Surface when a similar candidate is proposed — reviewer can "
            "decide whether to re-test the YELLOW or treat as duplicate.",
    },
    Shelf.RED_MOTIVATION: {
        "definition":
            "Paper that motivated a candidate that subsequently went RED. "
            "Cataloged so the same paper-driven hypothesis isn't re-tried "
            "without new evidence.",
        "queryability_hint":
            "Surface during new-candidate proposal — 'this paper's "
            "hypothesis was tested and failed; here's why'.",
    },
    Shelf.RED_CRITIQUE: {
        "definition":
            "Paper that explains the failure mode of a RED candidate. "
            "Often a meta-analysis (HXZ) or methodology paper (Bailey-LdP).",
        "queryability_hint":
            "Surface alongside RED_MOTIVATION to explain WHY the candidate "
            "failed, not just THAT it did.",
    },
    Shelf.DOCTRINE_METHOD: {
        "definition":
            "Framework / methodology paper applied across MANY candidates "
            "(e.g. Harvey-Liu-Zhu 2016 |t|>=3 bar, Bailey-LdP 2014 "
            "deflated SR, Linnainmaa-Roberts 2018 post-pub decay). These "
            "are gravity wells — referenced by nearly every lesson.",
        "queryability_hint":
            "Always include in DA briefing context; these define the "
            "rules we judge candidates against.",
    },
    Shelf.DORMANT_REVISIT: {
        "definition":
            "Paper whose hypothesis is not currently testable / actionable "
            "but would be if a new data source / market / tool arrives. "
            "Linked to a REDLesson.dormant_revisits entry.",
        "queryability_hint":
            "Surface when a new dataset / market access is acquired — "
            "system asks 'does this unlock any dormant tests?'",
    },
    Shelf.OTHER: {
        "definition":
            "Escape hatch. A paper that doesn't fit the 7 categorical "
            "shelves. MUST be paired with an explanatory note in the "
            "entry's `note` field — silent OTHERs accumulate as debt.",
        "queryability_hint":
            "Audit periodically: review every OTHER and try to re-shelf "
            "into a categorical bin.",
    },
}


# Import-time integrity check.
assert set(SHELF_DOCS.keys()) == set(Shelf), (
    f"SHELF_DOCS missing entries for: {set(Shelf) - set(SHELF_DOCS.keys())}"
)
