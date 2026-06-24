"""engine/research/chain_library.py — Frontier 4 (2026-06-01):
catalogue of pre-defined research chains.

Each chain is small (3-5 steps) and serves ONE clear research question.
Chains live in code (not YAML) so they're code-reviewed + version-
controlled; if a chain produces consistently useful results we can
externalize it later.

Reference chains:
  - paper_to_candidate   — arxiv search → check graveyard → suggest
                            variation seed → check library for cousins
  - family_audit         — list deployed sleeves → graveyard summary →
                            recent iterations → calibration KPI
"""
from __future__ import annotations

from engine.research.research_chain import Chain, Step


# ── 1. paper_to_candidate ────────────────────────────────────────────


PAPER_TO_CANDIDATE = Chain(
    chain_id="paper_to_candidate",
    description=(
        "Search arxiv for relevant papers → check graveyard for the "
        "first hit's family → check library for cousins → output a "
        "structured candidate idea. Stops cleanly if any step shows "
        "the family is RED (graveyard recommendation=block)."
    ),
    steps=[
        Step(
            name="find_paper",
            tool="arxiv_search",
            args={
                "query":       "{{initial.query}}",
                "max_results": 5,
            },
        ),
        Step(
            name="check_graveyard",
            tool="query_graveyard",
            args={
                # Family inferred from the user-supplied seed query, NOT
                # from the arxiv result, because arxiv titles often don't
                # map cleanly to our family taxonomy.
                "family":          "{{initial.family}}",
                "candidate_title": "{{initial.query}}",
            },
            guard="{{steps.find_paper.status}}",
        ),
        Step(
            name="check_library",
            tool="query_library",
            args={
                "family": "{{initial.family}}",
            },
            # Even if graveyard says "block" we still want to know what's
            # deployed — could be the existing sleeve already covers this.
            guard="{{steps.find_paper.status}}",
        ),
        Step(
            name="get_intuition_rules",
            tool="query_intuition_rules",
            args={
                "context_text": "{{initial.query}}",
            },
            on_failure="continue",
        ),
    ],
)


# ── 2. family_audit ──────────────────────────────────────────────────


FAMILY_AUDIT = Chain(
    chain_id="family_audit",
    description=(
        "Quick health check for a family: what's currently deployed, "
        "what's in the graveyard, how the council has been calibrated "
        "on this family lately. Used as a pre-flight before proposing "
        "any new candidate in the family."
    ),
    steps=[
        Step(
            name="deployed_in_family",
            tool="query_library",
            args={"family": "{{initial.family}}"},
        ),
        Step(
            name="graveyard_in_family",
            tool="query_graveyard",
            args={"family": "{{initial.family}}"},
        ),
        Step(
            name="recent_iterations",
            tool="query_l4_iterations",
            args={"limit": 20},
            on_failure="continue",
        ),
        Step(
            name="family_n_trials",
            tool="family_n_trials_lookup",
            args={"family": "{{initial.family}}"},
            on_failure="continue",
        ),
    ],
)


# ── Registry ─────────────────────────────────────────────────────────


CHAINS: dict[str, Chain] = {
    PAPER_TO_CANDIDATE.chain_id: PAPER_TO_CANDIDATE,
    FAMILY_AUDIT.chain_id:       FAMILY_AUDIT,
}


def get_chain(chain_id: str) -> Chain:
    """Return a chain by id. Raises KeyError if not registered."""
    if chain_id not in CHAINS:
        raise KeyError(
            f"unknown chain_id={chain_id!r}; available: "
            f"{sorted(CHAINS.keys())}"
        )
    return CHAINS[chain_id]


def list_chains() -> list[dict]:
    """Return chain summaries for the REST surface."""
    return [
        {
            "chain_id":    c.chain_id,
            "description": c.description,
            "n_steps":     len(c.steps),
            "step_names":  [s.name for s in c.steps],
        }
        for c in CHAINS.values()
    ]
