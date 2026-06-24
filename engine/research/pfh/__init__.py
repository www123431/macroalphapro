"""engine/research/pfh — Probabilistic Factor Hypothesizer.

The MVP of "engine that suggests what to test next" — uses STRUCTURED
research history (library + graveyard + intuition_rules + outcome_ledger
+ critic_calibration) as informative Bayesian prior to rank candidate
factor proposals by posterior P(success).

DESIGN DOCTRINES (recorded so future-me / reviewers can audit):

  1. NO LLM SCORING. All numerical scores are deterministic Python
     using a Beta-Binomial conjugate model with empirical Bayes
     shrinkage. LLM is invoked ONLY for narrative rationale text
     after scoring, and it can only reference evidence by ID.

  2. SMALL N → SHRINK, DON'T FIT. With N=36 labeled mechanisms (6
     GREEN + 2 YELLOW + 28 RED), any logistic regression or ML
     classifier would overfit catastrophically. Beta-Binomial with
     weak informative priors is the honest choice (Gelman: "don't
     fit a model more complex than your data supports").

  3. CREDIBLE INTERVALS, NOT POINT ESTIMATES. Every posterior P(success)
     is reported with [5th, 95th]-percentile credible interval. Users
     who skip the interval and read only the mean are making the
     classic small-sample point-estimate mistake.

  4. NO CIRCULAR LOOP. PFH's prior updates ONLY from outcomes of
     human-originated or council-disagreed candidates. Outcomes of
     PFH-suggested candidates are tracked SEPARATELY for calibration
     of PFH itself — they do NOT feed back into the prior. This is
     the same circuit-breaker doctrine as Frontier 3 (calibration
     feedback / human-gated rule promotion).

  5. OUTPUTS ARE COMPOSE-SPEC YAMLs. The closed loop precondition
     from Week 1c. PFH does NOT output free-text suggestions; it
     outputs `data/feature_store/_specs/<pfh-suggested-id>.yaml`
     stubs (status: pending_review) that the user can directly
     materialize after audit.

  6. EXPLICIT DEPENDENCY ANNOTATION. When PFH suggests a factor
     requiring a universe / signal / weighting that doesn't exist
     yet in the feature_store catalog, the proposal explicitly lists
     "needs_new_axis_components" so the user knows what to build
     before materializing.
"""
from engine.research.pfh.catalog import (
    LabeledMechanism,
    load_labeled_mechanisms,
    overall_base_rate,
    per_family_counts,
)
from engine.research.pfh.bayesian import (
    BetaBinomialPosterior,
    score_candidate,
)
from engine.research.pfh.generator import (
    CandidateProposal,
    generate_candidates,
)
from engine.research.pfh.proposer import (
    suggest_top_k,
    write_pfh_compose_spec,
)

__all__ = [
    "LabeledMechanism", "load_labeled_mechanisms",
    "overall_base_rate", "per_family_counts",
    "BetaBinomialPosterior", "score_candidate",
    "CandidateProposal", "generate_candidates",
    "suggest_top_k", "write_pfh_compose_spec",
]
