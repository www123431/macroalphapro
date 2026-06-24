"""engine/research/pfh/proposer.py — Score + rank + emit compose-spec YAMLs.

The end-to-end PFH pipeline:
  1. Read labeled mechanism dataset (catalog.load_labeled_mechanisms)
  2. Enumerate candidates (generator.generate_candidates)
  3. Score each via Beta-Binomial posterior on (family, anchor-family) cell
  4. Apply deterministic adjustments (cousin penalty, post-pub decay)
  5. Rank by posterior_mean
  6. Emit compose-spec YAML stubs for the top-K
  7. Persist scoring evidence to data/research/pfh_suggestions.jsonl

DOCTRINE: all numerical scoring is in this module + bayesian.py. No
LLM call. Rationale narrative (LLM-generated) is OPTIONAL and lives in
engine/research/pfh/rationale.py (Week 2.5+, not in MVP).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from engine.research.pfh.bayesian import (
    BetaBinomialPosterior, score_candidate,
)
from engine.research.pfh.catalog import (
    LabeledMechanism, load_labeled_mechanisms,
    overall_base_rate, per_family_counts,
)
from engine.research.pfh.constrained_generator import (
    generate_constrained_candidates,
)
from engine.research.pfh.generator import (
    CandidateProposal, generate_candidates,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PFH_LEDGER = REPO_ROOT / "data" / "research" / "pfh_suggestions.jsonl"
COMPOSE_SPECS_DIR = REPO_ROOT / "data" / "feature_store" / "_specs"


# ── Scored proposal ──────────────────────────────────────────────────


@dataclass
class ScoredProposal:
    """CandidateProposal + posterior + final score after adjustments."""
    proposal:          CandidateProposal
    posterior:         BetaBinomialPosterior
    cousin_penalty:    float          # multiplicative ∈ (0, 1]
    final_score:       float          # posterior_mean * cousin_penalty
    score_breakdown:   dict           # auditable decomposition

    def to_dict(self) -> dict:
        return {
            "proposal":        self.proposal.to_dict(),
            "posterior":       self.posterior.to_dict(),
            "cousin_penalty":  round(self.cousin_penalty, 4),
            "final_score":     round(self.final_score, 4),
            "score_breakdown": self.score_breakdown,
        }


# ── Scoring ──────────────────────────────────────────────────────────


_COUSIN_PENALTY_PER_RED = 0.15   # each cousin RED multiplies score by (1 - 0.15)


def _score_one_candidate(
    candidate:       CandidateProposal,
    labels:          list[LabeledMechanism],
    fam_counts:      dict,
    base_rate:       float,
    prior_strength:  float,
) -> ScoredProposal:
    """Compute posterior + adjustments for one candidate.

    The scoring CELL for the posterior is the candidate's family_normalized.
    Cousin penalty is applied based on the candidate's cousin_warnings
    list (already populated by the generator).
    """
    fam = candidate.family_normalized
    cell = fam_counts.get(fam, {"n_green": 0, "n_yellow": 0, "n_red": 0})

    posterior = score_candidate(
        n_green=cell["n_green"],
        n_yellow=cell["n_yellow"],
        n_red=cell["n_red"],
        base_rate=base_rate,
        prior_strength=prior_strength,
    )

    # Cousin penalty: 1 - 0.15 per RED warning (capped at 0.95 total
    # reduction so even heavily-warned candidates retain some signal).
    n_warnings = len(candidate.cousin_warnings)
    cousin_penalty = max(0.05, (1.0 - _COUSIN_PENALTY_PER_RED) ** n_warnings)

    final = posterior.posterior_mean * cousin_penalty

    return ScoredProposal(
        proposal=candidate,
        posterior=posterior,
        cousin_penalty=cousin_penalty,
        final_score=final,
        score_breakdown={
            "family":           fam,
            "base_rate":        round(base_rate, 4),
            "prior_strength":   prior_strength,
            "cell_n_green":     cell["n_green"],
            "cell_n_yellow":    cell["n_yellow"],
            "cell_n_red":       cell["n_red"],
            "posterior_mean":   posterior.posterior_mean,
            "credible_05_95":   [posterior.credible_05,
                                  posterior.credible_95],
            "n_cousin_warnings": n_warnings,
            "cousin_penalty":   round(cousin_penalty, 4),
            "final_score":      round(final, 4),
        },
    )


# ── Top-K aggregator ─────────────────────────────────────────────────


def _diversify_top_k(
    scored: list[ScoredProposal],
    k: int,
    *,
    max_per_family:   int = 2,
    max_per_universe: int = 2,
) -> list[ScoredProposal]:
    """Greedy 2-axis diversification: cap top-K by both family AND
    universe.

    WHY family cap: at small N, one family dominates raw posterior
    (e.g. earnings_underreaction has 2 GREEN, 0 RED → all cross-market
    variants tie at the top). Without the cap user gets a useless
    mono-mechanism suggestion list.

    WHY universe cap (added 2026-06-01 for cross-asset demo): ALL
    untested-cell candidates land at the SAME posterior mean (since
    they share base rate as prior and have 0 cell observations).
    Without universe cap, ties are broken by alphabetical order of the
    universe name, which means one universe dominates the top-K.
    Capping per universe forces the engine to spread across asset
    classes — essential when the catalog contains both equity and
    futures universes.
    """
    out: list[ScoredProposal] = []
    family_count:   dict[str, int] = {}
    universe_count: dict[str, int] = {}
    for s in scored:
        fam = s.proposal.family_normalized
        uni = s.proposal.universe or "_no_universe"
        if family_count.get(fam, 0) >= max_per_family:
            continue
        if universe_count.get(uni, 0) >= max_per_universe:
            continue
        out.append(s)
        family_count[fam] = family_count.get(fam, 0) + 1
        universe_count[uni] = universe_count.get(uni, 0) + 1
        if len(out) >= k:
            break
    # If caps prevented filling k, relax: take next-best ignoring caps
    if len(out) < k:
        seen = {id(s) for s in out}
        for s in scored:
            if id(s) in seen:
                continue
            out.append(s)
            if len(out) >= k:
                break
    return out


def suggest_top_k(
    k:                int = 5,
    *,
    labels:           Optional[list[LabeledMechanism]] = None,
    prior_strength:   float = 4.0,
    max_per_family:   int = 2,
    max_per_universe: int = 2,
    write_specs:      bool = False,
    write_ledger:     bool = True,
    mode:             str = "open",
) -> dict:
    """End-to-end PFH suggestion pipeline.

    Args:
      k:               number of top suggestions to return
      labels:          override label set (used for tests / counterfactual
                        runs). Defaults to load_labeled_mechanisms()
      prior_strength:  pseudo-observations of prior weight (default 4)
      max_per_family:  diversity cap — at most this many top-K slots
                        from the same family. Default 2.
      write_specs:     if True, materialize top-K as compose-spec YAML
                        stubs in data/feature_store/_specs/
      write_ledger:    if True, append the full run to pfh_suggestions.jsonl
      mode:            "open" — original generator, may suggest factors
                        needing new axis components (research mode)
                       "constrained" — only suggest factors composable
                        from EXISTING axis components (closed-loop mode).
                        Specs emitted in this mode are immediately
                        materialize-able with no human axis-component work.

    Returns dict with: base_rate / n_candidates_total / top / run_id / ts
    """
    if labels is None:
        labels = load_labeled_mechanisms()

    br = overall_base_rate(labels)
    base_rate = br["p_green"] or 0.5

    fam_counts = per_family_counts(labels)
    if mode == "constrained":
        candidates = generate_constrained_candidates()
    elif mode == "open":
        candidates = generate_candidates(labels)
    else:
        raise ValueError(f"mode must be 'open' or 'constrained'; got {mode!r}")

    scored: list[ScoredProposal] = [
        _score_one_candidate(c, labels, fam_counts, base_rate, prior_strength)
        for c in candidates
    ]
    scored.sort(key=lambda s: -s.final_score)
    top = _diversify_top_k(scored, k,
                            max_per_family=max_per_family,
                            max_per_universe=max_per_universe)

    run_id = f"pfh-{uuid.uuid4().hex[:12]}"
    ts = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    written_paths: list[str] = []
    if write_specs:
        COMPOSE_SPECS_DIR.mkdir(parents=True, exist_ok=True)
        for s in top:
            path = write_pfh_compose_spec(s, ts=ts, run_id=run_id)
            try:
                rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            except ValueError:
                # tmp_path fixtures live outside REPO_ROOT
                rel = str(path).replace("\\", "/")
            written_paths.append(rel)

    out = {
        "run_id":              run_id,
        "ts":                  ts,
        "mode":                mode,
        "base_rate_used":      base_rate,
        "n_candidates_total":  len(candidates),
        "n_scored":            len(scored),
        "k_requested":         k,
        "prior_strength":      prior_strength,
        "top":                 [s.to_dict() for s in top],
        "written_spec_paths":  written_paths,
        "labels_summary":      br,
    }

    if write_ledger:
        _append_to_ledger(out)

    return out


def _append_to_ledger(entry: dict) -> None:
    try:
        PFH_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with PFH_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        # Vector RAG (2026-06-02): incremental refresh — embeddings module
        # dedupes by row hash so this is O(new rows) not O(ledger).
        try:
            from engine.research import embeddings as _E
            _E.build_index("pfh_suggestions")
        except ImportError:
            pass
        except Exception:
            logger.warning("pfh_suggestions embedding refresh failed",
                            exc_info=True)
    except Exception:
        logger.exception("PFH ledger append failed (non-fatal)")


def read_pfh_history(limit: int = 50) -> list[dict]:
    """Read recent PFH suggestion runs newest-first."""
    if not PFH_LEDGER.is_file():
        return []
    out: list[dict] = []
    with PFH_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    out.reverse()
    return out[: max(1, int(limit))]


# ── Compose-spec YAML emitter ────────────────────────────────────────


def _resolve_universe_input_path(universe_ref: str) -> Optional[str]:
    """Look up a universe's input_path for the constrained-mode hash inputs."""
    from engine.research.pfh.axis_catalog import UNIVERSES_DIR
    p = UNIVERSES_DIR / f"{universe_ref}.yaml"
    if not p.is_file():
        return None
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return raw.get("input_path")
    except Exception:
        return None


def write_pfh_compose_spec(
    scored: ScoredProposal,
    *,
    ts: str,
    run_id: str,
) -> Path:
    """Write a PFH-suggested compose-spec YAML stub.

    The stub is intentionally INCOMPLETE — it references axis components
    that may not exist yet (needs_new_axes is the user's TODO list).
    Status is "pending_pfh_review" so the user knows this came from
    PFH and needs human audit BEFORE materialize.

    For CONSTRAINED-mode proposals (universe/signal/weighting all set
    + needs_new_axes empty), the writer populates inputs[] with the
    universe's real input_path so the materializer's reproducibility
    hash works correctly.

    Returns the path written.
    """
    prop = scored.proposal
    spec_id = prop.candidate_id

    # Inline the spec content. If any axis is None or marked NEW, we
    # write a placeholder ref so the YAML loads but materialize fails
    # with a clear error pointing at what's missing.
    universe_ref  = prop.universe     or "PLACEHOLDER_universe"
    signal_ref    = prop.signal_recipe or "PLACEHOLDER_signal"
    weighting_ref = prop.weighting    or "PLACEHOLDER_weighting"

    # Constrained-mode: resolve the universe's input_path so the
    # materializer can hash it correctly. Open-mode keeps placeholder.
    if prop.proposal_kind == "constrained":
        real_input_path = _resolve_universe_input_path(universe_ref)
        if real_input_path:
            inputs_list = [{"cache_path": real_input_path}]
        else:
            inputs_list = [{"cache_path": "data/feature_store/_specs/PFH_DEPENDENCIES_TBD"}]
    else:
        inputs_list = [{"cache_path": "data/feature_store/_specs/PFH_DEPENDENCIES_TBD"}]

    spec_doc = {
        "_schema_version": 1,
        "spec_id": spec_id,
        "version": 1,
        "description": (
            f"PFH-suggested factor (run {run_id}, generated {ts}). "
            f"Kind: {prop.proposal_kind}. Family: {prop.family_normalized}. "
            f"DO NOT materialize until human review confirms axis component "
            f"definitions are sound."
        ),
        "compose": {
            "universe":  {"ref": universe_ref},
            "signal":    {"ref": signal_ref},
            "weighting": {"ref": weighting_ref},
            "rebalance": {"freq": prop.rebalance},
        },
        "output": {
            "kind": "monthly_returns",
            "expected_date_range": {
                "start":   "2010-01-01",
                "end_min": "2023-12-01",
            },
            "expected_shape": {"n_rows": [50, 600]},
            "sanity": {
                "no_nan_after_first_observation": False,
                "annualized_vol_range":    [0.02, 0.50],
                "annualized_sharpe_range": [-2.0, 3.0],
            },
        },
        "inputs": inputs_list,
        "source_module_files": [
            "engine/feature_store/composer.py",
            "engine/feature_store/primitives.py",
            "engine/research/pfh/proposer.py",
        ],
        "audit": {
            "added_date":      ts[:10],
            "added_by":        "pfh",
            "status":          "pending_pfh_review",
            "pfh_run_id":      run_id,
            "pfh_score": {
                "posterior_mean":  scored.posterior.posterior_mean,
                "credible_05":     scored.posterior.credible_05,
                "credible_95":     scored.posterior.credible_95,
                "cousin_penalty":  round(scored.cousin_penalty, 4),
                "final_score":     round(scored.final_score, 4),
            },
            "pfh_evidence": {
                "derived_from":      prop.derived_from,
                "cousin_warnings":   prop.cousin_warnings,
                "needs_new_axes":    prop.needs_new_axes,
                "rationale_seeds":   prop.rationale_seeds,
                "score_breakdown":   scored.score_breakdown,
            },
            "notes": (
                "PFH MVP output (Week 2). LLM rationale narrative not "
                "yet generated (Week 2.5). Human reviewer responsible "
                "for: (1) defining the needs_new_axes components, "
                "(2) tightening expected_date_range/shape/sanity, "
                "(3) flipping status to 'pending_review' after audit."
            ),
        },
    }

    out_path = COMPOSE_SPECS_DIR / f"{spec_id}.yaml"
    out_path.write_text(
        yaml.safe_dump(spec_doc, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────


def _cli() -> None:
    """python -m engine.research.pfh.proposer <suggest|list>"""
    import sys
    args = sys.argv[1:]
    cmd = args[0] if args else "suggest"

    if cmd == "suggest":
        k = int(args[1]) if len(args) > 1 else 5
        out = suggest_top_k(k=k, write_specs=False, write_ledger=True)
        # Print compact summary
        print(json.dumps({
            "run_id":       out["run_id"],
            "base_rate":    out["base_rate_used"],
            "n_candidates": out["n_candidates_total"],
            "top": [
                {
                    "candidate_id":     s["proposal"]["candidate_id"],
                    "kind":             s["proposal"]["proposal_kind"],
                    "family":           s["proposal"]["family_normalized"],
                    "posterior_mean":   s["posterior"]["posterior_mean"],
                    "credible_05_95":   s["score_breakdown"]["credible_05_95"],
                    "cousin_penalty":   s["cousin_penalty"],
                    "final_score":      s["final_score"],
                    "needs_new_axes":   s["proposal"]["needs_new_axes"],
                    "cousin_warnings":  s["proposal"]["cousin_warnings"],
                }
                for s in out["top"]
            ],
        }, indent=2, default=str))
        return

    if cmd == "list":
        n = int(args[1]) if len(args) > 1 else 10
        runs = read_pfh_history(limit=n)
        print(json.dumps({"n": len(runs), "runs": runs},
                          indent=2, default=str))
        return

    print("usage: suggest [k] | list [n]",
          file=__import__("sys").stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    _cli()
