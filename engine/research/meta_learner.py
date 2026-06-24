"""engine/research/meta_learner.py — Phase 6.5 Tier 1 (Beta-Binomial family base rates).

Per [[project-meta-learner-design-2026-05-30]]: Tier 1 = Beta-Binomial
family base rates + cold-start published priors (HXZ 2020 / MP 2016).
Advisory-only — never auto-modifies gate criteria, never auto-deploys,
never overrides H7. Just annotates the prior odds.

What this is:
  - Per-family Beta(α, β) posterior over P(strict-gate pass)
  - Updated from data/research/gate_runs.jsonl (every gate run we've done)
  - Cold-start prior from published-paper base rates so unseen families
    don't get Beta(1, 1) flat-uniform

What this is NOT:
  - Auto-execution: never short-circuits the strict gate, never deploys
  - Verdict generator: outputs a prior; the actual verdict comes from gate
  - LLM-driven: pure deterministic Bayesian update; LLM only enters in
    Tier 2 for *cluster naming* of patterns, which is not in this build

Public API:
  MetaLearner.from_disk(...)                — build from gate_runs.jsonl
  ml.predict(family)                         — Beta posterior + 95% CI
  ml.compare(families)                       — rank by expected value
  ml.update(family, outcome)                  — append observation
  ml.cold_start_prior(family)                  — published baseline
  ml.summary()                                  — DataFrame of all families

Strict red lines (must hold across all callers):
  1. NEVER modify pass_criteria, alpha_t, deflsr_min, etc.
  2. NEVER classify verdict; only annotate prior_odds + sample_size
  3. NEVER auto-deploy regardless of prior strength
  4. NEVER override H7 (post-pub decay flag) — those are research checks,
     not learner outputs
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Iterable

import yaml

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_RUNS_PATH = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"
LIBRARY_DIR    = REPO_ROOT / "data" / "research" / "mechanism_library"


# ── Cold-start priors ──────────────────────────────────────────────────────
# Published-paper base rates from HXZ (2020) "Replicating Anomalies"
# and McLean-Pontiff (2016) "Does Academic Research Destroy Stock Return
# Predictability?". Values represent prior P(strict-gate pass) for a
# fresh candidate in each family, BEFORE seeing any data from our gate.
#
# Encoded as (alpha, beta) of a Beta(α, β) so we can update in closed form.
# alpha ≈ effective "success count"; beta ≈ effective "failure count".
# (alpha + beta) ~ "pseudo-observations" — keep this small (5-15) so
# real data dominates after a handful of observations.
#
# Source references:
#   HXZ 2020: replication success rates by anomaly category
#   MP 2016:  ~32% post-publication decay across 97 anomalies
#   AMP 2013: cross-asset momentum + carry survive in 4-asset-class data
COLD_START_PRIORS: dict[str, tuple[float, float]] = {
    # Higher-base-rate families (per HXZ 2020 replication)
    "carry":               (3.0, 7.0),     # 30%  — cross-asset carry well-replicated
    "tsmom":               (2.5, 7.5),     # 25%  — TSMOM post-MOP 2012 partial decay
    "momentum":            (2.0, 8.0),     # 20%  — cross-section momentum, decay
    "low_vol":             (2.0, 8.0),     # 20%  — BAB family, partial decay
    "value":               (1.5, 8.5),     # 15%  — value premium weak post-2010
    # Lower-base-rate families
    "quality":             (1.0, 9.0),     # 10%  — junk premium era 2010s
    "profitability":       (1.5, 8.5),     # 15%
    "growth":              (1.0, 9.0),     # 10%
    "residual_momentum":   (1.5, 8.5),     # 15%
    "post_earnings_drift": (2.0, 8.0),     # 20%  — well-replicated, anchor
    "vol_carry":           (1.0, 9.0),     # 10%  — vol-selling crisis-fragile
    # Speculative / less-replicated families
    "news_attention":      (0.5, 9.5),     # 5%
    "text_nlp":            (0.5, 9.5),     # 5%
    "insider":             (0.5, 9.5),     # 5%
    "holdings_13f":        (0.5, 9.5),     # 5%
    "options":             (0.5, 9.5),     # 5%
    "patents":             (0.5, 9.5),     # 5%
    "supply_chain":        (0.5, 9.5),     # 5%
    "merger_arb":          (0.5, 9.5),     # 5%
    # Unknown/default
    "unknown":             (1.0, 9.0),     # 10% — generic prior
}


# ── Observation type ───────────────────────────────────────────────────────

@dataclasses.dataclass
class GateObservation:
    """One gate-run outcome (parsed from gate_runs.jsonl)."""
    mechanism_id: str
    family:       str
    outcome:      str       # "pass" | "fail" | "yellow"
    sharpe:       float | None
    deflated_sr:  float | None
    alpha_t:      float | None
    timestamp:    str | None


def _classify_outcome(rec: dict) -> str:
    """Reduce verdict text → {pass, fail, yellow}.

    GREEN (any qualification) → pass
    YELLOW (no qualification) → yellow (we treat as failure for base rate
                                          since deploying YELLOW is rare,
                                          but exposed separately so the
                                          caller can choose)
    RED / None / anything else → fail
    """
    v = (rec.get("verdict") or "").strip().upper()
    if v.startswith("GREEN"):
        return "pass"
    if v.startswith("YELLOW"):
        return "yellow"
    return "fail"


def _family_lookup_from_library() -> dict[str, str]:
    """Read library YAMLs to map mechanism_id → family."""
    out: dict[str, str] = {}
    if not LIBRARY_DIR.exists():
        return out
    for yml in LIBRARY_DIR.glob("*.yaml"):
        try:
            with yml.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            mid = data.get("id") or yml.stem
            fam = data.get("family")
            if mid and fam:
                out[mid] = str(fam)
        except Exception as exc:
            logger.warning("library yaml parse failed %s: %s", yml, exc)
    return out


def _parse_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_gate_observations(
    *,
    gate_path: Path | None = None,
    family_map: dict[str, str] | None = None,
) -> list[GateObservation]:
    p = gate_path or GATE_RUNS_PATH
    fam_map = family_map if family_map is not None else _family_lookup_from_library()
    if not p.exists():
        return []
    obs: list[GateObservation] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        mech_id = rec.get("mechanism") or rec.get("mechanism_id")
        if not mech_id:
            continue
        family = (rec.get("family") or fam_map.get(mech_id) or "unknown")
        obs.append(GateObservation(
            mechanism_id=str(mech_id),
            family=str(family),
            outcome=_classify_outcome(rec),
            sharpe=_parse_float(rec.get("standalone_sharpe")),
            deflated_sr=_parse_float(rec.get("deflated_sr")),
            alpha_t=_parse_float(rec.get("alpha_t_ff5umd")),
            timestamp=rec.get("ts"),
        ))
    return obs


# ── Beta-Binomial core ─────────────────────────────────────────────────────

@dataclasses.dataclass
class FamilyPosterior:
    family:        str
    alpha:         float
    beta:          float
    n_observations: int
    prior_source:  str       # "cold_start" | "data-only"
    successes:     int
    failures:      int
    yellows:       int

    @property
    def mean(self) -> float:
        """Posterior expected P(pass)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed CI via Wilson approximation (good for small N).

        For better accuracy at tiny α+β we'd use scipy.stats.beta.ppf, but
        the project's stdlib-only constraint favors a closed-form approximation.
        """
        try:
            from scipy.stats import beta as _beta_dist
            lo = float(_beta_dist.ppf((1 - level) / 2, self.alpha, self.beta))
            hi = float(_beta_dist.ppf(1 - (1 - level) / 2, self.alpha, self.beta))
            return lo, hi
        except ImportError:
            # Wilson approximation around the mean
            p = self.mean
            n = max(self.alpha + self.beta, 1.0)
            z = 1.96 if level == 0.95 else 2.58
            denom = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
            return max(0.0, center - half), min(1.0, center + half)

    def to_dict(self) -> dict:
        lo, hi = self.credible_interval()
        return {
            "family":          self.family,
            "alpha":           round(self.alpha, 3),
            "beta":            round(self.beta, 3),
            "posterior_mean":  round(self.mean, 4),
            "ci_low":          round(lo, 4),
            "ci_high":         round(hi, 4),
            "n_observations":  self.n_observations,
            "successes":       self.successes,
            "failures":        self.failures,
            "yellows":         self.yellows,
            "prior_source":    self.prior_source,
        }


class MetaLearner:
    """In-memory Beta-Binomial state for all observed families.

    Construction:
      ml = MetaLearner()
      ml.bulk_update(observations)   # observations: Iterable[GateObservation]

    Or:
      ml = MetaLearner.from_disk()    # auto-loads gate_runs.jsonl
    """

    def __init__(self,
                  *,
                  yellow_counts_as_pass: bool = False,
                  cold_start_priors: dict[str, tuple[float, float]] | None = None):
        self.priors = dict(cold_start_priors or COLD_START_PRIORS)
        self.yellow_counts_as_pass = yellow_counts_as_pass
        # Per-family counts (not yet folded with prior)
        self._counts: dict[str, dict[str, int]] = {}

    # ── construction
    @classmethod
    def from_disk(cls,
                   *,
                   gate_path: Path | None = None,
                   family_map: dict[str, str] | None = None,
                   **kwargs) -> "MetaLearner":
        ml = cls(**kwargs)
        ml.bulk_update(_load_gate_observations(
            gate_path=gate_path, family_map=family_map,
        ))
        return ml

    # ── update API
    def update(self, family: str, outcome: str) -> None:
        """Append a single observation: outcome ∈ {pass, fail, yellow}."""
        if outcome not in ("pass", "fail", "yellow"):
            raise ValueError(f"outcome must be pass/fail/yellow, got {outcome!r}")
        c = self._counts.setdefault(family, {"pass": 0, "fail": 0, "yellow": 0})
        c[outcome] += 1

    def bulk_update(self, observations: Iterable[GateObservation]) -> None:
        for o in observations:
            self.update(o.family, o.outcome)

    # ── prior lookup
    def cold_start_prior(self, family: str) -> tuple[float, float]:
        return self.priors.get(family, self.priors["unknown"])

    # ── prediction
    def predict(self, family: str) -> FamilyPosterior:
        a0, b0 = self.cold_start_prior(family)
        c = self._counts.get(family, {"pass": 0, "fail": 0, "yellow": 0})
        s = c["pass"] + (c["yellow"] if self.yellow_counts_as_pass else 0)
        f = c["fail"] + (0 if self.yellow_counts_as_pass else c["yellow"])
        return FamilyPosterior(
            family=family,
            alpha=a0 + s,
            beta=b0 + f,
            n_observations=c["pass"] + c["fail"] + c["yellow"],
            prior_source=("cold_start" if (c["pass"] + c["fail"] + c["yellow"]) == 0
                            else "cold_start+data"),
            successes=c["pass"],
            failures=c["fail"],
            yellows=c["yellow"],
        )

    def compare(self, families: list[str]) -> list[FamilyPosterior]:
        """Sort by posterior mean (highest first)."""
        return sorted([self.predict(f) for f in families],
                       key=lambda p: -p.mean)

    # ── reporting
    def all_families(self) -> list[str]:
        """All families with observations OR cold-start priors."""
        return sorted(set(self.priors.keys()) | set(self._counts.keys()))

    def summary(self) -> list[dict]:
        return [self.predict(f).to_dict() for f in self.all_families()]

    def observed_families(self) -> list[str]:
        """Only families with at least one observation."""
        return sorted(self._counts.keys())


# ── Advisory annotation helpers ────────────────────────────────────────────

def annotate_candidate(
    family: str,
    *,
    ml: MetaLearner | None = None,
) -> dict:
    """Return advisory annotation for a candidate of given family.

    INTENT: Generator / Discovery callers attach this to candidate
    metadata so reviewers see the prior before running expensive gates.
    The annotation is NEVER fed back into a pass/fail decision —
    that's the strict gate's job.
    """
    ml = ml or MetaLearner.from_disk()
    post = ml.predict(family)
    lo, hi = post.credible_interval()
    return {
        "family":                family,
        "prior_pass_probability": round(post.mean, 4),
        "credible_interval_95":  [round(lo, 4), round(hi, 4)],
        "observations_in_family": post.n_observations,
        "prior_source":          post.prior_source,
        "advisory_note": (
            f"Family '{family}' has a {post.mean*100:.0f}% prior probability "
            f"of clearing the strict gate, based on "
            f"{post.n_observations} prior observations + cold-start prior. "
            f"This is ADVISORY only — the strict gate result remains "
            f"authoritative."
        ),
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 6.5 Tier 1 Meta-Learner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary",
                    help="Show all families with posterior P(pass) + CI")
    pp = sub.add_parser("predict",
                          help="Predict P(pass) for one family")
    pp.add_argument("family")

    cc = sub.add_parser("compare",
                          help="Rank multiple families")
    cc.add_argument("families", nargs="+")

    aa = sub.add_parser("annotate",
                          help="Advisory annotation for a candidate")
    aa.add_argument("family")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ml = MetaLearner.from_disk()
    if args.cmd == "summary":
        out = ml.summary()
    elif args.cmd == "predict":
        out = ml.predict(args.family).to_dict()
    elif args.cmd == "compare":
        out = [p.to_dict() for p in ml.compare(args.families)]
    elif args.cmd == "annotate":
        out = annotate_candidate(args.family, ml=ml)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
