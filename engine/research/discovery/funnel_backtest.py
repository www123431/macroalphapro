"""engine/research/discovery/funnel_backtest.py — replay historical data
through the current filter stack to validate calibration.

Per user 2026-05-30: "明明旧有的我们还没尝试完, 这就已经能开始用来验证".
We have:
  - 9 library YAMLs (each with known status_in_our_book: DEPLOYED / RED /
    UNTESTED) — known ground truth labels
  - 83 gate_runs.jsonl entries with verdicts — known empirical outcomes
  - 71 graveyard entries (mostly RED)
  - 76+ discovery_log entries from v3 smoke
We can run THIS HISTORICAL CORPUS back through:
  - credibility_scorer (regex+venue features) → does it pass papers
    that became DEPLOYED?
  - confidence_calculator → does it score known-GREEN abstracts high?
  - family_thresholds → which library mechanisms would have been
    bumped by family bonus?
  - graveyard → are graveyard hits correctly catching family RED?
Output: a calibration report + per-decision diff vs ground truth.

NOT FOR PRODUCTION USE — this is empirical validation tooling
that should run ONCE on history to surface false positives /
false negatives in current thresholds, then again only when
thresholds are re-tuned.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"
GATE_RUNS = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"
DISCOVERY_LOG = REPO_ROOT / "data" / "research" / "discovery_log.jsonl"


@dataclasses.dataclass
class HistoricalCandidate:
    """A historical paper/mechanism with known outcome."""
    source:       str               # "library" | "gate_run" | "discovery_log"
    name:         str               # mechanism id / arxiv id / title slug
    title:        str
    abstract:     str
    family:       str
    ground_truth: str               # "DEPLOYED" | "GREEN" | "YELLOW" | "RED" | "UNKNOWN"
    metadata:     dict              # full original record for audit


# ── Loaders for each historical source ────────────────────────────────────

def _load_library_candidates() -> list[HistoricalCandidate]:
    """Library YAMLs → known DEPLOYED / RED / UNTESTED labels."""
    out = []
    if not LIBRARY_DIR.exists():
        return out
    for fp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            entry = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("library yaml %s parse failed: %s", fp, exc)
            continue
        title = str(entry.get("title") or entry.get("id") or fp.stem)
        # Library YAMLs don't have full abstracts; use mechanism_economics
        # as best-effort proxy text for the calculator
        abstract = str(entry.get("mechanism_economics") or "")
        family = str(entry.get("family") or "unknown")
        status = str(entry.get("status_in_our_book") or "UNKNOWN").upper()
        out.append(HistoricalCandidate(
            source="library",
            name=str(entry.get("id") or fp.stem),
            title=title,
            abstract=abstract,
            family=family,
            ground_truth=status,
            metadata={"path": str(fp.relative_to(REPO_ROOT))},
        ))
    return out


def _load_gate_run_candidates(*, limit: int | None = None) -> list[HistoricalCandidate]:
    """gate_runs.jsonl entries — empirical GREEN/YELLOW/RED outcomes."""
    if not GATE_RUNS.exists():
        return []
    out = []
    for line in GATE_RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = (rec.get("verdict") or "").strip().upper()
        if v.startswith("GREEN"):
            gt = "GREEN"
        elif v.startswith("YELLOW"):
            gt = "YELLOW"
        elif v.startswith("RED"):
            gt = "RED"
        else:
            gt = "UNKNOWN"
        out.append(HistoricalCandidate(
            source="gate_run",
            name=str(rec.get("mechanism") or rec.get("name") or ""),
            title=str(rec.get("mechanism") or "(gate-run)"),
            abstract=str(rec.get("name") or ""),    # gate_runs have no abstract; only name
            family=str(rec.get("family") or "unknown"),
            ground_truth=gt,
            metadata={
                "sharpe":         rec.get("standalone_sharpe"),
                "alpha_t":        rec.get("alpha_t_ff5umd"),
                "deflated_sr":    rec.get("deflated_sr"),
                "ts":             rec.get("ts"),
            },
        ))
    if limit:
        out = out[-limit:]
    return out


# ── Per-candidate evaluation ──────────────────────────────────────────────

@dataclasses.dataclass
class FunnelDecision:
    """What the current funnel would do with this candidate."""
    candidate:                  HistoricalCandidate
    credibility_score:          float | None
    credibility_passes:         bool | None
    confidence_score:           float | None
    confidence_features_hit:    list[str]
    family_routing:             str | None
    routing_adjusted:           float | None
    meta_learner_prior:         float | None
    final_disposition:          str       # review / borderline / skip
    agrees_with_truth:          bool      # heuristic match per source


def _evaluate_one(cand: HistoricalCandidate) -> FunnelDecision:
    """Run a single candidate through credibility + confidence + family +
    meta-learner. Determine if this matches its known ground truth."""
    from engine.research.discovery.credibility_scorer import (
        PaperMetadata, score_paper,
    )
    from engine.research.discovery.confidence_calculator import compute_confidence
    from engine.research.discovery.family_thresholds import explain_routing
    from engine.research.meta_learner import MetaLearner

    # 1. Credibility (regex+venue, may be neutral without venue tag)
    cred = score_paper(PaperMetadata(
        title=cand.title, abstract=cand.abstract, authors="", venue="",
    ))

    # 2. Deterministic confidence
    det = compute_confidence(
        cand.title, cand.abstract, family_guess=cand.family,
    )

    # 3. Family-aware routing
    routing = explain_routing(det.confidence, cand.family)

    # 4. Meta-learner prior
    try:
        ml = MetaLearner.from_disk()
        prior_posterior = ml.predict(cand.family or "unknown")
        prior = prior_posterior.mean
    except Exception:
        prior = None

    # Final disposition reflects what the funnel would do TODAY
    disp = routing["routing"]

    # Heuristic ground-truth agreement check per source:
    # - library: DEPLOYED + funnel chose review = agree;
    #            RED + funnel chose skip = agree;
    #            mixed = disagree
    # - gate_run: GREEN should have been routed review; RED should
    #             have been routed skip (NOT borderline) for ideal
    #             calibration, but borderline is acceptable
    if cand.source == "library":
        if cand.ground_truth in ("DEPLOYED", "GREEN"):
            agrees = disp == "review"
        elif cand.ground_truth == "RED":
            agrees = disp in ("skip", "borderline")
        else:
            agrees = True   # UNKNOWN ground truth → don't penalize
    elif cand.source == "gate_run":
        if cand.ground_truth == "GREEN":
            agrees = disp in ("review", "borderline")
        elif cand.ground_truth == "RED":
            agrees = disp in ("skip", "borderline")
        else:
            agrees = True
    else:
        agrees = True

    return FunnelDecision(
        candidate=cand,
        credibility_score=cred.score,
        credibility_passes=cred.passes_filter,
        confidence_score=det.confidence,
        confidence_features_hit=det.positives_hit,
        family_routing=cand.family,
        routing_adjusted=routing["adjusted_confidence"],
        meta_learner_prior=prior,
        final_disposition=disp,
        agrees_with_truth=agrees,
    )


# ── Public report API ─────────────────────────────────────────────────────

def run_backtest(
    *,
    include_library: bool = True,
    include_gate_runs: bool = True,
    gate_runs_limit: int | None = None,
) -> dict:
    """Replay historical corpus through current funnel + produce
    calibration report."""
    candidates: list[HistoricalCandidate] = []
    if include_library:
        candidates.extend(_load_library_candidates())
    if include_gate_runs:
        candidates.extend(_load_gate_run_candidates(limit=gate_runs_limit))

    decisions = [_evaluate_one(c) for c in candidates]

    # Per-source / per-ground-truth aggregation
    by_source = Counter(d.candidate.source for d in decisions)
    by_truth = Counter(d.candidate.ground_truth for d in decisions)
    by_disposition = Counter(d.final_disposition for d in decisions)

    # Confusion matrix-like view: ground truth × disposition
    confusion: dict[tuple[str, str], int] = {}
    for d in decisions:
        key = (d.candidate.ground_truth, d.final_disposition)
        confusion[key] = confusion.get(key, 0) + 1

    # Agreement rate
    agree = sum(1 for d in decisions if d.agrees_with_truth)

    # False-positive (RED labeled, funnel chose review) and
    # false-negative (GREEN labeled, funnel chose skip) lists
    false_pos = [d for d in decisions
                    if d.candidate.ground_truth == "RED"
                    and d.final_disposition == "review"]
    false_neg = [d for d in decisions
                    if d.candidate.ground_truth in ("GREEN", "DEPLOYED")
                    and d.final_disposition == "skip"]

    return {
        "total":            len(decisions),
        "by_source":        dict(by_source),
        "by_ground_truth":  dict(by_truth),
        "by_disposition":   dict(by_disposition),
        "agreement_rate":   round(agree / max(len(decisions), 1), 4),
        "confusion": {
            f"{gt}→{disp}": n
            for (gt, disp), n in sorted(confusion.items())
        },
        "false_positives": [
            {"name": d.candidate.name, "family": d.candidate.family,
              "confidence": round(d.confidence_score or 0, 3),
              "adjusted_confidence": round(d.routing_adjusted or 0, 3)}
            for d in false_pos
        ],
        "false_negatives": [
            {"name": d.candidate.name, "family": d.candidate.family,
              "ground_truth": d.candidate.ground_truth,
              "confidence": round(d.confidence_score or 0, 3)}
            for d in false_neg
        ],
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-library", action="store_true")
    parser.add_argument("--no-gate-runs", action="store_true")
    parser.add_argument("--gate-runs-limit", type=int, default=None)
    parser.add_argument("--format", choices=["json", "human"], default="human")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_backtest(
        include_library=not args.no_library,
        include_gate_runs=not args.no_gate_runs,
        gate_runs_limit=args.gate_runs_limit,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, default=str))
        return
    # Human-friendly
    print("=" * 64)
    print("FUNNEL BACK-TEST — historical corpus calibration check")
    print("=" * 64)
    print(f"Total candidates: {report['total']}")
    print(f"By source:        {report['by_source']}")
    print(f"By ground truth:  {report['by_ground_truth']}")
    print(f"By disposition:   {report['by_disposition']}")
    print(f"Agreement rate:   {report['agreement_rate']:.1%}")
    print()
    print("Confusion (ground_truth -> disposition):")
    for k, n in report["confusion"].items():
        print(f"  {k:<32} {n}")
    print()
    print(f"FALSE POSITIVES (RED candidate funnel would review): {len(report['false_positives'])}")
    for fp in report["false_positives"][:5]:
        print(f"  - {fp['name']:<40} fam={fp['family']:<15} adj={fp['adjusted_confidence']}")
    print()
    print(f"FALSE NEGATIVES (GREEN/DEPLOYED candidate funnel would skip): {len(report['false_negatives'])}")
    for fn in report["false_negatives"][:5]:
        print(f"  - {fn['name']:<40} fam={fn['family']:<15} conf={fn['confidence']}")


if __name__ == "__main__":
    _cli()
