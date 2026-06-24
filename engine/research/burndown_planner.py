"""engine.research.burndown_planner — produce a DRY-RUN cron plan.

This is the "look before you leap" stage of burn-1. Plans are produced
WITHOUT dispatching. Each plan records:

  * the WeeklyUsage snapshot it saw
  * top-K ranked candidates (post family/global filter)
  * belief-1 predicted verdict distribution for each candidate (preview)
  * skipped-reason breakdown (FAMILY_CAP_HIT / GLOBAL_CAP_HIT /
    ALREADY_DISPATCHED / INELIGIBLE_STATE / NO_FAMILY)

burn-1b will turn plan execution on (via _enabled flag); until then,
plans are audit artifacts only.

Plans land at:
  data/cron_burndown/plans/<plan_id>.json   (one file per plan run)
  data/cron_burndown/plans/_index.jsonl     (append-only index for query)
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_DIR = _REPO_ROOT / "data" / "cron_burndown" / "plans"


@_dc.dataclass(frozen=True)
class CandidateWithPrediction:
    """A ranked candidate enriched with belief-1's predicted verdict dist."""
    hypothesis_id:           str
    family:                  str
    claim_short:             str
    age_days:                int
    rank_score:              float
    novelty_score:           float
    demand_score:            float
    recency_score:           float
    predicted_verdict_dist:  dict[str, float]
    predicted_load_bearing:  list[str]
    prediction_basis:        str

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


@_dc.dataclass(frozen=True)
class BurndownPlan:
    """The audit artifact a planning run produces."""
    plan_id:           str
    ts:                str
    target_k:          int
    actual_k:          int
    candidates:        list[CandidateWithPrediction]
    usage_before:      dict          # WeeklyUsage.to_dict()
    usage_summary:     str
    queue_size:        int           # eligible candidates before cap filter
    cap_status:        str           # human-readable cap snapshot
    skipped_counts:    dict[str, int]    # reason → count
    dry_run:           bool

    def to_dict(self) -> dict[str, Any]:
        d = _dc.asdict(self)
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d


# ── Diagnostic counters ────────────────────────────────────────────


def _walk_skipped_reasons(
    hyp_rows: list[dict],
    dispatched: set[str],
) -> dict[str, int]:
    """Count why candidates are excluded across the queue, for plan output."""
    counts = {
        "ALREADY_DISPATCHED":      0,
        "INELIGIBLE_STATE":        0,
        "NO_FAMILY":               0,
        "NON_FACTOR_FAMILY":       0,    # OTHER / ATTENTION / etc — not dispatchable
        "DOCTRINE_SIGNAL_META":    0,    # source:doctrine_signal-tagged meta claims
        "ENHANCE_CLASS":           0,    # addresses_decay_in non-null OR active_b_sleeve_scan
        "NON_PROPOSAL_TYPE":       0,    # factor_analysis / methodology / sleeve_improvement / unknown
    }
    from engine.research.burndown_ranker import (
        ELIGIBLE_REVIEW_STATES, DISPATCHABLE_FAMILIES,
        NON_FACTOR_TAG_PREFIXES,
    )
    from engine.research_store.hypothesis.classifier import classify_hypothesis_type
    for h in hyp_rows:
        hid = h.get("hypothesis_id")
        if hid and hid in dispatched:
            counts["ALREADY_DISPATCHED"] += 1
            continue
        if h.get("review_state") not in ELIGIBLE_REVIEW_STATES:
            counts["INELIGIBLE_STATE"] += 1
            continue
        fam = (h.get("mechanism_family") or "").upper()
        if not fam:
            counts["NO_FAMILY"] += 1
            continue
        if fam not in DISPATCHABLE_FAMILIES:
            counts["NON_FACTOR_FAMILY"] += 1
            continue
        hyp_tags = tuple(h.get("tags") or ())
        if any(any(t.startswith(p) for p in NON_FACTOR_TAG_PREFIXES) for t in hyp_tags):
            counts["DOCTRINE_SIGNAL_META"] += 1
            continue
        if h.get("addresses_decay_in"):
            counts["ENHANCE_CLASS"] += 1
            continue
        h_type = h.get("hypothesis_type")
        if h_type is None or h_type == "unknown":
            h_type = classify_hypothesis_type(h)
        if h_type != "factor_proposal":
            counts["NON_PROPOSAL_TYPE"] += 1
            continue
    return counts


# ── Public API ────────────────────────────────────────────────────


def plan(
    *,
    target_k:           int = 3,
    now:                Optional[_dt.datetime] = None,
    dry_run:            bool = True,
) -> BurndownPlan:
    """Build a burndown plan for today.

    The default `target_k=3` matches the user's stated daily cadence
    (3-5/day → 15-20/wk). The actual returned count may be smaller if
    caps bind or the queue is thin.
    """
    if now is None:
        now = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    from engine.research import burndown_caps, burndown_ranker
    from engine.research.belief import predict_verdict

    usage = burndown_caps.usage_last_7d(now=now)
    ranked = burndown_ranker.rank_candidates(
        top_k = target_k,
        now   = now,
        usage = usage,
    )

    # Diagnostic queue counts
    hyp_rows   = burndown_ranker.load_hypotheses()
    dispatched = burndown_ranker.load_dispatched_hypothesis_ids()
    skipped    = _walk_skipped_reasons(hyp_rows, dispatched)
    from engine.research_store.hypothesis.classifier import classify_hypothesis_type as _classify
    def _is_dispatchable_queue_row(h: dict) -> bool:
        if not h.get("hypothesis_id") or h["hypothesis_id"] in dispatched:
            return False
        if h.get("review_state") not in burndown_ranker.ELIGIBLE_REVIEW_STATES:
            return False
        if (h.get("mechanism_family") or "").upper() not in burndown_ranker.DISPATCHABLE_FAMILIES:
            return False
        hyp_tags = tuple(h.get("tags") or ())
        if any(any(t.startswith(p) for p in burndown_ranker.NON_FACTOR_TAG_PREFIXES) for t in hyp_tags):
            return False
        if h.get("addresses_decay_in"):
            return False
        h_type = h.get("hypothesis_type")
        if h_type is None or h_type == "unknown":
            h_type = _classify(h)
        if h_type != "factor_proposal":
            return False
        return True
    queue_size = sum(1 for h in hyp_rows if _is_dispatchable_queue_row(h))

    # Cap status snapshot for the plan summary
    cap_lines = []
    for fam in sorted(burndown_caps.WATCHED_FAMILIES):
        left = burndown_caps.family_capacity_left(fam, usage)
        used = usage.by_family.get(fam, 0)
        cap_lines.append(f"  {fam}: {used}/{burndown_caps.FAMILY_WEEKLY_CAP} (left {left})")
    cap_status = (
        f"Global: {usage.global_count}/{burndown_caps.WEEKLY_GLOBAL_SOFT_CAP} "
        f"(hard {burndown_caps.WEEKLY_GLOBAL_HARD_CAP})\n"
        + "\n".join(cap_lines)
    )

    # belief-1 preview per candidate — same code path the dispatcher will
    # take, but we do NOT log_prediction here (no dispatch yet).
    enriched: list[CandidateWithPrediction] = []
    for rc in ranked:
        pred = predict_verdict(
            subject_id   = rc.hypothesis_id,
            family       = rc.family,
            paper_year   = None,  # paper metadata not threaded yet
            signal_kind  = None,
            extra_inputs = {"plan_preview": True},
        )
        enriched.append(CandidateWithPrediction(
            hypothesis_id          = rc.hypothesis_id,
            family                 = rc.family,
            claim_short            = rc.claim_short,
            age_days               = rc.age_days,
            rank_score             = rc.rank_score,
            novelty_score          = rc.novelty_score,
            demand_score           = rc.demand_score,
            recency_score          = rc.recency_score,
            predicted_verdict_dist = dict(pred.predicted_verdict_dist),
            predicted_load_bearing = list(pred.predicted_load_bearing),
            prediction_basis       = pred.prediction_basis,
        ))

    return BurndownPlan(
        plan_id        = str(uuid.uuid4()),
        ts             = now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        target_k       = target_k,
        actual_k       = len(enriched),
        candidates     = enriched,
        usage_before   = usage.to_dict(),
        usage_summary  = burndown_caps.usage_summary(usage),
        queue_size     = queue_size,
        cap_status     = cap_status,
        skipped_counts = skipped,
        dry_run        = dry_run,
    )


def write_plan(
    plan_obj: BurndownPlan,
    *,
    out_dir: Optional[Path] = None,
) -> Path:
    """Persist a plan to disk. Returns the file path written."""
    d = out_dir or DEFAULT_PLAN_DIR
    d.mkdir(parents=True, exist_ok=True)
    date_str = plan_obj.ts[:10]
    out_path = d / f"{date_str}_{plan_obj.plan_id[:8]}.json"
    out_path.write_text(
        json.dumps(plan_obj.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Append index row
    index_path = d / "_index.jsonl"
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "plan_id":     plan_obj.plan_id,
            "ts":          plan_obj.ts,
            "target_k":    plan_obj.target_k,
            "actual_k":    plan_obj.actual_k,
            "queue_size":  plan_obj.queue_size,
            "dry_run":     plan_obj.dry_run,
            "file":        out_path.name,
        }, ensure_ascii=False) + "\n")
    logger.info("burndown plan written to %s", out_path)
    return out_path


def format_plan_human(p: BurndownPlan) -> str:
    """Pretty-print a plan for stdout / Inbox digest."""
    lines: list[str] = []
    lines.append(f"=== Burndown Plan {p.plan_id[:8]} @ {p.ts} ===")
    lines.append(f"dry_run={p.dry_run}  target_k={p.target_k}  actual_k={p.actual_k}")
    lines.append(f"queue size (eligible after dedup): {p.queue_size}")
    lines.append("")
    lines.append("--- Cap status (last 7d) ---")
    lines.append(p.usage_summary)
    lines.append("")
    if p.candidates:
        lines.append("--- Selected candidates ---")
        for i, c in enumerate(p.candidates, 1):
            green = c.predicted_verdict_dist.get("GREEN", 0.0)
            marg  = c.predicted_verdict_dist.get("MARGINAL", 0.0)
            red   = c.predicted_verdict_dist.get("RED", 0.0)
            lines.append(
                f"{i}. {c.hypothesis_id[:8]}  [{c.family}]  "
                f"score={c.rank_score:.3f}  age={c.age_days}d"
            )
            lines.append(
                f"   predict: GREEN={green:.2f} / MARG={marg:.2f} / RED={red:.2f}"
                + (f"  load-bearing={c.predicted_load_bearing}" if c.predicted_load_bearing else "")
            )
            lines.append(f"   claim: {c.claim_short[:140]}")
    else:
        lines.append("--- No candidates selected ---")
        lines.append("(check skipped_counts + cap status above)")
    lines.append("")
    lines.append("--- Skipped queue rows ---")
    for k, v in sorted(p.skipped_counts.items()):
        if v:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)
