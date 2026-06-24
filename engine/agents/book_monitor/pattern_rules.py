"""engine.agents.book_monitor.pattern_rules — Phase 2.0 step 9b.

Pattern detection rules for Employee D. Each rule is a PURE function:

  events: list[ResearchEvent]  →  list[PatternHit]

PatternHit is the rule's verdict — what subject + severity + summary
+ source events form the cluster. The runner (step 9c) consumes hits
and calls emit.doctrine_signal_detected for each.

Keeping rules pure functions (no I/O) means:
  - tests use synthetic event fixtures, NO live store dependency
  - dedup logic lives in the runner, not the rule (separation of
    concerns)
  - rules compose easily — runner just iterates RULES and concatenates
    hits

First canonical rule: family_red_cluster (≥3 RED verdicts in the same
mechanism family within a recency window). Rationale: Hou-Xue-Zhang
65% non-replication prior means RED verdicts are EXPECTED — but
clustered REDs in one family is a doctrine signal ("this family is
either over-mined or the regime broke"). Subsequent Employee A
synthesis runs should see the signal and prefer empty over more
candidates in that family.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
from collections import defaultdict
from typing import Iterable, Optional

from engine.research_store.schema import EventType, ResearchEvent, Verdict

logger = logging.getLogger(__name__)


def _valid_mechanism_family_values() -> frozenset[str]:
    """Lazy import — avoids circular dependency at module load.
    Returns the canonical enum values as a frozenset for fast lookup."""
    from engine.research_store.red_lessons.mechanism_families import (
        MechanismFamily,
    )
    return frozenset(m.value for m in MechanismFamily)


@_dc.dataclass(frozen=True)
class PatternHit:
    """One rule's finding. The runner converts each into a
    doctrine_signal_detected event.

    Fields:
      rule_name:        registered name in RULES dict (e.g. 'family_red_cluster')
      subject_id:       the thing being flagged (a representative factor or sleeve)
      severity:         INFO / WARN / CRITICAL — mapped to verdict by emit
      summary:          1-2 sentence human-readable; stored verbatim on event
      metrics:          rule-specific structured payload (family, counts, etc.)
      parent_event_ids: source events that triggered this hit (lineage)
      family:           family name when meaningful (filter helper)
    """
    rule_name:        str
    subject_id:       str
    severity:         str           # "INFO" / "WARN" / "CRITICAL"
    summary:          str
    metrics:          dict
    parent_event_ids: tuple[str, ...]
    family:           Optional[str] = None


# ────────────────────────────────────────────────────────────────────
# Time helpers
# ────────────────────────────────────────────────────────────────────
def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _cutoff_iso(window_days: int, now: Optional[str] = None) -> str:
    """Compute the ISO cutoff for a recency window relative to `now`."""
    base = (_dt.datetime.fromisoformat(now.replace("Z", ""))
            if now else _dt.datetime.utcnow())
    return (base - _dt.timedelta(days=window_days)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


# ────────────────────────────────────────────────────────────────────
# Rule: family_red_cluster
# ────────────────────────────────────────────────────────────────────
def check_family_red_cluster(
    events: Iterable[ResearchEvent],
    *,
    threshold:    int = 3,
    window_days:  int = 30,
    now:          Optional[str] = None,
) -> list[PatternHit]:
    """Flag any mechanism family with ≥ `threshold` RED factor verdicts
    in the last `window_days`.

    Severity:
      WARN     when threshold ≤ count < 2*threshold
      CRITICAL when count ≥ 2*threshold (overwhelming evidence the
               family is either over-mined or regime-broken)

    Subject convention: the most-recent RED's subject_id (a real
    registered factor); the cluster members live in
    metrics["red_subject_ids"]. parent_event_ids carry all member
    event_ids.

    `now` is overridable for deterministic testing (default utcnow).
    Time comparison is ISO-8601 string lexicographic, which is correct
    for the Z-suffixed UTC timestamps used by the store.
    """
    cutoff = _cutoff_iso(window_days, now=now)

    # Filter to RED factor verdicts in window
    in_window: list[ResearchEvent] = []
    for ev in events:
        if ev.event_type != EventType.factor_verdict_filed:
            continue
        if ev.verdict != Verdict.RED:
            continue
        if ev.ts < cutoff:
            continue
        in_window.append(ev)

    # DEDUP by (subject_id, source_hypothesis_id) — caught 2026-06-07
    # during manual close-loop of hid 47893a71: autopilot re-emitted
    # the same RED verdict 3× (same subject, same hyp), which made
    # the cluster threshold trigger phantom signals ("PROFITABILITY
    # family has 3 REDs" when reality is 1 RED triple-emitted).
    # Cluster size should count UNIQUE failure cases, not raw events.
    # Keep the NEWEST event per dedup-key (so the representative
    # carries the freshest metrics/summary).
    in_window.sort(key=lambda e: e.ts)
    deduped_by_key: dict[tuple[str, str], ResearchEvent] = {}
    for ev in in_window:
        # source_hypothesis_id, when present, is the canonical
        # identity (one hypothesis → one auto_<hash> subject). When
        # absent, fall back to subject_id alone.
        source_hyp = ""
        if isinstance(ev.metrics, dict):
            source_hyp = str(ev.metrics.get("source_hypothesis_id") or "")
        key = (ev.subject_id or "", source_hyp)
        deduped_by_key[key] = ev   # last write wins → newest (sorted asc)
    in_window_unique = list(deduped_by_key.values())

    # Group by family. Skip:
    #   - events with no family (can't cluster)
    #   - "OTHER" catch-all (not a meaningful "family over-mining"
    #     signal — surfaced as noise in 2026-06-07 live verification:
    #     20-subject "cluster" was mostly e2e/smoke test artifacts)
    #   - families not in the canonical MechanismFamily enum (e.g.
    #     lowercase "macro" from legacy data — older schema accepted
    #     free-form strings; today's data should be enum-clean. Skip
    #     for cluster detection so legacy garbage doesn't fire signals)
    valid_families = _valid_mechanism_family_values()
    by_family: dict[str, list[ResearchEvent]] = defaultdict(list)
    for ev in in_window_unique:
        fam = (ev.family or "").strip()
        if not fam:
            continue
        if fam == "OTHER":
            continue
        if fam not in valid_families:
            continue
        by_family[fam].append(ev)

    hits: list[PatternHit] = []
    for fam, members in by_family.items():
        if len(members) < threshold:
            continue
        # Sort newest-first so subject is the most-recent RED
        members.sort(key=lambda e: e.ts, reverse=True)
        representative = members[0]

        severity = "CRITICAL" if len(members) >= 2 * threshold else "WARN"

        hits.append(PatternHit(
            rule_name        = "family_red_cluster",
            subject_id       = representative.subject_id,
            severity         = severity,
            summary          = (
                f"{fam} family: {len(members)} RED verdicts in last "
                f"{window_days} days — investigate over-mining or regime break"
            ),
            metrics          = {
                "family":            fam,
                "red_count":         len(members),
                "window_days":       window_days,
                "threshold":         threshold,
                "red_subject_ids":   [e.subject_id for e in members],
                "newest_red_ts":     representative.ts,
                "oldest_red_ts":     members[-1].ts,
            },
            parent_event_ids = tuple(e.event_id for e in members),
            family           = fam,
        ))

    # Sort hits by family for deterministic test ordering
    hits.sort(key=lambda h: h.family or "")
    return hits


# ────────────────────────────────────────────────────────────────────
# Registry — runner iterates this
# ────────────────────────────────────────────────────────────────────
# Map rule_name → check function. Each check function signature is
# (events, *, **rule_kwargs) → list[PatternHit]. Adding a rule = one
# entry here + the function.
RULES = {
    "family_red_cluster": check_family_red_cluster,
}
