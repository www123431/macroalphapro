"""engine.agents.book_monitor.runner — Phase 2.0 step 9c.

Glues pattern_rules + emit into a runnable pipeline:

  filter_events()  →  RULES.values()  →  dedup against prior signals
                                              ↓
                                  emit.doctrine_signal_detected
                                              ↓
                            Employee A's gatherer reads these on
                            next synthesis call (already wired)

The dedup is load-bearing: this runner is expected to fire daily
(scripts/run_book_monitor.py via cron). Without dedup, a persistent
3-RED cluster in MOMENTUM would emit a new `doctrine_signal_detected`
event every day for as long as the cluster persists, flooding
events.jsonl + double-counting in A's gatherer window.

Dedup rule: a fresh hit is one where NO `doctrine_signal_detected`
event exists in the last `dedup_window_days` (default 7) with the
same (rule_name, family-or-subject) tuple. This means:
  - the same MOMENTUM 3-RED cluster fires once a week max
  - if the cluster GROWS (4 → 5 REDs), severity may change but
    we still respect dedup (the runner doesn't re-fire just
    because metrics shifted). Operator can force re-fire by
    waiting out the window or running with force_emit=True.

Same fail-safe pattern as synthesis_runner: errors recorded in
result["errors"], never raised.
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Optional

from engine.agents.book_monitor.pattern_rules import (
    PatternHit, RULES,
)

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _cutoff_iso(days: int) -> str:
    return (_dt.datetime.utcnow() - _dt.timedelta(days=days)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


# ────────────────────────────────────────────────────────────────────
# Dedup
# ────────────────────────────────────────────────────────────────────
def _hit_dedup_key(h: PatternHit) -> str:
    """Stable key used to test whether a fresh hit is already covered
    by a recent doctrine_signal_detected event. Pattern-specific:
      - family_red_cluster keys on family
      - sleeve-level rules (future) key on subject_id
    Falls back to subject_id when family is None."""
    return f"{h.rule_name}::{h.family or h.subject_id}"


def _prior_signal_keys(*, dedup_window_days: int) -> set[str]:
    """Read recent doctrine_signal_detected events from store and
    return the set of dedup keys they cover."""
    from engine.research_store.store import filter_events
    since = _cutoff_iso(dedup_window_days)
    prior = filter_events(
        event_type="doctrine_signal_detected",
        since=since,
    )
    keys: set[str] = set()
    for ev in prior:
        pattern = ev.metrics.get("pattern_name") if ev.metrics else None
        fam = ev.family
        anchor = fam or ev.subject_id
        if pattern:
            keys.add(f"{pattern}::{anchor}")
    return keys


# ────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────
def run_book_monitor(
    *,
    events_window_days: int = 30,
    dedup_window_days:  int = 7,
    rule_kwargs:        Optional[dict[str, dict]] = None,
    dry_run:            bool = False,
    force_emit:         bool = False,
) -> dict:
    """Load recent events, run all registered rules, dedup, emit fresh
    signals.

    Args:
      events_window_days: recency window passed to rules (default 30).
      dedup_window_days:  window for checking prior signal emissions.
      rule_kwargs:        per-rule overrides, e.g.
                          {'family_red_cluster': {'threshold': 2}}.
      dry_run:            run rules + dedup but skip emit (preview).
      force_emit:         bypass dedup (use ONLY when intentionally
                          re-firing — e.g. operator pinned a signal
                          back to top of queue).

    Returns:
      {
        "run_ts":           iso,
        "dry_run":          bool,
        "n_events_scanned": int,
        "rules_run":        list[str],
        "hits":             [PatternHit-as-dict],
        "n_hits_total":     int,
        "n_hits_fresh":     int,   # post-dedup
        "n_emitted":        int,   # 0 on dry_run
        "event_ids":        list[str],
        "errors":           list[str],
      }
    """
    rule_kwargs = rule_kwargs or {}
    run_ts = _utc_iso()
    result: dict = {
        "run_ts":            run_ts,
        "dry_run":           dry_run,
        "n_events_scanned":  0,
        "rules_run":         list(RULES.keys()),
        "hits":              [],
        "n_hits_total":      0,
        "n_hits_fresh":      0,
        "n_emitted":         0,
        "event_ids":         [],
        "errors":            [],
    }

    # ── Load events ─────────────────────────────────────────────
    try:
        from engine.research_store.store import filter_events
        since = _cutoff_iso(events_window_days)
        events = filter_events(since=since)
        result["n_events_scanned"] = len(events)
    except Exception as exc:
        logger.exception("book_monitor: event load failed")
        result["errors"].append(f"load: {exc}")
        return result

    # ── Run each rule ───────────────────────────────────────────
    all_hits: list[PatternHit] = []
    for rule_name, check_fn in RULES.items():
        kwargs = rule_kwargs.get(rule_name, {})
        try:
            hits = check_fn(events, **kwargs)
            all_hits.extend(hits)
        except Exception as exc:
            logger.exception("book_monitor: rule %s failed", rule_name)
            result["errors"].append(f"rule:{rule_name}: {exc}")

    result["n_hits_total"] = len(all_hits)

    # ── Dedup ───────────────────────────────────────────────────
    fresh_hits: list[PatternHit]
    if force_emit:
        prior_keys: set[str] = set()
        fresh_hits = list(all_hits)
    else:
        try:
            prior_keys = _prior_signal_keys(dedup_window_days=dedup_window_days)
        except Exception as exc:
            logger.exception("book_monitor: prior signal read failed")
            result["errors"].append(f"dedup_read: {exc}")
            prior_keys = set()
        fresh_hits = [h for h in all_hits if _hit_dedup_key(h) not in prior_keys]

    result["n_hits_fresh"] = len(fresh_hits)

    # Serialize all hits for caller inspection (fresh + suppressed both
    # surfaced so UI can show "5 hits total, 2 fresh, 3 dedup-suppressed").
    result["hits"] = [
        {
            "rule_name":        h.rule_name,
            "subject_id":       h.subject_id,
            "severity":         h.severity,
            "summary":          h.summary,
            "metrics":          h.metrics,
            "family":           h.family,
            "parent_event_ids": list(h.parent_event_ids),
            "is_fresh":         _hit_dedup_key(h) not in prior_keys,
        }
        for h in all_hits
    ]

    if dry_run:
        return result

    # ── Emit fresh hits ─────────────────────────────────────────
    if not fresh_hits:
        return result

    try:
        from engine.research_store import emit
    except Exception as exc:
        result["errors"].append(f"emit_import: {exc}")
        return result

    for h in fresh_hits:
        try:
            eid = emit.doctrine_signal_detected(
                subject_id       = h.subject_id,
                pattern_name     = h.rule_name,
                metrics          = h.metrics,
                summary          = h.summary,
                severity         = h.severity,
                parent_event_ids = h.parent_event_ids,
                family           = h.family,
            )
            result["event_ids"].append(eid)
            result["n_emitted"] += 1
        except Exception as exc:
            logger.exception("book_monitor: emit failed for hit %s/%s",
                              h.rule_name, h.subject_id)
            result["errors"].append(
                f"emit:{h.rule_name}/{h.subject_id}: {exc}"
            )

    return result
