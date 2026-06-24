"""tests/test_book_monitor_pattern_rules.py — Phase 2.0 step 9b.

Pure-function tests for the pattern rules. No store I/O — synthetic
ResearchEvent fixtures cover the corner cases.

Covers `family_red_cluster`:
  - No events / no RED events / only out-of-window RED events  → []
  - Exactly threshold REDs in window → 1 hit (WARN)
  - 2*threshold REDs in window → 1 hit (CRITICAL)
  - Two families both clustering → 2 hits
  - REDs spanning multiple families don't cross-pollinate
  - subject_id = most recent RED's subject
  - parent_event_ids = all cluster members
  - GREEN / MARGINAL verdicts ignored
  - Non-factor_verdict_filed events ignored
  - Events with no family skipped (can't cluster)
"""
from __future__ import annotations

import datetime as _dt

from engine.agents.book_monitor.pattern_rules import (
    check_family_red_cluster,
    PatternHit,
    RULES,
)
from engine.research_store.schema import (
    EventType, ResearchEvent, SubjectType, Verdict,
)


# ─────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────
_NOW = "2026-06-06T12:00:00Z"


def _ts(days_ago: int, base: str = _NOW) -> str:
    base_dt = _dt.datetime.fromisoformat(base.replace("Z", ""))
    return (base_dt - _dt.timedelta(days=days_ago)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _verdict_event(
    *,
    subject_id: str,
    family:     str | None,
    verdict:    Verdict,
    days_ago:   int,
    event_id:   str | None = None,
    event_type: EventType = EventType.factor_verdict_filed,
) -> ResearchEvent:
    return ResearchEvent(
        event_id     = event_id or f"ev_{subject_id}_{days_ago}",
        event_type   = event_type,
        ts           = _ts(days_ago),
        session_id   = "test",
        actor        = "test",
        subject_type = SubjectType.factor,
        subject_id   = subject_id,
        verdict      = verdict,
        metrics      = {},
        artifacts    = {},
        parent_event_ids = (),
        family       = family,
        tags         = (),
        summary      = "test",
        git_sha      = "test",
    )


# ─────────────────────────────────────────────────────────────────────
# Empty / degenerate inputs
# ─────────────────────────────────────────────────────────────────────
def test_empty_input_returns_no_hits():
    assert check_family_red_cluster([], now=_NOW) == []


def test_only_green_events_returns_no_hits():
    """A cluster requires REDs; GREENs don't count."""
    events = [
        _verdict_event(subject_id=f"f{i}", family="EARNINGS_DRIFT",
                        verdict=Verdict.GREEN, days_ago=i)
        for i in range(5)
    ]
    assert check_family_red_cluster(events, now=_NOW) == []


def test_only_marginal_events_returns_no_hits():
    events = [
        _verdict_event(subject_id=f"f{i}", family="MOMENTUM",
                        verdict=Verdict.MARGINAL, days_ago=i)
        for i in range(5)
    ]
    assert check_family_red_cluster(events, now=_NOW) == []


def test_only_out_of_window_reds_returns_no_hits():
    """3 REDs but all > 30 days old → no cluster."""
    events = [
        _verdict_event(subject_id=f"f{i}", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=60 + i)
        for i in range(5)
    ]
    assert check_family_red_cluster(events, now=_NOW) == []


def test_non_factor_verdict_events_ignored():
    """E.g. decay_alert RED doesn't count toward factor-RED clusters."""
    events = [
        _verdict_event(subject_id=f"f{i}", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=i,
                        event_type=EventType.decay_alert)
        for i in range(5)
    ]
    assert check_family_red_cluster(events, now=_NOW) == []


def test_red_events_without_family_skipped():
    """Can't cluster what we can't group — events missing family
    must be skipped, not crash."""
    events = [
        _verdict_event(subject_id=f"f{i}", family=None,
                        verdict=Verdict.RED, days_ago=i)
        for i in range(5)
    ]
    assert check_family_red_cluster(events, now=_NOW) == []


# ─────────────────────────────────────────────────────────────────────
# Below / at / above threshold
# ─────────────────────────────────────────────────────────────────────
def test_two_reds_below_threshold_no_hit():
    events = [
        _verdict_event(subject_id=f"f{i}", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=i)
        for i in range(2)
    ]
    assert check_family_red_cluster(events, now=_NOW) == []


def test_three_reds_emits_warn_hit():
    events = [
        _verdict_event(subject_id=f"red_{i}", family="EARNINGS_DRIFT",
                        verdict=Verdict.RED, days_ago=i)
        for i in range(3)
    ]
    hits = check_family_red_cluster(events, now=_NOW)
    assert len(hits) == 1
    h = hits[0]
    assert h.rule_name == "family_red_cluster"
    assert h.severity == "WARN"
    assert h.family == "EARNINGS_DRIFT"
    assert h.metrics["red_count"] == 3
    assert h.metrics["window_days"] == 30
    assert h.metrics["threshold"] == 3


def test_six_reds_emits_critical_hit():
    """6 REDs = 2*threshold → CRITICAL."""
    events = [
        _verdict_event(subject_id=f"red_{i}", family="LOW_VOL",
                        verdict=Verdict.RED, days_ago=i)
        for i in range(6)
    ]
    hits = check_family_red_cluster(events, now=_NOW)
    assert len(hits) == 1
    assert hits[0].severity == "CRITICAL"
    assert hits[0].metrics["red_count"] == 6


def test_threshold_kwarg_overrides_default():
    """Caller can require fewer REDs (e.g. threshold=2 for a stricter
    monitor in audit mode)."""
    events = [
        _verdict_event(subject_id=f"red_{i}", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=i)
        for i in range(2)
    ]
    hits = check_family_red_cluster(events, now=_NOW, threshold=2)
    assert len(hits) == 1
    assert hits[0].metrics["threshold"] == 2


def test_window_days_kwarg_excludes_older_events():
    """7-day window should exclude REDs from day 20."""
    events = [
        _verdict_event(subject_id=f"red_{i}", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=i)
        for i in range(3)
    ] + [
        _verdict_event(subject_id=f"old_{i}", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=20 + i)
        for i in range(3)
    ]
    hits_30 = check_family_red_cluster(events, now=_NOW, window_days=30)
    hits_7  = check_family_red_cluster(events, now=_NOW, window_days=7)
    # 30-day window sees all 6 → CRITICAL
    assert hits_30[0].metrics["red_count"] == 6
    # 7-day window sees only 3 → WARN
    assert hits_7[0].metrics["red_count"] == 3
    assert hits_7[0].severity == "WARN"


# ─────────────────────────────────────────────────────────────────────
# Multi-family
# ─────────────────────────────────────────────────────────────────────
def test_two_families_both_clustering_returns_two_hits():
    events = (
        [_verdict_event(subject_id=f"ed_{i}", family="EARNINGS_DRIFT",
                         verdict=Verdict.RED, days_ago=i)
         for i in range(3)]
        +
        [_verdict_event(subject_id=f"mo_{i}", family="MOMENTUM",
                         verdict=Verdict.RED, days_ago=i+5)
         for i in range(3)]
    )
    hits = check_family_red_cluster(events, now=_NOW)
    assert len(hits) == 2
    families = {h.family for h in hits}
    assert families == {"EARNINGS_DRIFT", "MOMENTUM"}


def test_families_dont_cross_pollinate():
    """2 REDs in MOMENTUM + 2 REDs in CARRY — neither family hits
    threshold → no hits, even though TOTAL REDs is 4."""
    events = (
        [_verdict_event(subject_id=f"mo_{i}", family="MOMENTUM",
                         verdict=Verdict.RED, days_ago=i)
         for i in range(2)]
        +
        [_verdict_event(subject_id=f"ca_{i}", family="CARRY",
                         verdict=Verdict.RED, days_ago=i+3)
         for i in range(2)]
    )
    assert check_family_red_cluster(events, now=_NOW) == []


# ─────────────────────────────────────────────────────────────────────
# Hit payload details
# ─────────────────────────────────────────────────────────────────────
def test_subject_id_is_most_recent_red():
    """The representative subject should be the freshest RED — so the
    audit lineage anchors on the latest decision, not the oldest."""
    events = [
        _verdict_event(subject_id="oldest", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=10,
                        event_id="ev_old"),
        _verdict_event(subject_id="middle", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=5,
                        event_id="ev_mid"),
        _verdict_event(subject_id="freshest", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=1,
                        event_id="ev_new"),
    ]
    hits = check_family_red_cluster(events, now=_NOW)
    assert len(hits) == 1
    assert hits[0].subject_id == "freshest"
    assert hits[0].metrics["newest_red_ts"] > hits[0].metrics["oldest_red_ts"]


def test_parent_event_ids_carry_all_cluster_members():
    events = [
        _verdict_event(subject_id=f"f_{i}", family="MOMENTUM",
                        verdict=Verdict.RED, days_ago=i,
                        event_id=f"ev_{i}")
        for i in range(4)
    ]
    hits = check_family_red_cluster(events, now=_NOW)
    assert len(hits) == 1
    assert set(hits[0].parent_event_ids) == {"ev_0", "ev_1", "ev_2", "ev_3"}


def test_metrics_red_subject_ids_listed():
    """Audit consumers want the full list of REDs in the cluster —
    surfaced via metrics rather than tags so it's queryable JSON."""
    events = [
        _verdict_event(subject_id=f"factor_{i}", family="LOW_VOL",
                        verdict=Verdict.RED, days_ago=i)
        for i in range(3)
    ]
    hits = check_family_red_cluster(events, now=_NOW)
    assert set(hits[0].metrics["red_subject_ids"]) == {
        "factor_0", "factor_1", "factor_2"
    }


def test_summary_mentions_family_and_count():
    events = [
        _verdict_event(subject_id=f"x_{i}", family="EARNINGS_DRIFT",
                        verdict=Verdict.RED, days_ago=i)
        for i in range(4)
    ]
    s = check_family_red_cluster(events, now=_NOW)[0].summary
    assert "EARNINGS_DRIFT" in s
    assert "4" in s
    assert "30" in s


# ─────────────────────────────────────────────────────────────────────
# Registry — make sure RULES dict wires up
# ─────────────────────────────────────────────────────────────────────
def test_rule_registered_in_rules_dict():
    """The runner (step 9c) iterates RULES — the new rule MUST be
    registered for the runner to pick it up."""
    assert "family_red_cluster" in RULES
    assert RULES["family_red_cluster"] is check_family_red_cluster


# ─────────────────────────────────────────────────────────────────────
# Dedup regression tests — caught 2026-06-07 during manual close-loop
# of pending hid 47893a71. Autopilot was re-emitting the same RED
# verdict 3× (same subject_id, same source_hypothesis_id) and the
# cluster rule counted them as 3 distinct cluster members → phantom
# "PROFITABILITY family over-mined" signals → bogus 30d-pause
# proposals downstream. Cluster size should count UNIQUE failure
# cases, not raw event count.
# ─────────────────────────────────────────────────────────────────────
def _verdict_event_with_metrics(
    *, subject_id, family, days_ago, event_id, metrics,
):
    """Variant of _verdict_event that carries a metrics dict so we
    can test source_hypothesis_id dedup."""
    return ResearchEvent(
        event_id     = event_id,
        event_type   = EventType.factor_verdict_filed,
        ts           = _ts(days_ago),
        session_id   = "test",
        actor        = "test",
        subject_type = SubjectType.factor,
        subject_id   = subject_id,
        verdict      = Verdict.RED,
        metrics      = metrics,
        artifacts    = {},
        parent_event_ids = (),
        family       = family,
        tags         = (),
        summary      = "test",
        git_sha      = "test",
    )


def test_dedup_same_subject_3_emissions_does_not_trigger_cluster():
    """3 RED events with IDENTICAL subject_id (autopilot re-emit
    pattern) → dedup to 1 → below threshold → NO cluster signal."""
    events = [
        _verdict_event_with_metrics(
            subject_id="auto_aaa111", family="PROFITABILITY",
            days_ago=i, event_id=f"ev_{i}",
            metrics={"source_hypothesis_id": "hyp_X",
                      "deflated_sr": 0.16},
        )
        for i in range(5)   # 5 emissions of same subject/hyp
    ]
    hits = check_family_red_cluster(events, now=_NOW)
    assert hits == [], (
        "5 duplicate emissions of one subject MUST dedup to 1 → "
        "no cluster (regression from 2026-06-07 manual close-loop)"
    )


def test_dedup_by_subject_id_when_source_hyp_missing():
    """Older events may lack metrics.source_hypothesis_id — dedup
    must still happen by subject_id alone."""
    events = [
        _verdict_event_with_metrics(
            subject_id="auto_bbb", family="MOMENTUM",
            days_ago=i, event_id=f"ev_{i}",
            metrics={},   # no source_hypothesis_id
        )
        for i in range(4)
    ]
    assert check_family_red_cluster(events, now=_NOW) == []


def test_three_distinct_subjects_still_cluster():
    """Genuine cluster (3 different subjects, same family) → still
    triggers signal. Dedup must NOT over-suppress real clusters."""
    events = [
        _verdict_event_with_metrics(
            subject_id=f"auto_distinct_{i}", family="MOMENTUM",
            days_ago=i, event_id=f"ev_{i}",
            metrics={"source_hypothesis_id": f"hyp_{i}",
                      "deflated_sr": 0.2},
        )
        for i in range(3)
    ]
    hits = check_family_red_cluster(events, now=_NOW)
    assert len(hits) == 1
    assert hits[0].metrics["red_count"] == 3
    assert set(hits[0].metrics["red_subject_ids"]) == {
        "auto_distinct_0", "auto_distinct_1", "auto_distinct_2",
    }


def test_mixed_dedup_unique_subjects_counted_correctly():
    """Mix: subject A emitted 3×, subject B emitted 1×, subject C
    emitted 2× — all in MOMENTUM family. Dedup → 3 unique → 1 cluster
    with red_count=3."""
    events = []
    for sid, n_emissions, hyp in [
        ("auto_A", 3, "hyp_A"),
        ("auto_B", 1, "hyp_B"),
        ("auto_C", 2, "hyp_C"),
    ]:
        for i in range(n_emissions):
            events.append(_verdict_event_with_metrics(
                subject_id=sid, family="MOMENTUM",
                days_ago=i, event_id=f"ev_{sid}_{i}",
                metrics={"source_hypothesis_id": hyp},
            ))
    hits = check_family_red_cluster(events, now=_NOW)
    assert len(hits) == 1
    assert hits[0].metrics["red_count"] == 3   # 3 unique subjects
    assert set(hits[0].metrics["red_subject_ids"]) == {
        "auto_A", "auto_B", "auto_C",
    }


def test_other_family_skipped_from_cluster_detection():
    """family=OTHER is the catch-all (e2e test artifacts, ad-hoc
    smoke runs, etc.) — not a meaningful 'family over-mining'
    signal. 3 distinct subjects in OTHER → NO cluster signal.
    Caught in 2026-06-07 live verification."""
    events = [
        _verdict_event_with_metrics(
            subject_id=f"auto_other_{i}", family="OTHER",
            days_ago=i, event_id=f"ev_{i}",
            metrics={"source_hypothesis_id": f"hyp_{i}"},
        )
        for i in range(5)   # 5 distinct subjects, still no cluster
    ]
    assert check_family_red_cluster(events, now=_NOW) == []


def test_invalid_family_value_skipped():
    """Legacy events with family strings outside MechanismFamily
    enum (e.g. lowercase 'macro' from older schema) MUST NOT
    generate cluster signals. Today's data should be enum-clean;
    this filter prevents legacy garbage firing alerts."""
    events = [
        _verdict_event_with_metrics(
            subject_id=f"auto_{i}", family="macro",   # lowercase
            days_ago=i, event_id=f"ev_{i}",
            metrics={"source_hypothesis_id": f"hyp_{i}"},
        )
        for i in range(4)
    ]
    assert check_family_red_cluster(events, now=_NOW) == []


def test_valid_canonical_family_still_clusters():
    """Sanity: a VALID family (in MechanismFamily enum) still
    triggers cluster detection. Filter doesn't over-suppress."""
    events = [
        _verdict_event_with_metrics(
            subject_id=f"auto_{i}", family="EARNINGS_DRIFT",
            days_ago=i, event_id=f"ev_{i}",
            metrics={"source_hypothesis_id": f"hyp_{i}"},
        )
        for i in range(3)
    ]
    hits = check_family_red_cluster(events, now=_NOW)
    assert len(hits) == 1
    assert hits[0].family == "EARNINGS_DRIFT"


def test_dedup_keeps_newest_event_per_subject():
    """When dedup'd, the representative metrics should be from the
    NEWEST event (so the cluster summary reflects the latest stats)."""
    events = [
        # subject_X emitted 3 days ago with deflated_sr=0.50
        _verdict_event_with_metrics(
            subject_id="auto_X", family="VALUE",
            days_ago=3, event_id="ev_old",
            metrics={"source_hypothesis_id": "hyp_X",
                      "deflated_sr": 0.50},
        ),
        # subject_X RE-emitted 1 day ago with deflated_sr=0.10 (worse)
        _verdict_event_with_metrics(
            subject_id="auto_X", family="VALUE",
            days_ago=1, event_id="ev_new",
            metrics={"source_hypothesis_id": "hyp_X",
                      "deflated_sr": 0.10},
        ),
        # Two more distinct subjects to clear threshold
        _verdict_event_with_metrics(
            subject_id="auto_Y", family="VALUE",
            days_ago=2, event_id="ev_y",
            metrics={"source_hypothesis_id": "hyp_Y"},
        ),
        _verdict_event_with_metrics(
            subject_id="auto_Z", family="VALUE",
            days_ago=0, event_id="ev_z",
            metrics={"source_hypothesis_id": "hyp_Z"},
        ),
    ]
    hits = check_family_red_cluster(events, now=_NOW)
    assert len(hits) == 1
    # red_count = 3 distinct subjects (X dedup'd)
    assert hits[0].metrics["red_count"] == 3
    # parent_event_ids should NOT include ev_old (dedup'd out)
    assert "ev_old" not in hits[0].parent_event_ids
    assert "ev_new" in hits[0].parent_event_ids
