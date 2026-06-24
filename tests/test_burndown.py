"""burn-1a tests — caps + ranker + planner.

Covers:
  * caps: family/global counting from synthesized dispatch log
  * caps: refusal rows DON'T count; only successful template runs
  * caps: non-cron rows ignored entirely
  * caps: time window respected (>7d ago excluded)
  * ranker: skips already-dispatched
  * ranker: ineligible review_state filtered
  * ranker: missing family filtered
  * ranker: scoring ordering (novelty x demand x recency monotonic)
  * ranker: capacity filter respected when usage provided
  * planner: produces plan with predictions
  * planner: write + human format
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib

import pytest

from engine.research import burndown_caps, burndown_ranker, burndown_planner


# ── Cap counting ──────────────────────────────────────────────────


def _write_log(tmp_path, rows):
    p = tmp_path / "dispatch_log.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


_row_counter = [0]   # closure-style monotone counter, reset per call sequence

def _row(*, cron_run_id="cr-1", family="PROFITABILITY", success=True,
         ts="2026-06-11T09:00:00Z", spec_hash=None):
    """Synth a dispatch log row.

    spec_hash defaults to a per-call unique string so successive rows
    count as distinct trials under the spec_hash-dedup semantics
    introduced 2026-06-17 in burndown_caps. Tests that intentionally
    want collision can pass spec_hash="sh-1" (or any literal).
    """
    _row_counter[0] += 1
    sh = spec_hash if spec_hash is not None else f"sh-auto-{_row_counter[0]}"
    return {
        "dispatch_event_id": f"d-{family[:3]}-{_row_counter[0]}",
        "ts":                ts,
        "hypothesis_id":     f"h-{family[:3]}-{_row_counter[0]}",
        "spec_hash":         sh,
        "cron_run_id":       cron_run_id,
        "family_hint":       family,
        "refusal":           None if success else {"reason_code": "TEMPLATE_NOT_CERTIFIED"},
        "template_result":   {"verdict": "MARGINAL"} if success else None,
        "actor":             "engine.agents.strengthener.factor_dispatcher",
    }


def test_caps_counts_successful_cron_rows(tmp_path):
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    log = _write_log(tmp_path, [
        _row(family="PROFITABILITY", success=True, ts="2026-06-10T09:00:00Z"),
        _row(family="MOMENTUM",       success=True, ts="2026-06-09T09:00:00Z"),
        _row(family="PROFITABILITY", success=True, ts="2026-06-08T09:00:00Z"),
    ])
    usage = burndown_caps.usage_last_7d(log_path=log, now=now)
    assert usage.global_count == 3
    assert usage.by_family == {"PROFITABILITY": 2, "MOMENTUM": 1}


def test_caps_excludes_custom_code_required_verdict(tmp_path):
    # B.1 fix (2026-06-11): CUSTOM_CODE_REQUIRED / EXECUTION_ERROR /
    # PENDING_BUILD verdicts mean no real lens-stack ran; they must
    # NOT burn quota.
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    row_custom = {
        "dispatch_event_id": "d1", "ts": "2026-06-10T09:00:00Z",
        "hypothesis_id": "h1", "spec_hash": "sh1",
        "cron_run_id": "cr-1", "family_hint": "PROFITABILITY",
        "refusal": None,
        "template_result": {"verdict": "CUSTOM_CODE_REQUIRED"},
    }
    row_real = {
        "dispatch_event_id": "d2", "ts": "2026-06-10T10:00:00Z",
        "hypothesis_id": "h2", "spec_hash": "sh2",
        "cron_run_id": "cr-1", "family_hint": "PROFITABILITY",
        "refusal": None,
        "template_result": {"verdict": "MARGINAL"},
    }
    log = _write_log(tmp_path, [row_custom, row_real])
    usage = burndown_caps.usage_last_7d(log_path=log, now=now)
    assert usage.global_count == 1
    assert usage.by_family == {"PROFITABILITY": 1}


def test_caps_excludes_refusal_rows(tmp_path):
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    log = _write_log(tmp_path, [
        _row(family="PROFITABILITY", success=False, ts="2026-06-10T09:00:00Z"),
        _row(family="PROFITABILITY", success=True,  ts="2026-06-10T10:00:00Z"),
    ])
    usage = burndown_caps.usage_last_7d(log_path=log, now=now)
    assert usage.global_count == 1
    assert usage.by_family == {"PROFITABILITY": 1}


def test_caps_excludes_non_cron_rows(tmp_path):
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    log = _write_log(tmp_path, [
        _row(cron_run_id=None, family="PROFITABILITY", success=True, ts="2026-06-10T09:00:00Z"),
        _row(cron_run_id="cr-2", family="MOMENTUM",     success=True, ts="2026-06-10T09:30:00Z"),
    ])
    usage = burndown_caps.usage_last_7d(log_path=log, now=now)
    assert usage.global_count == 1
    assert usage.by_family == {"MOMENTUM": 1}


def test_caps_respects_7d_window(tmp_path):
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    log = _write_log(tmp_path, [
        _row(family="PROFITABILITY", success=True, ts="2026-06-10T09:00:00Z"),  # in
        _row(family="VALUE",          success=True, ts="2026-06-01T09:00:00Z"),  # >7d → out
    ])
    usage = burndown_caps.usage_last_7d(log_path=log, now=now)
    assert usage.global_count == 1
    assert usage.by_family == {"PROFITABILITY": 1}


def test_family_capacity_left_watched_vs_unwatched(tmp_path):
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    # Fill the family cap exactly — each row has a unique spec_hash so
    # all count under the 2026-06-17 dedup semantics.
    log = _write_log(tmp_path, [
        _row(family="PROFITABILITY", success=True,
              ts=f"2026-06-{10 - i:02d}T09:00:00Z")
        for i in range(burndown_caps.FAMILY_WEEKLY_CAP)
    ])
    usage = burndown_caps.usage_last_7d(log_path=log, now=now)
    assert burndown_caps.family_capacity_left("PROFITABILITY", usage) == 0
    # Unwatched family — always reports FAMILY_WEEKLY_CAP
    assert burndown_caps.family_capacity_left("SOMETHING_NEW", usage) == burndown_caps.FAMILY_WEEKLY_CAP


def test_global_caps(tmp_path):
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    rows = [
        _row(family="VOL", success=True, ts="2026-06-10T09:00:00Z")
        for _ in range(burndown_caps.WEEKLY_GLOBAL_SOFT_CAP)
    ]
    log = _write_log(tmp_path, rows)
    usage = burndown_caps.usage_last_7d(log_path=log, now=now)
    assert burndown_caps.global_capacity_left(usage) == 0
    ok, reason = burndown_caps.can_dispatch("VOL", usage)
    assert not ok
    assert "GLOBAL_SOFT_CAP_HIT" in reason


def test_caps_dedupes_same_spec_hash(tmp_path):
    # Bailey-LdP n_trials hygiene: 8 dispatches sharing the same
    # spec_hash count as 1 trial, not 8. Regression test for
    # GP/A audit follow-up 2026-06-17.
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    log = _write_log(tmp_path, [
        # Carr-2009-style cluster: 8 hyps from same paper → same spec
        _row(family="VOL_RISK_PREMIUM", spec_hash="vrp-carr-2009",
              ts="2026-06-10T09:00:00Z"),
        _row(family="VOL_RISK_PREMIUM", spec_hash="vrp-carr-2009",
              ts="2026-06-10T09:01:00Z"),
        _row(family="VOL_RISK_PREMIUM", spec_hash="vrp-carr-2009",
              ts="2026-06-10T09:02:00Z"),
        # Plus a distinct VRP spec — different hypothesis, should count
        _row(family="VOL_RISK_PREMIUM", spec_hash="vrp-bekaert-2010",
              ts="2026-06-10T10:00:00Z"),
    ])
    usage = burndown_caps.usage_last_7d(log_path=log, now=now)
    # 4 dispatches → 2 unique specs → 2 trials in this family
    assert usage.global_count == 2
    assert usage.by_family == {"VOL_RISK_PREMIUM": 2}


def test_caps_legacy_rows_without_spec_hash_count_individually(tmp_path):
    # Pre-2026-06-08 schema rows lack spec_hash; the legacy fallback
    # makes each one count via dispatch_event_id so historical totals
    # remain stable.
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    log = _write_log(tmp_path, [
        # Legacy schema: no spec_hash
        {"dispatch_event_id": "leg-1", "ts": "2026-06-10T09:00:00Z",
         "hypothesis_id": "h-1", "cron_run_id": "cr-1",
         "family_hint": "VALUE", "refusal": None,
         "template_result": {"verdict": "RED"}},
        {"dispatch_event_id": "leg-2", "ts": "2026-06-10T09:30:00Z",
         "hypothesis_id": "h-2", "cron_run_id": "cr-1",
         "family_hint": "VALUE", "refusal": None,
         "template_result": {"verdict": "RED"}},
    ])
    usage = burndown_caps.usage_last_7d(log_path=log, now=now)
    assert usage.global_count == 2
    assert usage.by_family == {"VALUE": 2}


# ── Ranker filtering ──────────────────────────────────────────────


def _write_hypotheses(tmp_path, rows):
    p = tmp_path / "hyp.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def _hyp(*, hid, family="PROFITABILITY", state="proposed", days_ago=10,
         claim="test claim"):
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    created = now - _dt.timedelta(days=days_ago)
    return {
        "hypothesis_id":   hid,
        "mechanism_family": family,
        "review_state":    state,
        "created_ts":      created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "claim":           claim,
    }


def test_ranker_skips_already_dispatched(tmp_path):
    hyp = _write_hypotheses(tmp_path, [
        _hyp(hid="h-a"),
        _hyp(hid="h-b"),
    ])
    log = _write_log(tmp_path, [
        _row(cron_run_id=None),
    ])
    # synth a dispatch row whose hypothesis_id matches h-a so it gets dedupped
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"hypothesis_id": "h-a", "ts": "2026-05-01T00:00:00Z"}) + "\n")

    gaps = tmp_path / "gaps.jsonl"
    gaps.touch()
    out = burndown_ranker.rank_candidates(
        top_k=5,
        hyp_path=hyp,
        dispatch_log_path=log,
        gaps_path=gaps,
        now=_dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc),
    )
    ids = [c.hypothesis_id for c in out]
    assert "h-a" not in ids
    assert "h-b" in ids


def test_ranker_filters_ineligible_state(tmp_path):
    hyp = _write_hypotheses(tmp_path, [
        _hyp(hid="h-a", state="proposed"),
        _hyp(hid="h-b", state="rejected"),
        _hyp(hid="h-c", state="superseded"),
    ])
    gaps = tmp_path / "gaps.jsonl"; gaps.touch()
    log = tmp_path / "dispatch_log.jsonl"; log.touch()
    out = burndown_ranker.rank_candidates(
        top_k=5,
        hyp_path=hyp, dispatch_log_path=log, gaps_path=gaps,
        now=_dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc),
    )
    ids = {c.hypothesis_id for c in out}
    assert ids == {"h-a"}


def test_ranker_skips_no_family(tmp_path):
    hyp = _write_hypotheses(tmp_path, [
        _hyp(hid="h-a", family="MOMENTUM"),
        _hyp(hid="h-b", family=""),
    ])
    gaps = tmp_path / "gaps.jsonl"; gaps.touch()
    log = tmp_path / "dispatch_log.jsonl"; log.touch()
    out = burndown_ranker.rank_candidates(
        top_k=5, hyp_path=hyp, dispatch_log_path=log, gaps_path=gaps,
        now=_dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc),
    )
    ids = {c.hypothesis_id for c in out}
    assert ids == {"h-a"}


def test_ranker_skips_non_dispatchable_families(tmp_path):
    # OTHER (meta-research), ATTENTION (thematic), HOLDINGS_BASED — none
    # of these have a confirmed template path, so cron must skip them.
    hyp = _write_hypotheses(tmp_path, [
        _hyp(hid="h-mom",   family="MOMENTUM"),
        _hyp(hid="h-other", family="OTHER"),
        _hyp(hid="h-att",   family="ATTENTION"),
        _hyp(hid="h-hold",  family="HOLDINGS_BASED"),
        _hyp(hid="h-val",   family="VALUE"),
    ])
    gaps = tmp_path / "gaps.jsonl"; gaps.touch()
    log  = tmp_path / "dispatch_log.jsonl"; log.touch()
    out = burndown_ranker.rank_candidates(
        top_k=10, hyp_path=hyp, dispatch_log_path=log, gaps_path=gaps,
        now=_dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc),
    )
    ids = {c.hypothesis_id for c in out}
    assert ids == {"h-mom", "h-val"}


def test_ranker_skips_enhance_class_hypotheses(tmp_path):
    """Phase 1 (2026-06-11): addresses_decay_in or active_b_sleeve_scan
    tag must exclude the row from forward cron — different statistical
    framework applies (paired bootstrap, not Bailey-LdP DSR)."""
    h_enhance_decay = _hyp(hid="h-decay", family="PROFITABILITY")
    h_enhance_decay["addresses_decay_in"] = "gp_at_2025"
    h_enhance_tag = _hyp(hid="h-scan", family="PROFITABILITY")
    h_enhance_tag["tags"] = ["source:active_b_sleeve_scan",
                                "sleeve:carry_g10_fx",
                                "improvement_kind:vol_target"]
    h_forward = _hyp(hid="h-real", family="PROFITABILITY")
    hyp = _write_hypotheses(tmp_path, [h_enhance_decay, h_enhance_tag, h_forward])
    gaps = tmp_path / "gaps.jsonl"; gaps.touch()
    log  = tmp_path / "dispatch_log.jsonl"; log.touch()
    out = burndown_ranker.rank_candidates(
        top_k=10, hyp_path=hyp, dispatch_log_path=log, gaps_path=gaps,
        now=_dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc),
    )
    ids = {c.hypothesis_id for c in out}
    assert ids == {"h-real"}


def test_ranker_skips_doctrine_signal_meta_claims(tmp_path):
    # A doctrine_signal-tagged hypothesis carries a valid family tag
    # (PROFITABILITY here) but is a META critique of that family, not a
    # new factor proposal. Must be skipped.
    h_meta = _hyp(hid="h-meta", family="PROFITABILITY")
    h_meta["tags"] = ["source:doctrine_signal", "pattern:family_red_cluster"]
    h_real = _hyp(hid="h-real", family="PROFITABILITY")
    h_real["tags"] = ["t3_llm_extraction"]
    hyp = _write_hypotheses(tmp_path, [h_meta, h_real])
    gaps = tmp_path / "gaps.jsonl"; gaps.touch()
    log  = tmp_path / "dispatch_log.jsonl"; log.touch()
    out = burndown_ranker.rank_candidates(
        top_k=10, hyp_path=hyp, dispatch_log_path=log, gaps_path=gaps,
        now=_dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc),
    )
    ids = {c.hypothesis_id for c in out}
    assert ids == {"h-real"}


def test_ranker_capacity_filter_applies(tmp_path, monkeypatch):
    hyp = _write_hypotheses(tmp_path, [
        _hyp(hid="h-a", family="PROFITABILITY"),
        _hyp(hid="h-b", family="MOMENTUM"),
        _hyp(hid="h-c", family="PROFITABILITY"),
    ])
    gaps = tmp_path / "gaps.jsonl"; gaps.touch()
    # Fill PROFITABILITY family cap with distinct-spec rows so the cap
    # binds under the 2026-06-17 spec_hash-dedup semantics.
    log_with_3prof = _write_log(tmp_path, [
        _row(family="PROFITABILITY", success=True,
              ts=f"2026-06-{10 - i:02d}T09:00:00Z")
        for i in range(burndown_caps.FAMILY_WEEKLY_CAP)
    ])
    now = _dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc)
    usage = burndown_caps.usage_last_7d(log_path=log_with_3prof, now=now)
    # PROFITABILITY at cap → only MOMENTUM hypothesis should survive
    out = burndown_ranker.rank_candidates(
        top_k=5, hyp_path=hyp,
        dispatch_log_path=log_with_3prof, gaps_path=gaps,
        now=now, usage=usage,
    )
    ids = {c.hypothesis_id for c in out}
    assert ids == {"h-b"}


def test_ranker_recency_ordering(tmp_path):
    # Two hypotheses, same family. Newer should score higher.
    hyp = _write_hypotheses(tmp_path, [
        _hyp(hid="h-new", family="MOMENTUM", days_ago=2),
        _hyp(hid="h-old", family="MOMENTUM", days_ago=365),
    ])
    gaps = tmp_path / "gaps.jsonl"; gaps.touch()
    log = tmp_path / "dispatch_log.jsonl"; log.touch()
    out = burndown_ranker.rank_candidates(
        top_k=2, hyp_path=hyp, dispatch_log_path=log, gaps_path=gaps,
        now=_dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc),
    )
    assert [c.hypothesis_id for c in out] == ["h-new", "h-old"]
    assert out[0].rank_score > out[1].rank_score


# ── Planner ───────────────────────────────────────────────────────


def test_planner_produces_predictions(tmp_path, monkeypatch):
    # Point all paths into tmp_path
    hyp = _write_hypotheses(tmp_path, [
        _hyp(hid="h-plan", family="MOMENTUM", days_ago=5),
    ])
    log = tmp_path / "dispatch_log.jsonl"; log.touch()
    gaps = tmp_path / "gaps.jsonl"; gaps.touch()

    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp)
    monkeypatch.setattr(burndown_ranker, "DEFAULT_DISPATCH_LOG", log)
    monkeypatch.setattr(burndown_ranker, "DEFAULT_GAPS_PATH", gaps)
    monkeypatch.setattr(burndown_caps, "DEFAULT_DISPATCH_LOG", log)

    p = burndown_planner.plan(target_k=2, dry_run=True,
                              now=_dt.datetime(2026, 6, 11, 12, 0,
                                               tzinfo=_dt.timezone.utc))
    assert p.target_k == 2
    assert p.actual_k == 1
    assert len(p.candidates) == 1
    cand = p.candidates[0]
    assert cand.hypothesis_id == "h-plan"
    assert set(cand.predicted_verdict_dist.keys()) == {"GREEN", "MARGINAL", "RED"}
    assert abs(sum(cand.predicted_verdict_dist.values()) - 1.0) < 1e-9
    assert "default prior" in cand.prediction_basis.lower() or \
           "observed family posterior" in cand.prediction_basis.lower() or \
           "family prior override" in cand.prediction_basis.lower()


def test_planner_write_and_format(tmp_path, monkeypatch):
    hyp = _write_hypotheses(tmp_path, [
        _hyp(hid="h-fmt", family="LOW_VOL", days_ago=1),
    ])
    log = tmp_path / "dispatch_log.jsonl"; log.touch()
    gaps = tmp_path / "gaps.jsonl"; gaps.touch()
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp)
    monkeypatch.setattr(burndown_ranker, "DEFAULT_DISPATCH_LOG", log)
    monkeypatch.setattr(burndown_ranker, "DEFAULT_GAPS_PATH", gaps)
    monkeypatch.setattr(burndown_caps, "DEFAULT_DISPATCH_LOG", log)

    p = burndown_planner.plan(target_k=3, dry_run=True,
                              now=_dt.datetime(2026, 6, 11, 12, 0,
                                               tzinfo=_dt.timezone.utc))
    out_dir = tmp_path / "plans"
    out_path = burndown_planner.write_plan(p, out_dir=out_dir)
    assert out_path.is_file()
    d = json.loads(out_path.read_text(encoding="utf-8"))
    assert d["plan_id"] == p.plan_id
    assert "candidates" in d

    # Index row appended
    idx = out_dir / "_index.jsonl"
    assert idx.is_file()
    idx_rows = [
        json.loads(ln) for ln in idx.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert any(r["plan_id"] == p.plan_id for r in idx_rows)

    text = burndown_planner.format_plan_human(p)
    assert "Burndown Plan" in text
    assert "LOW_VOL" in text or "Selected candidates" in text
