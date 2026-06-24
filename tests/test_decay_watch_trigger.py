"""tests/test_decay_watch_trigger.py — C of senior施工建议.

Locks the 3 decay-trigger criteria + severity mapping + the
research-auto-capital-human discipline (SUGGESTION not command).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def _windows(sharpes: list[float]) -> list[dict]:
    """Build a list-of-window dicts mimicking subsample_stability output."""
    return [
        {
            "start":      "2010-01",
            "end":        "2014-12",
            "n_months":   60,
            "sharpe_ann": s,
            "nw_t_stat":  s * 2.0,
            "ann_return": s * 0.04,
            "ann_vol":    0.04,
        } for s in sharpes
    ]


def _subsample_dict(
    *,
    sharpes:        list[float],
    worst_best:     float | None = None,
    monotone:       bool = False,
    n_splits:       int = 4,
    n_total_months: int = 240,
) -> dict:
    return {
        "n_splits":                n_splits,
        "n_total_months":          n_total_months,
        "windows":                 _windows(sharpes),
        "worst_best_sharpe_ratio": worst_best,
        "institutional_stable":    False,
        "monotone_decay":          monotone,
        "monotone_growth":         False,
        "decay_slope_per_year":    None,
        "decay_slope_t":           None,
    }


# ────────────────────────────────────────────────────────────────────
# Trigger A — worst/best collapse
# ────────────────────────────────────────────────────────────────────
def test_trigger_A_fires_below_bar():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[1.0, 0.5, 0.3, 0.1], worst_best=0.10)
    )
    assert "A" in out["triggers_hit"]


def test_trigger_A_does_not_fire_at_or_above_bar():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[1.0, 0.8, 0.6, 0.3], worst_best=0.30)
    )
    assert "A" not in out["triggers_hit"]


# ────────────────────────────────────────────────────────────────────
# Trigger B — monotone decay
# ────────────────────────────────────────────────────────────────────
def test_trigger_B_fires_on_strict_monotone():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[1.5, 1.0, 0.5, 0.1],
                          worst_best=0.10, monotone=True)
    )
    assert "B" in out["triggers_hit"]


def test_trigger_B_does_not_fire_when_not_monotone():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[1.0, 0.5, 0.7, 0.3],
                          worst_best=0.30, monotone=False)
    )
    assert "B" not in out["triggers_hit"]


# ────────────────────────────────────────────────────────────────────
# Trigger C — latest-vs-prior collapse
# ────────────────────────────────────────────────────────────────────
def test_trigger_C_fires_when_latest_half_of_prior():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    # latest 0.20 < 0.5 × 0.80 = 0.40
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[0.5, 0.7, 0.80, 0.20], worst_best=0.40)
    )
    assert "C" in out["triggers_hit"]
    assert out["latest_window_sharpe"] == pytest.approx(0.20)
    assert out["prior_window_sharpe"] == pytest.approx(0.80)


def test_trigger_C_does_not_fire_when_latest_close_to_prior():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[0.5, 0.6, 0.80, 0.65], worst_best=0.50)
    )
    assert "C" not in out["triggers_hit"]


def test_trigger_C_skipped_when_prior_is_non_positive():
    """Can't compute latest/prior ratio if prior <= 0 — undefined."""
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[0.5, 0.3, 0.0, 0.10], worst_best=0.20)
    )
    assert "C" not in out["triggers_hit"]
    assert out["latest_vs_prior_ratio"] is None


# ────────────────────────────────────────────────────────────────────
# Severity mapping
# ────────────────────────────────────────────────────────────────────
def test_severity_RED_when_3_of_3_fire():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[1.5, 1.0, 0.5, 0.1],
                          worst_best=0.07, monotone=True)
    )
    assert out["n_triggers"] == 3
    assert out["severity"] == "RED"


def test_severity_MARGINAL_when_2_of_3_fire():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    # Trigger A (worst/best=0.10 < 0.20)
    # Trigger C (latest 0.10 < 0.5 × 0.50 = 0.25)
    # Trigger B does NOT fire (not strictly monotone)
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[1.0, 0.7, 0.50, 0.10],
                          worst_best=0.10, monotone=False)
    )
    assert out["n_triggers"] == 2
    assert out["severity"] == "MARGINAL"


def test_severity_NEUTRAL_when_only_1_fires():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[1.0, 0.8, 0.7, 0.6],
                          worst_best=0.15, monotone=False)
    )
    assert out["n_triggers"] == 1
    assert out["severity"] == "NEUTRAL"


def test_severity_NEUTRAL_when_none_fire():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    out = evaluate_subsample_for_decay(
        _subsample_dict(sharpes=[1.0, 1.1, 1.0, 0.9],
                          worst_best=0.80, monotone=False)
    )
    assert out["n_triggers"] == 0
    assert out["severity"] == "NEUTRAL"


# ────────────────────────────────────────────────────────────────────
# Emit helper — research_store integration
# ────────────────────────────────────────────────────────────────────
def test_emit_skipped_when_no_trigger():
    """No trigger fires → no emit, returns None."""
    from engine.research import decay_watch_trigger as dwt
    out = dwt.emit_decay_alert_from_subsample(
        subject_id="any_sleeve",
        subsample_output=_subsample_dict(
            sharpes=[1.0, 1.1, 1.0, 0.9], worst_best=0.80,
        ),
    )
    assert out is None


def test_emit_skipped_when_only_1_trigger_default_threshold():
    """min_triggers_for_emit defaults to 2; single trigger logged
    only at NEUTRAL, no event emitted."""
    from engine.research import decay_watch_trigger as dwt
    out = dwt.emit_decay_alert_from_subsample(
        subject_id="any_sleeve",
        subsample_output=_subsample_dict(
            sharpes=[1.0, 0.8, 0.7, 0.6], worst_best=0.15,  # A only
        ),
    )
    assert out is None


def test_emit_calls_research_store_when_2_triggers():
    """2+ triggers → MARGINAL emit. Mock emit.decay_alert to confirm
    it's called with the right verdict + tags + summary."""
    from engine.research import decay_watch_trigger as dwt
    captured = {}
    def _fake_emit(*, subject_id, verdict, metrics, artifacts,
                      summary, parent_event_ids, tags, actor):
        captured.update(locals())
        return "fake_event_id_123"

    with patch("engine.research_store.emit.decay_alert", _fake_emit):
        out = dwt.emit_decay_alert_from_subsample(
            subject_id="cross_asset_carry",
            subsample_output=_subsample_dict(
                sharpes=[1.5, 1.0, 0.5, 0.1],
                worst_best=0.07, monotone=True,
            ),
            parent_event_ids=("verdict_event_abc",),
        )
    assert out == "fake_event_id_123"
    assert captured["subject_id"] == "cross_asset_carry"
    assert captured["verdict"] == "RED"          # 3-of-3
    assert "decay_watch" in captured["tags"]
    assert "review_recommended" in captured["tags"]
    assert captured["parent_event_ids"] == ("verdict_event_abc",)


def test_emit_summary_carries_SUGGESTION_not_command():
    """research-auto-capital-human discipline: summary MUST say
    'SUGGESTION' / 'review' — never 'reduce' or 'kill'."""
    from engine.research import decay_watch_trigger as dwt
    captured = {}
    def _fake_emit(*, subject_id, verdict, metrics, artifacts,
                      summary, parent_event_ids, tags, actor):
        captured["summary"] = summary
        return "eid"

    with patch("engine.research_store.emit.decay_alert", _fake_emit):
        dwt.emit_decay_alert_from_subsample(
            subject_id="cross_asset_carry",
            subsample_output=_subsample_dict(
                sharpes=[1.5, 1.0, 0.5, 0.1],
                worst_best=0.07, monotone=True,
            ),
        )
    s = captured["summary"]
    assert "SUGGESTION" in s
    assert "review" in s.lower()
    # Hard negative checks: must NOT contain capital-action commands
    for forbidden in ("reduce capital", "kill", "decommission",
                          "halt trading", "stop trading"):
        assert forbidden not in s.lower(), (
            f"summary contains command-tone phrase {forbidden!r}: {s!r}"
        )


def test_emit_metrics_payload_carries_windows_and_triggers():
    from engine.research import decay_watch_trigger as dwt
    captured = {}
    def _fake_emit(*, subject_id, verdict, metrics, artifacts,
                      summary, parent_event_ids, tags, actor):
        captured["metrics"] = metrics
        return "eid"
    with patch("engine.research_store.emit.decay_alert", _fake_emit):
        dwt.emit_decay_alert_from_subsample(
            subject_id="cross_asset_carry",
            subsample_output=_subsample_dict(
                sharpes=[1.5, 1.0, 0.5, 0.1],
                worst_best=0.07, monotone=True,
            ),
        )
    m = captured["metrics"]
    assert set(m["triggers_hit"]) == {"A", "B", "C"}
    assert m["severity"] == "RED"
    assert m["worst_best_sharpe_ratio"] == 0.07
    assert m["monotone_decay"] is True
    assert len(m["windows"]) == 4


# ────────────────────────────────────────────────────────────────────
# K — cron dedup semantics (should_emit_for_subject)
# ────────────────────────────────────────────────────────────────────
def _mk_event(event_id, tags, metrics=None, parents=()):
    from types import SimpleNamespace
    return SimpleNamespace(
        event_id=event_id, event_type="decay_alert",
        subject_id="dedup_sleeve", verdict="MARGINAL",
        ts="2026-06-10T05:30:00Z", summary="x",
        metrics=metrics or {}, tags=tags,
        parent_event_ids=parents, family=None, actor="test",
    )


def _eval(triggers, severity):
    return {"triggers_hit": triggers, "severity": severity,
              "n_triggers": len(triggers)}


def _patch_store_and_chain(monkeypatch, events):
    """Patch the store + ack-chain walker that should_emit_for_subject
    reads. The chain walker is the real one from api.main — events
    just need correct tags + parent_event_ids."""
    from engine.research_store import store
    monkeypatch.setattr(store, "filter_events",
                          lambda **kw: events)


def test_dedup_emits_when_no_prior_alert(monkeypatch):
    from engine.research.decay_watch_trigger import should_emit_for_subject
    _patch_store_and_chain(monkeypatch, [])
    ok, reason = should_emit_for_subject("dedup_sleeve",
                                            _eval(["A", "B"], "MARGINAL"))
    assert ok
    assert reason == "no_prior_alert"


def test_dedup_skips_when_open_alert_exists(monkeypatch):
    """An un-acked alert for the subject blocks re-emission —
    alert-fatigue protection. The #1 reason institutional alert
    systems get ignored is daily repeats of the same finding."""
    from engine.research.decay_watch_trigger import should_emit_for_subject
    original = _mk_event("orig", ("decay_watch", "review_recommended"),
                            metrics={"triggers_hit": ["A", "B"],
                                       "severity": "MARGINAL"})
    _patch_store_and_chain(monkeypatch, [original])
    ok, reason = should_emit_for_subject("dedup_sleeve",
                                            _eval(["A", "B"], "MARGINAL"))
    assert not ok
    assert reason == "open_alert_exists"


def test_dedup_skips_acked_same_signature(monkeypatch):
    """Acked alert + identical finding → skip. The principal already
    reviewed THIS exact fact; re-alerting disrespects the ack."""
    from engine.research.decay_watch_trigger import should_emit_for_subject
    original = _mk_event("orig", ("decay_watch", "review_recommended"),
                            metrics={"triggers_hit": ["A", "B"],
                                       "severity": "MARGINAL"})
    ack = _mk_event("ack1", ("decay_watch", "acknowledged"),
                       metrics={"action": "reviewed_no_action"},
                       parents=("orig",))
    _patch_store_and_chain(monkeypatch, [ack, original])
    ok, reason = should_emit_for_subject("dedup_sleeve",
                                            _eval(["A", "B"], "MARGINAL"))
    assert not ok
    assert reason == "acked_same_signature"


def test_dedup_emits_on_signature_escalation(monkeypatch):
    """Acked alert but the finding ESCALATED (A,B MARGINAL →
    A,B,C RED). Past ack doesn't cover new evidence — emit."""
    from engine.research.decay_watch_trigger import should_emit_for_subject
    original = _mk_event("orig", ("decay_watch", "review_recommended"),
                            metrics={"triggers_hit": ["A", "B"],
                                       "severity": "MARGINAL"})
    ack = _mk_event("ack1", ("decay_watch", "acknowledged"),
                       metrics={"action": "reviewed_no_action"},
                       parents=("orig",))
    _patch_store_and_chain(monkeypatch, [ack, original])
    ok, reason = should_emit_for_subject("dedup_sleeve",
                                            _eval(["A", "B", "C"], "RED"))
    assert ok
    assert reason.startswith("signature_changed:")


def test_dedup_emit_helper_integration(monkeypatch):
    """emit_decay_alert_from_subsample(dedup=True) returns None when
    the dedup gate says skip — and does NOT call emit."""
    from engine.research import decay_watch_trigger as dwt
    monkeypatch.setattr(dwt, "should_emit_for_subject",
                          lambda s, ev: (False, "open_alert_exists"))
    called = []
    import engine.research_store.emit as emit_mod
    monkeypatch.setattr(emit_mod, "decay_alert",
                          lambda **kw: called.append(kw) or "eid")
    out = dwt.emit_decay_alert_from_subsample(
        subject_id="dedup_sleeve",
        subsample_output=_subsample_dict(
            sharpes=[1.5, 1.0, 0.5, 0.1],
            worst_best=0.07, monotone=True,
        ),
        dedup=True,
    )
    assert out is None
    assert called == []


def test_dedup_default_off_preserves_backfill_behavior(monkeypatch):
    """dedup defaults to False — the original backfill semantics
    (always emit when triggers fire) are unchanged. Only --cron
    opts in."""
    from engine.research import decay_watch_trigger as dwt
    # Even with a blocking gate patched in, dedup=False ignores it
    monkeypatch.setattr(dwt, "should_emit_for_subject",
                          lambda s, ev: (False, "open_alert_exists"))
    captured = {}
    import engine.research_store.emit as emit_mod
    monkeypatch.setattr(emit_mod, "decay_alert",
                          lambda **kw: captured.update(kw) or "eid_x")
    out = dwt.emit_decay_alert_from_subsample(
        subject_id="dedup_sleeve",
        subsample_output=_subsample_dict(
            sharpes=[1.5, 1.0, 0.5, 0.1],
            worst_best=0.07, monotone=True,
        ),
        # dedup omitted → False
    )
    assert out == "eid_x"


def test_min_triggers_threshold_overridable():
    """Caller can force emit on a single NEUTRAL trigger via
    min_triggers_for_emit=1 — useful for backfill verbose mode."""
    from engine.research import decay_watch_trigger as dwt
    fired = []
    def _fake_emit(**kw):
        fired.append(kw)
        return "eid"
    with patch("engine.research_store.emit.decay_alert", _fake_emit):
        out = dwt.emit_decay_alert_from_subsample(
            subject_id="any_sleeve",
            subsample_output=_subsample_dict(
                sharpes=[1.0, 0.8, 0.7, 0.6], worst_best=0.15,  # A only
            ),
            min_triggers_for_emit=1,
        )
    assert out == "eid"
    assert fired[0]["verdict"] == "MARGINAL"   # NEUTRAL routes as MARGINAL emit
