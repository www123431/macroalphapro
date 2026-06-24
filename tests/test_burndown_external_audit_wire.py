"""Phase 1.2 (2026-06-13): external_audit wire-in to burndown_executor.

Mitigation #1 of self-audit blind-spots doctrine: every cron-emitted
GREEN/MARGINAL/RED verdict must get an independent adversarial review.

Tests:
  - Wire path actually fires audit_verdict_event on GREEN verdict
  - Non-verdict outcomes (refusal, INSUFFICIENT_DATA) skip audit
  - Disabled env var (BURNDOWN_EXTERNAL_AUDIT_DISABLED=1) skips audit
  - Weekly budget cap blocks audit when exhausted
  - Audit exception does NOT bubble out (cron must not crash)
  - cost gets recorded to budget ledger
  - severity=skipped does NOT consume budget
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.research import burndown_executor as bx


@pytest.fixture
def tmp_budget(tmp_path, monkeypatch):
    """Isolate budget ledger to a tmp file per test."""
    p = tmp_path / "external_audit_budget.jsonl"
    return p


@pytest.fixture
def disable_audit_env(monkeypatch):
    """Default: audit DISABLED for tests so unrelated tests don't fire it."""
    monkeypatch.setenv("BURNDOWN_EXTERNAL_AUDIT_DISABLED", "1")


@pytest.fixture
def enable_audit_env(monkeypatch):
    monkeypatch.delenv("BURNDOWN_EXTERNAL_AUDIT_DISABLED", raising=False)


def _green_outcome():
    return bx.ExecutionOutcome(
        hypothesis_id     = "test-hyp",
        family            = "TEST_FAMILY",
        cron_run_id       = "cron-test",
        extraction_ok     = True,
        extraction_error  = None,
        spec_hash         = "abc",
        refusal_reason    = None,
        verdict           = "GREEN",
        decay_severity    = None,
        dispatch_event_id = "evt-test",
        prediction_id     = "pred-test",
        ran_at            = "2026-06-13T00:00:00Z",
    )


# ── Audit firing logic ───────────────────────────────────────────


def test_audit_fires_on_green_verdict(enable_audit_env, tmp_budget):
    """A GREEN verdict from cron path must trigger audit_verdict_event."""
    call_count = {"n": 0, "events": []}

    def fake_audit(event):
        call_count["n"] += 1
        call_count["events"].append(event)
        rec = MagicMock()
        rec.severity = "no_issue"
        rec.cost_estimate_usd = 0.005
        rec.audit_id = "audit-fake"
        rec.flagged_categories = []
        return rec

    outcome = _green_outcome()
    tr = {"summary": "test verdict", "metrics": {"sharpe_gross": 0.5}}
    bx._maybe_audit_verdict(outcome, tr, budget_path=tmp_budget, audit_fn=fake_audit)

    assert call_count["n"] == 1
    ev = call_count["events"][0]
    assert ev["verdict"] == "GREEN"
    assert ev["subject_id"] == "test-hyp"
    assert ev["family"] == "TEST_FAMILY"
    assert ev["event_id"] == "evt-test"


def test_audit_skips_on_refusal(enable_audit_env, tmp_budget):
    """Refusal outcomes (verdict=None) must NOT fire audit."""
    call_count = {"n": 0}

    def fake_audit(event):
        call_count["n"] += 1
        return MagicMock()

    outcome = bx.ExecutionOutcome(
        hypothesis_id="h", family="F", cron_run_id="c",
        extraction_ok=True, extraction_error=None, spec_hash=None,
        refusal_reason="SIGNAL_INPUT_UNKNOWN", verdict=None,
        decay_severity=None, dispatch_event_id=None,
        prediction_id=None, ran_at="2026-06-13T00:00:00Z",
    )
    bx._maybe_audit_verdict(outcome, {}, budget_path=tmp_budget, audit_fn=fake_audit)
    assert call_count["n"] == 0


def test_audit_skips_on_insufficient_data(enable_audit_env, tmp_budget):
    """INSUFFICIENT_DATA / INSUFFICIENT_HISTORY verdicts must NOT fire audit."""
    call_count = {"n": 0}
    def fake_audit(event):
        call_count["n"] += 1
        return MagicMock()

    for v in ("INSUFFICIENT_DATA", "INSUFFICIENT_HISTORY", "EXECUTION_ERROR",
              "PENDING_TEMPLATE_BUILD"):
        outcome = _green_outcome()
        outcome = bx.ExecutionOutcome(**{**outcome.to_dict(), "verdict": v})
        bx._maybe_audit_verdict(outcome, {}, budget_path=tmp_budget, audit_fn=fake_audit)
    assert call_count["n"] == 0


def test_audit_skipped_when_env_disabled(disable_audit_env, tmp_budget):
    """BURNDOWN_EXTERNAL_AUDIT_DISABLED=1 must short-circuit audit."""
    call_count = {"n": 0}
    def fake_audit(event):
        call_count["n"] += 1
        return MagicMock()

    bx._maybe_audit_verdict(
        _green_outcome(), {}, budget_path=tmp_budget, audit_fn=fake_audit,
    )
    assert call_count["n"] == 0


# ── Budget cap ────────────────────────────────────────────────────


def test_audit_skipped_when_budget_exhausted(enable_audit_env, tmp_budget):
    """When weekly spend ≥ budget cap, audit must skip without calling
    audit_fn."""
    # Pre-populate budget file at cap
    tmp_budget.parent.mkdir(parents=True, exist_ok=True)
    cap = bx.EXTERNAL_AUDIT_WEEKLY_BUDGET_USD
    with tmp_budget.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "week": bx._week_iso_now(),
            "ts": bx._utc_iso(),
            "audit_id": "exhaust-test",
            "cost_usd": cap + 0.01,
            "severity": "concern",
        }) + "\n")

    call_count = {"n": 0}
    def fake_audit(event):
        call_count["n"] += 1
        return MagicMock()

    bx._maybe_audit_verdict(
        _green_outcome(), {}, budget_path=tmp_budget, audit_fn=fake_audit,
    )
    assert call_count["n"] == 0


def test_budget_spend_records_with_severity(enable_audit_env, tmp_budget):
    """When audit returns non-skipped severity, a budget row gets appended."""
    def fake_audit(event):
        rec = MagicMock()
        rec.severity = "concern"
        rec.cost_estimate_usd = 0.0085
        rec.audit_id = "rec-1"
        rec.flagged_categories = ["spanning", "multi_test"]
        return rec

    bx._maybe_audit_verdict(
        _green_outcome(), {"summary": "x", "metrics": {}},
        budget_path=tmp_budget, audit_fn=fake_audit,
    )
    assert tmp_budget.is_file()
    rows = [json.loads(ln) for ln in tmp_budget.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1
    assert rows[0]["audit_id"] == "rec-1"
    assert abs(rows[0]["cost_usd"] - 0.0085) < 1e-9
    assert rows[0]["severity"] == "concern"


def test_skipped_severity_does_not_consume_budget(enable_audit_env, tmp_budget):
    """severity='skipped' (stub provider returned nothing) → no budget row."""
    def fake_audit(event):
        rec = MagicMock()
        rec.severity = "skipped"
        rec.cost_estimate_usd = 0.0
        rec.audit_id = "skip-1"
        rec.flagged_categories = []
        return rec

    bx._maybe_audit_verdict(
        _green_outcome(), {}, budget_path=tmp_budget, audit_fn=fake_audit,
    )
    assert not tmp_budget.is_file()


# ── Robustness ────────────────────────────────────────────────────


def test_audit_exception_never_raises(enable_audit_env, tmp_budget):
    """audit_fn raising must NOT bubble up — cron path must keep running."""
    def boom(event):
        raise RuntimeError("provider exploded")

    # Should NOT raise
    bx._maybe_audit_verdict(
        _green_outcome(), {}, budget_path=tmp_budget, audit_fn=boom,
    )


def test_execute_one_includes_audit_call(monkeypatch, tmp_budget, enable_audit_env):
    """End-to-end: ExecutionStream.execute_one wires the audit call after
    dispatch. Stub dispatcher returns a GREEN verdict; audit_fn observed."""
    audit_calls = []

    def fake_audit(event):
        audit_calls.append(event)
        rec = MagicMock()
        rec.severity = "no_issue"
        rec.cost_estimate_usd = 0.001
        rec.audit_id = "stream-1"
        rec.flagged_categories = []
        return rec

    # Patch the module's lazy-resolved audit fn by patching _maybe_audit_verdict
    # at the helper level (cleaner than mocking external_audit module).
    orig = bx._maybe_audit_verdict
    def patched(outcome, tr, **kw):
        return orig(outcome, tr, budget_path=tmp_budget, audit_fn=fake_audit)
    monkeypatch.setattr(bx, "_maybe_audit_verdict", patched)

    # Stub spec_extractor + dispatcher
    fake_spec = MagicMock()
    fake_spec.universe = "ken_french_ff5_mom"
    fake_spec.signal_kind = "factor_combination"

    def fake_extract(*args, **kwargs):
        return fake_spec

    def fake_dispatch(spec, **kwargs):
        return {
            "refusal": None,
            "template_result": {
                "verdict": "GREEN",
                "summary": "test pass",
                "metrics": {"sharpe_gross": 0.8},
            },
            "spec_hash": "h-1",
            "dispatch_event_id": "evt-stream-1",
            "prediction_id": "pred-stream-1",
        }

    # Stub the hypothesis loader
    fake_hyp = MagicMock()
    fake_hyp.id = "test-hyp-stream"
    monkeypatch.setattr(bx, "_load_hypothesis_by_id", lambda hid, **kw: fake_hyp)

    stream = bx.BurndownExecutor(
        cron_run_id="run-test",
        spec_extractor_fn=fake_extract,
        dispatcher_fn=fake_dispatch,
    )

    candidate = MagicMock()
    candidate.hypothesis_id = "test-hyp-stream"
    candidate.family = "TEST_FAMILY"

    outcome = stream.execute_one(candidate)
    assert outcome.verdict == "GREEN"
    assert len(audit_calls) == 1
    assert audit_calls[0]["verdict"] == "GREEN"
    assert audit_calls[0]["event_id"] == "evt-stream-1"
