"""External adversarial audit substrate tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research import external_audit as ea


# ── Stub provider behavior ────────────────────────────────────────


def test_stub_provider_returns_skipped():
    provider = ea._StubProvider()
    response, severity, flagged, cost = provider.adversarial_audit(
        subject_payload={"event_id": "test"},
        prompt="test prompt",
    )
    assert severity == "skipped"
    assert flagged == []
    assert cost == 0.0
    assert "stub" in response.lower()


def test_stub_is_default_provider():
    """No env var configured → stub provider used."""
    import os
    monkey_env = os.environ.copy()
    os.environ.pop("EXTERNAL_AUDIT_PROVIDER", None)
    try:
        p = ea._get_active_provider()
        assert p.name == "stub"
    finally:
        os.environ.clear()
        os.environ.update(monkey_env)


# ── Register / lookup ─────────────────────────────────────────────


class _FakeProvider:
    name = "fake"
    def adversarial_audit(self, *, subject_payload, prompt):
        return ("FAKE ISSUE", "concern", ["statistical"], 0.05)


def test_register_provider_makes_it_lookup_able(monkeypatch):
    fake = _FakeProvider()
    ea.register_provider(fake)
    monkeypatch.setenv("EXTERNAL_AUDIT_PROVIDER", "fake")
    p = ea._get_active_provider()
    assert p.name == "fake"


def test_unknown_provider_falls_back_to_stub(monkeypatch):
    monkeypatch.setenv("EXTERNAL_AUDIT_PROVIDER", "does_not_exist")
    p = ea._get_active_provider()
    assert p.name == "stub"


# ── audit_verdict_event ───────────────────────────────────────────


def _ev(*, event_id="e1", verdict="GREEN", family="TEST_FAM",
         summary="test summary", metrics=None):
    return {
        "event_id": event_id,
        "subject_id": "subj-" + event_id,
        "verdict": verdict,
        "family": family,
        "summary": summary,
        "metrics": metrics or {},
    }


def test_audit_skipped_when_stub_provider(tmp_path):
    log = tmp_path / "audits.jsonl"
    record = ea.audit_verdict_event(_ev(), log_path=log)
    assert record.severity == "skipped"
    assert record.provider == "stub"
    # Should still write a log row
    rows = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1
    assert rows[0]["audit_id"] == record.audit_id


def test_audit_with_explicit_provider_records_severity(tmp_path):
    log = tmp_path / "audits.jsonl"
    record = ea.audit_verdict_event(
        _ev(verdict="GREEN", family="X"),
        provider=_FakeProvider(),
        log_path=log,
    )
    assert record.severity == "concern"
    assert "statistical" in record.flagged_categories
    assert record.cost_estimate_usd == pytest.approx(0.05)


def test_audit_provider_exception_recorded_as_skipped(tmp_path):
    class _BrokenProvider:
        name = "broken"
        def adversarial_audit(self, **kw):
            raise RuntimeError("intentional")
    log = tmp_path / "audits.jsonl"
    record = ea.audit_verdict_event(
        _ev(), provider=_BrokenProvider(), log_path=log,
    )
    assert record.severity == "skipped"
    assert "PROVIDER_ERROR" in record.audit_response


def test_audit_payload_includes_ff_complement_alpha(tmp_path):
    """Verdict payload sent to provider must include the key spanning
    metrics so external reviewer can sanity-check them."""
    captured = {}
    class _CapturingProvider:
        name = "capturing"
        def adversarial_audit(self, *, subject_payload, prompt):
            captured["payload"] = subject_payload
            captured["prompt"] = prompt
            return ("ok", "no_issue", [], 0.01)
    log = tmp_path / "audits.jsonl"
    ea.audit_verdict_event(
        _ev(metrics={
            "sharpe_gross": 0.70, "nw_t_gross": 5.02,
            "capm_alpha_t": 2.30, "ff_complement_alpha_t": 0.11,
            "ff_complement_anchor": ["MKT_RF", "SMB", "RMW", "CMA"],
            "strategy_family": "COMBINATION_HML_MOM",
        }),
        provider=_CapturingProvider(),
        log_path=log,
    )
    assert "ff_complement_alpha_t" in captured["payload"]["key_metrics"]
    assert "strategy_family" in captured["payload"]["key_metrics"]


# ── recent_audits / severity_breakdown ────────────────────────────


def test_recent_audits_filter_by_window(tmp_path):
    log = tmp_path / "audits.jsonl"
    log.write_text("\n".join([
        json.dumps({"audit_id": "old", "ts": "2025-01-01T00:00:00Z",
                     "severity": "concern", "flagged_categories": []}),
        json.dumps({"audit_id": "new", "ts": "2026-06-13T00:00:00Z",
                     "severity": "noted", "flagged_categories": []}),
    ]) + "\n", encoding="utf-8")
    out = ea.recent_audits(days_back=30, log_path=log)
    ids = {a["audit_id"] for a in out}
    assert "new" in ids
    assert "old" not in ids


def test_severity_breakdown_excludes_skipped():
    audits = [
        {"severity": "critical"},
        {"severity": "concern"},
        {"severity": "concern"},
        {"severity": "noted"},
        {"severity": "skipped"},
        {"severity": "skipped"},
    ]
    breakdown = ea.severity_breakdown(audits)
    assert breakdown == {"critical": 1, "concern": 2, "noted": 1, "no_issue": 0}
