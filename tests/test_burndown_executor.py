"""burn-1b executor tests — execution path with mocked extractor + dispatcher.

We don't call real Sonnet in tests; the executor accepts injected
spec_extractor_fn and dispatcher_fn for that exact reason.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from engine.research import burndown_executor


REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Helpers ────────────────────────────────────────────────────────


class _FakeSpec:
    """Minimal stand-in for a FactorSpec; executor only reads attributes
    on the dispatch_result it gets back."""
    def __init__(self, hypothesis_id="h-fake", signal_kind="cross_sec"):
        self.hypothesis_id = hypothesis_id
        self.signal_kind   = signal_kind


class _FakeCandidate:
    def __init__(self, hid, family):
        self.hypothesis_id = hid
        self.family = family


def _ok_spec(_hyp):
    return _FakeSpec()


def _bad_spec(_hyp):
    return None


def _raising_spec(_hyp):
    raise RuntimeError("extractor blew up")


def _ok_dispatch_green(_spec, **kwargs):
    return {
        "hypothesis_id":     _spec.hypothesis_id,
        "spec_hash":         "spec-hash-1",
        "refusal":           None,
        "template_result":   {
            "verdict":    "GREEN",
            "summary":    "fake green",
            "metrics":    {
                "sharpe": 1.5,
                "oos_triple": {"severity": "none", "decay_pct": -0.05},
            },
        },
        "dispatch_event_id": "dev-1",
        "prediction_id":     "pred-1",
    }


def _refusal_dispatch(_spec, **kwargs):
    return {
        "hypothesis_id":     _spec.hypothesis_id,
        "spec_hash":         "spec-hash-1",
        "refusal":           {"reason_code": "TEMPLATE_NOT_CERTIFIED", "detail": "..."},
        "template_result":   None,
        "dispatch_event_id": "dev-2",
        "prediction_id":     "pred-2",
    }


# ── Tests ──────────────────────────────────────────────────────────


def test_executor_returns_extract_failure_when_extract_returns_none(tmp_path, monkeypatch):
    # Point HYPOTHESES_PATH at a tmp file with one row matching candidate
    from engine.research import burndown_ranker
    hyp_path = tmp_path / "hyp.jsonl"
    hyp_path.write_text(
        json.dumps({
            "hypothesis_id":  "h-1",
            "source_paper_id": "p-1",
            "version":         1,
            "schema_version":  1,
            "source_chunk_ids":[],
            "verbatim_quotes": [],
            "claim":           "x",
            "mechanism_family":"PROFITABILITY",
            "predicted_direction": "positive",
            "predicted_magnitude": "x",
            "required_data":   [],
            "test_methodology":"x",
            "extraction_method": "llm_extract",
            "review_state": "proposed",
            "created_ts": "2026-06-01T00:00:00Z",
            "created_by": "test",
        }) + "\n", encoding="utf-8",
    )
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp_path)

    ex = burndown_executor.BurndownExecutor(
        cron_run_id="cr-test",
        spec_extractor_fn=_bad_spec,
        dispatcher_fn=_ok_dispatch_green,
    )
    cand = _FakeCandidate("h-1", "PROFITABILITY")
    out = ex.execute_one(cand)
    assert out.extraction_ok is False
    assert out.extraction_error == "EXTRACT_RETURNED_NONE"
    assert out.verdict is None


def test_executor_handles_extract_exception(tmp_path, monkeypatch):
    from engine.research import burndown_ranker
    hyp_path = tmp_path / "hyp.jsonl"
    hyp_path.write_text(json.dumps({
        "hypothesis_id":  "h-2",
        "source_paper_id": "p-1",
        "version":         1,
        "schema_version":  1,
        "source_chunk_ids":[],
        "verbatim_quotes": [],
        "claim":           "x",
        "mechanism_family":"PROFITABILITY",
        "predicted_direction": "positive",
        "predicted_magnitude": "x",
        "required_data":   [],
        "test_methodology":"x",
        "extraction_method": "llm_extract",
        "review_state": "proposed",
        "created_ts": "2026-06-01T00:00:00Z",
        "created_by": "test",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp_path)

    ex = burndown_executor.BurndownExecutor(
        cron_run_id="cr-x",
        spec_extractor_fn=_raising_spec,
        dispatcher_fn=_ok_dispatch_green,
    )
    out = ex.execute_one(_FakeCandidate("h-2", "PROFITABILITY"))
    assert out.extraction_ok is False
    assert out.extraction_error.startswith("EXTRACT_EXCEPTION")


def test_executor_returns_full_outcome_on_green(tmp_path, monkeypatch):
    from engine.research import burndown_ranker
    hyp_path = tmp_path / "hyp.jsonl"
    hyp_path.write_text(json.dumps({
        "hypothesis_id":  "h-3",
        "source_paper_id": "p-1",
        "version":         1,
        "schema_version":  1,
        "source_chunk_ids":[],
        "verbatim_quotes": [],
        "claim":           "x",
        "mechanism_family":"VALUE",
        "predicted_direction": "positive",
        "predicted_magnitude": "x",
        "required_data":   [],
        "test_methodology":"x",
        "extraction_method": "llm_extract",
        "review_state": "proposed",
        "created_ts": "2026-06-01T00:00:00Z",
        "created_by": "test",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp_path)

    ex = burndown_executor.BurndownExecutor(
        cron_run_id="cr-test",
        spec_extractor_fn=_ok_spec,
        dispatcher_fn=_ok_dispatch_green,
    )
    out = ex.execute_one(_FakeCandidate("h-3", "VALUE"))
    assert out.extraction_ok is True
    assert out.verdict == "GREEN"
    assert out.refusal_reason is None
    assert out.decay_severity == "none"
    assert out.dispatch_event_id == "dev-1"
    assert out.prediction_id == "pred-1"
    assert out.cron_run_id == "cr-test"


def test_executor_records_refusal_without_burning_slot(tmp_path, monkeypatch):
    from engine.research import burndown_ranker
    hyp_path = tmp_path / "hyp.jsonl"
    hyp_path.write_text(json.dumps({
        "hypothesis_id":  "h-4",
        "source_paper_id": "p-1",
        "version":         1,
        "schema_version":  1,
        "source_chunk_ids":[],
        "verbatim_quotes": [],
        "claim":           "x",
        "mechanism_family":"LOW_VOL",
        "predicted_direction": "positive",
        "predicted_magnitude": "x",
        "required_data":   [],
        "test_methodology":"x",
        "extraction_method": "llm_extract",
        "review_state": "proposed",
        "created_ts": "2026-06-01T00:00:00Z",
        "created_by": "test",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp_path)

    ex = burndown_executor.BurndownExecutor(
        cron_run_id="cr-r",
        spec_extractor_fn=_ok_spec,
        dispatcher_fn=_refusal_dispatch,
    )
    out = ex.execute_one(_FakeCandidate("h-4", "LOW_VOL"))
    assert out.extraction_ok is True
    assert out.refusal_reason == "TEMPLATE_NOT_CERTIFIED"
    assert out.verdict is None


def test_executor_missing_hypothesis_id(tmp_path, monkeypatch):
    from engine.research import burndown_ranker
    hyp_path = tmp_path / "hyp.jsonl"
    hyp_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp_path)

    ex = burndown_executor.BurndownExecutor(
        cron_run_id="cr-miss",
        spec_extractor_fn=_ok_spec,
        dispatcher_fn=_ok_dispatch_green,
    )
    out = ex.execute_one(_FakeCandidate("missing", "LOW_VOL"))
    assert out.extraction_ok is False
    assert out.extraction_error == "HYPOTHESIS_NOT_FOUND"


# ── execute_plan + cap mid-run ─────────────────────────────────────


class _FakePlan:
    def __init__(self, candidates):
        self.candidates = candidates


def test_execute_plan_runs_each_candidate(tmp_path, monkeypatch):
    from engine.research import burndown_ranker, burndown_caps
    hyp_path = tmp_path / "hyp.jsonl"
    rows = []
    for hid, fam in [("h-A", "VALUE"), ("h-B", "LOW_VOL")]:
        rows.append(json.dumps({
            "hypothesis_id":  hid,
            "source_paper_id": "p-1", "version":1, "schema_version":1,
            "source_chunk_ids":[], "verbatim_quotes":[],
            "claim":"x", "mechanism_family":fam,
            "predicted_direction":"positive","predicted_magnitude":"x",
            "required_data":[], "test_methodology":"x",
            "extraction_method":"llm_extract",
            "review_state": "proposed",
            "created_ts": "2026-06-01T00:00:00Z",
            "created_by": "test",
        }))
    hyp_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp_path)

    log_path = tmp_path / "dispatch_log.jsonl"; log_path.touch()
    monkeypatch.setattr(burndown_caps, "DEFAULT_DISPATCH_LOG", log_path)

    ex = burndown_executor.BurndownExecutor(
        cron_run_id="cr-multi",
        log_path=log_path,
        spec_extractor_fn=_ok_spec,
        dispatcher_fn=_ok_dispatch_green,
    )
    plan = _FakePlan([
        _FakeCandidate("h-A", "VALUE"),
        _FakeCandidate("h-B", "QUALITY"),
    ])
    out = ex.execute_plan(plan, respect_caps_mid_run=False)
    assert len(out) == 2
    assert {o.verdict for o in out} == {"GREEN"}


def test_executor_logs_extract_returned_none_to_dispatch_log(tmp_path, monkeypatch):
    """burn-1c.1: EXTRACT_RETURNED_NONE must write a dispatch_log row so
    the next cron round's ranker dedups this hypothesis."""
    from engine.research import burndown_ranker
    from engine.agents.strengthener import factor_dispatcher

    hyp_path = tmp_path / "hyp.jsonl"
    hyp_path.write_text(json.dumps({
        "hypothesis_id":  "h-extract-none",
        "source_paper_id": "p-1", "version": 1, "schema_version": 1,
        "source_chunk_ids": [], "verbatim_quotes": [],
        "claim": "x", "mechanism_family": "PROFITABILITY",
        "predicted_direction": "positive", "predicted_magnitude": "x",
        "required_data": [], "test_methodology": "x",
        "extraction_method": "llm_extract",
        "review_state": "proposed",
        "created_ts": "2026-06-01T00:00:00Z", "created_by": "test",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp_path)

    log_path = tmp_path / "dispatch_log.jsonl"
    # Point factor_dispatcher's default log path at tmp via record_extraction_failure
    monkeypatch.setattr(factor_dispatcher, "FACTOR_DISPATCH_LOG_PATH", log_path)

    ex = burndown_executor.BurndownExecutor(
        cron_run_id="cr-extract-none",
        spec_extractor_fn=_bad_spec,
        dispatcher_fn=_ok_dispatch_green,
    )
    out = ex.execute_one(_FakeCandidate("h-extract-none", "PROFITABILITY"))
    assert out.extraction_ok is False
    assert out.extraction_error == "EXTRACT_RETURNED_NONE"
    assert out.dispatch_event_id is not None       # log row written

    # Verify dispatch_log row exists with refusal reason_code
    rows = [
        json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["hypothesis_id"] == "h-extract-none"
    assert rows[0]["refusal"]["reason_code"] == "EXTRACT_RETURNED_NONE"
    assert rows[0]["template_result"] is None

    # Now verify ranker dedups: same hypothesis should NOT re-rank
    ranked = burndown_ranker.rank_candidates(
        top_k=5, hyp_path=hyp_path, dispatch_log_path=log_path,
        gaps_path=tmp_path / "empty.jsonl",
        now=_dt.datetime(2026, 6, 11, 12, 0, tzinfo=_dt.timezone.utc),
    )
    assert all(c.hypothesis_id != "h-extract-none" for c in ranked)


def test_executor_passes_source_through_to_dispatcher(tmp_path, monkeypatch):
    """burn-1c: executor's source field must reach the dispatcher
    as cron_source kwarg so dispatch_log rows can be audited."""
    from engine.research import burndown_ranker
    hyp_path = tmp_path / "hyp.jsonl"
    hyp_path.write_text(json.dumps({
        "hypothesis_id":  "h-src",
        "source_paper_id": "p-1", "version": 1, "schema_version": 1,
        "source_chunk_ids": [], "verbatim_quotes": [],
        "claim": "x", "mechanism_family": "VALUE",
        "predicted_direction": "positive", "predicted_magnitude": "x",
        "required_data": [], "test_methodology": "x",
        "extraction_method": "llm_extract",
        "review_state": "proposed",
        "created_ts": "2026-06-01T00:00:00Z", "created_by": "test",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp_path)

    captured = {}
    def _capture_dispatcher(_spec, **kwargs):
        captured.update(kwargs)
        return _ok_dispatch_green(_spec, **kwargs)

    ex = burndown_executor.BurndownExecutor(
        cron_run_id="cr-src",
        spec_extractor_fn=_ok_spec,
        dispatcher_fn=_capture_dispatcher,
        source="auto",
    )
    ex.execute_one(_FakeCandidate("h-src", "VALUE"))
    assert captured.get("cron_source") == "auto"
    assert captured.get("cron_run_id") == "cr-src"
    # No force_reason → no human_override
    assert "human_override" not in captured


def test_executor_force_reason_attaches_human_override(tmp_path, monkeypatch):
    from engine.research import burndown_ranker
    hyp_path = tmp_path / "hyp.jsonl"
    hyp_path.write_text(json.dumps({
        "hypothesis_id":  "h-force",
        "source_paper_id": "p-1", "version": 1, "schema_version": 1,
        "source_chunk_ids": [], "verbatim_quotes": [],
        "claim": "x", "mechanism_family": "VALUE",
        "predicted_direction": "positive", "predicted_magnitude": "x",
        "required_data": [], "test_methodology": "x",
        "extraction_method": "llm_extract",
        "review_state": "proposed",
        "created_ts": "2026-06-01T00:00:00Z", "created_by": "test",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(burndown_ranker, "HYPOTHESES_PATH", hyp_path)

    captured = {}
    def _capture(_spec, **kwargs):
        captured.update(kwargs)
        return _ok_dispatch_green(_spec, **kwargs)

    ex = burndown_executor.BurndownExecutor(
        cron_run_id="cr-force",
        spec_extractor_fn=_ok_spec,
        dispatcher_fn=_capture,
        source="manual",
        force_reason="smoke testing new portfolio_overlay template",
    )
    ex.execute_one(_FakeCandidate("h-force", "VALUE"))
    override = captured.get("human_override", "")
    assert "force" in override
    assert "smoke testing" in override


def test_summarize_outcomes_basic():
    outs = [
        burndown_executor.ExecutionOutcome(
            hypothesis_id="h1", family="VALUE", cron_run_id="cr",
            extraction_ok=True, extraction_error=None, spec_hash="sh",
            refusal_reason=None, verdict="GREEN", decay_severity="none",
            dispatch_event_id="de", prediction_id="pr", ran_at="t",
        ),
        burndown_executor.ExecutionOutcome(
            hypothesis_id="h2", family="LOW_VOL", cron_run_id="cr",
            extraction_ok=True, extraction_error=None, spec_hash="sh",
            refusal_reason="TEMPLATE_NOT_CERTIFIED", verdict=None,
            decay_severity=None, dispatch_event_id="de", prediction_id=None,
            ran_at="t",
        ),
        burndown_executor.ExecutionOutcome(
            hypothesis_id="h3", family="MOMENTUM", cron_run_id="cr",
            extraction_ok=False, extraction_error="EXTRACT_RETURNED_NONE",
            spec_hash=None, refusal_reason=None, verdict=None,
            decay_severity=None, dispatch_event_id=None, prediction_id=None,
            ran_at="t",
        ),
    ]
    text = burndown_executor.summarize_outcomes(outs)
    assert "extraction failed:       1" in text
    assert "TEMPLATE_NOT_CERTIFIED: 1" in text
    assert "GREEN: 1" in text


def test_write_outcomes_round_trip(tmp_path):
    outs = [burndown_executor.ExecutionOutcome(
        hypothesis_id="h", family="VALUE", cron_run_id="cr",
        extraction_ok=True, extraction_error=None, spec_hash="sh",
        refusal_reason=None, verdict="GREEN", decay_severity="none",
        dispatch_event_id="de", prediction_id="pr", ran_at="t",
    )]
    out_path = burndown_executor.write_outcomes(outs, "plan-uuid-1234", out_dir=tmp_path)
    assert out_path.is_file()
    d = json.loads(out_path.read_text(encoding="utf-8"))
    assert d["plan_id"] == "plan-uuid-1234"
    assert len(d["outcomes"]) == 1
    assert d["outcomes"][0]["verdict"] == "GREEN"
