"""tests/test_chief_of_staff_memo.py — Phase 2.0 step 14b.

Mocked LLM tests for the weekly memo generator + integration into
the orchestrator runner.
"""
from __future__ import annotations

import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Fixture session_result
# ─────────────────────────────────────────────────────────────────────
def _session_result(**over):
    base = {
        "d_result": {
            "n_events_scanned": 441,
            "n_hits_total": 6, "n_hits_fresh": 0, "n_emitted": 0,
            "hits": [
                {"rule_name": "family_red_cluster", "family": "CARRY",
                 "severity": "CRITICAL", "is_fresh": False,
                 "metrics": {"red_count": 10}},
                {"rule_name": "family_red_cluster", "family": "HOLDINGS_BASED",
                 "severity": "CRITICAL", "is_fresh": False,
                 "metrics": {"red_count": 7}},
            ],
            "errors": [],
        },
        "a_result": {
            "snapshot": {"recent_summaries": 6, "deployed_sleeves": 5,
                          "recent_events": 40, "doctrine_snippets": 0},
            "n_candidates": 0, "n_written": 0, "candidates": [],
            "errors": [],
        },
        "b_result": {
            "n_candidates": 0, "n_reviewed": 0, "n_persisted": 0,
            "verdicts": [], "errors": [],
        },
    }
    base.update(over)
    return base


def _valid_memo_payload(**over):
    base = {
        "headline":   "Quiet week — substrate sparse, A returned empty, B queue unchanged at 0",
        "bullets": [
            "D: 6 family-level RED clusters all dedup-suppressed from earlier emits (CARRY 10, HOLDINGS_BASED 7, OTHER 26, EARNINGS_DRIFT 4, PROFITABILITY 4, macro 4)",
            "A: Sonnet returned 0 candidates on substrate of 6 papers / 5 sleeves / 40 events / 0 doctrine — honest empty path per HXZ 65% prior",
            "B: nothing to review (A has not persisted any LLM_SYNTHESIS rows yet); /approvals queue still empty",
            "No errors across D, A, B substeps",
            "Substrate enrichment is the binding constraint — same as last 2 sessions",
        ],
        "whats_next": "Consider INGESTing 2-3 strong candidate papers manually via /research/papers/new to give A a richer prompt. The system is correctly disciplined; what's missing is signal-rich substrate.",
    }
    base.update(over)
    return base


def _mock_llm(monkeypatch, *, tool_input=None, text="", raise_exc=None,
                tool_name="emit_weekly_memo"):
    from engine.agents.chief_of_staff import memo as mm
    from engine.llm.call import LLMCallResult, ToolCall
    def _fake_call(**kw):
        if raise_exc is not None:
            raise raise_exc
        tcs = ()
        if tool_input is not None:
            tcs = (ToolCall(id="tc", name=tool_name, input=tool_input),)
        return LLMCallResult(
            text=text, tool_calls=tcs, stop_reason="tool_use",
            model="claude-sonnet-4-6", provider="anthropic",
            cost_usd=0.04, latency_ms=8000,
            cache_read_tokens=0, raw_usage={},
        )
    monkeypatch.setattr(mm, "llm_call", _fake_call)


# ─────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────
def test_generate_memo_returns_typed_memo(tmp_path, monkeypatch):
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, tool_input=_valid_memo_payload())
    memos_path = tmp_path / "weekly_memos.jsonl"
    m = generate_memo(
        session_id     = "cos-test",
        session_result = _session_result(),
        pending_b      = 0,
        memos_path     = memos_path,
    )
    assert m is not None
    assert m.session_id == "cos-test"
    assert m.headline.startswith("Quiet week")
    assert len(m.bullets) == 5
    assert "substrate" in m.whats_next.lower()
    assert m.model == "claude-sonnet-4-6"


def test_memo_persisted_to_jsonl(tmp_path, monkeypatch):
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, tool_input=_valid_memo_payload())
    memos_path = tmp_path / "weekly_memos.jsonl"
    generate_memo(session_id="cos-test",
                    session_result=_session_result(), pending_b=0,
                    memos_path=memos_path)
    assert memos_path.is_file()
    row = json.loads(memos_path.read_text(encoding="utf-8").strip())
    assert row["session_id"] == "cos-test"
    assert isinstance(row["bullets"], list)


def test_last_memos_loaded_into_context(tmp_path, monkeypatch):
    """The prior-context window must pass last memos to the LLM
    (continuity / delta narrative). Verify the prompt builder sees
    them."""
    from engine.agents.chief_of_staff.memo import generate_memo
    memos_path = tmp_path / "weekly_memos.jsonl"
    # Pre-seed two prior memos
    prior = [
        {"session_id": "cos-old1", "headline": "Last week head",
         "bullets": ["b1", "b2", "b3"], "whats_next": "x",
         "generated_ts": "2026-05-30T00:00:00Z",
         "model": "claude-sonnet-4-6"},
        {"session_id": "cos-old2", "headline": "Older head",
         "bullets": ["b1"], "whats_next": "y",
         "generated_ts": "2026-05-23T00:00:00Z",
         "model": "claude-sonnet-4-6"},
    ]
    with memos_path.open("w", encoding="utf-8") as f:
        for p in prior:
            f.write(json.dumps(p) + "\n")

    captured = {}
    from engine.agents.chief_of_staff import memo as mm
    from engine.llm.call import LLMCallResult, ToolCall
    def _fake_call(**kw):
        captured["user"] = kw["user"]
        return LLMCallResult(
            text="", tool_calls=(ToolCall(id="x", name="emit_weekly_memo",
                                            input=_valid_memo_payload()),),
            stop_reason="tool_use", model="claude-sonnet-4-6",
            provider="anthropic", cost_usd=0.04, latency_ms=1,
            cache_read_tokens=0, raw_usage={},
        )
    monkeypatch.setattr(mm, "llm_call", _fake_call)

    generate_memo(session_id="cos-new", session_result=_session_result(),
                    pending_b=0, memos_path=memos_path)
    assert "Last week head" in captured["user"]
    assert "Older head" in captured["user"]


def test_prompt_surfaces_session_substep_data(monkeypatch):
    """User message MUST include D/A/B summaries with concrete counts
    so the model can write specific bullets."""
    from engine.agents.chief_of_staff.memo import _format_input
    msg = _format_input(session_id="cos-x",
                          session_result=_session_result(),
                          last_memos=[], pending_b=3)
    assert "D (book monitor)" in msg
    assert "A (synthesis)" in msg
    assert "B (strengthener)" in msg
    assert "CARRY" in msg                    # family from D hit
    assert "HOLDINGS_BASED" in msg
    assert "pending /approvals (cumulative): 3" in msg


def test_system_prompt_carries_load_bearing_rules():
    from engine.agents.chief_of_staff.memo import _SYSTEM_PROMPT
    assert "30-second scan" in _SYSTEM_PROMPT
    assert "If nothing happened" in _SYSTEM_PROMPT
    assert "Empty bullets array is INVALID" in _SYSTEM_PROMPT
    assert "capital decisions" in _SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────
def test_bullets_under_three_rejects(tmp_path, monkeypatch):
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, tool_input=_valid_memo_payload(
        bullets=["only one bullet"]))
    assert generate_memo(session_id="x",
                           session_result=_session_result(),
                           pending_b=0,
                           memos_path=tmp_path / "m.jsonl") is None


def test_bullets_over_seven_truncates_not_rejects(tmp_path, monkeypatch):
    """Fix (2026-06-08): >7 bullets → truncate to 7, don't drop the
    whole memo. Previous behavior wasted Sonnet calls when LLM emitted
    8-9 bullets due to schema overshoot."""
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, tool_input=_valid_memo_payload(
        bullets=[f"bullet {i}" for i in range(8)]))
    m = generate_memo(session_id="x", session_result=_session_result(),
                       pending_b=0, memos_path=tmp_path / "m.jsonl")
    assert m is not None
    assert len(m.bullets) == 7   # truncated, not dropped


def test_empty_headline_rejects(tmp_path, monkeypatch):
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, tool_input=_valid_memo_payload(headline=""))
    assert generate_memo(session_id="x", session_result=_session_result(),
                           pending_b=0, memos_path=tmp_path / "m.jsonl") is None


def test_oversize_headline_truncates_not_rejects(tmp_path, monkeypatch):
    """Fix (2026-06-08): headline >120 chars → truncate to 117+'…',
    don't drop the memo. Previous behavior wasted Sonnet calls every
    weekly run because the model routinely emitted 125-140 char
    headlines slightly over the schema's maxLength=120."""
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, tool_input=_valid_memo_payload(
        headline="x" * 200))
    m = generate_memo(session_id="x", session_result=_session_result(),
                       pending_b=0, memos_path=tmp_path / "m.jsonl")
    assert m is not None
    assert len(m.headline) == 118   # 117 'x' + '…' = 118 chars total
    assert m.headline.endswith("…")
    # Empty headline still drops the memo (it's a hard requirement)
    _mock_llm(monkeypatch, tool_input=_valid_memo_payload(headline=""))
    assert generate_memo(session_id="y", session_result=_session_result(),
                           pending_b=0, memos_path=tmp_path / "m2.jsonl") is None


def test_empty_whats_next_rejects(tmp_path, monkeypatch):
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, tool_input=_valid_memo_payload(whats_next=""))
    assert generate_memo(session_id="x", session_result=_session_result(),
                           pending_b=0, memos_path=tmp_path / "m.jsonl") is None


def test_llm_exception_returns_none(tmp_path, monkeypatch):
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, raise_exc=RuntimeError("api down"))
    assert generate_memo(session_id="x", session_result=_session_result(),
                           pending_b=0, memos_path=tmp_path / "m.jsonl") is None


def test_no_tool_call_returns_none(tmp_path, monkeypatch):
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, tool_input=None, text="I'd rather write prose")
    assert generate_memo(session_id="x", session_result=_session_result(),
                           pending_b=0, memos_path=tmp_path / "m.jsonl") is None


def test_wrong_tool_name_returns_none(tmp_path, monkeypatch):
    from engine.agents.chief_of_staff.memo import generate_memo
    _mock_llm(monkeypatch, tool_input=_valid_memo_payload(),
                tool_name="some_other_tool")
    assert generate_memo(session_id="x", session_result=_session_result(),
                           pending_b=0, memos_path=tmp_path / "m.jsonl") is None


# ─────────────────────────────────────────────────────────────────────
# Integration with orchestrator
# ─────────────────────────────────────────────────────────────────────
def test_runner_runs_memo_after_b_and_includes_in_result(monkeypatch):
    """The orchestrator must call generate_memo after B and surface the
    result in SessionResult.memo + on the session emit."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.book_monitor import runner as d_mod
    from engine.agents.papers_curator import synthesis_runner as a_mod
    from engine.agents.strengthener import runner as b_mod
    from engine.agents.chief_of_staff import runner as cos_mod
    from engine.agents.chief_of_staff import memo as mm
    from engine.research_store import emit
    from engine.agents.chief_of_staff.memo import WeeklyMemo

    monkeypatch.setattr(d_mod, "run_book_monitor",
        lambda **kw: {"n_events_scanned": 100, "n_hits_total": 0,
                       "n_hits_fresh": 0, "n_emitted": 0,
                       "event_ids": [], "errors": []})
    monkeypatch.setattr(a_mod, "run_synthesis_pipeline",
        lambda **kw: {"snapshot": {}, "candidates": [], "n_candidates": 0,
                       "written_hypothesis_ids": [], "n_written": 0,
                       "errors": [], "event_id": "ev_a"})
    monkeypatch.setattr(b_mod, "run_strengthener_pipeline",
        lambda **kw: {"n_candidates": 0, "n_reviewed": 0, "n_persisted": 0,
                       "verdicts": [], "errors": []})
    monkeypatch.setattr(cos_mod, "_count_pending_b_approvals", lambda: 0)

    captured_memo_call = {}
    fake_memo = WeeklyMemo(
        session_id="cos-test", headline="Test headline",
        bullets=("b1","b2","b3"), whats_next="do X",
        generated_ts="2026-06-06T13:00:00Z", model="claude-sonnet-4-6",
    )
    def _fake_gen(**kw):
        captured_memo_call.update(kw)
        return fake_memo
    monkeypatch.setattr(mm, "generate_memo", _fake_gen)

    captured_emit = {}
    def _fake_emit(**kw):
        captured_emit.update(kw)
        return "ev_session"
    monkeypatch.setattr(emit, "chief_of_staff_session_run", _fake_emit)

    result = run_weekly_session(session_id="cos-test")
    # Memo was called
    assert captured_memo_call.get("session_id") == "cos-test"
    # Memo dict on result
    assert result.memo is not None
    assert result.memo["headline"] == "Test headline"
    # Emit got the headline metric
    assert captured_emit["memo_headline"] == "Test headline"


def test_runner_dry_run_skips_memo(monkeypatch):
    """dry_run must NOT call generate_memo — preview runs shouldn't
    pollute weekly_memos.jsonl or burn cost."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.book_monitor import runner as d_mod
    from engine.agents.papers_curator import synthesis_runner as a_mod
    from engine.agents.strengthener import runner as b_mod
    from engine.agents.chief_of_staff import runner as cos_mod
    from engine.agents.chief_of_staff import memo as mm

    monkeypatch.setattr(d_mod, "run_book_monitor",
        lambda **kw: {"n_events_scanned": 0, "n_hits_total": 0,
                       "n_hits_fresh": 0, "n_emitted": 0,
                       "event_ids": [], "errors": []})
    monkeypatch.setattr(a_mod, "run_synthesis_pipeline",
        lambda **kw: {"snapshot": {}, "candidates": [], "n_candidates": 0,
                       "written_hypothesis_ids": [], "n_written": 0,
                       "errors": [], "event_id": None})
    monkeypatch.setattr(b_mod, "run_strengthener_pipeline",
        lambda **kw: {"n_candidates": 0, "n_reviewed": 0, "n_persisted": 0,
                       "verdicts": [], "errors": []})
    monkeypatch.setattr(cos_mod, "_count_pending_b_approvals", lambda: 0)

    called = []
    monkeypatch.setattr(mm, "generate_memo", lambda **kw: called.append(kw) or None)

    result = run_weekly_session(dry_run=True)
    assert called == []
    assert result.memo is None


def test_runner_memo_failure_doesnt_kill_session(monkeypatch):
    """Memo step failing must NOT block the session emit — the substep
    results are still valuable."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.book_monitor import runner as d_mod
    from engine.agents.papers_curator import synthesis_runner as a_mod
    from engine.agents.strengthener import runner as b_mod
    from engine.agents.chief_of_staff import runner as cos_mod
    from engine.agents.chief_of_staff import memo as mm
    from engine.research_store import emit

    monkeypatch.setattr(d_mod, "run_book_monitor",
        lambda **kw: {"n_events_scanned": 0, "n_hits_total": 0,
                       "n_hits_fresh": 0, "n_emitted": 0,
                       "event_ids": [], "errors": []})
    monkeypatch.setattr(a_mod, "run_synthesis_pipeline",
        lambda **kw: {"snapshot": {}, "candidates": [], "n_candidates": 0,
                       "written_hypothesis_ids": [], "n_written": 0,
                       "errors": [], "event_id": None})
    monkeypatch.setattr(b_mod, "run_strengthener_pipeline",
        lambda **kw: {"n_candidates": 0, "n_reviewed": 0, "n_persisted": 0,
                       "verdicts": [], "errors": []})
    monkeypatch.setattr(cos_mod, "_count_pending_b_approvals", lambda: 0)
    def _boom(**kw):
        raise RuntimeError("memo broke")
    monkeypatch.setattr(mm, "generate_memo", _boom)

    emit_called = []
    monkeypatch.setattr(emit, "chief_of_staff_session_run",
                          lambda **kw: emit_called.append(kw) or "ev_session")

    result = run_weekly_session()
    # Session emit STILL fires
    assert emit_called
    assert result.session_event_id == "ev_session"
    # Memo error recorded
    assert any("memo" in e for e in result.errors)
    assert result.memo is None
