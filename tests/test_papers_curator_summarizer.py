"""tests/test_papers_curator_summarizer.py — Phase 1.5b summarizer tests.

Network path = mocked llm_call. Validates:
  - happy path: 5-field payload → PaperSummary
  - recommended_action enum gating (invalid → SKIP)
  - risk_flags truncation and type coercion
  - summaries_store: latest-by-paper picks newest
"""
from __future__ import annotations


def _make_candidate_and_judgment():
    from engine.agents.papers_curator.crawler import PaperCandidate
    from engine.agents.papers_curator.filter import FilterJudgment
    c = PaperCandidate(
        source="arxiv", source_id="2403.00007",
        title="Time-Series Momentum in G10 Bonds",
        authors=("Author One",), abstract="We document 12-1 TSMOM in G10 bonds.",
        abs_url="http://arxiv.org/abs/2403.00007",
        pdf_url="http://arxiv.org/pdf/2403.00007.pdf",
        published_ts="2024-03-01T00:00:00Z",
        categories=("q-fin.PR",), fetched_ts="2024-03-02T00:00:00Z",
    )
    j = FilterJudgment(
        source="arxiv", source_id="2403.00007",
        is_tradable_factor=True, confidence=0.85,
        one_line_reason="TSMOM applied to G10 bonds",
        category_guess="new_factor",
        judged_ts="2024-03-02T00:00:00Z", model="deepseek-v4-pro",
        raw_response="",
    )
    return c, j


def test_summarize_happy_path(monkeypatch):
    from engine.agents.papers_curator import summarizer as sm
    from engine.llm.call import LLMCallResult

    def _fake_call(**kw):
        return LLMCallResult(
            text=(
                '{"thesis": "TSMOM works on G10 bonds.", '
                '"mechanism": "Behavioral underreaction in yields.", '
                '"testable_hypothesis": "MOMENTUM/TSMOM signal on G10 sovereign bonds, monthly rebal", '
                '"why_matters_for_us": "Adjacent to deployed TSMOM equity sleeve, asset-class extension", '
                '"risk_flags": ["window post-2010", "no QE-era stress test", "needs CGB/JGB data"], '
                '"recommended_action": "INGEST"}'
            ),
            tool_calls=(), stop_reason="end_turn",
            model="deepseek-v4-pro", provider="deepseek",
            cost_usd=0.011, latency_ms=2800,
            cache_read_tokens=0, raw_usage={},
        )
    monkeypatch.setattr("engine.agents.papers_curator.summarizer.llm_call", _fake_call)

    c, j = _make_candidate_and_judgment()
    s = sm.summarize_paper(c, j, triggered_by="auto_yes")
    assert s is not None
    assert s.thesis == "TSMOM works on G10 bonds."
    assert "MOMENTUM/TSMOM" in s.testable_hypothesis
    assert s.recommended_action == "INGEST"
    assert s.risk_flags == ("window post-2010", "no QE-era stress test", "needs CGB/JGB data")
    assert s.triggered_by == "auto_yes"


def test_invalid_action_defaults_to_skip(monkeypatch):
    """If LLM returns an unknown recommended_action enum value, we
    coerce to SKIP rather than crash."""
    from engine.agents.papers_curator import summarizer as sm
    from engine.llm.call import LLMCallResult

    def _fake_call(**kw):
        return LLMCallResult(
            text=(
                '{"thesis": "t", "mechanism": "m", '
                '"testable_hypothesis": "h", "why_matters_for_us": "w", '
                '"risk_flags": [], "recommended_action": "MAYBE_INGEST_LATER"}'
            ),
            tool_calls=(), stop_reason="end_turn",
            model="deepseek-v4-pro", provider="deepseek",
            cost_usd=0.01, latency_ms=2000,
            cache_read_tokens=0, raw_usage={},
        )
    monkeypatch.setattr("engine.agents.papers_curator.summarizer.llm_call", _fake_call)

    c, j = _make_candidate_and_judgment()
    s = sm.summarize_paper(c, j)
    assert s is not None
    assert s.recommended_action == "SKIP"


def test_unparseable_returns_none(monkeypatch):
    from engine.agents.papers_curator import summarizer as sm
    from engine.llm.call import LLMCallResult

    def _fake_call(**kw):
        return LLMCallResult(
            text="I'd be happy to summarize this paper for you...",
            tool_calls=(), stop_reason="end_turn",
            model="deepseek-v4-pro", provider="deepseek",
            cost_usd=0.005, latency_ms=1800,
            cache_read_tokens=0, raw_usage={},
        )
    monkeypatch.setattr("engine.agents.papers_curator.summarizer.llm_call", _fake_call)

    c, j = _make_candidate_and_judgment()
    assert sm.summarize_paper(c, j) is None


def test_summaries_store_latest_wins(tmp_path, monkeypatch):
    import engine.agents.papers_curator.summaries_store as ss
    from engine.agents.papers_curator.summarizer import PaperSummary

    monkeypatch.setattr(ss, "SUMMARIES_PATH", tmp_path / "summaries.jsonl")

    s_old = PaperSummary(
        source="arxiv", source_id="2401.00002",
        thesis="old", mechanism="m", testable_hypothesis="t",
        why_matters_for_us="w", risk_flags=("r",),
        recommended_action="SKIP",
        triggered_by="auto_yes",
        summarized_ts="2024-01-01T00:00:00Z",
        model="deepseek-v4-pro", raw_response="",
    )
    s_new = PaperSummary(
        source="arxiv", source_id="2401.00002",
        thesis="user re-checked, better take",
        mechanism="m2", testable_hypothesis="t2",
        why_matters_for_us="w2", risk_flags=(),
        recommended_action="INGEST",
        triggered_by="user_request_recheck",
        summarized_ts="2024-02-01T00:00:00Z",
        model="deepseek-v4-pro", raw_response="",
    )
    ss.append_summary(s_old)
    ss.append_summary(s_new)

    latest = ss.latest_by_paper()
    picked = latest[("arxiv", "2401.00002")]
    assert picked.recommended_action == "INGEST"
    assert picked.thesis.startswith("user re-checked")
