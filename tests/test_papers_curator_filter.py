"""tests/test_papers_curator_filter.py — Phase 1.5 filter unit tests.

Verifies:
  - JSON parser handles raw JSON, fenced JSON, JSON-with-prose
  - judge_paper happy path with mocked llm_call
  - judgments_store append + latest-by-paper round-trip
"""
from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────
# _parse_judgment_json — handles all the Deepseek-text shapes we've seen
# ──────────────────────────────────────────────────────────────────────
def test_parse_raw_json():
    from engine.agents.papers_curator.filter import _parse_judgment_json
    out = _parse_judgment_json(
        '{"is_tradable_factor": true, "confidence": 0.85, '
        '"one_line_reason": "new momentum signal", "category_guess": "new_factor"}'
    )
    assert out is not None
    assert out["is_tradable_factor"] is True
    assert out["category_guess"] == "new_factor"


def test_parse_fenced_json():
    from engine.agents.papers_curator.filter import _parse_judgment_json
    out = _parse_judgment_json(
        '```json\n'
        '{"is_tradable_factor": false, "confidence": 0.9, '
        '"one_line_reason": "pure theory", "category_guess": "theory"}\n'
        '```'
    )
    assert out is not None
    assert out["is_tradable_factor"] is False


def test_parse_json_with_prose():
    from engine.agents.papers_curator.filter import _parse_judgment_json
    out = _parse_judgment_json(
        'Here is my judgment:\n'
        '{"is_tradable_factor": true, "confidence": 0.7, '
        '"one_line_reason": "x", "category_guess": "refinement"}\n'
        'Hope this helps.'
    )
    assert out is not None
    assert out["confidence"] == 0.7


def test_parse_unparseable_returns_none():
    from engine.agents.papers_curator.filter import _parse_judgment_json
    assert _parse_judgment_json("not json at all") is None
    assert _parse_judgment_json("") is None
    assert _parse_judgment_json(None) is None  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────
# judge_paper with mocked llm_call
# ──────────────────────────────────────────────────────────────────────
def _make_candidate():
    from engine.agents.papers_curator.crawler import PaperCandidate
    return PaperCandidate(
        source       = "arxiv",
        source_id    = "2401.99999",
        title        = "A New Cross-Sectional Momentum Signal in Crypto",
        authors      = ("Test Author",),
        abstract     = "We construct a 12-1 momentum signal in crypto.",
        abs_url      = "http://arxiv.org/abs/2401.99999",
        pdf_url      = "http://arxiv.org/pdf/2401.99999.pdf",
        published_ts = "2024-01-01T00:00:00Z",
        categories   = ("q-fin.PR",),
        fetched_ts   = "2024-01-02T00:00:00Z",
    )


def test_judge_paper_happy_path(monkeypatch):
    """Mock llm_call to return a clean JSON; judge_paper must map it
    into a FilterJudgment."""
    from engine.agents.papers_curator import filter as filter_mod
    from engine.llm.call import LLMCallResult

    def _fake_call(**kw):
        return LLMCallResult(
            text=(
                '{"is_tradable_factor": true, "confidence": 0.82, '
                '"one_line_reason": "new crypto momentum signal with backtest", '
                '"category_guess": "new_factor"}'
            ),
            tool_calls=(),
            stop_reason="end_turn",
            model="deepseek-v4-pro",
            provider="deepseek",
            cost_usd=0.0012,
            latency_ms=2100,
            cache_read_tokens=0,
            raw_usage={},
        )

    # Patch the locally-imported reference inside filter.judge_paper —
    # the function does `from engine.llm.call import call as llm_call`
    # at runtime, so patch on the source module.
    monkeypatch.setattr("engine.agents.papers_curator.filter.llm_call", _fake_call)

    j = filter_mod.judge_paper(_make_candidate())
    assert j is not None
    assert j.is_tradable_factor is True
    assert j.confidence == 0.82
    assert j.category_guess == "new_factor"
    assert j.model == "deepseek-v4-pro"
    assert j.source == "arxiv"
    assert j.source_id == "2401.99999"


def test_judge_paper_returns_none_on_llm_failure(monkeypatch):
    """If llm_call raises, judge_paper returns None (caller leaves for
    next run rather than crashing the daily pipeline)."""
    from engine.agents.papers_curator import filter as filter_mod

    def _raises(**kw):
        raise RuntimeError("deepseek api down")
    monkeypatch.setattr("engine.agents.papers_curator.filter.llm_call", _raises)

    assert filter_mod.judge_paper(_make_candidate()) is None


def test_judge_paper_returns_none_on_unparseable(monkeypatch):
    """If LLM returns unparseable text, judge_paper returns None."""
    from engine.agents.papers_curator import filter as filter_mod
    from engine.llm.call import LLMCallResult

    def _fake_call(**kw):
        return LLMCallResult(
            text="I am happy to help! Let me think about this...",
            tool_calls=(),
            stop_reason="end_turn",
            model="deepseek-v4-pro",
            provider="deepseek",
            cost_usd=0.001,
            latency_ms=1500,
            cache_read_tokens=0,
            raw_usage={},
        )
    monkeypatch.setattr("engine.agents.papers_curator.filter.llm_call", _fake_call)
    assert filter_mod.judge_paper(_make_candidate()) is None


# ──────────────────────────────────────────────────────────────────────
# judgments_store — latest-by-paper picks newest by judged_ts
# ──────────────────────────────────────────────────────────────────────
def test_judgments_store_latest_by_paper(tmp_path, monkeypatch):
    import engine.agents.papers_curator.judgments_store as js
    from engine.agents.papers_curator.filter import FilterJudgment

    monkeypatch.setattr(js, "JUDGMENTS_PATH", tmp_path / "judgments.jsonl")

    j_old = FilterJudgment(
        source="arxiv", source_id="2401.00001",
        is_tradable_factor=False, confidence=0.5,
        one_line_reason="old judgment", category_guess="theory",
        judged_ts="2024-01-01T00:00:00Z", model="deepseek-v4-pro",
        raw_response="",
    )
    j_new = FilterJudgment(
        source="arxiv", source_id="2401.00001",
        is_tradable_factor=True, confidence=0.8,
        one_line_reason="re-judged with better prompt", category_guess="new_factor",
        judged_ts="2024-02-01T00:00:00Z", model="deepseek-v4-pro",
        raw_response="",
    )
    js.append_judgment(j_old)
    js.append_judgment(j_new)

    all_rows = js.load_judgments()
    assert len(all_rows) == 2

    latest = js.latest_by_paper()
    assert ("arxiv", "2401.00001") in latest
    picked = latest[("arxiv", "2401.00001")]
    assert picked.is_tradable_factor is True   # newer one wins
    assert picked.judged_ts == "2024-02-01T00:00:00Z"
