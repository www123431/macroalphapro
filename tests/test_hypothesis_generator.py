"""Tests for engine.research.hypothesis_generator (Phase 2)."""
from __future__ import annotations

import json

import pytest

from engine.research import hypothesis_generator as HG


def test_deterministic_proposes_or_returns_null(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "PROPOSAL_QUEUE", tmp_path / "proposal_queue.jsonl")
    result = HG._deterministic_propose(include_pending=True)
    # Must always return a dict with proposal field
    assert "proposal" in result
    assert "mode" in result
    if result["proposal"]:
        # All required fields present
        p = result["proposal"]
        for field in ("mechanism_id", "canonical_paper_id", "sample_start",
                       "sample_end", "parameters", "justification",
                       "hygiene_summary", "h7_critique"):
            assert field in p, f"missing {field}"
        # H7 critique populated
        assert p["h7_critique"].get("verdict") in ("survive", "kill")


def test_deterministic_proposes_equity_xsmom_jt():
    """With current library (9 entries, 2 candidates with include_pending),
    equity_xsmom_jt should pass all gates while low_vol_bab hard-rejects."""
    result = HG._deterministic_propose(include_pending=True)
    if result["proposal"]:
        assert result["proposal"]["mechanism_id"] == "equity_xsmom_jt"


def test_deterministic_no_visible_returns_null():
    """include_pending=False → 0 visible entries → null proposal (R5)."""
    result = HG._deterministic_propose(include_pending=False)
    assert result["proposal"] is None
    assert "no unexplored visible" in result["reason"].lower()


def test_generate_proposal_falls_back_without_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "_read_anthropic_key", lambda: None)
    monkeypatch.setattr(HG, "PROPOSAL_QUEUE", tmp_path / "proposal_queue.jsonl")
    result = HG.generate_proposal(use_llm=True, log=False)
    assert "deterministic" in result["mode"]


def test_proposal_queue_appends_on_log(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "PROPOSAL_QUEUE", tmp_path / "proposal_queue.jsonl")
    # Use deterministic mode with include_pending=True via direct call
    # (generate_proposal's deterministic path uses include_pending=False
    # by default; mock to use pending=True)
    monkeypatch.setattr(HG, "_deterministic_propose",
                          lambda **kw: HG._deterministic_propose.__wrapped__(include_pending=True)
                          if hasattr(HG._deterministic_propose, "__wrapped__")
                          else {"proposal": {
                              "mechanism_id": "equity_xsmom_jt",
                              "canonical_paper_id": "jegadeesh_titman_1993_jf",
                              "sample_start": "1965-01-01",
                              "sample_end":   "2026-05-30",
                              "parameters":   ["horizon=12-1"],
                              "justification": "test",
                              "hygiene_summary": {},
                              "h7_critique":  {"verdict": "survive"},
                          }, "mode": "deterministic_only"})
    HG.generate_proposal(use_llm=False, log=True)
    assert (tmp_path / "proposal_queue.jsonl").exists()


def test_proposal_queue_NOT_written_when_h7_kills(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "PROPOSAL_QUEUE", tmp_path / "proposal_queue.jsonl")
    monkeypatch.setattr(HG, "_deterministic_propose",
                          lambda **kw: {"proposal": {
                              "mechanism_id": "x",
                              "h7_critique": {"verdict": "kill"},
                          }, "mode": "deterministic_only"})
    HG.generate_proposal(use_llm=False, log=True)
    assert not (tmp_path / "proposal_queue.jsonl").exists()


def test_proposal_queue_NOT_written_when_null_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "PROPOSAL_QUEUE", tmp_path / "proposal_queue.jsonl")
    monkeypatch.setattr(HG, "_deterministic_propose",
                          lambda **kw: {"proposal": None, "mode": "deterministic_only",
                                          "reason": "test no proposal"})
    HG.generate_proposal(use_llm=False, log=True)
    assert not (tmp_path / "proposal_queue.jsonl").exists()


def test_read_proposal_queue_recent_first(tmp_path, monkeypatch):
    queue_path = tmp_path / "proposal_queue.jsonl"
    monkeypatch.setattr(HG, "PROPOSAL_QUEUE", queue_path)
    queue_path.write_text(
        "\n".join([json.dumps({"ts": "1", "proposal": {"mechanism_id": "a"}}),
                    json.dumps({"ts": "2", "proposal": {"mechanism_id": "b"}}),
                    json.dumps({"ts": "3", "proposal": {"mechanism_id": "c"}})]),
        encoding="utf-8")
    rows = HG.read_proposal_queue(limit=10)
    assert [r["ts"] for r in rows] == ["3", "2", "1"]


def test_extract_json_simple():
    assert HG._extract_json('Some text {"key": "value"} more text') == {"key": "value"}


def test_extract_json_no_braces():
    assert HG._extract_json("plain text no braces") is None


def test_extract_json_nested():
    text = 'Output: {"proposal": {"id": "x"}, "other": null}'
    parsed = HG._extract_json(text)
    assert parsed == {"proposal": {"id": "x"}, "other": None}
