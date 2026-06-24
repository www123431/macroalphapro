"""Tests for engine.research.mutation_proposer (Phase 1 ①).

Critical properties:
1. Whitelist enforcement: non-whitelisted mutation_type → REJECT
2. Sign-flip detection in keyword scan → REJECT
3. Signal-construction change → REJECT
4. n_trials must be exactly 1 → REJECT otherwise
5. cited_diagnosis_ts required → REJECT if missing
6. justification too short → REJECT
7. mutation_seq cap = 2 → 3rd rejected
8. existing-mutation count gate
9. Valid proposal passes
10. Deterministic mode: missing diagnosis → returns null proposal
11. Deterministic mode: GREEN verdict → returns null proposal
12. LLM mode falls back when no API key
13. Ledger append on valid proposal + log=True
14. Ledger NOT touched on dry_run / invalid
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research import mutation_proposer as MP


@pytest.fixture
def tmp_ledgers(tmp_path, monkeypatch):
    gate = tmp_path / "gate_runs.jsonl"
    diag = tmp_path / "diagnostic_reports.jsonl"
    mut = tmp_path / "mutation_proposals.jsonl"
    monkeypatch.setattr(MP, "GATE_LEDGER", gate)
    monkeypatch.setattr(MP, "DIAGNOSTIC_LEDGER", diag)
    monkeypatch.setattr(MP, "MUTATION_LEDGER", mut)
    return {"gate": gate, "diag": diag, "mut": mut}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ── Validator tests ─────────────────────────────────────────────────────

def test_non_whitelist_type_rejected():
    p = {
        "candidate_name": "x_v1", "mutation_type": "signal_threshold",
        "old_value": "0.5", "new_value": "0.6",
        "justification": "Diagnosis says threshold is wrong, change it. Citing diag ts.",
        "cited_diagnosis_ts": "2026-05-29T00:00:00Z",
    }
    r = MP.validate_mutation_proposal(p)
    assert r.ok is False
    assert any("not in whitelist" in reason for reason in r.reasons)


def test_sign_flip_rejected():
    p = {
        "candidate_name": "x_v1", "mutation_type": "weighting",
        "old_value": "EW long-top short-bottom",
        "new_value": "EW long-bottom short-top (sign flip)",
        "justification": "Need to flip the sign to recover alpha. Cite diagnosis.",
        "cited_diagnosis_ts": "2026-05-29T00:00:00Z",
    }
    r = MP.validate_mutation_proposal(p)
    assert r.ok is False
    assert any("sign-flip" in reason.lower() for reason in r.reasons)


def test_signal_construction_change_rejected():
    p = {
        "candidate_name": "x_v1", "mutation_type": "cost_model",
        "old_value": "12bp execution cost",
        "new_value": "use a different signal formula altogether",
        "justification": "Diagnosis says signal formula is suboptimal. Cite diag.",
        "cited_diagnosis_ts": "2026-05-29T00:00:00Z",
    }
    r = MP.validate_mutation_proposal(p)
    assert r.ok is False
    assert any("NEW CANDIDATE" in reason for reason in r.reasons)


def test_n_trials_must_be_1():
    p = {
        "candidate_name": "x_v1", "mutation_type": "sample_window",
        "old_value": "2010-2024", "new_value": "2007-2024 to include GFC",
        "justification": "Diagnosis flags missing 2008 stress period. Adding it.",
        "cited_diagnosis_ts": "2026-05-29T00:00:00Z",
        "n_trials_added": 3,
    }
    r = MP.validate_mutation_proposal(p)
    assert r.ok is False
    assert any("n_trials_added" in reason for reason in r.reasons)


def test_missing_cited_diagnosis_rejected():
    p = {
        "candidate_name": "x_v1", "mutation_type": "sample_window",
        "old_value": "2010-2024", "new_value": "2007-2024 to include 2008 GFC",
        "justification": "Just adding 2008 since it's missing from sample. Reasonable.",
    }
    r = MP.validate_mutation_proposal(p)
    assert r.ok is False
    assert any("cited_diagnosis_ts" in reason for reason in r.reasons)


def test_short_justification_rejected():
    p = {
        "candidate_name": "x_v1", "mutation_type": "sample_window",
        "old_value": "2010-2024", "new_value": "2007-2024",
        "justification": "Add GFC.",
        "cited_diagnosis_ts": "2026-05-29T00:00:00Z",
    }
    r = MP.validate_mutation_proposal(p)
    assert r.ok is False
    assert any("justification" in reason.lower() for reason in r.reasons)


def test_mutation_seq_cap(tmp_ledgers):
    p = {
        "candidate_name": "x_v1", "mutation_type": "sample_window",
        "old_value": "2010-2024",
        "new_value": "extend to 2007 to include 2008 GFC per diagnosis",
        "justification": "Diagnosis explicitly cites missing 2008 stress. Adding window.",
        "cited_diagnosis_ts": "2026-05-29T00:00:00Z",
        "mutation_seq": 3,
    }
    r = MP.validate_mutation_proposal(p)
    assert r.ok is False
    assert any("p-hacking" in reason for reason in r.reasons)


def test_existing_mutation_count_blocks_third(tmp_ledgers):
    # Seed 2 prior mutations
    _write_jsonl(tmp_ledgers["mut"], [
        {"candidate_name": "x_v1", "mutation_type": "sample_window",
         "mutation_seq": 1},
        {"candidate_name": "x_v1", "mutation_type": "cost_model",
         "mutation_seq": 2},
    ])
    p = {
        "candidate_name": "x_v1", "mutation_type": "horizon",
        "old_value": "60d", "new_value": "30d per diagnostic horizon cite",
        "justification": "Diagnosis explicitly suggests shorter horizon based on event speed.",
        "cited_diagnosis_ts": "2026-05-29T00:00:00Z",
    }
    r = MP.validate_mutation_proposal(p)
    assert r.ok is False
    assert any("max" in reason.lower() and "per original" in reason
                 for reason in r.reasons)


def test_valid_proposal_passes(tmp_ledgers):
    p = {
        "candidate_name": "x_v1", "mutation_type": "sample_window",
        "old_value": "2018-2024",
        "new_value": "extend to 2007-2024 to include 2008 GFC per diagnosis",
        "justification":
            "Diagnostic ts 2026-05-29T... cites 'missed 2008 GFC as canonical stress'. "
            "Extending sample to include this period addresses the regime gap.",
        "cited_diagnosis_ts": "2026-05-29T10:00:00Z",
    }
    r = MP.validate_mutation_proposal(p)
    assert r.ok is True, r.reasons


# ── Deterministic proposer tests ───────────────────────────────────────

def test_deterministic_no_diagnosis_returns_null(tmp_ledgers):
    res = MP._propose_deterministic("nonexistent_candidate")
    assert res["proposal"] is None
    assert "no diagnostic" in res["reason"]


def test_deterministic_green_returns_null(tmp_ledgers):
    _write_jsonl(tmp_ledgers["gate"], [{"name": "x_v1", "verdict": "GREEN"}])
    _write_jsonl(tmp_ledgers["diag"], [{
        "candidate": "x_v1", "timestamp": "2026-05-29T10:00:00Z",
        "refined_diagnosis": "missed 2008 GFC was the cause"
    }])
    res = MP._propose_deterministic("x_v1")
    assert res["proposal"] is None
    assert "GREEN" in res["reason"]


def test_deterministic_pattern_match(tmp_ledgers):
    _write_jsonl(tmp_ledgers["gate"], [
        {"name": "qx_v1", "verdict": "RED"}
    ])
    _write_jsonl(tmp_ledgers["diag"], [{
        "candidate": "qx_v1", "timestamp": "2026-05-29T10:00:00Z",
        "refined_diagnosis":
            "the sample missed 2008 GFC and 2018 vol-mageddon. "
            "ROOT CAUSE: missed 2008 stress was decisive."
    }])
    res = MP._propose_deterministic("qx_v1")
    assert res["proposal"] is not None
    assert res["proposal"]["mutation_type"] == "sample_window"
    assert res["validation"]["ok"] is True


# ── Public entry / LLM fallback / ledger ───────────────────────────────

def test_propose_falls_back_without_api_key(tmp_ledgers, monkeypatch):
    monkeypatch.setattr(MP, "_read_anthropic_key", lambda: None)
    _write_jsonl(tmp_ledgers["gate"], [{"name": "v1", "verdict": "RED"}])
    _write_jsonl(tmp_ledgers["diag"], [{
        "candidate": "v1", "timestamp": "2026-05-29T10:00:00Z",
        "refined_diagnosis": "missed 2008 stress as cause"
    }])
    res = MP.propose_mutation("v1", use_llm=True, log=False)
    assert "deterministic" in res["mode"]


def test_ledger_appends_on_valid_proposal(tmp_ledgers):
    _write_jsonl(tmp_ledgers["gate"], [{"name": "v1", "verdict": "RED"}])
    _write_jsonl(tmp_ledgers["diag"], [{
        "candidate": "v1", "timestamp": "2026-05-29T10:00:00Z",
        "refined_diagnosis":
            "the sample missed 2008 GFC; ROOT CAUSE: sample missed 2008 stress."
    }])
    res = MP.propose_mutation("v1", use_llm=False, log=True)
    assert res["proposal"] is not None
    assert tmp_ledgers["mut"].exists()
    rows = [json.loads(l) for l in tmp_ledgers["mut"].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["candidate_name"] == "v1"


def test_ledger_not_touched_on_no_proposal(tmp_ledgers):
    # No diagnosis → no proposal → no ledger entry
    res = MP.propose_mutation("ghost_v1", use_llm=False, log=True)
    assert res["proposal"] is None
    assert not tmp_ledgers["mut"].exists()


def test_read_mutation_ledger_returns_recent_first(tmp_ledgers):
    _write_jsonl(tmp_ledgers["mut"], [
        {"candidate_name": "a", "ts": 1},
        {"candidate_name": "b", "ts": 2},
        {"candidate_name": "c", "ts": 3},
    ])
    rows = MP.read_mutation_ledger(limit=10)
    assert [r["candidate_name"] for r in rows] == ["c", "b", "a"]
