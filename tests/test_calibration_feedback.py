"""Tests for Frontier 3 (2026-06-01) — calibration feedback loop.

Pure-Python paths (find / cluster / queue) are tested deterministically
with synthetic ledger data. LLM call (synthesize_proposed_rule) is
mocked so tests are fast + cost-free.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from unittest import mock

import pytest

from engine.research.calibration_feedback import (
    WrongCluster,
    _pattern_key,
    append_proposed_rule,
    cluster_council_wrong,
    find_council_wrong_iterations,
    read_proposed_rules,
    review_proposed_rule,
    run_calibration_scan,
    synthesize_proposed_rule,
)


@pytest.fixture
def isolated_ledgers(tmp_path, monkeypatch):
    """Redirect both the L4 iterations ledger AND the proposed-rules
    ledger to tmp files."""
    l4_path  = tmp_path / "l4_iterations.jsonl"
    prop_path = tmp_path / "proposed_intuition_rules.jsonl"
    monkeypatch.setattr(
        "engine.research.outcome_ledger.L4_LEDGER_PATH", l4_path,
    )
    monkeypatch.setattr(
        "engine.research.calibration_feedback.PROPOSED_RULES_LEDGER", prop_path,
    )
    return l4_path, prop_path


def _write_l4_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _make_wrong_row(
    *, family: str, council: str, pipeline: str, iteration_id: str,
    days_ago: int = 1,
    council_rationale: str = "council was confident",
    pipeline_rationale: str = "pipeline rejected on stat grounds",
) -> dict:
    ts = (_dt.datetime.utcnow() - _dt.timedelta(days=days_ago))
    return {
        "ts":           ts.isoformat(timespec="seconds") + "Z",
        "iteration_id": iteration_id,
        "verdict_alignment": "council_wrong",
        "proposal":     {"title": f"prop-{iteration_id}",
                          "family": family,
                          "proposed_role": "alpha_seeker"},
        "council":      {"consensus": council,
                          "rationale": council_rationale},
        "pipeline":     {"final_decision": pipeline,
                          "rationale": pipeline_rationale},
    }


# ── find_council_wrong_iterations ────────────────────────────────────────


def test_find_council_wrong_respects_window(isolated_ledgers):
    l4_path, _ = isolated_ledgers
    _write_l4_rows(l4_path, [
        _make_wrong_row(family="f1", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id="i1",
                          days_ago=1),
        _make_wrong_row(family="f1", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id="i2",
                          days_ago=45),
    ])
    out = find_council_wrong_iterations(since_days=30)
    assert len(out) == 1
    assert out[0]["iteration_id"] == "i1"


def test_find_council_wrong_filters_aligned_iterations(isolated_ledgers):
    """Only rows with alignment=council_wrong are returned; agree /
    pipeline_resolved rows must be excluded."""
    l4_path, _ = isolated_ledgers
    _write_l4_rows(l4_path, [
        {"ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
         "iteration_id": "agree-1",
         "verdict_alignment": "agree",
         "proposal": {"family": "f"},
         "council": {"consensus": "APPROVE"},
         "pipeline": {"final_decision": "PROMOTE_TO_GATE"}},
        _make_wrong_row(family="f", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id="wrong-1"),
    ])
    out = find_council_wrong_iterations(since_days=30)
    assert len(out) == 1
    assert out[0]["iteration_id"] == "wrong-1"


# ── cluster_council_wrong ────────────────────────────────────────────────


def test_cluster_groups_by_family_and_direction():
    rows = [
        _make_wrong_row(family="A", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id="a1"),
        _make_wrong_row(family="A", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id="a2"),
        _make_wrong_row(family="B", council="REJECT",
                          pipeline="PROMOTE_TO_GATE", iteration_id="b1"),
    ]
    clusters = cluster_council_wrong(rows)
    assert len(clusters) == 2
    biggest = clusters[0]
    assert biggest.family == "A"
    assert biggest.n == 2
    assert set(biggest.iteration_ids) == {"a1", "a2"}


def test_cluster_does_not_merge_different_pipeline_decisions():
    """Same family but different pipeline_decision = different root
    cause cluster (e.g. HARD_REJECT vs SOFT_REJECT)."""
    rows = [
        _make_wrong_row(family="A", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id="a1"),
        _make_wrong_row(family="A", council="APPROVE",
                          pipeline="SOFT_REJECT", iteration_id="a2"),
    ]
    clusters = cluster_council_wrong(rows)
    assert len(clusters) == 2


def test_pattern_key_format():
    row = _make_wrong_row(family="earnings_drift", council="APPROVE",
                            pipeline="HARD_REJECT", iteration_id="x")
    assert _pattern_key(row) == "earnings_drift::APPROVE::HARD_REJECT"


# ── synthesize_proposed_rule (LLM mocked) ────────────────────────────────


def _mock_anthropic_response(rule_json: dict):
    """Build a fake anthropic.Client whose messages.create() returns the
    given rule JSON inside a single text block."""
    fake_response = mock.MagicMock()
    fake_block = mock.MagicMock()
    fake_block.type = "text"
    fake_block.text = json.dumps(rule_json)
    fake_response.content = [fake_block]
    fake_client = mock.MagicMock()
    fake_client.messages.create.return_value = fake_response
    return fake_client


def test_synthesize_proposed_rule_returns_meta_enriched_dict(monkeypatch):
    cluster = WrongCluster(
        pattern_key="A::APPROVE::HARD_REJECT",
        family="A", council_consensus="APPROVE", pipeline_decision="HARD_REJECT",
        iteration_ids=["i1", "i2"], n=2,
        sample_proposals=[{"title": "X", "proposed_role": "alpha_seeker"}],
        sample_council_rationales=["council said clean"],
        sample_pipeline_rationales=["pipeline failed multi-test"],
    )
    canned = {
        "id":          "family_a_multi_test_blind_spot",
        "category":    "statistical",
        "severity":    "HARD_WARN",
        "when":        "Proposal in family A claims clean Sharpe.",
        "then":        "Check multi-test correction (Bailey-LdP).",
        "evidence_source": "calibration_cluster:auto",
        "rationale":   "council missed multi-test in family A twice.",
    }
    fake_client = _mock_anthropic_response(canned)
    monkeypatch.setattr(
        "engine.research.calibration_feedback._load_anthropic_key",
        lambda: "fake-key",
    )
    with mock.patch("anthropic.Anthropic", return_value=fake_client):
        out = synthesize_proposed_rule(cluster)
    assert out["id"] == canned["id"]
    assert out["category"] == "statistical"
    assert out["_meta"]["cluster_pattern_key"] == cluster.pattern_key
    assert out["_meta"]["cluster_size"] == 2
    assert out["_meta"]["cluster_iteration_ids"] == ["i1", "i2"]


def test_synthesize_proposed_rule_raises_on_unparseable_llm_output(monkeypatch):
    cluster = WrongCluster(
        pattern_key="A::APPROVE::HARD_REJECT",
        family="A", council_consensus="APPROVE", pipeline_decision="HARD_REJECT",
        iteration_ids=["i1"], n=1,
        sample_proposals=[], sample_council_rationales=[],
        sample_pipeline_rationales=[],
    )
    fake_response = mock.MagicMock()
    fake_block = mock.MagicMock()
    fake_block.type = "text"
    fake_block.text = "this is not json at all"
    fake_response.content = [fake_block]
    fake_client = mock.MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr(
        "engine.research.calibration_feedback._load_anthropic_key",
        lambda: "fake-key",
    )
    with mock.patch("anthropic.Anthropic", return_value=fake_client):
        with pytest.raises(RuntimeError, match="unparseable JSON"):
            synthesize_proposed_rule(cluster)


# ── proposed-rule queue ──────────────────────────────────────────────────


def test_append_and_read_proposed_rule(isolated_ledgers):
    _, prop_path = isolated_ledgers
    pid = append_proposed_rule({"id": "test_rule",
                                  "category": "statistical",
                                  "severity": "HARD_WARN"})
    assert pid.startswith("prop-")
    rules = read_proposed_rules()
    assert len(rules) == 1
    assert rules[0]["proposal_id"] == pid
    assert rules[0]["status"] == "pending"


def test_review_proposed_rule_atomic_swap(isolated_ledgers):
    _, prop_path = isolated_ledgers
    pid = append_proposed_rule({"id": "r1", "category": "regime",
                                  "severity": "HARD_WARN"})
    out = review_proposed_rule(pid, status="accepted",
                                 reviewer="zhang", note="looks right")
    assert out["status"] == "accepted"
    assert out["reviewed_by"] == "zhang"
    assert out["review_note"] == "looks right"
    # Re-read confirms persistence
    rules = read_proposed_rules()
    assert rules[0]["status"] == "accepted"
    # No tmp file left behind
    assert not (prop_path.with_suffix(".jsonl.tmp")).exists()


def test_review_proposed_rule_rejects_invalid_status(isolated_ledgers):
    pid = append_proposed_rule({"id": "r1"})
    with pytest.raises(ValueError, match="status must be"):
        review_proposed_rule(pid, status="pending")


def test_review_proposed_rule_unknown_id_raises(isolated_ledgers):
    append_proposed_rule({"id": "r1"})
    with pytest.raises(KeyError):
        review_proposed_rule("prop-doesnotexist", status="accepted")


# ── run_calibration_scan ─────────────────────────────────────────────────


def test_scan_skips_clusters_below_threshold(isolated_ledgers):
    """Cluster of size 1 must NOT trigger synthesis even when scan
    is called — saves LLM cost on one-off noise."""
    l4_path, _ = isolated_ledgers
    _write_l4_rows(l4_path, [
        _make_wrong_row(family="lonely", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id="x1"),
    ])
    out = run_calibration_scan(min_cluster_size=2)
    assert out["n_wrong_iterations"] == 1
    assert out["n_actionable_clusters"] == 0
    assert out["n_synthesized"] == 0


def test_scan_synthesizes_only_actionable_clusters(isolated_ledgers, monkeypatch):
    l4_path, _ = isolated_ledgers
    _write_l4_rows(l4_path, [
        _make_wrong_row(family="A", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id="a1"),
        _make_wrong_row(family="A", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id="a2"),
        _make_wrong_row(family="B", council="REJECT",
                          pipeline="PROMOTE_TO_GATE", iteration_id="b1"),  # singleton
    ])
    canned = {"id": "synth_rule", "category": "statistical",
                "severity": "HARD_WARN",
                "when": "Family A claims clean.", "then": "Multi-test.",
                "evidence_source": "auto", "rationale": "ok"}
    fake_client = _mock_anthropic_response(canned)
    monkeypatch.setattr(
        "engine.research.calibration_feedback._load_anthropic_key",
        lambda: "fake",
    )
    with mock.patch("anthropic.Anthropic", return_value=fake_client):
        out = run_calibration_scan(min_cluster_size=2)
    assert out["n_clusters"] == 2
    assert out["n_actionable_clusters"] == 1
    assert out["n_synthesized"] == 1
    # The queue now has 1 proposed rule
    queued = read_proposed_rules()
    assert len(queued) == 1
    assert queued[0]["rule"]["id"] == "synth_rule"


def test_scan_continues_when_one_cluster_synthesis_fails(
    isolated_ledgers, monkeypatch,
):
    l4_path, _ = isolated_ledgers
    _write_l4_rows(l4_path, [
        _make_wrong_row(family="A", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id=f"a{i}")
        for i in range(2)
    ] + [
        _make_wrong_row(family="B", council="APPROVE",
                          pipeline="HARD_REJECT", iteration_id=f"b{i}")
        for i in range(2)
    ])

    call_counter = {"n": 0}

    def fake_synthesize(cluster, *, api_key=None, model=None):
        call_counter["n"] += 1
        if cluster.family == "A":
            raise RuntimeError("simulated LLM blip")
        return {"id": "ok_rule", "category": "regime",
                "severity": "HARD_WARN",
                "when": "x", "then": "y",
                "evidence_source": "auto", "rationale": "z",
                "_meta": {"cluster_pattern_key": cluster.pattern_key,
                          "cluster_size": cluster.n,
                          "cluster_iteration_ids": cluster.iteration_ids}}

    monkeypatch.setattr(
        "engine.research.calibration_feedback.synthesize_proposed_rule",
        fake_synthesize,
    )
    out = run_calibration_scan(min_cluster_size=2)
    assert out["n_actionable_clusters"] == 2
    assert out["n_synthesized"] == 1
    assert len(out["errors"]) == 1
    assert out["errors"][0]["pattern_key"].startswith("A::")
