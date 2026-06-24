"""Tests for Frontier A (2026-06-01) — per-critic calibration.

Synthetic data: we directly build council + pipeline_report dicts that
mirror the real shape, then verify the calibration math.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from engine.research.critic_calibration import (
    _aggregate,
    _interpret_marginal,
    append_critic_calibration_rows,
    classify_critic_alignment,
    compute_critic_accuracy,
    compute_critic_marginal_info,
    compute_pairwise_critic_agreement,
    critic_calibration_report,
)


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    fake = tmp_path / "critic_calibration.jsonl"
    monkeypatch.setattr(
        "engine.research.critic_calibration.CRITIC_CALIBRATION_LEDGER", fake,
    )
    return fake


def _verdict(name: str, v: str, conf: float = 0.7) -> dict:
    return {"agent_name": name, "verdict": v, "confidence": conf}


def _seed(
    ledger,
    *,
    iteration_id: str,
    critics: list[dict],
    pipeline: str,
    consensus: str = "APPROVE",
    family: str = "f1",
    role: str = "alpha_seeker",
) -> None:
    """Synthesize a council + write per-critic rows directly via the
    public append fn — this exercises the real classifier path."""
    append_critic_calibration_rows(
        iteration_id=iteration_id,
        council={"consensus": consensus, "verdicts": critics, "rationale": ""},
        proposal={"family": family, "proposed_role": role,
                   "title": iteration_id},
        pipeline_report={"final_decision": pipeline, "ran": True},
    )


# ── classify_critic_alignment ────────────────────────────────────────


def test_pass_with_promote_is_agree():
    assert classify_critic_alignment("PASS", "PROMOTE_TO_GATE") == "agree"


def test_pass_with_reject_is_critic_wrong():
    assert classify_critic_alignment("PASS", "HARD_REJECT") == "critic_wrong"


def test_fail_with_reject_is_agree():
    assert classify_critic_alignment("FAIL", "HARD_REJECT") == "agree"


def test_fail_with_promote_is_critic_wrong():
    assert classify_critic_alignment("FAIL", "PROMOTE_AS_REPLACEMENT") == "critic_wrong"


def test_warn_with_borderline_is_agree():
    assert classify_critic_alignment("WARN", "BORDERLINE_REVIEW") == "agree"


def test_warn_with_promote_is_pipeline_resolved():
    """Critic said 'uncertain'; pipeline gave a definite answer — not a
    miscalibration, just an upgrade in resolution."""
    assert classify_critic_alignment("WARN", "PROMOTE_TO_GATE") == "pipeline_resolved"


def test_no_pipeline_is_not_runnable():
    assert classify_critic_alignment("PASS", None) == "not_runnable"


# ── append_critic_calibration_rows ───────────────────────────────────


def test_append_writes_one_row_per_critic(isolated_ledger):
    n = append_critic_calibration_rows(
        iteration_id="iter-1",
        council={"consensus": "APPROVE",
                  "verdicts": [_verdict("theorist", "PASS"),
                                _verdict("DA",       "WARN")],
                  "rationale": ""},
        proposal={"family": "fam", "proposed_role": "alpha_seeker"},
        pipeline_report={"final_decision": "PROMOTE_TO_GATE", "ran": True},
    )
    assert n == 2
    rows = [json.loads(line)
             for line in isolated_ledger.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    assert len(rows) == 2
    by_name = {r["critic_agent_name"]: r for r in rows}
    assert by_name["theorist"]["alignment"] == "agree"
    assert by_name["DA"]["alignment"] == "pipeline_resolved"


def test_append_skips_when_no_verdicts(isolated_ledger):
    n = append_critic_calibration_rows(
        iteration_id="iter-empty",
        council={"verdicts": []},
        proposal={"family": "f"},
        pipeline_report={"final_decision": "HARD_REJECT"},
    )
    assert n == 0
    assert not isolated_ledger.exists() or isolated_ledger.read_text() == ""


# ── compute_critic_accuracy ──────────────────────────────────────────


def test_accuracy_simple(isolated_ledger):
    # theorist: 3 PASSes; 2 hit PROMOTE (agree), 1 hits REJECT (wrong)
    for i in range(2):
        _seed(isolated_ledger, iteration_id=f"i{i}",
              critics=[_verdict("theorist", "PASS")],
              pipeline="PROMOTE_TO_GATE")
    _seed(isolated_ledger, iteration_id="i2",
          critics=[_verdict("theorist", "PASS")],
          pipeline="HARD_REJECT")
    out = compute_critic_accuracy("theorist")
    assert out["n_decided"] == 3
    assert out["accuracy"] == round(2/3, 3)


def test_accuracy_ignores_warn_rows_in_decided(isolated_ledger):
    """WARN + non-borderline pipeline = pipeline_resolved, not decided."""
    _seed(isolated_ledger, iteration_id="warn-x",
          critics=[_verdict("DA", "WARN")],
          pipeline="PROMOTE_TO_GATE")
    out = compute_critic_accuracy("DA")
    assert out["n_total"] == 1
    assert out["n_decided"] == 0
    assert out["accuracy"] is None


def test_accuracy_breaks_out_by_family(isolated_ledger):
    _seed(isolated_ledger, iteration_id="x1",
          critics=[_verdict("DA", "PASS")],
          pipeline="PROMOTE_TO_GATE", family="carry")
    _seed(isolated_ledger, iteration_id="x2",
          critics=[_verdict("DA", "PASS")],
          pipeline="HARD_REJECT", family="news_attention")
    out = compute_critic_accuracy("DA")
    assert out["by_family"]["carry"]["accuracy"] == 1.0
    assert out["by_family"]["news_attention"]["accuracy"] == 0.0


# ── compute_pairwise_critic_agreement ────────────────────────────────


def test_pairwise_agreement_full_match(isolated_ledger):
    for i in range(3):
        _seed(isolated_ledger, iteration_id=f"j{i}",
              critics=[_verdict("theorist", "PASS"),
                        _verdict("DA",       "PASS")],
              pipeline="PROMOTE_TO_GATE")
    out = compute_pairwise_critic_agreement()
    assert len(out["pairs"]) == 1
    p = out["pairs"][0]
    assert p["n_iterations"] == 3
    assert p["verdict_agreement_pct"] == 100.0
    assert p["alignment_agreement_pct"] == 100.0


def test_pairwise_agreement_partial(isolated_ledger):
    # 2 match, 1 mismatch
    _seed(isolated_ledger, iteration_id="k1",
          critics=[_verdict("theorist", "PASS"), _verdict("DA", "PASS")],
          pipeline="PROMOTE_TO_GATE")
    _seed(isolated_ledger, iteration_id="k2",
          critics=[_verdict("theorist", "PASS"), _verdict("DA", "PASS")],
          pipeline="PROMOTE_TO_GATE")
    _seed(isolated_ledger, iteration_id="k3",
          critics=[_verdict("theorist", "PASS"), _verdict("DA", "FAIL")],
          pipeline="HARD_REJECT")
    out = compute_pairwise_critic_agreement()
    p = out["pairs"][0]
    assert p["n_iterations"] == 3
    assert p["verdict_agreement_pct"] == round(2/3 * 100, 1)


# ── compute_critic_marginal_info ─────────────────────────────────────


def test_marginal_info_when_critic_carries_signal(isolated_ledger):
    """Set up so theorist is the lone correct vote when consensus
    should reject — without theorist, council would APPROVE wrongly."""
    # iteration: theorist=FAIL (correct), DA=PASS (wrong), pipeline=HARD_REJECT
    # full council: FAIL → REJECT ⇒ agree
    # without theorist: just PASS → APPROVE ⇒ council_wrong
    for i in range(25):  # need ≥ 20 decided
        _seed(isolated_ledger, iteration_id=f"m{i}",
              critics=[_verdict("theorist", "FAIL"),
                        _verdict("DA",       "PASS")],
              pipeline="HARD_REJECT",
              consensus="REJECT")
    out = compute_critic_marginal_info("theorist")
    assert out["full_council_accuracy"] == 1.0
    assert out["without_critic_accuracy"] == 0.0
    assert out["marginal_information_gain"] == 1.0
    assert "ADDS material" in out["interpretation"]


def test_marginal_info_when_critic_is_redundant(isolated_ledger):
    """Both critics say PASS, pipeline promotes. Removing either
    leaves the same verdict → marginal == 0."""
    for i in range(25):
        _seed(isolated_ledger, iteration_id=f"r{i}",
              critics=[_verdict("theorist", "PASS"),
                        _verdict("DA",       "PASS")],
              pipeline="PROMOTE_TO_GATE")
    out = compute_critic_marginal_info("theorist")
    assert out["marginal_information_gain"] == 0.0
    assert "REDUNDANT" in out["interpretation"]


def test_marginal_info_insufficient_data(isolated_ledger):
    _seed(isolated_ledger, iteration_id="few",
          critics=[_verdict("theorist", "PASS"), _verdict("DA", "PASS")],
          pipeline="PROMOTE_TO_GATE")
    out = compute_critic_marginal_info("theorist")
    assert "low confidence" in out["interpretation"]


# ── _aggregate (consensus replica) ───────────────────────────────────


def test_aggregate_replica_matches_council_rules():
    assert _aggregate(["PASS", "PASS"]) == "APPROVE"
    assert _aggregate(["PASS", "WARN"]) == "NEEDS_REVISION"
    assert _aggregate(["PASS", "FAIL"]) == "REJECT"
    assert _aggregate(["FAIL", "FAIL"]) == "REJECT"
    assert _aggregate([]) == "REJECT"  # conservative default


# ── End-to-end report ────────────────────────────────────────────────


def test_critic_calibration_report_includes_all_critics(isolated_ledger):
    _seed(isolated_ledger, iteration_id="r1",
          critics=[_verdict("theorist", "PASS"),
                    _verdict("DA",       "PASS"),
                    _verdict("book_critic", "FAIL")],
          pipeline="PROMOTE_TO_GATE")
    out = critic_calibration_report()
    assert out["n_distinct_critics"] == 3
    assert set(out["per_critic"].keys()) == {"theorist", "DA", "book_critic"}
    assert "pairwise_agreement" in out


# ── Integration: outcome_ledger emits critic rows ────────────────────


def test_outcome_ledger_emits_critic_rows(tmp_path, monkeypatch):
    """Writing an l4_iteration via append_l4_iteration must ALSO emit
    per-critic rows (Frontier A hook)."""
    fake_l4   = tmp_path / "l4_iter.jsonl"
    fake_crit = tmp_path / "critic_cal.jsonl"
    monkeypatch.setattr(
        "engine.research.outcome_ledger.L4_LEDGER_PATH", fake_l4,
    )
    monkeypatch.setattr(
        "engine.research.critic_calibration.CRITIC_CALIBRATION_LEDGER",
        fake_crit,
    )
    from engine.research.outcome_ledger import append_l4_iteration
    append_l4_iteration(
        workflow_id="wf-1",
        proposal={"family": "earnings_drift",
                   "proposed_role": "alpha_seeker"},
        council={"consensus": "APPROVE",
                  "verdicts": [_verdict("theorist", "PASS"),
                                _verdict("DA",       "PASS")],
                  "rationale": "clean"},
        pipeline_report={"final_decision": "PROMOTE_TO_GATE", "ran": True,
                          "step_results": []},
    )
    assert fake_l4.exists()
    assert fake_crit.exists()
    crit_rows = [json.loads(l)
                  for l in fake_crit.read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    assert len(crit_rows) == 2
    assert {r["critic_agent_name"] for r in crit_rows} == {"theorist", "DA"}
    assert all(r["alignment"] == "agree" for r in crit_rows)


# ── _interpret_marginal ──────────────────────────────────────────────


def test_interpret_marginal_buckets():
    assert "insufficient data" in _interpret_marginal(None, 100)
    assert "low confidence" in _interpret_marginal(0.1, 5)
    assert "ADDS material" in _interpret_marginal(0.10, 100)
    assert "REDUNDANT" in _interpret_marginal(0.00, 100)
    assert "HURTS" in _interpret_marginal(-0.05, 100)
