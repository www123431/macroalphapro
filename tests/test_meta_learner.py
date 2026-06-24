"""Tests for engine.research.meta_learner — Phase 6.5 Tier 1."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from engine.research import meta_learner
from engine.research.meta_learner import (
    COLD_START_PRIORS, FamilyPosterior, GateObservation, MetaLearner,
    _classify_outcome, annotate_candidate,
)


# ── outcome classification ─────────────────────────────────────────────────

def test_classify_green_is_pass():
    assert _classify_outcome({"verdict": "GREEN"}) == "pass"


def test_classify_green_with_qualifier_is_pass():
    assert _classify_outcome({"verdict": "GREEN — 4/4 strict bars cleared"}) == "pass"


def test_classify_yellow_is_yellow():
    assert _classify_outcome({"verdict": "YELLOW"}) == "yellow"


def test_classify_red_is_fail():
    assert _classify_outcome({"verdict": "RED"}) == "fail"


def test_classify_missing_is_fail():
    assert _classify_outcome({}) == "fail"
    assert _classify_outcome({"verdict": None}) == "fail"


# ── MetaLearner update/predict ────────────────────────────────────────────

def test_predict_unseen_family_uses_cold_start():
    ml = MetaLearner()
    p = ml.predict("carry")
    a, b = COLD_START_PRIORS["carry"]
    assert p.alpha == a and p.beta == b
    assert p.n_observations == 0
    assert p.prior_source == "cold_start"
    assert 0.0 < p.mean < 1.0


def test_predict_unknown_family_uses_unknown_prior():
    ml = MetaLearner()
    p = ml.predict("totally_made_up_family")
    a, b = COLD_START_PRIORS["unknown"]
    assert p.alpha == a and p.beta == b


def test_update_pass_increases_alpha():
    ml = MetaLearner()
    p0 = ml.predict("carry")
    ml.update("carry", "pass")
    p1 = ml.predict("carry")
    assert p1.alpha == p0.alpha + 1
    assert p1.beta == p0.beta
    assert p1.mean > p0.mean
    assert p1.n_observations == 1


def test_update_fail_increases_beta():
    ml = MetaLearner()
    p0 = ml.predict("carry")
    ml.update("carry", "fail")
    p1 = ml.predict("carry")
    assert p1.beta == p0.beta + 1
    assert p1.alpha == p0.alpha
    assert p1.mean < p0.mean


def test_update_yellow_default_is_failure_like():
    """YELLOW defaults to fail-like contribution unless overridden."""
    ml = MetaLearner(yellow_counts_as_pass=False)
    p0 = ml.predict("carry")
    ml.update("carry", "yellow")
    p1 = ml.predict("carry")
    assert p1.mean < p0.mean
    assert p1.failures == 0    # tracked as yellow separately
    assert p1.yellows == 1


def test_update_yellow_as_pass_mode():
    ml = MetaLearner(yellow_counts_as_pass=True)
    p0 = ml.predict("carry")
    ml.update("carry", "yellow")
    p1 = ml.predict("carry")
    assert p1.mean > p0.mean


def test_invalid_outcome_raises():
    ml = MetaLearner()
    with pytest.raises(ValueError):
        ml.update("carry", "neutral")


# ── bulk_update + observations ────────────────────────────────────────────

def test_bulk_update_from_observations():
    ml = MetaLearner()
    obs = [
        GateObservation("m1", "carry", "pass", 0.7, 0.95, 3.1, "2024-01-01"),
        GateObservation("m2", "carry", "fail", 0.1, 0.05, 0.5, "2024-02-01"),
        GateObservation("m3", "tsmom", "pass", 0.6, 0.85, 2.9, "2024-03-01"),
    ]
    ml.bulk_update(obs)
    p_carry = ml.predict("carry")
    assert p_carry.successes == 1
    assert p_carry.failures == 1
    p_tsmom = ml.predict("tsmom")
    assert p_tsmom.successes == 1
    assert p_tsmom.failures == 0


# ── credible interval ─────────────────────────────────────────────────────

def test_credible_interval_widens_with_few_obs():
    """Wide CI under sparse evidence."""
    ml = MetaLearner()
    p = ml.predict("unknown")
    lo, hi = p.credible_interval(0.95)
    assert hi - lo > 0.2     # wide on cold start


def test_credible_interval_narrows_with_many_obs():
    ml = MetaLearner()
    for _ in range(100):
        ml.update("test_family", "pass")
    for _ in range(50):
        ml.update("test_family", "fail")
    p = ml.predict("test_family")
    lo, hi = p.credible_interval(0.95)
    assert hi - lo < 0.15      # tight after 150 obs


def test_credible_interval_bounded_0_1():
    ml = MetaLearner()
    p = ml.predict("carry")
    lo, hi = p.credible_interval(0.95)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert lo <= hi


# ── compare ───────────────────────────────────────────────────────────────

def test_compare_ranks_higher_prior_first():
    ml = MetaLearner()
    ranked = ml.compare(["news_attention", "carry", "tsmom", "value"])
    # carry has highest cold-start (30%); news_attention has lowest (5%)
    assert ranked[0].family == "carry"
    assert ranked[-1].family == "news_attention"


# ── from_disk ─────────────────────────────────────────────────────────────

def test_from_disk_with_tmp_gate_runs(tmp_path):
    gate_path = tmp_path / "gate_runs.jsonl"
    with gate_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"mechanism": "m_a", "family": "carry",
                              "verdict": "GREEN"}) + "\n")
        f.write(json.dumps({"mechanism": "m_b", "family": "carry",
                              "verdict": "RED"}) + "\n")
        f.write(json.dumps({"mechanism": "m_c", "family": "quality",
                              "verdict": "RED"}) + "\n")
    ml = MetaLearner.from_disk(gate_path=gate_path, family_map={})
    assert ml.predict("carry").successes == 1
    assert ml.predict("carry").failures == 1
    assert ml.predict("quality").failures == 1


def test_from_disk_uses_library_lookup_when_no_family(tmp_path, monkeypatch):
    """If gate run has no family field, fall back to library YAML lookup."""
    gate_path = tmp_path / "gate_runs.jsonl"
    with gate_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"mechanism": "m_a", "verdict": "GREEN"}) + "\n")
    family_map = {"m_a": "carry"}
    ml = MetaLearner.from_disk(gate_path=gate_path, family_map=family_map)
    assert ml.predict("carry").successes == 1


def test_from_disk_missing_mechanism_id_is_skipped(tmp_path):
    gate_path = tmp_path / "gate_runs.jsonl"
    with gate_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"verdict": "GREEN"}) + "\n")        # no mechanism
        f.write(json.dumps({"mechanism": "m_a", "verdict": "RED"}) + "\n")
    ml = MetaLearner.from_disk(gate_path=gate_path, family_map={})
    # only m_a → "unknown" family
    assert ml.predict("unknown").failures == 1


# ── annotate_candidate (advisory contract) ────────────────────────────────

def test_annotate_returns_required_advisory_fields():
    ml = MetaLearner()
    ann = annotate_candidate("carry", ml=ml)
    assert "prior_pass_probability" in ann
    assert "credible_interval_95" in ann
    assert "observations_in_family" in ann
    assert "advisory_note" in ann
    assert "ADVISORY" in ann["advisory_note"]


def test_annotate_never_returns_verdict_or_passcriteria():
    """STRICT RED LINE: meta-learner cannot output anything resembling a
    decision. annotate_candidate must NOT contain 'verdict', 'pass_criteria',
    'recommendation', etc."""
    ml = MetaLearner()
    ann = annotate_candidate("carry", ml=ml)
    forbidden = ("verdict", "pass_criteria", "decision", "recommendation",
                  "auto_deploy", "should_run")
    keys_lower = " ".join(ann.keys()).lower()
    for f in forbidden:
        assert f not in keys_lower, f"forbidden field {f!r} in annotation"


# ── summary / all_families ─────────────────────────────────────────────────

def test_summary_includes_all_cold_start_families():
    ml = MetaLearner()
    families_in_summary = {row["family"] for row in ml.summary()}
    assert "carry" in families_in_summary
    assert "tsmom" in families_in_summary
    assert "unknown" in families_in_summary


def test_observed_families_only_returns_seen():
    ml = MetaLearner()
    ml.update("carry", "pass")
    ml.update("quality", "fail")
    assert set(ml.observed_families()) == {"carry", "quality"}


# ── prior_source field signals data vs cold-start ─────────────────────────

def test_prior_source_indicates_data_vs_cold_start():
    ml = MetaLearner()
    assert ml.predict("carry").prior_source == "cold_start"
    ml.update("carry", "pass")
    assert ml.predict("carry").prior_source == "cold_start+data"
