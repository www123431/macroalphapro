"""Tests for engine.research.rbg — Week 5 Research Brief Generator.

Covers:
  - Deterministic skeleton (sections from input data, no fabrication)
  - LLM-off mode produces full brief
  - LLM-on mode mocked so no network / cost
  - Validation flags fabricated evidence_ids
  - Persistence: write_brief_to_disk + sidecar metadata
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from engine.research.rbg.brief_generator import (
    _collect_evidence_ids,
    _section_evidence,
    _section_metrics,
    _section_predicted_concerns,
    _section_run_commands,
    _section_stop_criteria,
    _validate_prose_against_evidence,
    generate_brief,
    write_brief_to_disk,
)


def _fake_scored(**overrides) -> dict:
    """Build a representative ScoredProposal-like dict for testing."""
    base = {
        "proposal": {
            "candidate_id":      "pfh_constrained_test",
            "proposal_kind":     "constrained",
            "family_normalized": "momentum",
            "universe":          "equity_us_crsp_monthly",
            "signal_recipe":     "momentum_12_1",
            "weighting":         "decile_ls_10",
            "rebalance":         "monthly",
            "derived_from":      [],
            "cousin_warnings":   [],
            "needs_new_axes":    [],
            "rationale_seeds":   [],
        },
        "posterior": {
            "posterior_mean": 0.42,
            "credible_05":    0.18,
            "credible_50":    0.41,
            "credible_95":    0.71,
            "alpha_post":     2.5,
            "beta_post":      3.5,
            "n_green":        2,
            "n_yellow":       0,
            "n_red":          1,
        },
        "cousin_penalty": 1.0,
        "final_score":    0.42,
        "score_breakdown": {
            "family":         "momentum",
            "base_rate":      0.20,
            "prior_strength": 4.0,
            "cell_n_green":   2,
            "cell_n_yellow":  0,
            "cell_n_red":     1,
            "credible_05_95": [0.18, 0.71],
        },
    }
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k].update(v)
        else:
            base[k] = v
    return base


def _fake_materialized(sharpe: float = 0.41, vol: float = 0.18) -> dict:
    return {
        "spec_id":    "pfh_constrained_test",
        "validation": {
            "ok":                  True,
            "observed_n_rows":     129,
            "observed_start":      "2013-10-31",
            "observed_end":        "2024-06-30",
            "observed_ann_sharpe": sharpe,
            "observed_ann_vol":    vol,
        },
    }


# ── Evidence collection ────────────────────────────────────────────


def test_collect_evidence_ids_includes_axes():
    scored = _fake_scored()
    ids = _collect_evidence_ids(scored)
    assert "momentum" in ids
    assert "equity_us_crsp_monthly" in ids
    assert "momentum_12_1" in ids
    assert "decile_ls_10" in ids


def test_collect_evidence_ids_includes_derived_from():
    scored = _fake_scored()
    scored["proposal"]["derived_from"] = ["post_earnings_drift_pit_sn"]
    ids = _collect_evidence_ids(scored)
    assert "post_earnings_drift_pit_sn" in ids


# ── Deterministic skeleton ──────────────────────────────────────────


def test_section_metrics_includes_all_headline_numbers():
    text = _section_metrics(_fake_scored(), _fake_materialized())
    for needle in ("0.42", "0.18", "0.71", "Sharpe", "0.4100", "129"):
        assert needle in text, f"missing {needle!r}"


def test_section_metrics_handles_no_materialized():
    text = _section_metrics(_fake_scored(), None)
    assert "Sharpe" not in text  # only present when materialized given


def test_section_evidence_includes_axes_in_backticks():
    text = _section_evidence(_fake_scored())
    assert "`equity_us_crsp_monthly`" in text
    assert "`momentum_12_1`" in text
    assert "`decile_ls_10`" in text


def test_section_evidence_lists_cousin_warnings():
    scored = _fake_scored()
    scored["proposal"]["cousin_warnings"] = [
        "GRAVEYARD WARNING: 6 RED entries in family earnings",
    ]
    text = _section_evidence(scored)
    assert "GRAVEYARD WARNING" in text


def test_predicted_concerns_fires_on_wide_credible_interval():
    scored = _fake_scored()
    # CI width = 0.71 - 0.18 = 0.53 > 0.5 trigger
    text = _section_predicted_concerns(scored)
    assert "Behavioral theorist" in text


def test_predicted_concerns_fires_on_no_green_in_cell():
    scored = _fake_scored()
    scored["score_breakdown"]["cell_n_green"] = 0
    text = _section_predicted_concerns(scored)
    assert "publication-bias" in text


def test_predicted_concerns_fires_on_cousin_warnings():
    scored = _fake_scored()
    scored["proposal"]["cousin_warnings"] = ["w1", "w2"]
    text = _section_predicted_concerns(scored)
    assert "Devil's advocate" in text
    assert "cousin warning" in text


def test_run_commands_uses_actual_spec_id():
    text = _section_run_commands(_fake_scored())
    assert "pfh_constrained_test" in text
    assert "materialize_spec" in text
    assert "run_full_council" in text
    # CRITICAL: no unresolved template literals (this was a real bug)
    assert "{p.get(" not in text
    assert "{family" not in text
    assert "<FILL_IN>" not in text


def test_stop_criteria_flags_negative_sharpe():
    scored = _fake_scored()
    mat = _fake_materialized(sharpe=-0.30)
    text = _section_stop_criteria(scored, mat)
    assert "Already failing" in text


def test_stop_criteria_no_flag_on_positive_sharpe():
    scored = _fake_scored()
    mat = _fake_materialized(sharpe=0.40)
    text = _section_stop_criteria(scored, mat)
    assert "Already failing" not in text


# ── End-to-end generate_brief ──────────────────────────────────────


def test_generate_brief_structured_only_full_pipeline():
    """LLM off → full markdown emitted, no warnings, used_llm=False."""
    scored = _fake_scored()
    art = generate_brief(scored, materialized=_fake_materialized(),
                          use_llm=False)
    assert art.used_llm is False
    assert "Research Brief" in art.markdown
    assert "Headline metrics" in art.markdown
    assert "Evidence chain" in art.markdown
    assert "Predicted council concerns" in art.markdown
    assert "Run this now" in art.markdown
    assert "Stop criteria" in art.markdown
    assert art.validation_warnings == []


def test_generate_brief_with_llm_mocked():
    """LLM on (mocked) → prose section added, used_llm=True."""
    fake_block = mock.MagicMock(type="text",
                                  text="The hypothesis tests whether "
                                       "momentum returns persist in "
                                       "equity_us_crsp_monthly post-2014.")
    fake_resp = mock.MagicMock(content=[fake_block])
    fake_client = mock.MagicMock()
    fake_client.messages.create.return_value = fake_resp

    with mock.patch(
        "engine.research.rbg.brief_generator._load_anthropic_key",
        return_value="fake-key",
    ), mock.patch("anthropic.Anthropic", return_value=fake_client):
        art = generate_brief(_fake_scored(),
                              materialized=_fake_materialized(),
                              use_llm=True)
    assert art.used_llm is True
    assert "## Hypothesis" in art.markdown
    assert "momentum returns persist" in art.markdown


def test_generate_brief_no_api_key_fallback():
    """No API key → structured-only mode without raise."""
    with mock.patch(
        "engine.research.rbg.brief_generator._load_anthropic_key",
        return_value=None,
    ):
        art = generate_brief(_fake_scored(), use_llm=True)
    assert art.used_llm is False
    # The structured fallback message is included in the prose section
    assert "Structured-only mode" in art.markdown


# ── Validation ──────────────────────────────────────────────────────


def test_validate_prose_catches_fabricated_id():
    """LLM that invents a snake_case ID not in evidence should warn."""
    evidence = {"momentum_12_1", "equity_us_crsp_monthly"}
    fabricated_prose = "This factor relates to `fabricated_factor_id`."
    warnings = _validate_prose_against_evidence(fabricated_prose, evidence)
    assert warnings, "expected warning for fabricated_factor_id"


def test_validate_prose_clean_when_only_real_ids():
    evidence = {"momentum_12_1", "equity_us_crsp_monthly"}
    clean_prose = "Tests `momentum_12_1` on `equity_us_crsp_monthly`."
    warnings = _validate_prose_against_evidence(clean_prose, evidence)
    assert warnings == []


# ── Persistence ────────────────────────────────────────────────────


def test_write_brief_to_disk_creates_md_and_sidecar(tmp_path):
    art = generate_brief(_fake_scored(), use_llm=False)
    out_path = write_brief_to_disk(art, out_dir=tmp_path)
    assert out_path.is_file()
    assert out_path.suffix == ".md"
    sidecar = out_path.with_suffix(".meta.json")
    assert sidecar.is_file()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["spec_id"] == "pfh_constrained_test"
    assert meta["used_llm"] is False
    assert "warnings" in meta
    assert "evidence_ids" in meta


def test_write_brief_persists_full_markdown(tmp_path):
    art = generate_brief(_fake_scored(),
                          materialized=_fake_materialized(),
                          use_llm=False)
    out_path = write_brief_to_disk(art, out_dir=tmp_path)
    content = out_path.read_text(encoding="utf-8")
    # Sanity: markdown body matches
    assert content == art.markdown
