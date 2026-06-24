"""Tests for engine.research.pfh — Probabilistic Factor Hypothesizer.

Covers the senior-critique design points:
  1. Deterministic catalog
  2. YELLOW counts as 0.5 each direction in base rate
  3. Family alias bridges PEAD library/graveyard cells
  4. Beta-Binomial posterior shrinks toward base rate at small n
  5. Credible interval contains posterior mean
  6. Cousin penalty multiplies correctly + capped at 0.05
  7. Diversification cap enforces max_per_family
  8. PFH-emitted compose-spec YAML is loadable by composer
  9. Determinism: identical input → identical output

Plus integration: PFH end-to-end against real repo data, asserting top-K
spans multiple families.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from engine.research.pfh.bayesian import (
    BetaBinomialPosterior, _hyperprior_alpha_beta, score_candidate,
)
from engine.research.pfh.catalog import (
    LabeledMechanism, _FAMILY_ALIASES, _infer_market_from_name,
    _normalize_family, load_labeled_mechanisms,
    overall_base_rate, per_family_counts,
)
from engine.research.pfh.generator import (
    CandidateProposal, generate_candidates,
)
from engine.research.pfh.proposer import (
    _diversify_top_k, suggest_top_k,
)


# ── Catalog ──────────────────────────────────────────────────────────


def test_load_labeled_mechanisms_deterministic():
    """Same disk state → byte-identical output across calls."""
    a = load_labeled_mechanisms()
    b = load_labeled_mechanisms()
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.to_dict() == y.to_dict()


def test_labeled_includes_expected_greens_and_reds():
    """Headline check: the 6 GREEN library entries + the 24 graveyard
    RED entries are all present, with correct verdicts."""
    labels = load_labeled_mechanisms()
    by_verdict: dict[str, set[str]] = {"GREEN": set(), "YELLOW": set(),
                                         "RED": set()}
    for m in labels:
        by_verdict[m.verdict].add(m.name)
    expected_green = {
        "post_earnings_drift", "post_earnings_drift_pit_sn",
        "cross_asset_carry", "time_series_momentum",
        "crisis_hedge_tlt_gld", "tail_hedge_put_spread",
    }
    assert expected_green.issubset(by_verdict["GREEN"])
    # Graveyard contributes 24 RED minimum (24 graveyard + 4 library RED markers)
    assert len(by_verdict["RED"]) >= 24


def test_family_alias_bridges_pead():
    """forward-earnings information (graveyard) must alias to
    earnings_underreaction (library) — these are the same mechanism."""
    assert _normalize_family("forward-earnings information") \
            == "earnings_underreaction"
    # 6 graveyard entries should now land in the earnings_underreaction cell
    labels = load_labeled_mechanisms()
    fams = per_family_counts(labels)
    eu = fams.get("earnings_underreaction")
    assert eu is not None
    assert eu["n_green"] == 2, "expected 2 GREEN (D_PEAD + PIT SN)"
    assert eu["n_red"] >= 6, ("expected ≥6 RED from forward-earnings "
                                "information graveyard family aliased in")


def test_alias_table_documents_kept_separate_families():
    """The 4 non-aliased graveyard families must stay separate
    (over-aggregation would corrupt the prior)."""
    assert "cross_sectional_equity_published" not in _FAMILY_ALIASES
    assert "macro_trend_risk_parity"          not in _FAMILY_ALIASES
    assert "text_machine_learning"            not in _FAMILY_ALIASES


def test_overall_base_rate_yellow_weighted_as_half():
    """YELLOW contributes 0.5 to n_eff_green, not 1.0 or 0."""
    fake_labels = [
        LabeledMechanism("a", "f", "f", None, None, None, "GREEN",
                          None, None, None, "test", "p"),
        LabeledMechanism("b", "f", "f", None, None, None, "YELLOW",
                          None, None, None, "test", "p"),
        LabeledMechanism("c", "f", "f", None, None, None, "RED",
                          None, None, None, "test", "p"),
    ]
    br = overall_base_rate(fake_labels)
    assert br["n_green"]     == 1
    assert br["n_yellow"]    == 1
    assert br["n_red"]       == 1
    assert br["n_eff_green"] == 1.5
    assert abs(br["p_green"] - 0.5) < 1e-6


def test_market_inference_handles_known_patterns():
    assert _infer_market_from_name("China A-share PEAD", "forward-earnings") \
            == "cn_equity"
    assert _infer_market_from_name("G10 sovereign curve", "carry") == "futures"
    assert _infer_market_from_name("FX dollar factor", "fx") == "fx"


# ── Bayesian scoring ─────────────────────────────────────────────────


def test_hyperprior_centered_on_base_rate():
    a, b = _hyperprior_alpha_beta(0.20, prior_strength=4.0)
    # mean of Beta(α, β) = α / (α + β) — should equal base_rate
    # when prior_strength is the dominant signal
    # α = 1 + 0.20 * 4 = 1.8
    # β = 1 + 0.80 * 4 = 4.2
    # mean = 1.8 / 6.0 = 0.30 (shifted from 0.20 due to the +1 offset)
    assert abs(a - 1.8) < 1e-9
    assert abs(b - 4.2) < 1e-9


def test_hyperprior_degenerate_base_rate():
    """base_rate=0 or 1 should fall back to Jeffreys prior (0.5, 0.5)
    rather than producing a degenerate posterior."""
    assert _hyperprior_alpha_beta(0.0)  == (0.5, 0.5)
    assert _hyperprior_alpha_beta(1.0)  == (0.5, 0.5)


def test_score_candidate_zero_data_returns_prior_mean():
    """With no observations, posterior == prior."""
    p = score_candidate(n_green=0, n_yellow=0, n_red=0, base_rate=0.20)
    # alpha_post = 1.8, beta_post = 4.2 → mean = 0.30
    assert abs(p.posterior_mean - 0.30) < 1e-3


def test_score_candidate_data_shifts_posterior_toward_mle():
    """Adding 10 GREEN with 0 RED should pull posterior well above base rate."""
    p = score_candidate(n_green=10, n_yellow=0, n_red=0, base_rate=0.20)
    # MLE = 1.0; weakly informative prior shouldn't dominate at n=10
    assert p.posterior_mean > 0.70


def test_credible_interval_contains_posterior_mean():
    p = score_candidate(n_green=2, n_yellow=0, n_red=6, base_rate=0.20)
    assert p.credible_05 <= p.posterior_mean <= p.credible_95


def test_credible_interval_strict_ordering():
    p = score_candidate(n_green=2, n_yellow=0, n_red=6, base_rate=0.20)
    assert p.credible_05 < p.credible_50 < p.credible_95


def test_yellow_in_score_candidate_splits_evenly():
    """1 YELLOW should equal 0.5 GREEN + 0.5 RED in the effective counts."""
    p_yellow = score_candidate(n_green=0, n_yellow=2, n_red=0, base_rate=0.20)
    p_split  = score_candidate(n_green=1, n_yellow=0, n_red=1, base_rate=0.20)
    assert abs(p_yellow.posterior_mean - p_split.posterior_mean) < 1e-9


# ── Generator ────────────────────────────────────────────────────────


def test_generator_extension_emits_for_each_green_family():
    """One extension proposal per GREEN family."""
    labels = load_labeled_mechanisms()
    fams = per_family_counts(labels)
    n_green_families = sum(1 for f, c in fams.items() if c["n_green"] > 0)
    candidates = generate_candidates(
        labels,
        include_cross_market=False,
        include_untested_families=False,
    )
    assert len(candidates) == n_green_families


def test_generator_cross_market_warnings_fire_when_red_in_target():
    """If a graveyard entry exists in (family, target_market), the
    cross-market candidate must surface a GRAVEYARD WARNING."""
    # Construct labels with a GREEN in earnings_underreaction × us_equity
    # AND a RED in earnings_underreaction × cn_equity
    fake = [
        LabeledMechanism("us_pead", "earnings_underreaction",
                          "earnings_underreaction", None, None,
                          "us_equity", "GREEN", None, None, None,
                          "test", "p"),
        LabeledMechanism("cn_pead", "earnings_underreaction",
                          "earnings_underreaction", None, None,
                          "cn_equity", "RED", "post-2014 dead",
                          None, None, "test", "p"),
    ]
    cands = generate_candidates(
        fake, include_extensions=False, include_untested_families=False,
    )
    cn_targets = [c for c in cands if "cn_equity" in c.candidate_id]
    assert cn_targets
    assert any("GRAVEYARD WARNING" in w for w in cn_targets[0].cousin_warnings)


def test_generator_untested_family_skips_already_green():
    """If a seed family is already in our labels as GREEN, skip it."""
    # Force an "already GREEN" carry into labels (mimics real state)
    fake = [
        LabeledMechanism("carry", "carry", "carry", None, None,
                          None, "GREEN", None, None, None,
                          "test", "p"),
    ]
    cands = generate_candidates(
        fake, include_extensions=False, include_cross_market=False,
    )
    fams = {c.family_normalized for c in cands}
    assert "carry" not in fams


# ── Diversification ─────────────────────────────────────────────────


def test_diversify_top_k_respects_per_family_cap():
    """No family appears more than max_per_family times in top-K."""
    # Use a large per-universe cap so this test isolates the family
    # cap (default open-mode uses cross-market candidates with varied
    # families and a common derived universe — universe cap can prevent
    # filling k under both caps without relaxation).
    out = suggest_top_k(k=6, max_per_family=2, max_per_universe=10)
    families = [s["proposal"]["family_normalized"] for s in out["top"]]
    from collections import Counter
    counts = Counter(families)
    for fam, n in counts.items():
        assert n <= 2, f"family {fam} appears {n} times (cap is 2)"


def test_diversify_top_k_relaxes_when_pool_smaller():
    """If candidate pool can't fill k under cap, fill remainder by raw rank."""
    # Build a synthetic pool with only 1 family
    from engine.research.pfh.bayesian import score_candidate
    from engine.research.pfh.generator import CandidateProposal
    from engine.research.pfh.proposer import ScoredProposal
    fake_props = [
        CandidateProposal(f"c{i}", "extension", "only_fam")
        for i in range(5)
    ]
    fake_post = score_candidate(n_green=1, n_yellow=0, n_red=0, base_rate=0.2)
    scored = [ScoredProposal(p, fake_post, 1.0, 0.5, {}) for p in fake_props]
    out = _diversify_top_k(scored, k=4, max_per_family=2)
    assert len(out) == 4  # capped pool relaxed to fill k


# ── End-to-end + compose spec YAML output ───────────────────────────


def test_suggest_top_k_returns_structured_dict():
    out = suggest_top_k(k=3, write_specs=False, write_ledger=False)
    for required in ("run_id", "ts", "base_rate_used",
                      "n_candidates_total", "top"):
        assert required in out
    assert len(out["top"]) <= 3
    for s in out["top"]:
        assert "proposal" in s
        assert "posterior" in s
        assert "final_score" in s


def test_suggest_top_k_no_ledger_no_specs_no_side_effects(tmp_path, monkeypatch):
    """write_ledger=False + write_specs=False produces no file writes."""
    fake_ledger = tmp_path / "pfh.jsonl"
    fake_specs  = tmp_path / "_specs"
    monkeypatch.setattr(
        "engine.research.pfh.proposer.PFH_LEDGER", fake_ledger,
    )
    monkeypatch.setattr(
        "engine.research.pfh.proposer.COMPOSE_SPECS_DIR", fake_specs,
    )
    suggest_top_k(k=3, write_specs=False, write_ledger=False)
    assert not fake_ledger.exists()
    assert not fake_specs.exists()


def test_pfh_compose_spec_yaml_well_formed(tmp_path, monkeypatch):
    """Emitted compose-spec YAML must be parseable + identifiable as
    PFH-originated."""
    fake_specs = tmp_path / "_specs"
    fake_specs.mkdir()
    monkeypatch.setattr(
        "engine.research.pfh.proposer.COMPOSE_SPECS_DIR", fake_specs,
    )
    out = suggest_top_k(k=2, write_specs=True, write_ledger=False)
    written = out["written_spec_paths"]
    assert written, "expected at least 1 spec emitted"
    for relpath in written:
        # YAML emitter writes to the tmp specs dir
        files = list(fake_specs.glob("*.yaml"))
        assert files
        for fp in files:
            raw = yaml.safe_load(fp.read_text(encoding="utf-8"))
            assert "compose" in raw
            assert raw["audit"]["added_by"] == "pfh"
            assert raw["audit"]["status"] == "pending_pfh_review"
            assert "pfh_run_id" in raw["audit"]
            assert "pfh_score" in raw["audit"]
            assert "pfh_evidence" in raw["audit"]


def test_top_k_spans_multiple_families_on_real_data():
    """Headline diversity test: on real repo data, top-6 must span ≥3
    distinct families. If this regresses we're back to mono-family
    output that the senior critique flagged."""
    out = suggest_top_k(k=6, write_specs=False, write_ledger=False)
    families = {s["proposal"]["family_normalized"] for s in out["top"]}
    assert len(families) >= 3, \
        f"only {len(families)} families in top-6: {families}"


def test_pfh_is_deterministic():
    """Same disk state ⇒ identical scoring (modulo run_id + ts)."""
    a = suggest_top_k(k=4, write_ledger=False, write_specs=False)
    b = suggest_top_k(k=4, write_ledger=False, write_specs=False)
    # Drop run_id + ts which are non-deterministic by design
    a_scores = [(s["proposal"]["candidate_id"], s["final_score"])
                 for s in a["top"]]
    b_scores = [(s["proposal"]["candidate_id"], s["final_score"])
                 for s in b["top"]]
    assert a_scores == b_scores


# ── Cousin penalty ──────────────────────────────────────────────────


def test_cousin_penalty_floors_at_0_05():
    """Heavily-warned candidates retain at least 0.05 of raw posterior."""
    from engine.research.pfh.proposer import _score_one_candidate
    cand = CandidateProposal(
        "test", "untested_family", "earnings_underreaction",
        cousin_warnings=["w1", "w2", "w3", "w4", "w5",
                          "w6", "w7", "w8", "w9", "w10"],
    )
    labels = load_labeled_mechanisms()
    fams = per_family_counts(labels)
    br = overall_base_rate(labels)
    s = _score_one_candidate(cand, labels, fams,
                              base_rate=br["p_green"] or 0.5,
                              prior_strength=4.0)
    assert s.cousin_penalty >= 0.05


def test_cousin_penalty_multiplicative_per_warning():
    """N warnings → penalty = max(0.05, 0.85^N)."""
    from engine.research.pfh.proposer import _COUSIN_PENALTY_PER_RED
    expected_3 = (1.0 - _COUSIN_PENALTY_PER_RED) ** 3
    from engine.research.pfh.proposer import _score_one_candidate
    cand = CandidateProposal(
        "test", "extension", "carry",
        cousin_warnings=["w1", "w2", "w3"],
    )
    labels = load_labeled_mechanisms()
    fams = per_family_counts(labels)
    br = overall_base_rate(labels)
    s = _score_one_candidate(cand, labels, fams,
                              base_rate=br["p_green"] or 0.5,
                              prior_strength=4.0)
    assert abs(s.cousin_penalty - expected_3) < 1e-6
