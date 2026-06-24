"""belief-4 closed-loop prior calibration tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research import belief_prior_calibration as bpc


def _write_autopsies(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _autopsy_row(family: str, actual: str, aid: str = "a",
                   n_obs_months: int = 360) -> dict:
    return {
        "autopsy_id":         aid,
        "ts":                 "2026-06-12T00:00:00Z",
        "prediction_id":      "p-" + aid,
        "verdict_event_id":   "e-" + aid,
        "subject_id":         "s-" + aid,
        "strategy_family":    family,
        "actual_verdict":     actual,
        "brier_component":    0.20,
        "surprise_direction": "well_calibrated",
        "surprise_magnitude": 0.0,
        "n_obs_months":       n_obs_months,
    }


# ── below threshold ───────────────────────────────────────────────


def test_returns_none_below_min_autopsies(tmp_path):
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row("TEST_FAM", "GREEN", aid=f"a{i}")
        for i in range(4)        # 4 < MIN_AUTOPSIES_FOR_OVERRIDE (5)
    ])
    assert bpc.calibrated_family_prior("TEST_FAM", autopsies_path=ap) is None


def test_returns_none_when_file_missing(tmp_path):
    ap = tmp_path / "missing.jsonl"   # not created
    assert bpc.calibrated_family_prior("TEST_FAM", autopsies_path=ap) is None


def test_returns_none_when_family_not_present(tmp_path):
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row("OTHER_FAM", "GREEN", aid=f"a{i}")
        for i in range(10)
    ])
    assert bpc.calibrated_family_prior("TEST_FAM", autopsies_path=ap) is None


# ── Dirichlet posterior math ──────────────────────────────────────


def test_calibrated_prior_sums_to_one(tmp_path):
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row("TEST_FAM", "GREEN", aid=f"g{i}") for i in range(3)
    ] + [
        _autopsy_row("TEST_FAM", "MARGINAL", aid=f"m{i}") for i in range(2)
    ])
    out = bpc.calibrated_family_prior("TEST_FAM", autopsies_path=ap)
    assert out is not None
    s = sum(out.values())
    assert abs(s - 1.0) < 1e-9


def test_calibrated_prior_shifts_toward_observed(tmp_path):
    """5 RED observations should push the prior toward RED relative to
    the base prior."""
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row("UNKNOWN_FAM_FOR_TEST", "RED", aid=f"r{i}")
        for i in range(8)
    ])
    out = bpc.calibrated_family_prior("UNKNOWN_FAM_FOR_TEST", autopsies_path=ap)
    assert out is not None
    # Base prior is DEFAULT (GREEN=0.20). 8 RED observations + pseudo-count
    # should shift posterior RED up significantly.
    assert out["RED"] > 0.55   # base 0.40 → posterior pushed materially higher
    assert out["GREEN"] < 0.20   # base 0.20 → squeezed down


def test_calibrated_prior_smooths_at_low_n(tmp_path):
    """With only 5 autopsies of one verdict, posterior should still
    keep meaningful weight on other verdicts via the Dirichlet pseudo-count."""
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row("UNKNOWN_FAM_SMOOTH", "GREEN", aid=f"g{i}")
        for i in range(5)
    ])
    out = bpc.calibrated_family_prior("UNKNOWN_FAM_SMOOTH", autopsies_path=ap)
    assert out is not None
    # 5 GREEN + alpha (3) * default GREEN (0.20) = 5.6 / (5+3) = 0.70
    # Not 1.0 — Dirichlet smoothing preserves uncertainty
    assert 0.65 < out["GREEN"] < 0.80
    # Other verdicts still have some weight (from pseudo-count of base)
    assert out["MARGINAL"] > 0.10
    assert out["RED"] > 0.10


def test_calibrated_prior_uses_family_override_as_base(tmp_path):
    """When family has FAMILY_PRIOR_OVERRIDES entry, those become the
    base for Dirichlet pseudo-counts."""
    ap = tmp_path / "autopsies.jsonl"
    # PROFITABILITY has override {GREEN: 0.12, MARGINAL: 0.50, RED: 0.38}
    # 5 GREEN autopsies + base pseudo → posterior GREEN should still be
    # higher than the override's 0.12 (observations move it)
    _write_autopsies(ap, [
        _autopsy_row("PROFITABILITY", "GREEN", aid=f"g{i}") for i in range(5)
    ])
    out = bpc.calibrated_family_prior("PROFITABILITY", autopsies_path=ap)
    assert out is not None
    # 5 GREEN + 3 * 0.12 = 5.36 / (5+3) = 0.67
    assert 0.60 < out["GREEN"] < 0.75


# ── BUG-4 precision weighting ─────────────────────────────────────


def test_bug4_long_sample_weighs_more_than_short(tmp_path):
    """5 GREENs at N=720mo should give higher GREEN posterior than 5
    GREENs at N=60mo because the long-sample autopsies carry more
    statistical weight per Bayesian precision (1/SE^2 ∝ N)."""
    long_ap  = tmp_path / "long.jsonl"
    short_ap = tmp_path / "short.jsonl"
    _write_autopsies(long_ap, [
        _autopsy_row("LONG_FAM", "GREEN", aid=f"l{i}", n_obs_months=720)
        for i in range(5)
    ])
    _write_autopsies(short_ap, [
        _autopsy_row("SHORT_FAM", "GREEN", aid=f"s{i}", n_obs_months=60)
        for i in range(5)
    ])
    long_prior  = bpc.calibrated_family_prior("LONG_FAM",  autopsies_path=long_ap)
    short_prior = bpc.calibrated_family_prior("SHORT_FAM", autopsies_path=short_ap)
    assert long_prior is not None
    assert short_prior is not None
    # Long-sample 5×720mo evidence drives GREEN posterior way up.
    # Short-sample 5×60mo evidence is heavily smoothed back toward base prior.
    assert long_prior["GREEN"] > short_prior["GREEN"], (
        f"long-sample GREEN posterior {long_prior['GREEN']:.3f} "
        f"should exceed short-sample {short_prior['GREEN']:.3f}"
    )


def test_bug4_missing_n_obs_falls_back_to_unit_weight(tmp_path):
    """Autopsies without n_obs_months default to weight 1.0 (treat as
    reference sample size). This maintains backward compat with
    historical autopsies that didn't carry the field."""
    ap = tmp_path / "no_n.jsonl"
    rows = [
        _autopsy_row("NO_N_FAM", "MARGINAL", aid=f"a{i}", n_obs_months=0)
        for i in range(5)
    ]
    _write_autopsies(ap, rows)
    out = bpc.calibrated_family_prior("NO_N_FAM", autopsies_path=ap)
    assert out is not None   # 5 autopsies clear threshold; not None
    # Posterior shifts toward MARGINAL (5 unit-weight observations)
    assert out["MARGINAL"] > 0.5


# ── calibration_summary ───────────────────────────────────────────


def test_calibration_summary_below_threshold(tmp_path):
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row("TEST_X", "MARGINAL", aid=f"a{i}") for i in range(3)
    ])
    out = bpc.calibration_summary("TEST_X", autopsies_path=ap)
    assert out["n_autopsies"] == 3
    assert out["override_active"] is False
    assert out["calibrated_prior"] is None
    assert out["base_prior"] is not None


def test_calibration_summary_active_override(tmp_path):
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row("TEST_Y", "GREEN", aid=f"g{i}") for i in range(7)
    ])
    out = bpc.calibration_summary("TEST_Y", autopsies_path=ap)
    assert out["n_autopsies"] == 7
    assert out["override_active"] is True
    assert out["calibrated_prior"] is not None
    assert out["observed_counts"] == {"GREEN": 7, "MARGINAL": 0, "RED": 0}
    # delta_green should be POSITIVE (7 GREEN observations push prior up)
    assert out["delta_green"] > 0.0


# ── belief-1 integration smoke ────────────────────────────────────


def test_belief1_predict_uses_belief4_when_autopsies_sufficient(tmp_path, monkeypatch):
    """belief-1 predict_verdict should pick up the calibrated prior
    when ≥5 autopsies exist in the family."""
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row("TEST_BELIEF1_INT", "RED", aid=f"r{i}") for i in range(6)
    ])
    monkeypatch.setattr(bpc, "AUTOPSIES_PATH", ap)

    # Also need belief.py to NOT find historical observations that
    # override (factor_verdict_filed events for this family) — those
    # are read independently. For the smoke we patch _family_observed_dist
    # to return N=0 so belief-4 path is taken cleanly.
    from engine.research import belief as belief_module
    monkeypatch.setattr(
        belief_module, "_family_observed_dist",
        lambda fam: (dict(belief_module.DEFAULT_PRIOR), 0),
    )

    pred = belief_module.predict_verdict(
        subject_id="t-belief4",
        family="TEST_BELIEF1_INT",
    )
    # Without belief-4: default prior GREEN=0.20.
    # With belief-4 + 6 RED autopsies: RED shifts way up
    assert "calibration_source:belief_4" in pred.predicted_load_bearing
    assert pred.predicted_verdict_dist["RED"] > 0.45  # shifted from base 0.40
