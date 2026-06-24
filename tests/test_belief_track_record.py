"""Smoke + invariant tests for engine.research.belief_track_record."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from engine.research.belief_track_record import (
    BASELINE_PERFECT,
    BASELINE_RANDOM_3CLASS,
    RELIABILITY_BINS,
    build_track_record,
)


def _mk_autopsy(*, predicted: dict[str, float], actual: str,
                  family: str = "VRP", ts: str = "2026-06-15T00:00:00Z",
                  superseded: str | None = None) -> dict:
    """Synthesize a minimal autopsy row for tests."""
    p_actual = predicted.get(actual, 0.0)
    return {
        "autopsy_id":             "a-" + actual + "-" + family,
        "ts":                     ts,
        "prediction_id":          "p-" + family,
        "verdict_event_id":       "e-" + family,
        "subject_id":             "hid-" + family,
        "strategy_family":        family,
        "claim_family":           family,
        "predicted_verdict_dist": predicted,
        "actual_verdict":         actual,
        "brier_component":        (1.0 - p_actual) ** 2,
        "surprise_direction":     "well_calibrated"
                                   if max(predicted, key=predicted.get) == actual
                                   else "neutral",
        "surprise_magnitude":     0.0,
        "load_bearing_realized":  [],
        "prediction_basis_echo":  "",
        "superseded_by":          superseded,
        "bug1_correction":        False,
        "n_obs_months":           0,
    }


def _write_autopsies(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "autopsies.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_empty_autopsies_returns_skeleton(tmp_path: Path):
    p = _write_autopsies(tmp_path, [])
    tr = build_track_record(autopsies_path=p)
    assert tr["n_autopsies"] == 0
    assert tr["mean_brier_overall"] is None
    assert tr["baselines"]["random_3class"] == BASELINE_RANDOM_3CLASS
    assert tr["baselines"]["perfect"] == BASELINE_PERFECT
    assert tr["family_breakdown"] == []


def test_perfect_predictor_gets_zero_brier(tmp_path: Path):
    rows = [
        _mk_autopsy(predicted={"GREEN": 1.0, "MARGINAL": 0.0, "RED": 0.0},
                     actual="GREEN", family="VRP"),
        _mk_autopsy(predicted={"GREEN": 0.0, "MARGINAL": 1.0, "RED": 0.0},
                     actual="MARGINAL", family="VRP"),
    ]
    p = _write_autopsies(tmp_path, rows)
    tr = build_track_record(autopsies_path=p)
    assert tr["n_autopsies"] == 2
    assert tr["mean_brier_overall"] == 0.0


def test_random_predictor_brier_at_baseline(tmp_path: Path):
    uniform = {"GREEN": 1/3, "MARGINAL": 1/3, "RED": 1/3}
    rows = [_mk_autopsy(predicted=uniform, actual="GREEN", family="A"),
            _mk_autopsy(predicted=uniform, actual="MARGINAL", family="A"),
            _mk_autopsy(predicted=uniform, actual="RED", family="A")]
    p = _write_autopsies(tmp_path, rows)
    tr = build_track_record(autopsies_path=p)
    assert abs(tr["mean_brier_overall"] - BASELINE_RANDOM_3CLASS) < 1e-5


def test_superseded_rows_excluded(tmp_path: Path):
    bad = _mk_autopsy(predicted={"GREEN": 0.0, "MARGINAL": 0.0, "RED": 1.0},
                       actual="GREEN", family="A", superseded="other-id")
    good = _mk_autopsy(predicted={"GREEN": 1.0, "MARGINAL": 0.0, "RED": 0.0},
                        actual="GREEN", family="A")
    p = _write_autopsies(tmp_path, [bad, good])
    tr = build_track_record(autopsies_path=p)
    assert tr["n_autopsies"] == 1
    assert tr["mean_brier_overall"] == 0.0


def test_family_breakdown_sorted_by_n_desc(tmp_path: Path):
    big = [_mk_autopsy(predicted={"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2},
                          actual="GREEN", family="BIG")
           for _ in range(5)]
    small = [_mk_autopsy(predicted={"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2},
                            actual="GREEN", family="SMALL")
             for _ in range(2)]
    p = _write_autopsies(tmp_path, big + small)
    tr = build_track_record(autopsies_path=p)
    assert tr["family_breakdown"][0]["family"] == "BIG"
    assert tr["family_breakdown"][0]["n"] == 5
    assert tr["family_breakdown"][1]["family"] == "SMALL"


def test_reliability_diagram_bin_count(tmp_path: Path):
    rows = [_mk_autopsy(predicted={"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2},
                          actual="GREEN", family="A")]
    p = _write_autopsies(tmp_path, rows)
    tr = build_track_record(autopsies_path=p)
    assert len(tr["reliability"]) == RELIABILITY_BINS
    populated = [b for b in tr["reliability"] if b["n"] > 0]
    assert len(populated) == 1


def test_ensemble_blend_flag_intentional_state():
    """W7-arxiv-v07: ensemble blend ACTIVATED 2026-06-22.

    Per the W7-arxiv-v05 sweep on 92 autopsies (34% in-sample Brier
    reduction with consistent per-family signal), the ensemble blend
    was activated by intentional principal decision documented in
    belief.py constant block + arxiv paper Section 4.6 amendment.

    If this test fails because the flag was flipped back to False,
    that's also fine — it should fail loudly so a deactivation gets
    documented the same way the activation did.
    """
    from engine.research import belief
    assert belief.BELIEF_ENSEMBLE_BLEND_ENABLED is True, (
        "Ensemble blend was activated 2026-06-22; if you're disabling it, "
        "update the comment block in belief.py + amend arxiv paper "
        "Section 4.6 with the deactivation date + reason."
    )


def test_ensemble_blend_off_preserves_existing_pipeline(monkeypatch):
    """When the flag is OFF, _apply_ensemble_blend is a no-op."""
    from engine.research import belief
    monkeypatch.setattr(belief, "BELIEF_ENSEMBLE_BLEND_ENABLED", False)
    initial = {"GREEN": 0.20, "MARGINAL": 0.40, "RED": 0.40}
    out, note = belief._apply_ensemble_blend(initial.copy(), "VRP")
    assert out == initial
    assert note is None


def test_ensemble_blend_on_uses_family_optimal_w(monkeypatch):
    """When ON + family in FAMILY_OPTIMAL_W + n eligible, blend fires.
    W7-arxiv-v09 LOOCV correction: all w_fam = 1.0 (pure family-empirical
    beats per-family tuning in CV-honest test)."""
    from engine.research import belief
    monkeypatch.setattr(belief, "BELIEF_ENSEMBLE_BLEND_ENABLED", True)
    # Mock _raw_family_empirical_at to return a known dist with n=10
    def _fake_emp(family):
        return ({"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2}, 10)
    monkeypatch.setattr(belief, "_raw_family_empirical_at", _fake_emp)
    # VRP w_fam=1.0 post-v0.9 LOOCV correction → blend collapses to pure
    # family-empirical, completely ignoring pipeline_dist
    pipeline_dist = {"GREEN": 0.1, "MARGINAL": 0.5, "RED": 0.4}
    blended, note = belief._apply_ensemble_blend(pipeline_dist, "VRP")
    expected_green = 1.0 * 0.5 + 0.0 * 0.1   # = 0.5 (pure family-empirical)
    assert abs(blended["GREEN"] - expected_green) < 1e-9
    assert note is not None and "W7-ensemble blend" in note


def test_ensemble_blend_on_skips_if_insufficient_eligible(monkeypatch):
    """If <3 eligible family verdicts, no blend even with flag ON."""
    from engine.research import belief
    monkeypatch.setattr(belief, "BELIEF_ENSEMBLE_BLEND_ENABLED", True)
    def _fake_emp(family):
        return ({"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2}, 2)
    monkeypatch.setattr(belief, "_raw_family_empirical_at", _fake_emp)
    pipeline_dist = {"GREEN": 0.1, "MARGINAL": 0.5, "RED": 0.4}
    out, note = belief._apply_ensemble_blend(pipeline_dist, "VRP")
    assert out == pipeline_dist
    assert note is None


def test_ensemble_blend_on_uses_global_fallback_for_unknown_family(monkeypatch):
    """Family not in FAMILY_OPTIMAL_W → use GLOBAL_W_FALLBACK.
    W7-arxiv-v09: global fallback now 1.0 (pure family-empirical) per LOOCV."""
    from engine.research import belief
    monkeypatch.setattr(belief, "BELIEF_ENSEMBLE_BLEND_ENABLED", True)
    def _fake_emp(family):
        return ({"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2}, 5)
    monkeypatch.setattr(belief, "_raw_family_empirical_at", _fake_emp)
    pipeline_dist = {"GREEN": 0.1, "MARGINAL": 0.5, "RED": 0.4}
    out, note = belief._apply_ensemble_blend(pipeline_dist, "SOME_OBSCURE_FAM")
    # Global fallback w=1.0 → pure family-empirical
    expected_green = 1.0 * 0.5 + 0.0 * 0.1
    assert abs(out["GREEN"] - expected_green) < 1e-9


def test_sliding_window_respects_cutoff(tmp_path: Path):
    fresh = _mk_autopsy(predicted={"GREEN": 1.0, "MARGINAL": 0.0, "RED": 0.0},
                          actual="GREEN", family="A",
                          ts="2026-06-20T00:00:00Z")
    stale = _mk_autopsy(predicted={"GREEN": 0.0, "MARGINAL": 0.0, "RED": 1.0},
                          actual="GREEN", family="A",
                          ts="2026-01-01T00:00:00Z")
    p = _write_autopsies(tmp_path, [fresh, stale])
    now = _dt.datetime(2026, 6, 21)
    tr = build_track_record(autopsies_path=p, now=now)
    short = next(w for w in tr["windows"] if w["window_days"] == 30)
    long  = next(w for w in tr["windows"] if w["window_days"] == 90)
    assert short["n"] == 1  # only fresh
    assert short["mean_brier"] == 0.0
    assert long["n"] == 1   # still only fresh (stale is 5+ months back)
