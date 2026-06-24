"""belief-2 Inbox digest tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.inbox import composer
from engine.research import belief_autopsy


def _write_autopsies(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _autopsy_row(
    *,
    aid: str,
    family: str,
    actual: str,
    direction: str,
    brier: float,
    ts: str = "2026-06-12T00:00:00Z",
) -> dict:
    return {
        "autopsy_id":         aid,
        "ts":                 ts,
        "prediction_id":      "pred-" + aid,
        "verdict_event_id":   "ev-" + aid,
        "subject_id":         aid + "-sid",
        "strategy_family":    family,
        "claim_family":       family,
        "predicted_verdict_dist": {"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2},
        "actual_verdict":     actual,
        "brier_component":    brier,
        "surprise_direction": direction,
        "surprise_magnitude": 0.3,
        "load_bearing_realized": [],
        "prediction_basis_echo": "test",
    }


def test_digest_empty_when_no_autopsies(tmp_path, monkeypatch):
    ap = tmp_path / "autopsies.jsonl"
    monkeypatch.setattr(belief_autopsy, "AUTOPSIES_PATH", ap)
    out = composer.source_belief_autopsy_digest(min_for_summary=3)
    assert out == []


def test_digest_empty_when_below_min(tmp_path, monkeypatch):
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row(aid="a1", family="X", actual="GREEN",
                       direction="well_calibrated", brier=0.10),
    ])
    monkeypatch.setattr(belief_autopsy, "AUTOPSIES_PATH", ap)
    out = composer.source_belief_autopsy_digest(min_for_summary=3)
    assert out == []


def test_digest_info_tone_when_calibrated(tmp_path, monkeypatch):
    ap = tmp_path / "autopsies.jsonl"
    _write_autopsies(ap, [
        _autopsy_row(aid=f"a{i}", family="X",
                       actual="GREEN", direction="well_calibrated",
                       brier=0.10) for i in range(5)
    ])
    monkeypatch.setattr(belief_autopsy, "AUTOPSIES_PATH", ap)
    out = composer.source_belief_autopsy_digest(min_for_summary=3)
    assert len(out) == 1
    item = out[0]
    assert item["tone"] == "info"
    assert item["source"] == "belief_autopsy"
    assert item["lane"] == composer._LANE_ENGINE
    assert "5" in item["title"]
    assert "well_calibrated=5" in item["summary"]


def test_digest_warn_when_mean_brier_high(tmp_path, monkeypatch):
    """mean Brier > 0.50 (but no pattern flag) → warn tone."""
    ap = tmp_path / "autopsies.jsonl"
    rows = []
    for i in range(5):
        rows.append(_autopsy_row(
            aid=f"a{i}", family="X",
            actual="MARGINAL",
            direction="neutral",
            brier=0.55,
        ))
    _write_autopsies(ap, rows)
    monkeypatch.setattr(belief_autopsy, "AUTOPSIES_PATH", ap)
    out = composer.source_belief_autopsy_digest(min_for_summary=3)
    assert len(out) == 1
    assert out[0]["tone"] == "warn"


def test_digest_alert_when_green_overconfidence_flag(tmp_path, monkeypatch):
    """10+ rows with 70%+ over_predicted_green → alert tone."""
    ap = tmp_path / "autopsies.jsonl"
    rows = []
    for i in range(10):
        rows.append(_autopsy_row(
            aid=f"a{i}",
            family="X",
            actual="MARGINAL" if i < 8 else "GREEN",
            direction="over_predicted_green" if i < 8 else "well_calibrated",
            brier=0.50,
        ))
    _write_autopsies(ap, rows)
    monkeypatch.setattr(belief_autopsy, "AUTOPSIES_PATH", ap)
    out = composer.source_belief_autopsy_digest(min_for_summary=3)
    assert len(out) == 1
    item = out[0]
    assert item["tone"] == "alert"
    assert "GREEN_OVERCONFIDENCE" in item["title"]
    assert "advice" in item["summary"]


def test_digest_surfaces_family_hotspot(tmp_path, monkeypatch):
    ap = tmp_path / "autopsies.jsonl"
    rows = []
    # family X with high Brier (poor calibration)
    for i in range(3):
        rows.append(_autopsy_row(
            aid=f"x{i}", family="HOT_FAM",
            actual="RED", direction="over_predicted_green", brier=0.80,
        ))
    # family Y with low Brier (good calibration)
    for i in range(3):
        rows.append(_autopsy_row(
            aid=f"y{i}", family="COLD_FAM",
            actual="GREEN", direction="well_calibrated", brier=0.05,
        ))
    _write_autopsies(ap, rows)
    monkeypatch.setattr(belief_autopsy, "AUTOPSIES_PATH", ap)
    out = composer.source_belief_autopsy_digest(min_for_summary=3)
    assert len(out) == 1
    assert "HOT_FAM" in out[0]["summary"]
    md = out[0]["metadata"]
    assert md["family_brier"]["HOT_FAM"] > md["family_brier"]["COLD_FAM"]
