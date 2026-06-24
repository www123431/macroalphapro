"""belief-2 autopsy tests."""
from __future__ import annotations

import json
import pathlib

import pytest

from engine.research import belief_autopsy as ba


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ── Surprise math ──────────────────────────────────────────────────


def test_brier_component_zero_when_actual_is_certain():
    p = {"GREEN": 1.0, "MARGINAL": 0.0, "RED": 0.0}
    assert ba._brier_component(p, "GREEN") == 0.0


def test_brier_component_one_when_actual_was_zero():
    p = {"GREEN": 1.0, "MARGINAL": 0.0, "RED": 0.0}
    assert ba._brier_component(p, "RED") == 1.0


def test_brier_component_partial():
    p = {"GREEN": 0.7, "MARGINAL": 0.2, "RED": 0.1}
    assert ba._brier_component(p, "MARGINAL") == pytest.approx((1.0 - 0.2) ** 2)


def test_surprise_direction_well_calibrated():
    p = {"GREEN": 0.7, "MARGINAL": 0.2, "RED": 0.1}
    assert ba._surprise_direction(p, "GREEN") == "well_calibrated"


def test_surprise_direction_over_predicted_green():
    p = {"GREEN": 0.7, "MARGINAL": 0.2, "RED": 0.1}
    assert ba._surprise_direction(p, "MARGINAL") == "over_predicted_green"
    assert ba._surprise_direction(p, "RED")      == "over_predicted_green"


def test_surprise_direction_over_predicted_red():
    p = {"GREEN": 0.1, "MARGINAL": 0.2, "RED": 0.7}
    assert ba._surprise_direction(p, "GREEN")    == "over_predicted_red"
    assert ba._surprise_direction(p, "MARGINAL") == "over_predicted_red"


def test_surprise_direction_neutral_when_modal_marginal():
    p = {"GREEN": 0.3, "MARGINAL": 0.4, "RED": 0.3}
    assert ba._surprise_direction(p, "RED") == "neutral"


def test_surprise_magnitude_zero_when_modal_realized():
    p = {"GREEN": 0.7, "MARGINAL": 0.2, "RED": 0.1}
    assert ba._surprise_magnitude(p, "GREEN") == 0.0


def test_surprise_magnitude_positive_on_miss():
    p = {"GREEN": 0.7, "MARGINAL": 0.2, "RED": 0.1}
    # actual RED: distance = 0.7 - 0.1 = 0.6
    assert ba._surprise_magnitude(p, "RED") == pytest.approx(0.6)


# ── Load-bearing realized ──────────────────────────────────────────


def test_load_bearing_spanning_realized_when_capm_alpha_small():
    realized = ba._load_bearing_realized(
        ["spanning_risk"], {"capm_alpha_t": 0.5},
    )
    assert "spanning_risk" in realized


def test_load_bearing_spanning_realized_via_jk_boundary():
    realized = ba._load_bearing_realized(
        ["spanning_risk"], {"capm_alpha_t": 3.0, "jk_vs_a_t": 1.0},
    )
    assert "spanning_risk" in realized


def test_load_bearing_decay_realized_on_severe_oos():
    realized = ba._load_bearing_realized(
        ["post_publication_decay"],
        {"oos_triple": {"severity": "severe"}},
    )
    assert "post_publication_decay" in realized


def test_load_bearing_decay_not_realized_on_mild():
    realized = ba._load_bearing_realized(
        ["post_publication_decay"],
        {"oos_triple": {"severity": "mild"}},
    )
    assert "post_publication_decay" not in realized


# ── Join + autopsy build ───────────────────────────────────────────


def _pred(*, pid, sid, dist, basis="basis", load=(), inputs=None, ts="2026-06-11T00:00:00Z"):
    return {
        "prediction_id": pid,
        "ts": ts,
        "subject_id": sid,
        "family": "TEST_FAM",
        "predicted_verdict_dist": dist,
        "predicted_load_bearing": list(load),
        "prediction_basis": basis,
        "inputs": inputs or {},
    }


def _ev(*, eid, subj, verdict, metrics=None, ts="2026-06-11T01:00:00Z", family="TEST_FAM"):
    return {
        "event_id": eid,
        "event_type": "factor_verdict_filed",
        "ts": ts,
        "subject_id": subj,
        "verdict": verdict,
        "metrics": metrics or {},
        "family": family,
    }


def test_build_autopsy_basic():
    pred = _pred(pid="p1", sid="abc12345-hyp", dist={"GREEN": 0.7, "MARGINAL": 0.2, "RED": 0.1})
    ev = _ev(eid="e1", subj="tier_c_auto_abc12345_factor_combination",
              verdict="MARGINAL",
              metrics={"strategy_family": "COMBINATION_HML_MOM",
                        "claim_family":   "VALUE"})
    a = ba.build_autopsy(pred, ev)
    assert a.prediction_id == "p1"
    assert a.verdict_event_id == "e1"
    assert a.actual_verdict == "MARGINAL"
    assert a.surprise_direction == "over_predicted_green"
    assert a.brier_component == pytest.approx((1.0 - 0.2) ** 2)
    assert a.strategy_family == "COMBINATION_HML_MOM"
    assert a.claim_family == "VALUE"


def test_match_prediction_uses_subject_containment(tmp_path):
    pred = _pred(pid="p2", sid="6f1fbaf3-abc", dist={"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2})
    ev_matching = _ev(eid="e2", subj="tier_c_auto_6f1fbaf3_factor_combination",
                       verdict="GREEN")
    ev_other = _ev(eid="e_other", subj="tier_c_auto_otherxxx_cross_sec",
                    verdict="RED")
    events_by_sid = {
        ev_matching["subject_id"]: [ev_matching],
        ev_other["subject_id"]: [ev_other],
    }
    matched = ba._match_prediction_to_event(pred, events_by_sid)
    assert matched is not None
    assert matched["event_id"] == "e2"


# ── Idempotent backfill + dedup ────────────────────────────────────


def test_backfill_all_dedups(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    ev_path   = tmp_path / "events.jsonl"
    ap_path   = tmp_path / "autopsies.jsonl"

    _write_jsonl(pred_path, [
        _pred(pid="p1", sid="aaa1-x", dist={"GREEN": 0.6, "MARGINAL": 0.3, "RED": 0.1}),
    ])
    _write_jsonl(ev_path, [
        _ev(eid="e1", subj="tier_c_auto_aaa1xxxx_factor_combination",
             verdict="MARGINAL"),
    ])

    out1 = ba.backfill_all(
        predictions_path=pred_path, events_path=ev_path, autopsies_path=ap_path,
    )
    assert len(out1) == 1

    # Re-run: should produce 0 new autopsies (dedup by pair)
    out2 = ba.backfill_all(
        predictions_path=pred_path, events_path=ev_path, autopsies_path=ap_path,
    )
    assert len(out2) == 0

    # Existing autopsies file still has just 1 row
    rows = [json.loads(ln) for ln in ap_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1


def test_backfill_skips_prediction_without_event(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    ev_path   = tmp_path / "events.jsonl"
    ap_path   = tmp_path / "autopsies.jsonl"
    _write_jsonl(pred_path, [
        _pred(pid="p1", sid="nomatch-y", dist={"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2}),
    ])
    _write_jsonl(ev_path, [])
    assert ba.backfill_all(
        predictions_path=pred_path, events_path=ev_path, autopsies_path=ap_path,
    ) == []


# ── run_autopsy_for_verdict_event hook ─────────────────────────────


def test_run_autopsy_for_verdict_event_finds_and_writes(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    ev_path   = tmp_path / "events.jsonl"
    ap_path   = tmp_path / "autopsies.jsonl"
    _write_jsonl(pred_path, [
        _pred(pid="p1", sid="bbb2test-uuid", dist={"GREEN": 0.7, "MARGINAL": 0.2, "RED": 0.1}),
    ])
    _write_jsonl(ev_path, [
        _ev(eid="e1", subj="tier_c_auto_bbb2test_factor_combination",
             verdict="RED"),
    ])
    autopsy = ba.run_autopsy_for_verdict_event(
        "e1",
        predictions_path=pred_path, events_path=ev_path, autopsies_path=ap_path,
    )
    assert autopsy is not None
    assert autopsy.actual_verdict == "RED"
    assert autopsy.surprise_direction == "over_predicted_green"
    assert autopsy.brier_component > 0.8

    # Idempotent: second call returns None
    again = ba.run_autopsy_for_verdict_event(
        "e1",
        predictions_path=pred_path, events_path=ev_path, autopsies_path=ap_path,
    )
    assert again is None


# ── Pattern detection ──────────────────────────────────────────────


def test_detect_patterns_zero_when_empty(tmp_path):
    ap_path = tmp_path / "autopsies.jsonl"
    ap_path.touch()
    out = ba.detect_patterns(autopsies_path=ap_path)
    assert out["n_autopsies"] == 0


def test_detect_patterns_green_overconfidence_flag(tmp_path):
    """10 rows where 8/10 over-predicted GREEN → flag fires."""
    ap_path = tmp_path / "autopsies.jsonl"
    rows = []
    for i in range(10):
        direction = "over_predicted_green" if i < 8 else "well_calibrated"
        rows.append({
            "autopsy_id":         f"a{i}",
            "ts":                 f"2026-06-11T{i:02d}:00:00Z",
            "prediction_id":      f"p{i}",
            "verdict_event_id":   f"e{i}",
            "subject_id":         f"s{i}",
            "strategy_family":    "TEST_FAM",
            "predicted_verdict_dist": {"GREEN": 0.7, "MARGINAL": 0.2, "RED": 0.1},
            "actual_verdict":     "MARGINAL" if i < 8 else "GREEN",
            "brier_component":    0.5,
            "surprise_direction": direction,
            "surprise_magnitude": 0.5,
        })
    _write_jsonl(ap_path, rows)
    out = ba.detect_patterns(autopsies_path=ap_path)
    assert out["n_autopsies"] == 10
    flags = [p["pattern"] for p in out["patterns"]]
    assert "GREEN_OVERCONFIDENCE" in flags


def test_detect_patterns_no_flag_when_calibrated(tmp_path):
    ap_path = tmp_path / "autopsies.jsonl"
    rows = []
    for i in range(10):
        rows.append({
            "autopsy_id":         f"a{i}",
            "ts":                 f"2026-06-11T{i:02d}:00:00Z",
            "prediction_id":      f"p{i}",
            "verdict_event_id":   f"e{i}",
            "subject_id":         f"s{i}",
            "strategy_family":    "TEST_FAM",
            "predicted_verdict_dist": {"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2},
            "actual_verdict":     "GREEN",
            "brier_component":    0.25,
            "surprise_direction": "well_calibrated",
            "surprise_magnitude": 0.0,
        })
    _write_jsonl(ap_path, rows)
    out = ba.detect_patterns(autopsies_path=ap_path)
    assert out["patterns"] == []
    assert out["mean_brier"] == pytest.approx(0.25)
