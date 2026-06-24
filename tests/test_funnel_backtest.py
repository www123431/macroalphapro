"""Tests for engine.research.discovery.funnel_backtest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from engine.research.discovery import funnel_backtest as fb


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    gate = tmp_path / "gate_runs.jsonl"
    monkeypatch.setattr(fb, "LIBRARY_DIR", lib)
    monkeypatch.setattr(fb, "GATE_RUNS", gate)
    monkeypatch.setattr(fb, "REPO_ROOT", tmp_path)
    return {"lib": lib, "gate": gate, "tmp": tmp_path}


def _write_library_yaml(lib: Path, mid: str, *, status="DEPLOYED",
                            family="carry", title=None,
                            economics="Cross-asset roll yield carry."):
    (lib / f"{mid}.yaml").write_text(
        yaml.safe_dump({
            "id": mid,
            "title": title or mid,
            "family": family,
            "status_in_our_book": status,
            "mechanism_economics": economics,
        }),
        encoding="utf-8",
    )


def _write_gate_run(path: Path, *, mechanism, verdict, family=None):
    rec = {"mechanism": mechanism, "verdict": verdict,
            "standalone_sharpe": 0.1, "alpha_t_ff5umd": 0.5,
            "deflated_sr": 0.1, "ts": "2024-01-01"}
    if family:
        rec["family"] = family
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ── Loaders ────────────────────────────────────────────────────────────

def test_load_library_candidates_includes_all_yamls(isolated):
    _write_library_yaml(isolated["lib"], "carry_x")
    _write_library_yaml(isolated["lib"], "momentum_y", status="RED",
                            family="momentum")
    cands = fb._load_library_candidates()
    assert len(cands) == 2
    statuses = {c.ground_truth for c in cands}
    assert statuses == {"DEPLOYED", "RED"}


def test_load_library_skips_underscore_prefix(isolated):
    _write_library_yaml(isolated["lib"], "carry_x")
    (isolated["lib"] / "_draft.yaml").write_text(
        yaml.safe_dump({"id": "draft", "title": "x", "family": "x",
                          "status_in_our_book": "PENDING"}),
        encoding="utf-8",
    )
    cands = fb._load_library_candidates()
    names = [c.name for c in cands]
    assert "draft" not in names


def test_load_gate_runs_classifies_verdicts(isolated):
    _write_gate_run(isolated["gate"], mechanism="green_one", verdict="GREEN")
    _write_gate_run(isolated["gate"], mechanism="green_qual",
                       verdict="GREEN — 4/4 strict bars")
    _write_gate_run(isolated["gate"], mechanism="yellow_one", verdict="YELLOW")
    _write_gate_run(isolated["gate"], mechanism="red_one", verdict="RED")
    _write_gate_run(isolated["gate"], mechanism="other", verdict="WEIRD")
    cands = fb._load_gate_run_candidates()
    gts = {c.name: c.ground_truth for c in cands}
    assert gts["green_one"] == "GREEN"
    assert gts["green_qual"] == "GREEN"
    assert gts["yellow_one"] == "YELLOW"
    assert gts["red_one"] == "RED"
    assert gts["other"] == "UNKNOWN"


def test_load_gate_runs_limit_param(isolated):
    for i in range(10):
        _write_gate_run(isolated["gate"], mechanism=f"m{i}", verdict="RED")
    cands = fb._load_gate_run_candidates(limit=3)
    # Limit takes last N
    assert len(cands) == 3
    assert cands[-1].name == "m9"


def test_load_gate_runs_handles_malformed_lines(isolated):
    with isolated["gate"].open("w", encoding="utf-8") as f:
        f.write(json.dumps({"mechanism": "ok", "verdict": "GREEN"}) + "\n")
        f.write("not valid json\n")
        f.write(json.dumps({"mechanism": "ok2", "verdict": "RED"}) + "\n")
    cands = fb._load_gate_run_candidates()
    assert len(cands) == 2     # malformed line silently skipped


# ── Evaluation ─────────────────────────────────────────────────────────

def test_evaluate_one_returns_decision_with_all_fields(isolated):
    _write_library_yaml(isolated["lib"], "carry_x",
                            economics="Long-short carry portfolio with "
                                       "Sharpe 1.5 from 1990-2020 CRSP data.")
    cand = fb._load_library_candidates()[0]
    decision = fb._evaluate_one(cand)
    for field in ("credibility_score", "confidence_score",
                    "family_routing", "routing_adjusted",
                    "final_disposition", "agrees_with_truth"):
        assert hasattr(decision, field)


def test_evaluate_one_deployed_carry_agrees_when_routed_review_or_borderline(
    isolated,
):
    """A DEPLOYED carry mechanism with mechanism markers should route
    to review or borderline (not skip)."""
    _write_library_yaml(
        isolated["lib"], "carry_x", status="DEPLOYED", family="carry",
        economics="Long-short carry portfolio Sharpe 1.5 CRSP 1990-2020.",
    )
    cand = fb._load_library_candidates()[0]
    decision = fb._evaluate_one(cand)
    # We expect at least borderline (carry threshold 0.4)
    assert decision.final_disposition in ("review", "borderline")


def test_evaluate_one_red_routes_to_skip_or_borderline(isolated):
    """RED mechanisms should not be promoted to review."""
    _write_library_yaml(isolated["lib"], "junk_red", status="RED",
                            family="quality",
                            economics="Some weak abstract.")
    cand = fb._load_library_candidates()[0]
    decision = fb._evaluate_one(cand)
    # Should NOT be review tier
    assert decision.final_disposition != "review"


# ── Report ─────────────────────────────────────────────────────────────

def test_run_backtest_returns_dict_shape(isolated):
    _write_library_yaml(isolated["lib"], "a", status="DEPLOYED")
    _write_gate_run(isolated["gate"], mechanism="b", verdict="RED")
    report = fb.run_backtest()
    for key in ("total", "by_source", "by_ground_truth",
                  "by_disposition", "agreement_rate",
                  "confusion", "false_positives", "false_negatives"):
        assert key in report
    assert report["total"] == 2


def test_run_backtest_exclude_library(isolated):
    _write_library_yaml(isolated["lib"], "a", status="DEPLOYED")
    _write_gate_run(isolated["gate"], mechanism="b", verdict="RED")
    report = fb.run_backtest(include_library=False)
    assert "library" not in report["by_source"]


def test_run_backtest_no_data_returns_empty(isolated):
    report = fb.run_backtest()
    assert report["total"] == 0
    assert report["agreement_rate"] == 0.0


def test_run_backtest_agreement_rate_computes_correctly(isolated):
    """Two known-DEPLOYED carry-rich entries → both should agree
    (assuming current calculator picks them up)."""
    _write_library_yaml(
        isolated["lib"], "carry_a", status="DEPLOYED", family="carry",
        economics="Sharpe 1.5 long-short carry CRSP 1990-2020 monthly rebal.",
    )
    _write_library_yaml(
        isolated["lib"], "carry_b", status="DEPLOYED", family="carry",
        economics="Sharpe 2 carry portfolio CRSP 1990-2020 monthly rebal.",
    )
    report = fb.run_backtest()
    # Both DEPLOYED carry entries should agree (review or borderline)
    assert report["agreement_rate"] >= 0.5
