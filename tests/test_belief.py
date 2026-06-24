"""Belief Layer Phase 1 (belief-1, 2026-06-11) tests.

Covers:
  * predict_verdict produces a valid distribution
  * family priors override default for known families
  * n_trials penalty shrinks GREEN
  * post-publication age penalty shifts toward RED
  * load-bearing list surfaces correct assumptions
  * log_prediction writes valid jsonl
  * **structural invariant**: lens / strict_gate / template modules
    MUST NOT import engine.research.belief (air-gap doctrine)
"""
from __future__ import annotations

import ast
import json
import pathlib
import tempfile

import pytest

from engine.research import belief


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── Basic prediction math ───────────────────────────────────────────


def test_predicted_dist_sums_to_one_default():
    pred = belief.predict_verdict(subject_id="t1", family=None)
    s = sum(pred.predicted_verdict_dist.values())
    assert abs(s - 1.0) < 1e-9
    assert set(pred.predicted_verdict_dist) == {"GREEN", "MARGINAL", "RED"}


def test_predicted_dist_uses_default_for_unknown_family():
    pred = belief.predict_verdict(subject_id="t1", family="UNKNOWN_FAM_XYZ")
    # Default has GREEN=0.20
    assert pred.predicted_verdict_dist["GREEN"] == pytest.approx(0.20, abs=0.01)
    assert "default prior" in pred.prediction_basis.lower()


def test_predicted_dist_uses_family_override_when_no_observations(monkeypatch):
    # Force the observed-posterior path to return N=0 so we exercise the
    # override branch deterministically (the real events.jsonl may carry
    # accumulated PROFITABILITY verdicts which would otherwise dominate).
    monkeypatch.setattr(
        belief, "_family_observed_dist",
        lambda fam: (dict(belief.DEFAULT_PRIOR), 0),
    )
    pred = belief.predict_verdict(subject_id="t1", family="PROFITABILITY")
    # PROFITABILITY override has GREEN=0.12 — strictly lower than default 0.20
    assert pred.predicted_verdict_dist["GREEN"] < 0.20
    assert "family prior override" in pred.prediction_basis.lower()
    assert pred.family == "PROFITABILITY"


def test_family_case_insensitive(monkeypatch):
    monkeypatch.setattr(
        belief, "_family_observed_dist",
        lambda fam: (dict(belief.DEFAULT_PRIOR), 0),
    )
    a = belief.predict_verdict(subject_id="t1", family="profitability")
    b = belief.predict_verdict(subject_id="t1", family="PROFITABILITY")
    assert a.family == "PROFITABILITY"
    assert b.family == "PROFITABILITY"
    assert a.predicted_verdict_dist == b.predicted_verdict_dist


# ── Adjustments ─────────────────────────────────────────────────────


def test_n_trials_penalty_shrinks_green():
    dist_no_penalty = {"GREEN": 0.30, "MARGINAL": 0.40, "RED": 0.30}
    out = belief._apply_n_trials_penalty(dist_no_penalty, n_trials=20)
    assert out["GREEN"] < dist_no_penalty["GREEN"]
    assert out["MARGINAL"] > dist_no_penalty["MARGINAL"]
    # RED unchanged (penalty doesn't make real factor fake, just harder to claim)
    assert out["RED"] == pytest.approx(dist_no_penalty["RED"], abs=0.01)
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_n_trials_below_threshold_no_penalty():
    dist = {"GREEN": 0.30, "MARGINAL": 0.40, "RED": 0.30}
    out = belief._apply_n_trials_penalty(dist, n_trials=5)
    assert out == dist


def test_publication_age_penalty_shifts_to_red():
    dist = {"GREEN": 0.30, "MARGINAL": 0.40, "RED": 0.30}
    new_dist, applied = belief._apply_publication_age_penalty(
        dist, paper_year=2000, current_year=2026,
    )
    assert applied is True
    assert new_dist["RED"] > dist["RED"]
    assert new_dist["GREEN"] < dist["GREEN"]
    assert abs(sum(new_dist.values()) - 1.0) < 1e-9


def test_recent_paper_no_age_penalty():
    dist = {"GREEN": 0.30, "MARGINAL": 0.40, "RED": 0.30}
    new_dist, applied = belief._apply_publication_age_penalty(
        dist, paper_year=2020, current_year=2026,
    )
    assert applied is False
    assert new_dist == dist


def test_old_paper_marks_decay_load_bearing():
    pred = belief.predict_verdict(
        subject_id="t1", family="MOMENTUM",
        paper_year=1990, current_year=2026,
    )
    assert "post_publication_decay" in pred.predicted_load_bearing


def test_mature_family_marks_spanning_risk():
    pred = belief.predict_verdict(subject_id="t1", family="PROFITABILITY")
    assert "spanning_risk" in pred.predicted_load_bearing


# ── Logging ─────────────────────────────────────────────────────────


def test_log_prediction_writes_valid_jsonl(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = pathlib.Path(td) / "predictions.jsonl"
        monkeypatch.setattr(belief, "PREDICTIONS_PATH", tmp_path)
        pred = belief.predict_verdict(
            subject_id="t_log", family="QUALITY", paper_year=2015,
        )
        pid = belief.log_prediction(pred)
        assert pid == pred.prediction_id
        assert tmp_path.is_file()
        rows = [
            json.loads(ln) for ln in tmp_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["prediction_id"] == pid
        assert rows[0]["subject_id"] == "t_log"
        assert rows[0]["family"] == "QUALITY"
        assert "GREEN" in rows[0]["predicted_verdict_dist"]


def test_predict_and_log_returns_prediction(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = pathlib.Path(td) / "predictions.jsonl"
        monkeypatch.setattr(belief, "PREDICTIONS_PATH", tmp_path)
        pred = belief.predict_and_log(
            subject_id="t_pl", family="VALUE", signal_kind="cross_sec",
        )
        assert isinstance(pred, belief.Prediction)
        assert pred.subject_id == "t_pl"
        assert tmp_path.is_file()


# ── STRUCTURAL: air-gap invariant ───────────────────────────────────


# Files that legitimately MAY consume belief predictions.
# The air-gap protects VERDICT-COMPUTING code (lens / strict_gate /
# template) from seeing predictions and self-fulfilling them. Modules
# that produce predictions or display them (planners, dashboards) are
# safe by definition — they don't run lens math on factor data.
_BELIEF_CONSUMER_WHITELIST = frozenset({
    # The dispatcher entry hook — the only producer
    "engine/agents/strengthener/factor_dispatcher.py",
    # belief module itself + its tests
    "engine/research/belief.py",
    "tests/test_belief.py",
    # burn-1a planner — shows predictions in dry-run plans for principal
    # review. Does NOT compute verdicts; safe to consume.
    "engine/research/burndown_planner.py",
    # belief-2 autopsy — reads predictions + verdicts AFTER both produced;
    # writes parallel autopsies.jsonl. Doesn't compute verdicts.
    "engine/research/belief_autopsy.py",
    # belief-4 closed-loop prior — reads autopsies, exports calibrated
    # prior consumed BY belief.py itself. No verdict computation.
    "engine/research/belief_prior_calibration.py",
})


def _modules_under(*dirs: str) -> list[pathlib.Path]:
    """Return all .py files under the given repo-relative directories."""
    out: list[pathlib.Path] = []
    for d in dirs:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        out.extend(root.rglob("*.py"))
    return out


def _imports_belief(path: pathlib.Path) -> bool:
    """Return True if the file's AST contains any import of engine.research.belief."""
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "engine.research.belief" or mod.startswith("engine.research.belief."):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "engine.research.belief" or alias.name.startswith("engine.research.belief."):
                    return True
    return False


def test_air_gap_lens_strict_gate_template_must_not_import_belief():
    """STRUCTURAL INVARIANT (Belief Layer doctrine 2026-06-11):

    No module under engine/research/ (lens / strict_gate / template
    plumbing) or engine/agents/strengthener/templates/ may import
    engine.research.belief. The predict-then-observe contract requires
    that verdict-computing code CANNOT see its own prediction —
    self-fulfilling prophecies would invalidate the entire calibration
    project.

    Whitelist: dispatcher entry hook + belief module + this test file.
    """
    candidate_dirs = (
        "engine/research",
        "engine/agents/strengthener/templates",
        "engine/agents/strengthener",
    )
    offenders: list[str] = []
    for fp in _modules_under(*candidate_dirs):
        try:
            rel = fp.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        if rel in _BELIEF_CONSUMER_WHITELIST:
            continue
        if _imports_belief(fp):
            offenders.append(rel)
    assert not offenders, (
        f"Air-gap violation: these modules import engine.research.belief "
        f"but are not on the whitelist: {offenders}. "
        f"Move belief consumption OUTSIDE the lens/strict_gate/template "
        f"tree (e.g. into engine.research.belief_autopsy in Phase 2)."
    )
