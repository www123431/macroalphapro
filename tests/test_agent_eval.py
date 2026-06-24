"""tests/test_agent_eval.py — the agent-eval scorer must CATCH bad behavior + the static
contract holds for every persona. No live LLM (Tier B is gated)."""
import json
from pathlib import Path
from types import SimpleNamespace

from engine.agents.eval.contract import Expect, score_expectation, score_static_contract
from engine.agents.eval.cases import CASES
from engine.agents.eval.runner import run_static_eval, _personas, _wilson_ci
from engine.agents.eval.manifest import build_manifest, check_manifest, freeze_manifest


def _turn(text, tools=()):
    return SimpleNamespace(final_text=text, tool_calls_log=tuple(tools))


# ── Tier A static contract ─────────────────────────────────────────────────
def test_static_contract_all_personas_pass():
    rep = run_static_eval()
    failed = {a: [c["check"] for c in checks if not c["passed"]]
              for a, checks in rep["report"].items()}
    failed = {a: f for a, f in failed.items() if f}
    assert rep["all_pass"], f"static contract failures: {failed}"


# ── scorer CATCH tests (proves it measures, not rubber-stamps) ──────────────
def test_no_banned_catches_hedging():
    assert score_expectation(_turn("This might be decaying."), "", Expect("no_banned")).passed is False
    assert score_expectation(_turn("D_PEAD is not decaying."), "", Expect("no_banned")).passed is True


def test_grounded_catches_fabricated_number():
    tools = [{"name": "read_decay_sentinel_report", "result_preview": '{"rolling_sharpe": 1.26}'}]
    # 9.99 is nowhere in the tool output -> fabrication flagged
    assert score_expectation(_turn("Sharpe is 9.99", tools), "", Expect("grounded")).passed is False
    # 1.26 IS in the tool output -> grounded
    assert score_expectation(_turn("rolling Sharpe 1.26", tools), "", Expect("grounded")).passed is True


def test_tool_expectation():
    tools = [{"name": "read_decay_sentinel_report", "result_preview": "{}"}]
    assert score_expectation(_turn("ok", tools), "", Expect("tool", names=("read_decay_sentinel_report",))).passed
    assert not score_expectation(_turn("ok", ()), "", Expect("tool", names=("read_decay_sentinel_report",))).passed


def test_refuse_route_detects_refusal_and_peer():
    r = score_expectation(_turn("Out of my scope — ask the Anomaly Sentinel."), "",
                          Expect("refuse_route", targets=("anomaly_sentinel",)))
    assert r.passed is True
    # a compliant (non-refusing) answer must NOT score as a refusal
    assert not score_expectation(_turn("GLD z-score is 1.4."), "",
                                 Expect("refuse_route", targets=("anomaly_sentinel",))).passed


# ── grounding full-haystack override (the rigor fix) ─────────────────────────
def test_grounded_uses_full_tool_haystack_override():
    # number is NOT in the (empty) call preview but IS in the full haystack -> grounded
    r = score_expectation(_turn("rolling Sharpe 1.26", ()), "",
                          Expect("grounded"), tool_haystack='{"rolling_sharpe": 1.26}')
    assert r.passed is True
    # not in haystack -> flagged
    assert not score_expectation(_turn("Sharpe 9.99", ()), "",
                                 Expect("grounded"), tool_haystack='{"rolling_sharpe": 1.26}').passed


# ── Wilson CI sanity ──────────────────────────────────────────────────────────
def test_wilson_ci_bounds():
    lo, hi = _wilson_ci(3, 3)
    assert 0.0 <= lo <= hi <= 1.0 and lo > 0.3      # 3/3 -> interval skewed high but < 1 width
    assert _wilson_ci(0, 0) == (None, None)


# ── manifest gate (model/prompt-change governance) ───────────────────────────
def test_frozen_manifest_matches_current():
    # THE GATE: editing a persona prompt/model/tools without re-freezing fails here.
    r = check_manifest()
    assert r["clean"], f"manifest drift (re-run eval + `manifest --freeze`): {r}"


def test_manifest_detects_drift(tmp_path):
    cur = build_manifest()
    tampered = json.loads(json.dumps(cur))
    victim = sorted(tampered)[0]
    tampered[victim]["prompt_sha"] = "deadbeefdeadbeef"     # simulate a silent prompt edit
    fp = tmp_path / "frozen.json"
    fp.write_text(json.dumps(tampered), encoding="utf-8")
    r = check_manifest(fp)
    assert victim in r["changed"] and not r["clean"]


# ── case set integrity ───────────────────────────────────────────────────────
def test_every_case_resolves_to_a_persona():
    personas = _personas()
    missing = sorted({c.agent_id for c in CASES} - set(personas))
    assert not missing, f"cases reference unknown personas: {missing}"
    assert len(CASES) >= 8
