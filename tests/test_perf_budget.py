"""tests/test_perf_budget.py — Phase 4 perf/cost governance (cache surface + SLO)."""
import json
from pathlib import Path

from engine.agents.governance.perf_budget import (
    cache_surface_report, check_slo, _p95, _target_for, _est_tokens,
)


# ── cache surface ────────────────────────────────────────────────────────────
def test_cache_surface_covers_all_personas():
    rep = cache_surface_report()
    assert "decay_sentinel" in rep and "risk_manager" in rep
    for r in rep.values():
        assert r["est_prompt_tokens"] > 0
        assert r["cache_eligible"] in (True, False, None)
    # deepseek persona is provider-managed (not an anthropic cache verdict)
    assert rep["devils_advocate"]["provider"] == "deepseek"
    assert rep["devils_advocate"]["cache_eligible"] is None


def test_token_estimate_is_chars_over_4():
    assert _est_tokens("x" * 400) == 100


# ── p95 + role targets ───────────────────────────────────────────────────────
def test_p95_picks_high_quantile():
    assert _p95(list(range(1, 101))) >= 95
    assert _p95([]) != _p95([])      # nan


def test_role_targets():
    assert _target_for("ops_watchdog").role == "batch_cron"
    assert _target_for("decay_sentinel").role == "interactive"


# ── SLO compliance over synthetic metrics ────────────────────────────────────
def _write(tmp_path, rows):
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_slo_flags_slow_interactive_and_passes_cron(tmp_path):
    rows = ([{"agent_id": "fast", "latency_ms": 1000, "success": True}] * 3
            + [{"agent_id": "slow", "latency_ms": 120_000, "success": True}] * 3
            + [{"agent_id": "ops_watchdog", "latency_ms": 300_000, "success": True}] * 3)
    r = check_slo(_write(tmp_path, rows))
    a = r["agents"]
    assert a["fast"]["compliant"] is True
    assert a["slow"]["latency_ok"] is False and a["slow"]["compliant"] is False   # > 60s interactive
    assert a["ops_watchdog"]["role"] == "batch_cron" and a["ops_watchdog"]["compliant"] is True  # < 600s batch
    assert all(v["low_sample"] for v in a.values())          # n=3 < 20
    assert r["all_compliant"] is False


def test_slo_no_metrics_file(tmp_path):
    assert check_slo(tmp_path / "missing.jsonl")["available"] is False


def test_slo_flags_low_success_rate(tmp_path):
    rows = [{"agent_id": "flaky", "latency_ms": 500, "success": i != 0} for i in range(5)]  # 4/5=0.8
    a = check_slo(_write(tmp_path, rows))["agents"]["flaky"]
    assert a["success_ok"] is False and a["compliant"] is False
