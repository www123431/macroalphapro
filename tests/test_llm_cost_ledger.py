"""
tests/test_llm_cost_ledger.py — Unit tests for engine.llm_cost_ledger
(Sprint 2B, 2026-05-10).

Test isolation discipline: every test that writes the ledger uses tmp_path
+ monkeypatch on the path-resolver hooks (_ledger_path / _lock_path) so no
test pollutes the real `data/llm_cost_ledger.jsonl`.
(Per feedback_test_isolation_no_disk_pollution rule, 2026-05-09.)
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from engine import llm_cost_ledger as ledger
from engine.llm_cost_ledger import (
    ALLOWED_AGENT_IDS,
    ALLOWED_PROVIDERS,
    CostEntry,
    get_call_count,
    get_calls,
    get_lifetime_total,
    get_total_by_agent,
    get_total_by_provider,
    get_total_today,
    get_trailing_365d_total,
    integrity_check,
    record_call,
)


# ── Fixture: redirect ledger to tmp_path ────────────────────────────────────
@pytest.fixture
def tmp_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ledger path to tmp dir for full test isolation."""
    p = tmp_path / "llm_cost_ledger.jsonl"
    lock = tmp_path / "llm_cost_ledger.jsonl.lock"
    monkeypatch.setattr(ledger, "_ledger_path", lambda: p)
    monkeypatch.setattr(ledger, "_lock_path",   lambda: lock)
    return p


# ── Recording: happy path ───────────────────────────────────────────────────
def test_record_call_appends_one_jsonl_line(tmp_ledger: Path) -> None:
    record_call(
        agent_id="r_audit",
        provider="gemini",
        model="gemini-2.5-flash",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.001,
        latency_ms=1500,
    )
    assert tmp_ledger.exists()
    lines = tmp_ledger.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["agent_id"]          == "r_audit"
    assert parsed["provider"]          == "gemini"
    assert parsed["model"]             == "gemini-2.5-flash"
    assert parsed["prompt_tokens"]     == 100
    assert parsed["completion_tokens"] == 50
    assert parsed["cost_usd"]          == 0.001
    assert parsed["latency_ms"]        == 1500
    assert "ts" in parsed and parsed["ts"].endswith("Z")


def test_record_call_appends_multiple_calls(tmp_ledger: Path) -> None:
    for i in range(3):
        record_call(
            agent_id="s6_anomaly",
            provider="gemini",
            model="gemini-2.5-flash",
            prompt_tokens=10 * i,
            completion_tokens=20 * i,
            cost_usd=0.0001 * (i + 1),
            latency_ms=100 + i,
        )
    lines = tmp_ledger.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3


def test_record_call_returns_entry(tmp_ledger: Path) -> None:
    out = record_call(
        agent_id="deepseek",
        provider="deepseek",
        model="deepseek-v4-flash",
        prompt_tokens=10,
        completion_tokens=200,
        cost_usd=0.000056,
        latency_ms=1500,
    )
    assert isinstance(out, CostEntry)
    assert out.agent_id == "deepseek"
    assert out.cost_usd == 0.000056


def test_record_call_extra_metadata_preserved(tmp_ledger: Path) -> None:
    record_call(
        agent_id="tool1_decision_lineage",
        provider="gemini",
        model="gemini-2.5-flash",
        prompt_tokens=500,
        completion_tokens=300,
        cost_usd=0.0008,
        latency_ms=2500,
        scope="react_step",
        extra={"step_idx": 1, "tool_called": "query_p2_rag"},
    )
    [entry] = get_calls()
    assert entry.scope == "react_step"
    assert entry.extra == {"step_idx": 1, "tool_called": "query_p2_rag"}


# ── Recording: validation ───────────────────────────────────────────────────
def test_record_call_rejects_unknown_agent_id(tmp_ledger: Path) -> None:
    with pytest.raises(ValueError, match="agent_id .* not in ALLOWED_AGENT_IDS"):
        record_call(
            agent_id="not_a_real_agent",
            provider="gemini",
            model="gemini-2.5-flash",
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=0.0001,
            latency_ms=100,
        )
    assert not tmp_ledger.exists()


def test_record_call_rejects_unknown_provider(tmp_ledger: Path) -> None:
    with pytest.raises(ValueError, match="provider .* not in ALLOWED_PROVIDERS"):
        record_call(
            agent_id="r_audit",
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=0.0001,
            latency_ms=100,
        )


def test_record_call_rejects_negative_cost(tmp_ledger: Path) -> None:
    with pytest.raises(ValueError, match="cost_usd must be non-negative"):
        record_call(
            agent_id="r_audit",
            provider="gemini",
            model="gemini-2.5-flash",
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=-0.0001,
            latency_ms=100,
        )


# ── Querying: filter combinations ────────────────────────────────────────────
def _seed_three_calls(ts_offsets_days: list[int] | None = None) -> None:
    """Helper: write 3 calls across 2 agents and 2 providers."""
    today = datetime.datetime.utcnow().date()
    if ts_offsets_days is None:
        ts_offsets_days = [0, 0, 0]
    for offset, (agent, prov, cost) in zip(
        ts_offsets_days,
        [
            ("r_audit",  "gemini",   0.001),
            ("s6_anomaly", "gemini", 0.002),
            ("deepseek", "deepseek", 0.0001),
        ],
    ):
        ts = (today - datetime.timedelta(days=offset)).isoformat() + "T12:00:00Z"
        record_call(
            agent_id=agent,
            provider=prov,
            model="m",
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=cost,
            latency_ms=100,
            ts=ts,
        )


def test_get_calls_no_filter_returns_all(tmp_ledger: Path) -> None:
    _seed_three_calls()
    out = get_calls()
    assert len(out) == 3
    agent_ids = {e.agent_id for e in out}
    assert agent_ids == {"r_audit", "s6_anomaly", "deepseek"}


def test_get_calls_filter_by_agent(tmp_ledger: Path) -> None:
    _seed_three_calls()
    out = get_calls(agent_id="r_audit")
    assert len(out) == 1
    assert out[0].agent_id == "r_audit"


def test_get_calls_filter_by_provider(tmp_ledger: Path) -> None:
    _seed_three_calls()
    out = get_calls(provider="deepseek")
    assert len(out) == 1
    assert out[0].provider == "deepseek"


def test_get_calls_filter_by_date_range(tmp_ledger: Path) -> None:
    _seed_three_calls(ts_offsets_days=[0, 5, 10])
    today = datetime.datetime.utcnow().date()
    out = get_calls(since=today - datetime.timedelta(days=7), until=today)
    # Should match ts_offsets 0 and 5 but NOT 10
    assert len(out) == 2


def test_get_calls_limit_keeps_tail(tmp_ledger: Path) -> None:
    for i in range(5):
        record_call(
            agent_id="r_audit",
            provider="gemini",
            model="m",
            prompt_tokens=i,
            completion_tokens=i,
            cost_usd=0.0001 * (i + 1),
            latency_ms=100,
        )
    out = get_calls(limit=2)
    assert len(out) == 2
    # Last 2 = highest prompt_tokens (3 and 4)
    assert {e.prompt_tokens for e in out} == {3, 4}


# ── Aggregation API ─────────────────────────────────────────────────────────
def test_get_total_by_agent_aggregates_correctly(tmp_ledger: Path) -> None:
    _seed_three_calls()
    record_call(
        agent_id="r_audit", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.005, latency_ms=100,
    )
    agg = get_total_by_agent()
    assert "r_audit" in agg
    assert agg["r_audit"]["total_usd"] == round(0.001 + 0.005, 8)
    assert agg["r_audit"]["calls"] == 2
    assert agg["s6_anomaly"]["total_usd"] == 0.002
    assert agg["deepseek"]["total_usd"] == 0.0001
    assert agg["r_audit"]["providers"] == {"gemini": round(0.001 + 0.005, 8)}


def test_get_total_by_provider(tmp_ledger: Path) -> None:
    _seed_three_calls()
    agg = get_total_by_provider()
    assert agg["gemini"] == round(0.001 + 0.002, 8)
    assert agg["deepseek"] == 0.0001


def test_get_trailing_365d_total(tmp_ledger: Path) -> None:
    today = datetime.datetime.utcnow().date()
    # Inside 365d window
    record_call(
        agent_id="etf_holdings", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.10, latency_ms=100,
        ts=(today - datetime.timedelta(days=200)).isoformat() + "T00:00:00Z",
    )
    # Outside 365d window
    record_call(
        agent_id="etf_holdings", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.50, latency_ms=100,
        ts=(today - datetime.timedelta(days=400)).isoformat() + "T00:00:00Z",
    )
    out = get_trailing_365d_total("etf_holdings")
    assert out == 0.10  # only inside-window entry counts


def test_get_total_today_filters_to_utc_today(tmp_ledger: Path) -> None:
    today = datetime.datetime.utcnow().date()
    yesterday = today - datetime.timedelta(days=1)
    record_call(
        agent_id="rag_synthesis", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.001, latency_ms=100,
        ts=today.isoformat() + "T00:00:00Z",
    )
    record_call(
        agent_id="rag_synthesis", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.999, latency_ms=100,
        ts=yesterday.isoformat() + "T00:00:00Z",
    )
    assert get_total_today() == 0.001
    assert get_total_today(agent_id="rag_synthesis") == 0.001


def test_get_lifetime_total(tmp_ledger: Path) -> None:
    _seed_three_calls()
    assert get_lifetime_total() == round(0.001 + 0.002 + 0.0001, 8)
    assert get_lifetime_total(agent_id="r_audit") == 0.001


def test_get_call_count(tmp_ledger: Path) -> None:
    _seed_three_calls()
    assert get_call_count() == 3
    assert get_call_count(agent_id="r_audit") == 1


# ── Empty / missing ledger ──────────────────────────────────────────────────
def test_query_on_missing_ledger_returns_empty(tmp_ledger: Path) -> None:
    assert tmp_ledger.exists() is False
    assert get_calls() == []
    assert get_total_by_agent() == {}
    assert get_total_by_provider() == {}
    assert get_trailing_365d_total("r_audit") == 0.0
    assert get_lifetime_total() == 0.0
    assert get_call_count() == 0


# ── Resilience: malformed lines ─────────────────────────────────────────────
def test_malformed_line_skipped_not_raised(tmp_ledger: Path) -> None:
    """A corrupted line (e.g. partial write) should be skipped, not crash query."""
    record_call(
        agent_id="r_audit", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.001, latency_ms=100,
    )
    # Inject corruption
    with open(tmp_ledger, "a", encoding="utf-8") as f:
        f.write("THIS IS NOT JSON\n")
    record_call(
        agent_id="s6_anomaly", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.002, latency_ms=100,
    )
    out = get_calls()
    assert len(out) == 2  # bad line skipped
    assert {e.agent_id for e in out} == {"r_audit", "s6_anomaly"}


def test_empty_lines_skipped(tmp_ledger: Path) -> None:
    record_call(
        agent_id="r_audit", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.001, latency_ms=100,
    )
    with open(tmp_ledger, "a", encoding="utf-8") as f:
        f.write("\n\n\n")
    assert len(get_calls()) == 1


# ── Integrity check ─────────────────────────────────────────────────────────
def test_integrity_check_clean_ledger(tmp_ledger: Path) -> None:
    record_call(
        agent_id="r_audit", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.001, latency_ms=100,
    )
    record_call(
        agent_id="s6_anomaly", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.002, latency_ms=100,
    )
    out = integrity_check()
    assert out["exists"] is True
    assert out["total_lines"] == 2
    assert out["valid_entries"] == 2
    assert out["malformed_lines"] == []
    assert out["size_warn"] is False


def test_integrity_check_finds_malformed(tmp_ledger: Path) -> None:
    record_call(
        agent_id="r_audit", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.001, latency_ms=100,
    )
    with open(tmp_ledger, "a", encoding="utf-8") as f:
        f.write("CORRUPTED\n")
    out = integrity_check()
    assert out["total_lines"] == 2
    assert out["valid_entries"] == 1
    assert out["malformed_lines"] == [2]


def test_integrity_check_missing_ledger(tmp_ledger: Path) -> None:
    assert tmp_ledger.exists() is False
    out = integrity_check()
    assert out["exists"] is False
    assert out["total_lines"] == 0
    assert out["valid_entries"] == 0
    assert out["malformed_lines"] == []
    assert out["size_bytes"] == 0


# ── ALLOWED_* enumeration sanity ────────────────────────────────────────────
def test_allowed_agent_ids_includes_all_known_agents() -> None:
    """Sanity check: all 7 trackers from Sprint 2A audit are in allowlist."""
    expected = {
        "r_audit", "s6_anomaly", "rag_synthesis", "deepseek",
        "etf_holdings", "fomc_override", "tool1_decision_lineage",
    }
    assert expected.issubset(ALLOWED_AGENT_IDS)


def test_allowed_providers_includes_gemini_and_deepseek() -> None:
    assert {"gemini", "deepseek"}.issubset(ALLOWED_PROVIDERS)


# ── Cost rounding edge cases ────────────────────────────────────────────────
def test_cost_rounded_to_8_decimals(tmp_ledger: Path) -> None:
    out = record_call(
        agent_id="r_audit", provider="gemini", model="m",
        prompt_tokens=1, completion_tokens=1,
        cost_usd=1.234567890123456,  # more than 8 decimals
        latency_ms=100,
    )
    assert out.cost_usd == 1.23456789


def test_zero_cost_allowed(tmp_ledger: Path) -> None:
    """Zero cost is valid (e.g. cached call, dry-run)."""
    record_call(
        agent_id="r_audit", provider="gemini", model="m",
        prompt_tokens=0, completion_tokens=0, cost_usd=0.0, latency_ms=0,
    )
    assert get_call_count() == 1
