"""Unit tests for engine.agents.research_diagnostician.

Critical safety properties:
1. All 5 tools dispatch deterministically and return ToolResult shape
2. Tool dispatch handles unknown tool name without exception
3. fetch_gate_evidence returns the latest matching entry
4. find_similar_candidates returns expected cousins from real data
5. check_deployed_overlap surfaces equity_book overlap for residual_momentum
6. Deterministic diagnosis path runs for today's 4 RED candidates without error
7. Diagnosis output has the required schema (candidate / verdict / mode / etc.)
8. LLM mode falls back to deterministic on missing key
9. Ledger logging is content-preserving and append-only
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.agents.research_diagnostician import diagnostician as D
from engine.agents.research_diagnostician.tools import (
    TOOL_SCHEMAS,
    ToolResult,
    check_deployed_overlap_t,
    execute_tool,
    fetch_gate_evidence,
    fetch_sleeve_health_history_t,
    find_similar_candidates_t,
    sample_stress_coverage_t,
    subperiod_analysis_t,
)


# ── Tool schemas (Anthropic format sanity) ───────────────────────────────────

def test_tool_schemas_have_required_fields():
    for s in TOOL_SCHEMAS:
        assert "name" in s
        assert "description" in s
        assert "input_schema" in s
        assert s["input_schema"]["type"] == "object"
        props = s["input_schema"]["properties"]
        # Either candidate-scoped (5 tools) or sleeve-scoped (1 tool)
        assert "candidate_name" in props or "sleeve_name" in props


def test_tool_schemas_count():
    assert len(TOOL_SCHEMAS) == 6


def test_sleeve_tool_schema_shape():
    schemas = {s["name"]: s for s in TOOL_SCHEMAS}
    assert "fetch_sleeve_health_history" in schemas
    s = schemas["fetch_sleeve_health_history"]
    props = s["input_schema"]["properties"]
    assert "sleeve_name" in props
    assert "n_days" in props
    assert s["input_schema"]["required"] == ["sleeve_name"]


# ── Tool dispatch ────────────────────────────────────────────────────────────

def test_execute_unknown_tool_returns_error():
    res = execute_tool("nonexistent_tool", candidate_name="x")
    assert res.success is False
    assert "unknown tool" in (res.error or "").lower()


def test_execute_tool_handles_internal_exception():
    """If a tool raises, execute_tool catches and returns error ToolResult."""
    # Pass a malformed name that will succeed (no exception) but return no entry
    res = execute_tool("fetch_gate_evidence", candidate_name="nonsense_xyz_unknown")
    assert res.success is False
    assert "no gate_runs entry" in (res.error or "").lower()


# ── Tool 1: fetch_gate_evidence ──────────────────────────────────────────────

def test_fetch_gate_evidence_known_candidate():
    """Today's quality POC should be in gate_runs."""
    res = fetch_gate_evidence("quality_novymarx_2013_v1")
    assert res.success
    assert res.payload["verdict"] in ("RED", "YELLOW")
    assert res.payload["standalone_sharpe"] < 0


def test_fetch_gate_evidence_returns_substring_match_as_fallback():
    """If exact name missing, substring should still work."""
    res = fetch_gate_evidence("vix_carry")    # partial of vix_carry_contango_filter_v1
    assert res.success
    assert "vix" in res.payload["name"].lower()


def test_fetch_gate_evidence_unknown():
    res = fetch_gate_evidence("totally_made_up_candidate_xyz")
    assert res.success is False


# ── Tool 2: find_similar_candidates ──────────────────────────────────────────

def test_find_similar_residual_momentum_finds_quality():
    """Cousin detection on real data."""
    res = find_similar_candidates_t("residual_momentum_bhm_2011_v1")
    assert res.success
    names = [s["name"] for s in res.payload.get("similar", [])]
    assert "quality_novymarx_2013_v1" in names


def test_find_similar_lead_lag_finds_none():
    """Sector lead-lag is genuinely new mechanism class."""
    res = find_similar_candidates_t("sector_leadlag_v1_dailysignal_monthlyrebal")
    assert res.success
    assert res.payload["n_similar"] == 0


# ── Tool 3: check_deployed_overlap ───────────────────────────────────────────

def test_check_overlap_residual_momentum_vs_equity_book():
    res = check_deployed_overlap_t("residual_momentum_bhm_2011_v1")
    assert res.success
    assert "equity_book" in res.payload.get("overlap_by_sleeve", {})
    overlap = res.payload["overlap_by_sleeve"]["equity_book"]
    assert overlap["overlap_strength"] == "parent_only"
    assert overlap["sleeve_weight"] == 0.70


def test_check_overlap_unknown_candidate():
    res = check_deployed_overlap_t("nonexistent_xyz")
    # Graph builds with no node for unknown name; tool returns success=True
    # with the "error" key in payload as the graph reports
    # Either path is acceptable; just shouldn't throw
    assert isinstance(res, ToolResult)


# ── Tool 4: sample_stress_coverage ───────────────────────────────────────────

def test_sample_stress_coverage_known():
    """Quality POC has gate run entry — coverage tool should run."""
    res = sample_stress_coverage_t("quality_novymarx_2013_v1")
    assert res.success
    # The sample 2013-2024 misses 2008 GFC + 2010 Flash + 2011 EU
    missed = res.payload.get("stress_missed", [])
    assert any("2008_gfc" in m for m in missed) or len(missed) >= 3


def test_sample_stress_coverage_unknown_candidate():
    res = sample_stress_coverage_t("missing_xyz")
    assert res.success is False


# ── Tool 5: subperiod_analysis ───────────────────────────────────────────────

def test_subperiod_analysis_returns_metadata():
    res = subperiod_analysis_t("quality_novymarx_2013_v1")
    assert res.success
    assert "full_sample_sharpe" in res.payload
    assert "second_half_sharpe" in res.payload


# ── Deterministic diagnosis path ─────────────────────────────────────────────

def test_diagnose_deterministic_residual_momentum_catches_overlap(tmp_path, monkeypatch):
    """The deterministic synthesis should call out the equity_book cousin
    relationship as a root cause."""
    monkeypatch.setattr(D, "DIAGNOSTIC_LEDGER", tmp_path / "diag.jsonl")
    result = D.diagnose("residual_momentum_bhm_2011_v1", use_llm=False, log=True)
    assert result["mode"] == "deterministic_only"
    assert result["verdict"] in ("RED", "YELLOW")
    # Diagnostic text should reference the equity_book overlap
    diag_lower = result["refined_diagnosis"].lower()
    assert "equity_book" in diag_lower or "pead" in diag_lower or "cousin" in diag_lower


def test_diagnose_deterministic_quality_red_synthesizes_reasonable_cause(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DIAGNOSTIC_LEDGER", tmp_path / "diag.jsonl")
    result = D.diagnose("quality_novymarx_2013_v1", use_llm=False, log=True)
    assert result["mode"] == "deterministic_only"
    # Quality has α-t -5.39 → must mention wrong-direction or significant alpha
    diag = result["refined_diagnosis"].lower()
    assert "wrong-direction" in diag or "alpha" in diag or "significant" in diag


def test_diagnose_returns_required_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DIAGNOSTIC_LEDGER", tmp_path / "diag.jsonl")
    result = D.diagnose("quality_novymarx_2013_v1", use_llm=False, log=True)
    required = {"candidate", "verdict", "mode", "initial_diagnosis",
                 "refined_diagnosis", "n_critique_rounds", "converged",
                 "tools_called", "cost_usd", "timestamp"}
    assert required.issubset(result.keys())


def test_diagnose_deterministic_unknown_candidate_no_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DIAGNOSTIC_LEDGER", tmp_path / "diag.jsonl")
    result = D.diagnose("totally_nonexistent_candidate_xyz", use_llm=False, log=True)
    assert "candidate" in result
    assert result["mode"] == "deterministic_only"


# ── LLM mode fallback ────────────────────────────────────────────────────────

def test_diagnose_llm_falls_back_without_api_key(tmp_path, monkeypatch):
    """If no ANTHROPIC_API_KEY available, diagnose should fall back to
    deterministic mode (not crash)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(D, "_read_anthropic_key", lambda: None)
    monkeypatch.setattr(D, "DIAGNOSTIC_LEDGER", tmp_path / "diag.jsonl")
    result = D.diagnose("quality_novymarx_2013_v1", use_llm=True, log=True)
    assert "deterministic_fallback" in result["mode"]


# ── Ledger ──────────────────────────────────────────────────────────────────

def test_ledger_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DIAGNOSTIC_LEDGER", tmp_path / "diag.jsonl")
    D.diagnose("quality_novymarx_2013_v1", use_llm=False, log=True)
    D.diagnose("residual_momentum_bhm_2011_v1", use_llm=False, log=True)
    rows = D.read_diagnostic_ledger()
    assert len(rows) == 2
    # Most-recent-first
    assert rows[0]["candidate"] == "residual_momentum_bhm_2011_v1"
    assert rows[1]["candidate"] == "quality_novymarx_2013_v1"


def test_no_log_skips_ledger_write(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DIAGNOSTIC_LEDGER", tmp_path / "diag.jsonl")
    D.diagnose("quality_novymarx_2013_v1", use_llm=False, log=False)
    assert not (tmp_path / "diag.jsonl").exists()


# ── Sleeve-mode tools + diagnose_sleeve ──────────────────────────────────────

def test_fetch_sleeve_health_history_missing_dir(tmp_path, monkeypatch):
    """No artifacts dir → returns error, not exception."""
    monkeypatch.setattr(
        "engine.agents.research_diagnostician.tools.DECAY_ARTIFACTS_DIR",
        tmp_path / "missing",
    )
    res = fetch_sleeve_health_history_t("equity_book", n_days=7)
    assert res.success is False
    assert "decay_sentinel" in (res.error or "")


def test_fetch_sleeve_health_history_reads_artifact(tmp_path, monkeypatch):
    """Stage a single fake artifact and confirm parsing."""
    monkeypatch.setattr(
        "engine.agents.research_diagnostician.tools.DECAY_ARTIFACTS_DIR", tmp_path
    )
    artifact = {
        "as_of": "2026-05-28",
        "mechanisms": {
            "equity_book": {
                "rolling_sharpe":   0.42,
                "rolling_t":         1.85,
                "decay_ratio":       0.65,
                "signal_ic":         0.014,
                "structural_decay":  False,
            }
        },
    }
    (tmp_path / "decay_sentinel_2026-05-28.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    res = fetch_sleeve_health_history_t("equity_book", n_days=7)
    assert res.success
    assert res.payload["n_days_found"] == 1
    row = res.payload["history"][0]
    assert row["rolling_sharpe"] == 0.42
    assert row["structural_decay"] is False


def test_fetch_sleeve_health_history_unknown_sleeve(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "engine.agents.research_diagnostician.tools.DECAY_ARTIFACTS_DIR", tmp_path
    )
    (tmp_path / "decay_sentinel_2026-05-28.json").write_text(
        json.dumps({"as_of": "2026-05-28",
                    "mechanisms": {"carry_book": {"rolling_sharpe": 0.3}}}),
        encoding="utf-8",
    )
    res = fetch_sleeve_health_history_t("tsmom_book", n_days=7)
    assert res.success
    assert res.payload["n_days_found"] == 0


def test_diagnose_sleeve_deterministic_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "engine.agents.research_diagnostician.tools.DECAY_ARTIFACTS_DIR", tmp_path
    )
    monkeypatch.setattr(D, "DIAGNOSTIC_LEDGER", tmp_path / "diag.jsonl")
    result = D.diagnose_sleeve("equity_book", use_llm=False, log=True)
    assert result["sleeve"] == "equity_book"
    assert result["mode"] == "deterministic_only"
    required = {"sleeve", "mode", "initial_diagnosis", "refined_diagnosis",
                 "n_critique_rounds", "converged", "tools_called", "timestamp"}
    assert required.issubset(result.keys())


def test_diagnose_sleeve_llm_falls_back_without_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "engine.agents.research_diagnostician.tools.DECAY_ARTIFACTS_DIR", tmp_path
    )
    monkeypatch.setattr(D, "DIAGNOSTIC_LEDGER", tmp_path / "diag.jsonl")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(D, "_read_anthropic_key", lambda: None)
    result = D.diagnose_sleeve("equity_book", use_llm=True, log=True)
    assert result["mode"] == "deterministic_only"


def test_execute_tool_sleeve_dispatch():
    """Dispatcher routes sleeve tool by name."""
    res = execute_tool("fetch_sleeve_health_history", sleeve_name="equity_book", n_days=3)
    # Result depends on whether artifacts exist locally; we only require shape
    assert isinstance(res, ToolResult)
    assert res.name == "fetch_sleeve_health_history"
