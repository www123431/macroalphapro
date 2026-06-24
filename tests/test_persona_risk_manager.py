"""tests/test_persona_risk_manager.py — Persona MVP tests (parametrized over all agents).

Post-2026-05-19 refactor: tests parametrize over every AgentPersona
in engine.agents.persona. Adding a new agent = adding one entry to
_PERSONAS list. Shared invariants (banned phrases / NO EMOJIS rule /
READ-ONLY assertion / chat loop termination / unknown-tool handling)
run against each persona automatically.

Filename kept for git history continuity even though the test scope
expanded beyond RM alone.
"""
from __future__ import annotations

import json
import os
import re
from unittest.mock import patch

import pytest

from engine.agents.persona import (
    ANOMALY_SENTINEL,
    ATTRIBUTION_ANALYST,
    AUDIT_RECORDER,
    CHIEF_OF_STAFF,
    DECAY_SENTINEL,
    DEVILS_ADVOCATE,
    DQ_INSPECTOR,
    RISK_MANAGER,
    AgentPersona,
    AgentTurnResult,
    chat_turn,
)
from engine.agents.persona.tools import (
    TOOL_SCHEMAS,
    delegate_to_specialist,
    execute_tool,
    forensic_ticker_check,
    list_personas,
    lookup_strategy_status,
    query_audit_findings,
    query_audit_runs,
    query_recent_alerts,
    query_recent_anomalies,
    read_nav_history,
    read_today_book_state,
    select_tools,
)
from engine.llm.call import LLMCallResult, ToolCall


# Parametrize all persona tests over the current registry. To add a new
# agent, add one entry here — existing invariants enforced automatically.
_PERSONAS: list[AgentPersona] = [
    CHIEF_OF_STAFF,
    RISK_MANAGER, DQ_INSPECTOR, DEVILS_ADVOCATE,
    ANOMALY_SENTINEL, ATTRIBUTION_ANALYST, AUDIT_RECORDER,
    DECAY_SENTINEL,
]


# ──────────────────────────────────────────────────────────────────────────────
# Tools — schemas (persona-agnostic)
# ──────────────────────────────────────────────────────────────────────────────
class TestToolSchemas:
    def test_registry_has_expected_tools(self):
        # Tier 3a/3b memory tools added 2026-05-19 (Phase A.1).
        # Anomaly-sentinel tools added 2026-05-19 (Phase A.4 persona build).
        # Attribution + Audit Recorder tools added 2026-05-19 (Phase A.5/A.6).
        # Chief of Staff orchestration tools added 2026-05-19 (Phase A.7
        # spec id=74). DQ pre-batch + Tier 2.5 recall added 2026-05-19
        # (Phase A.7 Wave 4.2/4.3).
        names = {t["name"] for t in TOOL_SCHEMAS}
        assert names == {
            # cross-agent
            "query_recent_alerts",       # alert DB read
            "read_today_book_state",     # current book artifact
            "lookup_strategy_status",    # per-strategy signal
            "lookup_spec",               # Tier 3a structured spec lookup
            "read_project_memory",       # Tier 3b curated knowledge search
            # Anomaly Sentinel-owned
            "query_recent_anomalies",    # AnomalyFlag history
            "forensic_ticker_check",     # live z-score / volume / drawdown
            # Decay Sentinel-owned (2026-05-22 — book-health monitor)
            "read_decay_sentinel_report",  # deterministic decay/diversification report
            # Attribution Analyst-owned
            "read_nav_history",          # PortfolioNavSnapshot history
            # DQ Inspector-owned (Wave 4.2)
            "run_dq_pre_batch_check",    # live Mode 1-4 gates
            # Audit Recorder-owned
            "query_audit_findings",      # AuditFinding by severity/days/status
            "query_audit_runs",          # AuditRun by scope/days
            # Chief of Staff-owned (Supervisor pattern)
            "delegate_to_specialist",    # route to one specialist
            "list_personas",             # static six-specialist directory
            # Cross-session memory (Wave 4.3, available to CoS by default)
            "recall_past_turns",         # ChatTurnEmbedding cosine search
        }

    def test_every_schema_has_required_fields(self):
        for tool in TOOL_SCHEMAS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"
            assert "properties" in tool["input_schema"]

    def test_select_tools_preserves_order_and_typo_fails(self):
        # The per-agent subset helper is the seam that prevents tool
        # capability from silently widening when a new tool is added
        # to the shared registry.
        subset = select_tools(["lookup_spec", "read_project_memory"])
        assert [t["name"] for t in subset] == [
            "lookup_spec", "read_project_memory",
        ]
        import pytest
        with pytest.raises(ValueError, match="unknown tool name"):
            select_tools(["nonexistent_tool"])


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations (persona-agnostic)
# ──────────────────────────────────────────────────────────────────────────────
class TestToolImplementations:
    def test_query_recent_alerts_returns_valid_json(self):
        result = query_recent_alerts(days_back=7, severity_min="LIGHT")
        parsed = json.loads(result)
        assert "n_alerts" in parsed or "error" in parsed

    def test_lookup_strategy_unknown_name_returns_error(self):
        result = lookup_strategy_status(strategy_name="NOT_A_STRATEGY")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "available" in parsed

    def test_read_today_book_state_returns_json(self):
        result = read_today_book_state()
        parsed = json.loads(result)
        assert "error" in parsed or "as_of" in parsed

    def test_query_recent_anomalies_returns_valid_json(self):
        # Empty AnomalyFlag table (test DB) → graceful zero-row payload.
        result = query_recent_anomalies(days_back=30)
        parsed = json.loads(result)
        assert "n_flags" in parsed or "error" in parsed

    def test_query_recent_anomalies_filters_by_ticker(self):
        # Filter applied even when table empty — must not throw.
        result = query_recent_anomalies(ticker="DOES_NOT_EXIST", days_back=7)
        parsed = json.loads(result)
        # Either a clean zero-row payload, or a structural error (no
        # AnomalyFlag table in test DB). Neither should be a crash.
        assert "n_flags" in parsed or "error" in parsed

    def test_forensic_ticker_check_handles_bogus_ticker(self):
        # yfinance returns empty for unknown ticker; tool must return
        # structured error rather than raise.
        result = forensic_ticker_check("DEFINITELY_NOT_A_REAL_TICKER_XYZ123")
        parsed = json.loads(result)
        assert "error" in parsed

    def test_read_nav_history_returns_valid_json(self):
        # Empty PortfolioNavSnapshot table (test DB) → zero-row payload.
        result = read_nav_history(days_back=30)
        parsed = json.loads(result)
        assert "n_rows" in parsed or "error" in parsed

    def test_read_nav_history_clamps_days_back(self):
        # Excessive days_back must not crash — function clamps to 365.
        result = read_nav_history(days_back=99999)
        parsed = json.loads(result)
        assert "n_rows" in parsed or "error" in parsed

    def test_query_audit_findings_returns_valid_json(self):
        result = query_audit_findings(severity_min="LOW", days_back=7)
        parsed = json.loads(result)
        assert "n_findings" in parsed or "error" in parsed

    def test_query_audit_findings_severity_filter(self):
        # Filter applied; even when table empty must not throw.
        result = query_audit_findings(severity_min="HIGH", days_back=1)
        parsed = json.loads(result)
        assert "n_findings" in parsed or "error" in parsed

    def test_query_audit_runs_returns_valid_json(self):
        result = query_audit_runs(scope="all", days_back=30)
        parsed = json.loads(result)
        assert "n_runs" in parsed or "error" in parsed


class TestExecuteToolDispatch:
    def test_dispatches_known_tool(self):
        output, is_error = execute_tool("query_recent_alerts", {"days_back": 1})
        assert isinstance(json.loads(output), dict)
        # is_error may be True or False — depends whether tool internals
        # surfaced an error (e.g. no alerts table yet). Type must be bool.
        assert isinstance(is_error, bool)

    def test_unknown_tool_returns_error_flag(self):
        output, is_error = execute_tool("not_a_real_tool", {})
        parsed = json.loads(output)
        assert "error" in parsed and "unknown tool" in parsed["error"]
        assert is_error is True

    def test_bad_args_caught_and_flagged(self):
        output, is_error = execute_tool("lookup_strategy_status", {})
        assert "error" in json.loads(output)
        assert is_error is True

    def test_lookup_unknown_strategy_flags_is_error(self):
        """Tool returns error JSON → executor sets is_error=True."""
        output, is_error = execute_tool(
            "lookup_strategy_status", {"strategy_name": "BOGUS"}
        )
        assert "error" in json.loads(output)
        assert is_error is True

    def test_lookup_spec_unknown_id_flags_error(self):
        """Unknown spec_id returns is_error=True with helpful available_ids."""
        output, is_error = execute_tool("lookup_spec", {"spec_id": 99999})
        parsed = json.loads(output)
        assert is_error is True
        assert "error" in parsed
        assert "available_ids" in parsed   # actionable hint for the agent

    def test_read_project_memory_keyword_search_returns_matches(self):
        """Keyword search returns top-N matches with description previews."""
        output, is_error = execute_tool(
            "read_project_memory", {"query": "emoji", "mode": "keyword"}
        )
        parsed = json.loads(output)
        # If memory dir resolves on this host, expect hits; if not,
        # expect graceful error rather than crash
        assert is_error is False or (is_error and "error" in parsed)
        if not is_error:
            assert parsed.get("mode") == "keyword_search"
            assert parsed.get("n_hits", 0) >= 1   # "emoji" should find feedback file
            # top entries must have file + description fields
            for hit in parsed.get("top", []):
                assert "file" in hit
                assert "match_count" in hit

    def test_read_project_memory_missing_query_param_flags_error(self):
        """Missing required query → bad-args error."""
        output, is_error = execute_tool("read_project_memory", {})
        parsed = json.loads(output)
        assert is_error is True
        assert "error" in parsed

    def test_read_project_memory_invalid_mode_flags_error(self):
        """Unknown mode value returns structured error."""
        output, is_error = execute_tool(
            "read_project_memory",
            {"query": "anything", "mode": "fuzzy_match"},
        )
        parsed = json.loads(output)
        assert is_error is True
        assert "valid_modes" in parsed

    def test_read_project_memory_semantic_mode_uses_index(self, monkeypatch):
        """Phase A.7 Wave 3.3: semantic mode calls memory_index.search_memory
        and surfaces the structured results. Mocked so test doesn't load
        the real sentence-transformers model."""
        fake_hits = [
            {"file": "feedback_no_emojis_2026-05-19.md",
             "score": 0.812, "description": "no emojis anywhere"},
            {"file": "feedback_ui_institutional_minimum.md",
             "score": 0.341, "description": "UI tone discipline"},
        ]
        monkeypatch.setattr(
            "engine.agents.persona.memory_index.search_memory",
            lambda query, top_k=5: fake_hits,
        )
        output, is_error = execute_tool(
            "read_project_memory",
            {"query": "what is the rule about emojis", "mode": "semantic"},
        )
        parsed = json.loads(output)
        assert is_error is False
        assert parsed["mode"] == "semantic_search"
        assert parsed["n_hits"] == 2
        assert parsed["top"][0]["score"] == 0.812

    def test_read_project_memory_auto_prefers_exact_over_semantic(self, monkeypatch):
        """A query that matches a filename stem must short-circuit to
        exact_name_match even when semantic would also return results.
        Filename stem is more specific signal than embedding similarity."""
        called = {"semantic": False}

        def _fake_semantic(query, top_k=5):
            called["semantic"] = True
            return [{"file": "x.md", "score": 1.0, "description": ""}]
        monkeypatch.setattr(
            "engine.agents.persona.memory_index.search_memory", _fake_semantic,
        )
        output, _ = execute_tool(
            "read_project_memory",
            {"query": "feedback_no_emojis_2026-05-19", "mode": "auto"},
        )
        parsed = json.loads(output)
        # If the memory dir resolves on this host, exact match wins
        if "error" not in parsed:
            assert parsed["mode"] == "exact_name_match"
            assert called["semantic"] is False, (
                "auto mode hit semantic when it should have short-circuited "
                "to exact_name_match"
            )


# ──────────────────────────────────────────────────────────────────────────────
# AgentPersona registry — every persona must satisfy these invariants
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("persona", _PERSONAS, ids=lambda p: p.agent_id)
class TestPersonaInvariants:
    """Every AgentPersona instance must satisfy these constraints."""

    def test_role_id_present(self, persona):
        assert persona.role_id
        assert persona.role_id in persona.system_prompt

    def test_lists_banned_vocabulary(self, persona):
        for banned in ["maybe", "perhaps", "probably", "seems to", "i think"]:
            assert banned in persona.system_prompt.lower(), (
                f"{persona.agent_id}: banned vocab list missing {banned!r}"
            )

    def test_explicit_no_emojis_rule(self, persona):
        assert "NO EMOJIS" in persona.system_prompt

    def test_explicit_read_only_authority(self, persona):
        assert "READ-ONLY" in persona.system_prompt

    def test_no_emoji_chars_in_prompt(self, persona):
        emojis = re.findall(r"[\U0001F300-\U0001FAFF]", persona.system_prompt)
        assert emojis == [], (
            f"{persona.agent_id}: system prompt contains emoji {emojis!r}"
        )

    def test_spec_ref_populated(self, persona):
        # spec_ref must be non-empty (either "spec id=N hash=X" or
        # "doctrine: <memory-name>" for agents grounded in feedback files)
        assert persona.spec_ref
        assert len(persona.spec_ref) > 5

    def test_agent_id_is_in_allowed_set(self, persona):
        from engine.llm_cost_ledger import ALLOWED_AGENT_IDS
        assert persona.agent_id in ALLOWED_AGENT_IDS, (
            f"{persona.agent_id}: not in ALLOWED_AGENT_IDS; add to "
            f"engine.llm_cost_ledger.ALLOWED_AGENT_IDS frozenset"
        )

    def test_workload_is_routed(self, persona):
        from engine.llm.call import _WORKLOAD_ROUTING
        assert persona.workload in _WORKLOAD_ROUTING, (
            f"{persona.agent_id}: workload {persona.workload!r} not in router"
        )

    def test_tools_well_formed(self, persona):
        # Tools list may be empty (e.g. DA persona = evidence-only critique,
        # no tool use by design). When non-empty, each tool must be a valid
        # Anthropic schema.
        for tool in persona.tools:
            assert "name" in tool
            assert "input_schema" in tool

    def test_memory_tier_discipline_present(self, persona):
        """Every persona must instruct the model to not cite user's prior
        chat statements as ground-truth evidence for new decisions (HARKing
        defense via memory tier separation). The exact phrasing differs
        per persona but the concept must be explicit."""
        prompt_lower = persona.system_prompt.lower()
        # Either "memory" tier vocabulary or "evidence" boundary terminology
        # must appear, AND user's prior statements must be marked as
        # input/claim (not ground truth).
        tier_vocab = any(t in prompt_lower for t in [
            "memory tier", "evidence boundary", "harking",
            "conversation history", "input claims",
            "memory and evidence",
        ])
        assert tier_vocab, (
            f"{persona.agent_id}: system prompt lacks memory-tier / "
            f"evidence-boundary discipline language"
        )

    def test_tool_executor_callable(self, persona):
        result = persona.tool_executor("not_a_real_tool", {})
        # Must return (str, bool) tuple — never raise
        assert isinstance(result, tuple) and len(result) == 2
        output, is_error = result
        assert isinstance(output, str)
        assert isinstance(is_error, bool)
        # Unknown tool always flags as error (regardless of whether
        # persona has tools — defensive)
        assert is_error is True


# ──────────────────────────────────────────────────────────────────────────────
# RM-specific invariants (separate from DQ — distinguishes the personas)
# ──────────────────────────────────────────────────────────────────────────────
class TestRiskManagerSpecific:
    def test_role_is_head_of_risk(self):
        assert RISK_MANAGER.role_id == "head_of_risk_blackrock_slack"
        assert "Head of Risk" in RISK_MANAGER.system_prompt

    def test_lists_rm_modes_not_dq_modes(self):
        # RM scope includes single-ticker / sleeve drift / VaR / HHI etc.
        for token in ["Single-ticker", "Sleeve drift", "VaR", "HHI"]:
            assert token in RISK_MANAGER.system_prompt

    def test_redirects_data_freshness_to_dq(self):
        assert "DQ Inspector" in RISK_MANAGER.system_prompt


# ──────────────────────────────────────────────────────────────────────────────
# DQ-specific invariants
# ──────────────────────────────────────────────────────────────────────────────
class TestDQInspectorSpecific:
    def test_role_is_data_quality_inspector(self):
        assert DQ_INSPECTOR.role_id == "data_quality_inspector_blackrock_slack"
        assert "Data Quality Inspector" in DQ_INSPECTOR.system_prompt

    def test_lists_dq_modes_not_rm_modes(self):
        for token in ["FRED", "yfinance", "bab_compat", "NaN burst", "Row-count"]:
            assert token in DQ_INSPECTOR.system_prompt

    def test_redirects_strategy_pnl_to_rm(self):
        assert "Risk Manager" in DQ_INSPECTOR.system_prompt

    def test_owns_live_dq_gate_tool(self):
        """Phase A.7 Wave 4.2: DQ persona must expose run_dq_pre_batch_check
        so it can give a live verdict instead of only reading historical
        DataQualityAlert rows."""
        names = {t["name"] for t in DQ_INSPECTOR.tools}
        assert "run_dq_pre_batch_check" in names


# ──────────────────────────────────────────────────────────────────────────────
# DA-specific invariants (constrained-evidence reasoning)
# ──────────────────────────────────────────────────────────────────────────────
class TestDevilsAdvocateSpecific:
    def test_role_is_devils_advocate(self):
        assert DEVILS_ADVOCATE.role_id == "devils_advocate_constrained_evidence"
        assert "Devil's Advocate" in DEVILS_ADVOCATE.system_prompt

    def test_routes_to_deepseek_v4_pro(self):
        from engine.llm.call import _WORKLOAD_ROUTING
        provider, model = _WORKLOAD_ROUTING[DEVILS_ADVOCATE.workload]
        assert provider == "deepseek"
        assert "v4-pro" in model.lower() or "pro" in model.lower()

    def test_evidence_only_rule_explicit_in_prompt(self):
        # The constraint that neutralizes SimpleQA gap must be loud and clear
        prompt_lower = DEVILS_ADVOCATE.system_prompt.lower()
        assert "evidence" in prompt_lower
        assert "insufficient evidence" in prompt_lower
        assert "external knowledge" in prompt_lower

    def test_no_validate_or_congratulate(self):
        assert "do not validate" in DEVILS_ADVOCATE.system_prompt.lower()
        assert "do not congratulate" in DEVILS_ADVOCATE.system_prompt.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly Sentinel-specific invariants (per-ticker forensic scope)
# ──────────────────────────────────────────────────────────────────────────────
class TestAnomalySentinelSpecific:
    def test_role_is_anomaly_sentinel(self):
        assert ANOMALY_SENTINEL.role_id == "anomaly_sentinel_forensic"
        assert "Anomaly Sentinel" in ANOMALY_SENTINEL.system_prompt

    def test_owns_anomaly_specific_tools(self):
        # The 2 tools that belong to the Sentinel and only the Sentinel
        # (RM / DQ do not include them in their palettes).
        names = {t["name"] for t in ANOMALY_SENTINEL.tools}
        assert "query_recent_anomalies" in names
        assert "forensic_ticker_check" in names

    def test_does_not_own_book_level_tools(self):
        # Per-ticker forensic only — book state / alert DB belong to RM/DQ.
        names = {t["name"] for t in ANOMALY_SENTINEL.tools}
        assert "read_today_book_state" not in names
        assert "query_recent_alerts" not in names

    def test_routes_to_sonnet(self):
        from engine.llm.call import _WORKLOAD_ROUTING
        provider, model = _WORKLOAD_ROUTING[ANOMALY_SENTINEL.workload]
        assert provider == "anthropic"
        assert "sonnet" in model.lower()

    def test_redirects_book_level_to_rm(self):
        # Cross-agent reference doctrine: book-level → Risk Manager.
        assert "Risk Manager" in ANOMALY_SENTINEL.system_prompt

    def test_redirects_data_freshness_to_dq(self):
        assert "DQ Inspector" in ANOMALY_SENTINEL.system_prompt

    def test_lists_forensic_vocabulary(self):
        # The persona must use forensic-domain vocabulary in its prompt
        # so the model anchors on the right scope.
        prompt_lower = ANOMALY_SENTINEL.system_prompt.lower()
        for token in ["z-score", "volume", "drawdown",
                      "confidence_likert", "anomalyflag"]:
            assert token in prompt_lower, (
                f"Sentinel prompt missing forensic vocab {token!r}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Attribution Analyst-specific invariants (P&L decomposition scope)
# ──────────────────────────────────────────────────────────────────────────────
class TestAttributionAnalystSpecific:
    def test_role_is_attribution_analyst(self):
        assert ATTRIBUTION_ANALYST.role_id == "attribution_analyst_forensic"
        assert "Attribution Analyst" in ATTRIBUTION_ANALYST.system_prompt

    def test_owns_nav_history_tool(self):
        names = {t["name"] for t in ATTRIBUTION_ANALYST.tools}
        assert "read_nav_history" in names
        assert "read_today_book_state" in names
        assert "lookup_strategy_status" in names

    def test_does_not_own_audit_or_anomaly_tools(self):
        # Attribution stays clear of audit-trail + anomaly forensic scope.
        names = {t["name"] for t in ATTRIBUTION_ANALYST.tools}
        assert "query_audit_findings" not in names
        assert "forensic_ticker_check" not in names

    def test_routes_to_sonnet(self):
        from engine.llm.call import _WORKLOAD_ROUTING
        provider, model = _WORKLOAD_ROUTING[ATTRIBUTION_ANALYST.workload]
        assert provider == "anthropic"
        assert "sonnet" in model.lower()

    def test_states_factor_regression_limit_honestly(self):
        # Critical doctrine: do not fabricate factor betas / Brinson decomp.
        prompt_lower = ATTRIBUTION_ANALYST.system_prompt.lower()
        assert "no factor-regression tool" in prompt_lower
        assert "do not fabricate" in prompt_lower

    def test_redirects_per_ticker_to_sentinel(self):
        assert "Anomaly Sentinel" in ATTRIBUTION_ANALYST.system_prompt

    def test_redirects_risk_to_rm(self):
        assert "Risk Manager" in ATTRIBUTION_ANALYST.system_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Audit Recorder-specific invariants (report state, not rule on state)
# ──────────────────────────────────────────────────────────────────────────────
class TestAuditRecorderSpecific:
    def test_role_is_audit_recorder(self):
        assert AUDIT_RECORDER.role_id == "audit_recorder_governance"
        assert "Audit Recorder" in AUDIT_RECORDER.system_prompt

    def test_owns_audit_tools(self):
        names = {t["name"] for t in AUDIT_RECORDER.tools}
        assert "query_audit_findings" in names
        assert "query_audit_runs" in names
        assert "lookup_spec" in names
        assert "query_recent_alerts" in names

    def test_does_not_own_live_verdict_tools(self):
        # Audit Recorder reports HISTORY; it must not have a tool that
        # would tempt the model into rendering a live verdict.
        names = {t["name"] for t in AUDIT_RECORDER.tools}
        assert "read_today_book_state" not in names
        assert "forensic_ticker_check" not in names
        assert "read_nav_history" not in names

    def test_routes_to_sonnet(self):
        from engine.llm.call import _WORKLOAD_ROUTING
        provider, model = _WORKLOAD_ROUTING[AUDIT_RECORDER.workload]
        assert provider == "anthropic"
        assert "sonnet" in model.lower()

    def test_report_not_rule_doctrine_explicit(self):
        # Critical doctrine: REPORT state, do not RULE on state.
        prompt_lower = AUDIT_RECORDER.system_prompt.lower()
        assert "report state" in prompt_lower or "report" in prompt_lower
        assert "do not rule" in prompt_lower or "not a verdict" in prompt_lower

    def test_lists_governance_vocabulary(self):
        prompt_lower = AUDIT_RECORDER.system_prompt.lower()
        for token in ["auditfinding", "auditrun", "specregistry",
                      "amendment_log", "pendingapproval"]:
            assert token in prompt_lower, (
                f"Audit Recorder prompt missing governance vocab {token!r}"
            )

    def test_redirects_live_verdict_to_rm(self):
        assert "Risk Manager" in AUDIT_RECORDER.system_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Chief of Staff-specific invariants (Supervisor pattern, spec id=74)
# ──────────────────────────────────────────────────────────────────────────────
class TestChiefOfStaffSpecific:
    def test_role_is_chief_of_staff(self):
        assert CHIEF_OF_STAFF.role_id == "chief_of_staff_supervisor"
        assert "Chief of Staff" in CHIEF_OF_STAFF.system_prompt

    def test_owns_delegation_tool(self):
        # The ONE tool that distinguishes CoS from every other persona.
        # If this disappears, the Supervisor pattern collapses.
        names = {t["name"] for t in CHIEF_OF_STAFF.tools}
        assert "delegate_to_specialist" in names
        assert "list_personas" in names
        # Phase A.7 Wave 4.3: cross-session memory recall belongs to CoS.
        assert "recall_past_turns" in names

    def test_does_not_own_specialist_tools(self):
        # CoS must route, not access state directly. If CoS had
        # read_today_book_state or query_recent_alerts, the supervisor
        # pattern would degrade into "CoS does everything itself".
        names = {t["name"] for t in CHIEF_OF_STAFF.tools}
        for forbidden in (
            "read_today_book_state",
            "query_recent_alerts",
            "read_nav_history",
            "forensic_ticker_check",
            "query_recent_anomalies",
            "query_audit_findings",
            "query_audit_runs",
            "lookup_strategy_status",
        ):
            assert forbidden not in names, (
                f"CoS must NOT own {forbidden!r} — it should delegate. "
                f"If CoS owns this, the Supervisor pattern is broken."
            )

    def test_lists_all_six_specialists_in_prompt(self):
        # Routing rules can only work if the model knows the specialists
        # exist. Spec id=74 §3.2 routing table. (7 specialists since 2026-05-22.)
        for aid in ("risk_manager", "dq_inspector", "anomaly_sentinel",
                    "attribution_analyst", "audit_recorder",
                    "devils_advocate", "decay_sentinel"):
            assert aid in CHIEF_OF_STAFF.system_prompt, (
                f"CoS prompt missing specialist agent_id {aid!r}"
            )

    def test_routing_keywords_in_prompt(self):
        # Sample of high-signal keywords from spec §3.2 routing table.
        # If these vanish the routing accuracy degrades silently.
        prompt = CHIEF_OF_STAFF.system_prompt
        for kw in ("VaR", "FRED", "z-score", "NAV",
                   "amendment", "critique"):
            assert kw in prompt, (
                f"CoS prompt missing routing keyword {kw!r}"
            )

    def test_pattern_5_ban_explicit(self):
        # Structurally enforced, but the prompt must also describe it
        # so the model does not LIE about specialists having talked to
        # each other. Look for any phrasing of the ban.
        prompt_lower = CHIEF_OF_STAFF.system_prompt.lower()
        assert "pattern 5" in prompt_lower or "isolated" in prompt_lower, (
            "CoS prompt must reference Pattern 5 ban / isolation"
        )

    def test_routes_to_sonnet(self):
        from engine.llm.call import _WORKLOAD_ROUTING
        provider, model = _WORKLOAD_ROUTING[CHIEF_OF_STAFF.workload]
        assert provider == "anthropic"
        assert "sonnet" in model.lower()


# ──────────────────────────────────────────────────────────────────────────────
# delegate_to_specialist + list_personas direct tests
# ──────────────────────────────────────────────────────────────────────────────
class TestDelegationTool:
    def test_list_personas_returns_seven(self):
        # 2026-05-22: decay_sentinel wired in as the 7th CoS specialist.
        parsed = json.loads(list_personas())
        assert "specialists" in parsed
        assert len(parsed["specialists"]) == 7
        agent_ids = {s["agent_id"] for s in parsed["specialists"]}
        assert agent_ids == {
            "risk_manager", "dq_inspector", "anomaly_sentinel",
            "attribution_analyst", "audit_recorder", "devils_advocate",
            "decay_sentinel",
        }

    def test_delegate_unknown_specialist_returns_error(self):
        out = delegate_to_specialist("not_a_real_agent", "test query")
        parsed = json.loads(out)
        assert "error" in parsed
        assert "available" in parsed
        assert "risk_manager" in parsed["available"]

    def test_delegate_dispatches_isolated_chat_turn(self, monkeypatch):
        """delegate must call chat_turn with history=[] (isolation contract)
        and return a structured JSON with from_agent / answer / cost."""
        captured = {}

        class _StubResult:
            final_text       = "stub specialist answer"
            n_iterations     = 2
            total_cost_usd   = 0.0042
            total_latency_ms = 123
            stop_reason      = "end_turn"
            tool_calls_log   = ()   # Wave 3.1: delegate reads this for derived signal

        def _stub_chat_turn(persona, user_message, history=None,
                            max_tokens=None, effort=None):
            captured["agent_id"] = persona.agent_id
            captured["history"]  = history
            captured["msg"]      = user_message
            return _StubResult()

        monkeypatch.setattr(
            "engine.agents.persona.base.chat_turn", _stub_chat_turn,
        )

        out = delegate_to_specialist("risk_manager", "VaR status today?")
        parsed = json.loads(out)
        assert parsed["from_agent"] == "risk_manager"
        assert parsed["answer"] == "stub specialist answer"
        assert parsed["n_iterations"] == 2
        # The isolation contract: history MUST be empty list, not None
        # and not threaded from CoS context.
        assert captured["history"] == []
        assert captured["agent_id"] == "risk_manager"

    def test_delegate_catches_specialist_exception(self, monkeypatch):
        """If chat_turn raises, delegate must return structured error
        rather than propagate — CoS must keep running."""
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated specialist crash")
        monkeypatch.setattr(
            "engine.agents.persona.base.chat_turn", _boom,
        )

        out = delegate_to_specialist("dq_inspector", "anything")
        parsed = json.loads(out)
        assert "error" in parsed
        assert parsed.get("from_agent") == "dq_inspector"

    def test_delegate_surfaces_structured_signal(self, monkeypatch):
        """Phase A.7 Wave 3.1: delegate must return derived fields
        (tools_called / confidence_signal / answer_is_refusal /
        answer_is_empty) so CoS can reason programmatically over the
        specialist's response without trusting the LLM to self-label."""
        class _Result:
            final_text       = "K1 BAB is running clean today."
            n_iterations     = 3
            total_cost_usd   = 0.0089
            total_latency_ms = 450
            stop_reason      = "end_turn"
            tool_calls_log   = (
                {"name": "lookup_strategy_status", "input": {}, "result_preview": "..."},
                {"name": "query_recent_alerts",    "input": {}, "result_preview": "..."},
            )
        monkeypatch.setattr(
            "engine.agents.persona.base.chat_turn",
            lambda persona, user_message, history=None, max_tokens=None,
                   effort=None: _Result(),
        )

        out = delegate_to_specialist("risk_manager", "K1 BAB status?")
        parsed = json.loads(out)
        # Derived signal must be present
        assert parsed["tools_called"] == [
            "lookup_strategy_status", "query_recent_alerts",
        ]
        # 2+ iterations + tools_called + end_turn = high confidence
        assert parsed["confidence_signal"] == "high"
        assert parsed["answer_is_refusal"] is False
        assert parsed["answer_is_empty"] is False

    def test_delegate_flags_refusal(self, monkeypatch):
        """Refusal-language detector lets CoS see when a specialist
        bounced back — it can re-route to the proper specialist
        instead of relaying a bare refusal."""
        class _Result:
            final_text       = "Refused — Risk Manager is read-only."
            n_iterations     = 1
            total_cost_usd   = 0.001
            total_latency_ms = 60
            stop_reason      = "end_turn"
            tool_calls_log   = ()
        monkeypatch.setattr(
            "engine.agents.persona.base.chat_turn",
            lambda *a, **kw: _Result(),
        )

        out = delegate_to_specialist("risk_manager", "Force unhalt!")
        parsed = json.loads(out)
        assert parsed["answer_is_refusal"] is True
        # Single iteration + no tools = med confidence (clean termination
        # but specialist didn't consult any state).
        assert parsed["confidence_signal"] == "med"

    def test_delegate_flags_low_confidence_on_max_iterations(self, monkeypatch):
        """If specialist hits the iteration cap, confidence_signal=low
        so CoS knows the answer may be truncated."""
        class _Result:
            final_text       = "partial..."
            n_iterations     = 4   # = max_iterations cap below
            total_cost_usd   = 0.01
            total_latency_ms = 1200
            stop_reason      = "tool_use"
            tool_calls_log   = (
                {"name": "query_recent_alerts",    "input": {}},
                {"name": "lookup_strategy_status", "input": {}},
                {"name": "query_recent_alerts",    "input": {}},
                {"name": "lookup_strategy_status", "input": {}},
            )
        monkeypatch.setattr(
            "engine.agents.persona.base.chat_turn",
            lambda *a, **kw: _Result(),
        )

        out = delegate_to_specialist(
            "risk_manager", "anything", max_iterations=4,
        )
        parsed = json.loads(out)
        assert parsed["confidence_signal"] == "low"


# ──────────────────────────────────────────────────────────────────────────────
# Agent loop — parametrized mocked tests (works for any AgentPersona)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("persona", _PERSONAS, ids=lambda p: p.agent_id)
class TestAgentLoopMocked:
    def _mock_result(self, text: str, tool_calls=None,
                     stop_reason: str = "end_turn") -> LLMCallResult:
        return LLMCallResult(
            text              = text,
            tool_calls        = tuple(
                ToolCall(id=tc["id"], name=tc["name"], input=tc["input"])
                for tc in (tool_calls or [])
            ),
            stop_reason       = stop_reason,
            model             = "mock-model",
            provider          = "anthropic",
            cost_usd          = 0.001,
            latency_ms        = 100,
            cache_read_tokens = 0,
            raw_usage         = {},
        )

    def test_terminal_text_no_tools(self, persona):
        mock = self._mock_result(text=f"{persona.name} responds.")
        with patch("engine.llm.call.call", return_value=mock):
            r = chat_turn(persona, "status?")
        assert r.final_text == f"{persona.name} responds."
        assert r.n_iterations == 1
        assert r.tool_calls_log == ()

    def test_one_tool_round_trip(self, persona):
        # Skip personas configured for single-turn-only (max_iterations=1)
        # — they cannot complete a tool round-trip by design (e.g. DA).
        if persona.max_iterations < 2:
            pytest.skip(
                f"{persona.agent_id} is single-turn (max_iterations="
                f"{persona.max_iterations}); tool round-trip n/a"
            )
        iter1 = self._mock_result(
            text="",
            tool_calls=[{"id": "toolu_1", "name": "query_recent_alerts",
                         "input": {"days_back": 7}}],
            stop_reason="tool_use",
        )
        iter2 = self._mock_result(text="Zero alerts.")
        with patch("engine.llm.call.call", side_effect=[iter1, iter2]):
            r = chat_turn(persona, "alerts?")
        assert r.n_iterations == 2
        assert r.final_text == "Zero alerts."
        assert len(r.tool_calls_log) == 1

    def test_max_iterations_cap_reached(self, persona):
        looping = self._mock_result(
            text="",
            tool_calls=[{"id": "toolu_x", "name": "query_recent_alerts",
                         "input": {}}],
            stop_reason="tool_use",
        )
        with patch("engine.llm.call.call", return_value=looping):
            r = chat_turn(persona, "loop?")
        assert r.n_iterations == persona.max_iterations

    def test_history_threading(self, persona):
        prior = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        mock = self._mock_result(text="follow-up answer")
        with patch("engine.llm.call.call", return_value=mock):
            r = chat_turn(persona, "follow-up", history=prior)
        assert len(r.new_messages) == 2
        assert prior[0] not in r.new_messages

    def test_unknown_tool_returns_error_to_model(self, persona):
        iter1 = self._mock_result(
            text="",
            tool_calls=[{"id": "toolu_1", "name": "invent_a_tool", "input": {}}],
            stop_reason="tool_use",
        )
        iter2 = self._mock_result(text="Tool not found, recovered.")
        with patch("engine.llm.call.call", side_effect=[iter1, iter2]):
            r = chat_turn(persona, "use bogus tool")
        assert r.tool_calls_log[0]["name"] == "invent_a_tool"
        assert "error" in r.tool_calls_log[0]["result_preview"].lower() or \
               "unknown" in r.tool_calls_log[0]["result_preview"].lower()
        # is_error flag must be True so model knows to recover
        assert r.tool_calls_log[0]["is_error"] is True

    def test_is_error_flag_propagates_to_anthropic_block(self, persona):
        """After a failed tool call, the user-side message must contain
        a tool_result block with is_error=True (Anthropic protocol),
        so the model recovers rather than treating error as data."""
        iter1 = self._mock_result(
            text="",
            tool_calls=[{"id": "toolu_xx", "name": "bogus_tool", "input": {}}],
            stop_reason="tool_use",
        )
        iter2 = self._mock_result(text="Recovered.")
        with patch("engine.llm.call.call", side_effect=[iter1, iter2]):
            r = chat_turn(persona, "trigger error")
        # Find the user message containing tool_result
        user_msgs_with_tool_result = [
            m for m in r.new_messages
            if m["role"] == "user" and isinstance(m["content"], list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m["content"]
            )
        ]
        assert user_msgs_with_tool_result, "no tool_result message found"
        tool_result_blocks = [
            b for b in user_msgs_with_tool_result[0]["content"]
            if b.get("type") == "tool_result"
        ]
        assert tool_result_blocks[0].get("is_error") is True, (
            "tool_result block missing is_error=True for failed tool call"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Live ping (opt-in for BOTH personas)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    os.environ.get("ANTHROPIC_LIVE_TEST") != "1",
    reason="Set ANTHROPIC_LIVE_TEST=1 to run live API test",
)
@pytest.mark.parametrize("persona", _PERSONAS, ids=lambda p: p.agent_id)
class TestLiveAgent:
    def test_live_one_turn(self, persona):
        # Per-persona test prompt aligned with each agent's scope
        prompts = {
            "chief_of_staff":      "Quick status check: any HARD HALT findings today? Route appropriately.",
            "risk_manager":        "What is K1 BAB status today? One sentence.",
            "dq_inspector":        "Was FRED data fresh today? One sentence.",
            "devils_advocate":     "Critique this claim in one sentence: "
                                   "'Backtest Sharpe of 1.0 over 10 years means "
                                   "the strategy will work in the next 10 years.'",
            "anomaly_sentinel":    "What is SPY's current z-score? One sentence.",
            "attribution_analyst": "What was the NAV return last 7 days? One sentence.",
            "audit_recorder":      "How many open audit findings? One sentence.",
        }
        prompt = prompts[persona.agent_id]
        r = chat_turn(persona, prompt)
        assert r.final_text
        # Banned-phrase check
        from engine.agents.risk_manager.narrator import contains_banned_phrase
        bad = contains_banned_phrase(r.final_text)
        assert bad is None, f"{persona.agent_id}: banned phrase {bad!r}"
        # No emojis
        emojis = re.findall(r"[\U0001F300-\U0001FAFF]", r.final_text)
        assert emojis == [], f"{persona.agent_id}: emoji {emojis!r}"
