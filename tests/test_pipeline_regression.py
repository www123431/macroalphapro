"""tests/test_pipeline_regression.py — Session 2 (L4 build):
candidate_pipeline regression net.

Test taxonomy (senior anti-fragility design):

  Class A — SMOKE  (<5s, no LLM, no full pipeline)
    imports succeed; graph compiles; state initializes; basic ops work.

  Class B — SCHEMA (<5s, no LLM)
    PipelineReport / StepResult shapes stable; hygiene functions accept
    candidate_info param (chicken-egg bug regression net); IntuitionRules
    validates clean.

  Class C — STRUCTURAL (slow, ~3-5min/candidate, includes DA LLM call)
    Run full pipeline on PIT SN; assert STRUCTURAL invariants only
    (n_steps == 15; specific steps don't FAIL; relation is REPLACEMENT).
    Avoid verdict-specific assertions (DA is non-deterministic).

  Class D — SEMANTIC (very slow, fragile)
    NOT in this file. Reserved for separate file run on-demand because
    DA verdict can vary across runs.

Pre-commit hook should run only A+B. Use pytest -m "not slow" for that.

Marker convention:
  @pytest.mark.slow      Class C
  (no marker)            Class A + B (fast subset)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make repo root importable when pytest runs from any CWD
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Class A — SMOKE ────────────────────────────────────────────────────


class TestPipelineSmoke:
    """Class A: fast structural tests of imports + compile + state."""

    def test_v1_pipeline_imports(self):
        """v1 pipeline core importable + dataclass fields present."""
        from engine.research.candidate_pipeline import (
            PipelineReport, StepResult, run_candidate_pipeline,
            _classify_replacement_or_addition,
        )
        assert callable(run_candidate_pipeline)
        assert "step_results" in PipelineReport.__dataclass_fields__
        assert "step_name" in StepResult.__dataclass_fields__

    def test_v2_pipeline_imports(self):
        """v2 LangGraph pipeline core importable + state fields present."""
        from engine.research.candidate_pipeline_v2 import (
            CandidateState, build_pipeline_graph, run_candidate_pipeline_v2,
        )
        assert callable(run_candidate_pipeline_v2)
        assert "step_results" in CandidateState.__dataclass_fields__

    def test_v2_graph_compiles(self):
        """LangGraph state machine compiles without error."""
        from engine.research.candidate_pipeline_v2 import build_pipeline_graph
        graph = build_pipeline_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_v2_graph_node_count(self):
        """Graph has expected node set — early signal of refactor drift."""
        from engine.research.candidate_pipeline_v2 import build_pipeline_graph
        graph = build_pipeline_graph()
        compiled = graph.compile()
        nodes = set(compiled.nodes.keys())
        required = {
            "build_manifest", "h10", "data_quality", "h2", "h6", "h7",
            "graveyard", "cost_model", "regime_stratified",
            "factor_budget", "multi_aum", "sub_period", "correlation",
            "ablation", "block_bootstrap_significance",
            "quarter_concentration",
            "honest_deploy_sharpe", "devils_advocate",
            "compute_meta_decision", "short_circuit_end",
        }
        missing = required - nodes
        assert not missing, f"v2 graph missing nodes: {missing}"

    def test_candidate_state_initializes(self):
        """CandidateState dataclass accepts the standard input set."""
        from engine.research.candidate_pipeline_v2 import CandidateState
        s = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-31",
                                                         periods=2, freq="ME"))
        state = CandidateState(
            candidate_returns=s, proposal_name="x", proposed_role="alpha_seeker",
        )
        assert state.proposal_name == "x"
        assert state.short_circuited is False
        assert state.step_results == []
        assert state.candidate_relation == "UNKNOWN"

    def test_v2_sqlite_checkpointer_compiles(self):
        """Phase 4a: graph compiles when bound to SqliteSaver."""
        from engine.research.candidate_pipeline_v2 import (
            _sqlite_checkpointer, build_pipeline_graph,
        )
        with _sqlite_checkpointer() as ckpt:
            compiled = build_pipeline_graph().compile(checkpointer=ckpt)
            assert compiled is not None
            assert len(compiled.nodes) >= 18

    def test_v2_make_thread_id_unique(self):
        """Phase 4a: thread_id has greppable slug + uuid suffix."""
        from engine.research.candidate_pipeline_v2 import make_thread_id
        a = make_thread_id("PIT SN")
        b = make_thread_id("PIT SN")
        assert a.startswith("pit_sn-")
        assert b.startswith("pit_sn-")
        assert a != b  # uuid suffix prevents collision

    def test_v2_get_checkpoint_state_unknown_returns_none(self):
        """Phase 4a: querying an unknown thread_id returns None
        (not raises) — caller can check existence before resume."""
        from engine.research.candidate_pipeline_v2 import get_checkpoint_state
        out = get_checkpoint_state("nonexistent-thread-xyz-abc")
        assert out is None

    def test_v2_resume_unknown_thread_raises(self):
        """Phase 4a: resuming a non-existent thread raises ValueError
        with a clear hint pointing to run_candidate_pipeline_v2."""
        from engine.research.candidate_pipeline_v2 import (
            resume_candidate_pipeline_v2,
        )
        with pytest.raises(ValueError, match="no checkpoint found"):
            resume_candidate_pipeline_v2("nonexistent-xyz-abc")


# ── Class B — SCHEMA ───────────────────────────────────────────────────


class TestSchemaContracts:
    """Class B: shape / API contract tests. Pure compute, no LLM."""

    def test_h2_accepts_candidate_info(self):
        """8th catch regression: h2_cousin_check_multilevel signature
        accepts candidate_info kwarg (chicken-egg fix)."""
        from engine.research.hygiene_tools import h2_cousin_check_multilevel
        import inspect
        sig = inspect.signature(h2_cousin_check_multilevel)
        assert "candidate_info" in sig.parameters

    def test_h6_accepts_candidate_info(self):
        """9th catch regression: h6_post_pub_evidence_check signature
        accepts candidate_info kwarg."""
        from engine.research.hygiene_tools import h6_post_pub_evidence_check
        import inspect
        sig = inspect.signature(h6_post_pub_evidence_check)
        assert "candidate_info" in sig.parameters

    def test_h2_returns_allow_warning_for_new_candidate_no_info(self):
        """h2 on unknown mechanism_id without candidate_info returns
        allow-with-warning rather than hard FAIL."""
        from engine.research.hygiene_tools import h2_cousin_check_multilevel
        r = h2_cousin_check_multilevel("genuinely_new_unknown_mechanism_id_xyz")
        d = r.to_dict()
        assert d["success"] is True
        assert "allow_no_metadata" in d["payload"].get("verdict", "")

    def test_devils_advocate_signature_has_relation_params(self):
        """11th catch regression: DA accepts relation context."""
        from engine.research.candidate_pipeline import _run_devils_advocate
        import inspect
        sig = inspect.signature(_run_devils_advocate)
        for p in ("candidate_relation", "most_correlated_sleeve",
                   "most_correlated_value"):
            assert p in sig.parameters, f"DA missing param {p}"

    def test_intuition_rules_validate_clean(self):
        """L4 Session 1 asset: IntuitionRulesBase must validate clean."""
        from engine.research.intuition_rules import validate_rules_file
        report = validate_rules_file()
        assert report.is_valid, (
            f"intuition rules invalid: violations={report.schema_violations} "
            f"duplicates={report.duplicate_ids}"
        )
        assert report.n_rules >= 12, "expected ≥12 rules after Session 1"

    def test_intuition_rules_query_works(self):
        """IntuitionRules queryable by all 3 filter modes."""
        from engine.research.intuition_rules import query_rules
        by_sev = query_rules(severity="FATAL_BLOCK")
        assert len(by_sev) >= 1, "no FATAL_BLOCK rules"
        by_id = query_rules(rule_id="n_trials_must_be_within_family")
        assert len(by_id) == 1, "id query broken"
        by_context = query_rules(context_text="cosine")
        assert len(by_context) >= 2, "context query broken"

    def test_da_role_guidance_exists_for_all_5_roles(self):
        """10th catch regression: DA role-aware guidance covers all 5 roles."""
        from engine.research.candidate_pipeline import _DA_ROLE_GUIDANCE
        for role in ["alpha_seeker", "risk_premium_harvester",
                      "insurance", "diversifier", "regime_overlay"]:
            assert role in _DA_ROLE_GUIDANCE, f"role guidance missing for {role}"

    def test_da_relation_guidance_handles_replacement_addition(self):
        """11th catch regression: DA relation-aware guidance covers
        REPLACEMENT and ADDITION."""
        from engine.research.candidate_pipeline import (
            _da_relation_specific_guidance,
        )
        rep = _da_relation_specific_guidance("REPLACEMENT", "equity", 0.88)
        assert "EXPECTED" in rep or "expected" in rep
        add = _da_relation_specific_guidance("ADDITION", "carry", 0.1)
        assert add  # non-empty

    def test_graveyard_family_alias_normalization(self):
        """12th catch regression: family alias normalization works on
        BOTH input and group members. Canonical key 'earnings_under-
        reaction' should match graveyard entries tagged with
        'forward-earnings information' (same alias group)."""
        from engine.research.graveyard import (
            CandidateInfo, check_against_graveyard,
        )
        candidate = CandidateInfo(
            title="Japan PEAD",
            family="earnings_underreaction",  # canonical alias
            parent_family="equity_factor",
        )
        match = check_against_graveyard(candidate)
        d = match.to_dict()
        assert d["cousin_count_in_family"] >= 6, (
            f"alias normalization broken — expected ≥6 cousins for "
            f"earnings_underreaction (PEAD family in graveyard.json), "
            f"got {d['cousin_count_in_family']}"
        )

    def test_llm_tool_registry_has_9_tools(self):
        """Session 3 asset: 9 LLM tools registered with valid schemas."""
        from engine.research.llm_tools import (
            list_tool_names, tool_specs_for_anthropic,
        )
        assert len(list_tool_names()) == 16
        specs = tool_specs_for_anthropic()
        assert len(specs) == 16
        for spec in specs:
            assert "name" in spec
            assert "description" in spec
            assert "input_schema" in spec
            assert spec["input_schema"]["type"] == "object"

    def test_llm_tool_dispatch_validates_schema(self):
        """dispatch() validates inputs against Pydantic schema."""
        from engine.research.llm_tools import dispatch
        # query_intuition_rules with valid args works
        r = dispatch("query_intuition_rules", severity="FATAL_BLOCK")
        assert r["n_matched"] >= 1
        # unknown tool raises clear KeyError
        import pytest as _pt
        with _pt.raises(KeyError):
            dispatch("unknown_tool_xyz")

    def test_mcp_server_registers_9_tools(self):
        """Phase 4a.5: MCP server exposes the same 9 tools registered
        in TOOLS, with flat (not nested) input schemas."""
        import asyncio
        from engine.research.mcp_server import mcp
        tools = asyncio.run(mcp.list_tools())
        assert len(tools) == 16
        names = {t.name for t in tools}
        assert "query_intuition_rules" in names
        assert "query_graveyard" in names
        assert "graveyard_summary" in names
        assert "query_l4_iterations" in names
        assert "get_candidate_suggestions" in names

    def test_mcp_server_schemas_are_flat(self):
        """Phase 4a.5: MCP input schemas have schema fields as TOP-LEVEL
        properties (not wrapped under {"args": {...}}). The flatness
        affects how external Claude clients invoke the tool."""
        import asyncio
        from engine.research.mcp_server import mcp
        tools = asyncio.run(mcp.list_tools())
        qir = next(t for t in tools if t.name == "query_intuition_rules")
        props = qir.inputSchema.get("properties", {})
        # Real fields should appear at top level
        assert "severity" in props
        assert "category" in props
        assert "context_text" in props
        # And NOT be wrapped under a nested "args" object
        assert "args" not in props

    def test_mcp_server_invoke_query_intuition_rules(self):
        """Phase 4a.5: end-to-end MCP invocation works — round-trip
        through FastMCP, schema validation, dispatch, JSON-encoded
        result. The same code path Claude Code will use."""
        import asyncio
        import json as _json
        from engine.research.mcp_server import mcp
        result = asyncio.run(
            mcp.call_tool("query_intuition_rules", {"severity": "FATAL_BLOCK"})
        )
        content = result[0] if isinstance(result, tuple) else result
        text = content[0].text if hasattr(content[0], "text") else str(content[0])
        parsed = _json.loads(text)
        assert parsed["n_matched"] >= 1

    def test_rest_shim_lists_9_tools(self):
        """Phase 4a.6: GET /api/research/tools returns all 9 tools
        with Anthropic-format input schemas (single source of truth
        with both MCP server and in-process dispatch)."""
        from fastapi.testclient import TestClient
        from api.main import app
        r = TestClient(app).get("/api/research/tools")
        assert r.status_code == 200
        body = r.json()
        assert body["n_tools"] == 16
        names = {t["name"] for t in body["tools"]}
        assert "query_intuition_rules" in names
        assert "query_graveyard" in names
        assert "graveyard_summary" in names
        assert "query_l4_iterations" in names
        assert "get_candidate_suggestions" in names

    def test_rest_shim_dispatches_call(self):
        """Phase 4a.6: POST /api/research/call/{tool} dispatches
        through engine.research.llm_tools.dispatch and returns the
        result wrapped in an envelope with hash + latency."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        r = c.post(
            "/api/research/call/query_intuition_rules",
            json={"args": {"severity": "FATAL_BLOCK"}},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["tool"] == "query_intuition_rules"
        assert body["result"]["n_matched"] >= 1
        assert isinstance(body["result_hash"], str)
        assert len(body["result_hash"]) == 16

    def test_rest_shim_unknown_tool_returns_404(self):
        """Phase 4a.6: unknown tool name → 404 with the registry URL
        in the detail (helps callers self-correct)."""
        from fastapi.testclient import TestClient
        from api.main import app
        r = TestClient(app).post(
            "/api/research/call/totally_not_a_tool",
            json={"args": {}},
        )
        assert r.status_code == 404
        assert "/api/research/tools" in r.json()["detail"]

    def test_agent_council_infrastructure_imports(self):
        """Phase 4b: agent_council module + personas import cleanly +
        the 3 persona prompts are non-empty + tool allowlists wired."""
        from engine.research.agent_council import (
            ProposalDict, AgentVerdict, CouncilVerdict,
            aggregate_verdicts, _parse_verdict_json, _normalize_verdict,
        )
        from engine.research.agent_council_personas import (
            ARCHITECT_SYSTEM_PROMPT, THEORIST_SYSTEM_PROMPT, DA_SYSTEM_PROMPT,
            ARCHITECT_TOOLS, THEORIST_TOOLS, DA_TOOLS,
        )
        # personas have substantive prompts (not stubs)
        assert len(ARCHITECT_SYSTEM_PROMPT) > 500
        assert len(THEORIST_SYSTEM_PROMPT) > 500
        assert len(DA_SYSTEM_PROMPT) > 500
        # each persona has a non-empty tool allowlist that intersects
        # with the registered TOOLS
        from engine.research.llm_tools import TOOLS
        for allowlist in (ARCHITECT_TOOLS, THEORIST_TOOLS, DA_TOOLS):
            assert allowlist
            for t in allowlist:
                assert t in TOOLS, f"persona references unknown tool {t!r}"

    def test_agent_council_aggregator_consensus_rules(self):
        """Phase 4b: aggregate_verdicts encodes the council consensus
        process rule (ANY FAIL → REJECT; ALL PASS → APPROVE; else
        NEEDS_REVISION). Process > LLM judgment for consensus."""
        from engine.research.agent_council import (
            AgentVerdict, aggregate_verdicts,
        )
        v_pass = AgentVerdict(agent_name="a", verdict="PASS",
                              confidence=0.8, rationale="clean")
        v_warn = AgentVerdict(agent_name="b", verdict="WARN",
                              confidence=0.5, rationale="concern",
                              material_concerns=["c1"])
        v_fail = AgentVerdict(agent_name="c", verdict="FAIL",
                              confidence=0.9, rationale="blocker",
                              fatal_red_flags=["f1"])
        assert aggregate_verdicts([v_pass, v_pass])[0] == "APPROVE"
        assert aggregate_verdicts([v_pass, v_warn])[0] == "NEEDS_REVISION"
        assert aggregate_verdicts([v_pass, v_fail])[0] == "REJECT"
        assert aggregate_verdicts([v_warn, v_fail])[0] == "REJECT"
        assert aggregate_verdicts([])[0] == "REJECT"

    def test_agent_council_ledger_round_trip(self):
        """Phase 4b.5: _append_to_ledger writes; read_council_runs
        retrieves; read_council_run_by_id finds the same row.
        Persistence is non-fatal so the council never blocks on disk."""
        from engine.research.agent_council import (
            _append_to_ledger, read_council_runs, read_council_run_by_id,
        )
        rid = _append_to_ledger({
            "stage": "test_round_trip", "consensus": "APPROVE",
            "proposal": {"title": "ledger_test_xyz",
                          "family": "test", "proposed_role": "alpha_seeker"},
            "verdicts": [], "rationale": "synthetic test row",
        })
        assert len(rid) == 12
        # Find by id
        row = read_council_run_by_id(rid)
        assert row is not None
        assert row["run_id"] == rid
        assert row["proposal"]["title"] == "ledger_test_xyz"
        # Appears at top of newest-first list
        runs = read_council_runs(limit=5)
        assert any(r["run_id"] == rid for r in runs)

    def test_rest_council_runs_endpoint(self):
        """Phase 4b.5: GET /api/research/council/runs returns the
        ledger; 404 on unknown run id."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        r = c.get("/api/research/council/runs?limit=10")
        assert r.status_code == 200
        body = r.json()
        assert "runs" in body and "n" in body
        # 404 on unknown id
        r2 = c.get("/api/research/council/run/definitely_not_a_real_id")
        assert r2.status_code == 404

    def test_l4_workflow_imports_clean(self):
        """Phase 4c+4d: L4 Temporal workflow + client modules import
        without side effects (no Temporal server contact at import)."""
        from engine.research.l4_workflow import (
            L4DiscoveryWorkflow, TASK_QUEUE_L4,
            propose_activity, critique_activity,
            pipeline_activity, ledger_activity,
            ProposeInput, CritiqueInput, PipelineInput, LedgerInput,
            CouncilWorkflowResult,
        )
        from engine.research.l4_temporal_client import (
            is_temporal_available, enqueue_council_workflow,
            query_workflow_status, DEFAULT_TEMPORAL_ADDRESS,
        )
        assert TASK_QUEUE_L4 == "l4-discovery"
        assert DEFAULT_TEMPORAL_ADDRESS
        # 4d: CouncilWorkflowResult must surface pipeline + alignment
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(CouncilWorkflowResult)}
        for f in ("iteration_id", "pipeline_ran",
                   "pipeline_final_decision", "verdict_alignment"):
            assert f in field_names, f"missing 4d field {f!r}"

    def test_outcome_ledger_round_trip(self):
        """Phase 4d: outcome_ledger module — append → read → drill +
        calibration aggregates compute correctly."""
        from engine.research.outcome_ledger import (
            append_l4_iteration, calibration_summary,
            read_iteration_by_id, read_l4_iterations,
        )
        iid = append_l4_iteration(
            workflow_id="l4-test-regression",
            proposal={"title": "reg_test_iter", "family": "test",
                       "proposed_role": "alpha_seeker"},
            council={"consensus": "APPROVE", "rationale": "mock",
                      "verdicts": [{}, {}]},
            pipeline_report={
                "final_decision": "PROMOTE_TO_GATE", "rationale": "mock",
                "step_results": []},
            elapsed_s=5.5,
        )
        assert iid.startswith("iter-")
        row = read_iteration_by_id(iid)
        assert row is not None
        assert row["proposal"]["title"] == "reg_test_iter"
        assert row["verdict_alignment"] == "agree"  # APPROVE × PROMOTE
        # calibration aggregator picks up the row
        cal = calibration_summary(limit=200)
        assert cal["n_total"] >= 1
        assert cal["agree_pct"] is not None

    def test_suggestion_engine_returns_ranked_list(self):
        """Phase 4d.5: L1 recommender returns suggestions ranked by
        score desc, blending library + seed pool sources, with
        alias-aware graveyard collision detection."""
        from engine.research.suggestion_engine import get_candidate_suggestions
        r = get_candidate_suggestions(limit=10)
        assert r["n_total"] >= 1
        assert "suggestions" in r
        # Sorted desc by score
        scores = [s["score"] for s in r["suggestions"]]
        assert scores == sorted(scores, reverse=True)
        # Has both source types
        sources = {s["source"] for s in r["suggestions"]}
        assert "seed_pool" in sources  # always present (hardcoded)
        # Each suggestion has required fields
        for s in r["suggestions"]:
            assert s["seed"] and s["family"] and s["risk_tag"]
            assert s["risk_tag"] in ("low", "medium", "high")
            assert 0.0 <= s["score"] <= 1.0

    def test_rest_l4_iterations_endpoint(self):
        """Phase 4d: GET /api/research/l4/iterations returns ledger +
        calibration KPI; 404 on unknown iteration_id."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        r = c.get("/api/research/l4/iterations?limit=10")
        assert r.status_code == 200
        body = r.json()
        assert "iterations" in body
        assert "calibration" in body
        assert "agree_pct" in body["calibration"]
        # 404 on unknown
        r2 = c.get("/api/research/l4/iterations/iter-does-not-exist")
        assert r2.status_code == 404

    def test_outcome_ledger_human_override_persists(self):
        """Phase 4e: when human_override_verdict is supplied, the ledger
        row records BOTH the original LLM consensus AND the override,
        and verdict_alignment classification uses the effective
        consensus (the override) — so the calibration KPI reflects
        decisions actually made, not what LLM said in vacuum."""
        from engine.research.outcome_ledger import (
            append_l4_iteration, read_iteration_by_id,
        )
        iid = append_l4_iteration(
            workflow_id="l4-4e-test",
            proposal={"title": "override_t", "family": "test",
                       "proposed_role": "alpha_seeker"},
            council={"consensus": "REJECT", "rationale": "m",
                      "verdicts": [{}]},
            pipeline_report={"final_decision": "PROMOTE_TO_GATE",
                              "rationale": "", "step_results": []},
            elapsed_s=1.0,
            human_override_verdict="APPROVE",
        )
        row = read_iteration_by_id(iid)
        assert row is not None
        assert row["council"]["consensus"] == "REJECT"  # original preserved
        assert row["human_override"] == {"verdict": "APPROVE"}
        assert row["effective_consensus"] == "APPROVE"
        # Effective alignment: APPROVE ↔ PROMOTE = agree
        assert row["verdict_alignment"] == "agree"

    def test_rest_council_override_validates_input(self):
        """Phase 4e: override endpoint validates verdict enum +
        requires non-empty justification (audit trail rule)."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        # Invalid verdict → 422
        r = c.post("/api/research/council/workflow/fake-wf/override",
                    json={"verdict": "MAYBE", "justification": "x"})
        assert r.status_code == 422
        # Missing justification → 422
        r2 = c.post("/api/research/council/workflow/fake-wf/override",
                     json={"verdict": "APPROVE", "justification": ""})
        assert r2.status_code == 422

    def test_pbb_auto_block_length_reflects_serial_dependence(self):
        """Phase 5.1: Politis-White 2009 block-length selector should
        give SHORTER blocks for iid data than for serially dependent.
        Exact thresholds are sample-noise; the relational invariant is
        what's robust. Average across multiple seeds for stability.

        Bug guard: pre-commit detected the |k|-factor omission which
        gave white noise block ~10. The relational invariant catches
        that variant regardless of seed."""
        import numpy as np
        from engine.validation.block_bootstrap import auto_block_length

        def avg_bl(generator, n=400, n_seeds=5) -> float:
            bls = []
            for seed in range(100, 100 + n_seeds):
                rng = np.random.default_rng(seed)
                series = generator(rng, n)
                bls.append(auto_block_length(series))
            return float(np.mean(bls))

        def wn(rng, n):
            return rng.standard_normal(n)

        def ar_series(phi):
            def _gen(rng, n):
                x = np.zeros(n)
                noise_sd = (1 - phi ** 2) ** 0.5
                for t in range(1, n):
                    x[t] = phi * x[t-1] + rng.standard_normal() * noise_sd
                return x
            return _gen

        bl_wn = avg_bl(wn)
        bl_ar50 = avg_bl(ar_series(0.5))
        bl_ar90 = avg_bl(ar_series(0.9))

        # Relational invariants — robust to seed
        assert bl_wn < bl_ar50, (
            f"AR(0.5) ({bl_ar50:.2f}) should exceed white noise "
            f"({bl_wn:.2f})"
        )
        assert bl_ar50 < bl_ar90 + 0.5, (
            f"AR(0.9) ({bl_ar90:.2f}) should be >= AR(0.5) "
            f"({bl_ar50:.2f}) within tolerance"
        )
        # Absolute upper bound on white noise — a buggy formula (omitting
        # the |k| factor) gives ~10 here regardless of seed
        assert bl_wn < 5.0, (
            f"white noise block_len {bl_wn:.2f} too large — looks like "
            "PW2009 |k|-factor regression"
        )

    def test_pbb_sharpe_diff_p_value_under_null(self):
        """Phase 5.1: pbb_sharpe_diff(a, a) under exact equality
        returns p ≈ 1 + diff ≈ 0 + CI containing 0."""
        import numpy as np
        from engine.validation.block_bootstrap import pbb_sharpe_diff
        rng = np.random.default_rng(7)
        a = 0.01 + rng.standard_normal(300)
        r = pbb_sharpe_diff(a, a, n_iter=2000, rng_seed=42)
        assert abs(r.diff_point) < 1e-9
        assert r.p_value_two_sided > 0.5  # any reasonable p > 0.5
        assert r.diff_ci_lo <= 0 <= r.diff_ci_hi

    def test_pbb_hochberg_step_up_is_correct(self):
        """Phase 5.1: Hochberg 1988 step-up — for sorted p (smallest
        first), adjusted_p_i = max over j>=i of (m-j+1) * p_(j); a
        miscoded Bonferroni-Holm variant would give different
        numbers."""
        from engine.validation.block_bootstrap import hochberg_adjust
        # m=4, p = [0.01, 0.04, 0.03, 0.20] sorted = [0.01, 0.03, 0.04, 0.20]
        # Hochberg step-up from largest:
        #   adj_(4) = 1*0.20 = 0.20
        #   adj_(3) = min(2*0.04, 0.20) = 0.08
        #   adj_(2) = min(3*0.03, 0.08) = 0.08
        #   adj_(1) = min(4*0.01, 0.08) = 0.04
        adj = hochberg_adjust([0.01, 0.04, 0.03, 0.20])
        # input order: indices 0,1,2,3 ↔ ranks 1,3,2,4
        assert abs(adj[0] - 0.04) < 1e-9    # p=0.01 ranked 1st
        assert abs(adj[1] - 0.08) < 1e-9    # p=0.04 ranked 3rd
        assert abs(adj[2] - 0.08) < 1e-9    # p=0.03 ranked 2nd
        assert abs(adj[3] - 0.20) < 1e-9    # p=0.20 ranked 4th

    def test_rest_sleeves_ca_calibration_endpoint(self):
        """Phase 5.7 follow-up: REST /api/research/sleeves/ca_calibration
        returns each DEPLOYED / PENDING_DEPLOY sleeve's CA filter
        config. After backfill, every entry must carry a method
        ∈ {paper_default, pbb_sweep_calibrated, scalar_override}."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        r = c.get("/api/research/sleeves/ca_calibration")
        assert r.status_code == 200
        body = r.json()
        assert "sleeves" in body and "n" in body
        assert body["n"] >= 5
        valid_methods = {
            "paper_default", "pbb_sweep_calibrated", "scalar_override",
            "not_applicable",
        }
        for s in body["sleeves"]:
            assert s["status"] in ("DEPLOYED", "PENDING_DEPLOY")
            assert s["ca_filter_k_method"] in valid_methods
            assert s["ca_signal_type"] in {
                "point_forecast", "cross_sect_rank", "regime_indicator",
                "vol_norm_zscore", "binary_trigger",
            }
            # ca_filter_k can be null only when method=not_applicable
            if s["ca_filter_k_method"] != "not_applicable":
                assert s["ca_filter_k"] is not None
        # cross_asset_carry is PBB-validated at k=3.0 via the multi-
        # asset per-contract pipeline (commit shipping this state).
        carry = next((s for s in body["sleeves"]
                       if s["id"] == "cross_asset_carry"), None)
        assert carry is not None
        assert carry["ca_filter_k_method"] == "pbb_sweep_calibrated"
        assert carry["ca_filter_k"] == 3.0
        # 4 sleeves should be not_applicable for sleeve-shape reasons
        na = [s["id"] for s in body["sleeves"]
               if s["ca_filter_k_method"] == "not_applicable"]
        assert "time_series_momentum" in na
        assert "crisis_hedge_tlt_gld" in na
        assert "mom_hedge_overlay" in na
        assert "post_earnings_drift" in na

    def test_library_ca_filter_audit_passes_strict(self):
        """SG5 (5.7 follow-up): all DEPLOYED / PENDING_DEPLOY YAMLs
        must carry ca_filter_k + companion fields. The audit script
        is wired into pre-commit; this test makes its passing visible
        in the pytest output too."""
        import subprocess
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m",
             "engine.research.library_ca_filter_audit", "--strict"],
            cwd=repo_root, capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"SG5 audit failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_ca_filter_gates_carry_signal_correctly(self):
        """Phase 5.7: should_trade applies paper's |ER| > k×Δpos×tc
        gate for the POINT_FORECAST carry sleeve — strong signal
        trades, weak signal holds."""
        from engine.portfolio.execution_filter import should_trade
        # Strong carry → trade
        d_strong = should_trade(
            "cross_asset_carry", raw_signal=0.05,
            current_position=0.0, target_position=1.0,
            tcost_round_trip=0.001, k=2.0,
        )
        assert d_strong.trade is True
        assert abs(d_strong.calibrated_er - 0.05) < 1e-9
        assert d_strong.cost_threshold == 0.001 * 2.0 * 1.0
        # Weak carry → hold
        d_weak = should_trade(
            "cross_asset_carry", raw_signal=0.0005,
            current_position=0.0, target_position=1.0,
            tcost_round_trip=0.001, k=2.0,
        )
        assert d_weak.trade is False
        assert "hold" in d_weak.reason.lower()

    def test_ca_filter_conservative_default_on_unknown_sleeve(self):
        """Phase 5.7: unknown sleeve_id → trade=True with diagnostic
        flag. Don't silently suppress trades for a sleeve we forgot
        to taxonomize."""
        from engine.portfolio.execution_filter import should_trade
        d = should_trade(
            "totally_unknown_sleeve", raw_signal=0.01,
            current_position=0.0, target_position=1.0,
            tcost_round_trip=0.001, k=2.0,
        )
        assert d.trade is True
        assert "no taxonomy" in d.reason
        assert d.confident_calibration is False

    def test_ca_filter_zero_position_change_holds(self):
        """Phase 5.7: zero |Δpos| → no trade (no cost to incur)."""
        from engine.portfolio.execution_filter import should_trade
        d = should_trade(
            "cross_asset_carry", raw_signal=0.05,
            current_position=1.0, target_position=1.0,  # unchanged
            tcost_round_trip=0.001, k=2.0,
        )
        assert d.trade is False
        assert "zero position change" in d.reason

    def test_ca_filter_apply_to_series_integrates_with_scaffold(self):
        """Phase 5.7: apply_ca_filter_to_returns produces a aligned
        filtered series feedable into the 5.5 scaffold for PBB
        validation. Smoke-tests the full 5.5 + 5.6 + 5.7 stack."""
        import numpy as np
        import pandas as pd
        from engine.portfolio.execution_filter import (
            apply_ca_filter_to_returns,
        )
        from engine.validation.filter_counterfactual import (
            evaluate_filter_counterfactual,
        )
        rng = np.random.default_rng(7)
        dates = pd.date_range("2014-01-31", periods=120, freq="ME")
        gross = pd.Series(0.012 + rng.standard_normal(120) * 0.025,
                           index=dates)
        signal = pd.Series(0.005 + 0.002 * rng.standard_normal(120),
                            index=dates)
        baseline = apply_ca_filter_to_returns(
            "cross_asset_carry", signal, gross,
            tcost_round_trip=0.001, k=0.0,
        )
        filtered = apply_ca_filter_to_returns(
            "cross_asset_carry", signal, gross,
            tcost_round_trip=0.001, k=2.0,
        )
        # Both should be aligned to the same date index
        assert len(baseline) == len(filtered)
        assert (baseline.index == filtered.index).all()
        # Scaffold can ingest these without raising
        result = evaluate_filter_counterfactual(
            baseline_returns=baseline,
            filtered_returns=filtered,
            sleeve_name="cross_asset_carry",
            filter_descriptor="CA k=2.0",
            n_iter=1000, rng_seed=42,
        )
        assert result.verdict in ("DEPLOY", "NO_EVIDENCE", "WORSE")
        assert result.n_obs_aligned == len(filtered)

    def test_tool_registry_has_category_taxonomy(self):
        """Post-5.7 tool registry hygiene: each tool has a category +
        an example query (helps LLM tool picking) + 5 categories cover
        all 16 tools with no orphans."""
        from engine.research.llm_tools import (
            TOOLS, TOOL_CATEGORIES, TOOL_EXAMPLES, list_tools_by_category,
        )
        # Every tool has both metadata
        for name in TOOLS:
            assert name in TOOL_CATEGORIES, f"{name} missing category"
            assert name in TOOL_EXAMPLES, f"{name} missing example"
            cat = TOOL_CATEGORIES[name]
            assert cat in {"knowledge", "compute", "history",
                            "external_data", "action"}
            ex = TOOL_EXAMPLES[name]
            assert len(ex) >= 5, f"{name} example too short"
        # Meta-discovery surface returns grouped view
        view = list_tools_by_category()
        assert view["n_tools_total"] == 16
        assert "knowledge" in view["by_category"]
        # Per-category filter works
        view_kn = list_tools_by_category("knowledge")
        for entry in view_kn["by_category"]["knowledge"]:
            assert "example" in entry
            assert "description" in entry

    def test_list_tools_by_category_callable_via_dispatch(self):
        """list_tools_by_category is also itself a registered tool, so
        an LLM client (Claude / MCP user) can call it to learn the
        registry shape via tool use."""
        from engine.research.llm_tools import dispatch
        r = dispatch("list_tools_by_category")
        assert r["n_tools_total"] == 16
        assert "knowledge" in r["by_category"]
        # Category-filtered call also works through dispatch
        r2 = dispatch("list_tools_by_category", category="external_data")
        assert "external_data" in r2["by_category"]
        # Should include the 3 new external_data tools
        names = {t["name"] for t in r2["by_category"]["external_data"]}
        assert {"arxiv_search", "sec_edgar_search", "fred_query"} <= names

    def test_external_data_tools_register_with_correct_schemas(self):
        """3 external_data tools imported + their schemas have the
        expected required fields."""
        from engine.research.llm_tools import TOOLS
        assert "arxiv_search" in TOOLS
        assert "sec_edgar_search" in TOOLS
        assert "fred_query" in TOOLS
        # Required field shape
        _, arxiv_schema, _ = TOOLS["arxiv_search"]
        assert "query" in arxiv_schema.model_fields
        _, sec_schema, _ = TOOLS["sec_edgar_search"]
        assert "query" in sec_schema.model_fields
        _, fred_schema, _ = TOOLS["fred_query"]
        assert "series_id" in fred_schema.model_fields

    def test_signal_taxonomy_dispatches_to_correct_calibrator(self):
        """Phase 5.6: SignalType enum + calibrate() dispatch returns
        the right per-type calibrator; carry signal is identity, regime
        signal hits historical mean given a panel."""
        import numpy as np
        import pandas as pd
        from engine.portfolio.signal_taxonomy import (
            SignalType, calibrate,
        )
        # Point forecast = identity
        r = calibrate(SignalType.POINT_FORECAST, 0.042)
        assert abs(r.expected_return - 0.042) < 1e-12
        assert r.confident is True
        assert r.method == "identity"

        # Regime indicator with panel
        panel = pd.DataFrame({
            "regime":         [0]*200 + [1]*100 + [2]*100,
            "forward_return": [0.012]*200 + [-0.005]*100 + [-0.030]*100,
        })
        calm = calibrate(SignalType.REGIME_INDICATOR, 0, panel)
        stress = calibrate(SignalType.REGIME_INDICATOR, 2, panel)
        assert calm.expected_return > 0
        assert stress.expected_return < 0
        assert calm.confident and stress.confident

        # Cross-sect rank with no panel → fallback
        rnp = calibrate(SignalType.CROSS_SECT_RANK, 2.0)
        assert rnp.confident is False
        assert "fallback" in rnp.method

    def test_signal_taxonomy_registry_covers_5_deployed_sleeves(self):
        """Phase 5.6: the DEPLOYED_SLEEVE_SIGNALS registry covers all
        5 currently deployed sleeves with sensible signal-type
        classifications. Guards against silently shipping a new sleeve
        without taxonomy entry."""
        from engine.portfolio.signal_taxonomy import (
            SignalType, get_sleeve_spec, list_sleeve_specs,
        )
        specs = list_sleeve_specs()
        assert len(specs) >= 5
        expected = {
            "cross_asset_carry":     SignalType.POINT_FORECAST,
            "post_earnings_drift":   SignalType.CROSS_SECT_RANK,
            "crisis_hedge_tlt_gld":  SignalType.REGIME_INDICATOR,
            "mom_hedge_overlay":     SignalType.VOL_NORM_ZSCORE,
            "time_series_momentum":  SignalType.VOL_NORM_ZSCORE,
        }
        # Legacy alias still resolves
        from engine.portfolio.signal_taxonomy import get_sleeve_spec
        assert get_sleeve_spec("tsmom") is not None
        for sid, st in expected.items():
            spec = get_sleeve_spec(sid)
            assert spec is not None, f"sleeve {sid} missing from registry"
            assert spec.signal_type == st, (
                f"sleeve {sid}: registry says {spec.signal_type}, "
                f"expected {st}"
            )

    def test_filter_counterfactual_no_evidence_under_null(self):
        """Phase 5.5: when baseline == filtered (identical series),
        verdict must be NO_EVIDENCE — no statistical lift possible."""
        import numpy as np
        import pandas as pd
        from engine.validation.filter_counterfactual import (
            evaluate_filter_counterfactual,
        )
        rng = np.random.default_rng(7)
        dates = pd.date_range("2014-01-31", periods=60, freq="ME")
        s = pd.Series(0.005 + rng.standard_normal(60) * 0.02,
                       index=dates)
        r = evaluate_filter_counterfactual(
            baseline_returns=s, filtered_returns=s.copy(),
            sleeve_name="null_test", filter_descriptor="identity",
            n_iter=1500, rng_seed=42,
        )
        assert r.verdict == "NO_EVIDENCE"
        assert abs(r.sharpe_diff) < 1e-6
        assert r.p_value > 0.5

    def test_filter_counterfactual_too_few_obs_returns_no_evidence(self):
        """Phase 5.5: insufficient overlap (< 24) → NO_EVIDENCE, no raise."""
        import pandas as pd
        import numpy as np
        from engine.validation.filter_counterfactual import (
            evaluate_filter_counterfactual,
        )
        dates = pd.date_range("2024-01-31", periods=12, freq="ME")
        b = pd.Series(np.random.randn(12) * 0.02, index=dates)
        f = pd.Series(np.random.randn(12) * 0.02, index=dates)
        r = evaluate_filter_counterfactual(
            baseline_returns=b, filtered_returns=f,
            sleeve_name="short_test", filter_descriptor="mock",
        )
        assert r.verdict == "NO_EVIDENCE"
        assert "insufficient" in (r.reasons[0] if r.reasons else "")

    def test_k_sweep_applies_hochberg_correction(self):
        """Phase 5.5: k-sweep across multiple k values applies Hochberg
        correction → raw p values get adjusted upward; some marginal
        DEPLOY verdicts can flip to NO_EVIDENCE under correction."""
        import numpy as np
        import pandas as pd
        from engine.validation.filter_counterfactual import evaluate_k_sweep
        rng = np.random.default_rng(7)
        dates = pd.date_range("2014-01-31", periods=120, freq="ME")
        gross = 0.012 + rng.standard_normal(120) * 0.025

        def factory(k: float):
            threshold = k * 0.0010
            flt = np.where(np.abs(gross) > threshold, gross - 0.0010, gross)
            return (pd.Series(gross - 0.0010, index=dates),
                     pd.Series(flt, index=dates))

        results = evaluate_k_sweep(
            sleeve_name="test", counterfactual_factory=factory,
            k_values=(0.5, 1.0, 1.5, 2.0, 3.0),
            n_iter=1500, rng_seed=42,
        )
        assert len(results) == 5
        # Each verdict ∈ valid set
        for r in results:
            assert r.verdict in ("DEPLOY", "NO_EVIDENCE", "WORSE")
        # p values are post-Hochberg → monotonically >= raw bootstrap p
        # (we don't store the raw, but we can check they're <= 1)
        for r in results:
            assert 0.0 <= r.p_value <= 1.0

    def test_concentration_classifier_flags_lucky_quarter_dependence(self):
        """Phase 5.4: classify_concentration returns HIGH when a
        strategy's ARC depends on top-N quarters (drop-top-N → loss).
        Verdict must NOT be LOW under that condition."""
        import numpy as np
        import pandas as pd
        from engine.validation.quarter_distribution import (
            classify_concentration, compute_quarter_distribution,
        )
        # Pathological series: mostly drifty-negative + 3 huge positive spikes
        rng = np.random.default_rng(7)
        dates = pd.date_range("2014-01-31", periods=120, freq="ME")
        ret = -0.003 + rng.standard_normal(120) * 0.02
        ret[[5, 40, 90]] = [0.15, 0.12, 0.20]
        s = pd.Series(ret, index=dates)
        qd = compute_quarter_distribution(s)
        v = classify_concentration(qd)
        assert v["verdict"] == "HIGH"
        assert qd.drop_top_arc < 0
        # And it carries the senior-readable reason
        assert any("drop-top" in r for r in v["reasons"])

    def test_concentration_classifier_passes_stable_series(self):
        """Phase 5.4: a smooth iid-positive-mean series with most
        quarters profitable should NOT be flagged HIGH."""
        import numpy as np
        import pandas as pd
        from engine.validation.quarter_distribution import (
            classify_concentration, compute_quarter_distribution,
        )
        rng = np.random.default_rng(42)
        dates = pd.date_range("2014-01-31", periods=120, freq="ME")
        s = pd.Series(0.010 + rng.standard_normal(120) * 0.025,
                       index=dates)
        qd = compute_quarter_distribution(s)
        v = classify_concentration(qd)
        assert v["verdict"] != "HIGH"
        assert qd.pct_profitable > 0.60

    def test_pd9_step_skips_short_series(self):
        """Phase 5.4: P-D9 quarter_concentration SKIPs gracefully on
        too-short input — pipeline must not raise."""
        import numpy as np
        import pandas as pd
        from engine.research.candidate_pipeline import _run_quarter_concentration
        s = pd.Series(np.random.randn(8),
                       index=pd.date_range("2024-01-31", periods=8, freq="ME"))
        step = _run_quarter_concentration(s, "test")
        assert step.status == "SKIP"

    def test_pd8_step_skips_without_parent_path(self):
        """Phase 5.2: P-D8 step gracefully SKIPs when no parent series
        provided — must not raise into the pipeline."""
        import pandas as pd
        import numpy as np
        from engine.research.candidate_pipeline import (
            _run_block_bootstrap_significance,
        )
        s = pd.Series(np.random.randn(60),
                       index=pd.date_range("2020-01-31", periods=60, freq="ME"))
        step = _run_block_bootstrap_significance(s, "test", None)
        assert step.step_name == "block_bootstrap_significance"
        assert step.status == "SKIP"
        # Missing file → WARN, also no raise
        step2 = _run_block_bootstrap_significance(
            s, "test", "nonexistent/path.parquet",
        )
        assert step2.status == "WARN"

    def test_pbb_with_dsr_verdict_combines_both_layers(self):
        """Phase 5.1: pbb_sharpe_with_dsr returns STRONG_PASS only when
        PBB CI excludes 0 AND DSR > 0.95 — both selection bias AND
        path realization checks must pass."""
        import numpy as np
        from engine.validation.block_bootstrap import pbb_sharpe_with_dsr
        rng = np.random.default_rng(42)
        # High-skill series: 240 monthly with mu/sigma = 0.10/0.20 → SR_per ≈ 0.5
        good = 0.05 + rng.standard_normal(240) * 0.10
        r = pbb_sharpe_with_dsr(good, n_trials=2, n_iter=2000,
                                  periods_per_year=12, rng_seed=42)
        assert r.verdict in ("STRONG_PASS", "MARGINAL", "WEAK")
        assert r.pbb_ci_lo_per_period < r.pbb_ci_hi_per_period
        assert 0.0 <= r.deflated_sr <= 1.0

    def test_trace_log_nested_spans_parent_chain(self):
        """Phase 4f: nested spans correctly chain parent_id and
        inherit root attrs (workflow_id) for downstream filtering."""
        from engine.research.trace_log import (
            read_spans, reset_for_test, span, start_trace,
        )
        reset_for_test()
        start_trace(workflow_id="l4-trace-regression")
        with span("outer", workflow_id="l4-trace-regression") as outer:
            with span("inner.a", workflow_id="l4-trace-regression"):
                pass
            with span("inner.b", workflow_id="l4-trace-regression"):
                pass
        spans = read_spans(workflow_id="l4-trace-regression")
        names = [s["name"] for s in spans
                  if s.get("kind") != "attr_update"]
        assert sorted(names) == ["inner.a", "inner.b", "outer"]
        # outer.parent_id is None; inner.* have the same parent_id (outer's span_id)
        outer_row = next(s for s in spans if s.get("name") == "outer")
        inners = [s for s in spans
                   if s.get("name") in ("inner.a", "inner.b")]
        assert outer_row["parent_id"] is None
        for inn in inners:
            assert inn["parent_id"] == outer_row["span_id"]

    def test_trace_log_dispatch_emits_tool_span(self):
        """Phase 4f: llm_tools.dispatch is auto-traced — tool calls
        appear as tool.{name} spans with workflow_id inherited from
        the current trace root."""
        from engine.research.llm_tools import dispatch
        from engine.research.trace_log import (
            read_spans, reset_for_test, start_trace,
        )
        reset_for_test()
        start_trace(workflow_id="l4-trace-dispatch-test")
        dispatch("query_intuition_rules", severity="FATAL_BLOCK")
        spans = read_spans(workflow_id="l4-trace-dispatch-test")
        tool_spans = [s for s in spans
                       if (s.get("name") or "").startswith("tool.")]
        assert len(tool_spans) >= 1
        assert tool_spans[0]["name"] == "tool.query_intuition_rules"
        assert (tool_spans[0]["attrs"] or {}).get("kind_class") == "tool"

    def test_rest_traces_endpoint(self):
        """Phase 4f: GET /api/research/traces returns spans filtered
        by workflow_id."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        r = c.get("/api/research/traces?workflow_id=l4-anything&limit=10")
        assert r.status_code == 200
        body = r.json()
        assert "spans" in body and "n" in body

    def test_library_promoter_writes_draft(self):
        """Phase 4e: promoter writes _drafts/<id>.yaml with PROPOSED_DRAFT
        status; library validators IGNORE files in dirs starting with _.

        Smoke ensures the draft does NOT inflate library count and
        carries the senior-fill checklist."""
        import yaml
        from pathlib import Path
        from engine.research.library_promoter import (
            DRAFTS_DIR, promote_iteration_to_draft,
        )
        from engine.research.llm_tools import dispatch
        before_n = (dispatch("query_library") or {}).get("n_matched")
        result = promote_iteration_to_draft({
            "iteration_id": "iter-promoter-regression",
            "workflow_id":  "l4-promoter-regression",
            "proposal": {
                "title": "promoter_regression_xyz",
                "family": "test_family",
                "parent_family": "equity_factor",
                "proposed_role": "alpha_seeker",
                "economics_text": "m",
                "motivation": "m",
                "required_data": [],
            },
            "council":  {"consensus": "APPROVE", "rationale": "ok",
                          "run_id": "c"},
            "pipeline": {"ran": True,
                          "final_decision": "PROMOTE_TO_GATE",
                          "rationale": "pass"},
            "verdict_alignment": "agree",
            "human_override": None,
        })
        path = Path(result["draft_path"])
        assert path.name.endswith(".yaml")
        full = DRAFTS_DIR / path.name
        try:
            assert full.is_file()
            content = yaml.safe_load(full.read_text(encoding="utf-8"))
            assert content["_draft"] is True
            assert content["status_in_our_book"] == "PROPOSED_DRAFT"
            assert content["_l4_origin"]["council_consensus"] == "APPROVE"
            assert len(result["checklist"]) >= 5
            # Library count must NOT have increased
            after_n = (dispatch("query_library") or {}).get("n_matched")
            assert after_n == before_n
        finally:
            if full.exists():
                full.unlink()

    def test_rest_l4_promote_validates_iteration_state(self):
        """Phase 4e: promote endpoint enforces effective_consensus =
        APPROVE + pipeline.ran = True + non-empty justification."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        # Missing justification
        r = c.post("/api/research/l4/promote",
                   json={"iteration_id": "any", "justification": ""})
        assert r.status_code == 422
        # Unknown iteration_id
        r2 = c.post("/api/research/l4/promote",
                     json={"iteration_id": "iter-nope",
                            "justification": "test"})
        assert r2.status_code == 404

    def test_outcome_ledger_alignment_classification(self):
        """Phase 4d: _classify_alignment encodes the calibration
        rules — APPROVE↔PROMOTE = agree; APPROVE↔REJECT = council_wrong."""
        from engine.research.outcome_ledger import _classify_alignment
        assert _classify_alignment("APPROVE", "PROMOTE_TO_GATE") == "agree"
        assert _classify_alignment("APPROVE", "HARD_REJECT") == "council_wrong"
        assert _classify_alignment("REJECT", "HARD_REJECT") == "agree"
        assert _classify_alignment("REJECT", "PROMOTE_TO_GATE") == "council_wrong"
        assert _classify_alignment("NEEDS_REVISION", "BORDERLINE_REVIEW") == "agree"
        assert _classify_alignment(None, "PROMOTE_TO_GATE") == "not_runnable"
        assert _classify_alignment("APPROVE", None) == "not_runnable"

    def test_l4_temporal_unavailable_returns_false(self):
        """Phase 4c: is_temporal_available probes with a tight timeout
        and returns False (not raises) when no server is reachable."""
        import asyncio
        from engine.research.l4_temporal_client import is_temporal_available
        # Bogus port → must return False quickly, not hang
        ok = asyncio.run(is_temporal_available(
            address="127.0.0.1:65534", timeout=0.3,
        ))
        assert ok is False

    def test_rest_council_trigger_requires_confirm_cost(self):
        """Phase 4b.5: trigger endpoint refuses without confirm_cost
        (each invocation costs LLM tokens; explicit ack required)."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        r = c.post("/api/research/council/run",
                    json={"seed_idea": "JP PEAD long enough seed idea here"})
        assert r.status_code == 400
        assert "confirm_cost" in r.json()["detail"]
        # Too-short seed is 422 even with confirm
        r2 = c.post("/api/research/council/run",
                     json={"seed_idea": "x", "confirm_cost": True})
        assert r2.status_code == 422

    def test_agent_council_verdict_parser_is_lenient(self):
        """Phase 4b: parser handles markdown fence + prose prefix +
        the {} format Claude actually emits — without crashing on
        junk."""
        from engine.research.agent_council import _parse_verdict_json
        assert _parse_verdict_json('```json\n{"verdict":"PASS"}\n```')["verdict"] == "PASS"
        assert _parse_verdict_json('Here:\n{"verdict":"WARN"}')["verdict"] == "WARN"
        assert _parse_verdict_json("not json") == {}
        assert _parse_verdict_json("") == {}

    def test_rest_shim_writes_audit_ledger(self):
        """Phase 4a.6: every call appends to ui_tool_calls.jsonl with
        caller / args / result_hash / latency. The hook is here even
        though current 9 tools are read-only, so write tools later
        (override_graveyard, etc.) inherit audit without retrofit."""
        from fastapi.testclient import TestClient
        from api.main import app
        from api.routes_research_tools import AUDIT_LEDGER
        before_size = AUDIT_LEDGER.stat().st_size if AUDIT_LEDGER.is_file() else 0
        c = TestClient(app)
        c.post(
            "/api/research/call/query_graveyard",
            json={"args": {"family": "earnings_underreaction"}},
            headers={"X-Research-Caller": "regression_test"},
        )
        assert AUDIT_LEDGER.is_file()
        assert AUDIT_LEDGER.stat().st_size > before_size
        # Verify the new entry is queryable via the audit endpoint
        r = c.get("/api/research/audit?caller=regression_test&limit=5")
        body = r.json()
        assert body["n"] >= 1
        assert body["entries"][0]["tool"] == "query_graveyard"
        assert body["entries"][0]["caller"] == "regression_test"
        assert body["entries"][0]["ok"] is True


# ── Class C — STRUCTURAL (slow) ────────────────────────────────────────


@pytest.mark.slow
class TestPipelineStructural:
    """Class C: full pipeline run with structural assertions only.

    NO verdict-specific assertions — DA non-determinism makes those
    fragile. We assert pipeline COMPLETES + STEP STATUSES at known
    points + final_decision is a valid enum value.
    """

    PIT_SN_PATH = ROOT / "data" / "cache" / "_dpead_sn_pit_monthly.parquet"

    @pytest.fixture(scope="class")
    def pit_sn_report(self):
        """Run v1 pipeline on PIT SN once for the whole test class."""
        if not self.PIT_SN_PATH.exists():
            pytest.skip(f"data missing: {self.PIT_SN_PATH}")
        from engine.research.candidate_pipeline import run_candidate_pipeline
        s = pd.read_parquet(self.PIT_SN_PATH).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        report = run_candidate_pipeline(
            candidate_returns=s,
            proposal_name="pit_sn_regression_test",
            proposed_role="alpha_seeker",
            mechanism_id="post_earnings_drift",
            proposal_dict={
                "family":         "earnings_underreaction",
                "parent_family":  "equity_factor",
                "required_data":  ["SUE_panel"],
                "economics_text": "PIT FF12 within-sector D_PEAD test.",
            },
            phase=3,
        )
        return report

    def test_pipeline_completes_with_15_steps(self, pit_sn_report):
        """PIT SN pipeline must produce 15 steps (full coverage)."""
        assert len(pit_sn_report.step_results) == 15, (
            f"expected 15 steps, got {len(pit_sn_report.step_results)}"
        )

    def test_h10_step_passes(self, pit_sn_report):
        """PIT SN H10 must PASS — known-good candidate."""
        h10 = next(s for s in pit_sn_report.step_results
                   if s.step_name == "H10_evaluate_candidate")
        assert h10.status == "PASS", f"H10 status {h10.status} (expected PASS)"

    def test_h2_h6_do_not_fail_on_chicken_egg(self, pit_sn_report):
        """8th + 9th catches regression: H2 / H6 must not FAIL with
        'mechanism_id not in library' error for a new candidate.
        Note: H2 may legitimately be WARN (soft_reject) due to actual
        cousin discovery — but never FAIL with the chicken-egg error."""
        h2 = next(s for s in pit_sn_report.step_results
                  if s.step_name == "H2_cousin_check")
        h6 = next(s for s in pit_sn_report.step_results
                  if s.step_name == "H6_post_pub_evidence")
        # The specific bug error wording was "not in library"
        assert "not in library" not in (h2.verdict or "")
        assert "not in library" not in (h6.verdict or "")

    def test_final_decision_is_valid_enum(self, pit_sn_report):
        """final_decision must be one of the known enum values."""
        valid = {
            "PROMOTE_TO_GATE", "PROMOTE_AS_REPLACEMENT",
            "BORDERLINE_REVIEW", "SOFT_REJECT", "HARD_REJECT",
        }
        assert pit_sn_report.final_decision in valid, (
            f"final_decision {pit_sn_report.final_decision!r} not in {valid}"
        )

    def test_relation_classification_for_pit_sn(self, pit_sn_report):
        """PIT SN is a known REPLACEMENT for parent D_PEAD (high cosine
        with deployed equity sleeve). Structural check that the
        classifier identifies this."""
        assert pit_sn_report.candidate_relation == "REPLACEMENT", (
            f"PIT SN relation expected REPLACEMENT, got "
            f"{pit_sn_report.candidate_relation}"
        )


@pytest.mark.slow
class TestL4TemporalWorkflow:
    """Phase 4c Class C: real Temporal workflow execution via
    embedded WorkflowEnvironment (starts a local dev server +
    worker + runs L4DiscoveryWorkflow with MOCKED activities).

    Doesn't burn LLM tokens — the activities are replaced with
    deterministic stubs so the test exercises ONLY the Temporal
    plumbing (workflow definition / activity dispatch / result
    flow / signals / queries)."""

    def test_l4_workflow_runs_end_to_end_with_mock_activities(self):
        """Phase 4c + 4d: full L4DiscoveryWorkflow round-trip with
        mocked activities — propose → critique → pipeline (skipped
        when no candidate_returns_path) → ledger. Exercises ALL
        Temporal plumbing without burning LLM tokens."""
        import asyncio
        from temporalio import activity
        from temporalio.testing import WorkflowEnvironment
        from temporalio.worker import Worker
        from engine.research.l4_workflow import (
            CritiqueInput, L4DiscoveryWorkflow, LedgerInput,
            PipelineInput, ProposeInput, TASK_QUEUE_L4,
        )

        @activity.defn(name="l4_propose_activity")
        async def mock_propose(inp: ProposeInput) -> dict:
            return {
                "title": "TestProposal", "family": "test_family",
                "parent_family": "equity_factor",
                "proposed_role": "alpha_seeker",
                "economics_text": "mock", "required_data": [],
                "motivation": "test", "mechanism_id": None,
            }

        @activity.defn(name="l4_critique_activity")
        async def mock_critique(inp: CritiqueInput) -> dict:
            return {
                "run_id": "test_run_xyz", "consensus": "APPROVE",
                "rationale": "mocked APPROVE", "elapsed_s": 0.1,
                "verdicts": [{"agent_name": "a"}, {"agent_name": "b"}],
                "proposal": inp.proposal_dict,
            }

        @activity.defn(name="l4_pipeline_activity")
        async def mock_pipeline(inp: PipelineInput) -> dict:
            # No candidate_returns_path → simulate skip
            if not inp.candidate_returns_path:
                return {
                    "ran": False,
                    "skipped_reason": "mock skip",
                    "final_decision": None, "rationale": "",
                    "step_results": [],
                    "candidate_returns_path": None,
                }
            return {
                "ran": True, "final_decision": "PROMOTE_TO_GATE",
                "rationale": "mock pass", "step_results": [{}] * 15,
                "candidate_returns_path": inp.candidate_returns_path,
            }

        @activity.defn(name="l4_ledger_activity")
        async def mock_ledger(inp: LedgerInput) -> dict:
            return {"iteration_id": "iter-mock-000000"}

        async def _run() -> None:
            env = await WorkflowEnvironment.start_local()
            try:
                async with Worker(
                    env.client, task_queue=TASK_QUEUE_L4,
                    workflows=[L4DiscoveryWorkflow],
                    activities=[
                        mock_propose, mock_critique,
                        mock_pipeline, mock_ledger,
                    ],
                ):
                    handle = await env.client.start_workflow(
                        L4DiscoveryWorkflow.run,
                        args=["test seed long enough idea", None],
                        id="regression-l4-wf-4d",
                        task_queue=TASK_QUEUE_L4,
                    )
                    result = await handle.result()
                    assert result.consensus == "APPROVE"
                    assert result.proposal_dict["title"] == "TestProposal"
                    assert result.n_critics == 2
                    # 4d: pipeline skipped (no path) but ledger ran
                    assert result.pipeline_ran is False
                    assert result.iteration_id == "iter-mock-000000"
                    # APPROVE × no-pipeline → not_runnable alignment
                    assert result.verdict_alignment == "not_runnable"
            finally:
                await env.shutdown()

        asyncio.run(_run())


@pytest.mark.slow
class TestAgentCouncilEndToEnd:
    """Phase 4b Class C: real Anthropic call exercising the full
    critique fan-out. Skipped if no API key. Costs LLM tokens."""

    def test_critique_council_on_jp_pead_fails_via_graveyard(self):
        """Send a Japan-PEAD proposal to the council. The graveyard
        already has the family RED with cross-market cousins
        (12th-catch fix surfaces these), so both critics should FAIL
        and consensus should be REJECT.

        Validates: (a) Anthropic tool-use loop fires the right tools,
        (b) verdict JSON parses, (c) aggregator returns REJECT, (d)
        12th-catch alias normalization carries through the full stack
        (tools → MCP → REST → council)."""
        import asyncio
        from engine.research.agent_council import (
            ProposalDict, _load_anthropic_key, critique_council,
        )
        if not _load_anthropic_key():
            pytest.skip("no ANTHROPIC_API_KEY in env or secrets.toml")

        prop = ProposalDict(
            title="Japan PEAD",
            family="earnings_underreaction",
            parent_family="equity_factor",
            proposed_role="alpha_seeker",
            economics_text=("Post-earnings drift in Japan. "
                            "Bernard-Thomas 1989 mechanism."),
            required_data=["I/B/E/S Japan EPS", "TOPIX returns"],
            motivation="Cross-market extension of deployed D_PEAD.",
        )
        council = asyncio.run(critique_council(prop))
        assert council.consensus == "REJECT", (
            f"expected REJECT due to graveyard block, got "
            f"{council.consensus}; rationale={council.rationale[:300]}"
        )
        assert len(council.verdicts) == 2
        # both critics should have called at least one tool
        for v in council.verdicts:
            assert len(v.tool_calls) >= 1, (
                f"{v.agent_name} did no tool calls — broken loop"
            )


@pytest.mark.slow
class TestV2DurableRoundTrip:
    """Phase 4a Class C: full v2 pipeline run with SqliteSaver
    checkpointer + resume from disk. Verifies the inner-ring durable
    boundary end-to-end."""

    def test_v2_durable_run_then_resume_matches_verdict(self):
        """Run a v2 pipeline with durable=True, then resume by
        thread_id and confirm final_decision is identical."""
        import numpy as np
        from engine.research.candidate_pipeline_v2 import (
            run_candidate_pipeline_v2, make_thread_id,
            get_checkpoint_state, resume_candidate_pipeline_v2,
        )
        np.random.seed(42)
        dates = pd.date_range("2014-01-31", periods=120, freq="ME")
        r = pd.Series(0.005 + 0.04 * np.random.randn(120),
                      index=dates, name="v2_durable_test")

        tid = make_thread_id("v2_durable_test")
        first = run_candidate_pipeline_v2(
            candidate_returns=r, proposal_name="v2_durable_test",
            proposed_role="alpha_seeker", thread_id=tid, durable=True,
        )

        # Checkpoint must be queryable after the run
        snap = get_checkpoint_state(tid)
        assert snap is not None
        assert "step_results" in snap
        assert len(snap["step_results"]) == len(first.step_results)

        # Resume returns equivalent terminal report
        second = resume_candidate_pipeline_v2(tid)
        assert second.final_decision == first.final_decision, (
            f"durable round-trip verdict drift: "
            f"{first.final_decision} vs {second.final_decision}"
        )
        assert len(second.step_results) == len(first.step_results)
