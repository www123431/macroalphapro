"""Tests for engine.research.hygiene_tools — the 7 deterministic gates H1-H7."""
from __future__ import annotations

from engine.research.hygiene_tools import (
    DATA_INVENTORY, TOOL_SCHEMAS,
    h1_list_unexplored_library_entries,
    h2_cousin_check_multilevel,
    h3_check_data_inventory,
    h4_verify_paper_in_library,
    h5_count_free_params,
    h6_post_pub_evidence_check,
    h7_kill_this_proposal,
    execute_tool,
)


def test_tool_schemas_count():
    assert len(TOOL_SCHEMAS) == 7
    names = {s["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "h1_list_unexplored_library_entries",
        "h2_cousin_check_multilevel",
        "h3_check_data_inventory",
        "h4_verify_paper_in_library",
        "h5_count_free_params",
        "h6_post_pub_evidence_check",
        "h7_kill_this_proposal",
    }


def test_h1_no_visible_entries_returns_empty():
    # All YAMLs are audit_signature=pending → 0 visible
    r = h1_list_unexplored_library_entries(include_pending=False)
    assert r.success
    assert r.payload["n_unexplored"] == 0


def test_h1_include_pending_returns_2_candidates():
    r = h1_list_unexplored_library_entries(include_pending=True)
    assert r.success
    ids = {e["id"] for e in r.payload["entries"]}
    assert "equity_xsmom_jt" in ids
    assert "low_vol_bab" in ids


def test_h2_low_vol_bab_hard_rejects_against_quality_qmj_red():
    # low_vol_bab family=quality, quality_qmj is RED → hard_reject
    r = h2_cousin_check_multilevel("low_vol_bab")
    assert r.success
    assert r.payload["verdict"] == "hard_reject"
    assert any("quality_qmj" in reason for reason in r.payload["hard_reject_reasons"])


def test_h2_equity_xsmom_jt_soft_rejects():
    # equity_xsmom_jt shares parent_family=equity_factor with multiple but
    # different family from PEAD/quality, so soft_reject not hard
    r = h2_cousin_check_multilevel("equity_xsmom_jt")
    assert r.success
    assert r.payload["verdict"] in ("soft_reject", "allow")
    assert len(r.payload["L2_same_parent"]) >= 1


def test_h2_unknown_mechanism_fails():
    r = h2_cousin_check_multilevel("ghost_mechanism")
    assert r.success is False


def test_h3_known_data_passes():
    r = h3_check_data_inventory(["crsp_dsf", "compustat_annual"])
    assert r.success
    assert r.payload["all_present"] is True


def test_h3_unknown_data_fails():
    r = h3_check_data_inventory(["made_up_dataset", "crsp_dsf"])
    assert r.success
    assert r.payload["all_present"] is False
    assert "made_up_dataset" in r.payload["missing"]


def test_h3_declared_not_implemented_surfaced_separately():
    """Tokens like optionm_iv_surface are DECLARED but not yet wired —
    hygiene must catch them, but distinguish from 'unknown token'."""
    r = h3_check_data_inventory(["optionm_iv_surface", "made_up", "crsp_dsf"])
    assert r.success
    assert r.payload["all_present"] is False
    # optionm_iv_surface is in declared_not_implemented bucket
    assert "optionm_iv_surface" in r.payload["declared_not_implemented"]
    # made_up is in truly_missing bucket
    assert "made_up" in r.payload["truly_missing"]
    # Both rolled into missing
    assert "optionm_iv_surface" in r.payload["missing"]
    assert "made_up" in r.payload["missing"]


def test_h3_ibes_detail_is_declared_not_implemented():
    """ibes_detail used to be in DATA_INVENTORY but no real fetcher.
    After split, it must surface as declared_not_implemented."""
    r = h3_check_data_inventory(["ibes_detail"])
    assert r.payload["all_present"] is False
    assert "ibes_detail" in r.payload["declared_not_implemented"]


def test_h3_ibes_summary_is_truly_implemented():
    """ibes_summary really is wired (path_c.earnings_panel uses it)."""
    r = h3_check_data_inventory(["ibes_summary"])
    assert r.payload["all_present"] is True


def test_h4_verified_paper_passes():
    r = h4_verify_paper_in_library("jegadeesh_titman_1993_jf")
    assert r.success
    assert r.payload["verified"] is True
    assert r.payload["in_master_index"] is True


def test_h4_unknown_paper_rejected():
    r = h4_verify_paper_in_library("fake_paper_2025")
    assert r.success
    assert r.payload["in_master_index"] is False
    assert r.payload["verified"] is False


def test_h5_single_value_ok():
    r = h5_count_free_params(["lookback=12", "cost=12bp"])
    assert r.payload["verdict"] == "ok"
    assert r.payload["free_params"] == 2


def test_h5_grid_in_brackets_rejected():
    r = h5_count_free_params(["lookback in [3, 6, 12]"])
    assert r.payload["verdict"] == "reject_grid_hide"


def test_h5_grid_in_braces_rejected():
    r = h5_count_free_params(["mode in {EW, VW}"])
    assert r.payload["verdict"] == "reject_grid_hide"


def test_h5_range_call_rejected():
    r = h5_count_free_params(["lookback=range(3, 24)"])
    assert r.payload["verdict"] == "reject_grid_hide"


def test_h5_to_range_rejected():
    r = h5_count_free_params(["lookback from 3 to 24"])
    assert r.payload["verdict"] == "reject_grid_hide"


def test_h5_hyphen_single_value_not_rejected():
    """Regression: '12-1' is a canonical momentum horizon name, not a range."""
    r = h5_count_free_params(["horizon=12-1 (12-month lookback, 1-month skip)"])
    assert r.payload["verdict"] == "ok"


def test_h5_year_range_not_rejected():
    """Regression: '2010-2024' is a sample-window identifier, not a grid."""
    r = h5_count_free_params(["sample=2010-2024"])
    assert r.payload["verdict"] == "ok"


def test_h6_candidate_with_post_pub_replications_passes():
    r = h6_post_pub_evidence_check("equity_xsmom_jt")
    assert r.success
    assert r.payload["verdict"] == "ok"
    assert r.payload["n_qualifying"] >= 1


def test_h6_cousin_anchor_marked_not_applicable():
    """purpose=cousin_anchor entries don't need post-pub evidence (they're
    not selectable by generator anyway)."""
    r = h6_post_pub_evidence_check("post_earnings_drift")
    assert r.success
    assert r.payload.get("applicable") is False


def test_h7_kill_vague_proposal():
    proposal = {"mechanism_id": "x", "justification": "short"}
    r = h7_kill_this_proposal(proposal)
    assert r.payload["verdict"] == "kill"
    assert any("vague" in reason.lower() or "short" in reason.lower()
                 for reason in r.payload["kill_reasons"])


def test_h7_survive_complete_proposal():
    proposal = {
        "mechanism_id":   "equity_xsmom_jt",
        "sample_start":   "1965-01-01",
        "sample_end":     "2026-05-30",
        "parameters":     ["horizon=12-1"],
        "justification":  "A complete justification that is sufficiently long to pass the vagueness check, citing tool results from H1 H2 H3 H4 H5 H6 hygiene gates.",
        "h2_cousin_check_result": {"verdict": "soft_reject"},
        "h3_data_check_result":   {"all_present": True},
        "h4_paper_check_result":  {"verified": True},
        "h5_param_check_result":  {"verdict": "ok"},
        "h6_post_pub_check_result": {"verdict": "ok"},
    }
    r = h7_kill_this_proposal(proposal)
    assert r.payload["verdict"] == "survive", r.payload["kill_reasons"]


def test_h7_kill_on_h2_hard_reject():
    proposal = {
        "mechanism_id":   "x",
        "sample_start":   "1965-01-01",
        "sample_end":     "2026-05-30",
        "parameters":     ["horizon=12"],
        "justification":  "Long enough justification text that passes the vagueness check easily.",
        "h2_cousin_check_result": {"verdict": "hard_reject",
                                     "hard_reject_reasons": ["cousin of X RED"]},
    }
    r = h7_kill_this_proposal(proposal)
    assert r.payload["verdict"] == "kill"
    assert any("H2" in reason for reason in r.payload["kill_reasons"])


def test_h7_kill_on_h3_missing_data():
    proposal = {
        "mechanism_id":   "x",
        "sample_start":   "1965-01-01",
        "sample_end":     "2026-05-30",
        "parameters":     ["horizon=12"],
        "justification":  "Long enough justification text that passes the vagueness check.",
        "h3_data_check_result": {"all_present": False, "missing": ["intraday"]},
    }
    r = h7_kill_this_proposal(proposal)
    assert r.payload["verdict"] == "kill"
    assert any("H3" in reason for reason in r.payload["kill_reasons"])


def test_execute_tool_dispatch():
    r = execute_tool("h3_check_data_inventory", required_data=["crsp_dsf"])
    assert r.success
    assert r.payload["all_present"] is True


def test_execute_unknown_tool_returns_error():
    r = execute_tool("nonexistent_tool")
    assert r.success is False
    assert "unknown" in (r.error or "").lower()
