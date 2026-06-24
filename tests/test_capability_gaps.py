"""tests/test_capability_gaps.py — flex-3: refusal routing slips.

Locks: tier classification with live data probes, friction-preserving
statistical-gate guidance, ledger dedup by (hypothesis_id, signature),
demand aggregation, and the wired refusal surfaces.
"""
from __future__ import annotations

import json

import pytest


def _redirect_ledger(monkeypatch, tmp_path):
    import engine.research.capability_gaps as cg
    ledger = tmp_path / "gaps.jsonl"
    monkeypatch.setattr(cg, "GAPS_LEDGER", ledger)
    return ledger


# ────────────────────────────────────────────────────────────────────
# Guidance builders
# ────────────────────────────────────────────────────────────────────
def test_unsupported_signal_classifies_tier2_when_data_cached(monkeypatch):
    """With CRSP+Compustat caches present (live-probed), an unknown
    signal is a TIER_2 formula gap — registry entry, ~30 min."""
    import engine.research.capability_gaps as cg
    monkeypatch.setattr(cg, "_probe_cross_sec_data", lambda: {
        "crsp_msf_cached": True, "compustat_pit_cached": True,
        "compustat_any_cached": True})
    g = cg.guidance_unsupported_signal(("total_accruals",))
    assert g["gap_class"] == cg.GAP_TIER_2_SIGNAL
    assert "SignalDefinition" in g["next_action"]
    assert "30 min" in g["effort"]


def test_unsupported_signal_classifies_tier4_when_data_missing(monkeypatch):
    import engine.research.capability_gaps as cg
    monkeypatch.setattr(cg, "_probe_cross_sec_data", lambda: {
        "crsp_msf_cached": False, "compustat_pit_cached": False,
        "compustat_any_cached": False})
    g = cg.guidance_unsupported_signal(("anything",))
    assert g["gap_class"] == cg.GAP_TIER_4_DATA


def test_unsupported_universe_tier3_when_data_cached(monkeypatch):
    """commodity carry: settle parquet IS cached → TIER_3 template
    work, with the C-2f precedent named."""
    import engine.research.capability_gaps as cg
    monkeypatch.setattr(cg, "_probe_universe_data", lambda u: True)
    g = cg.guidance_unsupported_universe("carry", "commodity_futures_27")
    assert g["gap_class"] == cg.GAP_TIER_3_TEMPLATE
    assert "C-2f" in g["next_action"]


def test_unsupported_universe_tier4_when_no_data(monkeypatch):
    import engine.research.capability_gaps as cg
    monkeypatch.setattr(cg, "_probe_universe_data", lambda u: False)
    g = cg.guidance_unsupported_universe("vrp", "options_spx")
    assert g["gap_class"] == cg.GAP_TIER_4_DATA
    assert "LRV precedent" in g["effort"]


def test_unknown_universe_probe_returns_tier4_with_probe_advice():
    import engine.research.capability_gaps as cg
    g = cg.guidance_unsupported_universe("vrp", "weird_new_domain")
    assert g["gap_class"] == cg.GAP_TIER_4_DATA
    assert "_UNIVERSE_DATA_PROBES" in g["next_action"]


def test_statistical_gate_guidance_preserves_friction():
    """Locked qualification 1: stat-gate guidance explains WHY +
    legitimate paths; override always demands a written reason —
    never a one-click."""
    import engine.research.capability_gaps as cg
    g = cg.guidance_statistical_gate("N_TRIALS_HARD", "n=15/15")
    assert g["gap_class"] == cg.GAP_STAT_GATE
    assert "Bailey-LdP" in g["next_action"]
    assert "written" in g["next_action"]       # reason required
    assert "abandon the family" in g["next_action"]   # legit alternative
    g2 = cg.guidance_statistical_gate("WEEKLY_CAP", "6/5")
    assert "retrospective" in g2["next_action"]


# ────────────────────────────────────────────────────────────────────
# Ledger: dedup + aggregation
# ────────────────────────────────────────────────────────────────────
def test_ledger_dedups_by_hypothesis_and_signature(monkeypatch, tmp_path):
    """Locked qualification 3: extractor retries don't inflate."""
    import engine.research.capability_gaps as cg
    ledger = _redirect_ledger(monkeypatch, tmp_path)
    g = {"gap_class": cg.GAP_TIER_2_SIGNAL,
          "requested": ["total_accruals"],
          "next_action": "x", "effort": "y"}
    assert cg.log_gap(hypothesis_id="h1", guidance=g) is not None
    assert cg.log_gap(hypothesis_id="h1", guidance=g) is None   # dedup
    assert cg.log_gap(hypothesis_id="h2", guidance=g) is not None
    rows = [json.loads(l) for l
              in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2


def test_aggregate_counts_distinct_hypotheses(monkeypatch, tmp_path):
    import engine.research.capability_gaps as cg
    _redirect_ledger(monkeypatch, tmp_path)
    g_vrp = {"gap_class": cg.GAP_TIER_3_TEMPLATE,
               "requested": {"signal_kind": "vrp", "universe": "options"},
               "next_action": "build vrp template", "effort": "1-3 days"}
    for h in ("paper_a", "paper_b", "paper_c"):
        cg.log_gap(hypothesis_id=h, guidance=g_vrp)
    g_sig = {"gap_class": cg.GAP_TIER_2_SIGNAL,
               "requested": ["nsi"], "next_action": "x", "effort": "y"}
    cg.log_gap(hypothesis_id="paper_d", guidance=g_sig)
    agg = cg.aggregate_gaps()
    by_class = {a["gap_class"]: a for a in agg}
    assert by_class[cg.GAP_TIER_3_TEMPLATE]["demand_count"] == 3
    assert by_class[cg.GAP_TIER_2_SIGNAL]["demand_count"] == 1


def test_aggregate_excludes_statistical_gates(monkeypatch, tmp_path):
    """Stat-gate refusals are friction working as designed — not
    unmet demand. Excluded from the digest."""
    import engine.research.capability_gaps as cg
    _redirect_ledger(monkeypatch, tmp_path)
    cg.log_gap(hypothesis_id="h1", guidance={
        "gap_class": cg.GAP_STAT_GATE, "requested": None,
        "next_action": "x", "effort": "n/a"})
    assert cg.aggregate_gaps() == []


# ────────────────────────────────────────────────────────────────────
# Wired surfaces
# ────────────────────────────────────────────────────────────────────
def test_pending_build_carries_guidance_and_logs(monkeypatch, tmp_path):
    import engine.research.capability_gaps as cg
    ledger = _redirect_ledger(monkeypatch, tmp_path)
    from engine.agents.strengthener.factor_dispatcher import (
        _template_pending_build,
    )
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    spec = FactorSpec(
        hypothesis_id="pb_probe", signal_kind="vrp",
        universe="us_treasury_curve", date_range="2014-01:2024-12",
        signal_inputs=("optionmetrics.x",), rebal="monthly",
        weighting="equal_weight_basket",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="t", extracted_ts="2026-06-10T00:00:00Z", model="c",
    )
    r = _template_pending_build(spec)
    assert r.verdict == "PENDING_TEMPLATE_BUILD"
    assert r.metrics["guidance"]["gap_class"] in (
        cg.GAP_TIER_3_TEMPLATE, cg.GAP_TIER_4_DATA)
    # demand logged
    assert ledger.is_file()
    assert "pb_probe" in ledger.read_text(encoding="utf-8")
    # summary carries the next action (routing slip, not wall)
    assert len(r.summary) > 60


def test_unsupported_signal_template_carries_guidance(monkeypatch, tmp_path):
    _redirect_ledger(monkeypatch, tmp_path)
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        template_cross_sec_us_equities,
    )
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    spec = FactorSpec(
        hypothesis_id="us_probe", signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000", date_range="2014-01:2024-12",
        signal_inputs=("compustat.funda.total_accruals",),
        rebal="monthly", weighting="decile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="t", extracted_ts="2026-06-10T00:00:00Z", model="c",
    )
    r = template_cross_sec_us_equities(spec)
    assert r.verdict == "UNSUPPORTED_SIGNAL"
    assert "guidance" in r.metrics
    assert r.metrics["guidance"].get("gap_class")


def test_weekly_cap_refusal_carries_friction_guidance(monkeypatch, tmp_path):
    import datetime, json as _json
    from pathlib import Path
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    spec = FactorSpec(
        hypothesis_id="wc_probe", signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000", date_range="2014-01:2024-12",
        signal_inputs=("crsp.msf.ret",), rebal="monthly",
        weighting="decile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="t", extracted_ts="2026-06-10T00:00:00Z", model="c",
    )
    log = tmp_path / "log.jsonl"
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    log.write_text("\n".join(_json.dumps({"ts": now}) for _ in range(6))
                     + "\n", encoding="utf-8")
    r = fd.pre_dispatch_check(spec, spec_approved=True,
                                  family_hint="X", log_path=log)
    assert r.reason_code == "WEEKLY_CAP"
    g = r.metrics["guidance"]
    assert g["gap_class"] == "STATISTICAL_GATE"
    assert "retrospective" in g["next_action"]


# ────────────────────────────────────────────────────────────────────
# Composer digest
# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
# flex-7: refusal guidance registry (declarative, not per-site)
# ────────────────────────────────────────────────────────────────────
def test_build_refusal_attaches_guidance_to_metrics(monkeypatch, tmp_path):
    """Factory looks up the provider by reason_code and attaches the
    guidance dict — refusal sites don't import builders or wire
    try/except inline."""
    from engine.research import capability_gaps as cg
    _redirect_ledger(monkeypatch, tmp_path)
    r = cg.build_refusal(
        "TEMPLATE_NOT_CERTIFIED",
        detail="no cert.", metrics={},
        hypothesis_id="h_factory",
        context={"signal_kind": "carry",
                   "universe": "commodity_futures_27"},
    )
    g = r.metrics["guidance"]
    assert g["gap_class"] in (cg.GAP_TIER_3_TEMPLATE, cg.GAP_TIER_4_DATA)
    # detail enriched with next_action (the routing slip)
    assert len(r.detail) > len("no cert.")


def test_build_refusal_logs_user_demand_but_not_stat_gates(
    monkeypatch, tmp_path,
):
    """Tier-2/3/4 gaps log to ledger (build-priority signal); stat
    gates do not (friction is by design, not unmet demand)."""
    from engine.research import capability_gaps as cg
    _redirect_ledger(monkeypatch, tmp_path)
    cg.build_refusal(
        "TEMPLATE_NOT_CERTIFIED", detail="x", metrics={},
        hypothesis_id="h_demand",
        context={"signal_kind": "vrp", "universe": "options_spx"},
    )
    cg.build_refusal(
        "WEEKLY_CAP", detail="x", metrics={},
        hypothesis_id="h_capped",
        context={"week_count": 6, "cap": 5},
    )
    agg = cg.aggregate_gaps()
    classes = {a["gap_class"] for a in agg}
    assert cg.GAP_TIER_4_DATA in classes or cg.GAP_TIER_3_TEMPLATE in classes
    assert cg.GAP_STAT_GATE not in classes


def test_every_dispatch_refusal_code_has_a_provider():
    """Structural invariant: every reason_code that
    pre_dispatch_check can emit must have a registered guidance
    provider — flex-7 makes doctrine compliance structural, not by
    convention. Adding a new refusal class requires registering
    its provider in REFUSAL_GUIDANCE, or this test fails."""
    import ast, pathlib
    from engine.research.capability_gaps import REFUSAL_GUIDANCE
    src = (pathlib.Path(__file__).resolve().parents[1]
             / "engine" / "agents" / "strengthener"
             / "factor_dispatcher.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    declared: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "reason_code"
                          for t in node.targets)):
            if isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, str):
                declared.add(node.value.value)
        if isinstance(node, ast.keyword) and node.arg == "reason_code":
            if isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, str):
                declared.add(node.value.value)
    # Gates not routed through build_refusal yet (acceptable: they
    # don't carry user-facing demand — controlled enum violations or
    # spec hygiene). Listed explicitly so adding ANOTHER one without
    # a provider remains a failing test.
    EXEMPT = {"UNKNOWN_SIGNAL_KIND", "NOT_APPROVED", "SIGNAL_INPUT_UNKNOWN",
                "B_CLASS_OUT_OF_RANGE"}
    needs_provider = declared - EXEMPT
    missing = needs_provider - set(REFUSAL_GUIDANCE)
    assert not missing, (
        f"refusal codes without guidance provider: {sorted(missing)} — "
        "register one in capability_gaps.REFUSAL_GUIDANCE or add to "
        "EXEMPT with justification.")


def test_inbox_digest_renders_demand(monkeypatch, tmp_path):
    import engine.research.capability_gaps as cg
    _redirect_ledger(monkeypatch, tmp_path)
    g_vrp = {"gap_class": cg.GAP_TIER_3_TEMPLATE,
               "requested": {"signal_kind": "vrp", "universe": "options"},
               "next_action": "build vrp template",
               "effort": "1-3 days"}
    for h in ("p1", "p2", "p3"):
        cg.log_gap(hypothesis_id=h, guidance=g_vrp)
    from engine.inbox.composer import source_capability_gaps_digest
    items = source_capability_gaps_digest()
    assert len(items) == 1
    it = items[0]
    assert "×3" in it["title"]
    assert "vrp×options" in it["title"]
    assert it["tone"] == "warn"     # demand >= 3 escalates tone
    assert it["source"] == "capability_gap"
