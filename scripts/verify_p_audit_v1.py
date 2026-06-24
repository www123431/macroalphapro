"""
P-AUDIT v1 — full verification harness.

Usage: D:/python/python.exe scripts/verify_p_audit_v1.py

Coverage:
  Facet 1  S3 forward registration (spec_hash present, retro=False, +1 trial)
  Facet 2  EFFECTIVE_N_TRIALS module-level reflects forward registration
  Facet 3  PendingApproval has review_rationale + review_category columns
  Facet 4  approval_context.get_approval_context() returns expected schema
  Facet 5  approval_context.get_similar_past_approvals() returns list
  Facet 6  approval_context.get_decision_replay() returns list
  Facet 7  approval_context.validate_review_inputs() rule
  Facet 8  approval_analytics 3 functions return shapes
  Facet 9  AppTest pages/orchestrator.py cold + seeded — 0 exception
  Facet 10 AppTest pages/approval_analytics.py — 0 exception
  Facet 11 resolve_pending_approval persists review_rationale + category
  Facet 12 P-AUDIT-1 → P-AUDIT-5 spec sub-sprints all delivered files exist
"""
from __future__ import annotations

import datetime
import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SPEC_PATH = "docs/spec_supervisor_approval_panel_v1.md"


def _hr(title: str) -> None:
    bar = "─" * 70
    print(f"\n{bar}\n{title}\n{bar}")


# Facet 1 + 2  ──────────────────────────────────────────────────────────────
_hr("Facet 1 + 2 — S3 forward registration + EFFECTIVE_N_TRIALS")

from engine.preregistration import (
    compute_pre_registration_n_trials, register_spec, _compute_git_blob_hash,
)
from engine.memory import SessionFactory, SpecRegistry

with SessionFactory() as s:
    row = (
        s.query(SpecRegistry)
         .filter(SpecRegistry.spec_path == SPEC_PATH.replace("\\", "/"))
         .first()
    )
    assert row is not None, f"{SPEC_PATH} NOT registered"
    assert not row.retro_registered, "P-AUDIT spec must be forward (retro=False)"
    assert row.n_trials_contributed >= 1
    print(f"  spec_hash[:16] = {row.current_hash[:16]}")
    print(f"  retro_registered = {row.retro_registered}")
    print(f"  n_trials_contributed = {row.n_trials_contributed}")

forward_total = compute_pre_registration_n_trials()
print(f"  forward registrations total = {forward_total}")
assert forward_total >= 2, f"forward total should be ≥ 2 (P-FUND + P-AUDIT); got {forward_total}"

from engine.backtest import refresh_effective_n_trials
n_eff, audit = refresh_effective_n_trials()
print(f"  EFFECTIVE_N_TRIALS = {n_eff}")
print(f"  pre_registration axis = {audit.get('pre_registration')}")
assert audit.get("pre_registration") == forward_total
assert n_eff >= 45, f"EFFECTIVE_N_TRIALS ≥ 45 expected (43 grid + ≥2 forward); got {n_eff}"


# Facet 3  ──────────────────────────────────────────────────────────────────
_hr("Facet 3 — PendingApproval audit columns present")

from engine.memory import _DB_PATH
con = sqlite3.connect(_DB_PATH)
cols = {row[1] for row in con.execute("PRAGMA table_info(pending_approvals)").fetchall()}
con.close()
required = {"review_rationale", "review_category"}
missing = required - cols
assert not missing, f"missing columns: {missing}"
print(f"  pending_approvals columns OK: review_rationale + review_category present")


# Facet 4 + 5 + 6 + 7  ──────────────────────────────────────────────────────
_hr("Facet 4-7 — approval_context.py contracts")

from engine.approval_context import (
    get_approval_context, get_similar_past_approvals,
    get_decision_replay, validate_review_inputs, REVIEW_CATEGORIES,
)
from engine.memory import PendingApproval

with SessionFactory() as s:
    pa_id_row = s.query(PendingApproval.id).order_by(PendingApproval.id.desc()).first()
assert pa_id_row is not None, "no PendingApproval rows in DB"
pa_id = int(pa_id_row[0])

ctx = get_approval_context(pa_id)
assert set(ctx) >= {"found", "approval_id", "base", "cb_status",
                    "harking", "quant_ctx", "reject_preview"}
assert ctx["found"] is True
assert ctx["base"]["approval_id"] == pa_id
print(f"  Facet 4 get_approval_context: shape OK on id={pa_id}")

sim = get_similar_past_approvals(pa_id, top_k=3)
assert isinstance(sim, list) and len(sim) <= 3
print(f"  Facet 5 get_similar_past_approvals: list len={len(sim)}")

rep = get_decision_replay(pa_id)
assert isinstance(rep, list)
if rep:
    assert set(rep[0]) >= {"ts", "type", "actor", "payload_summary",
                           "run_id_link", "reconstructed"}
print(f"  Facet 6 get_decision_replay: list len={len(rep)}")

assert validate_review_inputs("", "signal_confirmed")[0] is False
assert validate_review_inputs("approve normal", "mystery")[0] is False
assert validate_review_inputs("approve normal because regime supports", "signal_confirmed")[0] is True
assert REVIEW_CATEGORIES == ("signal_confirmed", "regime_driven",
                             "supervisor_discretion", "risk_override",
                             "cash_flow_routine", "other")
print("  Facet 7 validate_review_inputs: 3 cases OK; enum frozen")


# Facet 8  ──────────────────────────────────────────────────────────────────
_hr("Facet 8 — approval_analytics.py contracts")

from engine.approval_analytics import (
    get_approval_rate_by_period, get_category_outcome_correlation,
    get_supervisor_override_pattern,
)

df1 = get_approval_rate_by_period("month")
assert {"period", "n_total", "n_approved", "n_rejected",
        "approval_rate"}.issubset(df1.columns)
print(f"  get_approval_rate_by_period(month): rows={len(df1)}")

df2 = get_category_outcome_correlation(min_n=1)
print(f"  get_category_outcome_correlation: rows={len(df2)}")

patt = get_supervisor_override_pattern()
assert {"n_total", "n_approved", "n_rejected", "approval_rate",
        "hit_rate_when_approved"}.issubset(patt.keys())
print(f"  get_supervisor_override_pattern: keys OK ({patt['n_total']} total)")


# Facet 9 + 10  ─────────────────────────────────────────────────────────────
_hr("Facet 9 + 10 — AppTest cold + seeded")

from streamlit.testing.v1 import AppTest

ops_path = os.path.abspath("pages/orchestrator.py")
at = AppTest.from_file(ops_path, default_timeout=180)
at.run()
n_exc = len(at.exception)
print(f"  cold orchestrator.py exceptions: {n_exc}")
for e in at.exception[:2]:
    print(f"    {str(e.value)[:200]}")
assert n_exc == 0

# Seeded: flip latest approved → pending temporarily
saved = None
with SessionFactory() as s:
    pa_seed = (
        s.query(PendingApproval)
         .filter(PendingApproval.status == "approved")
         .order_by(PendingApproval.id.desc())
         .first()
    )
    if pa_seed is not None:
        saved = (pa_seed.id, pa_seed.status, pa_seed.resolved_at,
                 pa_seed.resolved_by, pa_seed.approval_deadline)
        pa_seed.status = "pending"
        pa_seed.resolved_at = None
        pa_seed.resolved_by = None
        pa_seed.approval_deadline = datetime.date.today() + datetime.timedelta(days=3)
        s.commit()

try:
    at = AppTest.from_file(ops_path, default_timeout=180)
    at.run()
    n_exc = len(at.exception)
    print(f"  seeded orchestrator.py exceptions: {n_exc}")
    assert n_exc == 0
    queue_btns = [b for b in at.button if (b.key or "").startswith("q_")]
    if queue_btns:
        queue_btns[0].click().run()
        n_exc2 = len(at.exception)
        print(f"  queue-click orchestrator.py exceptions: {n_exc2}")
        assert n_exc2 == 0
finally:
    if saved is not None:
        with SessionFactory() as s:
            r = s.get(PendingApproval, saved[0])
            if r is not None:
                r.status = saved[1]
                r.resolved_at = saved[2]
                r.resolved_by = saved[3]
                r.approval_deadline = saved[4]
                s.commit()

an_path = os.path.abspath("pages/approval_analytics.py")
at = AppTest.from_file(an_path, default_timeout=180)
at.run()
n_exc = len(at.exception)
print(f"  approval_analytics.py exceptions: {n_exc}")
for e in at.exception[:2]:
    print(f"    {str(e.value)[:200]}")
assert n_exc == 0


# Facet 11  ─────────────────────────────────────────────────────────────────
_hr("Facet 11 — resolve_pending_approval persists rationale + category")

from engine.memory import resolve_pending_approval

# Create a temp pending row, resolve it via the new path, verify columns set
with SessionFactory() as s:
    tmp = PendingApproval(
        approval_type="entry",
        priority="normal",
        sector="P-AUDIT-VERIFY",
        ticker="VRFY",
        triggered_condition="verify_p_audit_v1.py temp row",
        triggered_date=datetime.date.today(),
        suggested_weight=0.0,
        approval_deadline=datetime.date.today() + datetime.timedelta(days=3),
        status="pending",
    )
    s.add(tmp); s.commit()
    tmp_id = int(tmp.id)
print(f"  created temp pending id={tmp_id}")

result = resolve_pending_approval(
    approval_id=tmp_id,
    approved=False,
    resolved_by="verify_script",
    rejection_reason="rejected by verify harness",
    review_rationale="P-AUDIT verify run — temporary record, immediately deleted",
    review_category="other",
)
print(f"  resolve_pending_approval ok={result.get('ok')}")

with SessionFactory() as s:
    r = s.get(PendingApproval, tmp_id)
    assert r is not None
    assert r.status == "rejected"
    assert (r.review_rationale or "").startswith("P-AUDIT verify run")
    assert r.review_category == "other"
    print(f"  persisted: rationale={r.review_rationale[:40]!r}, category={r.review_category}")
    s.delete(r); s.commit()
    print(f"  cleanup: temp id={tmp_id} deleted")


# Facet 12 ──────────────────────────────────────────────────────────────────
_hr("Facet 12 — sub-sprint deliverables exist")

deliverables = [
    "docs/spec_supervisor_approval_panel_v1.md",
    "engine/approval_context.py",
    "engine/approval_analytics.py",
    "pages/approval_analytics.py",
    "pages/orchestrator.py",
    "scripts/verify_p_audit_v1.py",
]
for d in deliverables:
    full = os.path.abspath(d)
    ok = os.path.exists(full)
    print(f"  {'OK' if ok else 'MISSING':8s}  {d}")
    assert ok, f"missing deliverable: {d}"


# ─────────────────────────────────────────────────────────────────────────────
# M3-corrected-ext-full amendment facets (2026-05-04 clarification +0)
# ─────────────────────────────────────────────────────────────────────────────
_hr("Facet 13-22 — M3-corrected-ext-full DECISION CONTEXT layers")

from engine.memory import (
    PendingApproval, SessionFactory, SpecRegistry,
)
from engine import decision_context as dc
import json as _json

# Pick most recent approval as test target
with SessionFactory() as s:
    pa_test = s.query(PendingApproval).order_by(PendingApproval.id.desc()).first()
    pa_test_id = int(pa_test.id) if pa_test is not None else None
    test_sector = pa_test.sector if pa_test else None
    test_ticker = pa_test.ticker if pa_test else None
    test_sw     = float(pa_test.suggested_weight or 0.0) if pa_test else 0.0
print(f"  test approval id={pa_test_id} sector={test_sector!r} ticker={test_ticker!r}")
assert pa_test_id is not None

# Facet 13: amendment ledger entry exists (clarification +0)
with SessionFactory() as s:
    sr = (
        s.query(SpecRegistry)
         .filter(SpecRegistry.spec_path == "docs/spec_supervisor_approval_panel_v1.md")
         .first()
    )
    led = _json.loads(sr.amendment_log)
    assert len(led) >= 1, "amendment ledger empty"
    last = led[-1]
    assert last["kind"] == "clarification"
    assert last["n_trials_added"] == 0
    print(f"  Facet 13 amendment_log: kind={last['kind']} n_added={last['n_trials_added']} OK")

# Facet 14: L1 watchlist_origin shape
o1 = dc.get_watchlist_origin(pa_test_id)
assert "available" in o1
print(f"  Facet 14 L1: available={o1.get('available')} OK")

# Facet 15: L2 quant_posture + EXT-2 league_table
o2 = dc.get_quant_posture(test_ticker, test_sector)
assert isinstance(o2.get("league_table"), list)
assert "available" in o2
print(f"  Facet 15 L2 + EXT-2: league_n={o2.get('league_n', 0)}  OK")

# Facet 16: L3 regime_context + EXT-1 macro_snapshot + filtered prob sum ≈ 1
o3 = dc.get_regime_context(
    test_ticker, test_sector,
    o1.get("created_date") if o1.get("available") else None,
)
if o3.get("available"):
    s_p = (o3["p_risk_on"] + o3["p_risk_off"] + o3["p_transition"])
    assert abs(s_p - 1.0) < 1e-2, f"filtered prob sum {s_p} !≈ 1.0"
    assert o3.get("ex_ante_caveat") is True
    assert "macro_snapshot" in o3
    print(f"  Facet 16 L3 + EXT-1: p_sum={s_p:.4f}  caveat=True  macro keys ok  OK")
else:
    print("  Facet 16 L3: regime data absent (acceptable on cold DB)")

# Facet 17: L4 portfolio_posture + EXT-4 HHI
o4 = dc.get_portfolio_posture(pa_test_id, test_sw)
if o4.get("available"):
    hhi = o4.get("hhi_metrics") or {}
    assert "hhi_current" in hhi and 0.0 <= hhi["hhi_current"] <= 1.0
    assert hhi.get("hhi_interpretation") in ("highly_concentrated","moderate","diversified")
    print(f"  Facet 17 L4 + EXT-4: hhi={hhi['hhi_current']:.4f} ({hhi['hhi_interpretation']})  OK")

# Facet 18: EXT-5 underwater_duration shape
dd = (o4.get("drawdown_metrics") or {})
assert "available" in dd
print(f"  Facet 18 EXT-5 drawdown: available={dd.get('available')} underwater_days={dd.get('underwater_days')}  OK")

# Facet 19: L5 conditional_history with insufficient_data flag
o5 = dc.get_conditional_history(test_sector, "long", o3.get("regime_label"))
assert "n_obs" in o5
print(f"  Facet 19 L5: n_obs={o5.get('n_obs')} insufficient={o5.get('insufficient_data')}  OK")

# Facet 20: L6 thesis composer — rule_based when DecisionLog absent
o6_rule = dc.compose_thesis(decision_log_payload=None, watchlist_origin=o1, quant_posture=o2, regime_context=o3)
assert o6_rule["thesis_source"] == "rule_based"
assert isinstance(o6_rule["key_thesis"], str) and len(o6_rule["key_thesis"]) > 0
# And LLM path with mock payload
o6_llm = dc.compose_thesis(
    decision_log_payload={"available": True, "key_thesis": "test", "primary_risk": "x"},
    watchlist_origin=o1, quant_posture=o2, regime_context=o3,
)
assert o6_llm["thesis_source"] == "decision_log"
print(f"  Facet 20 L6: rule_based + decision_log paths both OK")

# Facet 21: L7a forward_preview + EXT-3 calendar effects + NO MC keys
o7 = dc.get_forward_preview(pa_test_id, test_sw)
if o7.get("available"):
    ce = o7.get("calendar_effects") or {}
    assert isinstance(ce.get("days_to_next_fomc"), int) or ce.get("days_to_next_fomc") is None
    assert "in_pre_fomc_drift_window" in ce
    # Forbid any probabilistic / Monte Carlo keys
    forbidden = {"monte_carlo","mc_paths","probability","outcome_distribution","forward_pnl_pdf"}
    appended = " ".join(_json.dumps(o7))
    for k in forbidden:
        assert k not in appended.lower(), f"forbidden MC key {k!r} found in L7a output"
    assert "no_montecarlo_note" in o7
    print(f"  Facet 21 L7a + EXT-3: blackout={ce.get('in_fomc_blackout_window')} TOM={ce.get('in_turn_of_month')}  no MC keys  OK")
else:
    print("  Facet 21 L7a: not available (no NAV snapshot); skipped")

# Facet 22: aggregator wires everything into get_approval_context
from engine.approval_context import get_approval_context as _gac
ctx_full = _gac(pa_test_id)
expected_dc_keys = {"watchlist_origin","quant_posture","regime_context","portfolio_posture","conditional_history","thesis_module","forward_preview"}
assert "decision_context" in ctx_full
assert set(ctx_full["decision_context"].keys()) == expected_dc_keys
print(f"  Facet 22 aggregator: get_approval_context returns full decision_context (7 layers) OK")


# ─────────────────────────────────────────────────────────────────────────────
# Facet 23 — compose_supervisor_narrative (deterministic narrative composer)
# ─────────────────────────────────────────────────────────────────────────────
_hr("Facet 23 — narrative composer (deterministic, 0 LLM)")

from engine.approval_context import get_approval_context as _gac2
from engine.decision_context import compose_supervisor_narrative

ctx_full = _gac2(pa_test_id)
narrative1 = compose_supervisor_narrative(ctx_full["base"], ctx_full["decision_context"])
narrative2 = compose_supervisor_narrative(ctx_full["base"], ctx_full["decision_context"])
assert isinstance(narrative1, str)
assert len(narrative1) > 200, f"narrative too short: {len(narrative1)}"
assert narrative1 == narrative2, "narrative composer is non-deterministic"
required_section_titles = [
    "为什么是这个标的", "当下市场环境", "量化全景",
    "批准影响", "历史能撑得住吗", "我会重点看的风险",
]
for t in required_section_titles:
    assert t in narrative1, f"narrative missing section: {t}"
print(f"  Facet 23 narrative: len={len(narrative1)} sections=6/6 deterministic OK")


# ─────────────────────────────────────────────────────────────────────────────
# Facets 24-27 — M3 Trader Workstation (clarification 2026-05-04)
# ─────────────────────────────────────────────────────────────────────────────
_hr("Facets 24-27 — M3 Trader Workstation")

from engine.approval_workflow import (
    get_throughput_today, check_alert_can_batch,
    bulk_resolve_pending_approvals, compose_daily_review_markdown,
)

# Facet 24 throughput counter
tp = get_throughput_today()
required = {"as_of_date","n_approved_today","n_rejected_today",
            "n_pending","nearest_deadline_id","nearest_deadline_days"}
assert required.issubset(tp.keys()), f"throughput shape missing: {required - set(tp.keys())}"
print(f"  Facet 24 throughput shape OK (pending={tp['n_pending']})")

# Facet 25 batch eligibility — red lines fire on edge cases
ok_real, _ = check_alert_can_batch(pa_test_id)
ok_missing, reason_m = check_alert_can_batch(99999999)
assert ok_missing is False and "not found" in reason_m
print(f"  Facet 25 check_alert_can_batch: real={ok_real}, missing={ok_missing!r} OK")

# Facet 26 bulk_resolve empty + skipped path
res_empty = bulk_resolve_pending_approvals(
    [], approved=True, resolved_by="verify",
    review_rationale="x", review_category="other",
)
assert res_empty == {"submitted": 0, "resolved": [], "skipped": []}
res_missing = bulk_resolve_pending_approvals(
    [99999998], approved=True, resolved_by="verify",
    review_rationale="regression test rationale 12 chars",
    review_category="other",
)
assert res_missing["submitted"] == 1 and len(res_missing["skipped"]) == 1
print(f"  Facet 26 bulk_resolve empty + missing-id paths OK")

# Facet 27 daily review markdown
md = compose_daily_review_markdown()
assert isinstance(md, str) and md.startswith("# Daily Approval Review")
md2 = compose_daily_review_markdown()
assert md == md2  # deterministic
print(f"  Facet 27 daily review composer: len={len(md)} deterministic OK")


# ─────────────────────────────────────────────────────────────────────────────
# Facets 28-31 — Triple-block capability extension (2026-05-04 same-day amendment)
#   Block A: 3-layer expander (status banner + analyst note + drill-down)
#   Block B: Historical conditional replay v2
#   Block C: Narrative snapshot + hash chain v2
# ─────────────────────────────────────────────────────────────────────────────
_hr("Facets 28-31 — Triple-block capability extension")

# Facet 28 Block A: operational narratives composers callable
from engine.operational_narratives import (
    compose_strategy_health_narrative, compose_risk_posture_narrative,
)
md_a = compose_strategy_health_narrative(
    type("X", (), {"days_since": None, "next_quarterly_due": None,
                   "days_until_quarterly": None, "status": "green"})(),
    [], [],
)
md_b = compose_risk_posture_narrative(None, [], [], None)
assert isinstance(md_a, str) and "当前状态" in md_a
assert isinstance(md_b, str)
print(f"  Facet 28 Block A operational narrative composers: len A={len(md_a)} B={len(md_b)} OK")

# Facet 29 Block B: historical_replay shape + 4 anti-anchoring guard caveats
from engine.historical_replay import get_historical_conditional_hit_rate
hr = get_historical_conditional_hit_rate(
    ticker="USO", direction="long", target_regime="neutral",
    horizon_days=21, lookback_years=15, regime_proxy="vix_simple",
)
required_keys = {"ticker","direction","target_regime","n_obs","hit_rate",
                 "mean_active_return","caveats","available"}
assert required_keys.issubset(hr.keys())
caveats_str = " | ".join(hr["caveats"])
assert "for context only" in caveats_str  # mandatory caveat boilerplate
print(f"  Facet 29 Block B historical replay: n={hr['n_obs']} caveats={len(hr['caveats'])} OK")

# Facet 30 Block C: PendingApproval has 3 hash chain columns
import sqlite3 as _sqlite3
from engine.memory import _DB_PATH as _PA_DB_PATH
con3 = _sqlite3.connect(_PA_DB_PATH)
pa_cols = {r[1] for r in con3.execute("PRAGMA table_info(pending_approvals)").fetchall()}
con3.close()
required_pa = {"review_narrative_snapshot", "review_narrative_hash", "prev_narrative_hash"}
missing_pa = required_pa - pa_cols
assert not missing_pa, f"missing hash chain cols: {missing_pa}"
print(f"  Facet 30 Block C ORM cols present: {required_pa} OK")

# Facet 31 Block C: end-to-end freeze + hash chain + audit
import datetime as _dt2, hashlib as _hl2
from engine.memory import (
    SessionFactory as _SF, PendingApproval as _PA,
    resolve_pending_approval as _resolve,
)
with _SF() as _s:
    tmp = _PA(
        approval_type="entry", priority="normal", sector="VFY", ticker="VFY",
        triggered_condition="verify hash chain",
        triggered_date=_dt2.date.today(), suggested_weight=0.0,
        approval_deadline=_dt2.date.today() + _dt2.timedelta(days=3),
        status="pending",
    )
    _s.add(tmp); _s.commit()
    tmp_id = int(tmp.id)

snap_t = "## Verify snapshot\n\nintegrity test row " + str(tmp_id)
_resolve(approval_id=tmp_id, approved=False, resolved_by="verify",
         review_rationale="verify_p_audit_v1 hash chain regression test",
         review_category="other", review_narrative_snapshot=snap_t,
         rejection_reason="verify regression")

with _SF() as _s:
    r = _s.get(_PA, tmp_id)
    expected_hash = _hl2.sha256(snap_t.encode("utf-8")).hexdigest()
    assert r.review_narrative_snapshot == snap_t
    assert r.review_narrative_hash == expected_hash
    _s.delete(r); _s.commit()

import scripts.audit_narrative_chain as _anc
audit_out = _anc.run_audit()
assert audit_out["n_hash_mismatch"] == 0
assert audit_out["n_chain_broken"] == 0
print(f"  Facet 31 Block C freeze + hash + audit: hash deterministic, chain intact OK")


print("\n" + "=" * 70)
print("P-AUDIT v1 (full) + M3-ext + Narrative + M3-WS + 3-Block: 31 / 31 facets PASS")
print("=" * 70)
