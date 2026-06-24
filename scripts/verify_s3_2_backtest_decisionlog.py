"""S3.2 backtest n_trials + DecisionLog spec_hash integration verification.

Facets:
  A. DecisionLog.spec_hash column exists in DB
  B. _N_TRIALS_AUDIT now contains "pre_registration" key
  C. EFFECTIVE_N_TRIALS = sqrt(grid) + pre_registration contribution
  D. refresh_effective_n_trials picks up new amendments live
  E. save_decision accepts + persists spec_hash kwarg
  F. SectorPipelineAgent auto-injects spec_hash on its DecisionLog row
  G. cleanup smoke residue
"""
import sys, os, math, json, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import inspect
from engine.memory import (
    init_db, SessionFactory,
    DecisionLog, SpecRegistry,
    save_decision, engine,
)
import engine.backtest as bt
from engine.preregistration import (
    register_spec, amend_spec, compute_pre_registration_n_trials,
    _compute_git_blob_hash, _resolve_to_abs, _normalize_spec_path,
)

init_db()
SMOKE_SOURCE = "ui_s3_2_test"
SMOKE_SPEC_PATH = "docs/spec_s3_2_smoke.md"
SMOKE_SPEC_ABS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), SMOKE_SPEC_PATH,
)


def _cleanup():
    with SessionFactory() as s:
        s.query(DecisionLog).filter(
            DecisionLog.decision_source == SMOKE_SOURCE
        ).delete(synchronize_session=False)
        s.query(SpecRegistry).filter(
            SpecRegistry.spec_path == SMOKE_SPEC_PATH
        ).delete(synchronize_session=False)
        s.commit()
    if os.path.exists(SMOKE_SPEC_ABS):
        os.remove(SMOKE_SPEC_ABS)


_cleanup()  # belt + suspenders


# ─────────────────────────────────────────────────────────────────────────────
# A. column present
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — DecisionLog.spec_hash column")
print("=" * 70)
cols = {c["name"] for c in inspect(engine).get_columns("decision_logs")}
assert "spec_hash" in cols, "spec_hash column missing on decision_logs"
print(f"  OK: column present (total cols={len(cols)})")


# ─────────────────────────────────────────────────────────────────────────────
# B. audit dict has pre_registration
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — _N_TRIALS_AUDIT contains pre_registration key")
print("=" * 70)
bt.refresh_effective_n_trials()
audit = bt._N_TRIALS_AUDIT
print(f"  audit keys: {list(audit.keys())}")
print(f"  audit values: {audit}")
assert "pre_registration" in audit
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# C. arithmetic: effective = sqrt(grid_raw) + pre_reg
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — EFFECTIVE_N_TRIALS arithmetic")
print("=" * 70)
grid_keys = [k for k in audit if k != "pre_registration"]
grid_raw = 1
for k in grid_keys:
    grid_raw *= audit[k]
expected_effective = int(math.ceil(math.sqrt(grid_raw))) + audit["pre_registration"]
print(f"  grid_raw = {grid_raw}")
print(f"  sqrt(grid_raw) = {math.ceil(math.sqrt(grid_raw))}")
print(f"  pre_registration = {audit['pre_registration']}")
print(f"  expected effective = {expected_effective}")
print(f"  actual EFFECTIVE_N_TRIALS = {bt.EFFECTIVE_N_TRIALS}")
assert bt.EFFECTIVE_N_TRIALS == expected_effective
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# D. refresh picks up new amendments
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — refresh_effective_n_trials reflects new amendment")
print("=" * 70)
n_before = compute_pre_registration_n_trials()
print(f"  pre-reg n_trials before: {n_before}")

# Create + register a smoke spec with a forward (non-retro) amendment
with open(SMOKE_SPEC_ABS, "w", encoding="utf-8") as f:
    f.write("# Smoke S3.2 spec\n\nNW t >= 1.5\n")
register_spec(SMOKE_SPEC_ABS, retro=False)
# amendment: threshold_tweak adds 1 trial
with open(SMOKE_SPEC_ABS, "w", encoding="utf-8") as f:
    f.write("# Smoke S3.2 spec v2\n\nNW t >= 1.8\n")
amend_spec(SMOKE_SPEC_ABS, kind="threshold_tweak",
           reason="smoke amendment to test n_trials liveness")

n_after = compute_pre_registration_n_trials()
print(f"  pre-reg n_trials after register+amend: {n_after}")
delta = n_after - n_before
# Expected: register +1 (forward), amend threshold_tweak +1 = +2
assert delta == 2, f"expected +2, got +{delta}"
print(f"  OK: +{delta} (register +1, threshold_tweak +1)")

# Refresh and check effective updates
new_eff, new_audit = bt.refresh_effective_n_trials()
print(f"  refreshed EFFECTIVE_N_TRIALS = {new_eff}")
assert new_audit["pre_registration"] == n_after
print("  OK: refresh propagates to module constants")


# ─────────────────────────────────────────────────────────────────────────────
# E. save_decision accepts + persists spec_hash
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — save_decision persists spec_hash kwarg")
print("=" * 70)
test_hash = _compute_git_blob_hash(SMOKE_SPEC_ABS)
print(f"  computed git-blob hash: {test_hash[:12]}...")

saved_id = save_decision(
    tab_type="sector",
    ai_conclusion="测试 spec_hash 注入",
    vix_level=14.0,
    sector_name="XLK",
    ticker="XLK",
    news_summary="smoke",
    macro_regime="低波动/牛市",
    horizon="季度(3个月)",
    confidence_score=60,
    decision_date=datetime.date(2026, 4, 30),
    decision_source=SMOKE_SOURCE,
    spec_hash=test_hash,
)
print(f"  saved DecisionLog id={saved_id}")
with SessionFactory() as s:
    d = s.query(DecisionLog).filter(DecisionLog.id == saved_id).one()
    print(f"  read-back spec_hash: {d.spec_hash[:12] if d.spec_hash else None}...")
    assert d.spec_hash == test_hash
print("  OK: spec_hash round-trip")


# ─────────────────────────────────────────────────────────────────────────────
# F. SectorPipelineAgent auto-injects spec_hash
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — SectorPipelineAgent auto-injects sector_pipeline_unification spec_hash")
print("=" * 70)
import engine.debate as debate_mod
from engine.agents.sector_pipeline.agent import SectorPipelineAgent
from engine.agents.base import Trigger
from engine.agents.reflection import compute_embedding  # warmup ST model


class MockResp:
    def __init__(self, text): self.text = text


def fake_run_sector_debate(model, sector_name, vix,
                           macro_context="", news_context="",
                           historical_context="", valuation_context="",
                           quant_context=None, quant_gate=None,
                           max_rounds=2):
    return {
        "final_output": "[结论] 标配 XLK 60%",
        "final_xai":    {"overall_confidence": 60, "horizon": "季度(3个月)"},
        "final_data":   {"weight_adjustment_pct": 0.0},
        "weight_adjustment_pct": 0.0,
        "blue_output":  "blue test",
        "blue_xai":     {},
        "debate_history": [],
        "arbitration_notes": "",
    }


real_debate = debate_mod.run_sector_debate
debate_mod.run_sector_debate = fake_run_sector_debate
try:
    class M:
        def generate_content(self, prompt):
            return MockResp("dummy")
    agent = SectorPipelineAgent(model=M())
    # Use a DIFFERENT (sector, date) than facet E to avoid save_decision dedup.
    tr = Trigger(
        type="manual", source=SMOKE_SOURCE,
        payload={"sector": "XLE", "vix": 13.0,
                 "parent_decision_id": None,
                 "revision_reason": "", "overwrite": False, "history_prefix": ""},
    )
    agent.run(tr, datetime.date(2026, 4, 29))
    res = agent._last_result or {}
finally:
    debate_mod.run_sector_debate = real_debate

new_dec_id = res.get("saved_id")
expected_hash = _compute_git_blob_hash(
    _resolve_to_abs("docs/spec_sector_pipeline_unification.md")
)
print(f"  agent saved id={new_dec_id}")
print(f"  expected spec_hash (sector_pipeline_unification): {expected_hash[:12]}...")
with SessionFactory() as s:
    d = s.query(DecisionLog).filter(DecisionLog.id == new_dec_id).one()
    print(f"  actual spec_hash:                                {d.spec_hash[:12] if d.spec_hash else None}...")
    assert d.spec_hash == expected_hash
print("  OK: SectorPipelineAgent auto-injected the canonical spec_hash")


# ─────────────────────────────────────────────────────────────────────────────
# G. cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("G — cleanup smoke residue")
print("=" * 70)
_cleanup()
with SessionFactory() as s:
    n_smoke = s.query(DecisionLog).filter(
        DecisionLog.decision_source == SMOKE_SOURCE
    ).count()
    n_sr = s.query(SpecRegistry).filter(
        SpecRegistry.spec_path == SMOKE_SPEC_PATH
    ).count()
print(f"  smoke decision_logs left: {n_smoke}")
print(f"  smoke spec_registry left: {n_sr}")
print(f"  smoke spec file present:  {os.path.exists(SMOKE_SPEC_ABS)}")
assert n_smoke == 0 and n_sr == 0 and not os.path.exists(SMOKE_SPEC_ABS)

# Refresh once more so EFFECTIVE_N_TRIALS reflects the cleaned-up state
final_eff, final_audit = bt.refresh_effective_n_trials()
print(f"  post-cleanup EFFECTIVE_N_TRIALS = {final_eff}, "
      f"pre_registration = {final_audit['pre_registration']}")
print("  OK")

print()
print("=" * 70)
print("S3.2 verification PASS")
print("=" * 70)
