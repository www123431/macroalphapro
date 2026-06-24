"""S2.8 end-to-end closed-loop smoke test.

Exercise the full S2 loop in one script:
  Step 1: insert a sector_pipeline DecisionLog with active_return filled
  Step 2: invoke generate_reflections_for_pending — should write reflection #1
  Step 3: invoke SectorPipelineAgent on a new decision (fake debate, mock LLM)
          — should RETRIEVE reflection #1 + INJECT into historical_context
            + persist new DecisionLog with reflections_injected_count=1
  Step 4: verify the audit trail closes (DB rows + cross-references)

This is the spec §6 backfill loop + §5 retrieval hook running back-to-back —
the closed loop that makes the "self-reflecting agent" claim concrete.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import json

from engine.agents.reflection import (
    generate_reflections_for_pending,
)
from engine.memory import (
    AgentReflection,
    DecisionLog,
    SessionFactory,
    init_db,
    save_decision,
)

init_db()
today = datetime.date(2026, 4, 30)


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup any prior smoke residue
# ─────────────────────────────────────────────────────────────────────────────
SMOKE_SOURCE = "ui_s2_8_e2e_test"
print("Cleaning prior smoke residue...")
with SessionFactory() as s:
    n_dec = s.query(DecisionLog).filter(
        DecisionLog.decision_source == SMOKE_SOURCE
    ).delete(synchronize_session=False)
    n_ref = s.query(AgentReflection).delete()
    s.commit()
print(f"  cleared {n_dec} smoke decisions, {n_ref} reflections")


class MockResp:
    def __init__(self, text): self.text = text


GOOD_REFLECTION = (
    "[CONTEXT] Risk-on regime, VIX 13, momentum stack on tech. "
    "[DECISION] Long XLK at 0.65 confidence based on momentum + low-vol thesis. "
    "[OUTCOME] Realized active return +2.40% next month, in line with thesis. "
    "[LESSON] When VIX<15 and momentum IC>0.05, tech-tilted longs hit reliably."
)


def good_model():
    class M:
        def generate_content(self, prompt):
            return MockResp(GOOD_REFLECTION)
    return M()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — insert a DecisionLog representing a past LLM debate decision
#          whose realized outcome is already filled
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Step 1 — seed past DecisionLog with active_return = +0.024")
print("=" * 70)
past_id = save_decision(
    tab_type="sector",
    ai_conclusion="过去决策：建议超配 XLK 60% based on momentum + low-vol",
    vix_level=13.0,
    sector_name="XLK",
    ticker="XLK",
    news_summary="historical seed",
    macro_regime="低波动/牛市",
    horizon="季度(3个月)",
    confidence_score=65,
    decision_date=today - datetime.timedelta(days=30),
    decision_source=SMOKE_SOURCE,
)
with SessionFactory() as s:
    d = s.query(DecisionLog).filter(DecisionLog.id == past_id).one()
    d.direction = "超配"
    d.active_return = 0.024
    d.quant_p_noise = 0.18
    d.quant_test_r2 = 0.05
    s.commit()
print(f"  past decision id={past_id}, active_return=+0.024")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — backfill: generate reflection memo for that pending decision
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Step 2 — generate_reflections_for_pending → write reflection #1")
print("=" * 70)
backfill_summary = generate_reflections_for_pending(
    as_of=today, model=good_model(),
)
print(f"  backfill summary: {backfill_summary}")
assert backfill_summary["processed"] == 1
assert backfill_summary["failed"] == 0

with SessionFactory() as s:
    refl = s.query(AgentReflection).filter(
        AgentReflection.decision_ref_id == past_id
    ).one()
    refl_id = refl.id
    print(f"  persisted AgentReflection id={refl_id} for decision {past_id}")
    print(f"    agent_id={refl.agent_id}")
    print(f"    hit_flag={refl.hit_flag} (long +0.024 → expect hit)")
    print(f"    embedding model={refl.embedding_model}, dim={len(json.loads(refl.embedding))}")
    assert refl.hit_flag == "hit"
    assert refl.agent_id == "sector_pipeline"
    assert refl.decision_ref_id == past_id


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — simulate a NEW SectorPipelineAgent run that should retrieve
#          reflection #1 and prepend it into the debate's historical_context
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Step 3 — new SectorPipelineAgent decision retrieves reflection #1")
print("=" * 70)

import engine.debate as debate_mod
real_run_sector_debate = debate_mod.run_sector_debate
captured = {}

def fake_run_sector_debate(model, sector_name, vix,
                           macro_context="", news_context="",
                           historical_context="", valuation_context="",
                           quant_context=None, quant_gate=None,
                           max_rounds=2):
    captured["historical_context"] = historical_context
    captured["sector_name"] = sector_name
    return {
        "final_output":  "[结论] 超配 XLK 60% based on momentum stack + reflection memory",
        "final_xai":     {"overall_confidence": 65, "horizon": "季度(3个月)"},
        "final_data":    {"weight_adjustment_pct": 5.0},
        "weight_adjustment_pct": 5.0,
        "blue_output":   "blue analysis test",
        "blue_xai":      {},
        "debate_history":[],
        "arbitration_notes": "",
    }


from engine.agents.sector_pipeline.agent import SectorPipelineAgent
from engine.agents.base import Trigger

debate_mod.run_sector_debate = fake_run_sector_debate
try:
    agent = SectorPipelineAgent(model=good_model())  # mock LLM (only used by reflection layer; debate is faked)
    trigger = Trigger(
        type="manual",
        source=SMOKE_SOURCE,
        payload={"sector": "XLK", "vix": 13.0,
                 "parent_decision_id": None,
                 "revision_reason": "", "overwrite": False, "history_prefix": ""},
    )
    agent.run(trigger, today)
    new_dec_result = agent._last_result or {}
finally:
    debate_mod.run_sector_debate = real_run_sector_debate

new_dec_id = new_dec_result.get("saved_id")
print(f"  new DecisionLog id={new_dec_id}")
hist_ctx = captured.get("historical_context", "")
print(f"  historical_context length: {len(hist_ctx)}")
has_block = "Past Reflections" in hist_ctx and "End Reflections" in hist_ctx
mentions_reflection_id = f"id={refl_id}" not in hist_ctx  # reflection text included, but id literal not
print(f"  contains reflection block: {has_block}")
contains_lesson_text = "tech-tilted longs hit reliably" in hist_ctx
print(f"  contains lesson text from past reflection: {contains_lesson_text}")
assert has_block
assert contains_lesson_text, "the new debate did not see the prior LESSON"
print("  OK: new decision saw the past reflection's lesson in its prompt")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — audit trail closes: new DecisionLog has reflections_injected_*
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Step 4 — audit trail close (new DecisionLog -> AgentReflection)")
print("=" * 70)
with SessionFactory() as s:
    new_dec = s.query(DecisionLog).filter(DecisionLog.id == new_dec_id).one()
    cnt = new_dec.reflections_injected_count
    ids = json.loads(new_dec.reflections_injected_ids or "[]")
    print(f"  new DecisionLog.reflections_injected_count = {cnt}")
    print(f"  new DecisionLog.reflections_injected_ids   = {ids}")
    assert cnt == 1
    assert ids == [refl_id]

    # symmetric check: AgentReflection #1 still references past_id
    refl_check = s.query(AgentReflection).filter(
        AgentReflection.id == refl_id
    ).one()
    assert refl_check.decision_ref_id == past_id
    print(f"  AgentReflection.decision_ref_id = {refl_check.decision_ref_id} "
          f"(stable forward link to past decision)")

print("  OK: closed loop —")
print(f"    past DecisionLog id={past_id}")
print(f"      |-> AgentReflection id={refl_id} (decision_ref_id → {past_id})")
print(f"        |-> retrieved by NEW DecisionLog id={new_dec_id} "
      f"(reflections_injected_ids = [{refl_id}])")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Cleanup")
print("=" * 70)
with SessionFactory() as s:
    n_dec = s.query(DecisionLog).filter(
        DecisionLog.decision_source == SMOKE_SOURCE
    ).delete(synchronize_session=False)
    n_ref = s.query(AgentReflection).delete()
    s.commit()
print(f"  deleted {n_dec} smoke decisions, {n_ref} reflections")
with SessionFactory() as s:
    print(f"  agent_reflections rows remaining: {s.query(AgentReflection).count()}")
    print(f"  smoke decision rows remaining:    "
          f"{s.query(DecisionLog).filter(DecisionLog.decision_source == SMOKE_SOURCE).count()}")

print()
print("=" * 70)
print("S2.8 E2E verification PASS — full S2 closed loop verified")
print("=" * 70)
