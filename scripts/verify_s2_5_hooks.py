"""S2.5 hook verification — sector_pipeline + macro_research integration.

Strategy: monkeypatch the LLM model on each agent so we never burn real
quota during verification. Validate that:
  A. SectorPipelineAgent — historical_context contains the reflection block
     that retrieve_relevant_reflections produced
  B. MacroResearchAgent — _build_prompt receives reflection_block and the
     final prompt contains it
  C. Cold start (empty pool) — no exception, both agents proceed without
     reflections
  D. DecisionLog.reflections_injected_count / ids are written
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import json

from engine.agents.reflection import (
    ReflectionInput,
    build_and_persist_reflection,
)
from engine.memory import (
    AgentReflection,
    DecisionLog,
    SessionFactory,
    init_db,
)

init_db()
today = datetime.date(2026, 4, 30)


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup any stale rows so we measure the new path cleanly.
# ─────────────────────────────────────────────────────────────────────────────
with SessionFactory() as s:
    n_ref = s.query(AgentReflection).delete()
    s.commit()
print(f"Cleared {n_ref} stale reflections")


class MockResp:
    def __init__(self, text): self.text = text


def mock_model(text):
    class M:
        def generate_content(self, prompt):
            return MockResp(text)
    return M()


# ─────────────────────────────────────────────────────────────────────────────
# Seed 4 reflections covering tech / energy clusters (sector_pipeline + macro)
# ─────────────────────────────────────────────────────────────────────────────
SEED = [
    ("sector_pipeline", today - datetime.timedelta(days=30),
     "[CONTEXT] Risk-on regime, VIX 12, momentum strong tech XLK. "
     "[DECISION] Long XLK 0.65 confidence based on momentum stack. "
     "[OUTCOME] Realized +0.024 next month. "
     "[LESSON] Risk-on + low VIX + positive QL01 IC reliably supports tech longs."),
    ("sector_pipeline", today - datetime.timedelta(days=60),
     "[CONTEXT] OPEC supply cut announced, energy XLE momentum 3M positive. "
     "[DECISION] Long XLE 0.55 confidence on supply tightening. "
     "[OUTCOME] Realized +0.031, hit. "
     "[LESSON] Supply-side OPEC cuts during tightening cycles drive XLE momentum reliably."),
    ("macro_research", today - datetime.timedelta(days=30),
     "[CONTEXT] CPI surprise upside, growth slowing. "
     "[DECISION] Forecast risk-off regime 0.7 confidence over 1M. "
     "[OUTCOME] Regime indeed flipped risk-off. "
     "[LESSON] Stagflation prints precede regime flips at 1M horizon."),
    ("macro_research", today - datetime.timedelta(days=90),
     "[CONTEXT] Fed dovish pivot, yields softening, growth stable. "
     "[DECISION] Forecast risk-on regime 0.6 confidence over 3M. "
     "[OUTCOME] Realized regime stayed risk-on, market rallied. "
     "[LESSON] Fed dovish pivot + stable growth → risk-on persistence at 3M."),
]
seed_ids = []
for agent_id, dt, txt in SEED:
    inp = ReflectionInput(
        agent_id=agent_id,
        decision_date=dt,
        decision_summary={"sector": "test", "direction": "long",
                          "confidence": 0.6, "rationale_excerpt": "seeded"},
        realized_outcome=+0.020,
        factor_context={"factor_ic_top3": [{"name": "QL01", "ic": 0.07}]},
    )
    seed_ids.append(build_and_persist_reflection(inp, model=mock_model(txt)))
print(f"Seeded {len(seed_ids)} reflections (ids={seed_ids})")


# ─────────────────────────────────────────────────────────────────────────────
# A. SectorPipelineAgent — capture historical_context passed to debate
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("A — SectorPipelineAgent passes reflection block into debate")
print("=" * 70)

from engine.agents.sector_pipeline.agent import SectorPipelineAgent
from engine.agents.base import Trigger
import engine.debate as debate_mod

captured = {}
real_run_sector_debate = debate_mod.run_sector_debate

def fake_run_sector_debate(model, sector_name, vix,
                           macro_context="", news_context="",
                           historical_context="", valuation_context="",
                           quant_context=None, quant_gate=None,
                           max_rounds=2):
    captured["historical_context"] = historical_context
    captured["sector_name"] = sector_name
    return {
        "final_output":          "测试用：[结论] 标配 XLK 60%",
        "final_xai":             {"overall_confidence": 60, "horizon": "季度(3个月)"},
        "final_data":            {"weight_adjustment_pct": 0.0},
        "weight_adjustment_pct": 0.0,
        "blue_output":           "blue test",
        "blue_xai":              {},
        "debate_history":        [],
        "arbitration_notes":     "fake",
    }

debate_mod.run_sector_debate = fake_run_sector_debate
try:
    agent = SectorPipelineAgent(model=mock_model("dummy"))
    trigger = Trigger(
        type="manual",
        source="ui_smoke_test",
        payload={"sector": "XLK", "vix": 13.0,
                 "parent_decision_id": None,
                 "revision_reason": "", "overwrite": False, "history_prefix": ""},
    )
    run = agent.run(trigger, today)
finally:
    debate_mod.run_sector_debate = real_run_sector_debate

hist_ctx = captured.get("historical_context", "")
print(f"  historical_context length: {len(hist_ctx)}")
has_block = "Past Reflections" in hist_ctx and "End Reflections" in hist_ctx
print(f"  contains reflection block: {has_block}")
assert has_block, "reflection block missing from historical_context"
# Should be sector_pipeline scoped — never include macro_research text
assert "Stagflation" not in hist_ctx, "macro_research reflection leaked"
assert ("XLK" in hist_ctx) or ("XLE" in hist_ctx), "no sector_pipeline reflection injected"
print("  OK: reflection block prepended; agent isolation respected")

saved_id = (agent._last_result or {}).get("saved_id")
print(f"  saved DecisionLog id: {saved_id}")

# Verify D — audit columns populated
if saved_id:
    with SessionFactory() as s:
        dec = s.query(DecisionLog).filter(DecisionLog.id == saved_id).one()
        cnt = dec.reflections_injected_count
        ids = json.loads(dec.reflections_injected_ids) if dec.reflections_injected_ids else None
        print(f"  D — reflections_injected_count = {cnt}")
        print(f"  D — reflections_injected_ids   = {ids}")
        assert cnt is not None and cnt >= 1
        assert isinstance(ids, list) and len(ids) == cnt
        # ids should be subset of seeded sector_pipeline rows (1, 2)
        sector_ids = {seed_ids[0], seed_ids[1]}
        assert all(i in sector_ids for i in ids), \
               f"injected ids leak beyond sector_pipeline pool: {ids}"
        print("  OK: D audit columns populated and consistent")


# ─────────────────────────────────────────────────────────────────────────────
# B. MacroResearchAgent — capture prompt sent to LLM
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — MacroResearchAgent injects reflection block into prompt")
print("=" * 70)

from engine.agents.macro_research.agent import MacroResearchAgent

captured_prompt = {}

class CapturingModel:
    def generate_content(self, prompt):
        captured_prompt["text"] = prompt
        return MockResp(json.dumps({
            "regime_assessment": "risk-off",
            "key_macro_driver": "test driver",
            "tail_risk_narrative": "test narrative",
            "horizon": "1M",
            "confidence_raw": 0.6,
            "contradicts_current_regime": False,
        }))

macro_agent = MacroResearchAgent(model=CapturingModel())
trigger = Trigger(type="scheduled", source="smoke_test", payload={})
run = macro_agent.run(trigger, today)

prompt = captured_prompt.get("text", "")
print(f"  prompt len={len(prompt)}")
has_block = "Past Reflections" in prompt and "End Reflections" in prompt
print(f"  contains reflection block: {has_block}")
assert has_block, "reflection block missing from macro prompt"
# macro_research scoped — must NOT include sector_pipeline text
assert "OPEC" not in prompt, "sector_pipeline reflection leaked into macro prompt"
assert "Stagflation" in prompt or "Fed dovish" in prompt, \
       "no macro_research reflection injected"
print("  OK: reflection block injected; agent isolation respected")

# AgentRun summary should also carry the audit info
summary = run.summary or {}
print(f"  AgentRun summary reflections_injected_count = {summary.get('reflections_injected_count')}")
print(f"  AgentRun summary reflections_injected_ids   = {summary.get('reflections_injected_ids')}")
assert summary.get("reflections_injected_count", 0) >= 1
print("  OK: macro AgentRun summary carries audit info")


# ─────────────────────────────────────────────────────────────────────────────
# C. Cold start — empty pool, both hooks degrade gracefully
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — cold-start (empty reflection pool) graceful no-op")
print("=" * 70)
with SessionFactory() as s:
    s.query(AgentReflection).delete()
    s.commit()
print("  pool cleared")

captured.clear()
debate_mod.run_sector_debate = fake_run_sector_debate
try:
    agent2 = SectorPipelineAgent(model=mock_model("dummy"))
    trigger2 = Trigger(
        type="manual", source="ui_cold_test",
        payload={"sector": "XLE", "vix": 14.0,
                 "parent_decision_id": None,
                 "revision_reason": "", "overwrite": False, "history_prefix": ""},
    )
    run2 = agent2.run(trigger2, today)
finally:
    debate_mod.run_sector_debate = real_run_sector_debate

hist2 = captured.get("historical_context", "")
print(f"  cold sector_pipeline historical_context len: {len(hist2)}")
assert "Past Reflections" not in hist2, \
       "empty pool should not produce a reflection block"
print("  OK: empty pool → no reflection block in historical_context")

saved_id_2 = (agent2._last_result or {}).get("saved_id")
if saved_id_2:
    with SessionFactory() as s:
        dec = s.query(DecisionLog).filter(DecisionLog.id == saved_id_2).one()
        print(f"  cold DecisionLog id={saved_id_2}, count={dec.reflections_injected_count}, "
              f"ids={dec.reflections_injected_ids}")
        # count should be 0 (was set to 0 by update_reflections_injected)
        assert dec.reflections_injected_count == 0
        assert json.loads(dec.reflections_injected_ids) == []
        print("  OK: cold-start audit columns set to 0 / [] (not NULL)")

captured_prompt.clear()
macro2 = MacroResearchAgent(model=CapturingModel())
run3 = macro2.run(Trigger(type="scheduled", source="cold_test", payload={}), today)
prompt2 = captured_prompt.get("text", "")
assert "Past Reflections" not in prompt2
print("  OK: cold-start macro prompt has no reflection block")
sm3 = run3.summary or {}
assert sm3.get("reflections_injected_count", -1) == 0
print(f"  cold macro AgentRun count = {sm3.get('reflections_injected_count')}, ids={sm3.get('reflections_injected_ids')}")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup the smoke-test residue we created
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Cleanup smoke-test residue")
print("=" * 70)
with SessionFactory() as s:
    n_ref_left = s.query(AgentReflection).delete()
    # smoke test created two DecisionLog rows via fake_run_sector_debate
    # leave them — they're no harm, but mark as smoke-test by setting
    # decision_source already = ui_smoke_test / ui_cold_test (auditable)
    s.commit()
    n_dec_smoke = s.query(DecisionLog).filter(
        DecisionLog.decision_source.in_(["ui_smoke_test", "ui_cold_test"])
    ).count()
print(f"  cleared {n_ref_left} reflections; {n_dec_smoke} smoke DecisionLog rows kept (auditable)")

print()
print("=" * 70)
print("S2.5 verification PASS")
print("=" * 70)
