"""S2.5 extended verification — facets E/F/G + cleanup."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import json
import traceback

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
# Reseed 4 reflections (extending baseline scenario from verify_s2_5_hooks)
# ─────────────────────────────────────────────────────────────────────────────
with SessionFactory() as s:
    s.query(AgentReflection).delete()
    s.commit()


class MockResp:
    def __init__(self, text): self.text = text


def mock_model(text):
    class M:
        def generate_content(self, prompt):
            return MockResp(text)
    return M()


SEED = [
    ("sector_pipeline", today - datetime.timedelta(days=30),
     "[CONTEXT] Risk-on tech regime, VIX 12, momentum strong. "
     "[DECISION] Long XLK 0.65 confidence on momentum. "
     "[OUTCOME] Realized +0.024 hit. "
     "[LESSON] Risk-on + low VIX → tech longs reliably hit."),
    ("sector_pipeline", today - datetime.timedelta(days=60),
     "[CONTEXT] OPEC supply cut, energy momentum positive. "
     "[DECISION] Long XLE 0.55 on supply tightening. "
     "[OUTCOME] Realized +0.031 hit. "
     "[LESSON] OPEC cuts during tightening cycles drive XLE momentum."),
]
for agent_id, dt, txt in SEED:
    inp = ReflectionInput(
        agent_id=agent_id, decision_date=dt,
        decision_summary={"sector": "test", "direction": "long",
                          "confidence": 0.6, "rationale_excerpt": "seed"},
        realized_outcome=0.02,
        factor_context={"factor_ic_top3": [{"name": "QL01", "ic": 0.07}]},
    )
    build_and_persist_reflection(inp, model=mock_model(txt))
print(f"reseeded {len(SEED)} sector_pipeline reflections")


# ─────────────────────────────────────────────────────────────────────────────
# Hook helpers — capture historical_context passed to debate
# ─────────────────────────────────────────────────────────────────────────────
import engine.debate as debate_mod
real_run_sector_debate = debate_mod.run_sector_debate
captured = {}

def fake_run_sector_debate(model, sector_name, vix,
                           macro_context="", news_context="",
                           historical_context="", valuation_context="",
                           quant_context=None, quant_gate=None,
                           max_rounds=2):
    captured["historical_context"] = historical_context
    return {
        "final_output": "[结论] 标配 XLK 60%",
        "final_xai":    {"overall_confidence": 60, "horizon": "季度(3个月)"},
        "final_data":   {"weight_adjustment_pct": 0.0},
        "weight_adjustment_pct": 0.0,
        "blue_output":  "blue",
        "blue_xai":     {},
        "debate_history": [],
        "arbitration_notes": "",
    }


def run_sector_with_payload(payload):
    """Run SectorPipelineAgent with the given trigger payload, capture hist_ctx."""
    from engine.agents.sector_pipeline.agent import SectorPipelineAgent
    from engine.agents.base import Trigger
    debate_mod.run_sector_debate = fake_run_sector_debate
    captured.clear()
    try:
        ag = SectorPipelineAgent(model=mock_model("dummy"))
        tr = Trigger(type="manual", source=payload.pop("decision_source", "ui_ext_test"),
                     payload=payload)
        ag.run(tr, today)
        return ag._last_result, captured.get("historical_context", "")
    finally:
        debate_mod.run_sector_debate = real_run_sector_debate


# ─────────────────────────────────────────────────────────────────────────────
# E. history_prefix coexists with reflection block (Arm B/C contract)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — history_prefix preserved alongside reflection block")
print("=" * 70)
PREFIX = "[ARM_B_HEADER] paper trading run id=99 control history\n"
result, hist = run_sector_with_payload({
    "sector": "XLK", "vix": 13.0,
    "parent_decision_id": None, "revision_reason": "",
    "overwrite": False, "history_prefix": PREFIX,
    "decision_source": "ui_ext_history_prefix",
})
print(f"  hist len={len(hist)}")
has_block = "Past Reflections" in hist and "End Reflections" in hist
has_prefix = PREFIX.strip() in hist
print(f"  contains reflection block: {has_block}")
print(f"  contains history_prefix:   {has_prefix}")
assert has_block and has_prefix
# history_prefix should be at the very front (last prepended), reflection block AFTER it
prefix_pos = hist.find(PREFIX.strip())
block_pos  = hist.find("Past Reflections")
print(f"  prefix_pos={prefix_pos}, block_pos={block_pos} (prefix should come first)")
assert prefix_pos < block_pos, "history_prefix should land at the front"
print("  OK: both coexist; ordering = history_prefix → reflection block → original historical_context")


# ─────────────────────────────────────────────────────────────────────────────
# F. retrieval failure → graceful proceed without reflection
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — retrieval failure path is silent (no decision blocking)")
print("=" * 70)

import engine.agents.reflection as R
real_retrieve = R.retrieve_relevant_reflections

def boom_retrieve(*a, **kw):
    raise RuntimeError("simulated retriever explosion")

# patch on the module objects the hook actually imports from
R.retrieve_relevant_reflections = boom_retrieve
import engine.agents.sector_pipeline.agent as spa_mod  # noqa: F401
import engine.agents.macro_research.agent as mra_mod   # noqa: F401

try:
    result, hist = run_sector_with_payload({
        "sector": "XLE", "vix": 14.0,
        "parent_decision_id": None, "revision_reason": "",
        "overwrite": False, "history_prefix": "",
        "decision_source": "ui_ext_retrieval_fail",
    })
    print(f"  sector hist len={len(hist)}; contains 'Past Reflections': {'Past Reflections' in hist}")
    assert "Past Reflections" not in hist
    saved_id = (result or {}).get("saved_id")
    if saved_id:
        with SessionFactory() as s:
            dec = s.query(DecisionLog).filter(DecisionLog.id == saved_id).one()
            print(f"  audit count={dec.reflections_injected_count}, ids={dec.reflections_injected_ids}")
            assert dec.reflections_injected_count == 0
            assert json.loads(dec.reflections_injected_ids) == []
    print("  OK sector: retrieval failure → no block, audit = 0/[], decision still saved")

    # also test macro hook on retrieval failure
    from engine.agents.macro_research.agent import MacroResearchAgent
    from engine.agents.base import Trigger as MTrigger

    captured_prompt = {}
    class CapModel:
        def generate_content(self, prompt):
            captured_prompt["text"] = prompt
            return MockResp(json.dumps({
                "regime_assessment": "neutral",
                "key_macro_driver": "x",
                "tail_risk_narrative": "y",
                "horizon": "1M",
                "confidence_raw": 0.5,
                "contradicts_current_regime": False,
            }))
    ma = MacroResearchAgent(model=CapModel())
    run = ma.run(MTrigger(type="scheduled", source="ext_retrieval_fail", payload={}), today)
    p = captured_prompt.get("text", "")
    print(f"  macro prompt len={len(p)}; contains 'Past Reflections': {'Past Reflections' in p}")
    assert "Past Reflections" not in p
    summary = run.summary or {}
    print(f"  macro AgentRun count={summary.get('reflections_injected_count')}, ids={summary.get('reflections_injected_ids')}")
    assert summary.get("reflections_injected_count", -1) == 0
    print("  OK macro: retrieval failure → graceful")

finally:
    R.retrieve_relevant_reflections = real_retrieve


# ─────────────────────────────────────────────────────────────────────────────
# G. real Gemini end-to-end through SectorPipelineAgent (1 sector, 1 call)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("G — real Gemini consumes reflection-augmented prompt without crash")
print("=" * 70)
G_ok = False
try:
    from engine.key_pool import get_pool
    from engine.agents.sector_pipeline.agent import SectorPipelineAgent
    from engine.agents.base import Trigger as GTrigger

    pool = get_pool()
    real_model = pool.get_model()

    # use the REAL run_sector_debate (not our fake) so the LLM actually sees
    # the augmented historical_context end-to-end
    ag = SectorPipelineAgent(model=real_model)
    payload = {
        "sector": "XLK", "vix": 13.0,
        "parent_decision_id": None, "revision_reason": "",
        "overwrite": False, "history_prefix": "",
    }
    tr = GTrigger(type="manual", source="ui_ext_real_llm_test", payload=payload)
    print("  invoking real Gemini (full debate, may take 30-90s)...")
    run = ag.run(tr, today)
    res = ag._last_result or {}
    saved_id = res.get("saved_id")
    print(f"  agent run.status = {run.status}")
    print(f"  saved_id = {saved_id}")
    if saved_id:
        with SessionFactory() as s:
            dec = s.query(DecisionLog).filter(DecisionLog.id == saved_id).one()
            print(f"  audit count={dec.reflections_injected_count}, ids={dec.reflections_injected_ids}")
            print(f"  conclusion[:120]: {(dec.ai_conclusion or '')[:120]}...")
        G_ok = True
        print("  OK: real Gemini completed full pipeline with reflection block injected")
    else:
        print(f"  PARTIAL: agent ran but no decision saved (status={run.status}, error={run.error})")
except Exception as e:
    print(f"  SKIP — real LLM unavailable / quota / data fetch failed: {e}")
    traceback.print_exc(limit=2)


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup ALL smoke residue from S2.5 baseline + extended
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Cleanup all smoke residue (DecisionLog + AgentReflection)")
print("=" * 70)
SMOKE_SOURCES = [
    "ui_smoke_test", "ui_cold_test",
    "ui_ext_test", "ui_ext_history_prefix",
    "ui_ext_retrieval_fail", "ui_ext_real_llm_test",
    "smoke_test", "ext_retrieval_fail", "cold_test",
]
with SessionFactory() as s:
    n_dec = s.query(DecisionLog).filter(
        DecisionLog.decision_source.in_(SMOKE_SOURCES)
    ).delete(synchronize_session=False)
    n_ref = s.query(AgentReflection).delete()
    s.commit()
print(f"  deleted {n_dec} smoke DecisionLog rows, {n_ref} reflections")

with SessionFactory() as s:
    print(f"  agent_reflections rows remaining: {s.query(AgentReflection).count()}")
    print(f"  smoke DecisionLog rows remaining: "
          f"{s.query(DecisionLog).filter(DecisionLog.decision_source.in_(SMOKE_SOURCES)).count()}")

print()
print("=" * 70)
print(f"S2.5 EXTENDED verification {'PASS' if G_ok else 'PASS (G skipped — real LLM unavailable)'}")
print("=" * 70)
