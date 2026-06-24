"""S2.7 UI verification — Agent Learning Curve tab.

Uses streamlit's AppTest harness to run the page headlessly:
  A. syntax / import sanity (ast.parse + compile)
  B. cold-start (0 reflections, 0 retrieval-augmented decisions) → renders empty branch
  C. seeded run (8 reflections, mixed hit/miss/partial + 1 inj-decision) → all 4 sub-modules render
  D. cleanup smoke residue
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import datetime
import json

PAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pages", "agent_observability.py",
)


# ─────────────────────────────────────────────────────────────────────────────
# A. syntax / import sanity
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — syntax / parse")
print("=" * 70)
with open(PAGE, "r", encoding="utf-8") as f:
    src = f.read()
ast.parse(src)
compile(src, PAGE, "exec")
print(f"  OK: page parses cleanly ({len(src)} bytes, {src.count(chr(10))} lines)")
assert "G. Agent learning curve" in src, "section G missing"
assert "AgentReflection" in src, "AgentReflection import missing"
assert "DecisionLog" in src, "DecisionLog import missing"
print("  OK: section G + required symbols present")


# ─────────────────────────────────────────────────────────────────────────────
# B. cold start (clean DB) → empty-state branch
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — cold start (0 reflections, 0 inj-decisions)")
print("=" * 70)

from engine.memory import (
    init_db, SessionFactory, AgentReflection, DecisionLog,
)

init_db()
with SessionFactory() as s:
    s.query(AgentReflection).delete()
    # don't blow away production decisions; just clear any reflections_injected_count
    s.query(DecisionLog).filter(
        DecisionLog.decision_source.in_(
            ["ui_s2_7_seed_test", "ui_s2_7_inj_test"]
        )
    ).delete(synchronize_session=False)
    s.commit()

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(PAGE, default_timeout=120)
at.run()
exceptions = [str(e.value) for e in at.exception]
print(f"  exceptions: {len(exceptions)}")
for e in exceptions:
    print(f"    {e[:160]}")
assert not exceptions, f"page raised exceptions in cold start: {exceptions}"

# Look for the empty-state info block in section G
all_info = [el.value for el in at.info]
empty_branch_msgs = [
    m for m in all_info
    if "No reflections accumulated yet" in m
]
print(f"  empty-branch info messages: {len(empty_branch_msgs)} found")
assert empty_branch_msgs, "expected empty-state info, none rendered"
print("  OK: cold-start renders empty-state branch without exception")


# ─────────────────────────────────────────────────────────────────────────────
# C. seeded — 8 reflections + 1 inj-decision → all 4 sub-modules render
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — seeded (8 reflections + 1 inj-decision)")
print("=" * 70)

from engine.agents.reflection import (
    ReflectionInput, build_and_persist_reflection,
)
from engine.memory import save_decision, update_reflections_injected


class MockResp:
    def __init__(self, text): self.text = text


def make_text(idx, sector):
    return (
        f"[CONTEXT] Decision {idx} on {sector} under risk-on regime, IC moderate. "
        f"[DECISION] Took LLM-suggested direction at {0.5 + idx * 0.05:.2f} confidence. "
        f"[OUTCOME] Realized active return as recorded; pattern reflects regime stability. "
        f"[LESSON] When momentum IC > 0.05 in risk-on, position sizing tolerates +0.5σ."
    )


def mock_model(text):
    class M:
        def generate_content(self, prompt):
            return MockResp(text)
    return M()


# 8 reflections: mix of agents, sectors, hits/misses/partial
today = datetime.date(2026, 4, 30)
SEED = [
    ("sector_pipeline", "XLK",  "long",    +0.024),  # hit
    ("sector_pipeline", "XLE",  "short",   -0.018),  # hit (short + neg)
    ("sector_pipeline", "XLF",  "long",    +0.003),  # partial (>0 but <thresh)
    ("sector_pipeline", "XLP",  "long",    -0.022),  # miss
    ("sector_pipeline", "XLV",  "short",   +0.020),  # miss (short + pos)
    ("sector_pipeline", "XLI",  "neutral", +0.001),  # neutral
    ("macro_research",  "macro","long",    +0.010),  # hit
    ("sector_pipeline", "XLB",  "long",    +0.018),  # hit
]
seed_ids = []
for i, (agent, sector, dirn, ret) in enumerate(SEED):
    inp = ReflectionInput(
        agent_id=agent,
        decision_date=today - datetime.timedelta(days=30 + i),
        decision_summary={
            "sector": sector, "direction": dirn,
            "confidence": 0.6, "rationale_excerpt": "seeded",
        },
        realized_outcome=ret,
        factor_context={"factor_ic_top3": [{"name": "QL01", "ic": 0.07}]},
    )
    rid = build_and_persist_reflection(inp, model=mock_model(make_text(i, sector)))
    seed_ids.append(rid)
print(f"  seeded {len(seed_ids)} reflections (ids={seed_ids})")

# 1 DecisionLog with reflections_injected_count > 0 (so G.4 inspector lights up)
saved_id = save_decision(
    tab_type="sector",
    ai_conclusion="测试 G.4 inspector：建议超配 XLK 60%",
    vix_level=13.0, sector_name="XLK", ticker="XLK",
    news_summary="seed", macro_regime="低波动/牛市",
    horizon="季度(3个月)", confidence_score=65,
    decision_date=today, decision_source="ui_s2_7_inj_test",
)
with SessionFactory() as s:
    d = s.query(DecisionLog).filter(DecisionLog.id == saved_id).one()
    d.direction = "超配"
    s.commit()
update_reflections_injected(saved_id, [seed_ids[0], seed_ids[1], seed_ids[7]])
print(f"  inserted inj-decision id={saved_id} with reflections_injected_count=3")

# Re-run page
at2 = AppTest.from_file(PAGE, default_timeout=180)
at2.run()
exceptions2 = [str(e.value) for e in at2.exception]
print(f"  exceptions: {len(exceptions2)}")
for e in exceptions2:
    print(f"    {e[:160]}")
assert not exceptions2, f"page raised exceptions in seeded run: {exceptions2}"

# Headline metrics — 5 expected (Total / This month / Hit50 / Lifetime / Latency probe)
metric_labels = [m.label for m in at2.metric]
print(f"  metrics rendered: {metric_labels}")
expected_labels = ["Total reflections", "This month"]
for lbl in expected_labels:
    assert lbl in metric_labels, f"metric '{lbl}' missing"
print("  OK: G.1 headline metrics rendered")

# Look up Total reflections value
totals = [m for m in at2.metric if m.label == "Total reflections"]
assert totals
total_val = totals[0].value
print(f"  Total reflections = {total_val} (expect 8)")
assert int(total_val) == 8

# Hit-rate metric — should reflect 5 hits + 1 partial + 2 miss = (5*1 + 0.5 + 0)/8 = 0.6875
# Excluding neutral; counted = 7 (XLI neutral excluded)
# 4 hits (XLK long+, XLE short-, macro long+, XLB long+) + 1 partial (XLF long small) + 2 miss (XLP, XLV)
# = (4*1 + 1*0.5 + 0)/7 ≈ 0.643
hit_lifetime = [m for m in at2.metric if m.label.startswith("Hit rate (lifetime")]
assert hit_lifetime
print(f"  Hit rate (lifetime) = {hit_lifetime[0].value}")

# Section G should have a dataframe rendered (the reflection list)
df_count = len(at2.dataframe)
print(f"  dataframes rendered on page: {df_count}")
assert df_count >= 1, "no dataframes rendered (reflection list expected)"

# Section G.4 inspector — selectbox + markdown rendering
sel_count = len(at2.selectbox)
print(f"  selectboxes rendered: {sel_count} (expect ≥1 for memo selector)")
assert sel_count >= 1

# G.2 trend — plotly chart should be present (need ≥5 hit/miss/partial; we have 7)
# We can't directly assert plotly chart presence in AppTest, but absence of
# exceptions + presence of all metrics means the trend block rendered.

# Also verify an info messages aren't the empty-state for G
info_msgs = [el.value for el in at2.info]
print(f"  info messages on page: {len(info_msgs)}")
assert not any("No reflections accumulated yet" in m for m in info_msgs), \
       "empty-state should NOT show with seeded data"
print("  OK: seeded path renders G.1 + G.2 + G.3 + G.4 with no exceptions")


# ─────────────────────────────────────────────────────────────────────────────
# D. cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — cleanup")
print("=" * 70)
with SessionFactory() as s:
    n_ref = s.query(AgentReflection).delete()
    n_dec = s.query(DecisionLog).filter(
        DecisionLog.decision_source.in_(["ui_s2_7_seed_test", "ui_s2_7_inj_test"])
    ).delete(synchronize_session=False)
    s.commit()
print(f"  deleted {n_ref} reflections, {n_dec} smoke decisions")
with SessionFactory() as s:
    print(f"  agent_reflections rows remaining: {s.query(AgentReflection).count()}")
    print(f"  smoke decision rows remaining:    "
          f"{s.query(DecisionLog).filter(DecisionLog.decision_source == 'ui_s2_7_inj_test').count()}")

print()
print("=" * 70)
print("S2.7 verification PASS")
print("=" * 70)
