"""S2.3 reflection generator full verification (6 facets, ad-hoc)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import json
import math
import traceback

from engine.agents.reflection import (
    HIT_THRESHOLD,
    ReflectionInput,
    build_and_persist_reflection,
    compute_embedding,
    compute_hit_flag,
    generate_reflection_text,
    validate_reflection_schema,
)
from engine.memory import AgentReflection, SessionFactory, init_db

init_db()

GOOD = (
    "[CONTEXT] Macro regime risk-on, VIX 12, factor IC QL01 +0.08 a month ago. "
    "[DECISION] Predicted long XLK with conf 0.55 on momentum+low-vol stack. "
    "[OUTCOME] Realized return not yet backfilled at memo time. "
    "[LESSON] When VIX<15 and momentum IC>0.05 confidence ceiling 0.60."
)


class MockResp:
    def __init__(self, text):
        self.text = text


class MockModel:
    def generate_content(self, prompt):
        return MockResp(GOOD)


print("=" * 70)
print("Verification A — real LLM (frozen prompt actually produces 4 sections)")
print("=" * 70)
A_ok = False
try:
    from engine.key_pool import get_pool

    pool = get_pool()
    model = pool.get_model()
    inp = ReflectionInput(
        agent_id="sector_pipeline",
        decision_date=datetime.date(2026, 4, 30),
        decision_summary={
            "sector": "XLK",
            "direction": "long",
            "confidence": 0.65,
            "rationale_excerpt": "momentum + low-vol stack post B++ QL01 IC top",
        },
        realized_outcome=+0.024,
        factor_context={
            "factor_ic_top3": [
                {"name": "QL01", "ic": 0.08, "icir": 0.42},
                {"name": "CL01", "ic": 0.05, "icir": 0.30},
            ],
            "beta_market": 0.92,
            "alpha_residual": 0.005,
            "correlation_to_top_strategy": 0.34,
        },
        prior_reflections=[],
    )
    text = generate_reflection_text(
        decision_summary=inp.decision_summary,
        realized_outcome=inp.realized_outcome,
        factor_context=inp.factor_context,
        prior_reflections=inp.prior_reflections,
        model=model,
    )
    valid = validate_reflection_schema(text)
    print(f"  text len={len(text)}, schema_valid={valid}")
    print("  --- LLM output ---")
    print(text[:600])
    print("  --- end ---")
    if valid:
        rid = build_and_persist_reflection(inp, model=model)
        print(f"  OK persisted id={rid} via real LLM")
        A_ok = True
    else:
        print("  FAIL — schema invalid (caller would skip-and-log per design)")
except Exception as e:
    print(f"  SKIP — real LLM unavailable / quota: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("Verification B — pending state (realized_outcome=None, hit=pending)")
print("=" * 70)
inp_pending = ReflectionInput(
    agent_id="macro_research",
    decision_date=datetime.date(2026, 4, 30),
    decision_summary={
        "sector": "macro",
        "direction": "long",
        "confidence": 0.55,
        "rationale_excerpt": "risk-on regime",
    },
    realized_outcome=None,
    factor_context={"factor_ic_top3": []},
)
rid_b = build_and_persist_reflection(inp_pending, model=MockModel())
with SessionFactory() as s:
    row = s.query(AgentReflection).filter(AgentReflection.id == rid_b).one()
    print(f"  id={row.id} hit={row.hit_flag} (expect pending), realized={row.realized_outcome}")
    assert row.hit_flag == "pending"
    assert row.realized_outcome is None
print("  OK pending row stored")

print()
print("=" * 70)
print("Verification C — LLM garbage triggers RuntimeError (schema invalid)")
print("=" * 70)


class GarbageModel:
    def generate_content(self, prompt):
        return MockResp("hi")


try:
    build_and_persist_reflection(inp_pending, model=GarbageModel())
    print("  FAIL — should have raised")
except RuntimeError as e:
    print(f"  OK raised: {e}")


class WrongOrderModel:
    def generate_content(self, prompt):
        return MockResp(
            "[DECISION] x " + ("y" * 200) + " [CONTEXT] z [OUTCOME] w [LESSON] v"
        )


try:
    build_and_persist_reflection(inp_pending, model=WrongOrderModel())
    print("  FAIL — wrong-order should have raised")
except RuntimeError as e:
    print(f"  OK raised on wrong order: {e}")

print()
print("=" * 70)
print("Verification D — embedding determinism + reasonable cosine")
print("=" * 70)
v1 = compute_embedding("the cat sat on the mat")
v2 = compute_embedding("the cat sat on the mat")
v3 = compute_embedding("completely different text about quantum mechanics")
assert v1 == v2, "same text must produce same vector"
print(f"  OK same-text determinism (v1==v2, len={len(v1)})")


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


sim_self = cos(v1, v2)
sim_diff = cos(v1, v3)
print(f"  cos(same,same)={sim_self:.4f} (expect 1.0)")
print(f"  cos(same,diff)={sim_diff:.4f} (hash fallback ~0; ST would be ~0.05-0.4)")
assert abs(sim_self - 1.0) < 1e-9
print("  NOTE: hash-fallback similarity is not semantic — install sentence-transformers for real retrieval")

print()
print("=" * 70)
print("Verification E — DB index ix_reflection_agent_date present")
print("=" * 70)
from sqlalchemy import inspect

from engine.memory import engine as db_engine

ins = inspect(db_engine)
ix = ins.get_indexes("agent_reflections")
names = [i["name"] for i in ix]
print(f"  indexes on agent_reflections: {names}")
assert "ix_reflection_agent_date" in names
print("  OK composite index present")

print()
print("=" * 70)
print("Verification F — non-ASCII round-trip in factor_context + decision_summary")
print("=" * 70)
inp_zh = ReflectionInput(
    agent_id="sector_pipeline",
    decision_date=datetime.date(2026, 4, 29),
    decision_summary={
        "sector": "XLE",
        "direction": "short",
        "confidence": 0.6,
        "rationale_excerpt": "OPEC 减产预期 + 中国需求疲软",
    },
    realized_outcome=-0.018,
    factor_context={"note": "B++ QL01 IC 仍稳健 0.07; β-neutralized α 持平"},
)
rid_f = build_and_persist_reflection(inp_zh, model=MockModel())
with SessionFactory() as s:
    row = s.query(AgentReflection).filter(AgentReflection.id == rid_f).one()
    ds = json.loads(row.decision_summary)
    fc = json.loads(row.factor_context)
    print(f"  decision_summary[rationale]={ds['rationale_excerpt']}")
    print(f"  factor_context[note]={fc['note']}")
    assert "减产" in ds["rationale_excerpt"]
    assert "稳健" in fc["note"]
    assert row.hit_flag == "hit"  # short + realized -0.018 → hit
print("  OK Chinese characters round-trip + hit_flag correct")

print()
print("=" * 70)
print("Verification G — schema validator boundaries (149/150 + 800/801)")
print("=" * 70)


def make_text(total_target):
    base_overhead = len("[CONTEXT] c [DECISION] d [OUTCOME] o [LESSON] ")
    pad = max(0, total_target - base_overhead)
    return f"[CONTEXT] c [DECISION] d [OUTCOME] o [LESSON] {'x' * pad}"


for n in (149, 150, 800, 801):
    t = make_text(n)
    print(f"  len {len(t):3d}: valid={validate_reflection_schema(t)}")
assert not validate_reflection_schema(make_text(149))
assert validate_reflection_schema(make_text(150))
assert validate_reflection_schema(make_text(800))
assert not validate_reflection_schema(make_text(801))
print("  OK boundary checks")

print()
print("=" * 70)
print("Final: row count summary")
print("=" * 70)
with SessionFactory() as s:
    rows = s.query(AgentReflection).order_by(AgentReflection.id).all()
    for r in rows:
        print(
            f"  id={r.id} agent={r.agent_id:18s} date={r.decision_date} "
            f"hit={(r.hit_flag or ''):8s} len={len(r.reflection_text or '')}"
        )
    print(f"  total={len(rows)} rows")
print()
print(f"A_ok (real LLM): {A_ok}")
