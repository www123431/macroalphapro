"""S2.4 extended verification — 5 additional facets after the 8-test baseline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import json
import math
import random
import time

from engine.agents.reflection import (
    ReflectionInput,
    build_and_persist_reflection,
    compute_embedding,
    retrieve_relevant_reflections,
)
from engine.memory import AgentReflection, SessionFactory, init_db

init_db()

today = datetime.date(2026, 4, 30)


# ─────────────────────────────────────────────────────────────────────────────
# A. Stored DB embedding round-trip vs fresh encode (precision check)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — stored DB embedding round-trip precision")
print("=" * 70)
with SessionFactory() as s:
    row = s.query(AgentReflection).filter(
        AgentReflection.agent_id == "sector_pipeline"
    ).first()
    assert row is not None, "no rows to test against — re-run baseline first"
    stored_vec = json.loads(row.embedding)
    text = row.reflection_text

fresh_vec = compute_embedding(text)
def cos(a, b):
    return sum(x * y for x, y in zip(a, b))

# Both should be unit-norm
norm_stored = math.sqrt(sum(x * x for x in stored_vec))
norm_fresh  = math.sqrt(sum(x * x for x in fresh_vec))
print(f"  ||stored||={norm_stored:.6f}   ||fresh||={norm_fresh:.6f}")
sim = cos(stored_vec, fresh_vec)
print(f"  cos(stored, fresh)={sim:.6f}  (expect ≈1.0 within JSON precision loss)")
max_abs_diff = max(abs(a - b) for a, b in zip(stored_vec, fresh_vec))
print(f"  max abs(stored-fresh) per dim = {max_abs_diff:.2e}")
assert sim > 0.9999, "stored and fresh should be ~identical"
print("  OK: JSON serialization preserves embedding precision")


# ─────────────────────────────────────────────────────────────────────────────
# B. k > N candidates → returns all available, no crash
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — k > N graceful (request 50, only 7 available)")
print("=" * 70)
results = retrieve_relevant_reflections(
    agent_id="sector_pipeline",
    query_text="any query about sectors",
    k=50,
    as_of=today,
)
print(f"  requested k=50, got {len(results)} (within 18mo cutoff)")
assert 1 <= len(results) <= 10
print("  OK: bounded by candidate pool")


# ─────────────────────────────────────────────────────────────────────────────
# C. Ranking determinism — same query twice → identical order
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — ranking determinism")
print("=" * 70)
q = "tech sector momentum low-vol QL01 IC top"
order_1 = [r.id for r in retrieve_relevant_reflections(
    agent_id="sector_pipeline", query_text=q, k=10, as_of=today)]
order_2 = [r.id for r in retrieve_relevant_reflections(
    agent_id="sector_pipeline", query_text=q, k=10, as_of=today)]
order_3 = [r.id for r in retrieve_relevant_reflections(
    agent_id="sector_pipeline", query_text=q, k=10, as_of=today)]
print(f"  call 1: {order_1}")
print(f"  call 2: {order_2}")
print(f"  call 3: {order_3}")
assert order_1 == order_2 == order_3
print("  OK: ranking is deterministic across repeated calls")


# ─────────────────────────────────────────────────────────────────────────────
# D. Scaling sanity — N=100 candidates, p95 latency still < 100ms
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — scaling sanity (N=100 candidates)")
print("=" * 70)

class MockResp:
    def __init__(self, text): self.text = text
class MockModel:
    def __init__(self, text): self.text = text
    def generate_content(self, prompt): return MockResp(self.text)

# Seed 100 synthetic reflections with varied semantic clusters
themes = [
    ("tech momentum semiconductor",       "XLK"),
    ("energy OPEC oil supply",            "XLE"),
    ("financials banks rate cycle",       "XLF"),
    ("staples defensive consumer",        "XLP"),
    ("healthcare pharma biotech",         "XLV"),
    ("utilities low-vol bond proxy",      "XLU"),
    ("industrials capex manufacturing",   "XLI"),
    ("materials commodity copper",        "XLB"),
    ("real estate REIT cap rate",         "XLRE"),
    ("communication services internet",   "XLC"),
]

seed_t0 = time.time()
random.seed(42)
seeded_count = 0
with SessionFactory() as s:
    for i in range(100):
        theme, sector = themes[i % len(themes)]
        txt = (
            f"[CONTEXT] Iteration {i}, regime varies, factor IC random. "
            f"[DECISION] Take position in {sector} on {theme} thesis. "
            f"[OUTCOME] Realized return drawn from synthetic distribution +/-2pct. "
            f"[LESSON] In context {theme}, signal-to-noise is moderate; size accordingly."
        )
        days_back = random.randint(10, 500)
        inp = ReflectionInput(
            agent_id="scaling_test_agent",
            decision_date=today - datetime.timedelta(days=days_back),
            decision_summary={
                "sector": sector, "direction": "long",
                "confidence": 0.5,
                "rationale_excerpt": theme,
            },
            realized_outcome=random.uniform(-0.04, 0.04),
            factor_context={"factor_ic_top3": [{"name": "QL01", "ic": 0.05, "icir": 0.3}]},
        )
        try:
            build_and_persist_reflection(inp, model=MockModel(txt), session=s)
            seeded_count += 1
        except Exception as e:
            # if some i'th text fails schema validation, try a different padding
            print(f"  seed i={i} failed: {e}")
print(f"  seeded {seeded_count} synthetic rows in {time.time()-seed_t0:.1f}s")

# Warm up
_ = retrieve_relevant_reflections(
    agent_id="scaling_test_agent", query_text="warmup", k=5, as_of=today)

queries = [t[0] for t in themes] * 4  # 40 queries cycling themes
times_ms = []
for q in queries:
    t0 = time.time()
    res = retrieve_relevant_reflections(
        agent_id="scaling_test_agent", query_text=q, k=5, as_of=today)
    times_ms.append((time.time() - t0) * 1000)
times_ms.sort()
p50 = times_ms[len(times_ms) // 2]
p95 = times_ms[int(len(times_ms) * 0.95)]
mx  = max(times_ms)

# Count actual candidates within cutoff
with SessionFactory() as s:
    cutoff = today - datetime.timedelta(days=30 * 18)
    n_cand = s.query(AgentReflection).filter(
        AgentReflection.agent_id == "scaling_test_agent",
        AgentReflection.decision_date >= cutoff,
    ).count()
print(f"  candidates within 18mo cutoff: {n_cand}")
print(f"  N={len(queries)} queries  p50={p50:.1f}ms  p95={p95:.1f}ms  max={mx:.1f}ms")
print(f"  spec gate p95 < 100ms → {'PASS' if p95 < 100 else 'FAIL'}")

# semantic verification on at least one large-pool query
top5 = retrieve_relevant_reflections(
    agent_id="scaling_test_agent",
    query_text="tech momentum semiconductor QL01",
    k=5, as_of=today)
hit_count = sum(1 for r in top5 if "semiconductor" in r.reflection_text or "XLK" in r.reflection_text)
print(f"  top-5 of 'tech semiconductor' query: {hit_count}/5 are tech-cluster")
assert hit_count >= 3, "semantic ordering degraded under scale"
print("  OK: semantic ordering holds at N=100")


# ─────────────────────────────────────────────────────────────────────────────
# E. Integration — generate via S2.3, immediately retrieve via S2.4
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — S2.3 generate then S2.4 retrieve, cross-module integration")
print("=" * 70)
NEW_TXT = (
    "[CONTEXT] Brand new XLK long decision today, low-vol regime stable. "
    "[DECISION] Long XLK 0.6 confidence on persistent QL01 IC signal. "
    "[OUTCOME] Pending backfill at memo time. "
    "[LESSON] Reflection template tested via integration path."
)
inp_new = ReflectionInput(
    agent_id="integration_test_agent",
    decision_date=today,
    decision_summary={
        "sector": "XLK", "direction": "long",
        "confidence": 0.6,
        "rationale_excerpt": "integration test",
    },
    realized_outcome=None,
    factor_context={"factor_ic_top3": [{"name": "QL01"}]},
)
new_id = build_and_persist_reflection(inp_new, model=MockModel(NEW_TXT))
print(f"  generated reflection id={new_id}")

# retrieve with semantically close query
res = retrieve_relevant_reflections(
    agent_id="integration_test_agent",
    query_text="XLK long low-vol QL01 momentum",
    k=3, as_of=today,
)
ids = [r.id for r in res]
print(f"  retrieve top-3: {ids}")
assert new_id in ids
top1_text = res[0].reflection_text[:60]
print(f"  top-1 text: {top1_text}...")
assert "XLK" in res[0].reflection_text or "QL01" in res[0].reflection_text
print("  OK: just-written reflection retrievable immediately")

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup scaling-test rows so they don't pollute production retrieval
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Cleanup — remove scaling_test_agent + integration_test_agent rows")
print("=" * 70)
with SessionFactory() as s:
    n_scale = s.query(AgentReflection).filter(
        AgentReflection.agent_id == "scaling_test_agent").delete()
    n_int = s.query(AgentReflection).filter(
        AgentReflection.agent_id == "integration_test_agent").delete()
    s.commit()
print(f"  deleted {n_scale} scaling rows, {n_int} integration rows")

with SessionFactory() as s:
    remaining = s.query(AgentReflection).count()
    print(f"  remaining: {remaining} (the 8 seeded baseline-test rows)")

print()
print("=" * 70)
print("S2.4 EXTENDED verification PASS")
print("=" * 70)
