"""S2.4 RAG retriever verification.

Seeds 8 mock-LLM reflections covering 3 semantic clusters + 2 agents +
old/new dates, then exercises retrieve_relevant_reflections across:
 (1) semantic ordering — query close to a cluster picks that cluster
 (2) agent isolation — sector_pipeline query never returns macro_research rows
 (3) lookback cutoff — old reflections excluded
 (4) exclude_ids — caller can skip self
 (5) empty / no-candidate paths
 (6) latency p50 / p95
 (7) prompt formatting
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import json
import time

from engine.agents.reflection import (
    DEFAULT_LOOKBACK_MONTHS,
    ReflectionInput,
    build_and_persist_reflection,
    build_reflection_query,
    format_reflections_for_prompt,
    retrieve_relevant_reflections,
)
from engine.memory import AgentReflection, SessionFactory, init_db

init_db()

# Sanity: start clean (the user already truncated; this is belt-and-suspenders)
with SessionFactory() as s:
    n_before = s.query(AgentReflection).count()
    print(f"agent_reflections rows before seed: {n_before}")
    if n_before > 0:
        print("  (already seeded; clearing)")
        s.query(AgentReflection).delete()
        s.commit()

# ─────────────────────────────────────────────────────────────────────────────
# Seed: 8 reflections across 3 semantic clusters + 2 agents + recency
# ─────────────────────────────────────────────────────────────────────────────


class MockResp:
    def __init__(self, text):
        self.text = text


def mock_model(text):
    class M:
        def generate_content(self, prompt):
            return MockResp(text)
    return M()


# Cluster A: Tech sector / momentum / low-vol
TXT_A1 = (
    "[CONTEXT] Risk-on regime, VIX 12, QL01 IC peaked at +0.08 last month. "
    "[DECISION] Long XLK at 0.65 confidence on momentum + low-vol stack. "
    "[OUTCOME] Realized +0.024 next month, in line with thesis. "
    "[LESSON] When VIX<15 and momentum IC>0.05, tech-tilted longs hit reliably."
)
TXT_A2 = (
    "[CONTEXT] Continued risk-on, VIX 13, semiconductor momentum strong. "
    "[DECISION] Increased XLK exposure to 70% confidence on tech beta. "
    "[OUTCOME] Realized +0.018, slightly under expectation. "
    "[LESSON] Tech momentum persists in low-vol regime; size up when QL01 IC stays positive."
)

# Cluster B: Energy / OPEC / commodity
TXT_B1 = (
    "[CONTEXT] OPEC announced production cuts; XLE momentum positive 3 months. "
    "[DECISION] Long XLE at 0.55 confidence on supply-side narrative. "
    "[OUTCOME] Realized +0.031, above expectation. "
    "[LESSON] Supply-side OPEC narratives drive XLE momentum reliably during tightening cycles."
)
TXT_B2 = (
    "[CONTEXT] OPEC compliance weakening, China demand soft, oil prices wobbling. "
    "[DECISION] Short XLE at 0.6 confidence on demand-side fundamentals. "
    "[OUTCOME] Realized -0.018, hit. "
    "[LESSON] When OPEC compliance erodes and China PMI prints contractionary, energy shorts work."
)

# Cluster C: Defensive / rate-sensitive / financials
TXT_C1 = (
    "[CONTEXT] Fed pivoting dovish, yield curve un-inverting, banks near book value. "
    "[DECISION] Long XLF at 0.5 confidence on rate-cycle reversal. "
    "[OUTCOME] Realized +0.012, partial. "
    "[LESSON] Bank longs into Fed dovish pivot need NIM context, not just curve."
)
TXT_C2 = (
    "[CONTEXT] Treasury yields rising fast, banks underperforming. "
    "[DECISION] Short XLF at 0.6 confidence on duration mismatch. "
    "[OUTCOME] Realized -0.015. "
    "[LESSON] Rate-shock scenarios reward bank shorts; check duration gap data."
)

# Macro_research agent (different agent_id — must NEVER show up for sector_pipeline queries)
TXT_M1 = (
    "[CONTEXT] CPI surprise to upside +0.3pp, growth revision down. "
    "[DECISION] Forecast risk-off regime with 0.7 confidence over 1M horizon. "
    "[OUTCOME] Realized regime indeed flipped to risk-off, drawdown 4%. "
    "[LESSON] Stagflationary CPI prints reliably precede regime flips at 1M."
)

# OLD reflection (>18 months ago — should be filtered out by default cutoff)
TXT_OLD = (
    "[CONTEXT] 2024 era regime, low-vol, tech tilt, momentum IC +0.06. "
    "[DECISION] Long XLK with 0.5 confidence on standard momentum signal. "
    "[OUTCOME] Realized +0.015 partial. "
    "[LESSON] Pre-2025 momentum factor was weaker than current QL01 BAB variant."
)

today = datetime.date(2026, 4, 30)
seed_specs = [
    ("sector_pipeline", today - datetime.timedelta(days=30),  TXT_A1, "XLK", "long"),
    ("sector_pipeline", today - datetime.timedelta(days=60),  TXT_A2, "XLK", "long"),
    ("sector_pipeline", today - datetime.timedelta(days=90),  TXT_B1, "XLE", "long"),
    ("sector_pipeline", today - datetime.timedelta(days=120), TXT_B2, "XLE", "short"),
    ("sector_pipeline", today - datetime.timedelta(days=150), TXT_C1, "XLF", "long"),
    ("sector_pipeline", today - datetime.timedelta(days=180), TXT_C2, "XLF", "short"),
    ("macro_research",  today - datetime.timedelta(days=30),  TXT_M1, "macro", "long"),
    ("sector_pipeline", today - datetime.timedelta(days=800), TXT_OLD, "XLK", "long"),  # OLD
]

print()
print("=" * 70)
print("Seeding 8 reflections...")
print("=" * 70)
seed_ids = []
for agent_id, dt, txt, sector, direction in seed_specs:
    inp = ReflectionInput(
        agent_id=agent_id,
        decision_date=dt,
        decision_summary={
            "sector": sector,
            "direction": direction,
            "confidence": 0.6,
            "rationale_excerpt": "seeded for retrieval test",
        },
        realized_outcome=+0.020 if direction == "long" else -0.020,
        factor_context={
            "factor_ic_top3": [{"name": "QL01", "ic": 0.08, "icir": 0.42}]
        },
    )
    rid = build_and_persist_reflection(inp, model=mock_model(txt))
    seed_ids.append(rid)
    print(f"  id={rid} agent={agent_id} date={dt} sector={sector}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: semantic ordering — XLK tech query should pull TXT_A1/A2 first
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Test 1 — semantic ordering (XLK / momentum query)")
print("=" * 70)
q = build_reflection_query(
    decision_summary={
        "sector": "XLK",
        "direction": "long",
        "rationale_excerpt": "low-vol regime, momentum IC positive, tech tilt",
    },
    factor_context={"factor_ic_top3": [{"name": "QL01"}]},
)
print(f"  query: {q}")
results = retrieve_relevant_reflections(
    agent_id="sector_pipeline", query_text=q, k=3, as_of=today
)
print("  top-3 returned:")
for r in results:
    snippet = r.reflection_text[:80].replace("\n", " ")
    print(f"    id={r.id} date={r.decision_date}: {snippet}...")
top_texts = [r.reflection_text[:30] for r in results]
assert any("Risk-on" in r.reflection_text or "tech momentum" in r.reflection_text.lower()
           for r in results[:2]), "expected tech-cluster reflections in top-2"
print("  OK: top results are tech-cluster (semantic match)")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: semantic ordering — energy query should pull TXT_B1/B2 first
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Test 2 — semantic ordering (XLE / OPEC query)")
print("=" * 70)
q2 = "sector=XLE | direction=long | OPEC supply-side cuts oil price"
results2 = retrieve_relevant_reflections(
    agent_id="sector_pipeline", query_text=q2, k=3, as_of=today
)
for r in results2:
    snippet = r.reflection_text[:80].replace("\n", " ")
    print(f"    id={r.id} date={r.decision_date}: {snippet}...")
assert any("OPEC" in r.reflection_text for r in results2[:2]), "expected energy cluster top-2"
print("  OK: top results are energy-cluster")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: agent isolation — sector_pipeline query NEVER returns macro_research
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Test 3 — agent isolation")
print("=" * 70)
results3 = retrieve_relevant_reflections(
    agent_id="sector_pipeline",
    query_text="CPI surprise stagflation regime flip risk-off",
    k=10,
    as_of=today,
)
agents_returned = {r.agent_id for r in results3}
print(f"  agents in results: {agents_returned}")
assert agents_returned == {"sector_pipeline"}, f"leaked: {agents_returned}"
print("  OK: macro_research row filtered out even though semantically closest")

# query macro_research directly to confirm it CAN find that row
results3b = retrieve_relevant_reflections(
    agent_id="macro_research",
    query_text="CPI stagflation regime",
    k=5,
    as_of=today,
)
assert len(results3b) == 1 and "CPI" in results3b[0].reflection_text
print("  OK: macro_research query returns its own (1) row")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: lookback cutoff — OLD reflection (800 days back) excluded by default
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Test 4 — lookback cutoff (default 18 months)")
print("=" * 70)
q4 = "tech XLK momentum low-vol"
results4_default = retrieve_relevant_reflections(
    agent_id="sector_pipeline", query_text=q4, k=10, as_of=today
)
ids_default = [r.id for r in results4_default]
old_id = seed_ids[7]
assert old_id not in ids_default, f"OLD row {old_id} leaked via default cutoff"
print(f"  default lookback ({DEFAULT_LOOKBACK_MONTHS}mo): OLD id={old_id} excluded")

# Wider cutoff: now OLD should appear
results4_wide = retrieve_relevant_reflections(
    agent_id="sector_pipeline",
    query_text=q4,
    k=10,
    lookback_months=60,
    as_of=today,
)
ids_wide = [r.id for r in results4_wide]
assert old_id in ids_wide, f"OLD row {old_id} missing under wide cutoff"
print(f"  wide   lookback (60mo): OLD id={old_id} present")
print("  OK: cutoff active and tunable")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: exclude_ids
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Test 5 — exclude_ids")
print("=" * 70)
top1 = retrieve_relevant_reflections(
    agent_id="sector_pipeline", query_text=q4, k=1, as_of=today
)
assert len(top1) == 1
excluded = top1[0].id
top1_after = retrieve_relevant_reflections(
    agent_id="sector_pipeline",
    query_text=q4,
    k=1,
    exclude_ids=[excluded],
    as_of=today,
)
assert len(top1_after) == 1 and top1_after[0].id != excluded
print(f"  before exclude top1 id={excluded}")
print(f"  after  exclude top1 id={top1_after[0].id}")
print("  OK: exclude_ids honored")

# ─────────────────────────────────────────────────────────────────────────────
# Test 6: empty / no-candidate paths
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Test 6 — empty/no-candidate paths")
print("=" * 70)
assert retrieve_relevant_reflections(agent_id="sector_pipeline", query_text="", as_of=today) == []
print("  empty query → []")
assert retrieve_relevant_reflections(agent_id="nonexistent", query_text="anything", as_of=today) == []
print("  unknown agent → []")

# ─────────────────────────────────────────────────────────────────────────────
# Test 7: latency p50 / p95 (spec verdict requires p95 < 100ms)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Test 7 — retrieval latency")
print("=" * 70)
# Warm up (first call loads ST model + tokenizer)
_ = retrieve_relevant_reflections(
    agent_id="sector_pipeline", query_text="warmup", k=5, as_of=today
)

N = 30
queries = [
    "low-vol momentum tech XLK long",
    "OPEC supply cut energy XLE",
    "Fed dovish pivot banks XLF",
    "stagflation CPI regime flip",
    "QL01 BAB low-vol top factor",
] * (N // 5)

times_ms = []
for q in queries:
    t0 = time.time()
    _ = retrieve_relevant_reflections(
        agent_id="sector_pipeline", query_text=q, k=5, as_of=today
    )
    times_ms.append((time.time() - t0) * 1000)
times_ms.sort()
p50 = times_ms[len(times_ms) // 2]
p95 = times_ms[int(len(times_ms) * 0.95)]
print(f"  N={N} queries  candidates=7 (sector_pipeline within cutoff)")
print(f"  p50={p50:.1f} ms   p95={p95:.1f} ms   max={max(times_ms):.1f} ms")
print(f"  spec verdict gate: p95 < 100 ms → {'PASS' if p95 < 100 else 'FAIL'}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 8: prompt formatting
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Test 8 — format_reflections_for_prompt")
print("=" * 70)
top3 = retrieve_relevant_reflections(
    agent_id="sector_pipeline", query_text=q, k=3, as_of=today
)
formatted = format_reflections_for_prompt(top3, agent_id="sector_pipeline")
print(formatted)
assert "Past Reflections" in formatted
assert "End Reflections" in formatted
assert "[1]" in formatted and "[2]" in formatted and "[3]" in formatted
assert "Outcome:" in formatted
print()
print("  OK: prompt block well-formed")

# Empty case
empty_block = format_reflections_for_prompt([], agent_id="sector_pipeline")
assert empty_block == ""
print("  OK: empty list → empty block (no dead prompt section)")

print()
print("=" * 70)
print("S2.4 verification PASS" if p95 < 100 else "S2.4 verification PARTIAL (latency)")
print("=" * 70)
