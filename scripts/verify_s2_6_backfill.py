"""S2.6 backfill verification — 7 facets.

A. happy path: 3 pending DecisionLog rows → 3 reflections persisted, decision_ref_id linked
B. NOT EXISTS filter: rerun does NOT regenerate (idempotent)
C. cold start: 0 candidates → graceful no-op
D. active_return is None → row skipped (filter correctness)
E. tab_type != 'sector' → skipped (only sector_pipeline rows in scope)
F. daily cap enforced (cap=2 → only 2 written even with 3 pending)
G. LLM failure path: per-row exception counted as `failed`, not raised
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
class MockResp:
    def __init__(self, text): self.text = text


GOOD = (
    "[CONTEXT] Sector decision under risk-on regime, factor IC moderate. "
    "[DECISION] Took the LLM-suggested direction at the proposed confidence. "
    "[OUTCOME] Realized active return as recorded on the decision row. "
    "[LESSON] Combine factor IC with regime gates before committing to size."
)


def good_model():
    class M:
        def generate_content(self, prompt):
            return MockResp(GOOD)
    return M()


def garbage_model():
    class M:
        def generate_content(self, prompt):
            return MockResp("hi")  # too short → schema fail
    return M()


def make_test_decision(sector, direction, active_return, tab_type="sector",
                       decision_source="ui_s2_6_test"):
    """Insert a synthetic DecisionLog row that S2.6 should pick up."""
    saved_id = save_decision(
        tab_type=tab_type,
        ai_conclusion=f"测试结论：建议{direction} {sector}",
        vix_level=13.0,
        sector_name=sector,
        ticker=sector,
        news_summary="test",
        macro_regime="低波动/牛市",
        horizon="季度(3个月)",
        confidence_score=60,
        decision_date=today - datetime.timedelta(days=30),
        decision_source=decision_source,
    )
    # then attach active_return so it qualifies
    with SessionFactory() as s:
        d = s.query(DecisionLog).filter(DecisionLog.id == saved_id).one()
        d.active_return = active_return
        d.direction = direction
        s.commit()
    return saved_id


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup any prior S2.6 residue + reflections
# ─────────────────────────────────────────────────────────────────────────────
print("Cleaning prior smoke residue...")
with SessionFactory() as s:
    n_dec = s.query(DecisionLog).filter(
        DecisionLog.decision_source == "ui_s2_6_test"
    ).delete(synchronize_session=False)
    n_ref = s.query(AgentReflection).delete()
    s.commit()
print(f"  cleared {n_dec} decisions, {n_ref} reflections")


# ─────────────────────────────────────────────────────────────────────────────
# A. happy path — 3 pending → 3 persisted
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("A — happy path: 3 pending DecisionLog rows")
print("=" * 70)
ids_A = [
    make_test_decision("XLK", "超配", +0.024),
    make_test_decision("XLE", "低配", -0.018),
    make_test_decision("XLF", "标配", +0.005),
]
print(f"  inserted decisions: {ids_A}")

s = generate_reflections_for_pending(as_of=today, model=good_model())
print(f"  summary: {s}")
assert s["processed"] == 3, f"expected 3 processed, got {s['processed']}"
assert s["failed"] == 0
assert s["candidates"] == 3

# verify decision_ref_id linkage + hit_flag rules
with SessionFactory() as ss:
    rows = ss.query(AgentReflection).filter(
        AgentReflection.decision_ref_id.in_(ids_A)
    ).all()
    print(f"  persisted reflections: {len(rows)}")
    by_id = {r.decision_ref_id: r for r in rows}
    # XLK: 超配=long, +0.024 → hit
    assert by_id[ids_A[0]].hit_flag == "hit", by_id[ids_A[0]].hit_flag
    # XLE: 低配=short, -0.018 → hit (short + negative)
    assert by_id[ids_A[1]].hit_flag == "hit", by_id[ids_A[1]].hit_flag
    # XLF: 标配=neutral → neutral
    assert by_id[ids_A[2]].hit_flag == "neutral", by_id[ids_A[2]].hit_flag
    print("  OK: 3 reflections persisted with correct hit_flag")
    print(f"    XLK: hit={by_id[ids_A[0]].hit_flag} (long +0.024)")
    print(f"    XLE: hit={by_id[ids_A[1]].hit_flag} (short -0.018)")
    print(f"    XLF: hit={by_id[ids_A[2]].hit_flag} (neutral)")


# ─────────────────────────────────────────────────────────────────────────────
# B. idempotency — re-run should NOT regenerate
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — idempotency (NOT EXISTS filter)")
print("=" * 70)
s2 = generate_reflections_for_pending(as_of=today, model=good_model())
print(f"  re-run summary: {s2}")
assert s2["processed"] == 0, "rerun should not regenerate existing reflections"
assert s2["candidates"] == 0
print("  OK: no duplicates regenerated")


# ─────────────────────────────────────────────────────────────────────────────
# C. cold start — 0 candidates
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — cold start (clean DB)")
print("=" * 70)
with SessionFactory() as ss:
    ss.query(AgentReflection).delete()
    ss.query(DecisionLog).filter(
        DecisionLog.decision_source == "ui_s2_6_test"
    ).delete(synchronize_session=False)
    ss.commit()

s3 = generate_reflections_for_pending(as_of=today, model=good_model())
print(f"  summary: {s3}")
assert s3["processed"] == 0
assert s3["candidates"] == 0
assert s3["failed"] == 0
print("  OK: cold start no-op")


# ─────────────────────────────────────────────────────────────────────────────
# D. active_return = None → row should be skipped
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — active_return None is filtered out")
print("=" * 70)
filled_id = make_test_decision("XLK", "超配", +0.02)
unfilled_id = save_decision(
    tab_type="sector", ai_conclusion="pending realized",
    vix_level=13.0, sector_name="XLE", ticker="XLE",
    news_summary="test", macro_regime="低波动/牛市",
    horizon="季度(3个月)", confidence_score=55,
    decision_date=today - datetime.timedelta(days=30),
    decision_source="ui_s2_6_test",
)
# explicitly leave active_return None on unfilled_id
print(f"  filled id={filled_id}, unfilled id={unfilled_id}")

s4 = generate_reflections_for_pending(as_of=today, model=good_model())
print(f"  summary: {s4}")
assert s4["processed"] == 1, "only the filled row should be processed"

with SessionFactory() as ss:
    n_for_filled   = ss.query(AgentReflection).filter(
        AgentReflection.decision_ref_id == filled_id).count()
    n_for_unfilled = ss.query(AgentReflection).filter(
        AgentReflection.decision_ref_id == unfilled_id).count()
print(f"  reflections for filled: {n_for_filled} (expect 1)")
print(f"  reflections for unfilled: {n_for_unfilled} (expect 0)")
assert n_for_filled == 1
assert n_for_unfilled == 0
print("  OK: active_return filter works")


# ─────────────────────────────────────────────────────────────────────────────
# E. tab_type != 'sector' filtered out
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — tab_type != 'sector' (e.g. macro / audit) filtered out")
print("=" * 70)
# Reset
with SessionFactory() as ss:
    ss.query(AgentReflection).delete()
    ss.query(DecisionLog).filter(
        DecisionLog.decision_source == "ui_s2_6_test"
    ).delete(synchronize_session=False)
    ss.commit()

# 1 sector + 1 audit — only sector should be processed
sector_id = make_test_decision("XLK", "超配", +0.02, tab_type="sector")
audit_id  = make_test_decision("XLE", "超配", +0.02, tab_type="audit")
print(f"  sector id={sector_id}, audit id={audit_id}")

s5 = generate_reflections_for_pending(as_of=today, model=good_model())
print(f"  summary: {s5}")
assert s5["processed"] == 1
with SessionFactory() as ss:
    audit_refs = ss.query(AgentReflection).filter(
        AgentReflection.decision_ref_id == audit_id).count()
assert audit_refs == 0
print("  OK: only tab_type='sector' rows processed")


# ─────────────────────────────────────────────────────────────────────────────
# F. daily cap enforced
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — daily cap enforcement")
print("=" * 70)
# Reset
with SessionFactory() as ss:
    ss.query(AgentReflection).delete()
    ss.query(DecisionLog).filter(
        DecisionLog.decision_source == "ui_s2_6_test"
    ).delete(synchronize_session=False)
    ss.commit()

# Insert 3 candidates, set daily_cap=2 → should process only 2
ids_F = [
    make_test_decision("XLK", "超配", +0.02),
    make_test_decision("XLE", "低配", -0.02),
    make_test_decision("XLF", "超配", +0.015),
]
s6 = generate_reflections_for_pending(
    as_of=today, model=good_model(), max_per_call=10, daily_cap=2,
)
print(f"  daily_cap=2 summary: {s6}")
assert s6["processed"] == 2
print("  OK: cap=2 honored")

# Now run again — already wrote 2 today, cap should kick in skipped_daily_cap=True
s7 = generate_reflections_for_pending(
    as_of=today, model=good_model(), max_per_call=10, daily_cap=2,
)
print(f"  re-run summary: {s7}")
assert s7["skipped_daily_cap"] is True
assert s7["processed"] == 0
print("  OK: skipped_daily_cap flag set when budget exhausted")


# ─────────────────────────────────────────────────────────────────────────────
# G. LLM failure path — schema-invalid output → counted as failed
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("G — LLM/schema failure tallied as `failed`, not raised")
print("=" * 70)
# Reset and use very high cap so no daily-cap interference
with SessionFactory() as ss:
    ss.query(AgentReflection).delete()
    ss.query(DecisionLog).filter(
        DecisionLog.decision_source == "ui_s2_6_test"
    ).delete(synchronize_session=False)
    ss.commit()

ids_G = [
    make_test_decision("XLK", "超配", +0.02),
    make_test_decision("XLE", "低配", -0.02),
]

s8 = generate_reflections_for_pending(
    as_of=today, model=garbage_model(),
    max_per_call=10, daily_cap=100,
)
print(f"  garbage model summary: {s8}")
assert s8["processed"] == 0
assert s8["failed"] == 2, f"expected 2 failures, got {s8['failed']}"
print("  OK: 2 garbage outputs counted as failed; no exception escaped")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Cleanup")
print("=" * 70)
with SessionFactory() as ss:
    n_dec = ss.query(DecisionLog).filter(
        DecisionLog.decision_source == "ui_s2_6_test"
    ).delete(synchronize_session=False)
    n_ref = ss.query(AgentReflection).delete()
    ss.commit()
print(f"  deleted {n_dec} smoke decisions, {n_ref} reflections")
with SessionFactory() as ss:
    print(f"  agent_reflections rows remaining: {ss.query(AgentReflection).count()}")
    print(f"  smoke decision rows remaining:    "
          f"{ss.query(DecisionLog).filter(DecisionLog.decision_source == 'ui_s2_6_test').count()}")

print()
print("=" * 70)
print("S2.6 verification PASS")
print("=" * 70)
