"""S3.3 HARKing detection — 4-rule trigger verification.

Facets:
  R1 — Late silent edit  → CRITICAL
       (register spec, validate_reference stamps first_referenced_at,
        then edit file silently → current_hash != last_recorded_hash)
  R2 — Threshold drift   → HIGH
       (current_hash drift + threshold pattern in current text + no recent amendment)
  R3 — Unannounced trial → HIGH
       (DecisionLog.spec_hash not in registry's known hash set)
  R4 — Predictions rewrite → MEDIUM
       (amendment_log accumulates ≥2 hypothesis_amend entries)

  X. Cleanup smoke residue
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import (
    init_db, SessionFactory,
    SpecRegistry, HARKingFlag, DecisionLog,
    save_decision,
)
from engine.preregistration import (
    register_spec, amend_spec, validate_reference,
    detect_harking, _compute_git_blob_hash,
)

init_db()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_PATHS = {
    "r1": os.path.join(ROOT, "docs", "spec_s3_3_harking_r1.md"),
    "r2": os.path.join(ROOT, "docs", "spec_s3_3_harking_r2.md"),
    "r4": os.path.join(ROOT, "docs", "spec_s3_3_harking_r4.md"),
}
SMOKE_DEC_SOURCE = "ui_s3_3_test"


def _cleanup():
    """Remove smoke specs, smoke decisions, and smoke flags."""
    with SessionFactory() as s:
        s.query(HARKingFlag).filter(
            HARKingFlag.spec_path.like("docs/spec_s3_3_harking%")
        ).delete(synchronize_session=False)
        s.query(HARKingFlag).filter(
            HARKingFlag.spec_path == "(decision_logs)"
        ).filter(
            HARKingFlag.notes.like("%fakehash%")
        ).delete(synchronize_session=False)
        s.query(SpecRegistry).filter(
            SpecRegistry.spec_path.like("docs/spec_s3_3_harking%")
        ).delete(synchronize_session=False)
        s.query(DecisionLog).filter(
            DecisionLog.decision_source == SMOKE_DEC_SOURCE
        ).delete(synchronize_session=False)
        s.commit()
    for p in SMOKE_PATHS.values():
        if os.path.exists(p):
            os.remove(p)


_cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# R1: late silent edit
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("R1 — Late silent edit (CRITICAL)")
print("=" * 70)

with open(SMOKE_PATHS["r1"], "w", encoding="utf-8") as f:
    f.write("# R1 spec\n\nNW t >= 1.5\n")
register_spec(SMOKE_PATHS["r1"], retro=False)
ok, reason = validate_reference(SMOKE_PATHS["r1"])
assert ok and reason == "ok"
print(f"  registered + validate_reference set first_referenced_at")

# Now silent-edit the file (no amendment record)
with open(SMOKE_PATHS["r1"], "w", encoding="utf-8") as f:
    f.write("# R1 spec SILENTLY EDITED\n\nNW t >= 2.5\n")
# Update SpecRegistry.current_hash to reflect the new disk state
# (simulating "file changed in git but no amend_spec call")
new_hash = _compute_git_blob_hash(SMOKE_PATHS["r1"])
with SessionFactory() as s:
    row = s.query(SpecRegistry).filter(
        SpecRegistry.spec_path == "docs/spec_s3_3_harking_r1.md"
    ).one()
    row.current_hash = new_hash  # simulating drift detection
    s.commit()

flags = detect_harking()
r1_flags = [f for f in flags if f["rule"] == "R1"
            and f["spec_path"] == "docs/spec_s3_3_harking_r1.md"]
print(f"  R1 flags raised: {len(r1_flags)}")
for f in r1_flags:
    print(f"    severity={f['severity']} notes={f['notes'][:80]}")
assert len(r1_flags) == 1, f"expected 1 R1 flag, got {len(r1_flags)}"
assert r1_flags[0]["severity"] == "CRITICAL"
print("  OK: R1 fires CRITICAL")


# ─────────────────────────────────────────────────────────────────────────────
# R2: threshold drift without amendment (>7 days since amendment)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("R2 — Threshold drift no amendment (HIGH)")
print("=" * 70)

with open(SMOKE_PATHS["r2"], "w", encoding="utf-8") as f:
    f.write("# R2 spec\n\nSharpe >= 0.5\n")
register_spec(SMOKE_PATHS["r2"], retro=False)
# Drift current_hash WITHOUT recording an amendment
with open(SMOKE_PATHS["r2"], "w", encoding="utf-8") as f:
    f.write("# R2 spec drifted\n\nSharpe >= 0.8\nNW t >= 1.8\n")
new_hash = _compute_git_blob_hash(SMOKE_PATHS["r2"])
with SessionFactory() as s:
    row = s.query(SpecRegistry).filter(
        SpecRegistry.spec_path == "docs/spec_s3_3_harking_r2.md"
    ).one()
    row.current_hash = new_hash
    s.commit()
# No amendment in last 7 days (none ever) → R2 fires
flags = detect_harking()
r2_flags = [f for f in flags if f["rule"] == "R2"
            and f["spec_path"] == "docs/spec_s3_3_harking_r2.md"]
print(f"  R2 flags raised: {len(r2_flags)}")
for f in r2_flags:
    print(f"    severity={f['severity']} notes={f['notes'][:90]}")
assert len(r2_flags) == 1, f"expected 1 R2 flag, got {len(r2_flags)}"
assert r2_flags[0]["severity"] == "HIGH"
print("  OK: R2 fires HIGH")


# ─────────────────────────────────────────────────────────────────────────────
# R3: DecisionLog.spec_hash not in registry
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("R3 — Unannounced trial (HIGH)")
print("=" * 70)

UNKNOWN_HASH = "fakehash" + "0" * 32  # 40 chars, definitely not in registry
saved_id = save_decision(
    tab_type="sector",
    ai_conclusion="测试 R3 unannounced spec_hash",
    vix_level=14.0,
    sector_name="XLK",
    ticker="XLK",
    news_summary="r3 smoke",
    macro_regime="低波动/牛市",
    horizon="季度(3个月)",
    confidence_score=60,
    decision_date=datetime.date(2026, 4, 28),
    decision_source=SMOKE_DEC_SOURCE,
    spec_hash=UNKNOWN_HASH,
)
print(f"  inserted DecisionLog id={saved_id} with unknown spec_hash {UNKNOWN_HASH[:12]}...")

flags = detect_harking()
r3_flags = [f for f in flags if f["rule"] == "R3"]
print(f"  R3 flags raised: {len(r3_flags)}")
for f in r3_flags:
    print(f"    severity={f['severity']} notes={f['notes'][:80]}")
# At least one should be present and it should reference our unknown hash
matching = [f for f in r3_flags if "fakehash" in f["notes"]]
assert matching, f"R3 did not flag unknown spec_hash; got {r3_flags}"
assert matching[0]["severity"] == "HIGH"
print("  OK: R3 fires HIGH on unknown spec_hash")


# ─────────────────────────────────────────────────────────────────────────────
# R4: ≥2 hypothesis_amend entries
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("R4 — Predictions rewrite (MEDIUM)")
print("=" * 70)

with open(SMOKE_PATHS["r4"], "w", encoding="utf-8") as f:
    f.write("# R4 spec\n\nH1: TSMOM works\n")
register_spec(SMOKE_PATHS["r4"], retro=False)

with open(SMOKE_PATHS["r4"], "w", encoding="utf-8") as f:
    f.write("# R4 spec\n\nH1: TSMOM 12m works  (amended)\n")
amend_spec(SMOKE_PATHS["r4"], kind="hypothesis_amend",
           reason="reframe H1 hypothesis to time-series momentum 12m, qualifies as ≥20 chars")

with open(SMOKE_PATHS["r4"], "w", encoding="utf-8") as f:
    f.write("# R4 spec\n\nH1: TSMOM 12m + 6m mix\n")
amend_spec(SMOKE_PATHS["r4"], kind="hypothesis_amend",
           reason="add H1b about 6-month TSMOM corroboration, ≥20 chars qualifying")

flags = detect_harking()
r4_flags = [f for f in flags if f["rule"] == "R4"
            and f["spec_path"] == "docs/spec_s3_3_harking_r4.md"]
print(f"  R4 flags raised: {len(r4_flags)}")
for f in r4_flags:
    print(f"    severity={f['severity']} notes={f['notes']}")
assert len(r4_flags) == 1, f"expected 1 R4 flag, got {len(r4_flags)}"
assert r4_flags[0]["severity"] == "MEDIUM"
print("  OK: R4 fires MEDIUM after 2 hypothesis_amend")


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency: re-running detect_harking should NOT duplicate open flags
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Idempotency — re-detect does not duplicate open flags")
print("=" * 70)
flags_2nd = detect_harking()
new_smoke = [f for f in flags_2nd
             if f["rule"] in ("R1", "R2", "R3", "R4")
             and ("s3_3_harking" in f["spec_path"] or "fakehash" in f.get("notes", ""))]
print(f"  flags newly raised on 2nd call: {len(new_smoke)} (expect 0)")
assert len(new_smoke) == 0, f"got {len(new_smoke)} duplicates"
print("  OK: idempotent")


# ─────────────────────────────────────────────────────────────────────────────
# Persisted state in HARKingFlag table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Persistence — HARKingFlag rows present")
print("=" * 70)
with SessionFactory() as s:
    n_smoke_flags = s.query(HARKingFlag).filter(
        HARKingFlag.spec_path.like("docs/spec_s3_3_harking%")
    ).count()
    n_r3_smoke = s.query(HARKingFlag).filter(
        HARKingFlag.notes.like("%fakehash%")
    ).count()
    # R1 spec naturally also trips R2 (silent edit + threshold pattern + no
    # amendment), so we expect the R1 path to produce 2 flags. R2 spec
    # produces 1 R2. R4 spec produces 1 R4. Total = 4.
    print(f"  R1 spec(R1+R2)+R2 spec(R2)+R4 spec(R4) flag rows: {n_smoke_flags} (expect 4)")
    print(f"  R3 smoke flag rows: {n_r3_smoke} (expect 1)")
    assert n_smoke_flags == 4
    assert n_r3_smoke == 1


# ─────────────────────────────────────────────────────────────────────────────
# X. Cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("X — cleanup smoke residue")
print("=" * 70)
_cleanup()
with SessionFactory() as s:
    n_left_flags = s.query(HARKingFlag).filter(
        HARKingFlag.spec_path.like("docs/spec_s3_3%")
    ).count()
    n_left_specs = s.query(SpecRegistry).filter(
        SpecRegistry.spec_path.like("docs/spec_s3_3%")
    ).count()
    n_left_dec = s.query(DecisionLog).filter(
        DecisionLog.decision_source == SMOKE_DEC_SOURCE
    ).count()
print(f"  flags remaining:    {n_left_flags}")
print(f"  specs remaining:    {n_left_specs}")
print(f"  decisions left:     {n_left_dec}")
files_left = sum(1 for p in SMOKE_PATHS.values() if os.path.exists(p))
print(f"  smoke files remain: {files_left}")
assert n_left_flags == 0 and n_left_specs == 0 and n_left_dec == 0 and files_left == 0
print("  OK: clean")

print()
print("=" * 70)
print("S3.3 HARKing-rule verification PASS (4/4 rules + idempotency)")
print("=" * 70)
