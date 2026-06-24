"""S3.1 SpecRegistry + CLI + backfill verification.

Facets:
  A. ORM tables created (spec_registry + harking_flags)
  B. register_spec idempotent + retro flag honored
  C. amend_spec ledger appends + n_trials accumulates per kind
  D. validate_reference detects (i) ok, (ii) silent_edit, (iii) not_registered
  E. compute_pre_registration_n_trials sums only forward (non-retro) specs
  F. backfill: every docs/spec_*.md gets retro-registered
  G. cleanup smoke residue
"""
import sys, os, glob, json, datetime, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import (
    init_db, SessionFactory,
    SpecRegistry, HARKingFlag,
)
from engine.preregistration import (
    register_spec, amend_spec, validate_reference,
    list_specs, compute_pre_registration_n_trials,
    AMENDMENT_KINDS,
    _compute_git_blob_hash, _normalize_spec_path,
)

init_db()


# ─────────────────────────────────────────────────────────────────────────────
# A. ORM tables created
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — spec_registry + harking_flags tables present")
print("=" * 70)
from sqlalchemy import inspect
from engine.memory import engine
ins = inspect(engine)
tables = set(ins.get_table_names())
assert "spec_registry" in tables, "spec_registry missing"
assert "harking_flags" in tables, "harking_flags missing"
print("  OK: both tables present")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup any prior smoke residue
# ─────────────────────────────────────────────────────────────────────────────
SMOKE_PATH_PREFIX = "docs/spec_s3_1_smoke"
print("\n— preliminary cleanup —")
with SessionFactory() as s:
    n_dropped_specs = s.query(SpecRegistry).filter(
        SpecRegistry.spec_path.like(f"{SMOKE_PATH_PREFIX}%")
    ).delete(synchronize_session=False)
    n_dropped_flags = s.query(HARKingFlag).filter(
        HARKingFlag.spec_path.like(f"{SMOKE_PATH_PREFIX}%")
    ).delete(synchronize_session=False)
    s.commit()
print(f"  pre-cleared {n_dropped_specs} smoke specs, {n_dropped_flags} flags")


# Helper: write a synthetic smoke spec
def write_smoke_spec(idx: int, content: str = None) -> str:
    path = f"docs/spec_s3_1_smoke_{idx}.md"
    abs_p = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path,
    )
    with open(abs_p, "w", encoding="utf-8") as f:
        f.write(content or f"# Smoke spec {idx}\n\nNW t ≥ 1.5\n")
    return abs_p


# ─────────────────────────────────────────────────────────────────────────────
# B. register_spec
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — register_spec: idempotent + retro flag")
print("=" * 70)

p1 = write_smoke_spec(1)
rid_1a = register_spec(p1, retro=False)
rid_1b = register_spec(p1, retro=False)  # re-register
print(f"  register_spec(p1) -> id={rid_1a}; re-register -> id={rid_1b}")
assert rid_1a == rid_1b, "re-register should be idempotent (same id)"
print("  OK: idempotent")

p2 = write_smoke_spec(2)
rid_2 = register_spec(p2, retro=True)
with SessionFactory() as s:
    r2 = s.query(SpecRegistry).filter(SpecRegistry.id == rid_2).one()
    assert r2.retro_registered is True
    assert r2.n_trials_contributed == 0  # retro doesn't contribute
    r1 = s.query(SpecRegistry).filter(SpecRegistry.id == rid_1a).one()
    assert r1.retro_registered is False
    assert r1.n_trials_contributed == 1  # forward = +1 on register
print(f"  forward register: n_trials=1 (id={rid_1a})")
print(f"  retro register:   n_trials=0 (id={rid_2})")
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# C. amend_spec ledger + n_trials
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — amend_spec ledger + n_trials grading")
print("=" * 70)

# Modify p1 then amend
with open(p1, "w", encoding="utf-8") as f:
    f.write("# Smoke spec 1 v2\n\nNW t ≥ 1.8\n")  # threshold tweak
amend_spec(p1, kind="threshold_tweak",
           reason="raise NW t threshold from 1.5 to 1.8 per power analysis")

# Amend again with hypothesis amend (+3)
with open(p1, "w", encoding="utf-8") as f:
    f.write("# Smoke spec 1 v3\n\nNW t ≥ 1.8 + new H2\n")
amend_spec(p1, kind="hypothesis_amend",
           reason="add H2 about cross-asset spillover, ≥20 chars qualifying")

with SessionFactory() as s:
    r1 = s.query(SpecRegistry).filter(SpecRegistry.id == rid_1a).one()
    ledger = json.loads(r1.amendment_log)
    print(f"  amendments: {len(ledger)}")
    for e in ledger:
        print(f"    {e['kind']}: +{e['n_trials_added']} (cumulative file)")
    assert len(ledger) == 2
    assert ledger[0]["kind"] == "threshold_tweak"
    assert ledger[0]["n_trials_added"] == 1
    assert ledger[1]["kind"] == "hypothesis_amend"
    assert ledger[1]["n_trials_added"] == 3
    # Initial register +1; threshold +1; hypothesis +3 → cumulative 5
    print(f"  total n_trials_contributed: {r1.n_trials_contributed} (expect 5)")
    assert r1.n_trials_contributed == 5

# reason validation: too-short reason should raise
try:
    amend_spec(p1, kind="clarification", reason="x")
    assert False, "should have raised on short reason"
except ValueError as e:
    print(f"  OK: short-reason guard raised: {e}")

# unknown kind should raise
try:
    amend_spec(p1, kind="bogus", reason="some adequate-length explanation here")
    assert False, "should have raised on unknown kind"
except ValueError as e:
    print(f"  OK: unknown-kind guard raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# D. validate_reference
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — validate_reference")
print("=" * 70)

# Currently p1 is at v3, last amendment recorded that hash → ok
ok, reason = validate_reference(p1)
print(f"  validate(p1, hash matches): ok={ok} reason={reason}")
assert ok and reason == "ok"

# Silent-edit: modify p1 file without amending
with open(p1, "w", encoding="utf-8") as f:
    f.write("# Smoke spec 1 v4 SILENT\n\nNW t ≥ 2.0\n")
ok, reason = validate_reference(p1)
print(f"  validate(p1, silent edit): ok={ok} reason={reason}")
assert not ok and reason == "silent_edit_detected"

# Not registered
nreg_path = "docs/spec_s3_1_smoke_NEVER_REGISTERED.md"
abs_nreg = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), nreg_path,
)
with open(abs_nreg, "w", encoding="utf-8") as f:
    f.write("not registered\n")
ok, reason = validate_reference(abs_nreg)
print(f"  validate(not_registered): ok={ok} reason={reason}")
assert not ok and reason == "not_registered"
os.remove(abs_nreg)

# File missing
ok, reason = validate_reference("docs/spec_s3_1_smoke_DOES_NOT_EXIST.md")
print(f"  validate(missing): ok={ok} reason={reason}")
assert not ok and reason == "spec_file_missing"

print("  OK: all 4 cases covered")


# ─────────────────────────────────────────────────────────────────────────────
# E. compute_pre_registration_n_trials
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — compute_pre_registration_n_trials")
print("=" * 70)
n = compute_pre_registration_n_trials()
print(f"  pre-reg n_trials (forward only): {n}")
# Smoke setup: 1 forward spec with cumulative n_trials = 5; retro contributes 0
# Plus other forward specs may exist from prior tests, but we should at least
# count the smoke spec contribution (=5)
assert n >= 5, f"expected ≥5, got {n}"
print(f"  OK: ≥5 (smoke spec p1 contributed 5)")


# ─────────────────────────────────────────────────────────────────────────────
# F. backfill all real docs/spec_*.md
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — retro-backfill all docs/spec_*.md")
print("=" * 70)
specs_glob = sorted(glob.glob("docs/spec_*.md"))
# Exclude smoke spec
real_specs = [s for s in specs_glob if "s3_1_smoke" not in s]
print(f"  found {len(real_specs)} real spec files")
n_registered_now = 0
n_already = 0
for s in real_specs:
    with SessionFactory() as sess:
        existing = sess.query(SpecRegistry).filter(
            SpecRegistry.spec_path == _normalize_spec_path(s)
        ).first()
    if existing:
        n_already += 1
    else:
        register_spec(s, retro=True)
        n_registered_now += 1
print(f"  newly retro-registered: {n_registered_now}")
print(f"  already in registry:    {n_already}")
print(f"  total real spec rows: {n_registered_now + n_already}")
assert n_registered_now + n_already == len(real_specs)

# Verify the SSRN paper spec is registered with hash 03a6767... — actually
# we use git-blob hash here, not sha256; the SSRN paper spec hash in
# docs/spec_ssrn_paper_v1.md is a SHA-256 from S4 bookkeeping, the registry
# stores git-blob SHA-1. They are intentionally different hash schemes.
ssrn_path = _normalize_spec_path("docs/spec_ssrn_paper_v1.md")
with SessionFactory() as sess:
    ssrn_row = sess.query(SpecRegistry).filter(
        SpecRegistry.spec_path == ssrn_path
    ).one_or_none()
    if ssrn_row:
        print(f"  SSRN paper spec: id={ssrn_row.id} "
              f"git_blob_hash={ssrn_row.git_blob_hash[:12]} "
              f"retro={ssrn_row.retro_registered}")
        assert ssrn_row.retro_registered is True
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# G. cleanup smoke residue
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("G — cleanup smoke residue")
print("=" * 70)
with SessionFactory() as s:
    n_dropped = s.query(SpecRegistry).filter(
        SpecRegistry.spec_path.like(f"{SMOKE_PATH_PREFIX}%")
    ).delete(synchronize_session=False)
    s.commit()
# Remove smoke files
for idx in (1, 2):
    p = f"docs/spec_s3_1_smoke_{idx}.md"
    abs_p = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p,
    )
    if os.path.exists(abs_p):
        os.remove(abs_p)
print(f"  cleared {n_dropped} smoke registry rows + 2 smoke files")

with SessionFactory() as s:
    n_smoke_left = s.query(SpecRegistry).filter(
        SpecRegistry.spec_path.like(f"{SMOKE_PATH_PREFIX}%")
    ).count()
    assert n_smoke_left == 0
print("  OK: 0 smoke rows remaining")

print()
print("=" * 70)
print("S3.1 verification PASS")
print("=" * 70)
