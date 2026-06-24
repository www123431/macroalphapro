"""S3.3 UI panel + cron hook verification.

Facets:
  A. Section H present in pages/agent_observability.py + parses
  B. UI renders cold (no smoke data) without exception
  C. UI renders with seeded SpecRegistry + HARKingFlag rows
     (3 metrics for severity tiers + amendment timeline + breakdown)
  D. cron hook: orchestrator.run_daily contains detect_harking trigger
  E. cleanup
"""
import sys, os, ast, datetime, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import (
    init_db, SessionFactory,
    SpecRegistry, HARKingFlag,
)
from engine.preregistration import register_spec, amend_spec, _compute_git_blob_hash

init_db()

PAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pages", "agent_observability.py",
)
ORCHESTRATOR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "engine", "orchestrator.py",
)

# Cleanup any prior smoke residue
SMOKE_PATH = "docs/spec_s3_3_ui_smoke.md"
SMOKE_ABS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), SMOKE_PATH,
)


def _cleanup():
    with SessionFactory() as s:
        s.query(HARKingFlag).filter(
            HARKingFlag.spec_path.like("docs/spec_s3_3_ui_smoke%")
        ).delete(synchronize_session=False)
        s.query(SpecRegistry).filter(
            SpecRegistry.spec_path == SMOKE_PATH
        ).delete(synchronize_session=False)
        s.commit()
    if os.path.exists(SMOKE_ABS):
        os.remove(SMOKE_ABS)


_cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# A. parse + Section H present
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — page parses + Section H present")
print("=" * 70)
with open(PAGE, "r", encoding="utf-8") as f:
    src = f.read()
ast.parse(src)
compile(src, PAGE, "exec")
print(f"  OK: parses ({len(src)} bytes)")

required = [
    "H. Pre-Registration Enforcement",
    "EFFECTIVE_N_TRIALS breakdown",
    "Registered specifications",
    "HARKing flags",
    "SpecRegistry",
    "HARKingFlag",
]
missing = [r for r in required if r not in src]
assert not missing, f"missing anchors: {missing}"
print(f"  OK: {len(required)} anchors present")


# ─────────────────────────────────────────────────────────────────────────────
# B. UI renders cold (clear smoke first)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — UI renders cold (no smoke flags / no smoke specs)")
print("=" * 70)

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(PAGE, default_timeout=180)
at.run()
exceptions = [str(e.value) for e in at.exception]
print(f"  exceptions: {len(exceptions)}")
for e in exceptions[:3]:
    print(f"    {e[:160]}")
assert not exceptions, f"page raised: {exceptions}"

# Section H should at least show the EFFECTIVE_N_TRIALS metric tile
metric_labels = [m.label for m in at.metric]
print(f"  metrics rendered: {len(metric_labels)}")
print(f"    sample: {metric_labels[-6:]}")
assert "EFFECTIVE_N_TRIALS" in metric_labels, \
    f"EFFECTIVE_N_TRIALS metric missing"
print("  OK: cold render OK; EFFECTIVE_N_TRIALS metric present")


# ─────────────────────────────────────────────────────────────────────────────
# C. UI renders with seeded data (SpecRegistry + HARKingFlag)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — UI renders with seeded smoke data")
print("=" * 70)

# Seed: 1 forward spec + 1 amendment + 1 manual HARKingFlag of each severity
with open(SMOKE_ABS, "w", encoding="utf-8") as f:
    f.write("# UI smoke spec\n\nNW t >= 1.5\n")
register_spec(SMOKE_ABS, retro=False)

with open(SMOKE_ABS, "w", encoding="utf-8") as f:
    f.write("# UI smoke spec v2\n\nNW t >= 1.8\n")
amend_spec(SMOKE_ABS, kind="threshold_tweak",
           reason="UI smoke amendment to test timeline render, ≥20 chars")

# Inject 3 HARKing flags (one per severity tier) for UI exercise
now = datetime.datetime.utcnow()
with SessionFactory() as s:
    for sev in ("CRITICAL", "HIGH", "MEDIUM"):
        s.add(HARKingFlag(
            rule="R1" if sev == "CRITICAL" else "R2" if sev == "HIGH" else "R4",
            spec_path=SMOKE_PATH,
            severity=sev,
            detected_at=now,
            notes=f"smoke {sev} flag for UI render test",
        ))
    s.commit()

at2 = AppTest.from_file(PAGE, default_timeout=180)
at2.run()
exceptions2 = [str(e.value) for e in at2.exception]
print(f"  exceptions: {len(exceptions2)}")
for e in exceptions2[:3]:
    print(f"    {e[:160]}")
assert not exceptions2, f"page raised on seeded: {exceptions2}"

metric_labels2 = [m.label for m in at2.metric]
print(f"  metrics rendered: {len(metric_labels2)}")
# H.3 severity tiles
for label in ("Open CRITICAL", "Open HIGH", "Open MEDIUM"):
    assert label in metric_labels2, f"missing severity tile: {label}"
    val = next(m.value for m in at2.metric if m.label == label)
    print(f"    {label} = {val}")
print("  OK: severity tiles render")

# Spec dataframe should show the smoke spec
df_count = len(at2.dataframe)
print(f"  dataframes total: {df_count} (expected ≥3: audit + spec list + flags)")
assert df_count >= 3
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# D. cron hook in orchestrator.run_daily
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — orchestrator.run_daily wired to detect_harking")
print("=" * 70)
with open(ORCHESTRATOR, "r", encoding="utf-8") as f:
    osrc = f.read()
assert "detect_harking()" in osrc
assert "HARKing detection nightly hook" in osrc
# verify it's inside run_daily
run_daily_start = osrc.find("def run_daily(")
run_weekly_start = osrc.find("def run_weekly(", run_daily_start)
run_daily_body = osrc[run_daily_start:run_weekly_start]
assert "detect_harking" in run_daily_body, \
    "detect_harking call not inside run_daily"
print("  OK: run_daily contains detect_harking hook")
# Also ensure it's wrapped in try/except so failures don't block the cycle
assert "try:" in run_daily_body and "except Exception" in run_daily_body
# Specifically the detect_harking hook section
hook_idx = run_daily_body.find("detect_harking")
hook_window = run_daily_body[hook_idx-200:hook_idx+200]
assert "try:" in run_daily_body[:hook_idx], "detect_harking not in try/except"
print("  OK: hook wrapped in try/except (won't block daily cycle on failure)")


# ─────────────────────────────────────────────────────────────────────────────
# E. cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — cleanup smoke residue")
print("=" * 70)
_cleanup()
with SessionFactory() as s:
    n_specs = s.query(SpecRegistry).filter(
        SpecRegistry.spec_path == SMOKE_PATH).count()
    n_flags = s.query(HARKingFlag).filter(
        HARKingFlag.spec_path == SMOKE_PATH).count()
print(f"  specs left: {n_specs}, flags left: {n_flags}, file: {os.path.exists(SMOKE_ABS)}")
assert n_specs == 0 and n_flags == 0 and not os.path.exists(SMOKE_ABS)
print("  OK")

print()
print("=" * 70)
print("S3.3 UI + cron verification PASS")
print("=" * 70)
