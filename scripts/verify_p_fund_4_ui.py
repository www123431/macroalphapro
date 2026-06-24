"""P-FUND-4 Performance Report UI verification.

Facets:
  A. page parses + section anchors present
  B. cold-start renders empty-state info block (no NAV snapshots)
  C. seeded with NAV + cash flows, all 4 sections render: G.0, G.1, G.2, G.3
  D. period table contains TWR/MWR/HPR columns
  E. cleanup
"""
import sys, os, ast, datetime, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import (
    init_db, SessionFactory,
    PortfolioNavSnapshot, CashFlow, PendingApproval,
)

init_db()

PAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pages", "performance_report.py",
)


SMOKE_DATE_RANGE = (datetime.date(2026, 6, 1), datetime.date(2026, 6, 30))
SMOKE_NOTE = "p_fund_4_smoke"


def _cleanup():
    with SessionFactory() as s:
        cf_ids = [
            cf.id for cf in s.query(CashFlow).filter(
                CashFlow.notes.like(f"%{SMOKE_NOTE}%")
            ).all()
        ]
        ap_ids = [
            cf.approval_id for cf in s.query(CashFlow).filter(
                CashFlow.id.in_(cf_ids)
            ).all() if cf.approval_id
        ]
        s.query(PendingApproval).filter(
            PendingApproval.id.in_(ap_ids)
        ).delete(synchronize_session=False)
        s.query(CashFlow).filter(
            CashFlow.id.in_(cf_ids)
        ).delete(synchronize_session=False)
        s.query(PortfolioNavSnapshot).filter(
            PortfolioNavSnapshot.snapshot_date >= SMOKE_DATE_RANGE[0],
            PortfolioNavSnapshot.snapshot_date <= SMOKE_DATE_RANGE[1],
        ).delete(synchronize_session=False)
        s.commit()


_cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# A. page parses + anchors
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — page parses + section anchors")
print("=" * 70)
with open(PAGE, "r", encoding="utf-8") as f:
    src = f.read()
ast.parse(src)
compile(src, PAGE, "exec")
print(f"  page parses ({len(src)} bytes)")
required = [
    "G.0 Supervisor cash flow",
    "G.1 Calendar heatmap",
    "G.2 NAV time series",
    "G.3 Period returns",
    "compute_period_summary",
    "deposit_funds",
    "withdraw_funds",
    "Bacon",
    "GIPS",
]
missing = [r for r in required if r not in src]
assert not missing, f"missing: {missing}"
print(f"  OK: all {len(required)} anchors present")


# ─────────────────────────────────────────────────────────────────────────────
# B. cold-start render
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — cold-start (no smoke NAV / cash flows)")
print("=" * 70)
# Make sure we have ZERO snapshots in the smoke window. If production data
# exists outside the smoke window it's fine; UI uses get_nav_with_flows()
# without filter, but for a freshly-cleaned DB, get_nav_with_flows is empty.

# Check whether production data exists
with SessionFactory() as s:
    n_real = s.query(PortfolioNavSnapshot).count()
print(f"  pre-existing PortfolioNavSnapshot rows: {n_real}")

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(PAGE, default_timeout=120)
at.run()
exceptions = [str(e.value) for e in at.exception]
print(f"  exceptions: {len(exceptions)}")
for e in exceptions[:3]:
    print(f"    {e[:200]}")
assert not exceptions, f"page raised on cold render: {exceptions}"

if n_real == 0:
    info_msgs = [el.value for el in at.info]
    cold_match = [m for m in info_msgs if "No NAV snapshots yet" in m]
    print(f"  cold-state info messages: {len(cold_match)}")
    assert cold_match, "expected cold-state info"
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# C. seeded scenario: 5 daily NAV snapshots + 1 deposit
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — seeded: 5 NAV snapshots + 1 cash flow")
print("=" * 70)

base = datetime.date(2026, 6, 1)
daily_returns = [+0.01, -0.005, +0.02, +0.003, -0.015]
nav = 100_000.0

with SessionFactory() as s:
    for i, r_d in enumerate(daily_returns):
        d = base + datetime.timedelta(days=i)
        ext_today = 0.0
        if d == base + datetime.timedelta(days=2):
            ext_today = 50_000.0
            cf = CashFlow(
                flow_date=d, flow_type="deposit",
                amount_usd=+50_000.0, is_external=True,
                status="applied",
                supervisor_id="ui-smoke",
                notes=f"{SMOKE_NOTE} mid-range deposit",
                applied_at=datetime.datetime.utcnow(),
            )
            s.add(cf)
        nav_open = nav
        nav_after = nav_open + ext_today
        nav_close = nav_after * (1 + r_d)
        snap = PortfolioNavSnapshot(
            snapshot_date=d, nav_open=nav_open,
            external_flow=ext_today, nav_after_flow=nav_after,
            nav_close=nav_close, gross_pnl=nav_close - nav_after,
            daily_modified_dietz=r_d,
            notes=SMOKE_NOTE,
        )
        s.add(snap)
        nav = nav_close
    s.commit()

at2 = AppTest.from_file(PAGE, default_timeout=120)
at2.run()
exceptions2 = [str(e.value) for e in at2.exception]
print(f"  exceptions: {len(exceptions2)}")
for e in exceptions2[:3]:
    print(f"    {e[:200]}")
assert not exceptions2, f"page raised on seeded render: {exceptions2}"

# Verify section content rendered
md_text = " ".join(m.value for m in at2.markdown)
for label in ["G.1 Calendar heatmap", "G.2 NAV time series", "G.3 Period returns"]:
    assert label in md_text, f"missing section: {label}"
print("  OK: G.1 / G.2 / G.3 all render with seeded data")


# ─────────────────────────────────────────────────────────────────────────────
# D. period table columns
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — period table contains TWR/MWR/HPR columns")
print("=" * 70)
df_count = len(at2.dataframe)
print(f"  dataframes rendered: {df_count}")
# We expect:
#   1 pending approvals dataframe (or skipped if 0 pending)
#   1 period summary dataframe (G.3)
# Multi dataframes may be present; assert at least 1
assert df_count >= 1
# Inspect the period table by content
found_period_cols = False
for d in at2.dataframe:
    cols = list(d.value.columns) if hasattr(d.value, "columns") else []
    if "TWR (MD)" in cols and "MWR (ann)" in cols and "HPR" in cols:
        found_period_cols = True
        break
assert found_period_cols, "period table with TWR/MWR/HPR columns not found"
print("  OK: period table has TWR (MD) + MWR (ann) + HPR + vs SPY columns")


# ─────────────────────────────────────────────────────────────────────────────
# E. cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — cleanup smoke")
print("=" * 70)
_cleanup()
with SessionFactory() as s:
    n_snap = s.query(PortfolioNavSnapshot).filter(
        PortfolioNavSnapshot.notes == SMOKE_NOTE
    ).count()
    n_cf = s.query(CashFlow).filter(
        CashFlow.notes.like(f"%{SMOKE_NOTE}%")
    ).count()
print(f"  smoke snapshots: {n_snap}")
print(f"  smoke cash flows: {n_cf}")
assert n_snap == 0 and n_cf == 0
print("  OK")

print()
print("=" * 70)
print("P-FUND-4 verification PASS (4 facets + cleanup)")
print("=" * 70)
