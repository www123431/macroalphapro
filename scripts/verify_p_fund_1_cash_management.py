"""P-FUND-1 CashFlow ORM + deposit/withdraw API verification.

Facets:
  A. tables created (cash_flows + portfolio_nav_snapshots)
  B. deposit_funds default require_approval=True flow (status pending + PendingApproval row)
  C. approve_cash_flow flips pending -> applied + PendingApproval -> approved
  D. withdraw + insufficient balance re-validation on approval
  E. reject_cash_flow flips pending -> cancelled
  F. record_internal_flow auto-applied (no approval gate)
  G. get_current_cash_balance + get_cash_flow_history sums correctly
  H. cleanup
"""
import sys, os, datetime, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import inspect
from engine.memory import (
    init_db, SessionFactory, CashFlow, PortfolioNavSnapshot,
    PendingApproval, engine,
)
from engine.cash_management import (
    deposit_funds, withdraw_funds, record_internal_flow,
    approve_cash_flow, reject_cash_flow,
    get_cash_flow_history, get_current_cash_balance,
)

init_db()


SMOKE_SUPERVISOR = "ui_p_fund_1_test"


def _cleanup():
    with SessionFactory() as s:
        # Delete CashFlow rows by supervisor_id OR by linked notes
        cf_ids = [cf.id for cf in s.query(CashFlow).filter(
            (CashFlow.supervisor_id == SMOKE_SUPERVISOR)
            | (CashFlow.notes.like("%p_fund_1_test%"))
        ).all()]
        # Also delete the PendingApproval rows linked to those CashFlows
        approval_ids = [
            cf.approval_id for cf in s.query(CashFlow).filter(
                CashFlow.id.in_(cf_ids)
            ).all() if cf.approval_id
        ]
        s.query(PendingApproval).filter(
            (PendingApproval.id.in_(approval_ids))
            | (PendingApproval.approval_type == "cash_flow")
        ).delete(synchronize_session=False)
        s.query(CashFlow).filter(CashFlow.id.in_(cf_ids)).delete(synchronize_session=False)
        s.commit()


_cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# A. tables created
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — cash_flows + portfolio_nav_snapshots tables exist")
print("=" * 70)
ins = inspect(engine)
tables = set(ins.get_table_names())
assert "cash_flows" in tables
assert "portfolio_nav_snapshots" in tables
print("  OK: both tables present")

# Sanity: check key columns
cf_cols = {c["name"] for c in ins.get_columns("cash_flows")}
required_cf = {
    "id", "flow_date", "flow_type", "amount_usd", "is_external",
    "status", "supervisor_id", "approval_id", "notes",
    "created_at", "applied_at",
}
assert required_cf.issubset(cf_cols), f"missing CashFlow cols: {required_cf - cf_cols}"
print(f"  OK: CashFlow has all {len(required_cf)} required columns")

nav_cols = {c["name"] for c in ins.get_columns("portfolio_nav_snapshots")}
required_nav = {
    "snapshot_date", "nav_open", "external_flow", "nav_after_flow",
    "nav_close", "gross_pnl", "benchmark_close", "daily_modified_dietz",
}
assert required_nav.issubset(nav_cols)
print(f"  OK: PortfolioNavSnapshot has all {len(required_nav)} required columns")


# ─────────────────────────────────────────────────────────────────────────────
# B. deposit_funds default flow
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — deposit_funds default require_approval=True")
print("=" * 70)
cf_id_dep, ap_id_dep = deposit_funds(
    50_000.0, flow_date=datetime.date(2026, 4, 1),
    supervisor_id=SMOKE_SUPERVISOR,
    notes="p_fund_1_test deposit",
)
print(f"  deposit_funds returned cf_id={cf_id_dep}, approval_id={ap_id_dep}")
assert cf_id_dep is not None
assert ap_id_dep is not None
with SessionFactory() as s:
    cf = s.query(CashFlow).filter(CashFlow.id == cf_id_dep).one()
    pa = s.query(PendingApproval).filter(PendingApproval.id == ap_id_dep).one()
    print(f"  CashFlow: type={cf.flow_type} amount={cf.amount_usd:+.2f} status={cf.status}")
    print(f"  PendingApproval: type={pa.approval_type} status={pa.status}")
    assert cf.status == "pending"
    assert cf.amount_usd == +50_000.0
    assert cf.is_external is True
    assert pa.approval_type == "cash_flow"
    assert pa.status == "pending"
    assert pa.sector == "CASH" and pa.ticker == "USD"
print("  OK: pending state set; PendingApproval linked correctly")


# ─────────────────────────────────────────────────────────────────────────────
# C. approve flips state
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — approve_cash_flow flips pending -> applied")
print("=" * 70)
ok = approve_cash_flow(cf_id_dep)
print(f"  approve_cash_flow returned: {ok}")
assert ok is True
with SessionFactory() as s:
    cf = s.query(CashFlow).filter(CashFlow.id == cf_id_dep).one()
    pa = s.query(PendingApproval).filter(PendingApproval.id == ap_id_dep).one()
    print(f"  CashFlow.status: {cf.status}, applied_at: {cf.applied_at is not None}")
    print(f"  PendingApproval.status: {pa.status}, resolved_at: {pa.resolved_at is not None}")
    assert cf.status == "applied"
    assert cf.applied_at is not None
    assert pa.status == "approved"
    assert pa.resolved_at is not None
print("  OK: dual-table state machine intact")

# Idempotent re-approve
ok2 = approve_cash_flow(cf_id_dep)
print(f"  re-approve no-op: {ok2}")
assert ok2 is False
print("  OK: idempotent on already-applied")


# ─────────────────────────────────────────────────────────────────────────────
# D. withdraw + insufficient balance
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — withdraw + insufficient balance re-validation")
print("=" * 70)
# Current balance after approval = $50,000
balance_after_dep = get_current_cash_balance()
print(f"  balance after deposit: ${balance_after_dep:,.2f}")

# Try to withdraw $80,000 — pending, then approval should fail
cf_id_wd, ap_id_wd = withdraw_funds(
    80_000.0, flow_date=datetime.date(2026, 4, 5),
    supervisor_id=SMOKE_SUPERVISOR,
    notes="p_fund_1_test withdraw too much",
)
print(f"  pending withdraw $80k: cf_id={cf_id_wd}")
try:
    approve_cash_flow(cf_id_wd)
    assert False, "should have raised on insufficient balance"
except ValueError as e:
    print(f"  OK approval raised: {e}")
# Reject it to clean
rj = reject_cash_flow(cf_id_wd, reason="insufficient balance smoke test")
assert rj is True

# Now $20,000 should succeed
cf_id_wd2, ap_id_wd2 = withdraw_funds(
    20_000.0, flow_date=datetime.date(2026, 4, 6),
    supervisor_id=SMOKE_SUPERVISOR,
    notes="p_fund_1_test withdraw ok",
)
ok = approve_cash_flow(cf_id_wd2)
assert ok
balance_after_wd = get_current_cash_balance()
print(f"  balance after $20k withdraw: ${balance_after_wd:,.2f} (expect 30,000)")
assert abs(balance_after_wd - 30_000.0) < 1e-6
print("  OK: balance arithmetic correct + insufficient-balance guard works")


# ─────────────────────────────────────────────────────────────────────────────
# E. reject flips pending -> cancelled
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — reject_cash_flow flips pending -> cancelled")
print("=" * 70)
cf_id_rej, _ = deposit_funds(
    1_000.0, flow_date=datetime.date(2026, 4, 7),
    supervisor_id=SMOKE_SUPERVISOR, notes="p_fund_1_test to reject",
)
ok = reject_cash_flow(cf_id_rej, reason="testing rejection path")
print(f"  reject result: {ok}")
assert ok
with SessionFactory() as s:
    cf = s.query(CashFlow).filter(CashFlow.id == cf_id_rej).one()
    print(f"  CashFlow.status: {cf.status}")
    assert cf.status == "cancelled"
print("  OK")

# reason guard
try:
    reject_cash_flow(cf_id_rej, reason="x")
    assert False, "short reason should raise"
except ValueError:
    print("  OK: short-reason guard")


# ─────────────────────────────────────────────────────────────────────────────
# F. internal flow auto-applied
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — record_internal_flow auto-applied")
print("=" * 70)
cf_id_div = record_internal_flow(
    "dividend", 250.0, flow_date=datetime.date(2026, 4, 10),
    notes="p_fund_1_test SPY div",
)
print(f"  dividend cf_id={cf_id_div}")
with SessionFactory() as s:
    cf = s.query(CashFlow).filter(CashFlow.id == cf_id_div).one()
    print(f"  type={cf.flow_type} status={cf.status} is_external={cf.is_external}")
    assert cf.status == "applied"
    assert cf.is_external is False
    assert cf.approval_id is None
    assert cf.amount_usd == +250.0
print("  OK")

# Internal fee is signed negative
cf_id_fee = record_internal_flow(
    "interest", 50.0, flow_date=datetime.date(2026, 4, 11),
    notes="p_fund_1_test interest",
)
# Try internal flow with external type → should raise
try:
    record_internal_flow("deposit", 100.0, notes="should-not-pass")
    assert False
except ValueError as e:
    print(f"  OK: deposit-as-internal guard raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# G. balance + history
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("G — balance + history")
print("=" * 70)
balance = get_current_cash_balance()
# applied = +50000 (dep) - 20000 (wd) + 250 (div) + 50 (interest) = 30,300
print(f"  balance: ${balance:,.2f} (expect 30,300)")
assert abs(balance - 30_300.0) < 1e-6
print("  OK")

hist = get_cash_flow_history()
applied = [h for h in hist if h["status"] == "applied"]
all_rows = get_cash_flow_history(applied_only=False)
print(f"  applied rows: {len(applied)}")
print(f"  all rows (incl pending+cancelled): {len(all_rows)}")
ext_only = get_cash_flow_history(external_only=True)
ext_applied = [h for h in ext_only if h["status"] == "applied"]
print(f"  external-only applied rows: {len(ext_applied)}")
# 1 deposit + 1 withdraw = 2 external; 2 internal (div, interest)
assert len(ext_applied) == 2
print("  OK: filters work")


# ─────────────────────────────────────────────────────────────────────────────
# H. cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("H — cleanup")
print("=" * 70)
_cleanup()
with SessionFactory() as s:
    n_left = s.query(CashFlow).filter(
        CashFlow.supervisor_id == SMOKE_SUPERVISOR
    ).count()
    n_pa_left = s.query(PendingApproval).filter(
        PendingApproval.approval_type == "cash_flow"
    ).count()
print(f"  smoke CashFlow rows remaining: {n_left}")
print(f"  cash_flow PendingApproval rows remaining: {n_pa_left}")
assert n_left == 0 and n_pa_left == 0
print("  OK")

print()
print("=" * 70)
print("P-FUND-1 verification PASS (7 facets + cleanup)")
print("=" * 70)
