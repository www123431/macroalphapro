"""Global data-flow + performance audit (2026-05-04).

Probes:
  A. ORM schema vs SQLite actual columns (drift detection)
  B. Sign convention / hash convention / date handling cross-module
  C. Recently-added edge cases (P-FUND-2 NAV cold start, XIRR guards,
     EFFECTIVE_N_TRIALS staleness, drift vs tactical race, etc.)
  D. Indexing + N+1 hotspots
  E. SpecRegistry hash drift (silent edit detection)

Outputs JSON-ish findings list. Each finding has: id, severity, category,
description, recommendation.
"""
import sys, os, json, datetime, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import inspect
from engine.memory import (
    init_db, SessionFactory, engine,
    SimulatedPosition, PortfolioNavSnapshot, CashFlow,
    PendingApproval, AgentReflection, DecisionLog,
    SpecRegistry, HARKingFlag, AgentRun,
    PaperTradingRun,
)

init_db()

findings = []


def _add(severity, category, fid, desc, rec=""):
    findings.append({
        "id": fid, "severity": severity, "category": category,
        "description": desc, "recommendation": rec,
    })


# ─────────────────────────────────────────────────────────────────────────────
# A. ORM schema vs actual SQLite columns
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — Schema integrity (ORM vs SQLite)")
print("=" * 70)
ins = inspect(engine)

orm_classes = [
    SimulatedPosition, PortfolioNavSnapshot, CashFlow, PendingApproval,
    AgentReflection, DecisionLog, SpecRegistry, HARKingFlag,
    AgentRun, PaperTradingRun,
]
for cls in orm_classes:
    tbl = cls.__tablename__
    if not ins.has_table(tbl):
        _add("CRITICAL", "schema", f"missing_table_{tbl}",
             f"Table {tbl} declared in ORM but not present in SQLite",
             f"Run engine.memory._migrate_db() to create.")
        continue
    db_cols = {c["name"] for c in ins.get_columns(tbl)}
    orm_cols = {c.name for c in cls.__table__.columns}
    missing_in_db = orm_cols - db_cols
    extra_in_db = db_cols - orm_cols
    if missing_in_db:
        _add("HIGH", "schema", f"orm_extra_{tbl}",
             f"Table {tbl}: ORM declares {missing_in_db} but SQLite missing them",
             "Add to _migrate_db() ALTER list.")
    if extra_in_db:
        # Stale columns in DB but not in current ORM — usually OK (old fields)
        _add("LOW", "schema", f"db_extra_{tbl}",
             f"Table {tbl}: SQLite has {len(extra_in_db)} columns absent from ORM "
             f"(likely deprecated fields)",
             "Optional cleanup; not blocking.")
print(f"  {len([f for f in findings if f['category']=='schema'])} schema findings")


# ─────────────────────────────────────────────────────────────────────────────
# B. Cross-module data flow conventions
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — Sign / hash / date conventions")
print("=" * 70)

# B.1 CashFlow sign convention sanity (no rows with deposit + amount<0)
with SessionFactory() as s:
    bad_dep = s.query(CashFlow).filter(
        CashFlow.flow_type == "deposit",
        CashFlow.amount_usd < 0,
    ).count()
    bad_wd = s.query(CashFlow).filter(
        CashFlow.flow_type == "withdraw",
        CashFlow.amount_usd > 0,
    ).count()
    bad_div = s.query(CashFlow).filter(
        CashFlow.flow_type == "dividend",
        CashFlow.amount_usd < 0,
    ).count()
    bad_fee = s.query(CashFlow).filter(
        CashFlow.flow_type == "fee",
        CashFlow.amount_usd > 0,
    ).count()

if bad_dep + bad_wd + bad_div + bad_fee > 0:
    _add("HIGH", "sign", "cashflow_sign_violation",
         f"CashFlow sign convention violations: "
         f"deposit<0={bad_dep}, withdraw>0={bad_wd}, "
         f"dividend<0={bad_div}, fee>0={bad_fee}",
         "Inspect rows; cash_management.py guards new inserts but legacy may differ.")
else:
    print(f"  CashFlow sign convention: clean (all type-sign pairs consistent)")

# B.2 spec_hash column populated when present
with SessionFactory() as s:
    n_dec = s.query(DecisionLog).filter(DecisionLog.tab_type == "sector").count()
    n_with_hash = s.query(DecisionLog).filter(
        DecisionLog.tab_type == "sector",
        DecisionLog.spec_hash.isnot(None),
    ).count()
print(f"  DecisionLog.spec_hash population: {n_with_hash}/{n_dec} sector rows")
if n_dec > 0 and n_with_hash == 0:
    _add("MEDIUM", "spec_hash", "no_spec_hash_decisions",
         f"All {n_dec} sector DecisionLog rows have NULL spec_hash",
         "Auto-injection (P-FUND-4b) only kicks in for NEW decisions after the "
         "S3.2 wiring landed. Historical rows stay NULL — acceptable.")

# B.3 Date / DateTime convention: nav_close has snapshot_date (Date), not DateTime
with SessionFactory() as s:
    sample = s.query(PortfolioNavSnapshot).first()
    if sample:
        sd_type = type(sample.snapshot_date).__name__
        print(f"  PortfolioNavSnapshot.snapshot_date Python type: {sd_type}")
        if sd_type not in ("date", "datetime"):
            _add("HIGH", "date_type", "snapshot_date_type",
                 f"snapshot_date deserialized as {sd_type} (expected date)",
                 "Fix ORM Column type or migration.")


# ─────────────────────────────────────────────────────────────────────────────
# C. Edge cases in recently-added code
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — Recently-added edge cases")
print("=" * 70)

# C.1 P-FUND-2 cold-start: roll_daily_nav with 0 snapshots + 1 prior cash flow
# Should treat prior cash flow as funded capital (NAV bumped up before today)
from engine.portfolio_returns import roll_daily_nav
from engine.cash_management import deposit_funds, approve_cash_flow
SMOKE_SUP = "audit_c_smoke"


def _cleanup_C():
    with SessionFactory() as s:
        cf_ids = [cf.id for cf in s.query(CashFlow).filter(
            CashFlow.supervisor_id == SMOKE_SUP).all()]
        ap_ids = [cf.approval_id for cf in s.query(CashFlow).filter(
            CashFlow.id.in_(cf_ids)).all() if cf.approval_id]
        s.query(PendingApproval).filter(
            PendingApproval.id.in_(ap_ids)).delete(synchronize_session=False)
        s.query(CashFlow).filter(CashFlow.id.in_(cf_ids)).delete(synchronize_session=False)
        s.query(PortfolioNavSnapshot).filter(
            PortfolioNavSnapshot.notes == SMOKE_SUP).delete(synchronize_session=False)
        s.commit()


_cleanup_C()
# Cold deposit on 2099-01-01, no snapshots
cf_id, _ = deposit_funds(
    100_000.0, flow_date=datetime.date(2099, 1, 1),
    supervisor_id=SMOKE_SUP, notes="cold start audit", require_approval=False,
)


def _const_provider(tickers, date):
    return {t: 0.0 for t in tickers}


snap = roll_daily_nav(
    datetime.date(2099, 1, 2),
    return_provider=_const_provider, force=True,
)
# Update notes for cleanup tracking
with SessionFactory() as s:
    s.query(PortfolioNavSnapshot).filter(
        PortfolioNavSnapshot.snapshot_date.in_([
            datetime.date(2099, 1, 2),
        ])
    ).update({"notes": SMOKE_SUP})
    s.commit()
print(f"  cold-start nav_open: ${snap['nav_open']:,.2f}")
# nav_open should be initial_nav (1M) + prior_external (100k) = 1.1M
expected = 1_000_000.0 + 100_000.0
if abs(snap["nav_open"] - expected) > 0.01:
    _add("HIGH", "nav_rollup", "cold_start_prior_flow",
         f"Cold-start NAV {snap['nav_open']:,.2f} != expected "
         f"initial({1_000_000:,.0f}) + prior_flow({100_000:,.0f})",
         "Inspect roll_daily_nav prior_external query.")
else:
    print(f"  OK: cold-start with prior flow handled correctly (expected ${expected:,.0f})")
_cleanup_C()


# C.2 XIRR mono-sign guard (already tested in P-FUND-3 verify, re-confirm here)
from engine.performance_metrics import compute_xirr
try:
    compute_xirr([
        (datetime.date(2026, 1, 1), -100.0),
        (datetime.date(2026, 6, 1), -50.0),
    ])
    _add("HIGH", "xirr", "xirr_mono_sign_guard",
         "compute_xirr did NOT raise on mono-sign cash flows",
         "Should raise ValueError per design.")
except ValueError:
    print("  XIRR mono-sign guard: clean (raises as designed)")


# C.3 EFFECTIVE_N_TRIALS staleness — module value vs live computation
import engine.backtest as bt
from engine.preregistration import compute_pre_registration_n_trials
module_eff = bt.EFFECTIVE_N_TRIALS
live_eff, _ = bt.refresh_effective_n_trials()
print(f"  EFFECTIVE_N_TRIALS module={module_eff}, live={live_eff}")
if module_eff != live_eff:
    _add("MEDIUM", "ntrials", "effective_n_trials_stale",
         f"backtest.EFFECTIVE_N_TRIALS={module_eff} stale; live={live_eff}",
         "Call refresh_effective_n_trials() at top of backtest entry points "
         "(currently only refreshed by import or explicit call).")
else:
    print(f"  OK: EFFECTIVE_N_TRIALS consistent")


# C.4 SpecRegistry hash drift: live file hash vs stored current_hash
print()
from engine.preregistration import _compute_git_blob_hash, _resolve_to_abs
with SessionFactory() as s:
    rows = s.query(SpecRegistry).filter(SpecRegistry.status == "active").all()
    drifted = []
    missing = []
    for r in rows:
        abs_p = _resolve_to_abs(r.spec_path)
        if not os.path.exists(abs_p):
            missing.append(r.spec_path)
            continue
        live_hash = _compute_git_blob_hash(abs_p)
        if live_hash != r.current_hash:
            drifted.append((r.spec_path, r.current_hash[:12], live_hash[:12]))

print(f"  SpecRegistry total active: {len(rows)}")
print(f"  Drifted (current_hash != live file hash): {len(drifted)}")
print(f"  Missing files: {len(missing)}")

for path, stored, live in drifted[:5]:
    print(f"    DRIFT {path}: stored={stored} live={live}")
if drifted:
    _add("HIGH", "spec_hash", "specregistry_drift",
         f"{len(drifted)} active specs have drifted from registered hash",
         "These are silent edits — should walk through amend_spec() workflow "
         "or HARKing R1 will flag as CRITICAL.")
if missing:
    _add("HIGH", "spec_hash", "specregistry_missing_file",
         f"{len(missing)} spec files referenced in registry but missing from disk",
         "File deleted or moved; investigate.")


# C.5 Drift update vs apply_tactical race — synthesize test
# Already covered indirectly; check that apply_tactical_weight_update with no
# data does not crash
print()
from engine.portfolio_tracker import apply_tactical_weight_update
try:
    apply_tactical_weight_update(
        update_date=datetime.date(2099, 12, 31),
        sector_adjustments=None,
        new_entries=None,
    )
    print("  apply_tactical with no inputs: no-op OK")
except Exception as exc:
    _add("MEDIUM", "tactical", "apply_tactical_empty_crashes",
         f"apply_tactical_weight_update with empty args raised: {exc}",
         "Should be a no-op or raise informatively.")


# ─────────────────────────────────────────────────────────────────────────────
# D. Indexes + N+1 hotspots
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — Indexes + N+1 hotspots")
print("=" * 70)
critical_index_tables = {
    "decision_logs":          ["decision_date", "tab_type", "sector_name"],
    "agent_reflections":      ["agent_id", "decision_date"],
    "cash_flows":             ["flow_date"],
    "portfolio_nav_snapshots": [],  # snapshot_date is PK, indexed automatically
    "simulated_positions":    ["snapshot_date", "sector"],
    "spec_registry":          ["spec_path"],   # unique, indexed
    "harking_flags":          [],
    "agent_runs":             [],
    "paper_trading_runs":     ["as_of_date"],
}
for tbl, expected_cols in critical_index_tables.items():
    if not ins.has_table(tbl):
        continue
    indexes = ins.get_indexes(tbl)
    indexed_cols = set()
    for idx in indexes:
        indexed_cols.update(idx["column_names"] or [])
    # Check primary key (always indexed)
    pk = ins.get_pk_constraint(tbl)
    if pk:
        indexed_cols.update(pk.get("constrained_columns") or [])
    missing = [c for c in expected_cols if c not in indexed_cols]
    if missing:
        _add("LOW", "index", f"missing_index_{tbl}",
             f"Table {tbl}: columns {missing} likely benefit from indexes (high-cardinality filters)",
             "Add Index in ORM __table_args__.")
    else:
        print(f"  {tbl}: indexes look adequate")


# ─────────────────────────────────────────────────────────────────────────────
# E. Cross-module integration probes
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — Cross-module integration probes")
print("=" * 70)

# E.1 backtest result dataclass can be constructed cleanly
try:
    from engine.backtest import BacktestMetrics
    _ = BacktestMetrics(
        label="probe", ann_return=0, ann_vol=0, sharpe=0, dsr=0,
        n_months=0, max_drawdown=0, sortino=0, calmar=0, win_rate=0,
        n_trials=0, periods_per_year=12,
    )
    print("  BacktestMetrics construct: OK")
except Exception as exc:
    _add("HIGH", "integration", "backtest_metrics_init",
         f"BacktestMetrics construction failed: {exc}",
         "Check dataclass field defaults / required fields.")

# E.2 reflection retrieval gracefully returns [] when sentence-transformers absent
from engine.agents.reflection import retrieve_relevant_reflections
try:
    res = retrieve_relevant_reflections(
        agent_id="nonexistent_agent_audit", query_text="probe",
    )
    if not isinstance(res, list):
        _add("MEDIUM", "reflection", "retrieve_return_type",
             f"retrieve_relevant_reflections returned {type(res).__name__}, expected list")
    else:
        print(f"  reflection retrieve unknown agent: returns [] ({len(res)} items)")
except Exception as exc:
    _add("HIGH", "reflection", "retrieve_crashed",
         f"retrieve_relevant_reflections crashed on unknown agent: {exc}",
         "Should return empty list, not crash.")


# E.3 _get_nav fallback chain
from engine.portfolio_tracker import _get_nav
nav = _get_nav()
print(f"  _get_nav() returns: ${nav:,.2f}")
if nav <= 0:
    _add("HIGH", "nav", "get_nav_zero",
         f"_get_nav() returned {nav}; should always be > 0",
         "Check fallback chain: snapshot → SystemConfig → 1_000_000 default.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"AUDIT SUMMARY — {len(findings)} findings")
print("=" * 70)
for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
    bucket = [f for f in findings if f["severity"] == sev]
    if bucket:
        print(f"\n{sev} ({len(bucket)}):")
        for f in bucket:
            print(f"  [{f['category']}] {f['id']}")
            print(f"    {f['description']}")
            if f.get("recommendation"):
                print(f"    -> {f['recommendation']}")

# JSON dump
out_path = "scripts/audit_findings.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "audit_date": str(datetime.date.today()),
        "n_findings": len(findings),
        "by_severity": {
            sev: len([f for f in findings if f["severity"] == sev])
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        },
        "findings": findings,
    }, f, ensure_ascii=False, indent=2, default=str)
print(f"\nJSON dump: {out_path}")
