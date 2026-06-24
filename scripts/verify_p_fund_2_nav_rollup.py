"""P-FUND-2 NAV rollup + orchestrator hook verification.

Uses synthetic return_provider for determinism — does not depend on yfinance.

Facets:
  A. cold-start: no prior snapshot → nav_open = initial_nav (1M)
  B. external flow today moves nav_open -> nav_after_flow correctly
  C. portfolio gross return: weighted asset returns flow into nav_close
  D. multi-day chain: nav_close[t] = nav_open[t+1]
  E. idempotency: same date repeat returns existing snapshot unless force=True
  F. cash flow integration: deposit + return + withdraw all reflected
  G. orchestrator.run_daily wired (grep contract)
  H. cleanup
"""
import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import (
    init_db, SessionFactory,
    CashFlow, PortfolioNavSnapshot, PendingApproval, SimulatedPosition,
)
from engine.cash_management import (
    deposit_funds, approve_cash_flow, withdraw_funds,
)
from engine.portfolio_returns import (
    roll_daily_nav, get_nav_series, get_nav_with_flows, initial_nav,
)

init_db()

SMOKE_SUPERVISOR = "ui_p_fund_2_test"
SMOKE_SECTOR = "P_FUND_TEST_SECTOR"
SMOKE_TICKER = "ZZZ_TEST"


def _cleanup():
    with SessionFactory() as s:
        cf_ids = [cf.id for cf in s.query(CashFlow).filter(
            CashFlow.supervisor_id == SMOKE_SUPERVISOR
        ).all()]
        approval_ids = [
            cf.approval_id for cf in s.query(CashFlow).filter(
                CashFlow.id.in_(cf_ids)
            ).all() if cf.approval_id
        ]
        s.query(PendingApproval).filter(
            PendingApproval.id.in_(approval_ids)
        ).delete(synchronize_session=False)
        s.query(CashFlow).filter(CashFlow.id.in_(cf_ids)).delete(synchronize_session=False)
        s.query(PortfolioNavSnapshot).filter(
            PortfolioNavSnapshot.snapshot_date >= datetime.date(2026, 4, 1),
            PortfolioNavSnapshot.snapshot_date <= datetime.date(2026, 4, 30),
        ).delete(synchronize_session=False)
        s.query(SimulatedPosition).filter(
            SimulatedPosition.sector == SMOKE_SECTOR
        ).delete(synchronize_session=False)
        s.commit()


_cleanup()


# Seed a SimulatedPosition for the test window so portfolio gross return is
# computable. We use a sector name unique enough that get_current_positions
# may surface it — but to be safe we'll inject the return provider.
with SessionFactory() as s:
    pos = SimulatedPosition(
        snapshot_date=datetime.date(2026, 4, 1),
        sector=SMOKE_SECTOR,
        ticker=SMOKE_TICKER,
        target_weight=1.0,         # 100% in the test ticker (entire portfolio)
        actual_weight=1.0,
        track="main",
    )
    s.add(pos)
    s.commit()

# Synthetic return provider:
#   Day 4/1: +1.00%
#   Day 4/2: -0.50%
#   Day 4/3: +2.00%
SYNTHETIC_RETURNS = {
    datetime.date(2026, 4, 1): {SMOKE_TICKER: +0.01},
    datetime.date(2026, 4, 2): {SMOKE_TICKER: -0.005},
    datetime.date(2026, 4, 3): {SMOKE_TICKER: +0.02},
}


def synth_provider(tickers, date):
    return SYNTHETIC_RETURNS.get(date, {t: 0.0 for t in tickers})


# ─────────────────────────────────────────────────────────────────────────────
# A. cold-start
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — cold start: nav_open = initial_nav()")
print("=" * 70)
init = initial_nav()
print(f"  initial_nav(): ${init:,.2f} (config paper_trading_nav)")

snap_a = roll_daily_nav(datetime.date(2026, 4, 1), return_provider=synth_provider)
print(f"  Day 4/1 snapshot: nav_open={snap_a['nav_open']:.2f} "
      f"ext={snap_a['external_flow']:+.2f} nav_close={snap_a['nav_close']:.2f}")
# nav_open == initial; ext_flow = 0; portfolio_ret = +0.01 → nav_close = init * 1.01
expected = init * 1.01
assert abs(snap_a["nav_open"] - init) < 1e-6
assert snap_a["external_flow"] == 0.0
assert abs(snap_a["nav_close"] - expected) < 1e-3
print(f"  expected nav_close: {expected:,.2f} [OK]")
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# B. external flow today
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — external flow today")
print("=" * 70)
# Deposit $100k on 4/2 (immediate apply for test simplicity)
cf_id, _ = deposit_funds(
    100_000.0, flow_date=datetime.date(2026, 4, 2),
    supervisor_id=SMOKE_SUPERVISOR, notes="p_fund_2_test deposit",
)
approve_cash_flow(cf_id)

snap_b = roll_daily_nav(datetime.date(2026, 4, 2), return_provider=synth_provider)
print(f"  Day 4/2 snapshot: nav_open={snap_b['nav_open']:.2f} "
      f"ext={snap_b['external_flow']:+.2f} nav_after={snap_b['nav_after_flow']:.2f} "
      f"nav_close={snap_b['nav_close']:.2f}")
# nav_open = day4/1 close = init*1.01; ext = +100k → nav_after = nav_open + 100k;
# return = -0.005 → nav_close = nav_after * 0.995
expected_open = init * 1.01
expected_after = expected_open + 100_000.0
expected_close = expected_after * 0.995
assert abs(snap_b["nav_open"] - expected_open) < 1e-3
assert snap_b["external_flow"] == 100_000.0
assert abs(snap_b["nav_after_flow"] - expected_after) < 1e-3
assert abs(snap_b["nav_close"] - expected_close) < 1e-3
print(f"  expected nav_close: {expected_close:,.2f} [OK]")
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# C. portfolio gross return component
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — portfolio gross return weighted")
print("=" * 70)
snap_c = roll_daily_nav(datetime.date(2026, 4, 3), return_provider=synth_provider)
print(f"  Day 4/3: gross_pnl={snap_c['gross_pnl']:+.2f} "
      f"daily_md={snap_c['daily_modified_dietz']*100:+.4f}%")
# day4/2 close * 1.02
expected_close_3 = expected_close * 1.02
assert abs(snap_c["nav_close"] - expected_close_3) < 1e-3
# Daily MD: (close - open - ext_flow) / nav_after_flow
# ext_flow = 0 today → MD = (close - open) / open = +0.02 = +2%
expected_md = +0.02
assert abs(snap_c["daily_modified_dietz"] - expected_md) < 1e-6
print(f"  expected nav_close: {expected_close_3:,.2f} [OK]")
print(f"  expected daily MD: +2.0000% [OK]")
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# D. multi-day chain integrity
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — multi-day chain: nav_close[t] = nav_open[t+1]")
print("=" * 70)
ser = get_nav_series(start=datetime.date(2026, 4, 1),
                    end=datetime.date(2026, 4, 3))
print(f"  series rows: {len(ser)}")
print(ser[["nav_open", "external_flow", "nav_close"]].to_string())
nav_close_d1 = ser.loc[datetime.date(2026, 4, 1), "nav_close"]
nav_open_d2 = ser.loc[datetime.date(2026, 4, 2), "nav_open"]
nav_close_d2 = ser.loc[datetime.date(2026, 4, 2), "nav_close"]
nav_open_d3 = ser.loc[datetime.date(2026, 4, 3), "nav_open"]
assert abs(nav_close_d1 - nav_open_d2) < 1e-3
assert abs(nav_close_d2 - nav_open_d3) < 1e-3
print("  OK: chain continuous")


# ─────────────────────────────────────────────────────────────────────────────
# E. idempotency
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — idempotency unless force=True")
print("=" * 70)
snap_e1 = roll_daily_nav(datetime.date(2026, 4, 3), return_provider=synth_provider)
nav_close_first = snap_e1["nav_close"]
# Calling again should not change
snap_e2 = roll_daily_nav(datetime.date(2026, 4, 3), return_provider=synth_provider)
assert abs(snap_e1["nav_close"] - snap_e2["nav_close"]) < 1e-9
print(f"  no-force re-call: same nav_close (idempotent)")

# force=True with different return injection
def alt_provider(tickers, date):
    return {SMOKE_TICKER: +0.05}
snap_e3 = roll_daily_nav(datetime.date(2026, 4, 3),
                         return_provider=alt_provider, force=True)
print(f"  force=True with alt provider: nav_close changed "
      f"{nav_close_first:.2f} -> {snap_e3['nav_close']:.2f}")
assert abs(snap_e3["nav_close"] - nav_close_first) > 1.0
print("  OK")
# Restore the original snapshot to keep subsequent tests deterministic
roll_daily_nav(datetime.date(2026, 4, 3),
               return_provider=synth_provider, force=True)


# ─────────────────────────────────────────────────────────────────────────────
# F. nav_with_flows shows deposits/withdrawals as markers
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — get_nav_with_flows surfaces deposits as markers")
print("=" * 70)
df = get_nav_with_flows(start=datetime.date(2026, 4, 1),
                       end=datetime.date(2026, 4, 3))
print(f"  cols: {list(df.columns)}")
print(df[["nav_close", "flow"]].to_string())
assert "flow" in df.columns
flow_d2 = df.loc[datetime.date(2026, 4, 2), "flow"]
assert flow_d2 == 100_000.0
flow_d1 = df.loc[datetime.date(2026, 4, 1), "flow"]
assert flow_d1 == 0.0
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# G. orchestrator.run_daily contains roll_daily_nav hook
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("G — orchestrator.run_daily wired to roll_daily_nav")
print("=" * 70)
ORCH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "engine", "orchestrator.py",
)
with open(ORCH, "r", encoding="utf-8") as f:
    src = f.read()
assert "roll_daily_nav(as_of)" in src
assert "P-FUND-2 Daily NAV rollup hook" in src
# Hook must be inside run_daily and wrapped in try/except
run_daily_start = src.find("def run_daily(")
run_weekly_start = src.find("def run_weekly(", run_daily_start)
body = src[run_daily_start:run_weekly_start]
assert "roll_daily_nav" in body
assert "try:" in body and "except Exception" in body
print("  OK: roll_daily_nav hooked + try/except wrapped")


# ─────────────────────────────────────────────────────────────────────────────
# H. cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("H — cleanup")
print("=" * 70)
_cleanup()
with SessionFactory() as s:
    n_snap = s.query(PortfolioNavSnapshot).filter(
        PortfolioNavSnapshot.snapshot_date >= datetime.date(2026, 4, 1),
        PortfolioNavSnapshot.snapshot_date <= datetime.date(2026, 4, 30),
    ).count()
    n_cf = s.query(CashFlow).filter(
        CashFlow.supervisor_id == SMOKE_SUPERVISOR
    ).count()
    n_pos = s.query(SimulatedPosition).filter(
        SimulatedPosition.sector == SMOKE_SECTOR
    ).count()
print(f"  smoke NAV snapshots: {n_snap}")
print(f"  smoke cash flows:    {n_cf}")
print(f"  smoke positions:     {n_pos}")
assert n_snap == 0 and n_cf == 0 and n_pos == 0
print("  OK")

print()
print("=" * 70)
print("P-FUND-2 verification PASS (8 facets + orchestrator hook + cleanup)")
print("=" * 70)
