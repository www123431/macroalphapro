"""One-shot cleanup: orphan SimulatedPosition row (id=48) had ticker=Technology
which yfinance can't resolve. Fix: ticker -> XLK (canonical SPDR Tech ETF).

This is a data-cleanup script, not a verification script. It:
  1. Snapshots the row before update
  2. Updates ticker only (preserves sector / weights / dates / track)
  3. Verifies the update + yfinance resolution
  4. Re-runs get_current_positions to confirm the row now has ticker=XLK
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

from engine.memory import init_db, SessionFactory, SimulatedPosition
from engine.portfolio_tracker import get_current_positions

init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: snapshot orphan row
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Step 1 — snapshot the orphan row")
print("=" * 70)
with SessionFactory() as s:
    orphan = s.query(SimulatedPosition).filter(
        SimulatedPosition.sector == "Technology",
        SimulatedPosition.ticker == "Technology",
    ).first()
    if not orphan:
        print("  no orphan row found — nothing to fix")
        sys.exit(0)
    orphan_id = orphan.id
    snapshot_before = {
        "id":             orphan.id,
        "sector":         orphan.sector,
        "ticker":         orphan.ticker,
        "snapshot_date":  str(orphan.snapshot_date),
        "target_weight":  orphan.target_weight,
        "actual_weight":  orphan.actual_weight,
        "track":          orphan.track,
        "direction":      orphan.direction,
    }
print(f"  before: {snapshot_before}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: update ticker only
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Step 2 — UPDATE ticker = 'XLK' (preserving everything else)")
print("=" * 70)
with SessionFactory() as s:
    orphan = s.query(SimulatedPosition).filter(
        SimulatedPosition.id == orphan_id
    ).one()
    orphan.ticker = "XLK"
    s.commit()
print(f"  committed UPDATE id={orphan_id}: ticker -> XLK")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: read back + yfinance probe
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Step 3 — verify update + yfinance resolution")
print("=" * 70)
with SessionFactory() as s:
    after = s.query(SimulatedPosition).filter(
        SimulatedPosition.id == orphan_id
    ).one()
    snapshot_after = {
        "id":             after.id,
        "sector":         after.sector,
        "ticker":         after.ticker,
        "snapshot_date":  str(after.snapshot_date),
        "target_weight":  after.target_weight,
        "actual_weight":  after.actual_weight,
        "track":          after.track,
        "direction":      after.direction,
    }
print(f"  after:  {snapshot_after}")
assert snapshot_after["ticker"] == "XLK"
# Everything else should be untouched
for key in ("sector", "snapshot_date", "target_weight", "actual_weight", "track"):
    assert snapshot_after[key] == snapshot_before[key], \
        f"{key} should be unchanged (was {snapshot_before[key]}, now {snapshot_after[key]})"
print("  OK: only ticker changed; sector / weights / date / track preserved")

fi = yf.Ticker("XLK").fast_info
print(f"  yfinance XLK.last_price = {fi.last_price}")
assert fi.last_price and fi.last_price > 0
print("  OK: yfinance can fetch XLK quote")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: get_current_positions sees the fixed row with ticker=XLK
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Step 4 — get_current_positions sees ticker=XLK for the Technology sector")
print("=" * 70)
df = get_current_positions()
print(f"  per-sector latest rows: {len(df)}")
if "Technology" in df.index:
    tech_row = df.loc["Technology"]
    print(f"  Technology row: ticker={tech_row['ticker']} target_w={tech_row['target_weight']:+.4f}")
    assert tech_row["ticker"] == "XLK"
    print("  OK: live_dashboard / command_center / daily_batch will all see ticker=XLK now")
else:
    print("  (Technology not in per-sector view — that's actually fine; the orphan was a single row)")


print()
print("=" * 70)
print("Orphan ticker cleanup PASS")
print("=" * 70)
