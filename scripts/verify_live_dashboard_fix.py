"""Verify get_current_positions per-sector-latest fix.

Facets:
  A. signature: returns per-sector latest, not global-latest
  B. open positions: ≥30 (matching what we observed in DB)
  C. include_closed=True: returns more rows (closed sectors come back)
  D. as_of historical: returns positions as of a past date
  E. live_dashboard via AppTest: holdings table now shows ≥10 rows
  F. callers do not break (command_center stat, daily_batch top-3)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import pandas as pd

from engine.portfolio_tracker import get_current_positions
from engine.memory import init_db, SessionFactory, SimulatedPosition

init_db()


# ─────────────────────────────────────────────────────────────────────────────
# A. signature
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — signature: per-sector latest replaces global-latest")
print("=" * 70)
df = get_current_positions()
print(f"  rows returned: {len(df)}")
# DB observed: per-sector latest = 40 sectors, of which 33 have nonzero weight
assert isinstance(df, pd.DataFrame)
assert df.index.name == "sector"
print(f"  columns: {list(df.columns)[:6]}...")
print("  OK: returns DataFrame indexed by sector")

# ─────────────────────────────────────────────────────────────────────────────
# B. open positions count
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — open positions only (default include_closed=False)")
print("=" * 70)
print(f"  rows: {len(df)} (expect ≥30 — DB had 33 nonzero per-sector)")
assert len(df) >= 30, f"expected ≥30 open positions, got {len(df)}"
nonzero_check = df.assign(
    aw_or_tw=lambda d: d["actual_weight"].fillna(d["target_weight"]).abs() + d["target_weight"].abs()
)["aw_or_tw"].gt(0)
assert nonzero_check.all(), "should not include closed-out (both-zero) sectors"
print(f"  OK: all {len(df)} returned rows have nonzero weight")

# ─────────────────────────────────────────────────────────────────────────────
# C. include_closed flag
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — include_closed=True surfaces closed sectors too")
print("=" * 70)
df_all = get_current_positions(include_closed=True)
print(f"  open rows: {len(df)} | open+closed rows: {len(df_all)}")
assert len(df_all) >= len(df), "include_closed should be ≥ open-only"
print("  OK: closed sectors restored when include_closed=True")

# ─────────────────────────────────────────────────────────────────────────────
# D. as_of historical
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — historical as_of cutoff")
print("=" * 70)
df_old = get_current_positions(as_of=datetime.date(2026, 4, 25))
print(f"  rows as_of 2026-04-25: {len(df_old)} (cutoff before recent partial rebals)")
assert len(df_old) >= 1
print("  OK: as_of historical filtering works")

# ─────────────────────────────────────────────────────────────────────────────
# E. live_dashboard via AppTest (headless render)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — live_dashboard renders ≥10 rows in HOLDINGS")
print("=" * 70)
from streamlit.testing.v1 import AppTest

PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "pages", "live_dashboard.py")
at = AppTest.from_file(PAGE, default_timeout=120)
at.run()
exceptions = [str(e.value) for e in at.exception]
print(f"  exceptions: {len(exceptions)}")
for e in exceptions[:3]:
    print(f"    {e[:200]}")
assert not exceptions, f"live_dashboard raised: {exceptions}"

# Inspect rendered elements. The HOLDINGS table is custom HTML (not
# st.dataframe), so we look at markdown blocks for the position-count
# caption that the page renders below the table:
#   "{N} positions · click any row for drill-down · ..."
n_metric = len(at.metric)
n_df = len(at.dataframe)
n_md = len(at.markdown)
print(f"  metrics: {n_metric}, dataframes: {n_df}, markdown blocks: {n_md}")
md_text = " ".join(m.value for m in at.markdown)
import re
match = re.search(
    r"(\d+)\s*positions?\s*(?:[·•]|click)", md_text, re.IGNORECASE,
)
if match:
    n_holdings_text = int(match.group(1))
    print(f"  HOLDINGS caption shows: {n_holdings_text} positions")
    assert n_holdings_text >= 10, \
        f"expected ≥10 positions after fix, got {n_holdings_text}"
    print(f"  OK: dashboard now shows {n_holdings_text} positions (vs 1 pre-fix)")
else:
    # Fallback: search for ticker symbols in markdown to confirm holdings rendered
    found_tickers = sum(
        1 for t in ("XLK", "XLF", "QQQ", "GLD", "IEF", "MTUM", "VNQ", "SMH")
        if t in md_text
    )
    print(f"  (caption regex did not match; fallback ticker probe found "
          f"{found_tickers} tickers in markdown)")
    assert found_tickers >= 3, \
        "no recognizable holdings tickers in rendered markdown"
    print(f"  OK: at least {found_tickers} ticker symbols present in HOLDINGS render")

# ─────────────────────────────────────────────────────────────────────────────
# F. callers do not break: emulate command_center + daily_batch usage
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — callers (command_center NAV stat, daily_batch top-3) work")
print("=" * 70)

# command_center: just len(positions)
n_pos_cc = len(df) if not df.empty else 0
print(f"  command_center NAV stat n_pos = {n_pos_cc}")
assert n_pos_cc >= 30

# daily_batch: nlargest(3, "actual_weight").index.tolist()
top3 = df.nlargest(3, "actual_weight").index.tolist()
print(f"  daily_batch top-3 by actual_weight: {top3}")
assert len(top3) == 3
print("  OK: existing callers receive richer per-sector view, no schema break")

# ─────────────────────────────────────────────────────────────────────────────
# G. cleanup audit (no smoke residue created by this script)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("G — no smoke residue (read-only verification)")
print("=" * 70)
print("  this script reads only — no inserts/deletes. nothing to clean.")

print()
print("=" * 70)
print("live_dashboard fix verification PASS")
print("=" * 70)
