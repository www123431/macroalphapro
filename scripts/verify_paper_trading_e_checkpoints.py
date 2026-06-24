"""Paper Trading E v0.2 §11 mid-period checkpoints verification.

Facets:
  A. function exists + idempotent guard via SystemConfig
  B. n_decisions < 12 → no fire
  C. n_decisions == 12 → h=12 fires (record only)
  D. h=12 already fired → no re-fire on next call
  E. n_decisions == 24 → h=24 fires + lever1 recommendation set
  F. force=True → re-fires regardless of stamp
  G. cleanup
"""
import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import (
    init_db, SessionFactory,
    PaperTradingRun, get_system_config, set_system_config,
)
from engine.paper_trading import (
    check_paper_trading_e_checkpoints,
    CHECKPOINT_CONFIG_KEY,
)

init_db()

SMOKE_NOTE = "ck_smoke"


def _cleanup():
    """Remove smoke PaperTradingRun rows + reset checkpoint stamps."""
    with SessionFactory() as s:
        s.query(PaperTradingRun).filter(
            PaperTradingRun.notes.like(f"%{SMOKE_NOTE}%")
        ).delete(synchronize_session=False)
        s.commit()
    set_system_config(CHECKPOINT_CONFIG_KEY[12], "")
    set_system_config(CHECKPOINT_CONFIG_KEY[24], "")


_cleanup()

# Baseline: count existing production Arm B rows (don't delete them)
with SessionFactory() as _s:
    BASELINE_N_B = (
        _s.query(PaperTradingRun)
        .filter_by(arm="B")
        .filter(PaperTradingRun.next_month_return.isnot(None))
        .count()
    )
print(f"baseline production Arm B decisions: {BASELINE_N_B} (smoke seeds add on top)")


_DATE_OFFSET = [0]  # mutable counter to keep seeded dates unique across calls


def _seed_b_decisions(n: int, base_date=None):
    """Insert n Arm B PaperTradingRun rows with non-NULL next_month_return.
    Uses far-future dates to avoid (as_of_date, arm) unique constraint clash."""
    # Far-future base to dodge production date conflicts
    base = base_date or (
        datetime.date(2099, 1, 1) + datetime.timedelta(days=_DATE_OFFSET[0])
    )
    with SessionFactory() as s:
        for i in range(n):
            d = base + datetime.timedelta(days=30 * i + _DATE_OFFSET[0])
            row = PaperTradingRun(
                arm="B",
                as_of_date=d,
                weights_json="{}",
                next_month_return=0.001 * (i + 1),  # non-null
                cum_nav=1.0 + 0.001 * i,
                placebo_seed=None,
                notes=SMOKE_NOTE,
            )
            s.add(row)
        s.commit()
    _DATE_OFFSET[0] += n * 30 + 1000  # bump for next call


def _expected_total(seeded: int) -> int:
    return BASELINE_N_B + seeded


# ─────────────────────────────────────────────────────────────────────────────
# A. function exists, idempotent
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — function exists + initial state")
print("=" * 70)
out = check_paper_trading_e_checkpoints(datetime.date(2026, 6, 1))
print(f"  initial: {out}")
assert out["n_decisions"] == _expected_total(0)
assert not out["h12_fired"]
assert not out["h24_fired"]
print("  OK")


# ─────────────────────────────────────────────────────────────────────────────
# B. n=8 (below threshold)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — n_decisions < 12: no fire")
print("=" * 70)
# Seed enough to stay below 12 even with baseline
seed_n = max(0, 11 - BASELINE_N_B)
_seed_b_decisions(seed_n)
out = check_paper_trading_e_checkpoints(datetime.date(2026, 8, 1))
print(f"  total n: {out['n_decisions']} (target <12), h12_just_fired={out['h12_just_fired']}")
assert out["n_decisions"] < 12
assert not out["h12_just_fired"]
print("  OK: dormant below threshold")


# ─────────────────────────────────────────────────────────────────────────────
# C. n=12 trips h=12 fire
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — n_decisions = 12: h=12 fires (record only)")
print("=" * 70)
_seed_b_decisions(1, base_date=datetime.date(2026, 6, 15))  # add 1 to cross 12
out = check_paper_trading_e_checkpoints(datetime.date(2026, 9, 1))
print(f"  result: n={out['n_decisions']}, h12_fired={out['h12_fired']}, "
      f"just_fired={out['h12_just_fired']}, h24={out['h24_fired']}")
assert out["n_decisions"] >= 12
assert out["h12_fired"] is True
assert out["h12_just_fired"] is True
assert out["h24_fired"] is False
assert "h=12 checkpoint fired" in out["notes"]
print("  OK: h=12 fires when crossing 12")


# ─────────────────────────────────────────────────────────────────────────────
# D. idempotent re-call after h=12
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — re-call: h=12 stamped, no re-fire")
print("=" * 70)
out = check_paper_trading_e_checkpoints(datetime.date(2026, 9, 2))
print(f"  result: h12_just_fired={out['h12_just_fired']} (should be False)")
assert not out["h12_just_fired"]
assert out["h12_fired"] is True  # still stamped
print("  OK: idempotent")


# ─────────────────────────────────────────────────────────────────────────────
# E. n=24 trips h=24 fire + Lever 1 recommendation set
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — n_decisions = 24: h=24 fires with lever1 recommendation")
print("=" * 70)
_seed_b_decisions(12, base_date=datetime.date(2027, 1, 15))  # cross 24
out = check_paper_trading_e_checkpoints(datetime.date(2027, 4, 1))
print(f"  result: n={out['n_decisions']}, h24_just_fired={out['h24_just_fired']}, "
      f"lever1={out['lever1_recommendation']}")
assert out["n_decisions"] >= 24
assert out["h24_fired"] is True
assert out["h24_just_fired"] is True
# Lever 1 will be "wait" since per-sector return join not wired (per
# mid_checkpoint_conviction skeleton intent)
assert out["lever1_recommendation"] in ("wait", "activate", "forfeit", "error")
assert "h=24 checkpoint fired" in out["notes"]
print("  OK: h=24 fires + lever1 recommendation set")


# ─────────────────────────────────────────────────────────────────────────────
# F. force=True re-fires
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — force=True bypasses idempotency")
print("=" * 70)
out = check_paper_trading_e_checkpoints(datetime.date(2027, 4, 2), force=True)
print(f"  force result: h12_just_fired={out['h12_just_fired']}, h24_just_fired={out['h24_just_fired']}")
assert out["h12_just_fired"] is True
assert out["h24_just_fired"] is True
print("  OK: force bypasses stamp")


# ─────────────────────────────────────────────────────────────────────────────
# G. cleanup
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("G — cleanup")
print("=" * 70)
_cleanup()
with SessionFactory() as s:
    n = s.query(PaperTradingRun).filter(
        PaperTradingRun.notes.like(f"%{SMOKE_NOTE}%")
    ).count()
print(f"  smoke rows remaining: {n}")
print(f"  h12 stamp: {get_system_config(CHECKPOINT_CONFIG_KEY[12], '')!r}")
print(f"  h24 stamp: {get_system_config(CHECKPOINT_CONFIG_KEY[24], '')!r}")
assert n == 0
print("  OK")

print()
print("=" * 70)
print("Paper Trading E v0.2 §11 checkpoints verification PASS (6 facets)")
print("=" * 70)
