"""scripts/compare_book_pit_sn_vs_parent.py — head-to-head OLD (Config C
regime-conditional 5-sleeve with parent D_PEAD) vs NEW (Config D' with
PIT SN replacement) book.

Numbers reproduce the 2026-05-31 deploy-decision comparison published in
Memory project_combo_test_axes_not_independent_2026-05-31.md + the
session conversation.

Usage:
  python scripts/compare_book_pit_sn_vs_parent.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from engine.portfolio.combined_book import (
    DEFAULT_BOOK_VOL_TARGET,
    DEFAULT_CARRY_RISK_WEIGHT,
    DEFAULT_TSMOM_RISK_WEIGHT,
    build_carry_book,
    build_equity_book,
    build_equity_book_pit_sn,
    build_tsmom_book,
    scale_to_book_vol,
    voltarget,
)


def _stats(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    ann = float(r.mean() * 12)
    vol = float(r.std() * (12 ** 0.5))
    sharpe = ann / vol if vol > 0 else float("nan")
    cum = (1 + r).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    months_in_dd = int((cum < cum.cummax()).sum())
    return {"label": label, "n": len(r), "ann": ann, "vol": vol,
            "sharpe": sharpe, "maxdd": dd, "calmar": ann / abs(dd) if dd else float("nan"),
            "months_in_dd": months_in_dd,
            "worst_month": float(r.min()), "best_month": float(r.max()),
            "win_rate": float((r > 0).mean())}


def _print_row(s: dict, hi: str = ""):
    print(f"  {hi}{s['label']:<32} n={s['n']:>3}  ann={s['ann']:>+6.2%}  "
          f"vol={s['vol']:>5.2%}  Sharpe={s['sharpe']:>+5.3f}  "
          f"maxDD={s['maxdd']:>+6.2%}  Calmar={s['calmar']:>+4.2f}  "
          f"win={s['win_rate']:>4.1%}")


def main() -> int:
    print("=" * 100)
    print(" BOOK COMPARISON — OLD (parent D_PEAD) vs NEW (PIT SN replacement)")
    print(" 3-sleeve simplified comparison (Equity + Carry + TSMOM only, no regime overlay)")
    print("=" * 100)

    # Build the 3 sleeves with both equity variants
    eq_parent = build_equity_book()
    eq_pit_sn = build_equity_book_pit_sn()
    carry = build_carry_book()
    tsmom = build_tsmom_book()

    # Sleeve-level stats
    print("\n  Per-sleeve stats (cost-net monthly returns):")
    _print_row(_stats(eq_parent, "Equity (parent D_PEAD)"))
    _print_row(_stats(eq_pit_sn, "Equity (PIT SN)"))
    _print_row(_stats(carry, "Carry (cross-asset 4-leg)"))
    _print_row(_stats(tsmom, "TSMOM (5-leg)"))

    # Build book OLD = vol-target each sleeve then blend
    # OLD book: equity (parent) 70%, carry 25%, tsmom 5%
    eq_p_vt = voltarget(eq_parent, DEFAULT_BOOK_VOL_TARGET)
    cy_vt = voltarget(carry, DEFAULT_BOOK_VOL_TARGET)
    ts_vt = voltarget(tsmom, DEFAULT_BOOK_VOL_TARGET)

    w_eq_p = 1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT
    book_old = (w_eq_p * eq_p_vt + DEFAULT_CARRY_RISK_WEIGHT * cy_vt
                + DEFAULT_TSMOM_RISK_WEIGHT * ts_vt).dropna()
    book_old = scale_to_book_vol(book_old, DEFAULT_BOOK_VOL_TARGET)

    # NEW book: equity (PIT SN) 70%, carry 25%, tsmom 5%
    eq_n_vt = voltarget(eq_pit_sn, DEFAULT_BOOK_VOL_TARGET)
    book_new = (w_eq_p * eq_n_vt + DEFAULT_CARRY_RISK_WEIGHT * cy_vt
                + DEFAULT_TSMOM_RISK_WEIGHT * ts_vt).dropna()
    book_new = scale_to_book_vol(book_new, DEFAULT_BOOK_VOL_TARGET)

    print("\n" + "-" * 100)
    print(" BOOK-LEVEL HEAD-TO-HEAD (both at 10% target vol, equal weights, same regime overlay OFF)")
    print("-" * 100)
    s_old = _stats(book_old, "Config-OLD (parent D_PEAD)")
    s_new = _stats(book_new, "Config-NEW (PIT SN replace)")
    _print_row(s_old)
    _print_row(s_new, hi=">> ")

    # Sample-aligned delta
    aligned = pd.concat([book_old.rename("old"), book_new.rename("new")], axis=1).dropna()
    if len(aligned) > 12:
        old_a = aligned["old"]; new_a = aligned["new"]
        print(f"\n  SAMPLE-ALIGNED ({len(aligned)} months "
              f"{aligned.index.min().date()} → {aligned.index.max().date()}):")
        _print_row(_stats(old_a, "Config-OLD (aligned)"))
        _print_row(_stats(new_a, "Config-NEW (aligned)"), hi=">> ")
        sharpe_delta = _stats(new_a, "")["sharpe"] - _stats(old_a, "")["sharpe"]
        print(f"\n  Δ Sharpe (NEW - OLD): {sharpe_delta:+.3f}")

    print("\n" + "=" * 100)
    print(" HONEST DEPLOY EXPECTATION (after 5-stage haircut from P-D8 audit)")
    print("=" * 100)
    haircut_pit_sn = 1.38 / 2.10                    # sleeve haircut ratio
    print(f"  PIT SN sleeve haircut (5-stage): {(1-haircut_pit_sn)*100:.0f}%")
    print(f"  Apply only to PIT SN portion of book (70% equity × 100% SN portion):")
    # equity weight × sn-portion × (1 - haircut_ratio)
    eq_weight = 0.70
    sn_in_eq = 0.50    # PIT SN is ~50% of equity_book (rest is analyst_revision)
    book_haircut = eq_weight * sn_in_eq * (1 - haircut_pit_sn)
    print(f"  Book-level cumulative haircut from SN: {book_haircut*100:.1f}%")
    print(f"  Plus 5% capacity + 10% implementation (universal):")
    other_haircut = 0.05 + 0.10
    print(f"  Total book haircut: {(book_haircut + other_haircut)*100:.1f}%")
    if len(aligned) > 12:
        honest_sharpe = _stats(new_a, "")["sharpe"] * (1 - book_haircut - other_haircut)
        print(f"\n  Backtest book Sharpe (cost-net): {_stats(new_a, '')['sharpe']:.3f}")
        print(f"  HONEST DEPLOY estimate:          {honest_sharpe:.3f}")
        print(f"  vs Config-OLD honest:            "
              f"{_stats(old_a, '')['sharpe'] * (1 - 0.15):.3f}")
        print(f"  Δ honest deploy:                 "
              f"{honest_sharpe - _stats(old_a, '')['sharpe'] * (1 - 0.15):+.3f}")

    print("\n  [Note] No regime-overlay sleeves (crisis_hedge / mom_hedge) — those are Config-C")
    print("  specific add-ons. This compare is pure equity-replacement, holding regime overlay constant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
