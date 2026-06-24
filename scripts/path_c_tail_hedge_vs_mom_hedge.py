"""scripts/path_c_tail_hedge_vs_mom_hedge.py — head-to-head: put-spread
tail hedge vs current mom_hedge_overlay sleeve.

Applies Path C pre-commit acceptance criteria:
  D1. Crisis-period (SPX dd >= 5%) PnL > mom_hedge same period
  D2. Annualized drag <= mom_hedge drag
  D3. Cosine with book lower than mom_hedge cosine
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine.portfolio.combined_book import (
    build_carry_book, build_equity_book, build_mom_hedge_book,
    build_tsmom_book,
)


def _sharpe(s: pd.Series) -> float:
    if s.std() == 0:
        return 0.0
    return float((s.mean() * 12) / (s.std() * math.sqrt(12)))


def main() -> int:
    put_spread = pd.read_parquet(
        "data/cache/_tail_hedge_put_spread_monthly.parquet"
    ).iloc[:, 0]
    put_spread.index = pd.to_datetime(put_spread.index)

    print("=" * 95)
    print(" PATH C — put-spread tail hedge vs mom_hedge_overlay head-to-head")
    print("=" * 95)

    # mom_hedge_overlay (current sleeve in book)
    mom_hedge = build_mom_hedge_book()
    mom_hedge.index = pd.to_datetime(mom_hedge.index)

    # Align
    aligned = pd.concat([
        put_spread.rename("put_spread"),
        mom_hedge.rename("mom_hedge"),
    ], axis=1).dropna()
    n = len(aligned)
    print(f"\n  Aligned window: {n} months "
          f"({aligned.index.min().date()} → {aligned.index.max().date()})")

    # Standalone stats
    print(f"\n  [standalone stats — 5% notional for put_spread]")
    for col in ["put_spread", "mom_hedge"]:
        s = aligned[col]
        ann_ret = s.mean() * 12
        ann_vol = s.std() * math.sqrt(12)
        win = (s > 0).mean()
        worst = s.min()
        best = s.max()
        print(f"    {col:<12}  ann_ret={ann_ret:>+7.4f}  ann_vol={ann_vol:>6.4f}  "
              f"Sharpe={_sharpe(s):>+6.3f}  win={win:>4.1%}  "
              f"worst={worst:>+7.4f}  best={best:>+7.4f}")

    # ── D2: drag comparison ─────────────────────────────────────────
    print(f"\n  [D2] Annualized drag comparison:")
    drag_ps = aligned["put_spread"].mean() * 12
    drag_mh = aligned["mom_hedge"].mean() * 12
    print(f"    put_spread: {drag_ps:+.4f} = {drag_ps*100:+.2f}%/yr")
    print(f"    mom_hedge:  {drag_mh:+.4f} = {drag_mh*100:+.2f}%/yr")
    d2_pass = drag_ps >= drag_mh  # less negative = better
    print(f"    → {'D2 PASS' if d2_pass else 'D2 FAIL'} (put_spread drag {'<=' if d2_pass else '>'} mom_hedge)")

    # ── D1: crisis-period payoff ────────────────────────────────────
    # Define crisis = months where equity_book had >= 5% drawdown OR
    # SPX dropped >= 5% (we don't have SPX in book builders here, so
    # use equity_book as proxy)
    equity_book = build_equity_book()
    equity_book.index = pd.to_datetime(equity_book.index)
    aligned2 = aligned.join(equity_book.rename("equity"), how="inner")
    crisis_mask = aligned2["equity"] <= -0.05
    n_crisis = int(crisis_mask.sum())
    print(f"\n  [D1] Crisis-period PnL (equity_book month return <= -5%, "
          f"n={n_crisis}):")
    if n_crisis > 0:
        crisis_ps = aligned2.loc[crisis_mask, "put_spread"].mean()
        crisis_mh = aligned2.loc[crisis_mask, "mom_hedge"].mean()
        print(f"    put_spread mean in crisis: {crisis_ps:+.4f}")
        print(f"    mom_hedge  mean in crisis: {crisis_mh:+.4f}")
        d1_pass = crisis_ps > crisis_mh
        print(f"    → {'D1 PASS' if d1_pass else 'D1 FAIL'} "
              f"(put_spread {'better' if d1_pass else 'worse'} in crisis)")
    else:
        print(f"    insufficient crisis months in window; D1 SKIPPED")
        d1_pass = None

    # ── D3: cosine with book ────────────────────────────────────────
    print(f"\n  [D3] Cosine similarity with book (lower = better diversifier):")
    # Build a proxy book = equity + carry + tsmom (without hedges) to
    # avoid circularity
    carry = build_carry_book()
    tsmom = build_tsmom_book()
    carry.index = pd.to_datetime(carry.index)
    tsmom.index = pd.to_datetime(tsmom.index)
    proxy_book = (0.70 * equity_book + 0.25 * carry + 0.05 * tsmom).dropna()
    a2 = aligned.join(proxy_book.rename("book"), how="inner")

    def _cosine(x, y):
        x = x.values; y = y.values
        nx = np.linalg.norm(x); ny = np.linalg.norm(y)
        if nx * ny == 0:
            return float("nan")
        return float(x @ y / (nx * ny))

    cos_ps = _cosine(a2["put_spread"], a2["book"])
    cos_mh = _cosine(a2["mom_hedge"], a2["book"])
    print(f"    put_spread cosine with book: {cos_ps:+.4f}")
    print(f"    mom_hedge  cosine with book: {cos_mh:+.4f}")
    d3_pass = cos_ps < cos_mh
    print(f"    → {'D3 PASS' if d3_pass else 'D3 FAIL'} "
          f"(put_spread {'more' if d3_pass else 'less'} orthogonal)")

    # ── Verdict ─────────────────────────────────────────────────────
    print(f"\n  ── VERDICT ──")
    flags = [("D1 crisis-period", d1_pass),
             ("D2 drag",          d2_pass),
             ("D3 cosine",        d3_pass)]
    n_pass = sum(1 for _, x in flags if x)
    for label, val in flags:
        print(f"    {label:<25} {val if val is not None else 'SKIPPED'}")
    print(f"  → {n_pass}/3 acceptance criteria passed")
    if n_pass >= 2:
        print(f"  STRATEGY GREEN — candidate to REPLACE mom_hedge_overlay")
    elif n_pass == 1:
        print(f"  STRATEGY YELLOW — partial improvement; needs deeper look")
    else:
        print(f"  STRATEGY REJECTED — does not improve on mom_hedge")

    # Statistical nuance per [[feedback-sharpe-se-for-strategy-comparison]]
    print(f"\n  [statistical nuance]")
    se_ps = math.sqrt((1 + 0.5 * _sharpe(aligned['put_spread'])**2) / (n/12))
    se_mh = math.sqrt((1 + 0.5 * _sharpe(aligned['mom_hedge'])**2) / (n/12))
    print(f"    SE(Sharpe) put_spread: ≈{se_ps:.3f}")
    print(f"    SE(Sharpe) mom_hedge:  ≈{se_mh:.3f}")
    print(f"    D2 drag comparison is reasonable bcoz drag-of-hedge is")
    print(f"    an EXPECTED-VALUE param, not a noisy Sharpe race.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
