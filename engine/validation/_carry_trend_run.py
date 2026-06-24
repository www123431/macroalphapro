"""engine/validation/_carry_trend_run.py — STRENGTHEN mechanism 2: combine cross-asset
CARRY with cross-asset TREND (TSMOM). Carry is short-vol (loses in crises); trend is
long-vol (profits from the carry-unwind) -> the classic complementary pair (AQR "Carry
Everywhere" + Moskowitz-Ooi-Pedersen TSMOM). Combining directly hedges carry's only
caveat (regime dependence). Built from the SAME cached commodity + FX front returns —
no new data. TSMOM = sign(trailing-12m return) held one month, equal-weight across
instruments; carry = the cross-asset risk-parity carry already validated GREEN.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
from engine.validation.crossasset_carry import build_fx_carry, _xs_ls
from engine.validation.deflated_sharpe import probabilistic_sharpe_ratio


def shp(x):
    x = x.dropna(); return x.mean() * 12 / (x.std() * np.sqrt(12)) if x.std() > 0 else np.nan


def tstat(x):
    x = x.dropna(); return x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else np.nan


def tsmom(rwide, look=12):
    """Equal-weight time-series momentum: sign(trailing-`look`m return) * next-month
    return, averaged across instruments. Returns a monthly portfolio series."""
    tr = (1 + rwide.fillna(0)).rolling(look).apply(np.prod, raw=True) - 1
    sig = np.sign(tr)
    rows = []
    idx = list(rwide.index)
    for i in range(len(idx) - 1):
        t, nx = idx[i], idx[i + 1]
        s = sig.loc[t].dropna()
        nr = rwide.loc[nx].reindex(s.index).dropna()
        common = s.index.intersection(nr.index)
        if len(common) < 4:
            continue
        rows.append((nx, float((s.reindex(common) * nr.reindex(common)).mean())))
    return pd.Series(dict(rows)).sort_index()


def rp_combine(series_list):
    """Risk-parity (inverse-vol) combine of monthly series on the common window."""
    J = pd.concat(series_list, axis=1).dropna()
    w = {c: 1 / J[c].std() for c in J.columns}; W = sum(w.values())
    return (sum(w[c] * J[c] for c in J.columns) / W), J


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cw_c, rw_c = commodity_cr()
    cw_f, rw_f, _ = build_fx_carry()

    # CARRY legs (already GREEN combined)
    carry_c = _xs_ls(cw_c, rw_c, q=0.3).rename("carry_cmdty")
    carry_f = _xs_ls(cw_f, rw_f, q=0.4).rename("carry_fx")
    carry, _ = rp_combine([carry_c, carry_f]); carry = carry.rename("carry")

    # TREND legs
    trend_c = tsmom(rw_c).rename("trend_cmdty")
    trend_f = tsmom(rw_f).rename("trend_fx")
    trend, _ = rp_combine([trend_c, trend_f]); trend = trend.rename("trend")

    print("\n=== standalone (cross-asset, commodity+FX) ===")
    for nm, s in (("CARRY", carry), ("TREND(TSMOM)", trend)):
        print(f"  {nm:14s} Sharpe {shp(s):.2f} t={tstat(s):.2f} (n={s.dropna().size})")
    j = pd.concat([carry, trend], axis=1).dropna()
    print(f"  corr(carry, trend) = {j['carry'].corr(j['trend']):+.2f}  (complementary if low/neg)")

    comb, _ = rp_combine([carry, trend]); comb = comb.rename("carry+trend")
    print(f"\n=== COMBINED carry+trend (risk-parity) ===")
    print(f"  Sharpe {shp(comb):.2f} t={tstat(comb):.2f} PSR0 {probabilistic_sharpe_ratio(comb.dropna().values,0.0):.3f} (n={comb.dropna().size})")

    # the KEY test: does TREND rescue CARRY's bad regime?
    print("\n=== regime-hedge check (Sharpe by sub-period) ===")
    for cut0, cut1, lab in [(None, "2013-12-31", "2000-2013 (carry strong)"),
                            ("2013-12-31", "2018-12-31", "2014-2018 (carry weak)"),
                            ("2018-12-31", None, "2019-2026")]:
        def slc(s):
            x = s.dropna()
            if cut0: x = x[x.index > cut0]
            if cut1: x = x[x.index <= cut1]
            return x
        print(f"  {lab:26s} carry {shp(slc(carry)):+.2f} | trend {shp(slc(trend)):+.2f} | combined {shp(slc(comb)):+.2f}")
    yr = (comb.dropna().groupby(comb.dropna().index.year).mean() * 12)
    print(f"\n  combined yearly positive {int((yr>0).sum())}/{len(yr)}  worst yr {yr.min()*1:.1%}")


if __name__ == "__main__":
    main()
