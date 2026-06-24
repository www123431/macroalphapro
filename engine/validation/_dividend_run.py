"""engine/validation/_dividend_run.py — full audit of the dividend-change-drift
sleeve (Michaely-Thaler-Womack), with the CORRECTED deflated-Sharpe methodology.
The DECISIVE question for a true 2nd alpha: is payout-signaling drift (a) real and
tradeable after cost, and (b) ORTHOGONAL to the D_PEAD + analyst-revision earnings-
information book (a genuinely different mechanism, unlike guidance which was +0.48)?
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.validation.dividend_drift import build_div_change_signal, build_div_drift_sleeve, _monthly_returns
from engine.validation.after_cost import apply_cost
from engine.validation.deflated_sharpe import (probabilistic_sharpe_ratio,
                                                expected_max_sharpe, var_sr_from_trial_sharpes)

RT = 26.0  # ss_large/mid round-trip bps (dividend payers = stable large/mid), conservative


def _stats(net):
    net = net.dropna(); vol = net.std() * np.sqrt(12)
    return dict(n=len(net), ann=net.mean() * 12, sharpe=net.mean() * 12 / vol if vol > 0 else float("nan"),
                t=net.mean() / net.std() * np.sqrt(len(net)) if net.std() > 0 else float("nan"),
                psr0=probabilistic_sharpe_ratio(net.values, 0.0))


def _dpead_monthly():
    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet")
    s = d.iloc[:, 0]; s.index = pd.to_datetime(s.index)
    return ((1 + s).resample("ME").prod() - 1).rename("dpead")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sig = build_div_change_signal()
    mret = _monthly_returns()
    print(f"\ndividend-change events {len(sig)}, permnos {sig['permno'].nunique()}, "
          f"{sig['dclrdt'].min():%Y-%m}..{sig['dclrdt'].max():%Y-%m}")

    # pre-specified sleeve (hold=6, q=0.2, equal) — MTW drift is slow (~1yr)
    ls, lo, turn = build_div_drift_sleeve(hold=6, q=0.2, weight="equal", signal=sig, mret=mret)
    print(f"\n=== PRE-SPECIFIED dividend-drift sleeve (hold=6, q=0.2, equal) turn={turn:.1f}x ===")
    for nm, ser, legs in [("L/S", ls, 2), ("LONG-ONLY (deployable)", lo, 1)]:
        g = _stats(ser.dropna())
        net = apply_cost(ser.dropna(), legs * turn * RT / 10000.0, ppy=12)
        ns = _stats(net)
        print(f"  {nm:22s} gross ann {g['ann']:+.1%} Sharpe {g['sharpe']:.2f} | "
              f"cost {legs*turn*RT/10000.0:.2%} | NET ann {ns['ann']:+.1%} Sharpe {ns['sharpe']:.2f} "
              f"t={ns['t']:.2f} PSR0={ns['psr0']:.3f}")

    # grid (deployable long-only) + CORRECT deflated SR (actual cross-trial V)
    grid = [dict(hold=h, q=qq, weight=w) for h in (3, 6, 12) for qq in (0.1, 0.2) for w in ("equal", "mag")]
    nets = []
    for p in grid:
        _, longonly, tn = build_div_drift_sleeve(signal=sig, mret=mret, **p)
        net = apply_cost(longonly.dropna(), tn * RT / 10000.0, ppy=12)
        if len(net) >= 24:
            nets.append((p, net))
    sr_pp = np.array([n.mean() / n.std() for _, n in nets])
    V_act = var_sr_from_trial_sharpes(sr_pp)
    M = pd.DataFrame({i: n for i, (_, n) in enumerate(nets)}).dropna()
    avg_corr = (M.corr().values.sum() - len(M.columns)) / (len(M.columns) ** 2 - len(M.columns))
    N = len(nets)
    base_net = [n for p, n in nets if p == dict(hold=6, q=0.2, weight="equal")][0]
    dsr = probabilistic_sharpe_ratio(base_net.values, expected_max_sharpe(N, V_act))
    print(f"\n=== grid (long-only) N={N} avg corr={avg_corr:.3f} V_act={V_act:.5f} ===")
    print(f"  pre-specified long-only: net Sharpe {base_net.mean()*12/(base_net.std()*np.sqrt(12)):.2f} "
          f"net t {base_net.mean()/base_net.std()*np.sqrt(len(base_net)):.2f} | "
          f"PSR0 {probabilistic_sharpe_ratio(base_net.values,0.0):.3f} | deflated SR (correct) {dsr:.3f}")
    print("  grid net Sharpe range: %.2f .. %.2f" % (sr_pp.min()*np.sqrt(12), sr_pp.max()*np.sqrt(12)))

    # regime + yearly
    ser = base_net; mid = ser.index[len(ser)//2]
    print("\n=== regime (deployable long-only) ===")
    for lab, sub in (("first", ser[ser.index < mid]), ("second", ser[ser.index >= mid])):
        print(f"  {lab:7s} n={len(sub)} ann {sub.mean()*12:+.1%} t={sub.mean()/sub.std()*np.sqrt(len(sub)):.2f}")
    yr = (ser.groupby(ser.index.year).mean()*12)
    print(f"  yearly positive {int((yr>0).sum())}/{len(yr)}: " + str({int(y): round(float(v),3) for y,v in yr.items()}))

    # ORTHOGONALITY (the decisive check for a true 2nd mechanism)
    print("\n=== ORTHOGONALITY (overlapping months) ===")
    dp = _dpead_monthly()
    j = pd.concat([ls.rename("div_ls"), lo.rename("div_lo"), dp], axis=1).dropna()
    if len(j) > 6:
        print(f"  corr(dividend L/S, D_PEAD)       = {j['div_ls'].corr(j['dpead']):+.2f}  (n={len(j)})")
        print(f"  corr(dividend long-only, D_PEAD) = {j['div_lo'].corr(j['dpead']):+.2f}")
    try:
        from engine.validation.analyst_revision import build_revision_sleeve_buffered
        rev, _ = build_revision_sleeve_buffered(q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
        jr = pd.concat([ls.rename("div"), rev.rename("rev")], axis=1).dropna()
        if len(jr) > 6:
            print(f"  corr(dividend L/S, analyst-revision) = {jr['div'].corr(jr['rev']):+.2f}  (n={len(jr)})")
    except Exception as e:
        print("  revision corr skipped:", str(e)[:60])


if __name__ == "__main__":
    main()
