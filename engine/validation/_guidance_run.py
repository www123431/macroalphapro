"""engine/validation/_guidance_run.py — full audit of the management-guidance-drift
sleeve, with the CORRECTED deflated-Sharpe methodology (actual cross-trial variance,
not raw grid size). The decisive checks: (1) is the post-guidance drift real and
tradeable after cost; (2) is it ORTHOGONAL to the D_PEAD + analyst-revision book
(guidance issued WITH earnings could just re-package the earnings surprise -> high
corr -> no diversification, like the 13F result).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from engine.validation.guidance_drift import build_guidance_surprise, build_guidance_sleeve, _monthly_returns
from engine.validation.after_cost import apply_cost
from engine.validation.deflated_sharpe import (deflated_sharpe_ratio,
                                                probabilistic_sharpe_ratio,
                                                expected_max_sharpe,
                                                var_sr_from_trial_sharpes)

RT = 26.0  # ss_large round-trip bps (large analyst-covered guidance firms), conservative


def _stats(net):
    net = net.dropna()
    vol = net.std() * np.sqrt(12)
    sr = net.mean() * 12 / vol if vol > 0 else float("nan")
    return dict(n=len(net), ann=net.mean() * 12, vol=vol, sharpe=sr,
                t=net.mean() / net.std() * np.sqrt(len(net)) if net.std() > 0 else float("nan"),
                psr0=probabilistic_sharpe_ratio(net.values, 0.0))


def _dpead_monthly():
    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet")
    s = d.iloc[:, 0]; s.index = pd.to_datetime(s.index)
    return ((1 + s).resample("ME").prod() - 1).rename("dpead")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    surprise = build_guidance_surprise()
    mret = _monthly_returns()
    print(f"\nguidance events {len(surprise)}, permnos {surprise['permno'].nunique()}, "
          f"{surprise['anndats'].min():%Y-%m}..{surprise['anndats'].max():%Y-%m}")

    # ---- pre-specified sleeve (hold=2, q=0.2, equal) ----
    ls, lo, turn = build_guidance_sleeve(hold=2, q=0.2, weight="equal",
                                         surprise=surprise, mret=mret)
    print(f"\n=== PRE-SPECIFIED guidance sleeve (hold=2, q=0.2, equal) turn={turn:.1f}x ===")
    for nm, ser, legs in [("L/S", ls, 2), ("LONG-ONLY (deployable)", lo, 1)]:
        gross = _stats(ser)
        net = apply_cost(ser.dropna(), legs * turn * RT / 10000.0, ppy=12)
        ns = _stats(net)
        print(f"  {nm:22s} gross ann {gross['ann']:+.1%} Sharpe {gross['sharpe']:.2f} | "
              f"cost {legs*turn*RT/10000.0:.2%} | NET ann {ns['ann']:+.1%} Sharpe {ns['sharpe']:.2f} "
              f"t={ns['t']:.2f} PSR0={ns['psr0']:.3f}")

    # ---- small grid for robustness + CORRECT deflated SR (actual cross-trial V) ----
    grid = [dict(hold=h, q=qq, weight=w) for h in (1, 2, 3) for qq in (0.1, 0.2) for w in ("equal", "mag")]
    nets = []
    for p in grid:
        l, longonly, tn = build_guidance_sleeve(surprise=surprise, mret=mret, **p)
        net = apply_cost(longonly.dropna(), tn * RT / 10000.0, ppy=12)   # long-only = deployable
        if len(net) >= 24:
            nets.append((p, net))
    sr_pp = np.array([n.mean() / n.std() for _, n in nets])
    V_act = var_sr_from_trial_sharpes(sr_pp)
    M = pd.DataFrame({i: n for i, (_, n) in enumerate(nets)}).dropna()
    avg_corr = (M.corr().values.sum() - len(M.columns)) / (len(M.columns) ** 2 - len(M.columns))
    N = len(nets)
    print(f"\n=== grid (long-only) N={N}  avg corr={avg_corr:.3f}  V_actual={V_act:.5f} ===")
    # deployable pre-specified = hold2/q.2/equal
    base_net = [n for p, n in nets if p == dict(hold=2, q=0.2, weight="equal")][0]
    em = expected_max_sharpe(N, V_act)
    dsr = probabilistic_sharpe_ratio(base_net.values, em)
    print(f"  pre-specified long-only: net Sharpe {base_net.mean()*12/(base_net.std()*np.sqrt(12)):.2f} "
          f"net t {base_net.mean()/base_net.std()*np.sqrt(len(base_net)):.2f} | "
          f"PSR0 {probabilistic_sharpe_ratio(base_net.values,0.0):.3f} | "
          f"deflated SR (correct, N={N},V_act) {dsr:.3f}")
    print("  grid net Sharpe range: %.2f .. %.2f" % (sr_pp.min()*np.sqrt(12), sr_pp.max()*np.sqrt(12)))

    # ---- regime + yearly (deployable long-only, pre-specified) ----
    ser = base_net
    mid = ser.index[len(ser)//2]
    print("\n=== regime (deployable long-only) ===")
    for lab, sub in (("first", ser[ser.index < mid]), ("second", ser[ser.index >= mid])):
        print(f"  {lab:7s} n={len(sub)} ann {sub.mean()*12:+.1%} t={sub.mean()/sub.std()*np.sqrt(len(sub)):.2f}")
    yr = (ser.groupby(ser.index.year).mean()*12)
    print(f"  yearly positive {int((yr>0).sum())}/{len(yr)}: " + str({int(y): round(float(v),3) for y,v in yr.items()}))

    # ---- ORTHOGONALITY: corr vs D_PEAD + analyst-revision (the decisive check) ----
    print("\n=== ORTHOGONALITY (overlapping months) ===")
    dp = _dpead_monthly()
    j = pd.concat([ls.rename("guid_ls"), lo.rename("guid_lo"), dp], axis=1).dropna()
    if len(j) > 6:
        print(f"  corr(guidance L/S, D_PEAD)      = {j['guid_ls'].corr(j['dpead']):+.2f}  (n={len(j)})")
        print(f"  corr(guidance long-only, D_PEAD)= {j['guid_lo'].corr(j['dpead']):+.2f}")
    try:
        from engine.validation.analyst_revision import build_revision_sleeve_buffered
        rev, _ = build_revision_sleeve_buffered(q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
        jr = pd.concat([ls.rename("guid"), rev.rename("rev")], axis=1).dropna()
        if len(jr) > 6:
            print(f"  corr(guidance L/S, analyst-revision) = {jr['guid'].corr(jr['rev']):+.2f}  (n={len(jr)})")
    except Exception as e:
        print("  revision corr skipped:", str(e)[:60])


if __name__ == "__main__":
    main()
