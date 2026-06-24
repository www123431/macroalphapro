"""engine/validation/_options_pcs_run.py — gate the options put-call IV spread
(Cremers-Weinbaum). FIRST gate = residual alpha vs FF5+UMD; then orthogonality vs
D_PEAD, cost/turnover, corrected deflated SR.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.validation.options_pcs import build_pcs_signal, build_pcs_sleeve, _monthly_returns
from engine.validation.after_cost import apply_cost
from engine.validation.deflated_sharpe import probabilistic_sharpe_ratio, expected_max_sharpe, var_sr_from_trial_sharpes

RT = 26.0


def _ff_monthly():
    f = pd.read_parquet("data/cache/ff_factors_weekly.parquet"); f.index = pd.to_datetime(f.index)
    return (1 + f.drop(columns=["RF"], errors="ignore")).resample("ME").prod() - 1


def _dpead_monthly():
    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet").iloc[:, 0]; d.index = pd.to_datetime(d.index)
    return ((1 + d.clip(-0.2, 0.2)).resample("ME").prod() - 1).rename("PEAD")


def _alpha(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None); resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]; se = np.sqrt(np.diag((resid @ resid / dof) * np.linalg.inv(X1.T @ X1)))
    return beta[0], beta[0] / se[0], beta[1:]


def _stat(net):
    net = net.dropna(); vol = net.std() * np.sqrt(12)
    return dict(n=len(net), ann=net.mean() * 12, sharpe=net.mean() * 12 / vol if vol > 0 else np.nan,
                t=net.mean() / net.std() * np.sqrt(len(net)) if net.std() > 0 else np.nan)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sig = build_pcs_signal(); mret = _monthly_returns()
    print(f"\nPCS panel: {len(sig)} rows, {sig['permno'].nunique()} permnos, "
          f"{sig['month'].min():%Y-%m}..{sig['month'].max():%Y-%m}")
    ls, lo, turn = build_pcs_sleeve(q=0.2, signal=sig, mret=mret)
    ff = _ff_monthly(); pe = _dpead_monthly()
    facs = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]

    print(f"\n=== put-call IV spread sleeve (q=0.2) turn={turn:.1f}x ===")
    for nm, ser, legs in [("L/S", ls, 2), ("LONG-ONLY", lo, 1)]:
        g = _stat(ser.dropna()); net = apply_cost(ser.dropna(), legs * turn * RT / 10000.0, ppy=12); ns = _stat(net)
        print(f"  {nm:10s} gross ann {g['ann']:+.1%} Sharpe {g['sharpe']:.2f} | "
              f"NET ann {ns['ann']:+.1%} Sharpe {ns['sharpe']:.2f} t={ns['t']:.2f}")

    net_ls = apply_cost(ls.dropna(), 2 * turn * RT / 10000.0, ppy=12)
    J = pd.concat([net_ls.rename("y"), ff, pe], axis=1).dropna()
    a5, t5, _ = _alpha(J["y"].values, J[facs].values)
    a5p, t5p, _ = _alpha(J["y"].values, J[facs + ["PEAD"]].values)
    print(f"\n=== GATE 1: residual alpha (net L/S, n={len(J)}) ===")
    print(f"  raw net t={_stat(net_ls)['t']:.2f}")
    print(f"  alpha vs FF5+UMD:      {a5*12:+.2%}/yr  t={t5:.2f}   <-- FIRST gate (HLZ ~3.0)")
    print(f"  alpha vs FF5+UMD+PEAD: {a5p*12:+.2%}/yr  t={t5p:.2f}")
    print(f"\n=== GATE 2: orthogonality ===")
    jd = pd.concat([ls.rename("pcs"), pe], axis=1).dropna()
    print(f"  corr(PCS L/S, D_PEAD) = {jd['pcs'].corr(jd['PEAD']):+.2f}")
    # deflated SR over q grid
    nets = []
    for qq in (0.1, 0.2, 0.3):
        l2, lo2, t2 = build_pcs_sleeve(q=qq, signal=sig, mret=mret)
        n2 = apply_cost(l2.dropna(), 2 * t2 * RT / 10000.0, ppy=12)
        if len(n2) >= 24:
            nets.append(n2)
    sr = np.array([n.mean() / n.std() for n in nets]); V = var_sr_from_trial_sharpes(sr)
    dsr = probabilistic_sharpe_ratio(net_ls.values, expected_max_sharpe(len(nets), V))
    print(f"\n  deflated SR (correct) {dsr:.3f}  PSR0 {probabilistic_sharpe_ratio(net_ls.values,0.0):.3f}")
    ser = net_ls; mid = ser.index[len(ser)//2]
    print("  regime:", {lab: round(sub.mean()/sub.std()*np.sqrt(len(sub)), 2)
                         for lab, sub in (("1H", ser[ser.index < mid]), ("2H", ser[ser.index >= mid]))})


if __name__ == "__main__":
    main()
