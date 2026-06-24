"""engine/validation/_commodity_carry_run.py — gate genuine commodity carry.
DECISIVE tests: (1) residual alpha vs FF5+UMD (is it equity-orthogonal?), (2) the
INCREMENT over COMMODITY MOMENTUM built from the same front returns (is carry a
distinct signal, not the trend already in the CTA sleeve?), (3) cost (futures are
cheap + carry is low-turnover) + corrected deflated SR.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.validation.commodity_carry import build_carry_and_returns
from engine.validation.deflated_sharpe import (probabilistic_sharpe_ratio,
                                                expected_max_sharpe, var_sr_from_trial_sharpes)

RT = 10.0  # commodity futures round-trip bps (very liquid); carry is low-turnover


def _ff_monthly():
    f = pd.read_parquet("data/cache/ff_factors_weekly.parquet"); f.index = pd.to_datetime(f.index)
    return (1 + f.drop(columns=["RF"], errors="ignore")).resample("ME").prod() - 1


def _xs_ls(signal_wide, ret_wide, q=0.3):
    """generic cross-sectional L/S given a monthly signal-wide + next-month returns."""
    allm = sorted(set(signal_wide.index) | set(ret_wide.index))
    rows, ent, prevL = [], [], set()
    for i in range(len(allm) - 1):
        m, nxt = allm[i], allm[i + 1]
        if m not in signal_wide.index or nxt not in ret_wide.index:
            continue
        c = signal_wide.loc[m].dropna()
        if len(c) < 8:
            continue
        hi = c[c >= c.quantile(1 - q)].index; lo = c[c <= c.quantile(q)].index
        nr = ret_wide.loc[nxt]
        rl = nr.reindex(hi).dropna(); rs = nr.reindex(lo).dropna()
        if len(rl) < 2 or len(rs) < 2:
            continue
        rows.append((nxt, float(rl.mean() - rs.mean())))
        ent.append(len(set(hi) - prevL) / max(len(hi), 1)); prevL = set(hi)
    return pd.Series(dict(rows)).sort_index(), float(np.mean(ent) * 12) if ent else float("nan")


def _alpha(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None); resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]; se = np.sqrt(np.diag((resid @ resid / dof) * np.linalg.inv(X1.T @ X1)))
    return beta[0], beta[0] / se[0], beta[1:]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cwide, rwide = build_carry_and_returns()
    print(f"\ncommodities={cwide.shape[1]}  months={len(cwide)} {cwide.index.min():%Y-%m}..{cwide.index.max():%Y-%m}")

    # carry L/S + long-only
    carry_ls, turn = _xs_ls(cwide, rwide, q=0.3)
    # commodity 12-1m momentum (the control: carry must beat THIS to be distinct from trend)
    mom = (1 + rwide.fillna(0)).rolling(12).apply(np.prod, raw=True) - 1
    mom = mom - rwide   # 12-1 (skip most recent month)
    mom_ls, _ = _xs_ls(mom, rwide, q=0.3)

    def stat(net):
        net = net.dropna(); vol = net.std() * np.sqrt(12)
        return dict(n=len(net), ann=net.mean() * 12, sharpe=net.mean() * 12 / vol if vol > 0 else np.nan,
                    t=net.mean() / net.std() * np.sqrt(len(net)) if net.std() > 0 else np.nan)

    net = carry_ls - (turn * RT / 10000.0) / 12          # tiny cost, monthly
    gs, ns = stat(carry_ls), stat(net)
    print(f"\n=== COMMODITY CARRY L/S (q=0.3) turn={turn:.1f}x ===")
    print(f"  gross ann {gs['ann']:+.1%} Sharpe {gs['sharpe']:.2f} t={gs['t']:.2f} | "
          f"NET ann {ns['ann']:+.1%} Sharpe {ns['sharpe']:.2f} t={ns['t']:.2f} | PSR0 {probabilistic_sharpe_ratio(net.dropna().values,0.0):.3f}")
    print(f"  commodity 12-1 MOMENTUM L/S: Sharpe {stat(mom_ls)['sharpe']:.2f} t={stat(mom_ls)['t']:.2f}")

    # GATE 1: residual vs FF5+UMD ; GATE 2: increment over commodity momentum
    ff = _ff_monthly()
    J = pd.concat([net.rename("y"), ff, mom_ls.rename("CMOM")], axis=1).dropna()
    facs = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    a_ff, t_ff, _ = _alpha(J["y"].values, J[facs].values)
    a_inc, t_inc, _ = _alpha(J["y"].values, J[facs + ["CMOM"]].values)
    print(f"\n=== GATES (n={len(J)}) ===")
    print(f"  alpha vs FF5+UMD:            {a_ff*12:+.2%}/yr t={t_ff:.2f}  (equity-orthogonal?)")
    print(f"  alpha vs FF5+UMD+CMOM (INCR): {a_inc*12:+.2%}/yr t={t_inc:.2f}  <- increment over commodity momentum/CTA-trend")
    print(f"  corr(carry, commodity-momentum) = {J['y'].corr(J['CMOM']):+.2f}")

    # corrected deflated SR over q grid
    nets = []
    for qq in (0.2, 0.3, 0.4):
        l2, t2 = _xs_ls(cwide, rwide, q=qq); n2 = l2 - (t2 * RT / 10000.0) / 12
        if len(n2.dropna()) >= 24:
            nets.append(n2.dropna())
    sr = np.array([n.mean() / n.std() for n in nets]); V = var_sr_from_trial_sharpes(sr)
    dsr = probabilistic_sharpe_ratio(net.dropna().values, expected_max_sharpe(len(nets), V))
    print(f"\n  deflated SR (correct, N={len(nets)}) {dsr:.3f}")
    ser = net.dropna(); mid = ser.index[len(ser)//2]
    print("  regime:", {lab: round(s.mean()/s.std()*np.sqrt(len(s)), 2)
                        for lab, s in (("1H", ser[ser.index < mid]), ("2H", ser[ser.index >= mid]))})
    yr = (ser.groupby(ser.index.year).mean()*12)
    print(f"  yearly positive {int((yr>0).sum())}/{len(yr)}")


if __name__ == "__main__":
    main()
