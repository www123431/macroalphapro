"""Gate the cross-asset carry COMBINATION (commodity + FX): does diversification
lift it toward GREEN? Honest eval — risk-parity combine, corr, combined vs each
alone, residual vs FF5+UMD, corrected deflated SR, regime. No param-tuning to force.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.validation.crossasset_carry import build_fx_carry, build_rates_carry, build_commodity_carry_ls
from engine.validation.deflated_sharpe import (probabilistic_sharpe_ratio,
                                                expected_max_sharpe, var_sr_from_trial_sharpes)


def _ff_monthly():
    f = pd.read_parquet("data/cache/ff_factors_weekly.parquet"); f.index = pd.to_datetime(f.index)
    return (1 + f.drop(columns=["RF"], errors="ignore")).resample("ME").prod() - 1


def shp(x):
    x = x.dropna(); return x.mean() * 12 / (x.std() * np.sqrt(12)) if x.std() > 0 else np.nan


def tstat(x):
    x = x.dropna(); return x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else np.nan


def _alpha(y, X):
    X1 = np.column_stack([np.ones(len(X)), X]); beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta; dof = len(y) - X1.shape[1]
    se = np.sqrt(np.diag((resid @ resid / dof) * np.linalg.inv(X1.T @ X1)))
    return beta[0], beta[0] / se[0]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    legs = {"cmdty": build_commodity_carry_ls(), "fx": build_fx_carry()[2]}
    try:
        rt = build_rates_carry()[2]
        if rt.dropna().size >= 24:
            legs["rt"] = rt
        else:
            print("[note] rates leg dropped: only %d usable months (deferred UST settle sparse)" % rt.dropna().size)
    except Exception as e:
        print("[note] rates leg skipped:", str(e)[:60])
    for nm, s in legs.items():
        print(f"standalone {nm:9s}: Sharpe {shp(s):.2f} t={tstat(s):.2f} (n={s.dropna().size})")
    J = pd.concat([legs[k].rename(k) for k in legs], axis=1).dropna()
    print("\ncorr matrix of the three carry legs:")
    print(J.corr().round(2).to_string())
    # risk-parity (equal-vol) combine across the 3 legs
    w = {c: 1 / J[c].std() for c in J.columns}; W = sum(w.values())
    comb = (sum(w[c] * J[c] for c in J.columns) / W).rename("combined")
    print(f"\nCOMBINED (3-asset risk-parity) Sharpe {shp(comb):.2f} t={tstat(comb):.2f} (n={len(comb)})  "
          f"PSR0 {probabilistic_sharpe_ratio(comb.values,0.0):.3f}")

    # residual vs FF5+UMD (overlap), deflated SR over the q-grid trials (commodity 0.2/0.3/0.4)
    ff = _ff_monthly()
    J = pd.concat([comb.rename("y"), ff], axis=1).dropna()
    facs = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    a, t = _alpha(J["y"].values, J[facs].values)
    print(f"alpha vs FF5+UMD: {a*12:+.2%}/yr t={t:.2f} (n={len(J)}, equity-orthogonal?)")
    # honest deflated SR: trials = {each leg, combined} as the search
    trials = [legs[k].dropna() for k in legs] + [comb.dropna()]
    sr = np.array([x.mean() / x.std() for x in trials]); V = var_sr_from_trial_sharpes(sr)
    dsr = probabilistic_sharpe_ratio(comb.values, expected_max_sharpe(len(trials), V))
    print(f"deflated SR (correct, N={len(trials)}) {dsr:.3f}")
    mid = comb.index[len(comb) // 2]
    print("regime (combined):", {lab: round(shp(s), 2) for lab, s in
                                  (("1H", comb[comb.index < mid]), ("2H", comb[comb.index >= mid]))})
    for cut in ("2013-12-31", "2018-12-31"):
        r = comb[comb.index > cut]
        print(f"  since {cut[:4]}: Sharpe {shp(r):.2f} (n={r.dropna().size})")
    yr = (comb.dropna().groupby(comb.dropna().index.year).mean() * 12)
    print(f"  yearly positive {int((yr>0).sum())}/{len(yr)}")


if __name__ == "__main__":
    main()
