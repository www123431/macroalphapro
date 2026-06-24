"""engine/validation/_network_run.py — full audit of shared-analyst connected-firm
momentum (CMOM). The DECISIVE test for this momentum-family signal: after
RESIDUALIZING vs FF5+UMD+PEAD, is there orthogonal alpha (or is it just UMD in
disguise)? Plus cost, corrected deflated SR, regime, and orthogonality vs the book.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from engine.validation.network_momentum import build_cmom_signal, build_cmom_sleeve
from engine.validation.after_cost import apply_cost
from engine.validation.deflated_sharpe import probabilistic_sharpe_ratio, expected_max_sharpe, var_sr_from_trial_sharpes

RT = 26.0


def _factors_monthly():
    f = pd.read_parquet("data/cache/ff_factors_weekly.parquet")
    f.index = pd.to_datetime(f.index)
    fm = (1 + f.drop(columns=["RF"], errors="ignore")).resample("ME").prod() - 1
    return fm


def _dpead_monthly():
    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet"); s = d.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return ((1 + s).resample("ME").prod() - 1).rename("PEAD")


def ols_alpha(y, X):
    """OLS intercept (alpha) + t-stat; X without const (added here)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    return beta[0], beta[0] / se[0], dof


def _stats(net):
    net = net.dropna(); vol = net.std() * np.sqrt(12)
    return dict(n=len(net), ann=net.mean() * 12, sharpe=net.mean() * 12 / vol if vol > 0 else np.nan,
                t=net.mean() / net.std() * np.sqrt(len(net)) if net.std() > 0 else np.nan)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("building CMOM signal (co-coverage graph per year)...")
    sigret = build_cmom_signal(min_shared=1, form=1)
    ls, lo, turn = build_cmom_sleeve(q=0.2, sigret=sigret)
    print(f"\n=== shared-analyst CMOM sleeve (min_shared=1, form=1mo, q=0.2) turn={turn:.1f}x ===")
    for nm, ser, legs in [("L/S", ls, 2), ("LONG-ONLY", lo, 1)]:
        g = _stats(ser.dropna())
        net = apply_cost(ser.dropna(), legs * turn * RT / 10000.0, ppy=12)
        ns = _stats(net)
        print(f"  {nm:10s} gross ann {g['ann']:+.1%} Sharpe {g['sharpe']:.2f} t={g['t']:.2f} | "
              f"NET ann {ns['ann']:+.1%} Sharpe {ns['sharpe']:.2f} t={ns['t']:.2f}")

    # ---- RESIDUALIZE vs FF5+UMD+PEAD (the decisive test) ----
    fm = _factors_monthly(); pe = _dpead_monthly()
    net_ls = apply_cost(ls.dropna(), 2 * turn * RT / 10000.0, ppy=12)
    J = pd.concat([net_ls.rename("y"), fm, pe], axis=1).dropna()
    facs = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "PEAD"]
    a_raw, t_raw, dof = ols_alpha(J["y"].values, J[facs].values)
    # also vs CAPM-UMD only (is it just momentum?)
    a_umd, t_umd, _ = ols_alpha(J["y"].values, J[["Mkt-RF", "UMD"]].values)
    print(f"\n=== RESIDUALIZATION (net L/S, n={len(J)} months) ===")
    print(f"  raw net: ann {net_ls.mean()*12:+.1%} t={_stats(net_ls)['t']:.2f}")
    print(f"  alpha vs Mkt+UMD:        {a_umd*12:+.2%}/yr  t={t_umd:.2f}   (is it just momentum?)")
    print(f"  alpha vs FF5+UMD+PEAD:   {a_raw*12:+.2%}/yr  t={t_raw:.2f}   (orthogonal residual alpha?)")
    # UMD loading
    X1 = np.column_stack([np.ones(len(J)), J[facs].values])
    beta, *_ = np.linalg.lstsq(X1, J["y"].values, rcond=None)
    print("  factor loadings:", {f: round(b, 2) for f, b in zip(facs, beta[1:])})

    # corrected deflated SR over a small grid (form/min_shared/q)
    nets = []
    for form in (1, 6, 12):
        for ms in (1, 2):
            sg = build_cmom_signal(min_shared=ms, form=form)
            l2, lo2, t2 = build_cmom_sleeve(q=0.2, sigret=sg)
            n2 = apply_cost(l2.dropna(), 2 * t2 * RT / 10000.0, ppy=12)
            if len(n2) >= 24:
                nets.append(n2)
    sr_pp = np.array([n.mean() / n.std() for n in nets]); V = var_sr_from_trial_sharpes(sr_pp)
    dsr = probabilistic_sharpe_ratio(net_ls.values, expected_max_sharpe(len(nets), V))
    print(f"\n  grid N={len(nets)} net Sharpe range {sr_pp.min()*np.sqrt(12):.2f}..{sr_pp.max()*np.sqrt(12):.2f} "
          f"| deflated SR (correct) {dsr:.3f}  PSR0 {probabilistic_sharpe_ratio(net_ls.values,0.0):.3f}")

    # regime
    ser = net_ls; mid = ser.index[len(ser)//2]
    print("  regime:", {lab: round(sub.mean()/sub.std()*np.sqrt(len(sub)), 2)
                         for lab, sub in (("1H", ser[ser.index < mid]), ("2H", ser[ser.index >= mid]))})

    # ---- orthogonality vs book + the existing supply-chain sleeve ----
    print("\n=== ORTHOGONALITY ===")
    try:
        sc = pd.read_parquet("data/cache/_supplychain_mom_sleeve.parquet").iloc[:, 0]
        sc.index = pd.to_datetime(sc.index)
        jj = pd.concat([ls.rename("cmom"), sc.rename("sc")], axis=1).dropna()
        print(f"  corr(co-coverage CMOM, existing supply-chain mom) = {jj['cmom'].corr(jj['sc']):+.2f} (n={len(jj)})")
    except Exception as e:
        print("  supplychain corr skipped:", str(e)[:50])
    jd = pd.concat([ls.rename("cmom"), pe], axis=1).dropna()
    print(f"  corr(CMOM, D_PEAD) = {jd['cmom'].corr(jd['PEAD']):+.2f}")
    try:
        from engine.validation.analyst_revision import build_revision_sleeve_buffered
        rev, _ = build_revision_sleeve_buffered(q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
        jr = pd.concat([ls.rename("cmom"), rev.rename("rev")], axis=1).dropna()
        print(f"  corr(CMOM, analyst-revision) = {jr['cmom'].corr(jr['rev']):+.2f}")
    except Exception as e:
        print("  revision corr skipped:", str(e)[:50])


if __name__ == "__main__":
    main()
