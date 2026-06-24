"""Methodology check: is n_trials=grid_size the CORRECT deflated-Sharpe penalty
for a parameter grid over ONE signal? Bailey-Lopez de Prado's E[max SR] needs the
ACTUAL variance of SR across trials V (not a single-series theoretical V), and
assumes ~independent trials. Highly-correlated grid configs => small V and few
EFFECTIVE independent trials => the mechanical n_trials=72-with-theoretical-V is
a gross over-penalty. This quantifies the right number."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.validation._revision_optimize import load_inputs, sleeve, evaluate
from engine.validation.after_cost import apply_cost
from engine.validation.deflated_sharpe import (deflated_sharpe_ratio,
                                                probabilistic_sharpe_ratio,
                                                expected_max_sharpe,
                                                var_sr_from_trial_sharpes)

RT = 20.0


def main():
    revw, cvw, mret = load_inputs()
    grid = [dict(disp_pctile=d, weight=w, q_in=qi, q_out=qo)
            for d in (0.3, 0.4, 0.5, 0.6) for w in ("equal", "mag")
            for qi in (0.10, 0.15, 0.20) for qo in (0.30, 0.40, 0.50) if qo > qi]

    nets, sr_pp = {}, []
    for p in grid:
        ls, turn = sleeve(revw, cvw, mret, **p)
        ls = ls.dropna()
        if len(ls) < 24:
            continue
        net = apply_cost(ls, turn * RT / 10000.0, ppy=12)
        key = (p["disp_pctile"], p["weight"], p["q_in"], p["q_out"])
        nets[key] = net
        sr_pp.append(net.mean() / net.std())          # per-period net SR
    N = len(nets)
    sr_pp = np.array(sr_pp)

    # average pairwise correlation among the trial return series
    M = pd.DataFrame({k: v for k, v in nets.items()}).dropna()
    C = M.corr().values
    avg_corr = (C.sum() - np.trace(C)) / (C.size - len(C))
    # effective independent trials (avg-correlation adjustment)
    n_eff = 1 + (N - 1) * (1 - avg_corr)

    V_actual = var_sr_from_trial_sharpes(sr_pp)        # true cross-trial SR variance
    V_theo = (1.0 / (len(next(iter(nets.values()))) - 1)) * (1 + 0.5 * sr_pp.max() ** 2)

    print(f"trials N={N}  avg pairwise corr={avg_corr:.3f}  -> effective N_indep={n_eff:.1f}")
    print(f"SR(per-period) across trials: min {sr_pp.min():.3f} max {sr_pp.max():.3f} "
          f"=> V_actual={V_actual:.5f}   (theoretical single-series V={V_theo:.5f})")
    print(f"E[max SR] per-period:")
    for tag, nn, VV in [("N=72, V_actual (CORRECT)", N, V_actual),
                        ("N_eff, V_actual", int(round(n_eff)), V_actual),
                        ("N=72, V_theoretical (what I did = WRONG)", N, V_theo)]:
        em = expected_max_sharpe(nn, VV)
        print(f"   {tag:42s}: SR*_0={em:.4f} (ann {em*np.sqrt(12):.3f})")

    # evaluate two configs: PRE-SPECIFIED (theory default) and grid-MAX
    cfgs = {"pre-specified (disp .5, q .2/.4, equal) [theory default, NOT mined]":
            (0.5, "equal", 0.20, 0.40),
            "grid-MAX by net Sharpe (disp .6, q .15/.5, equal)":
            (0.6, "equal", 0.15, 0.50)}
    for name, key in cfgs.items():
        net = nets[key]; sr = net.mean() / net.std()
        print(f"\n=== {name} ===")
        print(f"   net ann {net.mean()*12:+.1%}  net Sharpe {sr*np.sqrt(12):.3f}  n={len(net)}mo")
        psr0 = probabilistic_sharpe_ratio(net.values, 0.0)
        print(f"   PSR vs 0 (sample+non-normality, NO multiple-test): {psr0:.4f}")
        for tag, nn, VV in [("N=72,V_actual (correct param-grid)", N, V_actual),
                            ("N_eff,V_actual", int(round(n_eff)), V_actual),
                            ("N=72,V_theoretical (my over-penalty)", N, V_theo)]:
            em = expected_max_sharpe(nn, VV)
            dsr = probabilistic_sharpe_ratio(net.values, em)
            print(f"   DSR [{tag:38s}] = {dsr:.4f}")


if __name__ == "__main__":
    main()
