"""engine/validation/_combined_book_run.py — DEPLOYMENT-GRADE combined earnings-
information book: D_PEAD + analyst-revision, causal equal-vol (risk-parity) weights
+ a vol-target risk overlay. Quantifies the IR uplift over D_PEAD alone (the defense
material) and shows the combined book's residual alpha vs FF5+UMD.

HONEST framing (not fake breadth): D_PEAD and revision share the SAME mechanism
(earnings-information underreaction, corr 0.64) — this is ONE deeply-engineered
strategy with the noise diversified away, NOT two independent strategies. The
vol-target overlay is RISK MANAGEMENT (cuts drawdown), not an alpha-timing regime
bet — the dispersion-state alpha overlay FAILED OOS earlier, so we don't claim it.
Costs applied uniformly (cost-model-consistency doctrine): one large-cap round-trip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.validation.analyst_revision import build_revision_sleeve_buffered
from engine.validation.after_cost import apply_cost
from engine.validation.deflated_sharpe import probabilistic_sharpe_ratio

RT = 30.0  # one consistent large-cap round-trip (bps), applied to both sleeves


def _ff_monthly():
    f = pd.read_parquet("data/cache/ff_factors_weekly.parquet"); f.index = pd.to_datetime(f.index)
    return (1 + f.drop(columns=["RF"], errors="ignore")).resample("ME").prod() - 1


def _dpead_monthly():
    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet").iloc[:, 0]
    d.index = pd.to_datetime(d.index)
    return ((1 + d.clip(-0.2, 0.2)).resample("ME").prod() - 1).rename("dpead")


def _metrics(r, ff):
    r = r.dropna()
    ann = r.mean() * 12; vol = r.std() * np.sqrt(12); sh = ann / vol
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    # residual alpha vs FF5+UMD
    J = pd.concat([r.rename("y"), ff], axis=1).dropna()
    facs = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    X = np.column_stack([np.ones(len(J))] + [J[f].values for f in facs])
    beta, *_ = np.linalg.lstsq(X, J["y"].values, rcond=None)
    resid = J["y"].values - X @ beta; dof = len(J) - X.shape[1]
    se = np.sqrt(np.diag((resid @ resid / dof) * np.linalg.inv(X.T @ X)))
    return dict(ann=ann, vol=vol, sharpe=sh, maxdd=dd,
                alpha=beta[0] * 12, t_alpha=beta[0] / se[0],
                psr0=probabilistic_sharpe_ratio(r.values, 0.0))


def voltarget(r, target=0.10, lookback=12, cap=2.0):
    """Causal vol-target: scale next month by target / trailing-realized-vol."""
    rv = r.rolling(lookback).std() * np.sqrt(12)
    w = (target / rv).clip(upper=cap).shift(1)
    return (w * r).rename(r.name + "_vt")


def main():
    ff = _ff_monthly()
    dp = _dpead_monthly()
    rev, rev_turn = build_revision_sleeve_buffered(q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
    rev = rev.rename("rev")
    # net of cost (consistent): D_PEAD turnover ~5x, revision measured turnover
    dp_net = apply_cost(dp, 5.0 * RT / 10000.0, ppy=12)
    rev_net = apply_cost(rev, rev_turn * RT / 10000.0, ppy=12)
    J = pd.concat([dp_net, rev_net], axis=1).dropna()

    # causal equal-vol (risk-parity) weights from trailing 12m vol
    vdp = J["dpead"].rolling(12).std().shift(1)
    vre = J["rev"].rolling(12).std().shift(1)
    wdp = (1 / vdp) / (1 / vdp + 1 / vre); wre = 1 - wdp
    comb = (wdp * J["dpead"] + wre * J["rev"]).dropna().rename("combined")
    comb_vt = voltarget(comb, target=0.10).dropna()

    print("=" * 92)
    print("COMBINED EARNINGS-INFORMATION BOOK — D_PEAD + analyst-revision (net, RT=%.0fbps)" % RT)
    print("=" * 92)
    print(f"corr(D_PEAD, revision) = {J['dpead'].corr(J['rev']):.2f}   (same mechanism — ONE strategy, noise diversified)")
    rows = {"D_PEAD alone": dp_net, "revision alone": rev_net,
            "COMBINED (eq-vol)": comb, "COMBINED + vol-target": comb_vt}
    print(f"\n{'book':24s} {'annRet':>7} {'vol':>6} {'Sharpe':>7} {'maxDD':>7} "
          f"{'alpha(FF5+UMD)':>14} {'t':>5} {'PSR0':>6}")
    base_sh = None
    for nm, r in rows.items():
        m = _metrics(r, ff)
        if nm == "D_PEAD alone":
            base_sh = m["sharpe"]
        print(f"{nm:24s} {m['ann']:6.1%} {m['vol']:5.1%} {m['sharpe']:7.2f} {m['maxdd']:7.1%} "
              f"{m['alpha']:13.1%} {m['t_alpha']:5.2f} {m['psr0']:6.3f}")

    sh_comb = _metrics(comb, ff)["sharpe"]; sh_vt = _metrics(comb_vt, ff)["sharpe"]
    print(f"\nIR UPLIFT (defense material):")
    print(f"  D_PEAD alone Sharpe        {base_sh:.2f}")
    print(f"  + revision (eq-vol)        {sh_comb:.2f}   (+{sh_comb-base_sh:.2f}, {(sh_comb/base_sh-1)*100:+.0f}%)")
    print(f"  + vol-target overlay       {sh_vt:.2f}   (+{sh_vt-base_sh:.2f} vs D_PEAD; overlay = risk mgmt)")
    print(f"  diversification source: corr 0.64 < 1 -> Grinold-Kahn breadth (noise reduction),")
    print(f"  NOT a 2nd independent mechanism (revision's D_PEAD-orthogonal alpha is only t=1.72).")


if __name__ == "__main__":
    main()
