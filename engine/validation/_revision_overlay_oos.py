"""engine/validation/_revision_overlay_oos.py — LEGITIMATE (non-p-hacked) test of
a dispersion-STATE overlay on the analyst-revision sleeve, to address the regime
caveat (alpha concentrated in high-dispersion years; first half t=1.41).

Anti-p-hacking discipline:
  * the conditioning variable (aggregate forecast dispersion) is economically
    pre-specified (Zhang 2006: revision drift stronger under high uncertainty) and
    was independently confirmed (yearly corr 0.55 with the sleeve's alpha);
  * the rule is CAUSAL — at each month the threshold is the EXPANDING median of
    PAST dispersion only (no look-ahead), so the whole 2013-2024 path is a real-
    time OOS test of the rule, not an in-sample fit;
  * ONE rule is pre-committed (full exposure when dispersion >= past-median, else
    half). Other (w_low, window) values are shown as a SENSITIVITY BAND, reported
    in full — NOT selected from. A single overlay = +1 trial (negligible).

Reports baseline vs overlaid: full-sample net Sharpe / t, first- vs second-half t,
yearly, so we can see whether the regime weakness is genuinely repaired OOS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from engine.validation._revision_optimize import load_inputs, sleeve
from engine.validation.after_cost import apply_cost
from engine.validation.deflated_sharpe import probabilistic_sharpe_ratio

RT = 20.0


def stat_block(net):
    net = net.dropna()
    sr = net.mean() / net.std()
    t = sr * np.sqrt(len(net))
    mid = net.index[len(net) // 2]
    fh, sh = net[net.index < mid], net[net.index >= mid]
    return dict(ann=net.mean() * 12, sharpe=sr * np.sqrt(12), t=t,
                fh_ann=fh.mean() * 12, fh_t=fh.mean() / fh.std() * np.sqrt(len(fh)),
                sh_ann=sh.mean() * 12, sh_t=sh.mean() / sh.std() * np.sqrt(len(sh)),
                psr0=probabilistic_sharpe_ratio(net.values, 0.0))


def main():
    revw, cvw, mret = load_inputs()
    # pre-specified sleeve (theory-default params; NOT mined)
    ls, turn = sleeve(revw, cvw, mret, q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
    net = apply_cost(ls.dropna(), turn * RT / 10000.0, ppy=12)

    # macro dispersion state = cross-sectional median CV per month, observable at t
    disp_state = cvw.median(axis=1)
    # dispersion KNOWN at the decision month (one month before the realized return)
    disp_dec = disp_state.shift(1).reindex(net.index)
    # CAUSAL threshold: expanding median of strictly-past decision dispersions
    thresh = disp_dec.expanding(min_periods=12).median().shift(1)

    def overlay(w_low, hi=1.0):
        exp = pd.Series(hi, index=net.index)
        mask = disp_dec < thresh        # low-uncertainty regime -> de-risk
        exp[mask & thresh.notna()] = w_low
        return (exp * net).rename(f"ov_{w_low}")

    print("=" * 74)
    print("DISPERSION-STATE OVERLAY — causal (expanding past-median), OOS by construction")
    print("=" * 74)
    base = stat_block(net)
    print("BASELINE (no overlay):")
    print(f"  full ann {base['ann']:+.1%} Sharpe {base['sharpe']:.2f} t={base['t']:.2f}  "
          f"| 1st-half ann {base['fh_ann']:+.1%} t={base['fh_t']:.2f}  "
          f"2nd-half ann {base['sh_ann']:+.1%} t={base['sh_t']:.2f}  PSR0={base['psr0']:.3f}")

    print("\nPRE-COMMITTED overlay (w_low=0.5):")
    ov = overlay(0.5)
    s = stat_block(ov)
    cov = pd.concat([net.rename('b'), ov.rename('o')], axis=1).dropna()
    print(f"  full ann {s['ann']:+.1%} Sharpe {s['sharpe']:.2f} t={s['t']:.2f}  "
          f"| 1st-half ann {s['fh_ann']:+.1%} t={s['fh_t']:.2f}  "
          f"2nd-half ann {s['sh_ann']:+.1%} t={s['sh_t']:.2f}  PSR0={s['psr0']:.3f}")
    frac_derisked = float(((disp_dec < thresh) & thresh.notna()).mean())
    print(f"  fraction of months de-risked: {frac_derisked:.0%}  corr(overlaid,base)={cov['b'].corr(cov['o']):.3f}")

    print("\nSENSITIVITY BAND (reported in full, NOT selected from):")
    print(f"  {'w_low':>6} {'full Sharpe':>12} {'full t':>8} {'1H t':>7} {'2H t':>7}")
    for wl in (0.0, 0.25, 0.5, 0.75, 1.0):
        st = stat_block(overlay(wl))
        print(f"  {wl:6.2f} {st['sharpe']:12.2f} {st['t']:8.2f} {st['fh_t']:7.2f} {st['sh_t']:7.2f}")

    # yearly of the pre-committed overlay
    yr = (ov.dropna().groupby(ov.dropna().index.year).mean() * 12)
    print("\n  pre-committed overlay yearly ann:",
          {int(y): round(float(v), 3) for y, v in yr.items()})


if __name__ == "__main__":
    main()
