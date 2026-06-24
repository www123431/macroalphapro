"""engine/validation/_book_config_run.py — configure the cross-asset CARRY sleeve
into the equity-alpha book (D_PEAD + analyst-revision). Shows the correlation, the
book Sharpe / maxDD across carry risk-allocations, and a recommended configuration.
Carry is equity-orthogonal -> adding it should raise book Sharpe + cut maxDD
(Grinold-Kahn), and being a DIFFERENT mechanism it is a true backup. Honest: this is
the equity-book + carry; the full 5-sleeve book (incl. insurance/CTA) diversifies more.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.validation.analyst_revision import build_revision_sleeve_buffered
from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
from engine.validation.crossasset_carry import build_fx_carry, _xs_ls


def rp_combine(series_list):
    J = pd.concat(series_list, axis=1).dropna()
    w = {c: 1 / J[c].std() for c in J.columns}; W = sum(w.values())
    return (sum(w[c] * J[c] for c in J.columns) / W), J

RT_EQ = 30.0   # equity sleeves round-trip bps
RT_CY = 10.0   # carry (liquid futures) round-trip bps


def mstats(r):
    r = r.dropna(); vol = r.std() * np.sqrt(12)
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    return dict(ann=r.mean() * 12, vol=vol, sharpe=r.mean() * 12 / vol if vol > 0 else np.nan,
                t=r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan, maxdd=dd)


def voltarget(r, target=0.10, lb=12):
    rv = r.rolling(lb).std() * np.sqrt(12)
    return ((target / rv).clip(upper=2.0).shift(1) * r)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # --- equity-alpha book: D_PEAD + revision (eq-vol, net) ---
    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet").iloc[:, 0]; d.index = pd.to_datetime(d.index)
    dp = ((1 + d.clip(-0.2, 0.2)).resample("ME").prod() - 1).rename("dp")
    dp_net = dp - 5.0 * RT_EQ / 10000.0 / 12
    rev, rev_turn = build_revision_sleeve_buffered(q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
    rev_net = (rev - rev_turn * RT_EQ / 10000.0 / 12).rename("rev")
    E = pd.concat([dp_net, rev_net], axis=1).dropna()
    vdp = E["dp"].rolling(12).std().shift(1); vre = E["rev"].rolling(12).std().shift(1)
    w = (1 / vdp) / (1 / vdp + 1 / vre)
    equity = (w * E["dp"] + (1 - w) * E["rev"]).dropna().rename("equity_book")

    # --- carry sleeve: commodity + FX (net, low cost) ---
    cw_c, rw_c = commodity_cr(); cw_f, rw_f, _ = build_fx_carry()
    carry_c = _xs_ls(cw_c, rw_c, q=0.3); carry_f = _xs_ls(cw_f, rw_f, q=0.4)
    carry_g, _ = rp_combine([carry_c.rename("c"), carry_f.rename("f")])
    carry = (carry_g - 4.0 * RT_CY / 10000.0 / 12).rename("carry")

    J = pd.concat([equity, carry], axis=1).dropna()
    print(f"\noverlap months: {len(J)}  ({J.index.min():%Y-%m}..{J.index.max():%Y-%m})")
    print(f"corr(equity book, carry) = {J['equity'].corr(J['carry']) if 'equity' in J else J.iloc[:,0].corr(J.iloc[:,1]):.2f}")
    e, c = J["equity_book"], J["carry"]
    for nm, s in (("equity book (D_PEAD+rev)", e), ("carry sleeve", c)):
        m = mstats(s); print(f"  {nm:26s} ann {m['ann']:+.1%} vol {m['vol']:.1%} Sharpe {m['sharpe']:.2f} maxDD {m['maxdd']:.1%}")

    # --- vol-target each to 10%, then combine at carry risk-weight grid ---
    ev = voltarget(e); cv = voltarget(c)
    K = pd.concat([ev.rename("e"), cv.rename("c")], axis=1).dropna()
    print(f"\n=== book configs (each vol-targeted ~10%, combine at carry risk-weight) ===")
    print(f"  {'carry wt':>9} {'annRet':>7} {'vol':>6} {'Sharpe':>7} {'maxDD':>7}")
    for wc in (0.0, 0.2, 0.3, 0.4, 0.5):
        book = (1 - wc) * K["e"] + wc * K["c"]
        m = mstats(book)
        tag = " <- equity only" if wc == 0 else (" <- risk-parity" if wc == 0.5 else "")
        print(f"  {wc:9.0%} {m['ann']:+7.1%} {m['vol']:6.1%} {m['sharpe']:7.2f} {m['maxdd']:7.1%}{tag}")

    # regime: does carry help in equity-book's worst stretch?
    bk30 = (0.7 * K["e"] + 0.3 * K["c"])
    print("\n=== regime (equity-only vs +30% carry), Sharpe by sub-period ===")
    mid = K.index[len(K) // 2]
    for lab, m0, m1 in (("first half", None, mid), ("second half", mid, None)):
        def slc(s):
            x = s.dropna()
            if m0 is not None: x = x[x.index >= m0]
            if m1 is not None: x = x[x.index < m1]
            return x
        print(f"  {lab:12s} equity {mstats(slc(K['e']))['sharpe']:+.2f} | +30%carry {mstats(slc(bk30))['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
