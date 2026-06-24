"""engine/validation/cta_trend.py — CTA replication: vol-scaled multi-lookback TSMOM (sleeve B).

⚠️ RECONCILIATION (2026-05-27): this is a FUTURES-UNIVERSE VARIANT of an ALREADY-EXISTING, registered,
validated trend sleeve — spec 75 `docs/spec_trend_crisis_hedge_sleeve_v1.md`, source
`engine/validation/tsmom_crisis.py` (same multi-speed 3/6/12 TSMOM mechanism). spec 75 ALREADY tested
futures vs ETF and found the **ETF-proxy wrapper is the DEPLOYABLE one** (full Sharpe 0.52 / OOS 0.38,
Alpaca-tradable) while yfinance continuous futures are WORSE (0.35, roll-stitch noise). This module's
WRDS-futures backtest (0.585) is HIGHER but NOT forward-deployable (WRDS = data not a broker; forward
marks would use the noisy yfinance futures, and a real futures venue was rejected for KYC). THEREFORE
the DEPLOYABLE trend sleeve = spec 75's ETF proxies on ALPACA, NOT this futures version. Keep this as a
logged alternative-wrapper backtest; the FuturesSimAdapter / futures_book work it feeds is genuinely
needed for CARRY (a real futures-curve strategy), not for the trend sleeve. Do NOT treat cta_trend as a
new sleeve or supersede PQTIX (spec 72) with it.


The FAITHFUL managed-futures sleeve. PQTIX (a mutual fund) can't be API-traded faithfully on any
paper broker, so the live book would mismatch the backtest. B replaces it with a SELF-BUILT trend
sleeve over the cached commodity + FX + rates futures universe — used in BOTH backtest and live
(Tradovate/IB futures) so live replicates backtest. Method = Moskowitz-Ooi-Pedersen / AQR
Hurst-Ooi-Pedersen: blended lookbacks (3/6/12m), positions scaled to constant per-instrument vol,
portfolio vol-targeted. No new data (reuses the carry front-return panels).

JUDGING: trend is an INSURANCE/diversification sleeve, judged on crisis-payoff + Sharpe>=0.3 +
diversification (low/neg corr to carry & equity) — the SAME bar PQTIX was held to — NOT the HLZ
alpha gate (that is for alpha sleeves). We still print the formal gate numbers for transparency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LOOKS = (3, 6, 12)
DEFAULT_TARGET_VOL = 0.10
VOL_WINDOW = 12


def _positions(rwide: pd.DataFrame, looks=DEFAULT_LOOKS,
               target_vol=DEFAULT_TARGET_VOL, vol_window=VOL_WINDOW) -> pd.DataFrame:
    """Per-instrument vol-scaled blended-lookback TSMOM positions (lagged 1m, no look-ahead)."""
    r = rwide.sort_index()
    vol = (r.rolling(vol_window, min_periods=6).std() * np.sqrt(12)).replace(0, np.nan)
    sig = sum(np.sign((1 + r.fillna(0)).rolling(L).apply(np.prod, raw=True) - 1)
              for L in looks) / len(looks)              # blended sign in [-1, 1]
    return (sig * (target_vol / vol)).shift(1)           # position known at t-1


def build_cta_returns(looks=DEFAULT_LOOKS, target_vol=DEFAULT_TARGET_VOL,
                      include_rates: bool = True, tc_bps_per_side: float = 0.0) -> pd.Series:
    """Self-built CTA trend sleeve monthly return (vol-targeted), commodity + FX (+ rates).
    tc_bps_per_side > 0 charges futures transaction cost on traded notional (|Δposition|)."""
    from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
    from engine.validation.crossasset_carry import build_fx_carry, build_rates_carry
    _, rw_c = commodity_cr()
    _, rw_f, _ = build_fx_carry()
    panels = [rw_c, rw_f]
    if include_rates:
        try:
            _, rw_r, _ = build_rates_carry()
            panels.append(rw_r)
        except Exception:
            pass
    rwide = pd.concat(panels, axis=1).sort_index()
    rwide = rwide.loc[:, ~rwide.columns.duplicated()]
    pos = _positions(rwide, looks=looks, target_vol=target_vol)
    gross = pos * rwide.sort_index()                      # realized instrument pnl
    if tc_bps_per_side > 0:
        turnover = pos.diff().abs().fillna(pos.abs())     # |Δnotional| per instrument-month
        cost = turnover * (tc_bps_per_side / 1e4)
        net = gross - cost
    else:
        net = gross
    port = net.mean(axis=1)                               # diversify across vol-scaled instruments
    pv = (port.rolling(12, min_periods=6).std() * np.sqrt(12)).replace(0, np.nan)
    return (port * (target_vol / pv).shift(1)).dropna().rename("cta_trend")


def _shp(x):
    x = x.dropna()
    return float(x.mean() * 12 / (x.std() * np.sqrt(12))) if x.std() > 0 else float("nan")


def _t(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if x.std() > 0 else float("nan")


def evaluate() -> dict:
    """CTA-appropriate evaluation: Sharpe, crisis/regime payoff, diversification vs carry."""
    from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
    from engine.validation.crossasset_carry import build_fx_carry, build_rates_carry, _xs_ls
    cta = build_cta_returns()
    # carry benchmark (the sleeve B must diversify, not dilute)
    cw_c, rw_c = commodity_cr(); cw_f, rw_f, _ = build_fx_carry()
    carry_c = _xs_ls(cw_c, rw_c, q=0.3); carry_f = _xs_ls(cw_f, rw_f, q=0.4)
    J = pd.concat([carry_c.rename("c"), carry_f.rename("f")], axis=1).dropna()
    carry = (J["c"] / J["c"].std() + J["f"] / J["f"].std())
    jc = pd.concat([cta.rename("cta"), carry.rename("carry")], axis=1).dropna()
    out = {"cta_sharpe": round(_shp(cta), 3), "cta_t": round(_t(cta), 2),
           "n_months": int(cta.dropna().size),
           "corr_vs_carry": round(float(jc["cta"].corr(jc["carry"])), 3)}
    # regime sub-periods (crisis-payoff lens)
    regimes = {}
    for c0, c1, lab in [(None, "2013-12-31", "2000-2013"), ("2013-12-31", "2018-12-31", "2014-2018"),
                        ("2018-12-31", None, "2019-2026")]:
        x = cta.dropna()
        if c0: x = x[x.index > c0]
        if c1: x = x[x.index <= c1]
        regimes[lab] = round(_shp(x), 2)
    out["regime_sharpe"] = regimes
    yr = (cta.dropna().groupby(cta.dropna().index.year).mean() * 12)
    out["yearly_positive"] = f"{int((yr > 0).sum())}/{len(yr)}"
    out["worst_year"] = round(float(yr.min()), 3)
    return out


if __name__ == "__main__":
    import json
    ev = evaluate()
    print("=== CTA trend sleeve (B) — insurance/diversification evaluation ===")
    print(json.dumps(ev, indent=2, ensure_ascii=False))
    # transparency: formal gate numbers (judged as alpha, which trend is NOT — for reference only)
    from engine.research.pipeline import run_gate
    g = run_gate(build_cta_returns(), name="cta_trend_replication", mechanism="trend",
                 n_trials=6, pead_control=True, log=False)
    print("\n=== formal gate (REFERENCE ONLY — trend judged on crisis-payoff, not HLZ alpha) ===")
    print(json.dumps({k: g.get(k) for k in
          ["standalone_sharpe", "alpha_t_ff5umd", "alpha_t_ff5umd_pead", "deflated_sr",
           "oos_sharpe", "corr_with_book", "verdict"]}, indent=2, ensure_ascii=False))
