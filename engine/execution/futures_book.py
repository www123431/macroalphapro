"""engine/execution/futures_book.py — run the futures sleeve (B + carry) through the realistic
FuturesSimAdapter and validate fidelity vs the frictionless backtest.

Step 1 (this module): historical REPLAY of sleeve B (CTA trend) through FuturesSimAdapter at
institutional scale — whole contracts + futures accounting + slippage. If the realistic-sim NAV
return tracks the frictionless cta_trend series (high corr, similar Sharpe, small tracking error),
the realism layer does NOT break the strategy → the durable internal futures sim is faithful.

Per-instrument exposure of B at month t (fraction of sleeve equity):
    weight_{t,i} = vt_t * pos_{t,i} / N_t
where pos = vol-scaled blended-lookback TSMOM positions (already lagged 1m), vt = portfolio
vol-target scalar (lagged), N = # active instruments. The month-t exposure earns r_{t,i}.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from engine.execution.futures_specs import contract_notional, yf_ticker
from engine.execution.futures_sim_adapter import FuturesSimAdapter
from engine.execution.rebalancer import rebalance


def b_weights_and_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(weights_{t,i}, returns_{t,i}) monthly for sleeve B over commodity + FX + rates."""
    from engine.validation.cta_trend import _positions, DEFAULT_TARGET_VOL
    from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
    from engine.validation.crossasset_carry import build_fx_carry, build_rates_carry
    _, rw_c = commodity_cr()
    _, rw_f, _ = build_fx_carry()
    _, rw_r, _ = build_rates_carry()
    rwide = pd.concat([rw_c, rw_f, rw_r], axis=1).sort_index()
    rwide = rwide.loc[:, ~rwide.columns.duplicated()]
    pos = _positions(rwide)
    port = (pos * rwide).mean(axis=1)
    pv = (port.rolling(12, min_periods=6).std() * np.sqrt(12)).replace(0, np.nan)
    vt = (DEFAULT_TARGET_VOL / pv).shift(1)
    n = pos.notna().sum(axis=1).replace(0, np.nan)
    weights = pos.mul(vt / n, axis=0)
    return weights, rwide


def _xs_positions(cwide: pd.DataFrame, q: float) -> pd.DataFrame:
    """Per-instrument carry L/S weights (+1/n_long for top-q carry, -1/n_short for bottom-q),
    indexed by the month the position is HELD (aligned to that month's return) — reconstructs the
    weights inside crossasset_carry._xs_ls so Σ weight×return reproduces the carry L/S series."""
    allm = sorted(cwide.index)
    rows = {}
    for i in range(len(allm) - 1):
        mth, nxt = allm[i], allm[i + 1]
        c = cwide.loc[mth].dropna()
        if len(c) < 4:
            continue
        hi = c[c >= c.quantile(1 - q)].index
        lo = c[c <= c.quantile(q)].index
        w = pd.Series(0.0, index=c.index)
        if len(hi):
            w[hi] = 1.0 / len(hi)
        if len(lo):
            w[lo] = -1.0 / len(lo)
        rows[nxt] = w
    return pd.DataFrame(rows).T.sort_index()


def combined_futures_weights(include_trend: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Net per-instrument weights of the futures sleeve. include_trend=True → carry (commodity⊕FX,
    risk-parity) ⊕ B (CTA trend) [used by the replay-validation, reproduces _carry_trend_run].
    include_trend=False → CARRY ONLY = the correct OPERATIONAL futures sleeve: the trend sleeve
    deploys on ALPACA ETFs (spec 75 / tsmom_crisis), NOT the futures sim (see cta_trend.py
    reconciliation). The forward runner uses carry-only."""
    from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
    from engine.validation.crossasset_carry import build_fx_carry
    cw_c, rw_c = commodity_cr()
    cw_f, rw_f, _ = build_fx_carry()
    cc_pos = _xs_positions(cw_c, 0.3)        # commodity carry L/S (q=0.3, matches build_commodity)
    cf_pos = _xs_positions(cw_f, 0.4)        # FX carry L/S (q=0.4, matches build_fx_carry)
    b_w, rwide = b_weights_and_returns()

    def ret_of(posdf):
        return (posdf * rwide.reindex(columns=posdf.columns)).sum(axis=1)

    cc_r, cf_r, b_r = ret_of(cc_pos), ret_of(cf_pos), ret_of(b_w)
    # risk-parity within carry (commodity vs FX), on the common window
    jc = pd.concat([cc_r.rename("c"), cf_r.rename("f")], axis=1).dropna()
    sc, sf = jc["c"].std(), jc["f"].std()
    wcc, wcf = (1 / sc) / (1 / sc + 1 / sf), (1 / sf) / (1 / sc + 1 / sf)
    carry_pos = cc_pos.mul(wcc).add(cf_pos.mul(wcf), fill_value=0.0)
    if not include_trend:
        return carry_pos.sort_index(), rwide          # CARRY ONLY (operational futures sleeve)
    carry_r = ret_of(carry_pos)
    # risk-parity between carry and B (the trend sleeve) — for the replay-validation only
    jt = pd.concat([carry_r.rename("carry"), b_r.rename("b")], axis=1).dropna()
    scy, sb = jt["carry"].std(), jt["b"].std()
    wcy, wb = (1 / scy) / (1 / scy + 1 / sb), (1 / sb) / (1 / scy + 1 / sb)
    net = carry_pos.mul(wcy).add(b_w.mul(wb), fill_value=0.0)
    return net.sort_index(), rwide


def _shp(x):
    x = x.dropna()
    return float(x.mean() * 12 / (x.std() * np.sqrt(12))) if x.std() > 0 else float("nan")


def replay_b(book_equity: float = 10_000_000.0, use_micro: bool = True,
             slippage_bps: float = 1.0) -> dict:
    """Replay B through FuturesSimAdapter (whole contracts). Compare realistic-sim NAV return to the
    frictionless cta_trend series."""
    from engine.validation.cta_trend import build_cta_returns
    weights, rwide = b_weights_and_returns()
    tradable = [s for s in weights.columns if contract_notional(s, use_micro=use_micro)]
    weights = weights[tradable]

    fs = FuturesSimAdapter(starting_equity=book_equity, use_micro=use_micro,
                           slippage_bps=slippage_bps, state_path=None)
    fs.seed_notionals_from_specs(tradable)

    months = [m for m in weights.index if weights.loc[m].notna().any()]
    nav = {}
    for t in months:
        w = weights.loc[t].dropna()
        target = {s: float(w[s]) for s in w.index if abs(w[s]) > 1e-9}
        rebalance(fs, target, allow_fractional=True)          # submit rounds to whole contracts
        r = rwide.loc[t].dropna() if t in rwide.index else pd.Series(dtype=float)
        held = fs.get_positions()
        eq = fs.mark({s: float(r.get(s, 0.0)) for s in held}, date=str(t)[:10])
        nav[t] = eq

    navs = pd.Series(nav).sort_index()
    sim_ret = navs.pct_change().dropna()
    frictionless = build_cta_returns()                        # gross frictionless B
    j = pd.concat([sim_ret.rename("sim"), frictionless.rename("fl")], axis=1).dropna()
    n_missing = len(weights.columns) - len(tradable)
    return {
        "book_equity": book_equity, "use_micro": use_micro, "n_instruments": len(tradable),
        "n_no_yf_or_spec": n_missing,
        "sim_sharpe": round(_shp(sim_ret), 3), "frictionless_sharpe": round(_shp(frictionless), 3),
        "corr_sim_vs_frictionless": round(float(j["sim"].corr(j["fl"])), 3) if len(j) > 12 else None,
        "sim_ann_ret_pct": round(float(sim_ret.mean() * 12 * 100), 2),
        "tracking_error_ann_pct": round(float((j["sim"] - j["fl"]).std() * np.sqrt(12) * 100), 2)
        if len(j) > 12 else None,
        "n_months": int(len(sim_ret)),
    }


def replay_combined(book_equity: float = 10_000_000.0, use_micro: bool = True,
                    slippage_bps: float = 1.0) -> dict:
    """Replay the FULL futures sleeve (carry⊕B) through FuturesSimAdapter; compare to the frictionless
    combined series (Σ net_weight × return = _carry_trend_run's risk-parity carry+trend)."""
    net, rwide = combined_futures_weights()
    tradable = [s for s in net.columns if contract_notional(s, use_micro=use_micro)]
    net = net[tradable]
    frictionless = (net * rwide.reindex(columns=tradable)).sum(axis=1).rename("fl")

    fs = FuturesSimAdapter(starting_equity=book_equity, use_micro=use_micro,
                           slippage_bps=slippage_bps, state_path=None)
    fs.seed_notionals_from_specs(tradable)
    nav = {}
    for t in [m for m in net.index if net.loc[m].abs().sum() > 1e-9]:
        w = net.loc[t].dropna()
        rebalance(fs, {s: float(w[s]) for s in w.index if abs(w[s]) > 1e-9}, allow_fractional=True)
        r = rwide.loc[t].dropna() if t in rwide.index else pd.Series(dtype=float)
        eq = fs.mark({s: float(r.get(s, 0.0)) for s in fs.get_positions()}, date=str(t)[:10])
        nav[t] = eq
    navs = pd.Series(nav).sort_index()
    sim_ret = navs.pct_change().dropna()
    j = pd.concat([sim_ret.rename("sim"), frictionless], axis=1).dropna()
    return {
        "book_equity": book_equity, "use_micro": use_micro, "n_instruments": len(tradable),
        "sim_sharpe": round(_shp(sim_ret), 3), "frictionless_sharpe": round(_shp(frictionless), 3),
        "corr_sim_vs_frictionless": round(float(j["sim"].corr(j["fl"])), 3) if len(j) > 12 else None,
        "tracking_error_ann_pct": round(float((j["sim"] - j["fl"]).std() * np.sqrt(12) * 100), 2)
        if len(j) > 12 else None,
        "gross_leverage": round(float(net.abs().sum(axis=1).dropna().iloc[-1]), 2),
        "n_months": int(len(sim_ret)),
    }


_FWD_STATE = "data/execution/futures_sim_state.json"
_FWD_PRICES = "data/execution/futures_fwd_prices.json"


def fetch_yf_prices(syms) -> dict[str, float]:
    """Latest continuous-futures close per sym via yfinance (free, durable forward marks)."""
    from engine.execution.futures_specs import yf_ticker
    try:
        import yfinance as yf
    except Exception:
        return {}
    tmap = {s: yf_ticker(s) for s in syms if yf_ticker(s)}
    if not tmap:
        return {}
    out = {}
    try:
        data = yf.download(list(set(tmap.values())), period="7d", progress=False, auto_adjust=True)
        close = data["Close"] if "Close" in data else data
        last = close.ffill().iloc[-1]
        for s, t in tmap.items():
            try:
                px = float(last[t]) if t in last.index else float(last)
                if px > 0:
                    out[s] = px
            except Exception:
                continue
    except Exception:
        pass
    return out


def run_futures_forward(book_equity: float = 10_000_000.0, use_micro: bool = True,
                        submit: bool = True, state_path: str = _FWD_STATE,
                        price_cache: str = _FWD_PRICES) -> dict:
    """Operational forward step for the futures sleeve: mark NAV by yfinance returns since the last
    run, then rebalance to the current carry⊕B target (whole contracts). Persists state → accrues a
    durable forward OOS. NOTE: signal freshness depends on the cached carry/B panels refreshing
    (B is yfinance-derivable; carry needs the futures CURVE → the documented WRDS/curve refresh item)."""
    import datetime as _dt
    net, _ = combined_futures_weights(include_trend=False)   # CARRY ONLY (trend → Alpaca ETF, spec 75)
    tgt_month = net.index[-1]
    w = net.loc[tgt_month].dropna()
    syms = [s for s in w.index if yf_ticker(s) and contract_notional(s, use_micro=use_micro)]

    fs = FuturesSimAdapter(starting_equity=book_equity, use_micro=use_micro, state_path=state_path)
    yf_now = fetch_yf_prices(syms)
    last = {}
    if os.path.exists(price_cache):
        last = json.load(open(price_cache, encoding="utf-8"))

    marked = {}
    if last:                                   # mark NAV by realized return since last run
        rets = {s: yf_now[s] / last[s] - 1 for s in yf_now if s in last and last[s] > 0}
        if submit and rets:
            fs.mark(rets, date=_dt.date.today().isoformat())
        marked = rets
    else:                                      # first run: anchor notionals to current prices
        fs.seed_notionals_from_specs(syms)

    target = {s: float(w[s]) for s in syms if abs(w[s]) > 1e-9}
    rep = rebalance(fs, target, allow_fractional=True, dry_run=not submit)
    if submit:
        fs.save()                              # persist contracts (first run has no mark → must save)
        if yf_now:
            os.makedirs(os.path.dirname(price_cache), exist_ok=True)
            json.dump({**last, **yf_now}, open(price_cache, "w", encoding="utf-8"), indent=2)

    return {"target_month": str(tgt_month)[:10], "n_target": len(target), "n_priced_yf": len(yf_now),
            "n_marked": len(marked), "equity": round(fs.get_account().equity, 2),
            "n_contracts_held": len(fs.get_positions()),
            "orders": rep.n_orders if hasattr(rep, "n_orders") else len(rep.orders),
            "no_yf": sorted(s for s in w.index if not yf_ticker(s))}


if __name__ == "__main__":
    print("=== B only ===")
    for eq, mic in [(10_000_000, True), (1_000_000, False)]:
        print(json.dumps(replay_b(book_equity=eq, use_micro=mic), ensure_ascii=False))
    print("=== COMBINED carry+B ===")
    for eq, mic in [(10_000_000, True), (10_000_000, False)]:
        print(json.dumps(replay_combined(book_equity=eq, use_micro=mic), ensure_ascii=False))
