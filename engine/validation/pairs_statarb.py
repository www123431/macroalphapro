"""engine/validation/pairs_statarb.py — track B: a GENUINELY different mechanism.

Everything validated so far (D_PEAD, analyst revision) is information-
underreaction. A multi-alpha fund needs a mechanism that is ORTHOGONAL at the
mechanism level, not just the return level — so that if underreaction decays
globally, the book does not die wholesale.

Statistical-arbitrage pairs trading (Gatev-Goetzmann-Rouwenhorst 2006, RFS):
relative-value MEAN REVERSION. Find pairs of stocks whose normalized prices
co-moved historically; when the spread diverges, long the laggard / short the
leader and bet on convergence. The edge is liquidity/demand-imbalance reversion,
NOT information — genuinely different mechanism, and market-neutral.

Method (GGR canonical):
  - Formation 12mo: normalized total-return price index (start=1) per name; for
    each pair, SSD = Σ(P_a − P_b)². Select the n_pairs with smallest SSD.
  - Trading next 6mo: re-index both to 1 at trade start; open when the spread
    diverges > z_open · σ_formation (long low / short high), close on spread sign
    flip (convergence) or window end.
  - Monthly-staggered cohorts (up to 6 active) → average → committed-capital
    daily return (P&L spread over ALL pairs incl. unopened = conservative).

Screened through alpha_factory.gate(); GREEN-only deploys. All cached, no WRDS.

VERDICT (2026-05-21, top-1500/2325, track B = genuinely different mechanism):
RED — but informative. Classic GGR distance pairs: gross monthly Sharpe 0.11,
net deflSR 0.01, residual t=-0.41, WEAK recently. DEAD, exactly as the literature
predicts (Do-Faff 2010: pairs profitability collapsed post-2002 from quant
adoption). A companion test of the MODERN variant — Avellaneda-Lee 2010 residual
reversion (market-neutralize each stock, trade mean-reversion of the idiosyncratic
s-score, weekly) — was also RED: gross Sharpe 0.60 / gross deflSR 0.67 but
residual alpha only t=1.07 (n.s.), WEAK recently, AND it is a HIGH-turnover
strategy so cost annihilates it (net deflSR 0.0 at ~50x turnover).
KEY FINDING: both stat-arb variants ARE genuinely orthogonal to PEAD (corr 0.03
and 0.06 — different MECHANISM, would diversify if they worked), but both are
dead/cost-annihilated in the liquid US universe. The mean-reversion family dies
the OPPOSITE way to PEAD: PEAD's alpha is small-cap-locked (cost wall on the
LONG side), reversal's alpha is turnover-locked (cost wall on TRADING FREQUENCY).
Same arbitrage wall, reached from the other side. This reinforces track A (take
the PROVEN PEAD edge to a less-arbitraged market) over hunting new mechanisms in
the most-arbitraged universe.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RET = "data/cache/crsp_hist_daily_ret.parquet"


def _wide_daily() -> pd.DataFrame:
    r = pd.read_parquet(_RET); r["date"] = pd.to_datetime(r["date"])
    return r.pivot_table(index="date", columns="permno", values="ret").sort_index()


def build_pairs_returns(n_pairs: int = 20, formation_days: int = 252,
                        trading_days: int = 126, z_open: float = 2.0,
                        max_names: int = 600) -> tuple[pd.Series, float]:
    """GGR pairs strategy → (daily committed-capital LS return series, avg
    annualized one-way turnover). Monthly-staggered 6mo cohorts."""
    wide = _wide_daily()
    dates = wide.index
    month_ends = pd.Series(dates).groupby([dates.year, dates.month]).last().values
    month_ends = pd.to_datetime(month_ends)

    daily_pl: dict[pd.Timestamp, list[float]] = {}   # date -> per-cohort committed returns
    open_events = 0; cohort_pair_days = 0

    for m in month_ends:
        f_idx = dates[dates <= m]
        if len(f_idx) < formation_days:
            continue
        form = wide.loc[f_idx[-formation_days:]]
        t_idx = dates[dates > m][:trading_days]
        if len(t_idx) < trading_days * 0.8:
            continue
        trade = wide.loc[t_idx]

        # names valid (no NaN) in BOTH windows -> no survivorship look-ahead
        valid = form.columns[(form.notna().all()) & (trade.notna().all())]
        if len(valid) < 30:
            continue
        if len(valid) > max_names:
            # keep the most-liquid proxy: lowest formation vol (stable, GGR-ish)
            vol = form[valid].std()
            valid = vol.nsmallest(max_names).index
        f = form[valid]; tr = trade[valid]

        # normalized formation price index (start 1)
        P = (1 + f.fillna(0)).cumprod().values            # (252, N)
        names = list(valid)
        # pairwise SSD via Gram: SSD[a,b] = s[a]+s[b]-2*G[a,b]
        G = P.T @ P
        s = np.einsum("ij,ij->j", P, P)
        SSD = s[:, None] + s[None, :] - 2 * G
        iu = np.triu_indices(len(names), k=1)
        ssd_pairs = SSD[iu]
        order = np.argsort(ssd_pairs)[:n_pairs]
        sigma_form = (P[:, iu[0]] - P[:, iu[1]]).std(axis=0)   # spread std per pair

        # trading: re-index to 1 at trade start
        Ptr = (1 + tr.fillna(0)).cumprod().values         # (126, N)
        rtr = tr.fillna(0).values                          # daily returns
        tdates = tr.index

        for k in order:
            ia, ib = iu[0][k], iu[1][k]
            sig = sigma_form[k]
            if sig <= 0:
                continue
            pos = 0  # +1: long a/short b ; -1: short a/long b
            for d in range(len(tdates)):
                spread = Ptr[d, ia] - Ptr[d, ib]
                # daily LS return for current position (committed capital)
                if pos != 0:
                    # long a short b => pos=+1 => r = r_a - r_b
                    pl = pos * (rtr[d, ia] - rtr[d, ib])
                    daily_pl.setdefault(tdates[d], []).append(pl)
                    cohort_pair_days += 1
                    if (pos == +1 and spread >= 0) or (pos == -1 and spread <= 0):
                        pos = 0   # converged
                else:
                    daily_pl.setdefault(tdates[d], []).append(0.0)
                    cohort_pair_days += 1
                    if spread > z_open * sig:
                        pos = -1; open_events += 1   # a rich -> short a/long b
                    elif spread < -z_open * sig:
                        pos = +1; open_events += 1   # a cheap -> long a/short b

    # committed-capital portfolio: mean across all pair-slots active that day
    ser = pd.Series({d: float(np.mean(v)) for d, v in daily_pl.items() if v}).sort_index()
    ser = ser.rename("pairs_statarb")
    # turnover: each open + each close ~ 2 one-way trades per opened event, /pairs
    ann_turnover = (open_events * 2) / max(cohort_pair_days / 252.0, 1e-9) / max(n_pairs, 1)
    return ser, float(ann_turnover)
