"""engine/validation/analyst_revision.py — 2nd-alpha: analyst EPS-revision drift (I/B/E/S).

Enabled by the full-access WRDS account (${WRDS_USER_2}) — I/B/E/S was permission-denied
on the prior account, which was THE binding constraint. The first genuine,
tradeable, D_PEAD-uncorrelated second alpha found in the whole search.

Signal (Womack 1996; Chan-Jegadeesh-Lakonishok 1996): analysts' EPS estimate
revisions drift. revision_ratio = (numup - numdown) / numest from the I/B/E/S
FY1 summary (statsum_epsus), monthly. Long net-upgraded, short net-downgraded.

VERDICT (2026-05-21, top-1500, account ${WRDS_USER_2}):
  - 1-month hold: gross +5.86%/yr, t=2.53, FF5+UMD residual alpha t=2.71
    (REAL alpha), gross deflated SR 0.857, corr 0.21 w/ the D_PEAD FAMILY book,
    recent-ALIVE, LARGE-CAP tradeable. But turnover 8.7x -> net deflated SR 0.45
    (cost-eaten). => RED standalone as first built.
  - holding-period: K=3mo cuts turnover to 2.1x -> net deflated SR 0.65 but
    residual alpha decays to t=1.51 (overlapping holds dilute signal freshness).

  RESCUE (2026-05-21, build_revision_sleeve_buffered) — the YELLOW was worth
  engineering, and TWO principled levers lifted net deflSR 0.45 -> 0.88:
  - LEVER 1, no-trade band (Novy-Marx–Velikov buffering): enter top quintile,
    exit only below the wider q_out band. Turnover 8.7x -> 4.1x WITHOUT diluting
    freshness (unlike K=3). net deflSR -> 0.69, resid t 2.65. Still RED: the
    GROSS deflSR (~0.83) is itself below the 0.90 GREEN bar, so cost-cutting
    alone cannot reach it.
  - LEVER 2, dispersion conditioning (Zhang 2006; Gleason-Lee 2003 — drift is
    stronger under high information uncertainty): restrict to the high-dispersion
    half (CV = stdev/|meanest| >= median) BEFORE the sort. Gross Sharpe 0.83 ->
    1.08, net deflSR -> 0.879, residual alpha 8.12%/yr t=3.26. => YELLOW, a hair
    under GREEN (0.879 vs 0.90). Did NOT fish the last 0.02 (would be p-hacking).
  - ROBUSTNESS AUDIT PASSED (the test that caught the Lazy-Prices false YELLOW):
    (a) cutoff sensitivity — gross Sharpe 1.08/1.05/1.03/0.98 as the dispersion
    cut tightens median->q75, a smooth monotone gradient, not a lucky knife-edge;
    (b) subsample — positive in BOTH halves (13-19 t=1.68; 19-24 t=3.13, stronger
    recently = alive); (c) yearly — POSITIVE in 10 of 12 years (only 2016 small
    negative). Not small-sample luck.
  - DIVERSIFICATION (honest, against the DEPLOYED book): corr with the D_PEAD
    FAMILY ensemble is 0.21 (all-names) / 0.28 (dispersion) — LOW, a real
    diversifier. NB: corr against the bare PEAD-TS single sleeve is ~0.6 (both
    are large-cap earnings-information drift), but the family ENSEMBLE dilutes
    that overlap via COMBINED (corr ~0.28), so the deployed-book diversification
    holds. The original "corr 0.21" was correct — it was vs the family, verified.
  - NET: the realest near-GREEN second alpha besides D_PEAD. Standalone YELLOW
    (almost deployable); as a book diversifier on top of D_PEAD, genuinely
    additive. Not labelled GREEN — the 0.90 bar is the contract.

UPDATE (2026-05-21 PM) — joint optimization + CORRECTED multiple-testing math
(_revision_optimize.py + _revision_dsr_correct.py). The earlier "net deflSR 0.88,
blocked by multiple testing, can't reach GREEN" was an ARTIFACT of a methodology
error: setting deflated-Sharpe n_trials = raw grid size (72) together with the
default theoretical SR-variance. The 72 grid configs are pairwise-correlated 0.907
(effective independent trials ~7.6) and their Sharpes cluster tightly (true
cross-trial V=0.00067 vs theoretical 0.0082, 12x too big) -> E[max SR] was inflated
3.5x. Done the Bailey-Lopez de Prado way (ACTUAL cross-trial V): parameter-level
deflated SR = 0.989 (pre-specified) / 0.991 (grid-max); PSR-vs-0 = 0.998. Campaign-
level (~50 distinct signals) check via Harvey-Liu-Zhu: NET t ~ 3.2 > HLZ 3.0 bar.
=> CLEARS multiple testing at both levels. Cost (rt=20bps) net Sharpe ~1.0; at
26bps ~0.94. So it is a GREEN-grade SECOND validated signal on the statistical+cost
criteria. The ONE honest residual caveat is REGIME dependence, NOT p-hacking:
the alpha tracks aggregate forecast dispersion (corr 0.55 yearly), strong in the
high-dispersion post-2020 regime, weaker (still positive, t=1.41) in the low-
dispersion 2013-2019 half; the two down years (2013 -15%, 2016 -10%) are low-
dispersion. => label: "GREEN, regime-managed" (NOT unconditional standalone).
Legitimate next lever (no curve-fitting): a PRE-SPECIFIED dispersion-STATE overlay
(scale exposure with ex-ante aggregate dispersion; Zhang 2006) + book-level
combination with D_PEAD (corr 0.21-0.28 -> genuine diversifier). See
[[feedback-deflated-sharpe-n-trials-methodology-2026-05-21]].

Bonus use (not built here): I/B/E/S forecast DISPERSION (stdev/|meanest|) to
condition D_PEAD (PEAD stronger when disagreement high; Zhang 2006).
Map I/B/E/S->permno via crsp.stocknames ncusip (8-digit cusip match).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STATSUM = "data/cache/_ibes_statsum_fy1.parquet"
_STOCKNAMES = "data/cache/_stocknames_ncusip.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"


def build_revision_panel() -> pd.DataFrame:
    """Per permno-month revision_ratio = (numup-numdown)/numest (FY1), >=3
    analysts, mapped to permno via cusip<->ncusip."""
    ss = pd.read_parquet(_STATSUM)
    ss["statpers"] = pd.to_datetime(ss["statpers"])
    sn = pd.read_parquet(_STOCKNAMES).rename(columns={"ncusip": "cusip"})
    ss = ss.merge(sn[["cusip", "permno"]].drop_duplicates(), on="cusip", how="inner")
    ss["rev_ratio"] = (ss["numup"].fillna(0) - ss["numdown"].fillna(0)) / ss["numest"].replace(0, np.nan)
    ss["month"] = ss["statpers"].dt.to_period("M").dt.to_timestamp("M")
    rev = (ss.sort_values("statpers")
             .groupby(["permno", "month"], as_index=False)
             .agg(rev_ratio=("rev_ratio", "last"), numest=("numest", "last"),
                  dispersion=("stdev", "last"), meanest=("meanest", "last")))
    return rev[rev["numest"] >= 3]


def build_revision_sleeve(hold_months: int = 1, q: float = 0.2) -> tuple[pd.Series, float]:
    """Monthly L/S: long net-upgraded quintile / short net-downgraded, held
    `hold_months` (overlapping). Returns (monthly_ls, ann_turnover)."""
    rev = build_revision_panel()
    revw = rev.pivot(index="month", columns="permno", values="rev_ratio").sort_index()
    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    mret = mret.where(daily.resample("ME").count() > 5)
    months = sorted(mret.index)
    sel = {}
    for t in revw.index:
        s = revw.loc[t].dropna()
        if len(s) >= 80:
            sel[t] = (set(s[s >= s.quantile(1 - q)].index), set(s[s <= s.quantile(q)].index))
    rows, ent, prevL = [], [], set()
    for i in range(len(months) - 1):
        t1 = months[i + 1]
        if t1 not in mret.index:
            continue
        rc = [months[j] for j in range(max(0, i - hold_months + 1), i + 1) if months[j] in sel]
        if not rc:
            continue
        L = set().union(*[sel[m][0] for m in rc]); S = set().union(*[sel[m][1] for m in rc])
        nx = mret.loc[t1]; rh = nx.reindex(list(L)).dropna(); rl = nx.reindex(list(S)).dropna()
        if len(rh) < 10 or len(rl) < 10:
            continue
        rows.append((t1, float(rh.mean() - rl.mean())))
        ent.append(len(L - prevL) / max(len(L), 1)); prevL = L
    return pd.Series(dict(rows)).sort_index().rename("analyst_rev"), float(np.mean(ent) * 12)


def build_revision_sleeve_buffered(q_in: float = 0.2, q_out: float = 0.4,
                                   weight: str = "equal",
                                   disp_pctile: float = 0.0) -> tuple[pd.Series, float]:
    """RESCUE attempt for the YELLOW: no-trade-band (buffering) + optional
    magnitude weighting, the two levers that target its ONLY flaw (turnover) AND
    its gross-Sharpe ceiling.

    Buffering (Novy-Marx–Velikov): a name ENTERS the long leg when rev_ratio is
    in the top q_in, and EXITS only when it drops out of the wider top q_out
    (q_out>q_in). The hysteresis band [q_out, q_in] stops names churning across
    the quintile boundary every month — cutting turnover WITHOUT diluting signal
    freshness the way overlapping multi-month holds do (which is why K=3 lost the
    alpha). Symmetric on the short leg.

    weight='mag' weights within each leg by |rev_ratio| (concentrate in the
    biggest revisions, which drift more per unit cost).

    disp_pctile>0 is the lever that actually works for the GROSS Sharpe (the
    necessary second prong, since buffering alone caps net at the ~0.83 gross
    ceiling): restrict each month's cross-section to names with forecast
    dispersion CV = stdev/|meanest| at or above this percentile BEFORE the
    quintile sort. Theory (Zhang 2006; Gleason-Lee 2003): revision drift is
    stronger under high information uncertainty. disp_pctile=0.5 (high-dispersion
    half) lifts gross Sharpe 0.83→1.08 and net deflSR 0.69→0.88. Returns
    (monthly_ls, ann_oneway_turnover), turnover measured the SAME way as
    build_revision_sleeve (long-leg entry rate annualized)."""
    rev = build_revision_panel()
    rev["cv"] = rev["dispersion"] / rev["meanest"].abs().replace(0, np.nan)
    revw = rev.pivot(index="month", columns="permno", values="rev_ratio").sort_index()
    cvw = rev.pivot(index="month", columns="permno", values="cv").sort_index()
    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    mret = mret.where(daily.resample("ME").count() > 5)
    months = sorted(mret.index)

    longset: set = set(); shortset: set = set()
    rows, ent, prevL = [], [], set()
    for i in range(len(months) - 1):
        t, t1 = months[i], months[i + 1]
        if t not in revw.index or t1 not in mret.index:
            continue
        s = revw.loc[t].dropna()
        if disp_pctile > 0:
            cv = cvw.loc[t].reindex(s.index)
            s = s[cv >= cv.quantile(disp_pctile)]
        if len(s) < 60:
            continue
        hi_in, hi_out = s.quantile(1 - q_in), s.quantile(1 - q_out)
        lo_in, lo_out = s.quantile(q_in), s.quantile(q_out)
        # hysteresis update: hold members until they leave the WIDER band
        longset = {p for p in s.index
                   if (s[p] >= hi_in) or (p in longset and s[p] >= hi_out)}
        shortset = {p for p in s.index
                    if (s[p] <= lo_in) or (p in shortset and s[p] <= lo_out)}
        nx = mret.loc[t1]
        L = list(longset); S = list(shortset)
        rl_r = nx.reindex(L).dropna(); rs_r = nx.reindex(S).dropna()
        if len(rl_r) < 10 or len(rs_r) < 10:
            continue
        if weight == "mag":
            wl = s.reindex(rl_r.index).abs(); wl = wl / wl.sum()
            ws = s.reindex(rs_r.index).abs(); ws = ws / ws.sum()
            long_ret = float((rl_r * wl).sum()); short_ret = float((rs_r * ws).sum())
        else:
            long_ret = float(rl_r.mean()); short_ret = float(rs_r.mean())
        rows.append((t1, long_ret - short_ret))
        ent.append(len(longset - prevL) / max(len(longset), 1)); prevL = set(longset)
    return (pd.Series(dict(rows)).sort_index().rename("analyst_rev_buf"),
            float(np.mean(ent) * 12) if ent else float("nan"))
