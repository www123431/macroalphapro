"""engine/validation/decay_sentinel.py — Decay Sentinel monitoring SUBSTANCE.

Right-sized agent for the now-TWO-mechanism book (per
project-agent-rightsizing-single-mechanism-2026-05-21). Single-mechanism risk was
"the one alpha decays"; the two-mechanism book adds the decisive new risk: the two
mechanisms STOP being independent (decay together / correlation rises) — the thing
that makes the book robust is corr≈0, so losing it is worse than either alpha softening.

This module is the deterministic monitoring substance (metrics + pre-specified alarm
thresholds); the agent persona / daily cron wrapper is a later, lighter layer
(substance-first doctrine). It tracks, per mechanism, the ROLLING health (Sharpe / t /
decay-ratio vs full sample) and, across mechanisms, the ROLLING correlation
(diversification integrity), and raises WARN/ALERT flags.

Mechanism 1 = equity earnings underreaction (D_PEAD + analyst-revision book).
Mechanism 2 = cross-asset carry (commodity + FX), spec id=77.

NB return-level Sharpe is the decay proxy available from return series; signal-level
IC / breadth / residual-alpha-t can be added when wired to the live signal pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MODULAR + CONFIG-DRIVEN design (do NOT hardcode the book in the monitoring logic).
#
#   layer 1  monitoring CORE   — pure functions on return/signal series + weights;
#                                generic over N mechanisms; never edited per-book.
#   layer 2  MechanismConfig   — the interface between core and config.
#   layer 3  PROVIDERS + build_mechanisms() — reads the LIVE book from the
#            single-source-of-truth registry (engine.strategies.get_registry()
#            sleeve_allocation_dict + load_all_strategy_returns_weekly, the same
#            source correlation_sentinel uses) so a book change (add/remove/reweight
#            a sleeve via adapters.py) propagates AUTOMATICALLY — no agent edit.
#            Research/candidate mechanisms not yet in the live registry (e.g. carry,
#            spec id=77 DRAFT) are added via one PROVIDERS entry with is_candidate=True.
# Adding a mechanism = a registry change (live) or one PROVIDERS row (candidate);
# the core logic and the report schema are untouched.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MechanismConfig:
    name: str
    returns: pd.Series                       # monthly return series
    weight: float                            # book risk-weight (from the registry)
    signal_panels: Optional[tuple] = None    # (signal_wide, ret_wide) for the IC-based
    is_candidate: bool = False               # structural-decay test; None -> return-only
    sleeve_id: Optional[str] = None          # registry sleeve this maps to (for the weight)
    role: str = "alpha"                      # alpha | insurance | trend | regime_premium
    # T2.2 (2026-06-05 audit S2 fix): blessed (honest_deploy) Sharpe from
    # the candidate pipeline's P-D8 calibration. When set, the structural-
    # decay bar scales as 0.5 x blessed_sharpe (with a floor) instead of
    # using the global SD_SHARPE_BAR. None falls back to the global.
    blessed_sharpe: Optional[float] = None


def _role_for(sleeve_id: "str | None", name: str) -> str:
    """A mechanism's ROLE decides HOW it's judged — a calm-period low Sharpe is decay
    for an ALPHA but BY DESIGN for INSURANCE (it pays in crises). Derived from the
    registry sleeve (single source of truth)."""
    s = (sleeve_id or "").lower()
    if "crisis" in s or "hedge" in s or "rms" in s:
        return "insurance"
    if "cta" in s or "trend" in s or "defensive" in s:
        return "trend"
    return "alpha"

# pre-specified alarm thresholds (documented; do not tune to suppress an alarm)
DECAY_WARN_FRAC = 0.50    # rolling Sharpe < 0.50 x full-sample Sharpe -> WARN
DECAY_ALERT_SHARPE = 0.0  # rolling Sharpe < 0 -> ALERT (mechanism not paying)
CORR_WARN = 0.40          # rolling |corr| > 0.40 (designed ~0.03-0.05) -> WARN
CORR_ALERT = 0.60         # rolling |corr| > 0.60 -> ALERT (diversification eroding)
ROLL = 36                 # rolling window (months)


def _ann_sharpe(r):
    r = r.dropna()
    return r.mean() * 12 / (r.std() * np.sqrt(12)) if len(r) > 1 and r.std() > 0 else np.nan


def mechanism_health(ret: pd.Series, window: int = ROLL) -> dict:
    """Full-sample + trailing-window Sharpe, the decay ratio (recent/full), and the
    rolling-Sharpe series, for one mechanism's monthly return stream."""
    ret = ret.dropna()
    full = _ann_sharpe(ret)
    roll = ret.rolling(window).apply(lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan, raw=False)
    recent = roll.iloc[-1] if len(roll.dropna()) else np.nan
    recent_t = (ret.iloc[-window:].mean() / ret.iloc[-window:].std() * np.sqrt(len(ret.iloc[-window:]))
                if len(ret) >= window and ret.iloc[-window:].std() > 0 else np.nan)
    return dict(full_sharpe=full, rolling_sharpe=recent, rolling_t=recent_t,
                decay_ratio=(recent / full) if (full and full > 0) else np.nan,
                roll_series=roll)


def cross_correlation(a: pd.Series, b: pd.Series, window: int = ROLL) -> dict:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    full = j["a"].corr(j["b"]) if len(j) > 2 else np.nan
    roll = j["a"].rolling(window).corr(j["b"])
    return dict(full_corr=full, rolling_corr=roll.iloc[-1] if len(roll.dropna()) else np.nan,
                roll_series=roll, n=len(j))


def downside_diagnostics(a: pd.Series, b: pd.Series, mkt: "pd.Series | None" = None) -> dict:
    """The metric that MATTERS for diversification: do the two co-MOVE on the DOWNSIDE
    (lose together) vs benign upside co-movement? Symmetric Pearson can't tell — high
    corr while both WIN (2020-22) is harmless; high corr while both LOSE is the real
    diversification failure. Returns: downside_corr (corr on both-negative months),
    co_drawdown_frac (% months both <0), both_down_combined (avg 50/50 return in those
    months), and stress_corr (corr when the market is in its worst quartile)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    bd = j[(j["a"] < 0) & (j["b"] < 0)]
    dn = j[(j["a"] < 0) | (j["b"] < 0)]
    down_corr = dn["a"].corr(dn["b"]) if len(dn) > 3 else np.nan
    stress_corr = np.nan
    if mkt is not None:
        m = mkt.reindex(j.index).dropna()
        if len(m) > 8:
            worst = j.loc[m[m <= m.quantile(0.25)].index]
            stress_corr = worst["a"].corr(worst["b"]) if len(worst) > 3 else np.nan
    return dict(downside_corr=down_corr,
                co_drawdown_frac=len(bd) / len(j) if len(j) else np.nan,
                both_down_combined=float((0.5 * bd["a"] + 0.5 * bd["b"]).mean()) if len(bd) else np.nan,
                both_down_n=len(bd), stress_corr=stress_corr)


def market_beta(ret: pd.Series, mkt: pd.Series, stress_only: bool = False) -> float:
    """OLS beta of a mechanism on the market (risk-appetite proxy). stress_only=True
    restricts to the market's worst-quartile months (the COMMON-FACTOR exposure that
    bites in stress, even when the unconditional beta is ~0)."""
    j = pd.concat([ret.rename("r"), mkt.rename("m")], axis=1).dropna()
    if stress_only:
        j = j[j["m"] <= j["m"].quantile(0.25)]
    if len(j) < 6 or j["m"].var() == 0:
        return np.nan
    return float(np.cov(j["r"], j["m"])[0, 1] / j["m"].var())


def crisis_payoff(ret: pd.Series, mkt: pd.Series) -> float:
    """INSURANCE / convex-hedge health = mean monthly return in the market's WORST-
    quartile months. Insurance (TLT/GLD crisis hedge, trend) is MEANT to drag in calm
    regimes — a calm-period low/negative Sharpe is BY DESIGN, not decay. The only
    question that matters for it is: does it still PAY when the market is stressed?
    > 0 = still hedging (doing its job); <= 0 = it stopped protecting (the real failure)."""
    j = pd.concat([ret.rename("r"), mkt.rename("m")], axis=1).dropna()
    if len(j) < 8:
        return np.nan
    worst = j[j["m"] <= j["m"].quantile(0.25)]
    return float(worst["r"].mean()) if len(worst) else np.nan


def evaluate_alarms(mechs: dict, xcorr: dict, dd: "dict | None" = None) -> list[tuple[str, str]]:
    """Return list of (level, message). level in {INFO, WARN, ALERT}. The diversification
    alarm fires on DOWNSIDE co-movement (lose together), NOT benign symmetric corr —
    high corr while both WIN is harmless; high corr while both LOSE is the failure."""
    out = []
    for name, h in mechs.items():
        if not np.isnan(h["rolling_sharpe"]):
            if h["rolling_sharpe"] < DECAY_ALERT_SHARPE:
                out.append(("ALERT", f"{name}: rolling {ROLL}m Sharpe {h['rolling_sharpe']:.2f} < 0 — mechanism not paying"))
            elif not np.isnan(h["decay_ratio"]) and h["decay_ratio"] < DECAY_WARN_FRAC:
                out.append(("WARN", f"{name}: rolling Sharpe {h['rolling_sharpe']:.2f} is {h['decay_ratio']:.0%} of full {h['full_sharpe']:.2f} — decaying"))
    # symmetric rolling corr is INFO only (it can't tell up- from down-co-movement)
    rc = abs(xcorr["rolling_corr"]) if not np.isnan(xcorr["rolling_corr"]) else np.nan
    if not np.isnan(rc) and rc > CORR_WARN:
        out.append(("INFO", f"symmetric rolling corr {xcorr['rolling_corr']:+.2f} elevated — check it's benign upside co-movement, not co-loss"))
    # the REAL diversification alarm = high DOWNSIDE/stress corr (lose together)
    if dd:
        dc = dd.get("downside_corr", np.nan); sc = dd.get("stress_corr", np.nan)
        worst = max([x for x in (dc, sc) if not np.isnan(x)], default=np.nan)
        if not np.isnan(worst):
            if worst > CORR_ALERT:
                out.append(("ALERT", f"DOWNSIDE/stress corr {worst:+.2f} > {CORR_ALERT} — mechanisms LOSE TOGETHER (diversification fails when it matters)"))
            elif worst > CORR_WARN:
                out.append(("WARN", f"DOWNSIDE/stress corr {worst:+.2f} > {CORR_WARN} — co-loss risk (full-sample corr understates stress)"))
    return out


def sentinel_report(mechanisms: "dict[str, MechanismConfig]", window: int = ROLL,
                    market: "pd.Series | None" = None) -> dict:
    """The DETERMINISTIC monitoring algorithm — pure math, NO LLM judgement (0-LLM-in-
    DECISION). Fully GENERIC over N mechanisms read from the book config: per-mechanism
    health, ALL pairwise symmetric + DOWNSIDE/stress correlations, common-factor betas,
    per-mechanism structural-decay (signal-IC where a panel exists, else return-only),
    and the disciplined re-allocation from the config weights. Adding an alpha extends
    every section automatically — the agent only NARRATES this output, never decides."""
    import itertools
    names = list(mechanisms)
    rets = {n: mechanisms[n].returns for n in names}
    health = {n: mechanism_health(rets[n], window) for n in names}
    betas = ({n: dict(beta=market_beta(rets[n], market), stress_beta=market_beta(rets[n], market, stress_only=True))
              for n in names} if market is not None else {})
    # ALL pairs (auto-extends as alphas are added)
    pairs = {}
    for a, b in itertools.combinations(names, 2):
        xc = cross_correlation(rets[a], rets[b], window)
        dd = downside_diagnostics(rets[a], rets[b], market)
        pairs[(a, b)] = dict(cross_corr=xc, downside=dd)
    roles = {n: mechanisms[n].role for n in names}
    crisis = ({n: crisis_payoff(rets[n], market) for n in names} if market is not None else {})
    # per-mechanism structural decay (signal-IC if the mechanism provides a panel; role-aware)
    decay = {}
    for n in names:
        ic_roll = None
        if mechanisms[n].signal_panels is not None:
            sig_wide, ret_wide = mechanisms[n].signal_panels
            ic_roll = rolling_signal_ic(sig_wide, ret_wide, window)
        decay[n] = assess_structural_decay(
            n, health[n]["roll_series"], ic_roll=ic_roll, role=roles[n],
            blessed_sharpe=mechanisms[n].blessed_sharpe,
            returns=rets[n],
        )
    base_weights = {n: mechanisms[n].weight for n in names}
    flags = {n: decay[n]["structural_decay"] for n in names}
    realloc = recommend_allocation(base_weights, flags, roles=roles)
    # ROLE-AWARE per-mechanism alarms (the refinement): an ALPHA is judged on Sharpe +
    # signal-IC; a convex HEDGE (insurance/trend) on CRISIS-PAYOFF (calm drag is BY
    # DESIGN, never an alarm); a REGIME_PREMIUM (carry) on SIGNAL-IC (it loses in
    # carry-unwinds by design, so a negative recent Sharpe alone is a regime drawdown).
    alarms = []
    for n in names:
        role = roles[n]; h = health[n]; dcy = decay[n]; rs = h["rolling_sharpe"]
        if role in ("insurance", "trend"):
            cp = crisis.get(n, np.nan)
            if not np.isnan(cp):
                if cp <= 0:
                    alarms.append(("ALERT", f"{n} [{role}]: crisis payoff {cp:+.2%}/mo <= 0 — no longer hedging in stress (its job)"))
                else:
                    alarms.append(("INFO", f"{n} [{role}]: crisis payoff {cp:+.2%}/mo (still protecting); calm rolling Sharpe {rs:+.2f} is by design, not decay"))
        elif role == "regime_premium":
            sic = dcy.get("signal_ic")
            if dcy["structural_decay"]:
                alarms.append(("ALERT", f"{n} [regime_premium]: signal-IC faded AND sustained low Sharpe — premium structurally gone, re-allocate"))
            elif sic is not None and sic <= SD_IC_BAR:
                alarms.append(("WARN", f"{n} [regime_premium]: signal-IC {sic:+.2f} <= 0 — premium thinning (returns alone not decisive for a regime premium)"))
            elif not np.isnan(rs) and rs < DECAY_ALERT_SHARPE:
                alarms.append(("INFO", f"{n} [regime_premium]: rolling Sharpe {rs:+.2f} < 0 but signal-IC intact — regime drawdown, HOLD (not decay)"))
        else:  # alpha
            if dcy["structural_decay"]:
                alarms.append(("ALERT", f"{n} [alpha]: STRUCTURAL DECAY confirmed ({dcy['reason']}) — re-allocate"))
            elif not np.isnan(rs) and rs < DECAY_ALERT_SHARPE:
                alarms.append(("ALERT", f"{n} [alpha]: rolling {window}m Sharpe {rs:.2f} < 0 — not paying"))
            elif not np.isnan(h["decay_ratio"]) and h["decay_ratio"] < DECAY_WARN_FRAC:
                alarms.append(("WARN", f"{n} [alpha]: rolling Sharpe {h['decay_ratio']:.0%} of full — decaying"))
    for (a, b), pv in pairs.items():
        alarms += [(lvl, f"({a},{b}) {msg}") for lvl, msg in evaluate_alarms({}, pv["cross_corr"], pv["downside"])]
    # OVERALL verdict is computed HERE (deterministic) — the narrator only states it,
    # never decides it (0-LLM-in-DECISION). INFO does not escalate the book verdict.
    levels = {lvl for lvl, _ in alarms}
    overall = "ACTION" if "ALERT" in levels else ("WATCH" if "WARN" in levels else "HEALTHY")
    realloc_action = any(flags.values())
    return dict(mechanisms=health, betas=betas, pairs=pairs, decay=decay, roles=roles, crisis=crisis,
                base_weights=base_weights, recommended_weights=realloc, alarms=alarms, window=window,
                overall=overall, realloc_action=realloc_action)


# --- structural-decay detection + disciplined re-allocation ---
SD_SHARPE_BAR = 0.15     # rolling Sharpe must be persistently below this …
SD_PERSIST = 18          # … for >= this many months (sustained, not a drawdown)
SD_IC_BAR = 0.0          # AND the signal-level IC must have faded to <= ~0 (premium gone)
SD_RECOVER = 0.40        # hysteresis: restore weight only once rolling Sharpe recovers above this
REALLOC_HAIRCUT = 0.50   # confirmed structural decay -> HALVE the weight (not zero)
# T2.2 (2026-06-05 audit S2 fix): when a sleeve has a blessed Sharpe
# (its calibrated honest_deploy_sharpe from the P-D8 audit), the decay
# bar = max(SD_SHARPE_BAR_FLOOR, blessed_sharpe x SD_BLESSED_FRACTION).
# Pre-T2.2 the global SD_SHARPE_BAR=0.15 worked OK for low-Sharpe sleeves
# (carry, blessed ~0.30 -> bar 0.15 ~= 50%) but was 7x too lenient for
# D_PEAD (blessed ~1.10 -> rolling Sharpe could drop to 0.16 ~= 85% decay
# yet stay above the bar). With per-sleeve calibration:
#   carry      blessed=0.30 -> bar 0.15 (status quo, floor binds)
#   D_PEAD     blessed=1.10 -> bar 0.55 (proper "halved" alarm)
#   tsmom      blessed=0.50 -> bar 0.25
#   analyst    blessed=0.65 -> bar 0.33
SD_BLESSED_FRACTION = 0.50
SD_SHARPE_BAR_FLOOR = 0.10


def _per_sleeve_bar(blessed_sharpe: "float | None") -> tuple[float, str]:
    """Resolve the structural-decay Sharpe bar for one mechanism.

    Returns (bar, source_label). source_label goes into the alarm
    message so the reviewer can tell which threshold fired.
    """
    if blessed_sharpe is None or not (isinstance(blessed_sharpe, (int, float))
                                      and blessed_sharpe > 0):
        return SD_SHARPE_BAR, "global"
    bar = max(SD_SHARPE_BAR_FLOOR, float(blessed_sharpe) * SD_BLESSED_FRACTION)
    return bar, f"per-sleeve (blessed={blessed_sharpe:.2f}*{SD_BLESSED_FRACTION:.0%})"


# T3.1 (2026-06-05 audit S3 fix): structural-break significance threshold.
# Chow test p-value < CHOW_P_THRESHOLD => post-break mean is significantly
# lower than pre-break mean => STRUCTURAL break (not cyclical).
CHOW_P_THRESHOLD = 0.05


def chow_test_decay(returns: pd.Series, break_n_periods: int = 18) -> dict:
    """Chow test for a level shift: did the last `break_n_periods`
    observations come from a distribution with a lower mean than
    the prior history?

    Mathematically a Chow test on a constant-only regression is
    equivalent to a one-sided two-sample t-test of mean equality.
    H0: mean_post >= mean_pre (no structural decline)
    H1: mean_post <  mean_pre (structural decline)

    Returns dict with:
      f_stat              chi-squared-style F statistic (t^2 form)
      p_value             one-sided p-value (lower = more break evidence)
      pre_mean / post_mean
      pre_n / post_n
      structural_break    bool: p_value < CHOW_P_THRESHOLD AND post < pre
      reason              short label for the alarm message

    Used by assess_structural_decay to distinguish a STRUCTURAL break
    (re-allocate) from a CYCLICAL drawdown (hold). Pre-T3.1, the only
    test was "rolling Sharpe below bar for N months", which can't
    tell the two apart — a 2008-style drawdown ticks the box even
    though the strategy is fine.
    """
    r = returns.dropna()
    if break_n_periods < 6 or len(r) < break_n_periods + 12:
        return dict(structural_break=False, reason="insufficient history for Chow",
                    f_stat=float("nan"), p_value=float("nan"),
                    pre_mean=float("nan"), post_mean=float("nan"),
                    pre_n=0, post_n=0)
    post = r.iloc[-break_n_periods:].values
    pre  = r.iloc[:-break_n_periods].values
    pre_mean  = float(pre.mean())
    post_mean = float(post.mean())
    # One-sided two-sample t-test (Welch — unequal variance is the safer
    # default; mechanism vol can shift with the level).
    try:
        from scipy import stats as _stats
        t_stat, p_two = _stats.ttest_ind(post, pre, equal_var=False)
        # Convert to one-sided: testing H1 mean_post < mean_pre,
        # one-sided p = p_two/2 if t < 0 else 1 - p_two/2
        if t_stat < 0:
            p_one = float(p_two) / 2.0
        else:
            p_one = 1.0 - float(p_two) / 2.0
        f_stat = float(t_stat) ** 2
    except Exception as exc:
        return dict(structural_break=False, reason=f"scipy unavailable: {exc}",
                    f_stat=float("nan"), p_value=float("nan"),
                    pre_mean=pre_mean, post_mean=post_mean,
                    pre_n=len(pre), post_n=len(post))
    is_break = (p_one < CHOW_P_THRESHOLD) and (post_mean < pre_mean)
    if is_break:
        reason = (f"Chow p={p_one:.3f}<{CHOW_P_THRESHOLD}: post-{break_n_periods} "
                  f"mean {post_mean:+.4f} vs prior {pre_mean:+.4f}")
    elif post_mean >= pre_mean:
        reason = f"Chow: post mean {post_mean:+.4f} >= pre {pre_mean:+.4f} (not declining)"
    else:
        reason = (f"Chow p={p_one:.3f}>={CHOW_P_THRESHOLD}: post-{break_n_periods} "
                  f"mean {post_mean:+.4f} not significantly < prior {pre_mean:+.4f} "
                  f"(cyclical drawdown, not structural)")
    return dict(structural_break=is_break, reason=reason,
                f_stat=float(f_stat), p_value=float(p_one),
                pre_mean=pre_mean, post_mean=post_mean,
                pre_n=int(len(pre)), post_n=int(len(post)))


def rolling_signal_ic(sig_wide: pd.DataFrame, ret_wide: pd.DataFrame, window: int = ROLL) -> pd.Series:
    """Per month: cross-sectional rank-IC between the signal and NEXT month's return
    (does high-signal still out-return low-signal?). Trailing-`window` mean. This is
    the 'is the premium STILL THERE' measure — robust to directional/regime losses
    (returns can be bad while the cross-sectional premium is intact)."""
    months = [m for m in sig_wide.index if m in ret_wide.index]
    ic = {}
    allm = sorted(ret_wide.index)
    pos = {m: i for i, m in enumerate(allm)}
    for m in months:
        i = pos.get(m)
        if i is None or i + 1 >= len(allm):
            continue
        s = sig_wide.loc[m].dropna(); nr = ret_wide.loc[allm[i + 1]]
        j = pd.concat([s.rename("s"), nr.rename("r")], axis=1).dropna()
        if len(j) >= 6:
            ic[m] = j["s"].rank().corr(j["r"].rank())
    return pd.Series(ic).sort_index().rolling(window).mean()


def assess_structural_decay(name: str, sharpe_roll: pd.Series,
                            ic_roll: "pd.Series | None" = None, role: str = "alpha",
                            blessed_sharpe: "float | None" = None,
                            returns: "pd.Series | None" = None) -> dict:
    """STRUCTURAL decay (re-allocate) vs CYCLICAL drawdown (hold). Requires SUSTAINED
    low rolling Sharpe AND (where a signal panel exists) a FADED signal-IC — returns
    being bad while IC stays positive = a directional/regime drawdown, NOT decay
    (cutting then would have sold carry's 2018 revival).

    ROLE-AWARE: insurance/trend are CONVEX HEDGES — a sustained low calm-Sharpe is BY
    DESIGN, so they are NEVER flagged structurally-decayed off the Sharpe path (they are
    judged on crisis-payoff instead). Their weight is set by their hedging role, not by
    a return premium that can fade.

    T2.2 (2026-06-05 audit S2 fix): blessed_sharpe (the sleeve's
    honest_deploy_sharpe from P-D8 calibration) lifts the decay bar
    above the global 0.15 floor for high-Sharpe sleeves. Without this,
    D_PEAD (blessed 1.10) had to drop 86% before tripping; now it
    trips at 50% (bar=0.55).

    T3.1 (2026-06-05 audit S3 fix): added Chow test gate. When a raw
    returns series is provided, the structural-decay flag REQUIRES
    both the Sharpe-dead condition AND a statistically significant
    structural break in the return mean (chow_test_decay p<0.05).
    Pre-T3.1, a 2008-style drawdown (rolling Sharpe sat below the bar
    for 18 months due to a single shock, not a regime change) would
    trip the flag — confusing drawdown with decay. With the Chow
    gate, decay is only flagged if the recent window's mean is
    significantly lower than the prior history, which a one-off
    drawdown won't usually satisfy.
    """
    sr = sharpe_roll.dropna()
    bar, bar_source = _per_sleeve_bar(blessed_sharpe)
    if role in ("insurance", "trend"):
        return dict(structural_decay=False,
                    reason=f"role={role}: convex hedge — judged on crisis-payoff, calm-Sharpe is by design",
                    rolling_sharpe=float(sr.iloc[-1]) if len(sr) else float("nan"),
                    signal_ic=None, sharpe_bar=bar, bar_source=bar_source,
                    chow=None)
    if len(sr) < SD_PERSIST:
        return dict(structural_decay=False, reason="insufficient history",
                    sharpe_bar=bar, bar_source=bar_source, chow=None)
    recent = sr.iloc[-SD_PERSIST:]
    sharpe_dead = bool((recent < bar).all())   # persistently near-dead at SLEEVE-SPECIFIC bar
    ic_faded = None
    if ic_roll is not None and len(ic_roll.dropna()):
        ic_faded = bool(ic_roll.dropna().iloc[-1] <= SD_IC_BAR)

    # T3.1: Chow test on the raw returns (when available). The gate fires
    # only if BOTH the Sharpe-dead path AND a significant level break agree.
    chow = None
    if returns is not None:
        chow = chow_test_decay(returns, break_n_periods=SD_PERSIST)
    chow_break = bool(chow["structural_break"]) if chow else None

    # decision: need persistence AND (premium gone, if measurable) AND (Chow break, if measurable)
    if ic_roll is not None:
        base_flag = sharpe_dead and bool(ic_faded)
        base_reason = (f"sharpe<{bar:.2f} ({bar_source}) for {SD_PERSIST}m={sharpe_dead}, "
                       f"signal-IC faded={ic_faded}")
    else:
        base_flag = sharpe_dead   # return-only: less reliable
        base_reason = (f"sharpe<{bar:.2f} ({bar_source}) for {SD_PERSIST}m={sharpe_dead} "
                       f"(RETURN-ONLY, no signal-IC — treat as candidate, confirm at signal level)")

    if chow is not None:
        # Combine: structural decay requires BOTH sharpe_dead AND chow_break
        flag = base_flag and chow_break
        if sharpe_dead and not chow_break:
            reason = (f"{base_reason}; CYCLICAL DRAWDOWN (Chow p={chow['p_value']:.3f}, "
                      f"post mean vs pre not significantly lower — HOLD, not decay)")
        else:
            reason = f"{base_reason}; {chow['reason']}"
    else:
        flag = base_flag
        reason = base_reason + " (Chow N/A — no returns supplied)"

    return dict(structural_decay=flag, reason=reason,
                rolling_sharpe=float(sr.iloc[-1]),
                signal_ic=float(ic_roll.dropna().iloc[-1]) if (ic_roll is not None and len(ic_roll.dropna())) else None,
                sharpe_bar=bar, bar_source=bar_source,
                chow=chow)


def recommend_allocation(base_weights: dict, structural_flags: dict,
                         roles: "dict | None" = None) -> dict:
    """Disciplined re-allocation: HALVE a confirmed-structurally-decayed mechanism's
    weight (not zero), redistribute to survivors, renormalise. Default = base weights
    (no action on drawdowns). Hysteresis is enforced by the caller persisting the flag
    until rolling Sharpe recovers above SD_RECOVER.

    ROLE-AWARE redistribution: freed weight flows to surviving RETURN sources
    (alpha / regime_premium) — NOT into insurance/trend hedges, whose size is set by
    their hedging role, not to fill a return gap. (If no return source survives, fall
    back to all survivors so the book stays invested.)"""
    w = dict(base_weights)
    roles = roles or {}
    if not any(structural_flags.values()):
        return w   # NO decay -> NO action: recommended == base verbatim (no renormalisation)
    freed = 0.0
    for name, decayed in structural_flags.items():
        if decayed and name in w:
            freed += w[name] * REALLOC_HAIRCUT
            w[name] *= (1 - REALLOC_HAIRCUT)
    return_survivors = [n for n in w if not structural_flags.get(n, False)
                        and roles.get(n, "alpha") in ("alpha", "regime_premium")]
    survivors = return_survivors or [n for n in w if not structural_flags.get(n, False)]
    if freed > 0 and survivors:
        sw = sum(w[n] for n in survivors)
        for n in survivors:
            w[n] += freed * (w[n] / sw if sw > 0 else 1 / len(survivors))
    tot = sum(w.values())
    return {n: v / tot for n, v in w.items()} if tot > 0 else w


# ── layer-3 PROVIDERS: per-mechanism data hookup (the ONE place to wire a mechanism)
def _equity_returns() -> pd.Series:
    from engine.validation.analyst_revision import build_revision_sleeve_buffered
    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet").iloc[:, 0]; d.index = pd.to_datetime(d.index)
    dp = ((1 + d.clip(-0.2, 0.2)).resample("ME").prod() - 1)
    rev, _ = build_revision_sleeve_buffered(q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
    E = pd.concat([dp.rename("dp"), rev.rename("rev")], axis=1).dropna()
    vdp = E["dp"].rolling(12).std().shift(1); vre = E["rev"].rolling(12).std().shift(1)
    w = (1 / vdp) / (1 / vdp + 1 / vre)
    return (w * E["dp"] + (1 - w) * E["rev"]).dropna()


def _carry_returns() -> pd.Series:
    from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
    from engine.validation.crossasset_carry import build_fx_carry, _xs_ls
    cw_c, rw_c = commodity_cr(); cw_f, rw_f, _ = build_fx_carry()
    cc = _xs_ls(cw_c, rw_c, q=0.3).rename("c"); cf = _xs_ls(cw_f, rw_f, q=0.4).rename("f")
    J = pd.concat([cc, cf], axis=1).dropna()
    wc, wf = 1 / J["c"].std(), 1 / J["f"].std()
    return ((wc * J["c"] + wf * J["f"]) / (wc + wf))


def _carry_signal() -> tuple:
    """signal panels for the IC-based structural-decay test (commodity leg)."""
    from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
    return commodity_cr()   # (carry_signal_wide, return_wide)


def _ac_returns() -> pd.Series:
    """AC_TLT_GLD monthly return (50/50 TLT/GLD) — the loader omits this sleeve, so
    rebuild it from the cross-asset price cache so all deployed sleeves are covered."""
    t = pd.read_parquet("data/cache/tsmom_crossasset_monthly.parquet"); t.index = pd.to_datetime(t.index)
    r = t[["TLT", "GLD"]].pct_change()
    return (0.5 * r["TLT"] + 0.5 * r["GLD"]).dropna().rename("AC_TLT_GLD")


def _dpead_signal() -> tuple:
    """D_PEAD signal panels for the IC-based structural-decay test: SUE cross-section
    (month x permno) vs the CRSP monthly return panel — does high-SUE still out-return
    low-SUE? Lets the equity leg use signal-gated decay, not return-only."""
    p = pd.read_parquet("data/cache/_pead_ts_panel_2014_2023.parquet").dropna(subset=["sue"])
    p["m"] = pd.to_datetime(p["rdq"]).dt.to_period("M").dt.to_timestamp("M")
    sue_wide = p.pivot_table(index="m", columns="permno", values="sue", aggfunc="last").sort_index()
    ret = pd.read_parquet("data/cache/crsp_hist_daily_ret.parquet"); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    ret_wide = ((1 + daily.fillna(0)).resample("ME").prod() - 1).where(daily.resample("ME").count() > 5)
    return (sue_wide, ret_wide)


# per-strategy signal panels (for signal-gated structural-decay) + return fallbacks
# (strategies the weekly loader omits). The ONE place to wire a deployed strategy's data.
SIGNAL_PROVIDERS: "dict[str, callable]" = {"D_PEAD": _dpead_signal}
RETURN_FALLBACKS: "dict[str, callable]" = {"AC_TLT_GLD": _ac_returns}


# CANDIDATE mechanisms = validated but NOT yet in the live registry (one row each).
# returns() -> monthly Series; signal() -> (sig_wide, ret_wide) for the IC test or None.
CANDIDATES: "dict[str, dict]" = {
    "cross_asset_carry": dict(returns=_carry_returns, signal=_carry_signal,
                              weight=0.30, sleeve_id=None, role="regime_premium"),  # spec id=77 DRAFT (regime-dependent risk premium)
}


def _to_monthly(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy(); weekly.index = pd.to_datetime(weekly.index)
    return (1 + weekly.fillna(0)).resample("ME").prod() - 1


# T2.2 (2026-06-05 audit S2 fix): blessed Sharpe per sleeve.
# Source: P-D8 honest_deploy_sharpe calibration from each sleeve's
# candidate pipeline report (data/research/*_pipeline_report.json or
# the strict-gate evidence doc). Hand-maintained for now — future
# refactor (T3+) should pull from library YAML.
# Sleeve key matches MechanismConfig.sleeve_id from get_registry().
_BLESSED_SHARPE_BY_SLEEVE: "dict[str, float]" = {
    # Mechanism 1: equity earnings underreaction
    "d_pead":                1.10,   # PEAD anchor, OOS Sharpe
    "analyst_revision":      0.65,   # conditional GREEN

    # Mechanism 2: cross-asset carry (spec 77 §9-§10)
    "cross_asset_carry":     0.83,   # 4-leg honest_deploy OOS (was 1.10 IS)

    # Mechanism: trend-following (TSMOM B-axis, 2026-05-29)
    "tsmom_futures":         0.50,   # post-FF5+UMD orthogonalization

    # Insurance roles — convex hedges, blessed_sharpe N/A
    # (role-aware logic in assess_structural_decay() skips them anyway)
    "path_c_tail_hedge":     None,   # judged on crisis payoff
    "mom_hedge_overlay":     None,   # being deprecated 2026-05-31
    "crisis_hedge":          None,
}


def build_mechanisms() -> "dict[str, MechanismConfig]":
    """CONFIG-DRIVEN: monitor the LIVE deployed strategies straight from the single-
    source-of-truth registry — names from get_registry(), effective book weights from
    StrategyModule.book_weight(sleeve_allocation), returns from
    load_all_strategy_returns_weekly() resampled to monthly (aligns with the monthly
    candidate series). A book change (add/remove/reweight a strategy via adapters.py)
    propagates AUTOMATICALLY. Validated-but-not-deployed mechanisms (carry, spec id=77)
    are added from CANDIDATES (one row). No agent edit on a book change."""
    out: dict = {}
    try:
        from engine.strategies import get_registry
        from engine.portfolio.replay_combined import load_all_strategy_returns_weekly
        reg = get_registry(); alloc = reg.sleeve_allocation_dict()
        mret = _to_monthly(load_all_strategy_returns_weekly())   # weekly -> monthly
        for name in reg.names():                                 # ALL deployed strategies
            s = reg.get(name)
            if name in mret.columns:
                rets = mret[name].dropna()
            elif name in RETURN_FALLBACKS:                       # loader omits it (e.g. AC) -> rebuild
                rets = RETURN_FALLBACKS[name]()
            else:
                logger.warning("no return series for deployed strategy %s — skipped", name); continue
            try:
                w = float(s.book_weight(alloc))
            except Exception:
                w = float(alloc.get(getattr(s.META, "sleeve_id", None), 0.0)) * float(getattr(s.META, "intra_sleeve_weight", 1.0))
            sig = SIGNAL_PROVIDERS[name]() if name in SIGNAL_PROVIDERS else None
            sid = getattr(s.META, "sleeve_id", None)
            out[name] = MechanismConfig(name=name, returns=rets, weight=w,
                                        signal_panels=sig, is_candidate=False,
                                        sleeve_id=sid, role=_role_for(sid, name),
                                        blessed_sharpe=_BLESSED_SHARPE_BY_SLEEVE.get((sid or "").lower()))
    except Exception as exc:                                     # never let a wiring hiccup break monitoring
        logger.warning("live registry/loader unavailable (%s) — monitoring candidates only", exc)
    for name, c in CANDIDATES.items():                           # validated, not-yet-deployed
        out[name] = MechanismConfig(name=name, returns=c["returns"](), weight=float(c["weight"]),
                                    signal_panels=(c["signal"]() if c["signal"] else None),
                                    is_candidate=True, sleeve_id=c["sleeve_id"],
                                    role=c.get("role", "alpha"),
                                    blessed_sharpe=_BLESSED_SHARPE_BY_SLEEVE.get((c["sleeve_id"] or "").lower()))
    return out


def _market_monthly():
    f = pd.read_parquet("data/cache/ff_factors_weekly.parquet"); f.index = pd.to_datetime(f.index)
    return ((1 + f["Mkt-RF"]).resample("ME").prod() - 1).rename("mkt")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mechs = build_mechanisms()                 # CONFIG-DRIVEN (weights from registry)
    rep = sentinel_report(mechs, market=_market_monthly())
    print("\n" + "=" * 76)
    print(f"DECAY SENTINEL — config-driven book health (rolling {rep['window']}m, {len(mechs)} mechanisms)")
    print("  [monitoring config DERIVED from get_registry() — deterministic, NO LLM judgement]")
    print("=" * 76)
    for n, h in rep["mechanisms"].items():
        b = rep["betas"].get(n, {}); dcy = rep["decay"][n]; role = rep["roles"][n]
        cand = "*" if mechs[n].is_candidate else " "
        cp = rep["crisis"].get(n, float("nan"))
        extra = (f"crisis-payoff {cp:+.2%}/mo" if role in ("insurance", "trend") and not np.isnan(cp)
                 else f"structural_decay={dcy['structural_decay']}")
        print(f"  {cand}{n:17s} [{role:14s}] wt {mechs[n].weight:.0%} | full Sh {h['full_sharpe']:.2f} "
              f"roll {h['rolling_sharpe']:+.2f} ({h['decay_ratio']:.0%}) | mkt-beta {b.get('beta',float('nan')):+.2f} | {extra}")
    print("\n  PAIRWISE DIVERSIFICATION (downside/stress = the ones that matter):")
    for (a, b), pv in rep["pairs"].items():
        xc, dd = pv["cross_corr"], pv["downside"]
        print(f"    {a} × {b}: symmetric roll {xc['rolling_corr']:+.2f} | DOWNSIDE {dd['downside_corr']:+.2f} | STRESS {dd['stress_corr']:+.2f}")
    print(f"\n  ALLOCATION: base(registry) {{{', '.join(f'{k}:{v:.0%}' for k,v in rep['base_weights'].items())}}}"
          f" -> recommended {{{', '.join(f'{k}:{v:.0%}' for k,v in rep['recommended_weights'].items())}}}")
    print("    (re-allocate ONLY on confirmed structural decay — signal-IC gated, halve+hysteresis, not drawdown-chasing)")
    print("\n  ALARMS:", "none." if not rep["alarms"] else "")
    for lvl, msg in rep["alarms"]:
        print(f"    [{lvl}] {msg}")
    print("=" * 76)


if __name__ == "__main__":
    main()
