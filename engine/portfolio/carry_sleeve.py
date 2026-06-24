"""engine/portfolio/carry_sleeve.py — deployable cross-asset CARRY sleeve return engine.

Increment 1 of deploying the validated-GREEN cross-asset carry sleeve into the live
(paper) book (user-approved 2026-05-24, "收益流 sleeve"). This module is the sleeve's
P&L ENGINE: it reproduces the EXACT validated construction — risk-parity (inverse-vol)
combine of commodity-carry-L/S + FX-carry-L/S (engine/validation/crossasset_carry.py,
spec id=77, GREEN: Sharpe 0.66 / t 3.36 / deflSR 0.998 / equity-orthogonal) — and
vol-targets it for sizing. It is NOT a proxy ETF and NOT a frozen backtest number; it
recomputes from the validated leg builders on the cached futures-curve data.

Deliberately ISOLATED + non-breaking: it does not touch the combined-book NAV pipeline
or the live sleeve allocation. Those are the governance-sensitive next increments
(framework return-stream hook → sleeve re-weight to include carry → NAV wiring → RM cap
→ spec-77 amend → futures-data refresh). Build them deliberately, with tests.

The pure functions (risk_parity_combine / vol_target) are unit-tested on synthetic data;
build_carry_sleeve_returns() reuses the validated builders and is exercised by a smoke
test that skips if the futures cache is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The validated GREEN book config vol-targets each sleeve to ~10% annual
# (project_commodity_carry_yellow_2026-05-21, commit db0897e). Keep that default so the
# deployed sleeve matches what the 0.96→1.04 book Sharpe was measured at.
DEFAULT_TARGET_ANNUAL_VOL: float = 0.10
MONTHS_PER_YEAR: int = 12


def risk_parity_combine(legs: dict[str, pd.Series]) -> pd.Series:
    """Inverse-vol (equal-risk) combine of carry legs — the EXACT validated logic from
    _crossasset_carry_run.py: w_c = 1/std(leg_c), normalized. Returns the combined
    monthly L/S series. NaNs in a leg are treated as 0 for that month (a leg can start
    later); a zero-vol or all-NaN leg is dropped."""
    J = pd.DataFrame(legs)
    if J.empty:
        return pd.Series(dtype=float, name="carry_combined")
    w = {c: 1.0 / J[c].std() for c in J.columns if J[c].std() and J[c].std() > 0}
    if not w:
        return pd.Series(dtype=float, name="carry_combined")
    W = sum(w.values())
    comb = sum(w[c] * J[c].fillna(0.0) for c in w) / W
    return comb.rename("carry_combined")


def vol_target(series: pd.Series, target_annual_vol: float = DEFAULT_TARGET_ANNUAL_VOL,
               periods_per_year: int = MONTHS_PER_YEAR) -> pd.Series:
    """Scale a return series to a target annualized vol (constant full-sample scalar —
    a sizing dial, NOT a return forecast). return ≈ Sharpe × target_vol; this sets the
    target_vol. Sharpe is unchanged by construction (it's a linear scaling)."""
    s = series.dropna()
    realized = s.std() * np.sqrt(periods_per_year)
    if not realized or realized <= 0:
        return series
    return (series * (target_annual_vol / realized)).rename(getattr(series, "name", None))


def build_carry_sleeve_returns(target_annual_vol: float = DEFAULT_TARGET_ANNUAL_VOL,
                               include_rates: bool = True,
                               include_rates_xc: bool = True) -> pd.Series:
    """The deployable cross-asset carry sleeve MONTHLY return series, vol-targeted.

    Spec 77 §9 amendment 2026-05-28 added US Treasury (UST2/5/10/30) as a 3rd carry
    leg, after the data-quality fix that resurrected the rates settlement curve.
    Spec 77 §10 amendment 2026-05-28 added cross-country G10 government bond futures
    (Bund/Gilt/CGB/AGB/JGB/BTP/OAT 10Y) as a 4th leg — the rates analogue of the
    9-currency FX expansion that pushed combined carry over institutional bars.

    Upgrade trajectory 2026-05-28:
      2-leg cmdty+FX (pre-amend, raw data)         Sharpe 0.66, t 3.36
      2-leg cmdty+FX (deduped)                     Sharpe 0.74, t 3.80
      3-leg +rates_us (§9)                         Sharpe 0.85, t 4.36
      4-leg +rates_xc (§10)                        Sharpe 1.10, t 5.63

    `include_rates_xc=False` reproduces the §9 3-leg path; `include_rates=False`
    reproduces the pre-amendment 2-leg path. Both kept for regression / A-B.
    """
    from engine.validation.crossasset_carry import (
        build_commodity_carry_ls, build_fx_carry, build_rates_carry, build_rates_xc_carry,
    )
    legs = {"cmdty": build_commodity_carry_ls(), "fx": build_fx_carry()[2]}
    if include_rates:
        legs["rates_us"] = build_rates_carry()[2]
    if include_rates_xc:
        legs["rates_xc"] = build_rates_xc_carry()[2]
    comb = risk_parity_combine(legs)
    return vol_target(comb, target_annual_vol)


def _xs_ls_yield(cwide: pd.DataFrame, q: float = 0.3) -> pd.Series:
    """SIGNAL companion to _xs_ls: the expected L/S carry yield for the
    next holding period, indexed at the REALIZATION month (same index
    as the returns series so they pair-align for 5.5 PBB k-sweep).

    For positions set at month mth realized at month nxt:
      signal[nxt] = mean(carry[hi-q at mth]) - mean(carry[lo-q at mth])
    This is the sleeve's commitment for that holding period — the
    EXPECTED RETURN under the carry hypothesis.
    """
    allm = sorted(cwide.index)
    rows: list[tuple[pd.Timestamp, float]] = []
    for i in range(len(allm) - 1):
        mth, nxt = allm[i], allm[i + 1]
        c = cwide.loc[mth].dropna()
        if len(c) < 4:
            continue
        hi = c[c >= c.quantile(1 - q)].index
        lo = c[c <= c.quantile(q)].index
        if len(hi) == 0 or len(lo) == 0:
            continue
        signal = float(c.reindex(hi).mean() - c.reindex(lo).mean())
        rows.append((nxt, signal))
    return pd.Series(dict(rows)).sort_index()


def build_carry_signal_panel(
    target_annual_vol: float = DEFAULT_TARGET_ANNUAL_VOL,
    include_rates: bool = True,
    include_rates_xc: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """⚠ INCOMPLETE — NOT VALID FOR CA FILTER CALIBRATION YET ⚠

    Senior post-audit 2026-06-01 found two abstraction defects:
      1. Returns are vol_target-scaled to a 10% target. Signal is
         ALSO vol_target-scaled here for "matching scales", which
         INFLATES the signal magnitude (divides by ~0.15 std → mean
         pushes from ~1-2% raw to ~9.6% scaled). This breaks the
         CA filter's |ER| > k × tcost comparison which assumes the
         signal IS the per-period expected return in return units.
      2. The aggregate (signal, returns) pair treats the 44-contract
         risk-parity sleeve as if it were a single-asset long-only —
         losing the per-contract trade granularity that CA gating
         actually operates on.

    Use cases that remain VALID for this function:
      - Visualizing the relationship between aggregate carry yield
        gap and realized sleeve returns over time
      - Computing cross-leg correlations of the yield-gap signal
      - DSR / Sharpe analysis of the SIGNAL itself (treating it as
        a hypothetical forecast)

    Use cases that are NOT yet valid:
      - PBB k-sweep for CA filter calibration → produces misleading
        NO_EVIDENCE due to the abstraction mismatch
      - Promoting cross_asset_carry.yaml ca_filter_k_method to
        pbb_sweep_calibrated based on this function

    The fix requires exposing per-contract (signal_panel,
    returns_panel, position_panel) and generalizing
    apply_ca_filter_to_returns to operate at the per-contract level.
    Tracked in [[project-multi-asset-ca-filter-gap-2026-06-01]].
    """
    from engine.validation.crossasset_carry import (
        build_fx_carry, build_rates_carry, build_rates_xc_carry,
    )
    from engine.validation.commodity_carry import (
        build_carry_and_returns as commodity_cr,
    )

    cmd_cw, cmd_rw = commodity_cr()
    fx_cw, fx_rw, _ = build_fx_carry()
    legs_signal: dict[str, pd.Series] = {
        "cmdty": _xs_ls_yield(cmd_cw, q=0.3),
        "fx":    _xs_ls_yield(fx_cw, q=0.4),
    }
    legs_returns: dict[str, pd.Series] = {
        "cmdty": build_commodity_carry_ls_returns_only(cmd_cw, cmd_rw),
        "fx":    build_commodity_carry_ls_returns_only(fx_cw, fx_rw, q=0.4),
    }
    if include_rates:
        rt_cw, rt_rw, rt_ls = build_rates_carry()
        legs_signal["rates_us"]  = _xs_ls_yield(rt_cw, q=0.5)
        legs_returns["rates_us"] = rt_ls
    if include_rates_xc:
        xc_cw, xc_rw, xc_ls = build_rates_xc_carry()
        legs_signal["rates_xc"]  = _xs_ls_yield(xc_cw, q=0.3)
        legs_returns["rates_xc"] = xc_ls

    # Risk-parity combine BOTH signal and returns using the same
    # inverse-vol weights — keeps signal scale consistent with realized
    # return scale so |expected_return| > k×tcost gate is meaningful.
    signal_combined  = risk_parity_combine(legs_signal)
    returns_combined = risk_parity_combine(legs_returns)
    signal_vt  = vol_target(signal_combined,  target_annual_vol)
    returns_vt = vol_target(returns_combined, target_annual_vol)

    # Align on common index
    aligned = pd.concat([signal_vt.rename("signal"),
                          returns_vt.rename("returns")],
                         axis=1).dropna()
    return aligned["signal"], aligned["returns"]


def build_commodity_carry_ls_returns_only(cw, rw, q: float = 0.3) -> pd.Series:
    """Inline copy of _xs_ls from engine.validation.crossasset_carry so
    we can re-use it across legs without re-importing. Same semantics."""
    allm = sorted(set(cw.index) | set(rw.index))
    rows: list[tuple[pd.Timestamp, float]] = []
    for i in range(len(allm) - 1):
        mth, nxt = allm[i], allm[i + 1]
        if mth not in cw.index or nxt not in rw.index:
            continue
        c = cw.loc[mth].dropna()
        if len(c) < 4:
            continue
        hi = c[c >= c.quantile(1 - q)].index
        lo = c[c <= c.quantile(q)].index
        nr = rw.loc[nxt]
        rl = nr.reindex(hi).dropna()
        rs = nr.reindex(lo).dropna()
        if len(rl) < 1 or len(rs) < 1:
            continue
        rows.append((nxt, float(rl.mean() - rs.mean())))
    return pd.Series(dict(rows)).sort_index()


def _per_contract_positions(
    cwide: pd.DataFrame, q: float,
) -> pd.DataFrame:
    """For each month, assign +1 to top-q carry contracts, -1 to
    bottom-q, 0 to the middle band. Output rows = month, cols = symbol,
    values ∈ {-1, 0, +1}. Position at month mth applies to returns
    realized in month mth+1 (held over the period)."""
    months = sorted(cwide.index)
    out = pd.DataFrame(0.0, index=months, columns=cwide.columns)
    for mth in months:
        c = cwide.loc[mth].dropna()
        if len(c) < 4:
            continue
        hi = c[c >= c.quantile(1 - q)].index
        lo = c[c <= c.quantile(q)].index
        out.loc[mth, hi] =  1.0
        out.loc[mth, lo] = -1.0
    return out


def build_carry_contract_panels(
    include_rates: bool = True,
    include_rates_xc: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phase: multi-asset CA filter rework.

    Expose per-contract (signal, returns, target_position) panels at
    monthly cadence — the right abstraction for CA filter gating on a
    multi-asset risk-parity sleeve (single-asset abstraction in
    build_carry_signal_panel was the defect caught in 2026-06-01
    post-audit, see [[project-multi-asset-ca-filter-gap-2026-06-01]]).

    Returns
    -------
    (signal_panel, returns_panel, position_panel) : tuple of DataFrames
      All indexed by monthly timestamps. Columns are namespaced
      "{leg}:{sym}" so different legs' symbol collisions don't merge.
        signal_panel:   raw carry yield per contract (return-equiv units)
        returns_panel:  realized next-period return per contract
        position_panel: intended position ∈ {-1, 0, +1} at month-end
                         (top-q long, bottom-q short, mid 0); applies to
                         the FOLLOWING month's returns_panel row.

    Notes
    -----
    - Per-leg q matches deployed sleeve (cmdty 0.3 / fx 0.4 /
      rates_us 0.5 / rates_xc 0.3). Position is assigned WITHIN the
      leg's universe, not pooled.
    - NO vol_target applied — caller decides scaling (CA filter wants
      return-unit signals; vol-target inflates the scale wrongly).
    """
    from engine.validation.crossasset_carry import (
        build_fx_carry, build_rates_carry, build_rates_xc_carry,
    )
    from engine.validation.commodity_carry import (
        build_carry_and_returns as commodity_cr,
    )

    legs: list[tuple[str, pd.DataFrame, pd.DataFrame, float]] = []
    cmd_cw, cmd_rw = commodity_cr()
    legs.append(("cmdty", cmd_cw, cmd_rw, 0.3))
    fx_cw, fx_rw, _ = build_fx_carry()
    legs.append(("fx", fx_cw, fx_rw, 0.4))
    if include_rates:
        rt_cw, rt_rw, _ = build_rates_carry()
        legs.append(("rates_us", rt_cw, rt_rw, 0.5))
    if include_rates_xc:
        xc_cw, xc_rw, _ = build_rates_xc_carry()
        legs.append(("rates_xc", xc_cw, xc_rw, 0.3))

    signal_parts: list[pd.DataFrame] = []
    returns_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    for leg_name, cw, rw, q in legs:
        pos = _per_contract_positions(cw, q)
        # Namespace columns to avoid collisions between legs
        sig = cw.rename(columns=lambda s: f"{leg_name}:{s}")
        ret = rw.rename(columns=lambda s: f"{leg_name}:{s}")
        pos = pos.rename(columns=lambda s: f"{leg_name}:{s}")
        signal_parts.append(sig)
        returns_parts.append(ret)
        position_parts.append(pos)

    # Outer concat — contracts present in only some legs are NaN
    # elsewhere; downstream caller skips NaN cells per the standard
    # CA filter pattern.
    signal_panel   = pd.concat(signal_parts,   axis=1).sort_index()
    returns_panel  = pd.concat(returns_parts,  axis=1).sort_index()
    position_panel = pd.concat(position_parts, axis=1).sort_index()

    return signal_panel, returns_panel, position_panel


def _daily_xs_ls(daily_ret_wide: pd.DataFrame, cwide_monthly: pd.DataFrame, q: float = 0.3) -> pd.Series:
    """Daily mark of the monthly-rebalanced cross-sectional carry L/S.

    Positions are set from the PRIOR month-end carry rank (long top-q / short bottom-q,
    matching the validated monthly _xs_ls timing) and held through the next month, marked
    DAILY on front-contract returns (daily equal-weight). Aggregating this to monthly is
    CLOSE to — not identical to — the monthly buy-and-hold _xs_ls; the small gap is the
    daily-rebalance vs monthly-hold convention (tested via correlation, not equality)."""
    months = sorted(cwide_monthly.index)
    d = daily_ret_wide.sort_index()
    pieces: list[pd.Series] = []

    def _ls_for(positions_month, lo_hi_window: pd.DataFrame) -> pd.Series | None:
        c = cwide_monthly.loc[positions_month].dropna()
        if len(c) < 4 or lo_hi_window.empty:
            return None
        hi = c[c >= c.quantile(1 - q)].index
        lo = c[c <= c.quantile(q)].index
        long_r = lo_hi_window.reindex(columns=hi).mean(axis=1)
        short_r = lo_hi_window.reindex(columns=lo).mean(axis=1)
        return (long_r - short_r).dropna()

    for i in range(len(months) - 1):
        mth, nxt = months[i], months[i + 1]
        win = d.loc[(d.index > mth) & (d.index <= nxt)]   # the days within holding month `nxt`
        seg = _ls_for(mth, win)
        if seg is not None:
            pieces.append(seg)
    # current (partial) month: hold the last complete month's positions so "today" marks too
    if months:
        last = months[-1]
        seg = _ls_for(last, d.loc[d.index > last])
        if seg is not None:
            pieces.append(seg)

    if not pieces:
        return pd.Series(dtype=float, name="carry_daily_ls")
    return pd.concat(pieces).sort_index().rename("carry_daily_ls")


def build_carry_daily_returns(target_annual_vol: float | None = None,
                              include_rates: bool = True,
                              include_rates_xc: bool = True) -> pd.Series:
    """The cross-asset carry sleeve's DAILY return series — what the live daily book marks.

    4-leg validated construction (commodity + FX + US-rates + G10-rates-XC per
    spec 77 §9 + §10 amendments 2026-05-28). Each leg marked daily from front-
    contract returns on the monthly-rebalanced positions, then inverse-vol
    combined. Optionally vol-targeted (252 trading days). Gross of cost by
    default; the monthly-consistency test guards that this aggregates back to the
    validated monthly series. `include_rates=False` reproduces the 2-leg
    pre-amendment path; `include_rates_xc=False` reproduces the §9 3-leg path."""
    import pandas as _pd
    from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
    from engine.validation.crossasset_carry import (
        FX, RATES, RATES_XC, _carry_and_returns,
        _RT_CONTR, _RT_PX, _RT_XC_CONTR, _RT_XC_PX, fetch_fx_futures,
    )
    cw_c, rd_c = commodity_cr(daily=True)
    ls_c = _daily_xs_ls(rd_c, cw_c, q=0.3)
    c_fx, p_fx = fetch_fx_futures()
    cw_f, rd_f = _carry_and_returns(c_fx, p_fx, FX, daily=True)
    ls_f = _daily_xs_ls(rd_f, cw_f, q=0.4)
    legs = {"cmdty": ls_c, "fx": ls_f}
    if include_rates:
        c_rt = _pd.read_parquet(_RT_CONTR)
        p_rt = _pd.read_parquet(_RT_PX)
        cw_r, rd_r = _carry_and_returns(c_rt, p_rt, RATES, daily=True)
        legs["rates_us"] = _daily_xs_ls(rd_r, cw_r, q=0.5)
    if include_rates_xc:
        c_xc = _pd.read_parquet(_RT_XC_CONTR)
        p_xc = _pd.read_parquet(_RT_XC_PX)
        cw_x, rd_x = _carry_and_returns(c_xc, p_xc, RATES_XC, daily=True)
        legs["rates_xc"] = _daily_xs_ls(rd_x, cw_x, q=0.3)
    daily = risk_parity_combine(legs).rename("carry_daily")
    if target_annual_vol:
        daily = vol_target(daily, target_annual_vol, periods_per_year=252)
    return daily


def sleeve_stats(series: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Quick descriptive stats for the sleeve return series (for surfacing/verification,
    not a re-validation — the GREEN verdict is spec-locked)."""
    s = series.dropna()
    if s.empty or s.std() == 0:
        return {"n": int(s.size), "ann_vol": 0.0, "ann_ret": 0.0, "sharpe": float("nan")}
    ann_vol = float(s.std() * np.sqrt(periods_per_year))
    ann_ret = float(s.mean() * periods_per_year)
    return {"n": int(s.size), "ann_vol": round(ann_vol, 4),
            "ann_ret": round(ann_ret, 4), "sharpe": round(ann_ret / ann_vol, 3)}
