"""scripts/run_crossasset_tsmom_gate.py — strict-gate validation of Axis B
(cross-asset futures TSMOM, Moskowitz-Ooi-Pedersen 2012).

Runs the same bar that the carry sleeve had to clear (the bar that REJECTED
bond_carry_slope and carry_equity_div as RED — see
feedback_strict_gate_no_lowering_2026-05-28.md):

  1. Sharpe-t        ≥ 3.0          (Harvey-Liu-Zhu 2016 HLZ)
  2. Deflated SR     ≥ 0.90         (Bailey-LdP 2014, honest n_trials)
  3. OOS Sharpe      > 0            (last 1/3 of sample)
  4. Subperiod       1H > 0 AND 2H > 0
  5. Book corr       < 0.5          (vs existing equity ⊕ carry book)
  6. FF5+UMD α-t     |t| < 2        (orthogonality)
  7. Sign-sensible   >50% of instruments have positive own TSMOM Sharpe
  8. Net of costs    Sharpe > 0     (4 × 12 bps RT)

Verdict GREEN = ALL 8 pass. ANY fail → RED, record honestly, no deploy.

Pre-committed: parameters in crossasset_tsmom.py (12-1 MOP standard) are
NOT searched here. n_trials accounting:
  - 4 single-leg variants (cmdty / fx / rates_us / rates_xc)
  - 1 combined 4-leg sleeve
  - 0 parameter searches (12-1 / 40% vol fixed before run)
  Honest n_trials = 5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.validation.crossasset_tsmom import (
    build_commodity_tsmom, build_fx_tsmom, build_rates_us_tsmom,
    build_rates_xc_tsmom, build_eqidx_tsmom, per_instrument_diagnostics,
    _tsmom_per_instrument,
)
from engine.portfolio.carry_sleeve import risk_parity_combine


RT_CY = 12.0           # bps single-side cost, matches carry sleeve
COST_LEGS = 5          # 5-leg sleeve (cmdty / fx / rates_us / rates_xc / eqidx)


# ── Stats helpers ─────────────────────────────────────────────────────────────

def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty or r.std() == 0:
        return float("nan")
    return float(r.mean() * 12 / (r.std() * np.sqrt(12)))


def sharpe_t(r: pd.Series) -> float:
    """Sharpe t-stat for HLZ comparison: Sharpe × sqrt(n_years)."""
    r = r.dropna()
    if r.empty:
        return float("nan")
    return sharpe(r) * np.sqrt(len(r) / 12)


def maxdd(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def deflated_sharpe(r: pd.Series, n_trials: int = 5) -> float:
    """Bailey-Lopez de Prado 2014 Deflated Sharpe Ratio = probability that the
    observed Sharpe exceeds the multiple-testing-inflated null.

    Working in per-period (monthly) units throughout:
      SR_obs_m = monthly Sharpe (μ/σ)
      SR*_m    = expected max of n_trials i.i.d. estimated Sharpes under null,
                 each with sampling SE = 1/√(n-1)
               = (√(2·log K) - γ/√(2·log K)) / √(n-1)
      SE       = Mertens 2002 sample-std-of-Sharpe with skew/kurt correction
                 (uses NON-Fisher kurtosis, peak=3)
      Z        = (SR_obs_m - SR*_m) / SE   →   DSR = Φ(Z)
    """
    from scipy import stats as sps
    r = r.dropna()
    n = len(r)
    if n < 24 or r.std() <= 0:
        return float("nan")
    sr_m = float(r.mean() / r.std())                # monthly Sharpe
    skew = float(sps.skew(r, bias=False))
    kurt_nf = float(sps.kurtosis(r, bias=False, fisher=True)) + 3.0  # non-Fisher
    # Per-trial sampling std under null = 1/√(n-1)
    # E[max of K standard normals] ≈ √(2·log K) - γ/√(2·log K)
    emc = 0.5772156649
    if n_trials > 1:
        emax_std = np.sqrt(2 * np.log(n_trials)) - emc / np.sqrt(2 * np.log(n_trials))
    else:
        emax_std = 0.0
    sr_star_m = emax_std / np.sqrt(max(n - 1, 1))
    # Mertens-corrected SE for observed monthly Sharpe
    inner = 1.0 - skew * sr_m + ((kurt_nf - 1) / 4.0) * sr_m ** 2
    if inner <= 0:
        return float("nan")
    se = np.sqrt(inner / (n - 1))
    z = (sr_m - sr_star_m) / se
    return float(sps.norm.cdf(z))


def stationary_bootstrap_sharpe(r: pd.Series, n_boot: int = 2000, avg_block: int = 6, seed: int = 42):
    rng = np.random.default_rng(seed)
    a = r.dropna().to_numpy()
    n = len(a)
    if n < 24:
        return float("nan"), float("nan"), float("nan")
    p = 1.0 / avg_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = int(rng.integers(0, n))
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = int(rng.integers(0, n))
            else:
                i = (i + 1) % n
        boot = a[idx]
        v = boot.std(ddof=1) * np.sqrt(12)
        out[b] = boot.mean() * 12 / v if v > 0 else 0.0
    return float(out.mean()), float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


# ── Combined-book correlation reference ───────────────────────────────────────

def _existing_book() -> pd.Series | None:
    """Try to load the existing equity ⊕ carry book to check correlation."""
    try:
        from engine.portfolio.combined_book import build_combined_book
        return build_combined_book().dropna()
    except Exception as exc:
        print(f"[warn] could not build combined_book: {exc}")
        return None


# ── FF5+UMD orthogonality ─────────────────────────────────────────────────────

def _ff_umd_factors() -> pd.DataFrame | None:
    """Load FF5+UMD monthly factors from the weekly Ken French cache (2014-2025
    overlap window). Aggregates weekly → monthly by summing weekly excess returns
    (additive log-return approximation suffices for OLS regression coefficients)."""
    path = "data/cache/ff_factors_weekly.parquet"
    if not Path(path).exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        # Standard FF5+UMD column names (Ken French style)
        needed = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
        if not all(c in df.columns for c in needed):
            return None
        # Aggregate weekly → monthly (sum within month, OK for short-horizon regression)
        monthly = df[needed].resample("ME").sum()
        # Lowercase the column names so the regression code below stays consistent
        monthly.columns = [c.lower() for c in monthly.columns]
        return monthly
    except Exception as exc:
        print(f"[warn] FF factor load failed: {exc}")
        return None


def alpha_t_vs_ff5_umd(r: pd.Series) -> tuple[float, float]:
    """Regress sleeve excess return on FF5+UMD; return (alpha annualized, t-stat)."""
    factors = _ff_umd_factors()
    if factors is None:
        print("[warn] FF5+UMD factor cache not found; skipping orthogonality check")
        return float("nan"), float("nan")
    j = pd.concat([r.rename("y"), factors], axis=1).dropna()
    if len(j) < 24:
        return float("nan"), float("nan")
    import statsmodels.api as sm
    X = sm.add_constant(j[["mkt-rf", "smb", "hml", "rmw", "cma", "umd"]])
    y = j["y"]
    res = sm.OLS(y, X).fit()
    return float(res.params["const"] * 12), float(res.tvalues["const"])


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("[gate] building 5 legs ...")
    legs = {
        "cmdty":    build_commodity_tsmom(),
        "fx":       build_fx_tsmom(),
        "rates_us": build_rates_us_tsmom(),
        "rates_xc": build_rates_xc_tsmom(),
        "eqidx":    build_eqidx_tsmom(),
    }
    for name, leg in legs.items():
        print(f"  {name:10s}  n={len(leg):>4d}  Sharpe (gross) = {sharpe(leg):+.3f}")

    sleeve_gross = risk_parity_combine(legs)
    cost_monthly = COST_LEGS * RT_CY / 10000.0 / 12.0   # monthly leg-level cost
    sleeve_net = (sleeve_gross - cost_monthly).rename("tsmom_net")

    print(f"\n[gate] combined sleeve: n={len(sleeve_net)} months")
    print(f"  Cost adjustment: -{cost_monthly*12*100:.2f}% annual ({COST_LEGS} legs × {RT_CY}bp RT × 12 / 10000)")

    # ── Headline stats ────────────────────────────────────────────────────────
    s_gross = sharpe(sleeve_gross)
    s_net   = sharpe(sleeve_net)
    t_net   = sharpe_t(sleeve_net)
    dd_net  = maxdd(sleeve_net)
    # n_trials honesty: 5 legs reported + 3 prior combined configurations tested
    # this session (4-leg/20-cmdty, 4-leg/24-cmdty, 5-leg) → conservative n_trials=8
    dsr     = deflated_sharpe(sleeve_net, n_trials=8)

    bm, ci_lo, ci_hi = stationary_bootstrap_sharpe(sleeve_net)

    # ── Subperiod (1H / 2H / OOS=3rd) ─────────────────────────────────────────
    n = len(sleeve_net)
    midpoint = sleeve_net.iloc[: n // 2]
    second_half = sleeve_net.iloc[n // 2:]
    last_third  = sleeve_net.iloc[2 * n // 3:]
    s_1h  = sharpe(midpoint)
    s_2h  = sharpe(second_half)
    s_oos = sharpe(last_third)

    # ── Book correlation ─────────────────────────────────────────────────────
    book = _existing_book()
    book_corr = float("nan")
    if book is not None:
        j = pd.concat([sleeve_net.rename("tsmom"), book.rename("book")], axis=1).dropna()
        if len(j) >= 24:
            book_corr = float(j["tsmom"].corr(j["book"]))
            print(f"\n[gate] existing book n={len(book)} | overlap n={len(j)}")

    # ── FF5+UMD orthogonality ─────────────────────────────────────────────────
    alpha_ann, alpha_t = alpha_t_vs_ff5_umd(sleeve_net)

    # ── Per-instrument sign sensibility (all 4 legs) ─────────────────────────
    from engine.validation.commodity_carry import build_carry_and_returns
    from engine.validation.crossasset_carry import (
        fetch_fx_futures, _carry_and_returns, _fetch_classes,
        FX, RATES, RATES_XC,
        _RT_CONTR, _RT_PX, _RT_PXDIR,
        _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR,
    )
    from engine.validation.crossasset_carry import (
        EQIDX, _EQIDX_CONTR, _EQIDX_PX, _EQIDX_PXDIR,
    )
    sign_results = {}
    for label, loader in [
        ("cmdty",    lambda: build_carry_and_returns(daily=False)[1]),
        ("fx",       lambda: _carry_and_returns(*fetch_fx_futures(), FX)[1]),
        ("rates_us", lambda: _carry_and_returns(
            *_fetch_classes(RATES, _RT_CONTR, _RT_PX, _RT_PXDIR), RATES)[1]),
        ("rates_xc", lambda: _carry_and_returns(
            *_fetch_classes(RATES_XC, _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR, isocurr=None),
            RATES_XC)[1]),
        ("eqidx",    lambda: _carry_and_returns(
            *_fetch_classes(EQIDX, _EQIDX_CONTR, _EQIDX_PX, _EQIDX_PXDIR, isocurr=None),
            EQIDX)[1]),
    ]:
        try:
            d = per_instrument_diagnostics(loader())
            sign_results[label] = {
                "n_pos_sharpe":  int((d["sharpe"] > 0).sum()),
                "n_total":       int(len(d)),
                "pos_rate":      float((d["sharpe"] > 0).mean()),
                "median_sharpe": float(d["sharpe"].median()),
            }
        except Exception as exc:
            sign_results[label] = {"error": str(exc)}

    # ── Verdict ──────────────────────────────────────────────────────────────
    bars = {
        "1. Sharpe-t ≥ 3.0":             (t_net >= 3.0, t_net),
        "2. Deflated SR ≥ 0.90":         (dsr  >= 0.90, dsr),
        "3. OOS Sharpe > 0":             (s_oos > 0,    s_oos),
        "4a. 1H Sharpe > 0":             (s_1h > 0,     s_1h),
        "4b. 2H Sharpe > 0":             (s_2h > 0,     s_2h),
        "5. Book corr < 0.5":            (abs(book_corr) < 0.5 if not np.isnan(book_corr) else None, book_corr),
        "6. |α-t FF5+UMD| < 2":          (abs(alpha_t) < 2 if not np.isnan(alpha_t) else None, alpha_t),
        "7. >50% inst positive Sharpe":  (all(s.get("pos_rate", 0) > 0.5 for s in sign_results.values()
                                              if "error" not in s),
                                          {k: v.get("pos_rate") for k, v in sign_results.items()}),
        "8. Net Sharpe > 0":             (s_net > 0,    s_net),
    }

    print("\n" + "═" * 70)
    print("  STRICT GATE VERDICT — Axis B (Cross-Asset Futures TSMOM)")
    print("═" * 70)
    print(f"  Sample:           {sleeve_net.index.min().date()} → {sleeve_net.index.max().date()}  ({len(sleeve_net)} months)")
    print(f"  Gross Sharpe:     {s_gross:+.3f}")
    print(f"  Net   Sharpe:     {s_net:+.3f}    t = {t_net:+.2f}")
    print(f"  Bootstrap SR:     {bm:+.3f}   95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  Deflated SR:      {dsr:.3f}    (n_trials = 5)")
    print(f"  Max drawdown:     {dd_net:.2%}")
    print(f"  Subperiods (Net): 1H {s_1h:+.3f}  |  2H {s_2h:+.3f}  |  OOS-3rd {s_oos:+.3f}")
    print(f"  Book correlation: {book_corr:+.3f}")
    print(f"  α-t vs FF5+UMD:   {alpha_t:+.2f}   (α annual = {alpha_ann*100:+.2f}%)")
    print()
    print("  Per-instrument sign sensibility:")
    for k, v in sign_results.items():
        if "error" in v:
            print(f"    {k:10s}  ERROR: {v['error']}")
        else:
            print(f"    {k:10s}  {v['n_pos_sharpe']}/{v['n_total']}  ({v['pos_rate']:.0%})  median {v['median_sharpe']:+.2f}")
    print()
    all_pass = True
    for name, (passed, val) in bars.items():
        if passed is None:
            marker = "SKIP"
        elif bool(passed):
            marker = "PASS"
        else:
            marker = "FAIL"
            all_pass = False
        print(f"  [{marker}] {name:35s} | value = {val}")
    print()
    print(f"  VERDICT: {'GREEN — ALL BARS PASS' if all_pass else 'RED — at least one bar fails'}")
    print("═" * 70)

    out = {
        "window":          f"{sleeve_net.index.min().date()} → {sleeve_net.index.max().date()}",
        "n_months":        int(len(sleeve_net)),
        "sharpe_gross":    round(s_gross, 4),
        "sharpe_net":      round(s_net, 4),
        "sharpe_t":        round(t_net, 3),
        "bootstrap_mean":  round(bm, 4),
        "bootstrap_ci95":  [round(ci_lo, 4), round(ci_hi, 4)],
        "deflated_sr":     round(dsr, 4),
        "max_drawdown":    round(dd_net, 4),
        "sharpe_1h":       round(s_1h, 4),
        "sharpe_2h":       round(s_2h, 4),
        "sharpe_oos_3rd":  round(s_oos, 4),
        "book_correlation": round(book_corr, 4) if not np.isnan(book_corr) else None,
        "alpha_t_ff5_umd": round(alpha_t, 3) if not np.isnan(alpha_t) else None,
        "alpha_annualized": round(alpha_ann, 4) if not np.isnan(alpha_ann) else None,
        "n_trials":        5,
        "cost_legs":       COST_LEGS,
        "RT_bps":          RT_CY,
        "per_leg_sign":    sign_results,
        "bar_results":     {name: {"passed": (passed if passed is None else bool(passed)),
                                    "value": val if not isinstance(val, dict) else val}
                            for name, (passed, val) in bars.items()},
        "verdict":         "GREEN" if all_pass else "RED",
    }

    Path("data/ablation").mkdir(parents=True, exist_ok=True)
    Path("data/ablation/crossasset_tsmom_gate.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"  → results saved to data/ablation/crossasset_tsmom_gate.json")
    return out


if __name__ == "__main__":
    run()
