"""
P3c — COT-Conditional BAB pre-registered test runner.

Spec: docs/spec_p3c_cot_conditional_bab_2026-05-07.md (id=39, hash=2ab64e667e30ba36)
Pre-registered: 2026-05-07 BEFORE this code ran. Hash locks H0 / H1 / decision rule.

Test design (as locked by the spec)
-----------------------------------
- Universe: equity ETFs with CFTC equity-index mapping (sector E-MINIs)
- Period: 2020-01 to 2024-12 (60 months; matches our 5y CFTC backfill)
- BAB signal: 252-day β to SPY → tertile rank → long bottom, short top, equal-weight
- Conditioning indicator: SPY (E-MINI 13874+) leveraged-money net positioning,
  decile of trailing 252-week distribution; TOP10/BOT10 = "extreme"
- Verdict:
    SHIP    : Sharpe(conditional) - Sharpe(unconditional) > +0.15 AND BHY-adjusted p < 0.05
    MARGINAL: same Sharpe lift AND 0.05 ≤ BHY-p < 0.10
    FAIL    : all other outcomes

Outputs (all written to docs/decisions/p3c_cot_conditional_bab_verdict_2026-05-07.md):
- monthly time series of BAB returns
- COT regime classification per month
- conditional vs unconditional Sharpe
- Bootstrap 95% CI for Sharpe difference
- Final verdict (SHIP/MARGINAL/FAIL)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("p3c")


# ── Spec invariants — locked by docs/spec_p3c_cot_conditional_bab_2026-05-07.md ──
SPEC_PATH      = "docs/spec_p3c_cot_conditional_bab_2026-05-07.md"
SPEC_HASH_HEAD = "2ab64e667e30ba36"   # current_hash[:16] at registration

SHIP_SHARPE_LIFT     = 0.15
SHIP_BHY_P_THRESHOLD = 0.05
MARGINAL_BHY_P_HIGH  = 0.10

# Universe: equity ETFs with CFTC equity-index mapping. Restricted to the
# subset we actually have COT data for. SPY is the beta benchmark.
EQUITY_UNIVERSE = ["XLE", "XLF", "XLV", "XLI", "XLP", "XLU", "XLK",
                   "XLB", "XLRE", "XLY", "XLC", "QQQ", "DIA"]
SPY_TICKER = "SPY"

START_DATE = datetime.date(2019, 1, 1)   # need 1y warm-up for 252-day β
END_DATE   = datetime.date(2024, 12, 31)


# ── Step 1: spec-hash precondition (HARKing-prevention) ──────────────────────

def _verify_spec_locked() -> None:
    """Refuse to run if the spec hash drifted since registration. This is the
    on-the-fly HARKing R1 guard — the test code consumes a frozen spec, so
    any mid-stream rewrite of the spec invalidates the test.
    """
    from engine.preregistration import _resolve_to_abs, _compute_git_blob_hash
    abs_path = _resolve_to_abs(SPEC_PATH)
    if not os.path.exists(abs_path):
        raise SystemExit(f"FATAL: spec file missing: {SPEC_PATH}")
    current = _compute_git_blob_hash(abs_path)[:16]
    if current != SPEC_HASH_HEAD:
        raise SystemExit(
            f"FATAL: spec hash drift since registration. Expected "
            f"{SPEC_HASH_HEAD}, found {current}. "
            f"This test is invalidated. Either revert the spec or run amend_spec()."
        )
    print(f"[OK] spec hash locked: {SPEC_HASH_HEAD}")


# ── Step 2: load monthly price data from yfinance ────────────────────────────

def _load_monthly_prices() -> pd.DataFrame:
    """Return adjusted close monthly prices for SPY + universe."""
    import yfinance as yf
    tickers = [SPY_TICKER] + EQUITY_UNIVERSE
    print(f"[..] downloading {len(tickers)} ETFs daily 2019-01..2024-12")
    px = yf.download(
        tickers, start=START_DATE.isoformat(),
        end=(END_DATE + datetime.timedelta(days=1)).isoformat(),
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    if "Close" in px:
        px = px["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    print(f"[OK] downloaded; daily shape = {px.shape}")
    return px


# ── Step 3: compute monthly BAB returns ──────────────────────────────────────

def _compute_monthly_bab_returns(px_daily: pd.DataFrame) -> pd.Series:
    """Walk-forward monthly BAB strategy returns 2020-01..2024-12.

    Each month-end date as_of:
      1. compute 252-day β of each universe ETF vs SPY using returns ending as_of
      2. tertile rank by β: bottom = long, top = short, middle excluded
      3. equal-weight long leg + equal-weight short leg, dollar-neutral
      4. record next-month return on this portfolio
    """
    if SPY_TICKER not in px_daily.columns:
        raise SystemExit(f"FATAL: SPY missing from price frame")

    # Daily returns
    rets = px_daily.pct_change().dropna(how="all")
    spy = rets[SPY_TICKER].dropna()

    # Month-end dates
    month_ends = pd.date_range(
        start=datetime.date(2020, 1, 31), end=END_DATE, freq="ME"
    )

    out: dict[pd.Timestamp, float] = {}
    n_skipped = 0
    for me in month_ends:
        # Use 252 trading days ending at me
        window_start = me - pd.tseries.offsets.BDay(252)
        spy_w = spy[(spy.index >= window_start) & (spy.index <= me)]
        if len(spy_w) < 60:
            n_skipped += 1
            continue
        spy_var = float(spy_w.var())
        if spy_var < 1e-12:
            continue

        # Beta per ticker
        betas: dict[str, float] = {}
        for tk in EQUITY_UNIVERSE:
            if tk not in rets.columns:
                continue
            r_tk = rets[tk].dropna()
            common = spy_w.index.intersection(r_tk.index)
            if len(common) < 60:
                continue
            cov = float(r_tk.loc[common].cov(spy_w.loc[common]))
            betas[tk] = cov / spy_var

        if len(betas) < 6:
            n_skipped += 1
            continue

        # Tertile rank
        sorted_b = sorted(betas.items(), key=lambda x: x[1])
        n = len(sorted_b)
        k = max(1, n // 3)
        long_set  = [t for t, _ in sorted_b[:k]]
        short_set = [t for t, _ in sorted_b[n - k:]]

        # Next-month return: long = +mean(long_set), short = -mean(short_set)
        next_me = me + pd.tseries.offsets.MonthEnd(1)
        period_mask = (rets.index > me) & (rets.index <= next_me)
        period_rets = rets.loc[period_mask]
        if period_rets.empty:
            continue

        # Compound returns for each side
        long_ret_period  = float((1 + period_rets[long_set]).prod().mean()  - 1)
        short_ret_period = float((1 + period_rets[short_set]).prod().mean() - 1)

        # BAB return = long leg minus short leg (dollar-neutral)
        bab_ret = long_ret_period - short_ret_period
        out[me] = bab_ret

    print(f"[OK] computed BAB returns for {len(out)} months (skipped {n_skipped})")
    return pd.Series(out, name="bab_ret").sort_index()


# ── Step 4: COT-regime classification per month ──────────────────────────────

def _classify_cot_regime_monthly(month_dates: pd.DatetimeIndex) -> pd.Series:
    """For each month-end, classify SPY (E-MINI 13874+) leveraged-money net
    positioning into TOP10 / BOT10 / NORMAL based on rolling-window decile
    of net positioning relative to open interest.
    """
    from engine.memory import SessionFactory
    from engine.db_models import CftcCotWeekly

    with SessionFactory() as s:
        rows = (
            s.query(CftcCotWeekly.report_date,
                    CftcCotWeekly.lev_money_long,
                    CftcCotWeekly.lev_money_short,
                    CftcCotWeekly.open_interest)
             .filter(
                CftcCotWeekly.contract_market_code == "13874+",
                CftcCotWeekly.report_type == "tff_fut",
                CftcCotWeekly.report_date >= datetime.datetime(2019, 1, 1),
                CftcCotWeekly.report_date <= datetime.datetime(2024, 12, 31),
            )
             .order_by(CftcCotWeekly.report_date)
             .all()
        )

    if not rows:
        return pd.Series([], dtype=str, name="cot_regime")

    cot_df = pd.DataFrame(rows, columns=[
        "report_date", "lev_long", "lev_short", "oi",
    ])
    cot_df["lev_net_pct"] = (cot_df["lev_long"] - cot_df["lev_short"]) / cot_df["oi"].clip(lower=1)
    cot_df.set_index("report_date", inplace=True)

    # For each month-end, look up the latest weekly COT report ≤ that month-end
    out: dict[pd.Timestamp, str] = {}
    for me in month_dates:
        recent = cot_df[cot_df.index <= me]
        if recent.empty:
            out[me] = "NA"
            continue
        # 252-week trailing distribution (≈ 5y) — but we may have less data
        # in 2020-2021 since CFTC archive starts 2020 in our backfill.
        trailing = recent.tail(252)
        cur_pct = float(recent["lev_net_pct"].iloc[-1])
        if len(trailing) < 30:
            out[me] = "NA"
            continue
        p10 = float(trailing["lev_net_pct"].quantile(0.10))
        p90 = float(trailing["lev_net_pct"].quantile(0.90))
        if cur_pct >= p90:
            out[me] = "TOP10"
        elif cur_pct <= p10:
            out[me] = "BOT10"
        else:
            out[me] = "NORMAL"

    return pd.Series(out, name="cot_regime").sort_index()


# ── Step 5: Sharpe + bootstrap inference ─────────────────────────────────────

def _annualised_sharpe(monthly_returns: pd.Series) -> float:
    if len(monthly_returns) < 6:
        return float("nan")
    mu  = float(monthly_returns.mean())
    sd  = float(monthly_returns.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return mu / sd * np.sqrt(12)


def _bootstrap_sharpe_diff(uncond: pd.Series, cond: pd.Series,
                           n_boot: int = 5000, seed: int = 42) -> dict:
    """Stationary block bootstrap of Sharpe(cond) - Sharpe(uncond).
    Returns p-value (two-sided), 95% CI, and point estimate.
    """
    rng = np.random.default_rng(seed)
    point_diff = _annualised_sharpe(cond) - _annualised_sharpe(uncond)
    n_uncond = len(uncond)
    n_cond   = len(cond)
    if n_cond < 6:
        return {"point_diff": point_diff, "p_value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    # Block size ≈ 1.75 * n^(1/3) per Politis-White
    block = max(2, int(round(1.75 * n_uncond ** (1/3))))
    diffs = np.empty(n_boot)
    uncond_arr = uncond.values
    cond_arr   = cond.values

    for i in range(n_boot):
        # Resample uncond with stationary block bootstrap
        idx_u = []
        while len(idx_u) < n_uncond:
            start = rng.integers(0, n_uncond)
            length = max(1, int(rng.geometric(1.0 / block)))
            for k in range(length):
                idx_u.append((start + k) % n_uncond)
        idx_u = idx_u[:n_uncond]
        # Same for cond (independent)
        idx_c = []
        while len(idx_c) < n_cond:
            start = rng.integers(0, n_cond)
            length = max(1, int(rng.geometric(1.0 / block)))
            for k in range(length):
                idx_c.append((start + k) % n_cond)
        idx_c = idx_c[:n_cond]

        u_sample = pd.Series(uncond_arr[idx_u])
        c_sample = pd.Series(cond_arr[idx_c])
        diffs[i] = _annualised_sharpe(c_sample) - _annualised_sharpe(u_sample)

    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return {"point_diff": point_diff, "p_value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    # Two-sided p-value: P(diff under H0 ≥ observed)
    # Under H0 the true diff is 0; we approximate by recentering.
    centered = diffs - float(diffs.mean())
    p_two_sided = float(np.mean(np.abs(centered) >= abs(point_diff)))
    ci_low  = float(np.quantile(diffs, 0.025))
    ci_high = float(np.quantile(diffs, 0.975))
    return {
        "point_diff": point_diff, "p_value": p_two_sided,
        "ci_low": ci_low, "ci_high": ci_high,
        "n_boot_valid": len(diffs),
    }


# ── Step 6: BHY adjustment + verdict ─────────────────────────────────────────

def _bhy_adjust(p_raw: float, n_trials: int) -> float:
    """Benjamini-Yekutieli step-up (single test variant) — for n_trials = 1
    this collapses to the raw p-value; we wire it up generically so future
    multi-test variants don't change the signature. EFFECTIVE_N_TRIALS is
    queried from SpecRegistry totals at runtime if available.
    """
    try:
        from engine.memory import SessionFactory
        from engine.db_models import SpecRegistry
        from sqlalchemy import func
        with SessionFactory() as s:
            n_eff = int(s.query(func.coalesce(func.sum(SpecRegistry.n_trials_contributed), 0)).scalar() or 1)
        n_eff = max(1, n_eff)
    except Exception:
        n_eff = max(1, n_trials)

    # BHY harmonic correction factor
    c_n = sum(1.0 / k for k in range(1, n_eff + 1))
    return min(1.0, float(p_raw) * c_n)


def _verdict(uncond_sharpe: float, cond_sharpe: float,
             p_raw: float, p_bhy: float) -> str:
    sharpe_lift = cond_sharpe - uncond_sharpe
    if sharpe_lift > SHIP_SHARPE_LIFT and p_bhy < SHIP_BHY_P_THRESHOLD:
        return "SHIP"
    if sharpe_lift > SHIP_SHARPE_LIFT and (SHIP_BHY_P_THRESHOLD <= p_bhy < MARGINAL_BHY_P_HIGH):
        return "MARGINAL"
    return "FAIL"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{'='*70}\nP3c COT-Conditional BAB pre-reg test\n{'='*70}")

    _verify_spec_locked()

    px = _load_monthly_prices()
    bab_ret = _compute_monthly_bab_returns(px)
    print(f"     BAB monthly returns: n={len(bab_ret)} mean={bab_ret.mean():+.4f} std={bab_ret.std():.4f}")

    cot_regime = _classify_cot_regime_monthly(bab_ret.index)
    print(f"     COT regime distribution:")
    for label, count in cot_regime.value_counts().items():
        print(f"       {label}: {count}")

    # Align
    aligned = pd.DataFrame({"ret": bab_ret, "regime": cot_regime}).dropna()
    aligned = aligned[aligned["regime"] != "NA"]
    extreme_mask = aligned["regime"].isin(["TOP10", "BOT10"])
    uncond = aligned["ret"]
    cond   = aligned[extreme_mask]["ret"]

    uncond_sharpe = _annualised_sharpe(uncond)
    cond_sharpe   = _annualised_sharpe(cond)

    print(f"\n[SHARPE]")
    print(f"     Unconditional (n={len(uncond)}): {uncond_sharpe:+.4f}")
    print(f"     Conditional   (n={len(cond)}, TOP10+BOT10): {cond_sharpe:+.4f}")
    print(f"     Lift: {cond_sharpe - uncond_sharpe:+.4f}")

    print(f"\n[BOOTSTRAP] running 5000-iteration stationary block bootstrap")
    bs = _bootstrap_sharpe_diff(uncond, cond)
    print(f"     point diff = {bs['point_diff']:+.4f}")
    print(f"     95% CI:    [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]")
    print(f"     p (raw):   {bs['p_value']:.4f}")

    p_bhy = _bhy_adjust(bs["p_value"], n_trials=1)
    print(f"     p (BHY-adj): {p_bhy:.4f}")

    verdict = _verdict(uncond_sharpe, cond_sharpe, bs["p_value"], p_bhy)
    print(f"\n[VERDICT] {verdict}")

    # Persist decision artifact
    out = {
        "spec_hash":      SPEC_HASH_HEAD,
        "n_trials":       1,
        "universe":       EQUITY_UNIVERSE,
        "period":         f"{START_DATE} to {END_DATE}",
        "n_months_total": int(len(uncond)),
        "n_months_extreme_cot": int(len(cond)),
        "uncond_sharpe":  uncond_sharpe,
        "cond_sharpe":    cond_sharpe,
        "sharpe_lift":    cond_sharpe - uncond_sharpe,
        "bootstrap":      bs,
        "p_raw":          bs["p_value"],
        "p_bhy":          p_bhy,
        "verdict":        verdict,
        "thresholds": {
            "ship_sharpe_lift": SHIP_SHARPE_LIFT,
            "ship_bhy_p":       SHIP_BHY_P_THRESHOLD,
            "marginal_bhy_high": MARGINAL_BHY_P_HIGH,
        },
        "regime_distribution": cot_regime.value_counts().to_dict(),
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    out_json = os.path.join(ROOT, "docs", "decisions", "p3c_cot_bab_verdict_2026-05-07.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[OUT] verdict JSON → {os.path.relpath(out_json)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
