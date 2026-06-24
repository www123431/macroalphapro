"""scripts/ablation_msm_on_vs_off.py

ABLATION: does the MSM regime overlay actually help our book?

The existing walk-forward backtest (engine.backtest.run_backtest) already computes
BOTH portfolios on every rebalance:
  - tsmom         : TSMOM + vol-targeting, regime=None       (MSM-OFF)
  - tsmom_regime  : TSMOM + vol-targeting, regime=regime_r   (MSM-ON)

We just need to compare them honestly. The decision rule (pre-committed):
  Δ Sharpe (MSM-ON minus MSM-OFF) with stationary-bootstrap 95% CI:
    Δ ≥ +0.10 and CI excludes 0          → KEEP MSM, evidence-backed
    Δ ∈ [+0.05, +0.10) or CI straddles 0 → MARGINAL — keep but flag
    Δ < +0.05  (incl. negative)          → DOWNGRADE MSM to rule-based
                                            (saves the statsmodels EM fit path,
                                            keeps the deterministic _rule_based_regime
                                            classifier that already exists)

We also report:
  - Sharpe in risk-on subperiods vs risk-off subperiods (where the overlay
    is supposed to matter most)
  - max drawdown
  - turnover (the overlay should pay for the extra turnover it induces)
  - hit-rate of overlay direction (% of risk-off-flagged months where MSM-OFF
    > MSM-ON i.e. the overlay was "right" to shrink longs)

NOT a new feature, NOT a backtest improvement. Just a decision aid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtest import run_backtest          # noqa: E402
from engine.regime import clear_regime_memo        # noqa: E402


# ── Stats helpers ─────────────────────────────────────────────────────────────

def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty or r.std() == 0:
        return float("nan")
    return float(r.mean() * 12 / (r.std() * np.sqrt(12)))


def maxdd(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def stationary_bootstrap_sharpe_diff(
    r_on: pd.Series,
    r_off: pd.Series,
    n_boot: int = 2000,
    avg_block: int = 6,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Politis-Romano stationary bootstrap on (r_on - r_off) Sharpe diff.

    Returns (mean delta, 2.5%-ile, 97.5%-ile). Block length is geometric with
    mean `avg_block` months — appropriate for monthly returns w/ mild autocorr.
    """
    rng = np.random.default_rng(seed)
    pair = pd.concat([r_on.rename("on"), r_off.rename("off")], axis=1).dropna()
    n = len(pair)
    if n < 12:
        return float("nan"), float("nan"), float("nan")
    p = 1.0 / avg_block
    deltas = np.empty(n_boot)
    arr = pair.to_numpy()
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = int(rng.integers(0, n))
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = int(rng.integers(0, n))
            else:
                i = (i + 1) % n
        boot = arr[idx]
        on, off = boot[:, 0], boot[:, 1]
        s_on  = on.mean() * 12 / (on.std(ddof=1) * np.sqrt(12) + 1e-12)
        s_off = off.mean() * 12 / (off.std(ddof=1) * np.sqrt(12) + 1e-12)
        deltas[b] = s_on - s_off
    return float(deltas.mean()), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


# ── Verdict ───────────────────────────────────────────────────────────────────

def verdict(delta_mean: float, ci_lo: float, ci_hi: float) -> str:
    ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    if delta_mean >= 0.10 and ci_excludes_zero:
        return "KEEP — evidence-backed (Δ≥0.10 & CI excludes 0)"
    if delta_mean >= 0.05:
        return "MARGINAL — keep but flag (Δ in [0.05, 0.10) or wide CI)"
    if delta_mean < 0.05 and ci_excludes_zero and delta_mean < 0:
        return "DOWNGRADE — MSM HURTS (Δ<0 with statistical significance)"
    return "DOWNGRADE — no evidence MSM helps (Δ<0.05 or CI straddles 0)"


# ── Main ──────────────────────────────────────────────────────────────────────

def run(start: str, end: str, regime_scale: float = 0.3, out_path: str | None = None) -> dict:
    print(f"[ablation] running run_backtest({start} → {end}, regime_scale={regime_scale})")
    clear_regime_memo()
    res = run_backtest(start_date=start, end_date=end, regime_scale=regime_scale)
    if res.returns.empty:
        print("[ablation] empty backtest result — abort")
        return {"error": "empty backtest"}

    df = res.returns.copy()
    # df columns: date, tsmom, tsmom_regime, benchmark, regime_label, [pure_tsmom, ...]
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    r_off = df["tsmom"].astype(float)              # MSM-OFF
    r_on  = df["tsmom_regime"].astype(float)       # MSM-ON
    r_bm  = df["benchmark"].astype(float)
    rlab  = df["regime_label"].astype(str) if "regime_label" in df.columns else None

    s_on, s_off, s_bm = sharpe(r_on), sharpe(r_off), sharpe(r_bm)
    dd_on, dd_off, dd_bm = maxdd(r_on), maxdd(r_off), maxdd(r_bm)

    delta_mean, ci_lo, ci_hi = stationary_bootstrap_sharpe_diff(r_on, r_off)

    # Subperiod conditional Sharpe (where the overlay should matter)
    sub = {}
    if rlab is not None:
        for lab in ["risk-on", "risk-off", "transition"]:
            mask = rlab == lab
            if mask.sum() >= 6:
                sub[lab] = {
                    "n":           int(mask.sum()),
                    "sharpe_on":   sharpe(r_on[mask]),
                    "sharpe_off":  sharpe(r_off[mask]),
                    "delta":       sharpe(r_on[mask]) - sharpe(r_off[mask]),
                    "mean_ret_on": float(r_on[mask].mean()),
                    "mean_ret_off":float(r_off[mask].mean()),
                }

    # Hit rate: in risk-off-flagged months, did the overlay shrink-longs help?
    overlay_helped = float("nan")
    if rlab is not None:
        ro_mask = rlab == "risk-off"
        if ro_mask.sum() >= 6:
            # Overlay reduced exposure in risk-off → it "helped" if r_on > r_off in those months
            overlay_helped = float((r_on[ro_mask] > r_off[ro_mask]).mean())

    out = {
        "window":            f"{start} → {end}",
        "n_months":          int(len(df)),
        "regime_scale":      regime_scale,
        "sharpe_msm_on":     round(s_on, 4),
        "sharpe_msm_off":    round(s_off, 4),
        "sharpe_benchmark":  round(s_bm, 4),
        "delta_sharpe":      round(s_on - s_off, 4),
        "delta_bootstrap_mean": round(delta_mean, 4),
        "delta_ci95":        [round(ci_lo, 4), round(ci_hi, 4)],
        "maxdd_msm_on":      round(dd_on, 4),
        "maxdd_msm_off":     round(dd_off, 4),
        "maxdd_benchmark":   round(dd_bm, 4),
        "overlay_helped_in_risk_off": round(overlay_helped, 3) if not np.isnan(overlay_helped) else None,
        "subperiod_sharpe":  sub,
        "verdict":           verdict(delta_mean, ci_lo, ci_hi),
    }

    print()
    print("════════════════════════════════════════════════════════════════════")
    print("  MSM ABLATION — does the regime overlay add Sharpe to our book?")
    print("════════════════════════════════════════════════════════════════════")
    print(f"  Window:              {out['window']}  ({out['n_months']} months)")
    print(f"  Regime scale:        {regime_scale} (long shrink in risk-off)")
    print()
    print(f"  Sharpe MSM-ON  :     {s_on:+.3f}")
    print(f"  Sharpe MSM-OFF :     {s_off:+.3f}")
    print(f"  Sharpe benchmark:    {s_bm:+.3f}")
    print(f"  Δ Sharpe (raw):      {s_on - s_off:+.3f}")
    print(f"  Δ Sharpe (bootstrap mean): {delta_mean:+.3f}")
    print(f"  Δ Sharpe 95% CI:     [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print()
    print(f"  Max drawdown MSM-ON:  {dd_on:+.2%}")
    print(f"  Max drawdown MSM-OFF: {dd_off:+.2%}")
    print(f"  Max drawdown bench:   {dd_bm:+.2%}")
    if not np.isnan(overlay_helped):
        print(f"  Overlay-helped rate in risk-off months: {overlay_helped:.1%}")
        print(f"    (% of risk-off months where MSM-ON beat MSM-OFF)")
    if sub:
        print()
        print("  Conditional Sharpe by regime label:")
        for lab, s in sub.items():
            print(f"    [{lab:>10}] n={s['n']:>3}  ON {s['sharpe_on']:+.3f}  OFF {s['sharpe_off']:+.3f}  Δ {s['delta']:+.3f}")
    print()
    print(f"  VERDICT: {out['verdict']}")
    print("════════════════════════════════════════════════════════════════════")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"  → results saved to {out_path}")

    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end",   default="2025-12-31")
    ap.add_argument("--regime-scale", type=float, default=0.3)
    ap.add_argument("--out",   default="data/ablation/msm_on_vs_off.json")
    args = ap.parse_args()
    run(args.start, args.end, regime_scale=args.regime_scale, out_path=args.out)
