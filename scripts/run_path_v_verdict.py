"""
scripts/run_path_v_verdict.py — Path V Cross-Sectional Momentum verdict.

Runs all 5 gates per spec §2.6 (post-Amendment-1 cd9fecf5):
  G1 PRIMARY  Sharpe (net 10bp TC × turnover, ann.) ≥ 0.30
  G2          Newey-West HAC t-stat (lag-8) > 1.96
  G3          max ρ vs each of K1/D-PEAD/PATH_N/CTA weekly ≤ 0.25
  G4          Bootstrap 95% CI (1000 samples, 12-wk stationary block)
              excludes 0
  G5          ≥ 1 of 3 crisis windows non-negative
              (2018-Q4 / 2020-COVID / 2022)

Decision rule (per spec §2.6):
  5/5    → PASS
  4/5    → MARGINAL
  ≤3/5   → FAIL

Writes capability evidence MD + verdict JSON sidecar.
"""
from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (per spec)
# ─────────────────────────────────────────────────────────────────────────────
WEEKLY_RFR = 0.04 / 52.0

G1_SHARPE_THRESHOLD = 0.30
G2_T_THRESHOLD      = 1.96
G3_RHO_THRESHOLD    = 0.25
NW_LAG              = 8
BOOTSTRAP_N         = 1000
BOOTSTRAP_BLOCK     = 12

CRISIS_WINDOWS = {
    "2018_Q4":    (pd.Timestamp("2018-10-01"), pd.Timestamp("2018-12-31")),
    "2020_COVID": (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30")),
    "2022_full":  (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────
def annualized_sharpe(returns_weekly: pd.Series, rfr_weekly: float = WEEKLY_RFR) -> float:
    excess = returns_weekly - rfr_weekly
    return float(excess.mean() / excess.std() * math.sqrt(52))


def newey_west_t(excess_returns: np.ndarray, lag: int = NW_LAG) -> float:
    """Newey-West HAC t-stat for H0: mean = 0 (one-sided)."""
    x = np.asarray(excess_returns, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 2 * lag + 5:
        return float("nan")
    mu = x.mean()
    e  = x - mu
    # Sample variance (lag-0)
    gamma0 = float((e * e).mean())
    # Bartlett-weighted autocovariances
    s = gamma0
    for k in range(1, lag + 1):
        w_k = 1.0 - k / (lag + 1)
        gamma_k = float((e[k:] * e[:-k]).mean())
        s += 2.0 * w_k * gamma_k
    if s <= 0:
        return float("nan")
    se = math.sqrt(s / T)
    return float(mu / se)


def stationary_bootstrap_sharpe_ci(
    returns_weekly: pd.Series,
    n_resample:     int = BOOTSTRAP_N,
    block_mean:     int = BOOTSTRAP_BLOCK,
    rfr_weekly:     float = WEEKLY_RFR,
    seed:           int = 42,
) -> tuple[float, float]:
    """Politis-Romano 1994 stationary block bootstrap CI for Sharpe."""
    rng = np.random.default_rng(seed)
    x = returns_weekly.dropna().to_numpy()
    T = len(x)
    if T < 50:
        return (float("nan"), float("nan"))
    p = 1.0 / block_mean
    sharpes = np.empty(n_resample, dtype=float)
    for b in range(n_resample):
        sample = np.empty(T, dtype=float)
        i = rng.integers(0, T)
        for t in range(T):
            sample[t] = x[i]
            if rng.random() < p:
                i = rng.integers(0, T)
            else:
                i = (i + 1) % T
        excess = sample - rfr_weekly
        if excess.std() <= 0:
            sharpes[b] = 0.0
            continue
        sharpes[b] = excess.mean() / excess.std() * math.sqrt(52)
    return (float(np.percentile(sharpes, 2.5)),
            float(np.percentile(sharpes, 97.5)))


def crisis_returns(returns_weekly: pd.Series) -> dict[str, float]:
    """Cumulative return per crisis window."""
    out = {}
    idx = pd.to_datetime(returns_weekly.index)
    s = returns_weekly.copy()
    s.index = idx
    for label, (start, end) in CRISIS_WINDOWS.items():
        sub = s.loc[start:end].dropna()
        if len(sub) == 0:
            out[label] = float("nan")
        else:
            out[label] = float((1.0 + sub).prod() - 1.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Path V Cross-Sectional Momentum verdict run ===")
    print()

    # Load Path V returns (net)
    pv_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_path_v_csm_weekly.parquet"
    pv = pd.read_parquet(pv_path)
    pv.index = pd.to_datetime(pv.index)
    pv_net = pv["net"].dropna()
    print(f"Path V net returns: n={len(pv_net)} weeks, "
          f"{pv_net.index.min().date()} → {pv_net.index.max().date()}")

    # Load existing 4-strategy returns for G3 ρ check
    existing_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_per_strategy_returns_weekly.parquet"
    existing = pd.read_parquet(existing_path).astype("float64").fillna(0.0)
    existing.index = pd.to_datetime(existing.index)
    print(f"Existing 4 sleeves: n={len(existing)} weeks, "
          f"cols={list(existing.columns)}")
    print()

    # ── G1 PRIMARY ──
    sharpe_net = annualized_sharpe(pv_net)
    g1_pass = sharpe_net >= G1_SHARPE_THRESHOLD
    print(f"G1 PRIMARY   Sharpe (net, ann.) = {sharpe_net:+.4f}   "
          f"threshold ≥ {G1_SHARPE_THRESHOLD}   →   {'PASS' if g1_pass else 'FAIL'}")

    # ── G2 ──
    excess = (pv_net - WEEKLY_RFR).to_numpy()
    nw_t = newey_west_t(excess, lag=NW_LAG)
    g2_pass = (not math.isnan(nw_t)) and (nw_t > G2_T_THRESHOLD)
    print(f"G2           Newey-West t (lag-8) = {nw_t:+.4f}   "
          f"threshold > {G2_T_THRESHOLD}   →   {'PASS' if g2_pass else 'FAIL'}")

    # ── G3 ──
    # Align on overlap
    common_idx = pv_net.index.intersection(existing.index)
    pv_aligned = pv_net.loc[common_idx]
    existing_aligned = existing.loc[common_idx]
    rho_vec = {col: float(pv_aligned.corr(existing_aligned[col]))
               for col in existing_aligned.columns}
    max_abs_rho = max(abs(v) for v in rho_vec.values())
    g3_pass = max_abs_rho <= G3_RHO_THRESHOLD
    print(f"G3           max |ρ| vs existing sleeves = {max_abs_rho:+.4f}   "
          f"threshold ≤ {G3_RHO_THRESHOLD}   →   {'PASS' if g3_pass else 'FAIL'}")
    for col, r in rho_vec.items():
        print(f"               ρ(PathV, {col:<12}) = {r:+.4f}")

    # ── G4 ──
    ci_lo, ci_hi = stationary_bootstrap_sharpe_ci(pv_net)
    g4_pass = (not math.isnan(ci_lo)) and (ci_lo > 0)
    print(f"G4           Bootstrap 95% CI on Sharpe = [{ci_lo:+.4f}, {ci_hi:+.4f}]   "
          f"excludes 0?   →   {'PASS' if g4_pass else 'FAIL'}")

    # ── G5 ──
    crisis_ret = crisis_returns(pv_net)
    n_non_negative = sum(1 for v in crisis_ret.values()
                          if not math.isnan(v) and v >= 0)
    g5_pass = n_non_negative >= 1
    print(f"G5           Crisis non-negative count = {n_non_negative} of 3   "
          f"threshold ≥ 1   →   {'PASS' if g5_pass else 'FAIL'}")
    for label, r in crisis_ret.items():
        sign = "PASS" if (not math.isnan(r) and r >= 0) else "FAIL"
        print(f"               [{sign}] {label}: {r*100:+.3f}%")
    print()

    # ── Decision rule ──
    n_pass = sum([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass])
    if n_pass == 5:
        verdict = "PASS"
    elif n_pass == 4:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {n_pass}/5 gates PASS  →  VERDICT: {verdict}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Save verdict JSON ──
    today = datetime.date.today()
    verdict_payload = {
        "spec_id": 69,
        "spec_hash_current": "cd9fecf5",
        "spec_hash_v1": "5bd744dd",
        "amendment": "scope_narrow universe → S&P 500 (CRSP DSF panel)",
        "run_date": today.isoformat(),
        "window": {"start": str(pv_net.index.min().date()),
                   "end":   str(pv_net.index.max().date()),
                   "n_weeks": int(len(pv_net))},
        "verdict":  verdict,
        "n_pass":   int(n_pass),
        "gates": {
            "G1_sharpe_net_ann":   {"value": sharpe_net, "threshold": G1_SHARPE_THRESHOLD, "pass": g1_pass},
            "G2_newey_west_t":     {"value": nw_t,       "threshold": G2_T_THRESHOLD,      "pass": g2_pass},
            "G3_max_abs_rho":      {"value": max_abs_rho,"threshold": G3_RHO_THRESHOLD,    "pass": g3_pass,
                                     "rho_by_sleeve": rho_vec},
            "G4_bootstrap_ci_95":  {"lo": ci_lo, "hi": ci_hi, "pass": g4_pass},
            "G5_crisis_non_neg":   {"count": n_non_negative, "threshold": 1, "pass": g5_pass,
                                     "returns": crisis_ret},
        },
        "summary_stats": {
            "weekly_mean_net":   float(pv_net.mean()),
            "weekly_std_net":    float(pv_net.std()),
            "annualized_vol":    float(pv_net.std() * math.sqrt(52)),
            "weekly_mean_gross": float(pv["gross"].mean()),
            "weekly_mean_tc":    float(pv["tc"].mean()),
            "annual_tc_drag":    float(pv["tc"].sum() / (len(pv) / 52.0)),
        },
    }
    out_dir = REPO_ROOT / "data" / "portfolio_replay"
    out_path = out_dir / f"path_v_verdict_{today.isoformat()}.json"
    out_path.write_text(json.dumps(verdict_payload, indent=2, default=str),
                         encoding="utf-8")
    print()
    print(f"Verdict saved: {out_path}")


if __name__ == "__main__":
    main()
