"""scripts/audit_gpa_ff5_anchor.py — GP/A external anchor audit (D step).

Regresses the 2026-06-08 GP/A EW PnL series (spec_hash dc4cf6beaa247880,
PROFITABILITY family) on Ken French FF5 + Momentum monthly factors. Tests
whether GP/A has α BEYOND known anchor factors — the critical question
for a clean PROMOTE decision.

Why this matters: Novy-Marx 2013 §3 reports α > 0 for GP/A vs CAPM /
FF3 / FF3+MOM. FF5 (added RMW + CMA in 2015) was DESIGNED to absorb the
profitability premium via RMW. If our GP/A spec is SUBSUMED by RMW, the
GREEN verdict is real-but-not-new — it's repackaged RMW exposure, not
incremental alpha.

Hypotheses:
  H1: α-t > 1.96 vs FF5+MOM → GP/A has incremental α — strong PROMOTE candidate
  H2: 1.65 ≤ α-t ≤ 1.96    → boundary; needs further investigation
  H3: α-t < 1.65            → SUBSUMED by FF5; PROMOTE rejects

Methodology:
  - Test asset: GP/A monthly L/S PnL (zero-cost, no RF subtraction needed)
  - Model: MKT_RF, SMB, HML, RMW, CMA, MOM (FF5 + UMD)
  - HAC SE: Newey-West lag 6 (consistent with template default)
  - Bailey-LdP n_trials threshold NOT applied here (single-factor audit,
    n_trials=1 in PROFITABILITY family at the time of dispatch)

Run:
    python scripts/audit_gpa_ff5_anchor.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

GPA_PNL_PARQUET = _REPO_ROOT / "data" / "research_store" / "tier_c_pnl" / "dc4cf6beaa247880_GREEN.parquet"
KF_DAILY        = _REPO_ROOT / "data" / "cache" / "ken_french_ff5_mom_daily.parquet"
OUT_DIR         = _REPO_ROOT / "data" / "research_store" / "audit" / "gpa_ff5_anchor_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_gpa_pnl() -> pd.Series:
    df = pd.read_parquet(GPA_PNL_PARQUET)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    # Use the 13bp net column to match the verdict (cost-aware)
    s = df["pnl_net_13bp"].dropna()
    # Normalize index to month-end
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    return s


def _load_kf_factors_monthly() -> pd.DataFrame:
    """Daily KF FF5+MOM → monthly compounded returns."""
    df = pd.read_parquet(KF_DAILY)
    # Pre-2026 cache uses date index already; if columns expressed as decimals
    # (e.g. 0.0067 = 0.67%) we keep them as-is — they're already fractional
    monthly = (1.0 + df).resample("ME").prod() - 1.0
    return monthly.dropna(how="all")


def _spanning_regression(y: pd.Series, X: pd.DataFrame) -> dict:
    """OLS with Newey-West HAC SE (lag 6).
    No RF subtraction — y is assumed zero-cost (L/S).
    """
    df = pd.concat({"y": y, **{c: X[c] for c in X.columns}}, axis=1).dropna()
    n = len(df)
    if n < 24:
        return {"n": n, "alpha": None, "alpha_t": None, "betas": {},
                "r_squared": None, "error": "insufficient_obs"}

    import statsmodels.api as sm
    X_const = sm.add_constant(df[list(X.columns)].values)
    ols = sm.OLS(df["y"].values, X_const).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6},
    )
    return {
        "n":          n,
        "alpha":      float(ols.params[0]),
        "alpha_t":    float(ols.tvalues[0]),
        "betas":      {c: float(ols.params[1 + i])  for i, c in enumerate(X.columns)},
        "beta_ts":    {c: float(ols.tvalues[1 + i]) for i, c in enumerate(X.columns)},
        "r_squared":  float(ols.rsquared),
        "r_squared_adj": float(ols.rsquared_adj),
        "f_pvalue":   float(ols.f_pvalue) if math.isfinite(ols.f_pvalue) else None,
    }


def _verdict(alpha_t: float | None) -> tuple[str, str]:
    if alpha_t is None or not math.isfinite(alpha_t):
        return "UNKNOWN", "Regression failed"
    a = abs(alpha_t)
    if a >= 1.96:
        return ("NOT_SUBSUMED",
                f"alpha-t={alpha_t:+.2f} clears 1.96 threshold — "
                "GP/A has α beyond FF5+MOM")
    if a >= 1.65:
        return ("INDETERMINATE",
                f"alpha-t={alpha_t:+.2f} in (1.65, 1.96) — boundary, "
                "investigate further")
    return ("SUBSUMED",
            f"alpha-t={alpha_t:+.2f} < 1.65 — FF5+MOM fully explains GP/A "
            "(profitability premium ≈ RMW exposure)")


def main():
    print("Loading GP/A PnL series + KF FF5+MOM factors...")
    gpa = _load_gpa_pnl()
    kf  = _load_kf_factors_monthly()
    # Align indices: KF month-end vs GP/A month-end (already month-end)
    kf.index = kf.index.to_period("M").to_timestamp("M")
    print(f"  GP/A series:  n={len(gpa)}  range={gpa.index.min().date()} → {gpa.index.max().date()}")
    print(f"  KF factors:   n={len(kf)}    range={kf.index.min().date()} → {kf.index.max().date()}")
    print(f"  Overlap:      n={len(gpa.index.intersection(kf.index))} months")
    print()

    runs = [
        ("CAPM",       ["MKT_RF"]),
        ("FF3",        ["MKT_RF", "SMB", "HML"]),
        ("FF3+MOM",    ["MKT_RF", "SMB", "HML", "MOM"]),
        ("FF5",        ["MKT_RF", "SMB", "HML", "RMW", "CMA"]),
        ("FF5+MOM",    ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]),
    ]

    print(f"{'model':<12}{'n':>5}{'alpha%/mo':>12}{'alpha-t':>10}{'R^2':>8}{'beta_RMW':>10}{'t_RMW':>8}  verdict")
    print("=" * 110)

    all_results = {}
    for model_name, cols in runs:
        X = kf[cols]
        r = _spanning_regression(gpa, X)
        v, msg = _verdict(r.get("alpha_t"))
        if r.get("alpha") is None:
            print(f"  {model_name:<12} n={r.get('n')} ERROR: {r.get('error','?')}")
            all_results[model_name] = r
            continue
        a_pct = r["alpha"] * 100  # to percent/month
        beta_rmw = r["betas"].get("RMW", 0.0)
        t_rmw    = r["beta_ts"].get("RMW", float("nan"))
        rmw_str = f"{beta_rmw:+.2f}" if "RMW" in cols else "—"
        trmw_str = f"{t_rmw:+.2f}"    if "RMW" in cols else "—"
        print(f"  {model_name:<12}{r['n']:>5}{a_pct:>+10.3f}%{r['alpha_t']:>10.2f}"
              f"{r['r_squared']:>8.3f}{rmw_str:>10}{trmw_str:>8}  {v}: {msg[:50]}")
        all_results[model_name] = {**r, "verdict": v, "verdict_msg": msg}

    # Headline takeaway: the FF5+MOM result is the most stringent
    headline = all_results.get("FF5+MOM", {})
    print()
    print("=" * 110)
    print("HEADLINE — GP/A vs FF5+MOM (most stringent anchor)")
    print("=" * 110)
    if headline.get("alpha") is not None:
        ann_alpha = headline["alpha"] * 12
        print(f"  α (monthly):    {headline['alpha']*100:+.3f}%   (annualized: {ann_alpha*100:+.2f}%)")
        print(f"  α-t (HAC L6):   {headline['alpha_t']:+.3f}")
        print(f"  R²:             {headline['r_squared']:.4f}  (adj: {headline['r_squared_adj']:.4f})")
        print(f"  n months:       {headline['n']}")
        print(f"  Verdict:        {headline['verdict']}")
        print()
        print("  Betas (HAC t-stat):")
        for c, b in headline["betas"].items():
            t = headline["beta_ts"][c]
            sig = "***" if abs(t) >= 2.58 else "**" if abs(t) >= 1.96 else "*" if abs(t) >= 1.65 else ""
            print(f"    {c:<8} β={b:+.3f}  t={t:+.2f}  {sig}")

    out_json = OUT_DIR / "gpa_ff5_anchor_stats.json"
    out_json.write_text(json.dumps({
        "subject":  "tier_c_auto_seed_gpa_cross_sectional_rank",
        "parent_verdict_event_id":   "704b792e-fb8c-4f93-95df-585f6818ab20",
        "audit_window":              [str(gpa.index.min().date()),
                                       str(gpa.index.max().date())],
        "results_by_model":          {k: {kk: vv for kk,vv in v.items()
                                            if kk not in ("error",)}
                                       for k, v in all_results.items()},
        "method":                    "OLS with Newey-West HAC SE lag 6, "
                                      "test_asset = GP/A monthly net 13bp L/S, "
                                      "no RF subtraction (zero-cost portfolio)",
    }, indent=2, default=str))
    print()
    print(f"Stats saved → {out_json}")


if __name__ == "__main__":
    main()
