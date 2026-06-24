"""scripts/audit_deployed_sleeves_rigor.py — Deployed Sleeve Rigor Audit.

Runs each deployed sleeve's PnL through the L2-4 + L2-5 + L2-6 lite
rigor stack (same lens that demolished GP/A 2026-06-09) and writes
a single markdown audit report.

NOT a SLM action — purely research-side observability. Per A+B
doctrine [[project-a-plus-b-substrate-first-roadmap-2026-06-05]],
findings inform the principal; they do NOT auto-trigger
DECOMMISSION / ramp-down / ramp-up.

Scope (4/5 deployed sleeves; tsmom requires running engine backtest):
  - equity_book (PIT SN)            — published equity factor
  - cross_asset_carry               — KMPV 2013 carry premium
  - crisis_hedge_tlt_gld            — diversifier (no alpha claim)
  - mom_hedge_overlay               — insurance (no alpha claim)
  - cross_asset_tsmom               — SKIPPED (needs backtest run)

Output: docs/audit/deployed_sleeve_rigor_<date>.md
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.research.anchor_regression import (
    compute_for_tier_c_pnl_series as anchor_stage1,
)
from engine.research.subsample_stability import (
    compute_for_tier_c_pnl_series as subsample,
)
from engine.research.industry_attribution import (
    compute_for_tier_c_with_stage1_residual as industry_stage2,
)
from engine.research.cross_asset_attribution import (
    compute_for_tier_c_with_macro as macro_stage3,
)


# ────────────────────────────────────────────────────────────────────
# Sleeve manifest — deployed strategies + their PnL artifact paths
# ────────────────────────────────────────────────────────────────────
SLEEVES = [
    {
        "name":             "equity_book",
        "role":             "alpha",        # legacy field name retained
        "investment_role":  "alpha",        # 7-axis (Phase 1)
        "asset_class":      "equity",
        "label":            "PIT SN (D_PEAD + IBES combo)",
        "paper_cite":       "Boehmer-Jones-Wu 2008-style short-interest signal",
        "parquet":          "data/cache/_dpead_pit_sn_ibes_combo_monthly.parquet",
        "column":           "combo",
        "expect_alpha":     True,
        "expect_tier":      "C",
        "expect_spanning_risk": "HIGH",
    },
    {
        "name":             "cross_asset_carry",
        "role":             "alpha",
        "investment_role":  "alpha",
        "asset_class":      "cross_asset",
        "label":            "G10 carry 4-leg",
        "paper_cite":       "Koijen-Moskowitz-Pedersen-Vrugt 2013, Hassan-Mertens 2017",
        "parquet":          "data/research/carry_run_2026-06-05/cross_asset_carry_4leg_monthly_returns.parquet",
        "column":           "cross_asset_carry_long_short",
        "expect_alpha":     True,
        "expect_tier":      "C",
        "expect_spanning_risk": "MED-HIGH",
    },
    {
        "name":             "crisis_hedge_tlt_gld",
        "role":             "diversifier",
        "investment_role":  "diversifier",  # routes to Tier D per A3
        "asset_class":      "cross_asset",
        "label":            "TLT + GLD overlay",
        "paper_cite":       "ad-hoc diversifier; not an alpha claim",
        "parquet":          "data/cache/_crisis_hedge_monthly.parquet",
        "column":           "ac_monthly",
        "expect_alpha":     False,
        "expect_tier":      "D",
        "expect_spanning_risk": "N/A",
    },
    {
        "name":             "mom_hedge_overlay",
        "role":             "insurance",
        "investment_role":  "insurance",    # routes to Tier D per A3
        "asset_class":      "equity",
        "label":            "MTUM short β-overlay",
        "paper_cite":       "ad-hoc insurance; not an alpha claim",
        "parquet":          "data/cache/_mom_hedge_monthly.parquet",
        "column":           "mom_hedge",
        "expect_alpha":     False,
        "expect_tier":      "D",
        "expect_spanning_risk": "N/A",
    },
]


def _to_pnl_series_df(returns: pd.Series) -> pd.DataFrame:
    """Convert a single net-return series to the pnl_series_df shape
    that our rigor stack expects. Deployed sleeves don't have
    gross/turnover persisted, so we use net as both gross and net
    (acknowledged as approximation in the audit report)."""
    return pd.DataFrame({
        "pnl_gross":    returns,   # approximation — true gross unavailable
        "pnl_net_13bp": returns,   # the actual deployed-sleeve net return
        "pnl_net_80bp": returns,   # placeholder; cost stress N/A here
        "turnover":     np.nan,    # not available
    }, index=returns.index)


def _audit_one_sleeve(s: dict) -> dict:
    """Run a single sleeve through L2-4 Stage 1 + L2-5 + L2-6 Stage 2.
    Returns dict with all three lens outputs (or None on per-lens failure)
    plus per-sleeve summary stats."""
    p = REPO_ROOT / s["parquet"]
    if not p.exists():
        return {**s, "error": f"parquet missing: {p}"}
    df_raw = pd.read_parquet(p)
    if s["column"] not in df_raw.columns:
        return {**s, "error": f"column '{s['column']}' not in {list(df_raw.columns)}"}

    series = df_raw[s["column"]].dropna()
    series.index = pd.DatetimeIndex(series.index)
    n_months = len(series)
    sharpe = float(series.mean() / series.std() * (12 ** 0.5)) \
        if series.std() > 0 else float("nan")
    ann_ret = float(series.mean() * 12 * 100)
    ann_vol = float(series.std() * (12 ** 0.5) * 100)

    pnl_df = _to_pnl_series_df(series)

    out = {**s, "n_months": n_months, "date_start": str(series.index.min().date()),
              "date_end": str(series.index.max().date()),
              "sharpe": sharpe, "ann_return_pct": ann_ret, "ann_vol_pct": ann_vol,
              "error": None}

    # L2-4 Stage 1
    try:
        out["stage1"] = anchor_stage1(pnl_df)
    except Exception as exc:
        out["stage1"] = None
        out["stage1_error"] = str(exc)

    # L2-5 subsample
    try:
        out["subsample"] = subsample(pnl_df, n_splits=4)
    except Exception as exc:
        out["subsample"] = None
        out["subsample_error"] = str(exc)

    # L2-6 Stage 2 — industry extension (equity sleeves only;
    # cross-asset sleeves get industry skipped to avoid mis-spec)
    out["stage2"] = None
    if out.get("stage1") and s.get("asset_class") == "equity":
        try:
            out["stage2"] = industry_stage2(out["stage1"], pnl_df)
        except Exception as exc:
            out["stage2_error"] = str(exc)

    # Cross-asset macro extension (all sleeves)
    out["stage3_macro"] = None
    if out.get("stage1"):
        try:
            include_ind = (s.get("asset_class") == "equity")
            out["stage3_macro"] = macro_stage3(
                out["stage1"], out.get("stage2"), pnl_df,
                include_industry=include_ind,
            )
        except Exception as exc:
            out["stage3_macro_error"] = str(exc)

    return out


def _fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or not (isinstance(x, (int, float)) and np.isfinite(x)):
        return "N/A"
    return f"{x*100:+.{digits}f}%"


def _fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None or not (isinstance(x, (int, float)) and np.isfinite(x)):
        return "N/A"
    return f"{x:+.{digits}f}"


def _write_audit_report(results: list[dict], out_path: Path) -> None:
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    lines = [
        f"# Deployed Sleeve Rigor Audit — {today}",
        "",
        "**What this is**: each deployed sleeve's monthly PnL run through "
        "Tier C's L2-4 (Ken French FF5+MOM anchor regression) + L2-5 "
        "(4-split subsample stability + McLean-Pontiff decay) + L2-6 "
        "(12-Industry JOINT regression per post-FWL-fix mechanics) + "
        "cross-asset macro extension. **As of Phase 2 Commits 1-4 "
        "(2026-06-09): LRV 2011 HML_FX + DOL panel auto-included when "
        "cached** — replaces lite macro proxy for FX carry attribution.",
        "",
        "**What this is NOT**: any kind of SLM action. Per A+B doctrine "
        "[[project-a-plus-b-substrate-first-roadmap-2026-06-05]] capital "
        "decisions stay HUMAN. Findings inform; do not auto-decommission.",
        "",
        "**Caveats baked in**:",
        "- Deployed sleeves' gross PnL + turnover are NOT persisted. We "
        "  use NET return as both gross AND net (Stage 1 gross-vs-net "
        "  delta is mechanical 0 here; ignore that column).",
        "- Sleeves with `role=diversifier` or `role=insurance` are NOT "
        "  alpha-claim sleeves; spanning critique is N/A by design. "
        "  Phase 1 commit c31a81f6 routes them to Tier D separately.",
        "- cross_asset_tsmom is SKIPPED (needs engine backtest run; "
        "  add to follow-on audit).",
        "- LRV FX carry anchors window: 2002-04 → 2026-01 (binding "
        "  constraint = JPY 3M interbank rate series). For sleeves with "
        "  shorter history (e.g., equity_book 2014+), LRV adds nothing; "
        "  for cross_asset_carry (1999+) it adds the canonical academic "
        "  FX carry attribution.",
        "",
        "---",
        "",
        "## Summary table",
        "",
        "| sleeve | role | n_mo | Sharpe | ann_ret | α₁ t (FF5+MOM) | α₂ t (+ Industry) | **α₃ t (+ Macro = full)** | Δα₁→α₃ | subsample stable? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['name']} | ERROR | — | — | — | — | — | — | — |")
            continue
        s1 = r.get("stage1") or {}
        s2 = r.get("stage2") or {}
        s3 = r.get("stage3_macro") or {}
        ss = r.get("subsample") or {}
        s1_t = _fmt_num(s1.get("alpha_nw_t"))
        s2_t = _fmt_num(s2.get("alpha_full_nw_t")) if s2 else "—"
        s3_t = _fmt_num(s3.get("alpha_full_nw_t"))
        stable = ss.get("institutional_stable")
        stable_str = "✓" if stable else ("✗" if stable is False else "?")
        # Compute crude decay if 4 splits available
        decay_pct = "N/A"
        windows = ss.get("windows") or []
        sharpes = [w.get("sharpe_ann") for w in windows
                     if w.get("sharpe_ann") is not None]
        if len(sharpes) >= 4:
            first = sum(sharpes[:2]) / 2
            second = sum(sharpes[2:]) / 2
            if first > 0:
                decay_pct = f"{(1 - second/first)*100:+.0f}%"
        # Δα₁→α₃ (FF5+MOM-only → full joint with macro)
        delta_full = s3.get("delta_vs_ff5mom_nw_t")
        delta_str = _fmt_num(delta_full)
        lines.append(
            f"| **{r['name']}** | {r['role']} | {r.get('n_months','?')} | "
            f"{r.get('sharpe',float('nan')):+.2f} | "
            f"{r.get('ann_return_pct',float('nan')):+.1f}% | "
            f"{s1_t} | {s2_t} | **{s3_t}** | {delta_str} | {stable_str} |"
        )

    # NOTE: 'stage2' below contains JOINT-model output per
    # 2026-06-09 FWL bug fix. Field names changed:
    # alpha_nw_t → alpha_full_nw_t; old "Stage 2 α" was a FWL
    # artifact and has been removed entirely.
    lines += ["", "---", "", "## Per-sleeve detail", ""]
    for r in results:
        lines += [f"### {r['name']} — `role={r.get('role','?')}`", ""]
        if r.get("error"):
            lines += [f"**ERROR**: {r['error']}", "", ""]
            continue
        lines += [
            f"- **Label**: {r.get('label','?')}",
            f"- **Paper / lineage**: {r.get('paper_cite','?')}",
            f"- **Window**: {r['date_start']} → {r['date_end']} "
            f"({r['n_months']} months)",
            f"- **Headline Sharpe**: {r['sharpe']:+.3f}  "
            f"(ann_ret {r['ann_return_pct']:+.2f}%, "
            f"ann_vol {r['ann_vol_pct']:.2f}%)",
            f"- **Expected spanning risk** (pre-audit): "
            f"{r.get('expect_spanning_risk','?')}",
            "",
        ]
        s1 = r.get("stage1")
        if s1:
            lines += [
                "**L2-4 Stage 1 (FF5+MOM anchor regression)**",
                "",
                f"- residual α NW-t = **{s1['alpha_nw_t']:+.3f}**",
                f"- residual α annual = {s1['alpha_annual']*100:+.3f}%",
                f"- R² = {s1['r2']:.4f}",
                f"- joint F-test p-value = "
                f"{s1.get('joint_loading_f_test',{}).get('f_pvalue',float('nan')):.4g}",
            ]
            # Top 3 anchor loadings by |t|
            betas = s1.get("betas") or {}
            beta_t = s1.get("beta_nw_t") or {}
            top3 = sorted(beta_t.items(), key=lambda kv: -abs(kv[1]))[:3]
            if top3:
                lines.append("- Top 3 anchor loadings:")
                for name, t in top3:
                    b = betas.get(name, float("nan"))
                    sig = "***" if abs(t) > 2.58 else (
                        "**" if abs(t) > 1.96 else
                        "*" if abs(t) > 1.65 else "")
                    lines.append(f"    - {name}: β={b:+.3f} t={t:+.2f} {sig}")
            lines.append("")
        else:
            lines += [f"**L2-4 Stage 1**: FAILED — {r.get('stage1_error','?')}", ""]

        s2 = r.get("stage2")
        if s2:
            jf_p = (s2.get("industry_joint_f_test") or {}).get("f_pvalue")
            jf_str = f"{jf_p:.4g}" if jf_p is not None else "N/A"
            lines += [
                "**L2-6 Joint Model (FF5+MOM + 12-Industry)**  "
                "*(post-FWL-fix 2026-06-09)*",
                "",
                f"- α_full NW-t = **{s2['alpha_full_nw_t']:+.3f}**  "
                f"(vs α₁ FF5+MOM-only = {s1['alpha_nw_t']:+.3f})",
                f"- α_full annual = {s2['alpha_full_annual']*100:+.3f}%",
                f"- Δα (t-stat approx) = "
                f"{(s2.get('delta_alpha_nw_t_approx') or 0):+.3f} "
                f"(positive = industry ATE alpha)",
                f"- joint R² = {s2['r2_full']:.4f}",
                f"- industry-subset F-test p-value = {jf_str}  "
                "(H0: all 12 industry γ = 0; p < 0.01 → industries "
                "add explanation)",
            ]
            betas = s2.get("industry_betas") or {}
            beta_t = s2.get("industry_beta_nw_t") or {}
            top3 = sorted(beta_t.items(), key=lambda kv: -abs(kv[1]))[:3]
            if top3:
                lines.append("- Top 3 industry tilts (joint model):")
                for name, t in top3:
                    b = betas.get(name, float("nan"))
                    sig = "***" if abs(t) > 2.58 else (
                        "**" if abs(t) > 1.96 else
                        "*" if abs(t) > 1.65 else "")
                    lines.append(f"    - {name}: β={b:+.3f} t={t:+.2f} {sig}")
            lines.append("")
        elif r.get("stage1") and r.get("asset_class") == "equity":
            lines += [
                f"**L2-6 Joint Model**: skipped — "
                f"{r.get('stage2_error','no error logged')}",
                "",
            ]
        elif r.get("asset_class") == "cross_asset":
            lines += [
                "**L2-6 Industry Joint Model**: SKIPPED for cross-asset sleeve "
                "(US-equity industry panel is mis-specified; see cross-asset "
                "macro section below)",
                "",
            ]

        # Stage 3 — cross-asset macro extension
        s3 = r.get("stage3_macro")
        if s3:
            mf = s3.get("macro_joint_f_test") or {}
            mf_p = mf.get("f_pvalue")
            mf_str = f"{mf_p:.4g}" if mf_p is not None else "N/A"
            fx_f = s3.get("fx_carry_joint_f_test") or {}
            fx_p = fx_f.get("f_pvalue") if fx_f else None
            fx_str = f"{fx_p:.4g}" if fx_p is not None else None
            lines += [
                f"**Cross-Asset Joint Model** "
                f"(`{s3.get('model_form','?')}`)",
                "",
                f"- α_full NW-t = **{s3['alpha_full_nw_t']:+.3f}**  "
                f"(α₁ FF5+MOM = {s1.get('alpha_nw_t',float('nan')):+.3f}, "
                f"Δ = {s3.get('delta_vs_ff5mom_nw_t',float('nan')):+.3f})",
                f"- α_full annual = {s3['alpha_full_annual']*100:+.3f}%",
                f"- joint R² (full) = {s3['r2_full']:.4f}",
                f"- macro-subset F-test p = {mf_str}",
            ]
            if fx_str is not None:
                lines.append(
                    f"- **LRV FX carry-subset F-test p = {fx_str}**  "
                    "(HML_FX + DOL panel orthogonality test)"
                )
            mbetas = s3.get("macro_betas") or {}
            mbeta_t = s3.get("macro_beta_nw_t") or {}
            if mbetas:
                sorted_m = sorted(mbeta_t.items(),
                                     key=lambda kv: -abs(kv[1]))[:5]
                lines.append("- Macro loadings (joint, sorted by |t|):")
                for k, t in sorted_m:
                    b = mbetas.get(k, float("nan"))
                    sig = "***" if abs(t) > 2.58 else (
                        "**" if abs(t) > 1.96 else
                        "*" if abs(t) > 1.65 else "")
                    lines.append(f"    - {k}: β={b:+.5f} t={t:+.3f} {sig}")
            fx_betas = s3.get("fx_carry_betas") or {}
            fx_beta_t = s3.get("fx_carry_beta_nw_t") or {}
            if fx_betas:
                lines.append("- **LRV FX carry loadings** "
                                "(Lustig-Roussanov-Verdelhan 2011):")
                for k in fx_betas:
                    b = fx_betas[k]
                    t = fx_beta_t.get(k, float("nan"))
                    sig = "***" if abs(t) > 2.58 else (
                        "**" if abs(t) > 1.96 else
                        "*" if abs(t) > 1.65 else "")
                    lines.append(f"    - **{k}**: β={b:+.5f} "
                                    f"t={t:+.3f} {sig}")
            lines.append("")
        elif r.get("stage1"):
            lines += [
                f"**Cross-Asset Macro Extension**: skipped — "
                f"{r.get('stage3_macro_error','no error logged')}",
                "",
            ]

        ss = r.get("subsample")
        if ss:
            wb = ss.get("worst_best_sharpe_ratio")
            wb_str = f"{wb:.3f}" if wb is not None else "N/A"
            dt = ss.get("decay_slope_t")
            dt_str = f"{dt:+.3f}" if dt is not None else "N/A"
            ds = ss.get("decay_slope_per_year") or 0
            lines += [
                "**L2-5 Subsample stability (4-split)**",
                "",
                f"- worst/best Sharpe ratio = {wb_str}  "
                f"(institutional_stable = {ss.get('institutional_stable','?')})",
                f"- monotone_decay = {ss.get('monotone_decay','?')}; "
                f"monotone_growth = {ss.get('monotone_growth','?')}",
                f"- decay slope = {ds*100:+.4f}%/yr  (NW t = {dt_str})",
                "",
                "Per-window:",
                "",
                "| window | n | Sharpe | NW-t | ann_ret |",
                "|---|---:|---:|---:|---:|",
            ]
            for w in ss.get("windows") or []:
                s_v = w.get("sharpe_ann")
                t_v = w.get("nw_t_stat")
                r_v = w.get("ann_return")
                s_str = f"{s_v:+.3f}" if s_v is not None else "N/A"
                t_str = f"{t_v:+.2f}" if t_v is not None else "N/A"
                r_str = f"{r_v*100:+.2f}%" if r_v is not None else "N/A"
                lines.append(
                    f"| {w.get('start','?')}→{w.get('end','?')} | "
                    f"{w.get('n_months','?')} | "
                    f"{s_str} | {t_str} | {r_str} |"
                )
            lines.append("")
        lines += ["---", ""]

    lines += [
        "## How to read this report",
        "",
        "**For role=alpha sleeves** (`equity_book`, `cross_asset_carry`):",
        "- Stage1 α NW-t < 1.96 → factor's apparent alpha is largely spanned "
        "  by FF5+MOM; allocation should be justified beyond 'it's a real factor'",
        "- Stage2 α NW-t < 1.0 → factor is ALSO an industry tilt; the unique "
        "  alpha after both peels is statistically zero (GP/A pattern)",
        "- worst/best Sharpe < 0.40 → REGIME-DEPENDENT, not a stable alpha",
        "- post-pub decay > 32% → McLean-Pontiff signature; OOS expectation "
        "  should be discounted",
        "",
        "**For role=diversifier / insurance**:",
        "- α t-stats are not load-bearing — purpose is correlation structure, "
        "  not alpha",
        "- Look at the windows table for crisis-period behavior instead",
        "",
        "**Action protocol per A+B doctrine**: This report is a research "
        "artifact. SLM DECOMMISSION / RAMP_DOWN remain human decisions, "
        "informed by but not auto-triggered by these findings.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def main():
    print("Auditing deployed sleeves through L2-4/L2-5/L2-6 rigor stack...")
    results = []
    for s in SLEEVES:
        print(f"  [{s['name']}] ...", end=" ")
        r = _audit_one_sleeve(s)
        results.append(r)
        if r.get("error"):
            print(f"ERROR: {r['error']}")
        else:
            s1_t = (r.get("stage1") or {}).get("alpha_nw_t")
            s2_t = (r.get("stage2") or {}).get("alpha_full_nw_t")
            s3_t = (r.get("stage3_macro") or {}).get("alpha_full_nw_t")
            print(f"a1(FF5+MOM)={_fmt_num(s1_t)}  "
                    f"a2(+Industry)={_fmt_num(s2_t)}  "
                    f"a3(+Macro)={_fmt_num(s3_t)}")

    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    out_path = REPO_ROOT / "docs" / "audit" / f"deployed_sleeve_rigor_{today}.md"
    _write_audit_report(results, out_path)
    print()
    print(f"=== Audit complete — see {out_path.relative_to(REPO_ROOT)} ===")
    return results


if __name__ == "__main__":
    main()
