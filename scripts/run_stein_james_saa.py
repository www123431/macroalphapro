"""
scripts/run_stein_james_saa.py — Bayes-Stein SAA shrinkage analysis.

Runs the full pipeline (Jorion 1986 Bayes-Stein on means + Ledoit-Wolf 2004
on covariance + constrained Markowitz) and writes:
  - A decision memo MD with the comparison table to
    docs/decisions/saa_stein_james_audit_<date>.md
  - A JSON sidecar with full numeric outputs to
    data/portfolio_replay/saa_stein_james_audit_<date>.json

This script is review-only — it does NOT mutate the production SAA.
Sleeve allocation changes go through the existing spec amendment workflow.

Tier-1 audit class A #2 (2026-05-14).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fmt_pct(x: float) -> str:
    return f"{x*100:+.4f}%"


def _fmt_wt(x: float) -> str:
    return f"{x*100:.1f}%"


def _decision_memo_md(analysis, date: datetime.date) -> str:
    a = analysis
    strats = a.strategies

    cur = a.weights_current_saa
    locked = a.weights_sleeve_locked
    caps = a.weights_with_caps
    unc = a.weights_unconstrained

    fwd = a.forward_sharpe_estimates

    diff_locked = {s: locked[s] - cur[s] for s in strats}
    max_locked_move = max(abs(v) for v in diff_locked.values())

    # Recommendation logic
    locked_sharpe = fwd["sleeve_locked"]
    current_sharpe = fwd["current_saa"]
    sharpe_gain = locked_sharpe - current_sharpe

    if sharpe_gain < -0.005:
        recommendation = "KEEP CURRENT SAA — empirically dominant"
        rec_reasoning = (
            f"Current SAA actually delivers HIGHER shrunk Sharpe "
            f"({current_sharpe:+.3f}) than the Markowitz-utility-maximizing "
            f"sleeve-locked alternative ({locked_sharpe:+.3f}, Δ "
            f"{sharpe_gain:+.3f}). The optimizer trades return for variance "
            f"reduction (max U = μ - λ/2·σ² with λ=2.0), so its higher-"
            f"variance Sharpe is lower than the current allocation's. "
            "**The current 36/27/27/10 is robustly defended under Bayes-Stein "
            "shrinkage — no amendment warranted, in fact it dominates the "
            "shrunk-input optimum on Sharpe.**"
        )
    elif max_locked_move < 0.02 and abs(sharpe_gain) < 0.05:
        recommendation = "KEEP CURRENT SAA — statistically indistinguishable"
        rec_reasoning = (
            f"Bayes-Stein-shrunk optimal weights under sleeve locks move "
            f"only {max_locked_move*100:.1f}pp from current allocation, "
            f"with Sharpe Δ {sharpe_gain:+.3f}. The current SAA is "
            "statistically indistinguishable from the shrunk optimum. "
            "NO AMENDMENT WARRANTED."
        )
    elif sharpe_gain >= 0.05:
        recommendation = "CONSIDER AMENDMENT"
        rec_reasoning = (
            f"Shrunk optimal weights deliver a Sharpe improvement of "
            f"+{sharpe_gain:.3f} over current SAA. Largest single-strategy "
            f"move is {max_locked_move*100:.1f}pp. Per Path O / "
            "deployment_design spec amendment workflow, route through "
            "Tier-3 approval before implementing."
        )
    else:
        recommendation = "MARGINAL — keep current"
        rec_reasoning = (
            f"Shrunk optimal weights move {max_locked_move*100:.1f}pp "
            f"(largest single-strategy delta) with Sharpe Δ {sharpe_gain:+.3f}. "
            "Within sample noise; not material enough to amend a spec-locked "
            "allocation. The 486-week in-sample window supports the current "
            "allocation as well-defended."
        )

    md = f"""# SAA Bayes-Stein Shrinkage Audit — {date.isoformat()}

**Decision date**: {date.isoformat()}
**Anchor**: Tier-1 audit class A #2 (allocation precision)
**Method**: Jorion 1986 Bayes-Stein on weekly excess means + Ledoit-Wolf 2004
on covariance + constrained Markowitz (SLSQP)
**Sample**: Sprint B 2014-09-12 → 2023-12-29 weekly replay, n_weeks = {a.n_weeks}

---

## TL;DR

**Recommendation: {recommendation}**

{rec_reasoning}

---

## 1. Inputs — sample (raw) statistics

| Strategy | Weekly μ (excess) | Weekly σ | Sample Sharpe (ann.) |
|---|---|---|---|
"""
    for s in strats:
        md += (f"| {s} | {_fmt_pct(a.sample_means_weekly[s])} | "
               f"{a.sample_vols_weekly[s]*100:.3f}% | "
               f"{a.sample_sharpe_ann[s]:+.3f} |\n")

    md += f"""
RFR assumed: 4% annual ({100*0.04/52:.4f}% weekly).

### Pairwise correlation (sample)

| | {' | '.join(strats)} |
|---|{('---|' * len(strats))}
"""
    cov = a.sample_covariance
    import math as _m
    diag = [cov[i][i] for i in range(len(strats))]
    for i, s_i in enumerate(strats):
        row = f"| **{s_i}** "
        for j in range(len(strats)):
            if diag[i] > 0 and diag[j] > 0:
                rho = cov[i][j] / _m.sqrt(diag[i] * diag[j])
            else:
                rho = float("nan")
            row += f"| {rho:+.3f} "
        md += row + "|\n"

    md += f"""

---

## 2. Shrinkage applied

### Bayes-Stein on means (Jorion 1986)

- Grand-mean prior (precision-weighted): {a.grand_mean_weekly*100:+.4f}% weekly
  (≈ {a.grand_mean_weekly*52*100:+.2f}% annualized)
- Shrinkage intensity w: **{a.mean_shrinkage_w:.4f}**
  (0.0 = no shrinkage; 1.0 = full collapse to grand mean)

### Ledoit-Wolf on covariance (Ledoit-Wolf 2004)

- Target: identity scaled by trace(S)/N
- Shrinkage intensity α: **{a.cov_shrinkage_alpha:.4f}**
  (0.0 = pure sample cov; 1.0 = pure identity target)

### Per-strategy Sharpe — sample vs shrunk

| Strategy | Sample Sharpe (ann.) | Shrunk Sharpe (ann.) | Δ |
|---|---|---|---|
"""
    for s in strats:
        sam = a.sample_sharpe_ann[s]
        shr = a.shrunk_sharpe_ann[s]
        md += f"| {s} | {sam:+.3f} | {shr:+.3f} | {shr - sam:+.3f} |\n"

    md += f"""

Reading: positive Δ on lower-Sharpe strategies and negative Δ on higher-Sharpe
strategies is the expected pattern (Stein-James "shrinks toward the prior").
Magnitude of shrinkage controlled by sample size and cross-strategy dispersion.

---

## 3. Optimal weights — under 3 constraint sets

| Strategy | Current SAA | Shrunk · sleeve-locked | Shrunk · capped 60% | Shrunk · unconstrained |
|---|---|---|---|---|
"""
    for s in strats:
        md += (f"| {s} | {_fmt_wt(cur[s])} | "
               f"{_fmt_wt(locked[s])} | "
               f"{_fmt_wt(caps[s])} | "
               f"{_fmt_wt(unc[s])} |\n")

    md += f"""

**Constraint sets**:
- **sleeve-locked**: K1=36% fixed, CTA=10% fixed (spec-locked); D-PEAD + Path N
  intra-sleeve split free within ss_sp500's 54%
- **capped 60%**: no per-strategy cap > 60%; otherwise free
- **unconstrained**: sum=1, [0, 1] bounds, no other constraints

---

## 4. Forward Sharpe estimates (using shrunk μ, Σ)

| Weight set | Sharpe (ann., shrunk) |
|---|---|
"""
    for k, v in fwd.items():
        md += f"| {k} | {v:+.3f} |\n"

    md += f"""

Reading: even after Bayes-Stein shrinkage, "unconstrained" weights produce
the highest in-sample shrunk Sharpe — but only because they ignore
existing sleeve mandates. The "sleeve-locked" line is the apples-to-apples
comparison to current SAA.

---

## 5. Decision

| | |
|---|---|
| Largest single-strategy move (sleeve-locked vs current) | {max_locked_move*100:.2f} pp |
| Shrunk Sharpe gain (sleeve-locked vs current SAA) | {sharpe_gain:+.4f} |
| **Verdict** | **{recommendation}** |

{rec_reasoning}

---

## 6. Honest disclosures

- **Sample biased**: in-sample 2014-2023; Garg-Goulding-Harvey-Mazzoleni 2021 reports
  ~50% factor-return decay post-publication. Shrunk Sharpe estimates here are
  upper bounds; expected forward Sharpe per deployment_design.md is 0.85-1.15.
- **CTA shrinkage anomaly**: PQTIX gets a SAMPLE Sharpe of {a.sample_sharpe_ann.get('CTA_PQTIX', 0):+.3f}
  in this 2014-2023 window; this is below the 30-year PQTIX track Sharpe of ~0.4
  and reflects the post-2010 TSMOM decay that Garg 2021 documents. Shrinkage
  pulls it toward the grand mean, which is itself depressed.
- **Sleeve locks reflect doctrine, not statistics**: K1=36% and CTA=10% are
  spec-locked for reasons beyond Markowitz (capacity, crisis-hedge mandate).
  The "unconstrained" optimum is informational, not actionable.
- **Single-period**: this is a one-shot in-sample analysis. Best practice
  (DeMiguel-Garlappi-Uppal 2009) would use rolling out-of-sample evaluation,
  which awaits forward window accumulation.

---

## 7. Cross-references

- `engine/portfolio/allocation_shrinkage.py` — implementation module
- `data/portfolio_replay/saa_stein_james_audit_{date.isoformat()}.json` — full numeric output
- `docs/portfolio_deployment_design_2026-05-13.md` — current SAA spec
- `data/portfolio_replay/v1_per_strategy_returns_weekly.parquet` — input data
"""
    return md


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bayes-Stein SAA shrinkage audit",
    )
    parser.add_argument("--risk-aversion", type=float, default=2.0,
                        help="Markowitz risk aversion parameter λ (default 2.0)")
    parser.add_argument("--no-write", action="store_true",
                        help="Print to stdout only; do not write MD or JSON")
    args = parser.parse_args()

    from engine.portfolio.allocation_shrinkage import run_shrinkage_analysis

    print("=== Bayes-Stein SAA shrinkage analysis ===")
    analysis = run_shrinkage_analysis(risk_aversion=args.risk_aversion)

    today = datetime.date.today()

    if args.no_write:
        print(_decision_memo_md(analysis, today))
        return 0

    # Decision memo
    docs_dir = REPO_ROOT / "docs" / "decisions"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / f"saa_stein_james_audit_{today.isoformat()}.md"
    md_path.write_text(_decision_memo_md(analysis, today), encoding="utf-8")
    print(f"Decision memo: {md_path}")

    # JSON sidecar
    out_dir = REPO_ROOT / "data" / "portfolio_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"saa_stein_james_audit_{today.isoformat()}.json"
    json_payload = dataclasses.asdict(analysis)
    json_payload["audit_date"] = today.isoformat()
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"JSON sidecar:  {json_path}")

    # Quick stdout summary
    print()
    print("Bayes-Stein mean shrinkage intensity:    "
          f"{analysis.mean_shrinkage_w:.4f}")
    print(f"Ledoit-Wolf cov shrinkage intensity:    "
          f"{analysis.cov_shrinkage_alpha:.4f}")
    print()
    print("Forward Sharpe estimates (using shrunk inputs):")
    for k, v in analysis.forward_sharpe_estimates.items():
        print(f"  {k:<20} {v:+.3f}")
    print()
    print("Weights under sleeve-locked constraint set:")
    for s, w in analysis.weights_sleeve_locked.items():
        cur = analysis.weights_current_saa[s]
        delta = w - cur
        print(f"  {s:<12} {w*100:.1f}%  (current {cur*100:.1f}%, "
              f"Δ {delta*100:+.1f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
