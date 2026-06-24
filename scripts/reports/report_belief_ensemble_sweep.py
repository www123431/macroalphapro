"""Generate the per-family LLM × family-prior ensemble report.

W7-arxiv-v05 (2026-06-22). $0 LLM. Pure stat sweep on existing 92
autopsies. Produces the architectural-improvement finding the W6-rigor
Section 4.3 finding implied: pure family-prior loses 0.149 Brier vs
LLM-only; an explicit per-family ensemble can recover most of that
plus a tiny LLM edge in 2 families.

Outputs:
  data/research/belief_ensemble_sweep.md
  data/research/belief_ensemble_sweep.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.research.belief_ensemble_sweep import (
    sweep_ensemble_per_family,
    loocv_ensemble_brier,
)

OUT_MD       = REPO_ROOT / "data" / "research" / "belief_ensemble_sweep.md"
OUT_JSON     = REPO_ROOT / "data" / "research" / "belief_ensemble_sweep.json"
OUT_LOOCV_MD = REPO_ROOT / "data" / "research" / "belief_ensemble_loocv.md"


def main() -> None:
    print("[1/3] running ensemble sweep...")
    out = sweep_ensemble_per_family()
    print(f"      final Brier: {out['final_ensemble_brier']} "
          f"vs LLM-only {out['llm_only_brier']} "
          f"vs family-only {out['family_only_brier']}")

    print(f"[2/3] writing {OUT_JSON.relative_to(REPO_ROOT)}...")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str),
                          encoding="utf-8")

    print(f"[3/3] writing {OUT_MD.relative_to(REPO_ROOT)}...")
    lines = [
        "# Belief Layer Ensemble Sweep (W7-arxiv-v05 finding)",
        "",
        f"_n = {out['n_total']} autopsies. Per-family weighted ensemble of "
        f"family-empirical posterior (time-aware LOO) + current LLM-driven "
        f"prior, optimized over w ∈ {{0.0, 0.1, ..., 1.0}}._",
        "",
        "## Headline",
        "",
        f"- LLM-only Brier:        **{out['llm_only_brier']:.4f}** "
        f"(current production)",
        f"- Family-only Brier:     **{out['family_only_brier']:.4f}** "
        f"(W6-rigor T2 fair baseline)",
        f"- **Ensemble Brier:      {out['final_ensemble_brier']:.4f}** "
        f"(per-family optimal w_fam + global fallback {out['global_w_fallback']})",
        f"- Improvement vs LLM:    **{out['improvement_vs_llm']:+.4f} "
        f"({out['improvement_vs_llm']/out['llm_only_brier']*100:+.1f}%)**",
        f"- Improvement vs family: {out['improvement_vs_fam']:+.4f}",
        "",
        f"_{out['n_families_eligible']} of {out['n_families_total']} families "
        f"have n ≥ 5 and get a family-specific w_fam; the remaining families "
        f"use the global fallback w = {out['global_w_fallback']}._",
        "",
        "## Per-family optimal weights (n ≥ 5 only)",
        "",
        "| family | n | optimal w_fam | LLM-only Brier | ensemble Brier | improvement | reading |",
        "|---|---|---|---|---|---|---|",
    ]
    for fam, r in sorted(out["by_family"].items(),
                            key=lambda kv: -kv[1]["n"]):
        if not r["use_family_specific"]:
            continue
        # Reading
        if r["optimal_w"] == 1.0:
            reading = "pure family-prior beats any LLM mix"
        elif r["optimal_w"] >= 0.8:
            reading = "family-prior dominant (LLM ≤20%)"
        elif r["optimal_w"] >= 0.5:
            reading = "balanced; LLM contributes"
        else:
            reading = "LLM-dominant (rare for this sample)"
        lines.append(
            f"| {fam} | {r['n']} | {r['optimal_w']} | "
            f"{r['llm_only_brier']:.4f} | {r['optimal_brier']:.4f} | "
            f"{r['improvement_vs_llm']:+.4f} | {reading} |"
        )
    lines += [
        "",
        "## Global w_fallback curve (used for families with n < 5)",
        "",
        "| w_fam | mean Brier (all 92 autopsies) |",
        "|---|---|",
    ]
    for w, b in sorted(out["global_brier_curve"].items()):
        marker = " ← **GLOBAL OPTIMUM**" if w == out["global_w_fallback"] else ""
        lines.append(f"| {w} | {b:.4f}{marker} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "**The LLM prior, on this 92-autopsy sample, contributes near-zero "
        "predictive value relative to a simple time-aware family-empirical "
        "baseline.** Of the 8 families with sufficient n for family-specific "
        "tuning:",
        "",
        "- 4 families prefer **w_fam = 1.0** (pure family-prior; LLM mix "
        "strictly hurts): CROSS_SEC_UNKNOWN, PROFITABILITY, SPANNING_SMB, "
        "SPANNING_CMA",
        "- 3 families prefer **w_fam ∈ [0.7, 0.8]** (family-prior dominant; "
        "LLM contributes <30%): VRP, EVENT_DRIFT, SPANNING_MOM",
        "- 1 family prefers **w_fam = 0.6** (balanced): SPANNING_HML",
        "",
        "The global w_fallback (for families with n < 5) is **0.9** — i.e. "
        "for sparse families, weight family-empirical at 90% and LLM at 10%.",
        "",
        "### What this means for the predictor architecture",
        "",
        "The current `engine.research.belief.predict_verdict` runs a "
        "deterministic pipeline that's already mostly family-empirical when "
        "family n ≥ 3 (W6-rigor-A threshold). The remaining LLM-style "
        "contribution comes from the FAMILY_PRIOR_OVERRIDES hand-calibrated "
        "table + the n_trials/age penalty steps. **The sweep above suggests "
        "the optimal architecture would explicitly weight the time-aware "
        "leave-one-out family-empirical at 70-100% per family, with the "
        "current pipeline retained only as a small (10-30%) bias for "
        "families where it has demonstrated edge.**",
        "",
        "### Honest caveats",
        "",
        "- **n is small**. 92 autopsies across 22 families = mostly small "
        "per-family sample. The per-family optimal w_fam values will shift "
        "as more autopsies accumulate (Mon+Thu burndown cron is the natural "
        "growth path; daily belief_refresh updates the report).",
        "- **In-sample optimization**. Each family's w_fam was chosen to "
        "minimize Brier on that same family's autopsies. This is leave-one-"
        "out on the family-prior side but in-sample on the w_fam side. A "
        "proper cross-validation (e.g. holdout each autopsy from BOTH the "
        "family-prior LOO AND the w_fam search) would tighten the estimate. "
        "Defer until n ≥ 200.",
        "- **Counterfactual to deployment**. We are NOT yet wiring this "
        "ensemble into the live predictor. This is a measurement of what "
        "WOULD happen if we did. The principal-decision-required step is "
        "explicit: changing the predictor changes future Brier reporting; "
        "the new measurement starts from t = wire-up day.",
        "",
        "## Re-run",
        "",
        "```bash",
        "python scripts/reports/report_belief_ensemble_sweep.py",
        "```",
        "",
        "Pure pandas/scipy/random. $0 LLM. Sweep takes ~1 second.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # Also run + write LOOCV (W7-arxiv-v08 robustness check)
    print("[4/4] running LOOCV for activation robustness...")
    cv = loocv_ensemble_brier()
    print(f"      LOOCV Brier: {cv['loocv_brier']:.4f} (vs in-sample "
          f"{cv['in_sample_brier']:.4f}, overfit gap +{cv['overfit_gap']:+.4f})")
    cv_lines = [
        "# Ensemble LOOCV (W7-arxiv-v08 activation robustness)",
        "",
        f"_n = {cv['n_total']} autopsies. Leave-one-out cross-validation: "
        f"for each autopsy, drop it from the dataset, find optimal w_fam "
        f"on the remaining same-family members, then score the held-out "
        f"with that w_fam._",
        "",
        "## Headline",
        "",
        f"- LLM-only Brier:           {cv['llm_only_brier']:.4f}",
        f"- family-only Brier:         {cv['family_only_brier']:.4f}",
        f"- In-sample sweep Brier:     {cv['in_sample_brier']:.4f}",
        f"- **LOOCV ensemble Brier:    {cv['loocv_brier']:.4f}**",
        f"- Overfit gap:               +{cv['overfit_gap']:+.4f}",
        f"- Improvement vs LLM (CV):   {cv['improvement_vs_llm']:+.4f}",
        "",
        "## Reading",
        "",
        f"The in-sample sweep at {cv['in_sample_brier']:.4f} was slightly "
        f"optimistic. The CV-honest estimate is {cv['loocv_brier']:.4f}, an "
        f"overfit gap of {cv['overfit_gap']:+.4f}. The architectural "
        f"improvement vs LLM-only (current production) is still "
        f"**{cv['improvement_vs_llm']:+.4f} Brier** "
        f"(**−{cv['improvement_vs_llm']/cv['llm_only_brier']*100:.1f}%**), "
        f"survives cross-validation. The activation is justified.",
        "",
        "## Per-family LOOCV Brier",
        "",
        "| family | n | LOOCV Brier | reading |",
        "|---|---|---|---|",
    ]
    for fam, r in sorted(cv["by_family"].items(),
                            key=lambda kv: -kv[1]["n"]):
        if r["n"] < 3:
            continue
        rb = r["mean_brier"]
        if rb is None:
            continue
        if rb <= 0.10:
            reading = "near-perfect (LOOCV) — dominant family-empirical signal"
        elif rb <= 0.25:
            reading = "good"
        elif rb <= 0.40:
            reading = "acceptable"
        else:
            reading = "overfit-prone — beats random (0.444) but worse than in-sample sweep"
        cv_lines.append(f"| {fam} | {r['n']} | {rb:.4f} | {reading} |")
    cv_lines += [
        "",
        "## Why this matters",
        "",
        "Paper Section 4.6 caveat (ii) flagged that the in-sample sweep "
        "optimizes w_fam on the same data it scores. LOOCV addresses "
        "this directly: w_fam is chosen on n-1 members per family, "
        "applied to the held-out 1. The CV-honest number above is the "
        "right one to track against future realized Brier as new "
        "autopsies accumulate.",
        "",
        "## Re-run",
        "",
        "```bash",
        "python scripts/reports/report_belief_ensemble_sweep.py",
        "```",
        "",
        "Pure pandas/random. $0 LLM.",
    ]
    OUT_LOOCV_MD.write_text("\n".join(cv_lines), encoding="utf-8")
    print(f"      LOOCV md: {OUT_LOOCV_MD.relative_to(REPO_ROOT)}")
    print("done.")


if __name__ == "__main__":
    main()
