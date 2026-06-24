"""Generate matplotlib figures for the belief track record + arxiv paper.

Built 2026-06-22 as v0.2 follow-up to docs/arxiv_preprint_draft_2026-06-22.md.

Three figures, all $0 LLM, all reproducible from existing data:
  Fig 1: Reliability diagram (modal-class confidence vs observed correct,
         with y=x diagonal for perfect calibration)
  Fig 2: Per-family Brier with bootstrap 95% CI bars + FDR-significant flag
  Fig 3: Brier vs 4 baselines (predictor, always-MARG, uniform, fair
         time-aware family-prior LOO)

Outputs:
  docs/figs/belief_fig1_reliability_diagram.png
  docs/figs/belief_fig2_family_brier_ci.png
  docs/figs/belief_fig3_baseline_comparison.png

Run:  python scripts/reports/report_belief_track_record_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from engine.research.belief_track_record_rigor import (
    BASELINE_RANDOM_3CLASS,
    run_all_rigor_tests,
)

OUT_DIR = REPO_ROOT / "docs" / "figs"


def fig1_reliability_diagram(rigor: dict) -> Path:
    """Modal-class confidence vs observed-correct rate, with diagonal."""
    t6 = rigor["T6_hosmer_lemeshow"]
    bins = t6.get("bin_details") or []
    populated = [(b["mean_p"], b["observed_correct"] / b["n"]
                    if b.get("n", 0) > 0 else None,
                    b.get("n", 0))
                   for b in bins if not b.get("skipped")]
    # bin_details has observed_correct as INT count; need to convert to rate
    # Recompute cleanly from the structure we have
    plot_points = []
    for b in bins:
        if b.get("skipped") or b.get("n", 0) == 0:
            continue
        mean_p = b.get("mean_p")
        observed_correct_count = b.get("observed_correct", 0)
        n = b.get("n", 0)
        observed_rate = observed_correct_count / n if n > 0 else 0
        plot_points.append((mean_p, observed_rate, n))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration (y=x)")
    if plot_points:
        xs = [p[0] for p in plot_points]
        ys = [p[1] for p in plot_points]
        sizes = [10 + 5 * p[2] for p in plot_points]
        sc = ax.scatter(xs, ys, s=sizes, c="#2c5aa0", alpha=0.7,
                         edgecolors="black", linewidth=0.5, zorder=3)
        for x, y, n in plot_points:
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                         xytext=(7, 4), fontsize=8)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("predicted probability of modal class")
    ax.set_ylabel("observed fraction correct in bin")
    ax.set_title(f"Fig 1. Reliability diagram (n={rigor['n_autopsies']})\n"
                  f"Hosmer-Lemeshow chi-square test: p = {t6.get('p_value', 'n/a')}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "belief_fig1_reliability_diagram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2_family_brier_ci(rigor: dict) -> Path:
    """Per-family Brier with bootstrap 95% CI bars; FDR-significant highlighted."""
    families = rigor["T4_per_family_fdr"]["families"]
    plot_fams = [f for f in families if f.get("ci_95_lo") is not None]
    plot_fams.sort(key=lambda f: f["mean_brier"])

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(plot_fams))))
    y_pos = np.arange(len(plot_fams))
    means = [f["mean_brier"] for f in plot_fams]
    los = [f["ci_95_lo"] for f in plot_fams]
    his = [f["ci_95_hi"] for f in plot_fams]
    errors = np.array([[m - lo, hi - m] for m, lo, hi in zip(means, los, his)]).T
    fdr_sig = [bool(f.get("fdr_significant_worse_than_baseline"))
                for f in plot_fams]
    colors = ["#c0392b" if s else "#2c5aa0" for s in fdr_sig]
    ax.barh(y_pos, means, xerr=errors, color=colors, alpha=0.7,
             edgecolor="black", linewidth=0.5, capsize=3)
    ax.axvline(BASELINE_RANDOM_3CLASS, color="black", linestyle="--",
                alpha=0.6, label=f"random baseline ({BASELINE_RANDOM_3CLASS:.3f})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{f['family']} (n={f['n']})" for f in plot_fams],
                         fontsize=8)
    ax.set_xlabel("mean Brier (bootstrap 95% CI)")
    ax.set_title(f"Fig 2. Per-family Brier (n={rigor['n_autopsies']}, "
                  f"FDR-significant-worse families in red)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    out = OUT_DIR / "belief_fig2_family_brier_ci.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig3_baseline_comparison(rigor: dict) -> Path:
    """Predictor vs 4 baselines, with bootstrap delta CIs."""
    comparisons = rigor["T2_baseline_comparison"]["comparisons"]
    t2_ta = rigor.get("T2_time_aware_family_prior", {})

    rows = []
    for c in comparisons:
        rows.append((c["baseline"], c["baseline_mean_brier"],
                       c["delta_ci_95_lo"], c["delta_ci_95_hi"]))
    # Add the fair time-aware family-prior
    if t2_ta and t2_ta.get("n"):
        rows.append((
            "family_prior\n(fair time-aware)",
            t2_ta["time_aware_family_prior_brier"],
            t2_ta["delta_ci_95_lo"],
            t2_ta["delta_ci_95_hi"],
        ))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    predictor_mean = rigor["T1_overall_brier_bootstrap"]["observed_mean"]

    labels = [r[0] for r in rows]
    baseline_means = [r[1] for r in rows]
    x = np.arange(len(rows))
    w = 0.35
    ax.bar(x - w/2, [predictor_mean] * len(rows), w,
            label=f"predictor (Brier {predictor_mean:.3f})",
            color="#2c5aa0", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, baseline_means, w, label="baseline",
            color="#888888", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axhline(BASELINE_RANDOM_3CLASS, color="red", linestyle="--",
                alpha=0.5, label=f"random ref ({BASELINE_RANDOM_3CLASS:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=12)
    ax.set_ylabel("mean Brier")
    ax.set_title(f"Fig 3. Predictor vs baselines (paired bootstrap, n={rigor['n_autopsies']})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Annotate delta significance
    for i, (label, baseline_m, lo, hi) in enumerate(rows):
        if lo < 0 and hi < 0:
            tag = "predictor better*"
            color = "green"
        elif lo > 0 and hi > 0:
            tag = "**predictor LOSES***"
            color = "darkred"
        else:
            tag = "n.s."
            color = "gray"
        ax.text(i, max(predictor_mean, baseline_m) + 0.02,
                 tag, ha="center", fontsize=8,
                 color=color, fontweight="bold")
    out = OUT_DIR / "belief_fig3_baseline_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    print("[1/4] running rigor tests...")
    rigor = run_all_rigor_tests()
    print(f"      n_autopsies = {rigor['n_autopsies']}")

    print("[2/4] generating Fig 1 (reliability diagram)...")
    f1 = fig1_reliability_diagram(rigor)
    print(f"      wrote {f1.relative_to(REPO_ROOT)}")

    print("[3/4] generating Fig 2 (per-family Brier CI)...")
    f2 = fig2_family_brier_ci(rigor)
    print(f"      wrote {f2.relative_to(REPO_ROOT)}")

    print("[4/4] generating Fig 3 (baseline comparison)...")
    f3 = fig3_baseline_comparison(rigor)
    print(f"      wrote {f3.relative_to(REPO_ROOT)}")
    print("done.")


if __name__ == "__main__":
    main()
