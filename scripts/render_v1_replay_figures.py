"""scripts/render_v1_replay_figures.py

Render the 3 publication-grade static figures for the 2026-06-03
v1_replay analysis docs (TC ablation / regime decomposition /
per-sleeve FF5+MOM α).

Output:
  docs/figures/tc_ablation_v1_2026-06-03.png
  docs/figures/regime_sharpe_v1_2026-06-03.png
  docs/figures/per_sleeve_alpha_t_v1_2026-06-03.png

Style: clean, single-color where possible, no chart-junk, no emoji,
no 3D, no gradients. Bloomberg-terminal / FT-style.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Common style
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

# ───────────────────────── FIG 1: TC ablation curve ──────────────────
def render_tc_ablation() -> Path:
    src = REPO / "data" / "tc_ablation_v1_2026-06-03.json"
    d = json.load(open(src))
    rows = d["results"]
    tcs    = [r["tc_bps_one_way"]      for r in rows]
    sharpe = [r["sharpe_annualized"]   for r in rows]
    drag   = [r["annual_drag_pct"]*100 for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # Sharpe line
    ax.plot(tcs, sharpe, marker="o", color="#1a4d8c", linewidth=2,
            markersize=7, label="Sharpe (annualized)")
    for x, y in zip(tcs, sharpe):
        ax.annotate(f"{y:+.2f}", xy=(x, y), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=9, color="#1a4d8c")

    # Bands — institutional / retail / broken
    ax.axvspan(0,  15,  alpha=0.08, color="#1a8c3a", label="Institutional band")
    ax.axvspan(15, 60,  alpha=0.08, color="#c08a1a", label="Retail band")
    ax.axvspan(60, 105, alpha=0.08, color="#a33333", label="Uninvestable")

    # HLZ-equivalent guide at Sharpe = 1.0
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(102, 1.02, "Sharpe = 1.0", fontsize=8, color="gray",
            ha="right", va="bottom")

    ax.set_xlabel("Transaction cost (one-way, basis points)")
    ax.set_ylabel("Sharpe (annualized)")
    ax.set_title("TC ablation · v1 combined book · 2014-09 → 2023-12")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, max(sharpe) * 1.25)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # Subtitle: turnover assumption
    eff = d["effective_annual_turnover"] * 100
    fig.text(0.50, -0.02,
             f"Effective book turnover = {eff:.0f}%/yr one-way (cadence-anchored, equal-weight 4 sleeves).",
             ha="center", fontsize=8, style="italic", color="#555")

    out = FIG / "tc_ablation_v1_2026-06-03.png"
    fig.savefig(out)
    plt.close(fig)
    return out

# ───────────────────────── FIG 2: Regime Sharpe bar chart ────────────
def render_regime_sharpe() -> Path:
    src = REPO / "data" / "regime_decomposition_v1_2026-06-03.json"
    rows = json.load(open(src))
    labels = [r["label"]              for r in rows]
    sharpes= [r["sharpe_annualized"]  for r in rows]
    n_obs  = [r["n_obs"]              for r in rows]

    # Color: red if negative, green if positive
    colors = ["#a33333" if s < 0 else "#1a8c3a" for s in sharpes]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, sharpes, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # Annotate Sharpe value + n
    for i, (s, n) in enumerate(zip(sharpes, n_obs)):
        x_offset = 0.05 if s >= 0 else -0.05
        ha       = "left" if s >= 0 else "right"
        ax.text(s + x_offset, i, f"{s:+.2f}  (n={n})",
                va="center", ha=ha, fontsize=8.5,
                color="#222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(1.0, -0.5, "Sharpe = 1.0", fontsize=8, color="gray",
            ha="center", va="bottom")

    ax.set_xlabel("Sharpe (annualized)")
    ax.set_title("Regime decomposition · v1 combined book · 9 named windows")
    ax.set_xlim(min(sharpes) * 1.4, max(sharpes) * 1.4)

    fig.text(0.50, -0.02,
             "+ in every crisis (2018-Q4, 2020-COVID, 2022) ex Q1-2018 vol-mageddon. "
             "GFC-2008 out-of-sample (data starts 2014).",
             ha="center", fontsize=8, style="italic", color="#555")

    out = FIG / "regime_sharpe_v1_2026-06-03.png"
    fig.savefig(out)
    plt.close(fig)
    return out

# ─────────────── FIG 3: Per-sleeve α vs t_NW scatter w/ HLZ line ─────
def render_per_sleeve_scatter() -> Path:
    src = REPO / "data" / "ff5_mom_per_sleeve_v1_2026-06-03.json"
    d = json.load(open(src))

    # Combined book reference (from prior FF5+MOM doc)
    combined = {
        "label": "COMBINED",
        "alpha_annualized": 0.06279,
        "alpha_tstat_NW":   3.276,
        "alpha_t_clears_HLZ": True,
    }

    points = []
    for name, r in d.items():
        points.append({
            "label": name,
            "alpha_annualized": r["alpha_annualized"],
            "alpha_tstat_NW":   r["alpha_tstat_NW"],
            "alpha_t_clears_HLZ": r["alpha_t_clears_HLZ"],
        })
    points.append(combined)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for p in points:
        x = p["alpha_annualized"] * 100
        y = p["alpha_tstat_NW"]
        is_combined = (p["label"] == "COMBINED")
        is_hlz      = p["alpha_t_clears_HLZ"]
        face   = "#1a8c3a" if is_hlz else "#888888"
        edge   = "black" if is_combined else face
        marker = "D" if is_combined else "o"
        size   = 200 if is_combined else 140
        ax.scatter(x, y, s=size, c=face, edgecolors=edge, linewidth=1.5,
                   marker=marker, zorder=5, alpha=0.9)
        # Label offset — COMBINED upper-left into safe quadrant; others upper-right
        if is_combined:
            ox, oy, ha, va = -12, 16, "right", "bottom"
        else:
            ox, oy, ha, va = 10, 8, "left", "bottom"
        ax.annotate(p["label"], xy=(x, y), xytext=(ox, oy),
                    textcoords="offset points", fontsize=9.5,
                    ha=ha, va=va,
                    fontweight="bold" if is_combined else "normal")

    # HLZ |t|>=3 horizontal line + shaded "above bar" region
    ax.axhline(3.0, color="black", linestyle="-", linewidth=1.2)
    ax.text(0.5, 3.1, "HLZ 2016 bar: |t| ≥ 3", fontsize=9,
            color="black", va="bottom")
    ax.axhspan(3.0, 12, alpha=0.05, color="#1a8c3a")

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Alpha (% per year, annualized)")
    ax.set_ylabel("t-statistic (Newey-West HAC, 6 lags)")
    ax.set_title("Per-sleeve α vs t-stat · v1 (FF5+MOM regression)")
    ax.set_xlim(-1, 12)
    ax.set_ylim(0, 5)

    # Legend
    legend_handles = [
        plt.scatter([], [], s=140, c="#1a8c3a", marker="o", edgecolors="#1a8c3a",
                    label="Clears HLZ (|t|≥3)"),
        plt.scatter([], [], s=140, c="#888888", marker="o", edgecolors="#888888",
                    label="Sub-HLZ"),
        plt.scatter([], [], s=200, c="#1a8c3a", marker="D", edgecolors="black",
                    label="Combined book"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=8)

    fig.text(0.50, -0.02,
             "D_PEAD is the only sleeve clearing HLZ standalone (α=+7.6%/yr, t=+3.37). "
             "Combined book inherits significance from D_PEAD + diversification.",
             ha="center", fontsize=8, style="italic", color="#555")

    out = FIG / "per_sleeve_alpha_t_v1_2026-06-03.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for fn in (render_tc_ablation, render_regime_sharpe, render_per_sleeve_scatter):
        path = fn()
        print(f"  wrote {path.relative_to(REPO)}")
