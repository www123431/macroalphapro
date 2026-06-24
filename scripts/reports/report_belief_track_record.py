"""Generate the belief layer track-record report.

Pipeline:
  1. backfill_all() — refresh autopsies.jsonl from any new prediction↔
     verdict_event pairs (idempotent; no-op if nothing new)
  2. build_track_record() — aggregate Brier + family + reliability
  3. Write data/research/belief_track_record.json (machine-readable)
  4. Write data/research/belief_track_record.md  (human + showcase surface)

Run:  python scripts/reports/report_belief_track_record.py

This is the FIRST public-facing artifact of the belief layer — it
answers "does the predictor know what it's doing?" against 85+
realized verdicts. Run weekly; the markdown is suitable for inclusion
in /research UI surface OR external sharing (arxiv supplement).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.research.belief_autopsy import backfill_all
from engine.research.belief_track_record import build_track_record


OUT_JSON = REPO_ROOT / "data" / "research" / "belief_track_record.json"
OUT_MD   = REPO_ROOT / "data" / "research" / "belief_track_record.md"


def _format_brier_label(brier: float) -> str:
    """Senior-quant interpretation of a Brier score (3-class)."""
    if brier <= 0.15:
        return "excellent"
    if brier <= 0.30:
        return "good"
    if brier <= 0.40:
        return "acceptable"
    if brier <= 4.0 / 9.0:
        return "marginal (<= random baseline)"
    return "worse than random - investigate"


def render_markdown(tr: dict) -> str:
    n = tr["n_autopsies"]
    if n == 0:
        return ("# Belief Track Record\n\n"
                "No autopsies yet. Run backfill_all + accumulate verdicts.\n")
    mean = tr["mean_brier_overall"]
    lines: list[str] = [
        "# Belief Layer — Track Record",
        "",
        f"_As of {tr['as_of']}. n = {n} autopsies. "
        f"Mean Brier = **{mean:.4f}** ({_format_brier_label(mean)})._",
        "",
        "## What this measures",
        "",
        "Every FORWARD pipeline dispatch (factor verdict) is preceded by a "
        "prediction — a probability distribution over GREEN/MARGINAL/RED, "
        "logged BEFORE the statistical gate runs. The predictor is "
        "architecturally air-gapped from the verdict pipeline ("
        "no `engine.research.belief` import from any lens/template/gate). "
        "After the verdict lands, `belief_autopsy` joins prediction ↔ "
        "verdict and computes a Brier component per realization.",
        "",
        "**Brier component** = `(1 - p(actual_verdict))²`. Lower is better.",
        "",
        "| baseline | Brier |",
        "|---|---|",
        "| random uniform (1/3,1/3,1/3) | 0.444 |",
        "| perfect calibration            | 0.000 |",
        f"| **observed**                   | **{mean:.4f}** |",
        "",
        f"Observed Brier is **{(0.444 - mean) / 0.444:.1%}** "
        "better than a random-uniform baseline. Not impressive in "
        "absolute terms, but the predictor is honest enough to be "
        "tracked — which is the precondition for closed-loop "
        "improvement (belief phases 4-5).",
        "",
        "## Direction breakdown — calibration vs bias",
        "",
        "| direction | count | fraction |",
        "|---|---|---|",
    ]
    direction = tr["direction"]
    for label, count in sorted(direction["counts"].items(),
                                  key=lambda kv: -kv[1]):
        frac = direction["fractions"].get(label, 0.0)
        lines.append(f"| {label} | {count} | {frac:.1%} |")
    lines += [
        "",
        "**Senior reading**:",
        "- `well_calibrated` = predicted modal class matched actual",
        "- `over_predicted_green` / `over_predicted_red` = bias signals",
        "- if `over_predicted_green` ≫ `over_predicted_red`, the predictor "
        "is **optimistically biased** (classic LLM behavior; "
        "Tetlock 2015 calibration §)",
        "",
        "## Per-family Brier (sorted by sample size)",
        "",
        "| family | n | mean_brier | reading |",
        "|---|---|---|---|",
    ]
    for row in tr["family_breakdown"]:
        lines.append(
            f"| {row['family']} | {row['n']} | "
            f"{row['mean_brier']:.4f} | {_format_brier_label(row['mean_brier'])} |"
        )
    lines += [
        "",
        "**Worst families = highest-leverage improvement targets**. The "
        "predictor's prior for these families is mis-specified; either "
        "the family taxonomy is too coarse, or family-specific anchor "
        "evidence is missing from the prior.",
        "",
        "## Reliability diagram (modal-class confidence vs realized rate)",
        "",
        "When the predictor says `p(GREEN) = 0.6`, does GREEN actually "
        "happen 60% of the time? Perfect calibration → `observed_correct` "
        "tracks `mean_p_modal` along the y=x diagonal.",
        "",
        "| bin | n | mean p_modal | observed correct | calibration |",
        "|---|---|---|---|---|",
    ]
    for row in tr["reliability"]:
        if row["n"] == 0:
            cal = "—"
            p_str, c_str = "—", "—"
        else:
            p_str = f"{row['mean_p_modal']:.3f}"
            c_str = f"{row['observed_correct']:.3f}"
            gap = row["observed_correct"] - row["mean_p_modal"]
            if abs(gap) < 0.05:
                cal = "calibrated"
            elif gap > 0:
                cal = f"under-confident (+{gap:.2f})"
            else:
                cal = f"over-confident ({gap:.2f})"
        lines.append(
            f"| [{row['bin_lo']:.1f}, {row['bin_hi']:.1f}) | "
            f"{row['n']} | {p_str} | {c_str} | {cal} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Autopsy `ts` is the COMPUTATION time, not the realization time. "
        "All 85 autopsies were generated in one backfill pass, so "
        "sliding-window trend analysis is currently flat. Future "
        "predictions will accumulate over time and enable real trend.",
        "- Brier-only ≠ resolution. A predictor that always outputs "
        "(0.33, 0.34, 0.33) achieves Brier ≈ 0.45 (worse than ours), "
        "but a predictor that always outputs (0, 1, 0) when actual "
        "rate is 50% MARGINAL achieves Brier 0.25 (better than ours) "
        "while being uselessly over-confident. Reliability diagram "
        "(above) is the cross-check.",
        "- `superseded_by` rows excluded (BUG-1 spanning fix corrections).",
        "",
        "## Why this report exists",
        "",
        "Most quant research code is self-justifying — verdicts are "
        "written and never re-evaluated against the lab's own prior "
        "expectations. This report is the closed-loop honesty mechanism "
        "(Tetlock 2015 + López de Prado AFML Ch.13 + Bailey-LdP 2014 §5 "
        "prediction-vs-realization). Publishing the track record — "
        "including the failures — is what separates an epistemically-"
        "bounded research OS from another factor-zoo machine.",
        "",
        "## Files",
        "",
        "- Raw predictions: `data/research/predictions.jsonl`",
        "- Raw autopsies: `data/research/autopsies.jsonl`",
        "- Aggregated JSON: `data/research/belief_track_record.json`",
        "- This report: `data/research/belief_track_record.md`",
        "",
        "## Anchors",
        "",
        "- Brier 1950 — original Brier score definition",
        "- Tetlock 2015 *Superforecasting* Ch.5 — calibration vs resolution",
        "- López de Prado 2018 *AFML* Ch.13 — track-record interpretation",
        "- Bailey-López de Prado 2014 §5 — prediction-vs-realization auditing",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("[1/4] backfill_all - refreshing autopsies from new prediction/verdict pairs...")
    produced = backfill_all()
    print(f"      autopsies newly produced: {len(produced)}")

    print("[2/4] build_track_record - aggregating...")
    tr = build_track_record()
    print(f"      n_autopsies = {tr['n_autopsies']}, "
          f"mean Brier = {tr['mean_brier_overall']}")

    print(f"[3/4] writing {OUT_JSON.relative_to(REPO_ROOT)}...")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(tr, indent=2, default=str),
                          encoding="utf-8")

    print(f"[4/4] writing {OUT_MD.relative_to(REPO_ROOT)}...")
    OUT_MD.write_text(render_markdown(tr), encoding="utf-8")

    print("done.")


if __name__ == "__main__":
    main()
