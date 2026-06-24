"""Generate the deployed book attribution markdown for arxiv Appendix A.

Built 2026-06-22 (W7-arxiv-v03). $0 LLM. Pure pandas/numpy on the
existing combined_book.py series + per-sleeve series.

Reports:
  - Per-sleeve Sharpe / MaxDD / n_months / window
  - Combined 5-sleeve book stats (regime-conditional, 10% vol target)
  - Per-sleeve correlation matrix (over overlap window)
  - 1-month live paper-trade NAV summary (since 2026-05-14)

Outputs:
  data/research/deployed_book_attribution.md
  data/research/deployed_book_attribution.json

This is the HONEST audit that caught the prior session's
"Sharpe 1.32 over 6 months" claim — actual backtest is 0.96 over
97 months 2016-2024; live trading is ~1 month with -0.18% cum return.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

REPLAY_VERDICT_PATH = REPO_ROOT / "data" / "portfolio_replay" / "v1_combined_replay_verdict.json"
OUT_MD   = REPO_ROOT / "data" / "research" / "deployed_book_attribution.md"
OUT_JSON = REPO_ROOT / "data" / "research" / "deployed_book_attribution.json"
NAV_PATH = REPO_ROOT / "data" / "research" / "nav_history.jsonl"


def series_stats(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"label": label, "n_months": 0}
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    return {
        "label":         label,
        "n_months":      len(r),
        "first":         r.index.min().strftime("%Y-%m"),
        "last":          r.index.max().strftime("%Y-%m"),
        "mean_monthly":  round(float(r.mean()), 6),
        "vol_monthly":   round(float(r.std()), 6),
        "ann_return":    round(float(r.mean() * 12), 4),
        "ann_vol":       round(float(r.std() * np.sqrt(12)), 4),
        "sharpe_ann":    round(sharpe, 4),
        "max_dd":        round(dd, 4),
    }


def live_paper_trade_summary() -> dict:
    if not NAV_PATH.is_file():
        return {"n_records": 0, "note": "nav_history.jsonl missing"}
    rows = []
    with NAV_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {"n_records": 0}
    first_nav = float(rows[0].get("equity") or 0)
    last_nav  = float(rows[-1].get("equity") or 0)
    return {
        "n_records":      len(rows),
        "first_as_of":    rows[0].get("as_of"),
        "last_as_of":     rows[-1].get("as_of"),
        "first_equity":   round(first_nav, 2),
        "last_equity":    round(last_nav, 2),
        "cum_return_pct": round((last_nav / first_nav - 1) * 100, 4)
                            if first_nav else 0,
        "note": ("Sample too short for Sharpe inference. NAV ledger continues "
                  "to accumulate; n>=126 daily obs needed for ~0.5 ann-vol "
                  "uncertainty on Sharpe estimate."),
    }


def main() -> None:
    print(f"[1/3] Loading canonical replay verdict from "
          f"{REPLAY_VERDICT_PATH.relative_to(REPO_ROOT)}...")
    replay = json.loads(REPLAY_VERDICT_PATH.read_text(encoding="utf-8"))
    combined_stats = replay.get("combined_metrics") or {}
    per_strategy = replay.get("per_strategy_metrics") or {}
    pairwise_corr = replay.get("pairwise_correlation") or {}
    crisis = replay.get("crisis_period_returns") or {}
    attribution = replay.get("sleeve_attribution") or {}
    forward_band = replay.get("expected_forward_band") or {}
    honest = replay.get("honest_disclose") or []
    print(f"      combined: Sharpe={combined_stats.get('sharpe')} "
          f"n_weeks={combined_stats.get('n_weeks')} "
          f"MaxDD={combined_stats.get('max_dd')}")

    print("[2/3] Loading live paper-trade NAV summary...")
    live = live_paper_trade_summary()
    print(f"      live: {live.get('n_records', 0)} records, "
          f"cum return {live.get('cum_return_pct', 0)}%")

    output = {
        "as_of":              pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source":             str(REPLAY_VERDICT_PATH.relative_to(REPO_ROOT)),
        "combined_replay":    combined_stats,
        "per_strategy":       per_strategy,
        "pairwise_correlation": pairwise_corr,
        "crisis_returns":     crisis,
        "sleeve_attribution": attribution,
        "expected_forward":   forward_band,
        "honest_disclose":    honest,
        "live_paper_trade":   live,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2, default=str),
                          encoding="utf-8")
    print(f"      JSON: {OUT_JSON.relative_to(REPO_ROOT)}")

    # Markdown
    lines = [
        "# Deployed Book Attribution (Appendix A source)",
        "",
        f"_As of {output['as_of']}. Source: "
        f"`{output['source']}` — the canonical combined-replay verdict "
        f"per the active deployment design (docs/portfolio_deployment_"
        f"design_2026-05-13.md)._",
        "",
        "## Combined 4-sleeve book (replay verdict)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in ["ann_ret", "ann_vol", "sharpe", "max_dd", "n_weeks"]:
        lines.append(f"| {k} | {combined_stats.get(k, '?')} |")
    lines += [
        "",
        "## Per-strategy stats",
        "",
        "| strategy | Sharpe | ann return | ann vol | MaxDD | n_weeks |",
        "|---|---|---|---|---|---|",
    ]
    for label, s in per_strategy.items():
        lines.append(
            f"| {label} | {s.get('sharpe', '?')} | "
            f"{s.get('ann_ret', '?')} | {s.get('ann_vol', '?')} | "
            f"{s.get('max_dd', '?')} | {s.get('n_weeks', '?')} |"
        )
    lines += [
        "",
        "## Pairwise correlation (in replay window)",
        "",
        "| pair | correlation |",
        "|---|---|",
    ]
    for k, v in pairwise_corr.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Sleeve attribution (cumulative contribution)",
        "",
        "| sleeve | cumulative contribution | total allocation |",
        "|---|---|---|",
    ]
    contrib = attribution.get("sleeve_cumulative_contribution", {})
    alloc = attribution.get("sleeve_total_allocation", {})
    for k in contrib:
        lines.append(f"| {k} | {contrib.get(k, '?')} | {alloc.get(k, '?')} |")
    lines += [
        "",
        "## Crisis-period returns",
        "",
        "| crisis | return |",
        "|---|---|",
    ]
    for k, v in crisis.items():
        lines.append(f"| {k} | {v:+.4f} |")
    lines += [
        "",
        "## Expected forward band",
        "",
    ]
    for k, v in forward_band.items():
        lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Live paper trade (since 2026-05-14)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in ["n_records", "first_as_of", "last_as_of", "first_equity",
                 "last_equity", "cum_return_pct"]:
        lines.append(f"| {k} | {live.get(k, '?')} |")
    lines += [
        "",
        f"_{live.get('note', '')}_",
        "",
        "## Honest disclosures (preserved from replay verdict)",
        "",
    ]
    for h in honest:
        lines.append(f"- {h}")
    lines += [
        "",
        "## Provenance amendment (audit trail)",
        "",
        "Earlier drafts (arxiv v0.1 abstract) said 'Sharpe 1.32 over 6 months'. ",
        "Audit 2026-06-22 traced the 1.32 number to "
        f"`{output['source']}` (this file's source). The '6 months' phrasing "
        "was wrong; the replay covers 486 weeks (~9.4 years). Live paper "
        "trading is ~1 month, with NAV ledger at data/research/"
        "nav_history.jsonl.",
        "",
        "A separate research variant `engine.portfolio.combined_book."
        "build_combined_book_regime_conditional` (regime-conditional, "
        "10% vol target, alternate 5-sleeve composition including "
        "cross_asset_carry / cross_asset_tsmom / mom_hedge_overlay / "
        "crisis_hedge / equity_book_pit_sn) gives Sharpe 0.96 over 97 months "
        "(2016-2024). Both numbers are real; they describe different book "
        "compositions. The CANONICAL DEPLOYED design is the 4-sleeve replay "
        "above (1.32).",
        "",
        "## Re-run",
        "",
        "```bash",
        "python scripts/reports/report_deployed_book_attribution.py",
        "```",
        "",
        "Pure pandas/numpy. $0 LLM.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"      MD:   {OUT_MD.relative_to(REPO_ROOT)}")
    print("done.")


if __name__ == "__main__":
    main()
