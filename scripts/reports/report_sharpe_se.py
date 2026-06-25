"""Lo (2002) standard error + 95% confidence interval for the reported
Sharpe ratios in the 4-sleeve canonical replay.

Reference:
    Lo, A. W. (2002). "The Statistics of Sharpe Ratios."
    Financial Analysts Journal 58(4), 36-52.

The IID-returns formula (Eq 3 in Lo 2002):
    Var(SR_p) = (1 + SR_p^2 / 2) / T
where SR_p is the per-period Sharpe and T is observation count.

For an annualized Sharpe at periods/year = q:
    Var(SR_annual) = q * Var(SR_p) = q/T + SR_annual^2 / (2T)

This script uses the IID version, which is the standard quant-interview
reference. For autocorrelation-adjusted SE (Mertens 2002 / Christie 2005)
we would need the underlying weekly returns series; the replay JSON
ships per-period summary metrics only.

Output:
    data/portfolio_replay/sharpe_se.md
    data/portfolio_replay/sharpe_se.json

Reproduces: arxiv preprint §A.1 SE column.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


T_WEEKS = 486          # weeks in the canonical 4-sleeve replay window
PERIODS_PER_YEAR = 52  # weekly frequency

# Source: data/portfolio_replay/v1_combined_replay_verdict.json (private)
# Hard-coded here because the replay JSON is excluded from public snapshot.
# Update both when the canonical replay is re-generated.
REPORTED_SHARPES = [
    ("Combined (4-sleeve replay)", 1.3165),
    ("K1_BAB",                     0.7624),
    ("D_PEAD",                     0.9312),
    ("PATH_N",                     0.7290),
    ("CTA_PQTIX",                  0.4298),
]

OUT_MD   = REPO_ROOT / "data" / "portfolio_replay" / "sharpe_se.md"
OUT_JSON = REPO_ROOT / "data" / "portfolio_replay" / "sharpe_se.json"


def lo_2002_se(sr_annual: float, T_periods: int, periods_per_year: int) -> float:
    """Standard error of annualized Sharpe ratio under IID returns."""
    q = periods_per_year
    var_ann = q / T_periods + (sr_annual ** 2) / (2 * T_periods)
    return math.sqrt(var_ann)


def _normal_sf(z: float) -> float:
    """Survival function of standard normal — 1 - Phi(z)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def main() -> None:
    out: dict = {
        "method":            "Lo (2002) Eq 3, IID returns",
        "T_periods":         T_WEEKS,
        "periods_per_year":  PERIODS_PER_YEAR,
        "years_effective":   round(T_WEEKS / PERIODS_PER_YEAR, 2),
        "rows": [],
    }
    rows_md = []
    for label, sr in REPORTED_SHARPES:
        se = lo_2002_se(sr, T_WEEKS, PERIODS_PER_YEAR)
        lo, hi = sr - 1.96 * se, sr + 1.96 * se
        t = sr / se
        p = _normal_sf(t)
        out["rows"].append({
            "series":      label,
            "sharpe":      round(sr, 4),
            "se":          round(se, 4),
            "ci_95_lo":    round(lo, 4),
            "ci_95_hi":    round(hi, 4),
            "t_stat":      round(t, 4),
            "p_one_sided": round(p, 6),
            "significant_5pct": p < 0.05,
        })
        rows_md.append(
            f"| {label} | {sr:.4f} | {se:.4f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{t:.2f} | {p:.4f} | {'✓' if p < 0.05 else '—'} |"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")

    md = [
        "# Sharpe Ratio Standard Errors — 4-sleeve canonical replay",
        "",
        f"_Lo (2002) Eq 3, IID returns. T = {T_WEEKS} weekly observations "
        f"(~{T_WEEKS/PERIODS_PER_YEAR:.1f} years), q = {PERIODS_PER_YEAR}._",
        "",
        "| series | SR (ann) | SE | 95% CI | t-stat | p (SR > 0) | sig 5% |",
        "|---|---|---|---|---|---|---|",
        *rows_md,
        "",
        "## Reading",
        "",
        "- **Combined Sharpe 1.32 is t ≈ 4.0** — significantly > 0 at "
        "p < 0.0001 even though the CI is wide [0.67, 1.96]. "
        "9.4 years of weekly data is enough to reject SR=0, not enough "
        "to discriminate 1.0 from 1.5.",
        "- **CTA_PQTIX standalone (SR 0.43) is NOT significant at 5%** "
        "(p = 0.094); its contribution to the combined book comes from "
        "diversification (near-zero correlation with the other three), "
        "not from standalone alpha. Honest disclosure.",
        "- **All other sleeves significant at 5%** but with CIs that "
        "include values low enough to be uninteresting — single-sleeve "
        "Sharpes in the [0.7, 0.9] range that *could* be as low as 0.1 "
        "in expectation.",
        "",
        "## Caveat — IID assumption",
        "",
        "The Lo (2002) Eq 3 formula assumes IID returns. Real strategy "
        "returns have non-zero autocorrelation; Mertens (2002) and "
        "Christie (2005) give adjusted SEs that are typically 10-30% "
        "wider for positively-autocorrelated streams. The IID SE here "
        "is therefore a LOWER BOUND on the true SE — actual uncertainty "
        "around the point estimate is slightly larger than what's "
        "tabulated above.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/reports/report_sharpe_se.py",
        "```",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")

    # Echo the table to stdout (ASCII-safe — Windows gbk console can't
    # render the markdown's check mark, so we just print the data rows)
    print()
    print(f"  {'series':<28} {'SR':>7}  {'SE':>7}  {'95% CI':<20}  {'t':>5}  {'p':>7}")
    for row in out["rows"]:
        ci = f"[{row['ci_95_lo']:+.3f}, {row['ci_95_hi']:+.3f}]"
        sig = "*" if row["significant_5pct"] else " "
        print(f"  {row['series']:<28} {row['sharpe']:>7.4f}  "
              f"{row['se']:>7.4f}  {ci:<20}  {row['t_stat']:>5.2f}  "
              f"{row['p_one_sided']:>7.4f} {sig}")


if __name__ == "__main__":
    main()
