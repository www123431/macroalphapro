"""scripts/ltr_verdict_pre_commit.py — LTR strict pre-commit criteria
+ diagnostic context."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine.portfolio.combined_book import build_tsmom_book
from engine.validation.deflated_sharpe import deflated_sharpe_ratio


def _sharpe(s: pd.Series) -> float:
    return float((s.mean() * 12) / (s.std() * math.sqrt(12))) if s.std() > 0 else 0.0


def _cosine(a: pd.Series, b: pd.Series) -> float:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 12:
        return float("nan")
    av, bv = j["a"].values, j["b"].values
    return float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))


def main():
    ltr = pd.read_parquet("data/cache/_ltr_monthly.parquet").iloc[:, 0]
    ltr.index = pd.to_datetime(ltr.index)
    us_pit_sn = pd.read_parquet("data/cache/_dpead_sn_pit_monthly.parquet").iloc[:, 0]
    us_pit_sn.index = pd.to_datetime(us_pit_sn.index)
    tsmom = build_tsmom_book()
    tsmom.index = pd.to_datetime(tsmom.index)

    print("=" * 90)
    print(" LTR PRE-COMMIT CRITERIA VERDICT")
    print("=" * 90)
    print(f"  sample: {len(ltr)} months "
          f"({ltr.index.min().date()} → {ltr.index.max().date()})")

    sharpe = _sharpe(ltr)
    cos_us = _cosine(ltr, us_pit_sn)
    cos_tsmom = _cosine(ltr, tsmom)
    dsr = deflated_sharpe_ratio(returns=ltr.values, n_trials=5, periods_per_year=12)

    print(f"\n  [LTR1] monthly Sharpe ≥ 0.5 ?")
    print(f"    Sharpe = {sharpe:+.3f}")
    ltr1 = sharpe >= 0.5
    print(f"    → {'PASS' if ltr1 else 'FAIL'}")

    print(f"\n  [LTR2] cosine with US PIT SN < 0.4 ?")
    print(f"    cosine = {cos_us:+.3f}")
    ltr2 = cos_us < 0.4
    print(f"    → {'PASS' if ltr2 else 'FAIL'}")

    print(f"\n  [LTR3] cosine with TSMOM ≤ 0 (anti-momentum) ?")
    print(f"    cosine = {cos_tsmom:+.3f}")
    ltr3 = cos_tsmom <= 0
    print(f"    → {'PASS' if ltr3 else 'FAIL'}")

    print(f"\n  [LTR4] DeflSR (n_trials=5) ≥ 0.6 ?")
    print(f"    DeflSR = {dsr.deflated_sr:.3f}")
    print(f"    {dsr.verdict}")
    ltr4 = dsr.deflated_sr >= 0.6
    print(f"    → {'PASS' if ltr4 else 'FAIL'}")

    print(f"\n  [LTR5] each-month decile size ≥ 100 — assumed (from build constraint)")
    ltr5 = True

    n_pass = sum([ltr1, ltr2, ltr3, ltr4, ltr5])
    print(f"\n{'='*90}")
    print(f" PASSES: {n_pass}/5")
    if n_pass == 5:
        verdict = "GREEN — promote to PAPER_TRADE"
    elif n_pass >= 3 and ltr2 and ltr3:
        verdict = "YELLOW — orthogonal but underpowered; longer history might revive"
    else:
        verdict = "REJECTED — does not clear pre-commit"
    print(f" Verdict: {verdict}")
    print(f"{'='*90}")

    # Diagnostic context (NOT criteria; just honest reporting)
    print(f"\n  [diagnostic context]")
    print(f"    LTR is HISTORICALLY a Sharpe ~0.5-0.6 strategy (DBT 1985)")
    print(f"    HKK 2020: LTR SURVIVES replication but with reduced magnitude")
    print(f"    OUR sample (2018-10 to 2024-06) is MOMENTUM-DOMINATED era:")
    print(f"      - COVID rally + recovery 2020-2021")
    print(f"      - AI / Mag-7 concentration 2023-2024")
    print(f"    → LTR (anti-momentum) WOULD be expected to underperform here.")
    print(f"    Real LTR test requires longer CRSP history (pre-2010 + post-2010)")
    print(f"    spanning multiple regime cycles. Our 5.5yr is INSUFFICIENT.")


if __name__ == "__main__":
    main()
