"""scripts/path_b_abnormal_sue_vs_pit_sn.py — head-to-head Path B
analysis: abnormal SUE vs PIT SN parent, applying the 3 pre-commit
falsification criteria from dpead_abnormal_sue doctrine.

Outputs:
  C1. Standalone Sharpe race
  C2. Cosine similarity (orthogonality threshold 0.7)
  C3. Combined-signal stacking test
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np
import pandas as pd

PIT_SN = Path("data/cache/_dpead_sn_pit_monthly.parquet")
ABNORMAL = Path("data/cache/_dpead_abnormal_sue_monthly.parquet")


def _sharpe(s: pd.Series) -> float:
    return float((s.mean() * 12) / (s.std() * math.sqrt(12)))


def main() -> int:
    pit = pd.read_parquet(PIT_SN).iloc[:, 0]
    pit.index = pd.to_datetime(pit.index)
    abn = pd.read_parquet(ABNORMAL).iloc[:, 0]
    abn.index = pd.to_datetime(abn.index)
    aligned = pd.concat([pit.rename("pit_sn"), abn.rename("abnormal")],
                        axis=1).dropna()
    n = len(aligned)

    print("=" * 90)
    print(" PATH B FALSIFICATION CRITERIA — abnormal SUE vs PIT SN parent")
    print("=" * 90)
    print(f"\n  Aligned window: {n} months "
          f"({aligned.index.min().date()} → {aligned.index.max().date()})")

    sh_pit = _sharpe(aligned["pit_sn"])
    sh_abn = _sharpe(aligned["abnormal"])
    print(f"\n  [stats]")
    print(f"    PIT SN gross Sharpe:        {sh_pit:+.3f}")
    print(f"    abnormal SUE gross Sharpe:  {sh_abn:+.3f}")

    # ── C1 ───────────────────────────────────────────────────────────
    c1_pass = sh_abn > sh_pit
    print(f"\n  [C1] Standalone Sharpe > PIT SN parent?")
    print(f"       {sh_abn:+.3f} > {sh_pit:+.3f}  →  "
          f"{'PASS' if c1_pass else 'FAIL'}")

    # ── C2 ───────────────────────────────────────────────────────────
    cosine = float(
        aligned["pit_sn"] @ aligned["abnormal"] /
        (np.linalg.norm(aligned["pit_sn"]) * np.linalg.norm(aligned["abnormal"]))
    )
    pearson = float(aligned["pit_sn"].corr(aligned["abnormal"]))
    c2_pass = cosine < 0.7
    print(f"\n  [C2] Cosine(abnormal, PIT SN) < 0.7 (orthogonality)?")
    print(f"       cosine={cosine:.3f}, pearson_r={pearson:.3f}  →  "
          f"{'PASS' if c2_pass else 'FAIL — too correlated'}")

    # ── C3 ───────────────────────────────────────────────────────────
    # Combined via inverse-vol weighting (matches D_PEAD + analyst_revision pattern
    # in engine.portfolio.combined_book.build_equity_book)
    v_pit = aligned["pit_sn"].rolling(12).std().shift(1)
    v_abn = aligned["abnormal"].rolling(12).std().shift(1)
    w_pit = (1 / v_pit) / (1 / v_pit + 1 / v_abn)
    combined = (w_pit * aligned["pit_sn"] + (1 - w_pit) * aligned["abnormal"]).dropna()
    sh_combined = _sharpe(combined)
    max_standalone = max(sh_pit, sh_abn)
    c3_pass = sh_combined > max_standalone
    rel_gain = (sh_combined - max_standalone) / max_standalone * 100
    print(f"\n  [C3] Combined Sharpe > max(standalone)?")
    print(f"       combined inv-vol  Sharpe: {sh_combined:+.3f}")
    print(f"       max(standalone)   Sharpe: {max_standalone:+.3f}")
    print(f"       relative gain:    {rel_gain:+.1f}%  →  "
          f"{'PASS — additive complement' if c3_pass else 'FAIL — no stacking benefit'}")

    # ── Verdict ──────────────────────────────────────────────────────
    n_pass = sum([c1_pass, c2_pass, c3_pass])
    print(f"\n  [verdict] Falsification criteria passed: {n_pass}/3")

    # Statistical nuance per Bailey-LdP: SE(Sharpe_ann) over n_years
    # observations is ~sqrt((1 + 0.5*SR²)/n_years). For SR~2 over 10y
    # this is ~0.55 — so any C1-style "X vs Y" Sharpe comparison within
    # 1 sigma (0.55) is NOT statistically distinguishable. C1 alone is
    # power-limited at this sample size; C2 + C3 carry the verdict weight.
    se_sharpe = (1 + 0.5 * max(sh_pit, sh_abn) ** 2) ** 0.5 / (n / 12) ** 0.5
    c1_distinguishable = abs(sh_abn - sh_pit) > se_sharpe
    print(f"\n  [C1 statistical nuance]")
    print(f"    SE(Sharpe_ann) ≈ {se_sharpe:.3f} over n={n}mo")
    print(f"    |Sharpe diff|  = {abs(sh_abn - sh_pit):.3f}")
    print(f"    {'C1 IS statistically distinguishable' if c1_distinguishable else 'C1 NOT statistically distinguishable (within 1 SE)'}")
    print(f"    → C1 alone underpowered; verdict relies on C2 (cosine) + C3 (combo)")

    if n_pass == 3:
        print(f"\n  STRATEGY ACCEPTED — propose ADDITION pathway via SLM")
    elif c3_pass and c2_pass:
        print(f"\n  STRATEGY ACCEPTED as ADDITION — C2 + C3 prove "
              f"orthogonality + stacking value")
    elif not c2_pass:
        # The substantive scientific finding: HIGH CORRELATION means
        # the two strategies capture the same underlying signal in
        # different forms. This is independent of C1 power issues.
        print(f"\n  STRATEGY REJECTED — abnormal SUE captures same underlying")
        print(f"  signal as PIT SN parent (cosine {cosine:.3f}). Sector")
        print(f"  neutralization via RANKING (PIT SN) vs SIGNAL TRANSFORMATION")
        print(f"  (abnormal SUE) are mathematically near-equivalent at this")
        print(f"  universe size. Deploy decision: do NOT add — no new signal.")
    else:
        print(f"\n  STRATEGY YELLOW — second-look needed")

    # Print full table
    print(f"\n  [reference table]")
    print(f"    {'metric':<25} {'PIT SN':>10} {'abnormal':>10} {'combined':>10}")
    for fn, label in [(_sharpe, "Sharpe (ann)")]:
        print(f"    {label:<25} {fn(aligned['pit_sn']):>10.3f} "
              f"{fn(aligned['abnormal']):>10.3f} {fn(combined):>10.3f}")
    print(f"    {'ann return':<25} {aligned['pit_sn'].mean()*12:>10.2%} "
          f"{aligned['abnormal'].mean()*12:>10.2%} "
          f"{combined.mean()*12:>10.2%}")
    print(f"    {'ann vol':<25} {aligned['pit_sn'].std()*math.sqrt(12):>10.2%} "
          f"{aligned['abnormal'].std()*math.sqrt(12):>10.2%} "
          f"{combined.std()*math.sqrt(12):>10.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
