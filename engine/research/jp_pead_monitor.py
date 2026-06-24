"""engine/research/jp_pead_monitor.py — Tier 1 monitoring-only sleeve
for JP PEAD. Tracks JP returns monthly to provide cross-market
robustness signal for US PIT SN; does NOT allocate capital.

Per 2026-05-31 verdict (commit c318a13): JP PEAD passed F1/F2/F3/F5
strongly but FAILED F4 (cosine 0.607 with US PIT SN). Deploying as
sleeve = wasted capacity (correlated signal). Monitoring-only gives
the robustness diagnostic at zero cost.

Diagnostic outputs:
  - JP vs US PIT SN trailing-3mo divergence
  - JP vs US PIT SN trailing-12mo cumulative gap
  - Alert thresholds (per Tier 1 doctrine):
    * |JP_3mo - US_3mo| > 1σ → flag potential US PEAD decay
    * JP 6mo consecutive NEG but US POS → JP-specific failure
    * BOTH 3mo consecutive NEG → behavioral PEAD family alarm
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
JP_RETURNS = REPO_ROOT / "data" / "cache" / "_jp_pead_monthly.parquet"
US_PIT_SN = REPO_ROOT / "data" / "cache" / "_dpead_sn_pit_monthly.parquet"


@dataclass
class JpUsDivergenceReport:
    n_aligned: int
    jp_trailing_3mo: float
    us_trailing_3mo: float
    diff_3mo: float
    diff_3mo_zscore: float
    diff_12mo_cum: float
    cosine_full: float
    alert_level: Literal["NONE", "INFO", "WARN", "CRITICAL"]
    alert_reason: str
    recommended_action: str


def evaluate_jp_us_divergence() -> JpUsDivergenceReport:
    """Run the divergence check; called by monthly tick + ad-hoc.

    Pure observational — no DB writes, no state changes. Caller
    decides what to do with alert_level.
    """
    jp = pd.read_parquet(JP_RETURNS).iloc[:, 0]
    jp.index = pd.to_datetime(jp.index)
    us = pd.read_parquet(US_PIT_SN).iloc[:, 0]
    us.index = pd.to_datetime(us.index)
    aligned = pd.concat([jp.rename("jp"), us.rename("us")], axis=1).dropna()
    n = len(aligned)
    if n < 12:
        return JpUsDivergenceReport(
            n_aligned=n, jp_trailing_3mo=0.0, us_trailing_3mo=0.0,
            diff_3mo=0.0, diff_3mo_zscore=0.0, diff_12mo_cum=0.0,
            cosine_full=float("nan"),
            alert_level="NONE",
            alert_reason="insufficient data (n<12)",
            recommended_action="continue accumulating data",
        )

    jp_3mo = float(aligned["jp"].tail(3).mean())
    us_3mo = float(aligned["us"].tail(3).mean())
    diff_3mo = jp_3mo - us_3mo

    # Z-score of 3mo diff: use historical |diff| std
    rolling_diff = aligned["jp"] - aligned["us"]
    diff_std = float(rolling_diff.std())
    diff_3mo_z = diff_3mo / diff_std if diff_std > 0 else 0.0

    diff_12mo_cum = float((1 + aligned["jp"].tail(12)).prod() -
                          (1 + aligned["us"].tail(12)).prod())

    import numpy as np
    jp_v = aligned["jp"].values
    us_v = aligned["us"].values
    cosine = float(jp_v @ us_v /
                   (np.linalg.norm(jp_v) * np.linalg.norm(us_v)))

    # Alert logic per Tier 1 doctrine
    jp_consec_neg = (aligned["jp"].tail(6) < 0).all()
    us_consec_neg = (aligned["us"].tail(6) < 0).all()
    both_3mo_neg = ((aligned["jp"].tail(3) < 0).all() and
                    (aligned["us"].tail(3) < 0).all())

    if both_3mo_neg:
        alert_level = "CRITICAL"
        alert_reason = ("US and JP PEAD BOTH negative for 3mo consecutive — "
                       "behavioral PEAD family alarm; investigate US sleeve "
                       "for decay")
        action = "review US PIT SN deploy decision; consider DECAY_WATCH"
    elif jp_consec_neg and not us_consec_neg:
        alert_level = "WARN"
        alert_reason = ("JP negative 6mo while US positive — JP-specific "
                       "failure (consider adding to graveyard)")
        action = "log JP-specific failure; do NOT promote JP PEAD"
    elif abs(diff_3mo_z) > 1.0:
        alert_level = "INFO"
        alert_reason = (f"JP-US 3mo divergence |z|={abs(diff_3mo_z):.2f} > 1.0 — "
                       f"watch but no action required")
        action = "log; continue monthly monitoring"
    else:
        alert_level = "NONE"
        alert_reason = "JP and US PEAD aligned; no divergence flag"
        action = "continue routine monthly tick"

    return JpUsDivergenceReport(
        n_aligned=n, jp_trailing_3mo=jp_3mo, us_trailing_3mo=us_3mo,
        diff_3mo=diff_3mo, diff_3mo_zscore=diff_3mo_z,
        diff_12mo_cum=diff_12mo_cum, cosine_full=cosine,
        alert_level=alert_level, alert_reason=alert_reason,
        recommended_action=action,
    )


def main():
    r = evaluate_jp_us_divergence()
    print("=" * 80)
    print(" JP PEAD MONITOR (Tier 1, capital-free robustness check)")
    print("=" * 80)
    print(f"  n_aligned:        {r.n_aligned} months")
    print(f"  JP trailing 3mo:  {r.jp_trailing_3mo:+.4f} ({r.jp_trailing_3mo*100:+.2f}%/mo)")
    print(f"  US trailing 3mo:  {r.us_trailing_3mo:+.4f} ({r.us_trailing_3mo*100:+.2f}%/mo)")
    print(f"  diff_3mo:         {r.diff_3mo:+.4f}")
    print(f"  diff_3mo z-score: {r.diff_3mo_zscore:+.3f}")
    print(f"  diff_12mo_cum:    {r.diff_12mo_cum:+.4f}")
    print(f"  cosine_full:      {r.cosine_full:+.3f}")
    print(f"\n  ALERT: {r.alert_level}")
    print(f"  reason: {r.alert_reason}")
    print(f"  action: {r.recommended_action}")


if __name__ == "__main__":
    main()
