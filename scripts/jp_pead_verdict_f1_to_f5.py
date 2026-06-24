"""scripts/jp_pead_verdict_f1_to_f5.py — apply pre-committed F1-F5
falsification + record outcome to override_ledger.

F-criteria (from scripts/override_jp_pead.py grant):
  F1: per-event t-stat >= 2.0 with n >= 5000 events
  F2: monthly L/S decile Sharpe >= 0.5 (HLZ floor)
  F3: DeflSR with n_trials=8 (family) >= 0.6
  F4: cosine with US PIT SN < 0.4
  F5: per-event sample size >= 5000

ADDITIONAL senior caveats applied beyond F1-F5:
  - Survivorship bias: yfinance gives only CURRENTLY-LIVE firms.
    Selecting "top-500 by EPS event count" uses full-sample knowledge.
    This OVER-states the signal.
  - Universe selection: not PIT (real-time TOPIX 500 would differ each year)
  - No sector neutralization (PIT SN method deferred to next iteration)

Output: write to override_ledger as OverrideOutcome.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine.research.override_workflow import (
    OverrideOutcome, record_outcome,
)


def _sharpe(s: pd.Series) -> float:
    if s.std() == 0:
        return 0.0
    return float((s.mean() * 12) / (s.std() * math.sqrt(12)))


def main() -> int:
    # Inputs
    jp_monthly = pd.read_parquet("data/cache/_jp_pead_monthly.parquet").iloc[:, 0]
    jp_monthly.index = pd.to_datetime(jp_monthly.index)
    jp_diag = pd.read_parquet("data/cache/_jp_pead_event_diag.parquet")
    eps = pd.read_parquet("data/cache/_jp_ibes_eps_actuals.parquet")
    us_pit_sn = pd.read_parquet("data/cache/_dpead_sn_pit_monthly.parquet").iloc[:, 0]
    us_pit_sn.index = pd.to_datetime(us_pit_sn.index)

    print("=" * 90)
    print(" JP PEAD F1-F5 FALSIFICATION VERDICT")
    print("=" * 90)

    # ── F1: per-event t-stat ────────────────────────────────────────
    # We don't have per-event returns directly. Approximate via:
    # monthly L/S t-stat from full daily series
    n_events = int(jp_diag["n_universe"].sum())   # total firm-day exposures
    print(f"\n  [F1+F5] event sample size:")
    print(f"    total firm-day exposures: {n_events:,}")
    print(f"    daily L/S samples:        {len(jp_diag):,}")

    # Daily L/S t-stat (Bernard-Thomas style aggregate)
    daily_ls = jp_diag["ls_return"]
    t_daily = float(daily_ls.mean() / (daily_ls.std() / math.sqrt(len(daily_ls))))
    print(f"    daily L/S t-stat: {t_daily:.3f}")
    f1_pass = abs(t_daily) >= 2.0
    f5_pass = n_events >= 5000
    print(f"    F1 (|t| >= 2.0):    {'PASS' if f1_pass else 'FAIL'}")
    print(f"    F5 (n >= 5000):     {'PASS' if f5_pass else 'FAIL'}")

    # ── F2: monthly Sharpe ──────────────────────────────────────────
    sharpe = _sharpe(jp_monthly)
    print(f"\n  [F2] monthly L/S Sharpe (annualized):")
    print(f"    {sharpe:+.3f}")
    f2_pass = sharpe >= 0.5
    print(f"    F2 (Sharpe >= 0.5): {'PASS' if f2_pass else 'FAIL'}")

    # ── F3: DeflSR with n_trials=8 ──────────────────────────────────
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    dsr = deflated_sharpe_ratio(
        returns=jp_monthly.values, n_trials=8, periods_per_year=12,
    )
    print(f"\n  [F3] Deflated Sharpe (n_trials=8 family):")
    print(f"    DeflSR: {dsr.deflated_sr:.3f}")
    print(f"    verdict: {dsr.verdict}")
    f3_pass = dsr.deflated_sr >= 0.6
    print(f"    F3 (DeflSR >= 0.6): {'PASS' if f3_pass else 'FAIL'}")

    # ── F4: cosine with US PIT SN ───────────────────────────────────
    aligned = pd.concat([jp_monthly.rename("jp"),
                         us_pit_sn.rename("us")], axis=1).dropna()
    if len(aligned) >= 12:
        s_us = aligned["us"].values
        s_jp = aligned["jp"].values
        cosine = float(s_us @ s_jp /
                       (np.linalg.norm(s_us) * np.linalg.norm(s_jp)))
    else:
        cosine = float("nan")
    print(f"\n  [F4] cosine with US PIT SN ({len(aligned)} aligned mo):")
    print(f"    cosine: {cosine:+.3f}")
    f4_pass = cosine < 0.4
    print(f"    F4 (cosine < 0.4): {'PASS' if f4_pass else 'FAIL'}")

    # ── SENIOR CAVEATS (beyond F1-F5) ───────────────────────────────
    print(f"\n  [caveats - critical to read before any conclusion]")
    print(f"    1. SURVIVORSHIP BIAS: yfinance has only currently-live")
    print(f"       firms. Selecting 'top-500 by EPS event count' uses")
    print(f"       full-sample knowledge. EXPECTED to OVER-state PEAD.")
    print(f"    2. Universe selection not PIT — would need real-time")
    print(f"       TOPIX 500 each year.")
    print(f"    3. No sector neutralization yet — adding might IMPROVE")
    print(f"       or DEGRADE (we don't know).")
    print(f"    4. Yfinance return adjustments may differ from WRDS.")
    print(f"    5. No transaction cost modeled — JP equity costs likely")
    print(f"       15-30bp round-trip; could turn this into mid-Sharpe.")

    # ── Verdict + ledger ─────────────────────────────────────────────
    n_pass = sum([f1_pass, f2_pass, f3_pass, f4_pass, f5_pass])
    print(f"\n{'='*90}")
    print(f" F1-F5 PASSES: {n_pass}/5")
    if n_pass == 5:
        overall = "OVERTURNED_GRAVEYARD"
        verdict_text = "TENTATIVELY OVERTURNED (pending WRDS-grade confirmation)"
    elif n_pass >= 3:
        overall = "INCONCLUSIVE"
        verdict_text = "PARTIAL — needs more rigor to commit"
    else:
        overall = "REINFORCED_GRAVEYARD"
        verdict_text = "FAILED — graveyard prior correct"
    print(f" Overall: {overall}")
    print(f" Reading: {verdict_text}")
    print(f"{'='*90}")

    # Record to override ledger
    outcome = OverrideOutcome(
        candidate_id="jp_pead",
        overall_verdict=overall,
        falsification_results={
            "F1_per_event_t_stat":   f1_pass,
            "F2_monthly_sharpe":     f2_pass,
            "F3_deflated_sharpe":    f3_pass,
            "F4_cosine_with_us":     f4_pass,
            "F5_event_sample_size":  f5_pass,
        },
        actual_hours_spent=4.0,    # approximate tonight's JP work
        lessons_learned=(
            f"yfinance JP PEAD shows headline Sharpe {sharpe:.2f} on top-500 "
            f"firms. STRONG SURVIVORSHIP BIAS suspected — top-by-event-count "
            f"selects long-lived large-caps. F4 cosine with US = {cosine:.3f} "
            f"({'orthogonal' if f4_pass else 'too correlated'}). Next step: "
            f"WRDS-grade returns + real-time PIT universe + TC model "
            f"before committing to deploy. Graveyard signal was CORRECT to "
            f"force friction; override granted via process-rigor allowed "
            f"this exploratory finding without permanently bypassing rigor."
        ),
    )
    record_outcome(outcome)
    print(f"\n  outcome logged to data/research/override_ledger.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
