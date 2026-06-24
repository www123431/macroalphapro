"""scripts/run_pipeline_on_ltr_and_issuance.py — eat-own-dogfood:
run candidate_pipeline 14-step audit on LTR and issuance series.

Per user direction: "立刻把 LTR 喂给 candidate_pipeline,让 loop 告诉
我们 ABCD 哪个对。"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import engine.research.sleeves  # noqa: F401

from engine.research.candidate_pipeline import run_candidate_pipeline


def run_one(label: str, series_path: str, mechanism_id: str | None = None,
            parent_returns_path: str | None = None,
            proposal_dict: dict | None = None):
    print("\n" + "=" * 95)
    print(f" CANDIDATE_PIPELINE on {label}")
    print("=" * 95)
    s = pd.read_parquet(series_path).iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    print(f"  series: n={len(s)} months "
          f"({s.index.min().date()} → {s.index.max().date()})")
    import math
    sh = (s.mean() * 12) / (s.std() * math.sqrt(12)) if s.std() > 0 else 0.0
    print(f"  gross Sharpe: {sh:+.3f}")

    report = run_candidate_pipeline(
        candidate_returns=s,
        proposal_name=label,
        proposed_role="alpha_seeker",
        mechanism_id=mechanism_id,
        proposal_dict=proposal_dict,
        parent_returns_path=parent_returns_path,
        phase=3,
    )

    print(f"\n  STEP RESULTS:")
    for i, step in enumerate(report.step_results, 1):
        d = step.to_dict()
        icon = {"PASS":"[OK]","WARN":"[WARN]","FAIL":"[FAIL]","SKIP":"[SKIP]","INFO":"[INFO]"}.get(d["status"], "[?]")
        verdict = d.get("verdict", "")
        if len(verdict) > 90:
            verdict = verdict[:87] + "..."
        print(f"  {i:2d} {icon:<6} {d.get('step_name','?'):<35} {verdict}")

    meta = report.to_dict()
    print(f"\n  ── meta_decision ──")
    print(f"  final_decision:    {meta.get('final_decision')}")
    rat = meta.get('rationale') or ''
    print(f"  rationale:         {rat[:300]}")
    print(f"  role_used:         {meta.get('role_used')}")
    print(f"  candidate_relation:{meta.get('candidate_relation')}")
    mc_s = meta.get('most_correlated_sleeve')
    mc_v = meta.get('most_correlated_value')
    if mc_s and mc_v is not None:
        print(f"  most_correlated:   {mc_s}  cosine={mc_v:+.3f}")


def main() -> int:
    run_one(
        label="LTR_long_history",
        series_path="data/cache/_ltr_monthly_long.parquet",
        mechanism_id="long_term_reversal",
        proposal_dict={
            "family":         "long_term_reversal",
            "parent_family":  "equity_factor",
            "required_data":  ["crsp_msf", "monthly_returns"],
            "economics_text": (
                "Long-term reversal (De Bondt-Thaler 1985): firms with "
                "poor 36-60 month past performance outperform firms with "
                "strong past performance, after skipping last 12 months "
                "to avoid momentum contamination. Mechanism: overreaction "
                "to long-run news that mean-reverts."
            ),
            # Post-pub replication evidence for H6
            "post_pub_decay": {
                "post_2020_replications": [
                    {"paper_id": "hou_xue_zhang_2020_rfs",
                     "delta_range_estimate": [-0.20, 0.00],
                     "notes": "HKK 2020 RFS classifies LTR as SURVIVES (large)"},
                ],
            },
        },
    )
    run_one(
        label="issuance_anomaly",
        series_path="data/cache/_issuance_monthly.parquet",
        mechanism_id="issuance",
        proposal_dict={
            "family":         "issuance",
            "parent_family":  "equity_factor",
            "required_data":  ["crsp_msf", "shares_outstanding"],
            "economics_text": (
                "Net share issuance anomaly (Pontiff-Woodgate 2008): "
                "firms reducing share count (buybacks) outperform firms "
                "issuing new shares. Corporate financing signal. "
                "Distinct from value / quality. HKK 2020 SURVIVES (large)."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
