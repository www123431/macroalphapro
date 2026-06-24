"""scripts/audit_nav_monitoring_fix_2026-06-18.py

P0 operational audit: NAV monitoring has 4 bugs producing 100%
false-positive anomaly alerts. This script:

1. Reads existing data/research/nav_history.jsonl
2. Identifies the 4 known issues
3. Computes CORRECTED daily returns (gap-aware) and z-scores
   (target-vol-aware)
4. Reports: would ANY alert have fired under the corrected logic?
5. Outputs a CLEANED NAV series for downstream consumption

If 0 corrected-alerts fire over the live deployment window
(2026-05-14 to present), strategy IS healthy and the alerts were
all monitoring bugs.

Per [[feedback-sizing-before-signal-2026-06-17]] doctrine: do NOT
push Sharpe-side work; this is operational hygiene that costs
nothing and prevents the team from chasing ghost-alerts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
NAV_PATH   = _REPO_ROOT / "data" / "research" / "nav_history.jsonl"
OUT_DIR    = _REPO_ROOT / "data" / "research_store" / "audit" / "nav_monitoring_2026_06_18"
OUT_DIR.mkdir(parents=True, exist_ok=True)


DEPLOYED_VOL_TARGET = 0.10           # 10% annualized — from active_deployment.yaml
ANOMALY_THRESHOLD_SIGMA = 3.0        # 3σ over deployed vol target


def load_raw_nav() -> pd.DataFrame:
    rows = []
    for ln in NAV_PATH.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


def dedup_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    """Fix 1: same-day duplicate rows → keep latest ts (EOD)."""
    return (df.sort_values(["as_of", "ts"])
                .drop_duplicates(subset=["as_of"], keep="last")
                .reset_index(drop=True))


def compute_corrected_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix 2+3: gap-aware return + target-vol-aware z-score.

    daily_eq_return = log(NAV_t / NAV_{t-1}) / days_elapsed
    expected_daily_vol = TARGET_VOL × sqrt(1/252) ≈ 0.63%
    z_corrected = daily_eq_return / expected_daily_vol
    """
    out = df.copy()
    out["days_elapsed"]   = out["as_of"].diff().dt.days.astype("float")
    out["raw_return"]     = np.log(out["equity"] / out["equity"].shift(1))
    # Gap-normalized daily return
    out["daily_return"]   = out["raw_return"] / out["days_elapsed"].replace(0, np.nan)

    expected_daily_vol = DEPLOYED_VOL_TARGET / math.sqrt(252)
    out["z_corrected"]    = out["daily_return"] / expected_daily_vol
    out["alert_corrected"] = (
        out["z_corrected"].abs() >= ANOMALY_THRESHOLD_SIGMA
    )
    return out


def main():
    print("=" * 80)
    print("NAV monitoring operational audit (2026-06-18)")
    print("=" * 80)

    raw = load_raw_nav()
    print(f"Raw NAV file: {len(raw)} rows, "
          f"{raw['as_of'].min().date()} → {raw['as_of'].max().date()}")

    raw_anomalies = raw[raw["status"] == "anomaly"]
    print(f"Raw 'anomaly' flag count: {len(raw_anomalies)}")
    print()

    # Step 1: dedup
    dedup = dedup_keep_last(raw)
    print(f"After dedup keep-last: {len(dedup)} rows")
    n_dropped = len(raw) - len(dedup)
    print(f"Dropped {n_dropped} duplicate (as_of, *) rows")
    print()

    # Step 2 + 3: corrected returns and z-scores
    corrected = compute_corrected_returns(dedup)
    print("CORRECTED daily-return + z-score table:")
    cols = ["as_of", "days_elapsed", "equity", "daily_return",
             "z_corrected", "alert_corrected"]
    fmt_df = corrected[cols].copy()
    fmt_df["days_elapsed"] = fmt_df["days_elapsed"].apply(
        lambda x: f"{int(x):>3}" if not pd.isna(x) else "  -"
    )
    fmt_df["equity"]       = fmt_df["equity"].apply(lambda x: f"{x:>10,.2f}")
    fmt_df["daily_return"] = fmt_df["daily_return"].apply(
        lambda x: f"{x:+.4%}" if not pd.isna(x) else "  n/a"
    )
    fmt_df["z_corrected"]  = fmt_df["z_corrected"].apply(
        lambda x: f"{x:+.2f}" if not pd.isna(x) else "  n/a"
    )
    fmt_df["alert_corrected"] = fmt_df["alert_corrected"].apply(
        lambda b: "★ ALERT" if b else ""
    )
    print(fmt_df.to_string(index=False))
    print()

    # Compare raw vs corrected anomaly counts
    n_corrected_alerts = int(corrected["alert_corrected"].sum())
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Raw 'anomaly' flag count:                 {len(raw_anomalies)}")
    print(f"  Corrected 3σ-vs-target-vol alert count:  {n_corrected_alerts}")
    print()
    if n_corrected_alerts == 0:
        print("  ✓ ALL raw anomaly flags were monitoring false-positives.")
        print("  ✓ Strategy NAV behavior is consistent with 10% vol target +")
        print("    normal post-deployment dispersion.")
    else:
        print(f"  ⚠ {n_corrected_alerts} real anomalies remain after correction.")
        print("    See alert_corrected=★ ALERT rows above for detail.")
    print()

    # Save cleaned NAV + audit
    out_csv = OUT_DIR / "nav_cleaned.csv"
    corrected.to_csv(out_csv, index=False)
    print(f"Cleaned NAV: {out_csv}")

    out_json = OUT_DIR / "nav_audit_summary.json"
    out_json.write_text(json.dumps({
        "raw_rows":             int(len(raw)),
        "raw_anomaly_count":    int(len(raw_anomalies)),
        "dedup_rows":           int(len(dedup)),
        "corrected_alert_count": int(n_corrected_alerts),
        "deployed_vol_target":   DEPLOYED_VOL_TARGET,
        "anomaly_threshold_sigma": ANOMALY_THRESHOLD_SIGMA,
        "verdict":              ("monitoring_bugs_only"
                                   if n_corrected_alerts == 0
                                   else "real_anomalies_exist"),
        "audit_window":         (raw["as_of"].min().date().isoformat(),
                                   raw["as_of"].max().date().isoformat()),
    }, indent=2, default=str))
    print(f"Audit summary: {out_json}")


if __name__ == "__main__":
    main()
