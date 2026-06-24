"""scripts/backfill_decay_alerts.py — one-time decay-alert backfill.

C of senior施工建议 (locked memory project_tier_c_senior_construction_
plan_2026-06-09). After C ships the decay_watch_trigger module, this
script runs ONCE over the deployed-sleeve manifest from
scripts/audits/audit_deployed_sleeves_rigor.py, runs subsample_stability on
each sleeve's persisted PnL, and emits `decay_alert` for any sleeve
that fires ≥2 of the 3 trigger criteria.

Why this exists:
  B.2 + L4 audit found cross_asset_carry has window 1 (2002-2007)
  Sharpe +1.11 dominating 22 years of total return, with W2-W4 ≈ 0.
  That's a TEXTBOOK McLean-Pontiff decay pattern and the system has
  NO surface telling the principal "review allocation" — until now.

This script is IDEMPOTENT in practice (events are immutable; running
again just emits a new event with the same metrics — downstream
consumers should de-dup by subject+window). But it's intended as a
ONE-TIME backfill; the standing wiring would be a cron job or
inline-in-dispatch trigger that runs as part of normal Tier C
auditing. That cron is deferred — this backfill alone exposes the
known carry decay to the principal.

Usage:
  python scripts/backfill_decay_alerts.py [--dry-run]

Flags:
  --dry-run   Show what would be emitted; don't touch the event store.

Exits:
  0  even when no alerts fired
  >0 on hard error (subject registration fail / parquet IO error)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _ensure_subjects_registered() -> None:
    """Register the 4 deployed-sleeve subjects if not already in the
    registry. emit.decay_alert validates the subject_id strictly per
    research-store doctrine; if these subjects don't exist, emit
    raises. Idempotent — register_subject is a no-op when subject
    already exists."""
    from engine.research_store import registry
    from engine.research_store.schema import SubjectType

    SLEEVE_SUBJECTS = [
        ("equity_book",
         "PIT SN (D_PEAD + IBES combo) — equity book deployed sleeve."),
        ("cross_asset_carry",
         "G10 4-leg carry overlay — cross-asset deployed sleeve."),
        ("crisis_hedge_tlt_gld",
         "TLT + GLD diversifier overlay — deployed sleeve."),
        ("mom_hedge_overlay",
         "MTUM short β-overlay insurance — deployed sleeve."),
    ]
    for name, description in SLEEVE_SUBJECTS:
        if registry.resolve(name) is None:
            registry.register_subject(
                name,
                subject_type=SubjectType.sleeve,
                family="deployed_book",
                description=description,
            )
            print(f"  Registered subject: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                            help="Don't touch the event store; print findings only.")
    # K (2026-06-10): cron mode. Enables dedup (skip when an open
    # alert exists or the same finding was already acked; re-emit
    # only on signature escalation). Registered as a weekly Windows
    # task via scripts/install_research_cron.py.
    parser.add_argument("--cron", action="store_true",
                            help="Cron mode: dedup against existing open/acked alerts.")
    args = parser.parse_args()

    # Late imports — sys.path tweaked above
    import pandas as pd
    from scripts.audits.audit_deployed_sleeves_rigor import SLEEVES, _to_pnl_series_df
    from engine.research.subsample_stability import (
        compute_for_tier_c_pnl_series as subsample,
    )
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
        emit_decay_alert_from_subsample,
    )

    if not args.dry_run:
        _ensure_subjects_registered()

    print()
    print("=" * 72)
    print(f"Decay alert backfill — {len(SLEEVES)} deployed sleeves")
    print(f"Mode: {'DRY-RUN (no emit)' if args.dry_run else 'LIVE'}")
    print("=" * 72)

    emitted = 0
    skipped = 0
    errored = 0

    for sleeve in SLEEVES:
        name = sleeve["name"]
        parquet = REPO_ROOT / sleeve["parquet"]
        col = sleeve["column"]
        print()
        print(f"── {name} ────────────────────────────")
        print(f"   role={sleeve.get('investment_role')}, "
                f"asset_class={sleeve.get('asset_class')}")
        if not parquet.exists():
            print(f"   SKIP: parquet missing at {parquet}")
            skipped += 1
            continue
        try:
            df_raw = pd.read_parquet(parquet)
            if col not in df_raw.columns:
                print(f"   SKIP: column {col!r} not in {list(df_raw.columns)}")
                skipped += 1
                continue
            series = df_raw[col].dropna()
            series.index = pd.DatetimeIndex(series.index)
            artifacts = {
                "pnl_series_df":   _to_pnl_series_df(series),
                "pnl_default_col": "pnl_net_13bp",
                "pnl_gross_col":   "pnl_gross",
            }
            sub_out = subsample(
                artifacts["pnl_series_df"], n_splits=4,
                artifacts=artifacts,
            )
            if sub_out is None:
                print("   SKIP: subsample returned None (insufficient months)")
                skipped += 1
                continue
        except Exception as exc:
            print(f"   ERROR: {type(exc).__name__}: {exc}")
            errored += 1
            continue

        # Evaluate triggers
        ev = evaluate_subsample_for_decay(sub_out)
        wbr = ev["worst_best_sharpe_ratio"]
        wbr_str = "None" if wbr is None else f"{wbr:.3f}"
        print(f"   subsample windows: {len(sub_out.get('windows') or [])} "
                f"(n_total={sub_out.get('n_total_months')}mo)")
        print(f"   worst/best Sharpe: {wbr_str}")
        print(f"   monotone_decay:    {ev['monotone_decay']}")
        if ev["latest_vs_prior_ratio"] is not None:
            print(f"   latest/prior:      "
                    f"{ev['latest_vs_prior_ratio']:.3f}")
        print(f"   triggers fired:    {ev['triggers_hit']} "
                f"(n={ev['n_triggers']}, severity={ev['severity']})")
        print(f"   summary:           {ev['summary']}")

        if ev["n_triggers"] < 2:
            print("   -> No emit (< 2 triggers; SUGGESTION threshold).")
            continue

        if args.dry_run:
            print(f"   -> DRY-RUN: would emit decay_alert "
                    f"({ev['severity']}, {ev['n_triggers']} triggers)")
            continue

        event_id = emit_decay_alert_from_subsample(
            subject_id        = name,
            subsample_output  = sub_out,
            parent_event_ids  = (),
            extra_tags        = ("cron",) if args.cron else ("backfill",),
            dedup             = args.cron,
        )
        if event_id:
            print(f"   -> EMITTED decay_alert event_id={event_id}")
            emitted += 1
        elif args.cron:
            print("   -> Dedup skip (open alert exists or same "
                    "finding already acked).")
            skipped += 1

    print()
    print("=" * 72)
    print(f"SUMMARY: emitted={emitted}, skipped={skipped}, errored={errored}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
