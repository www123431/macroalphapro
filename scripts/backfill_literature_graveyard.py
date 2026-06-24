"""scripts/backfill_literature_graveyard.py — T4.6 (2026-06-05).

Backfill RED verdicts from legacy ledgers (factory_ledger.jsonl,
gate_runs.jsonl) into research_store as factor_verdict_filed events
so they're visible to graveyard_collision.check_collision().

Why this exists
---------------
The T4.5 audit found that direction_proposer was marking all 20 top
candidates as graveyard CLEAN — including obvious anchor factors
like BAB / QMJ / value-momentum combos. Root cause: research_store
held only 25 path-* test results, while 43 additional RED verdicts
(carry / vrp / insider / news_attention / lazy_prices / merger_arb
/ patents / supply-chain mom / KOR PEAD / Korean variants / cmdty
carry / sector lead-lag / bond_xsmom / vix carry / credit carry /
g10 XC carry EM extension / and others) lived only in legacy
ledgers and weren't queried by the collision check.

After backfill, graveyard_collision sees the full ~68 RED corpus
and the semantic S4 dim (T4.5 commit 27caf9ae) has many more
chances to fire for paper-derived candidates.

Idempotency
-----------
- Subject registration is idempotent (re-register returns the
  existing Subject; no errors).
- Event emission DOES create a new event_id on every call. So
  running this twice will create duplicate events. To prevent that,
  we check `store.filter_events(subject_id=..., verdict=RED)`
  before each emit and skip if a RED already exists for that subject.

Usage
-----
  python scripts/backfill_literature_graveyard.py            # full backfill
  python scripts/backfill_literature_graveyard.py --dry-run  # plan only, no writes
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Family mapping by name prefix. Order matters — first match wins.
# Falls back to OTHER for unmatched names; emit still works (family
# is informational, not the gating field).
_NAME_TO_FAMILY = [
    ("carry_equity_div",         "CARRY"),
    ("bond_carry",               "CARRY"),
    ("bond_xsmom",               "CROSS_ASSET_MOMENTUM"),
    ("cmdty_fx_rates_carry",     "CARRY"),
    ("carry_tsmom",              "CARRY"),
    ("g10_xc_carry",             "CARRY"),
    ("vix_conditional_carry",    "CARRY"),
    ("credit_spread_carry",      "CARRY"),
    ("vix_carry",                "CARRY"),
    ("iii4_credit_spread_carry", "CARRY"),
    ("vrp_",                     "VOL_RISK_PREMIUM"),
    ("iv_skew",                  "OPTIONS_IMPLIED"),
    ("K1_BAB",                   "LOW_VOL"),
    ("insider_",                 "HOLDINGS_BASED"),
    ("lazy_prices",              "ATTENTION"),
    ("lazy_lm",                  "SENTIMENT"),
    ("news_attention",           "ATTENTION"),
    ("news_ess",                 "SENTIMENT"),
    ("analyst_revision",         "EARNINGS_DRIFT"),
    ("sue_rev",                  "EARNINGS_DRIFT"),
    ("KOR_PEAD",                 "EARNINGS_DRIFT"),
    ("supplychain_mom",          "SUPPLY_CHAIN"),
    ("merger_arb",               "OTHER"),       # special category, no family
    ("patents_ie",               "OTHER"),       # innovation/intangibles
    ("regime_overlay",           "OTHER"),
    ("sector_leadlag",           "CROSS_ASSET_MOMENTUM"),
    ("PATH_",                    "OTHER"),
    ("CTA_",                     "MOMENTUM"),
    ("AC_proxy",                 "OTHER"),
]


def _family_for(name: str) -> str:
    for prefix, fam in _NAME_TO_FAMILY:
        if name.lower().startswith(prefix.lower()) or prefix.lower() in name.lower():
            return fam
    return "OTHER"


def _factory_ledger_reds() -> list[dict]:
    p = REPO_ROOT / "data" / "validation" / "factory_ledger.jsonl"
    if not p.is_file():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get("light", r.get("verdict", ""))).upper() == "RED":
                out.append(r)
    return out


def _gate_runs_reds() -> list[dict]:
    p = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"
    if not p.is_file():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get("verdict", "")).upper() == "RED":
                out.append(r)
    return out


def _summary_from_factory(row: dict) -> str:
    reasons = row.get("reasons") or []
    head = "; ".join(str(x) for x in reasons[:3])
    stats = (f"deflated_sr={row.get('deflated_sr', 'NA')} "
             f"net={row.get('net_deflated_sr', 'NA')} "
             f"alpha_t={row.get('residual_alpha_t', 'NA')}")
    return (f"[backfilled from factory_ledger] {head} | {stats}")[:395]


def _summary_from_gate(row: dict) -> str:
    bars = row.get("bars") or {}
    head = (f"deflated_sr={row.get('deflated_sr', 'NA')} "
            f"oos_sharpe={row.get('oos_sharpe', 'NA')} "
            f"alpha_t={row.get('alpha_t_ff5umd', 'NA')} "
            f"book_corr={row.get('corr_with_book', 'NA')} "
            f"mechanism={row.get('mechanism', 'NA')}")
    bars_s = f"bars: {bars}" if bars else ""
    return (f"[backfilled from gate_runs] FAIL: {head} | {bars_s}")[:395]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be emitted, do nothing")
    args = ap.parse_args()

    # Deferred so --help doesn't pay import cost
    from engine.research_store import registry, store
    from engine.research_store import emit as rs_emit
    from engine.research_store.schema import SubjectType

    factory_reds = _factory_ledger_reds()
    gate_reds    = _gate_runs_reds()
    print(f"factory_ledger REDs:  {len(factory_reds)}")
    print(f"gate_runs REDs:       {len(gate_reds)}")
    print()

    # Build the unified backfill plan: (subject_id, family, summary, ts, source_path)
    plan: list[dict] = []
    for r in factory_reds:
        name = r.get("name") or ""
        if not name:
            continue
        plan.append({
            "subject_id": name,
            "family":     _family_for(name),
            "summary":    _summary_from_factory(r),
            "ts":         r.get("ts", ""),
            "source":     "data/validation/factory_ledger.jsonl",
            "raw":        r,
        })
    for r in gate_reds:
        name = r.get("name") or ""
        if not name:
            continue
        plan.append({
            "subject_id": name,
            "family":     _family_for(name),
            "summary":    _summary_from_gate(r),
            "ts":         r.get("ts", ""),
            "source":     "data/research/gate_runs.jsonl",
            "raw":        r,
        })

    print(f"Plan total: {len(plan)}")
    fam_dist = Counter(p["family"] for p in plan)
    print("Family distribution in backfill plan:")
    for f, n in fam_dist.most_common():
        print(f"  {n:>3}  {f}")
    print()

    # Skip-existing pass — query research_store for already-emitted REDs
    existing = store.filter_events(event_type="factor_verdict_filed",
                                    verdict="RED", limit=500)
    existing_subjects = {ev.subject_id for ev in existing}
    skipped = [p for p in plan if p["subject_id"] in existing_subjects]
    todo    = [p for p in plan if p["subject_id"] not in existing_subjects]
    print(f"Already in research_store as RED:  {len(skipped)}")
    print(f"To emit this run:                 {len(todo)}")
    print()

    if args.dry_run:
        print("--dry-run set; no writes.")
        print()
        print("Sample plan rows:")
        for p in todo[:5]:
            print(f"  subject={p['subject_id'][:40]:<40} family={p['family']:<22}")
            print(f"    summary: {p['summary'][:120]}")
        return 0

    n_ok = 0
    n_fail = 0
    for p in todo:
        # Register subject (idempotent)
        try:
            registry.register_subject(
                p["subject_id"],
                subject_type=SubjectType.factor,
                family=p["family"],
                description=f"Backfilled from {p['source']} (T4.6)",
                created_by="t4_6_backfill",
            )
        except Exception as e:
            print(f"  REGISTER_FAIL  {p['subject_id'][:40]:<40}  {e}")
            n_fail += 1
            continue

        # Emit verdict
        try:
            ev_id = rs_emit.factor_verdict(
                subject_id=p["subject_id"],
                verdict="RED",
                metrics={
                    "backfilled_from": p["source"],
                    "original_ts":     p["ts"],
                },
                artifacts={"source_ledger": p["source"]},
                summary=p["summary"],
                family=p["family"],
                tags=("literature_graveyard", "T4.6_backfill"),
                actor="t4_6_backfill",
            )
            n_ok += 1
            print(f"  OK  {ev_id[:8]}  {p['subject_id'][:40]:<40}  fam={p['family']}")
        except Exception as e:
            print(f"  EMIT_FAIL  {p['subject_id'][:40]:<40}  {type(e).__name__}: {str(e)[:80]}")
            n_fail += 1

    print()
    print("=" * 60)
    print(f"Done. ok={n_ok}  fail={n_fail}  skipped={len(skipped)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
