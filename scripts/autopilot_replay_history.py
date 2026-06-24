"""scripts/autopilot_replay_history.py — F14a soak proxy (2026-06-05).

Replaces the original "1-week wall-clock soak before F14b" gate with a
backward replay: simulate what compute_dry_run_plan() WOULD have output
on each of the last N calendar days, given only the corpus state that
existed at end-of-day-T.

Why this exists. F14a is a pure function of (catalog, library, specs,
verdicts) at time T. Wall-clock soak adds no signal — the function has
no time-accumulation behavior (unlike paper-trade, which accumulates
real P&L). What we actually want to verify before promoting to F14b:

  1. Day-over-day selection STABILITY. If today's top-1 is gone tomorrow
     and back the day after, the selection rule is unstable.
  2. CELL DIVERSITY. Top-N should cover >= 3 distinct (family,
     signal_type) cells. If all 5 picks live in one cell, PER_CELL_CAP
     is broken or the corpus is too narrow.
  3. REDUNDANCY-trigger COVERAGE. Over a 7-day window, did STRONG WARN
     ever fire? If never, the rule is dead code; if always, the corpus
     is poisoned.
  4. STICKY high-priority candidates. The same convergence-cluster
     spec landing rank-1 day after day = the rule is finding the
     genuine top — exactly what we want.

All four are measurable in ONE run by replaying back-dated state.

Implementation: monkeypatch the 2 loaders that drive the catalog +
the autopilot's `_latest_specs_by_hyp`, filtering rows by their
embedded timestamp. Library YAMLs are point-in-time files w/o per-row
ts — use today's state (changes slowly enough that 7-day drift is
typically zero or one entry).

Run:
  python scripts/autopilot_replay_history.py [--days 7] [--top 5]

Output:
  - stdout: per-day picks table + stability/diversity/redundancy summary
  - data/autopilot/_replay/replay_<ts>.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_OUT_DIR = REPO_ROOT / "data" / "autopilot" / "_replay"


# ──────────────────────────────────────────────────────────────────────
# Monkeypatch: filter all jsonl rows + verdict events + spec dataclasses
# by an as-of timestamp set by the replay driver.
# ──────────────────────────────────────────────────────────────────────
_AS_OF: Optional[str] = None      # iso 'YYYY-MM-DDTHH:MM:SSZ' upper bound


def _row_ts(row: dict) -> str:
    """Best-effort ts extraction across all our jsonl row shapes."""
    return (
        (row.get("extraction") or {}).get("extracted_ts")
        or row.get("created_ts")
        or row.get("ts")
        or row.get("plan_ts")
        or ""
    )


def _install_monkeypatches() -> None:
    """Wrap mechanism_catalog._load_jsonl, _factor_verdicts, and
    autopilot._latest_specs_by_hyp so they respect _AS_OF.

    Each wrapper falls through (no filter) when _AS_OF is None — so
    importing this script doesn't perturb anything outside replay mode.
    """
    import engine.research_store.mechanism_catalog as mc
    import engine.agents.autopilot as ap

    orig_load_jsonl   = mc._load_jsonl
    orig_verdicts     = mc._factor_verdicts
    orig_latest_specs = ap._latest_specs_by_hyp

    def filtered_load_jsonl(path):
        rows = orig_load_jsonl(path)
        if _AS_OF is None:
            return rows
        out = []
        for r in rows:
            ts = _row_ts(r)
            # No timestamp → keep (we'd rather over-include than drop
            # rows we can't time-attribute; same-day error is small)
            if not ts or ts <= _AS_OF:
                out.append(r)
        return out

    def filtered_verdicts():
        rows = orig_verdicts()
        if _AS_OF is None:
            return rows
        return [r for r in rows if (r.get("ts") or "") <= _AS_OF]

    def filtered_latest_specs():
        d = orig_latest_specs()
        if _AS_OF is None:
            return d
        out = {}
        for hid, s in d.items():
            ts = (s.extraction.extracted_ts or s.created_ts or "")
            if not ts or ts <= _AS_OF:
                out[hid] = s
        return out

    mc._load_jsonl         = filtered_load_jsonl
    mc._factor_verdicts    = filtered_verdicts
    ap._latest_specs_by_hyp = filtered_latest_specs


# ──────────────────────────────────────────────────────────────────────
# Replay driver
# ──────────────────────────────────────────────────────────────────────
def _set_as_of(target_eod_utc: _dt.datetime) -> None:
    global _AS_OF
    _AS_OF = target_eod_utc.strftime("%Y-%m-%dT23:59:59Z")


def _summarize_decision(d) -> dict:
    """Per-candidate row for the per-day JSON."""
    return {
        "rank":               d.rank,
        "source_hypothesis_id": d.source_hypothesis_id,
        "spec_hash":          d.spec_hash,
        "cell":               f"{d.family}/{d.signal_type}",
        "action":             d.action,
        "in_convergence":     d.cell_in_convergence,
        "n_red_in_cluster":   d.redundancy_n_red,
        "redundancy_advice":  d.redundancy_advice,
    }


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    ap_arg = argparse.ArgumentParser()
    ap_arg.add_argument("--days", type=int, default=7,
                         help="Number of past calendar days to replay (default 7)")
    ap_arg.add_argument("--top", type=int, default=5)
    args = ap_arg.parse_args()

    _install_monkeypatches()

    from engine.agents.autopilot import compute_dry_run_plan

    # Build the day list — today inclusive, walking backwards
    today = _dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    target_days = [today - _dt.timedelta(days=k) for k in range(args.days)]
    target_days.reverse()   # ascending: oldest first

    per_day_plans: list[dict] = []
    print(f"Replaying autopilot over {args.days} days (top_n={args.top})")
    print()
    print(f"{'date':<12} {'ready':>6} {'test':>5} {'skip':>5} {'cells':>6}  top-1 cell")
    print("-" * 72)

    for d in target_days:
        _set_as_of(d)
        try:
            plan = compute_dry_run_plan(top_n=args.top)
        except Exception as exc:
            logger.exception("replay failed on %s: %s", d.date().isoformat(), exc)
            continue
        decisions = [_summarize_decision(dec) for dec in plan.decisions]
        cells = [d["cell"] for d in decisions if d["action"] == "WOULD_TEST"]
        unique_cells = len(set(cells))
        top1 = cells[0] if cells else "(empty)"
        per_day_plans.append({
            "date":               d.date().isoformat(),
            "as_of_ts":           _AS_OF,
            "n_ready_specs":      plan.n_ready_specs,
            "n_would_test":       plan.n_would_test,
            "n_would_skip":       plan.n_would_skip,
            "unique_cells":       unique_cells,
            "decisions":          decisions,
            "top1_cell":          top1,
        })
        print(f"{d.date().isoformat()}  "
               f"{plan.n_ready_specs:>5}  {plan.n_would_test:>4}  "
               f"{plan.n_would_skip:>4}  {unique_cells:>5}   {top1}")

    print()
    print("=" * 72)
    print("STABILITY / DIVERSITY / COVERAGE")
    print("=" * 72)

    # 1. Day-over-day selection stability via Jaccard on spec_hash sets
    if len(per_day_plans) >= 2:
        jaccs = []
        for a, b in zip(per_day_plans[:-1], per_day_plans[1:]):
            sa = {d["spec_hash"] for d in a["decisions"] if d["action"] == "WOULD_TEST"}
            sb = {d["spec_hash"] for d in b["decisions"] if d["action"] == "WOULD_TEST"}
            jaccs.append(_jaccard(sa, sb))
        avg_j = sum(jaccs) / len(jaccs)
        print(f"  selection stability (Jaccard, adjacent days):")
        for i, j in enumerate(jaccs):
            a_d, b_d = per_day_plans[i]['date'], per_day_plans[i+1]['date']
            print(f"    {a_d} -> {b_d}:  {j:.2f}")
        print(f"  avg Jaccard: {avg_j:.2f}  "
               + ("OK  (sticky, mostly same picks)" if avg_j >= 0.6
                   else "BAD (churning — selection rule unstable)" if avg_j < 0.4
                   else "~   (some churn — investigate)"))
    else:
        avg_j = None

    # 2. Cell diversity per day
    avg_div = sum(p["unique_cells"] for p in per_day_plans) / max(1, len(per_day_plans))
    print()
    print(f"  cell diversity (mean distinct cells in top-{args.top} per day): {avg_div:.1f}")
    print(f"    target: >= 3   {'OK' if avg_div >= 3 else 'BAD (corpus too narrow OR PER_CELL_CAP broken)'}")

    # 3. STRONG WARN trigger coverage over the window
    n_skips_total = sum(p["n_would_skip"] for p in per_day_plans)
    days_w_skip = sum(1 for p in per_day_plans if p["n_would_skip"] > 0)
    print()
    print(f"  STRONG WARN triggers: {n_skips_total} total across {days_w_skip}/{len(per_day_plans)} days")
    if n_skips_total == 0:
        print(f"    INFO redundancy guard never fired in this window. Either the rule is "
               f"dead code OR the corpus genuinely has no redundancy. Construct a "
               f"deployed-overlap test to verify the rule is reachable before F14b.")
    else:
        print(f"    OK   redundancy guard reachable and triggered organically")

    # 4. Top-1 stickiness  (sanity for "is the convergence picker doing its job")
    top1_counts = Counter(p["top1_cell"] for p in per_day_plans)
    most_common, n = top1_counts.most_common(1)[0]
    print()
    print(f"  most-frequent top-1 cell: {most_common}  ({n}/{len(per_day_plans)} days)")
    if n == len(per_day_plans):
        print(f"    OK   same convergence cell ranks #1 every day = stable priority")
    elif n >= len(per_day_plans) * 0.7:
        print(f"    OK   mostly stable top-1 with some rotation")
    else:
        print(f"    BAD  top-1 rotates aggressively — convergence sort may be ineffective")

    # Persist
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_path = _OUT_DIR / f"replay_{ts}.json"
    out_path.write_text(json.dumps({
        "ts":               ts,
        "days":             args.days,
        "top_n":            args.top,
        "per_day":          per_day_plans,
        "avg_jaccard":      avg_j,
        "avg_cell_diversity": avg_div,
        "n_strong_warn_total": n_skips_total,
        "days_w_strong_warn": days_w_skip,
    }, indent=2), encoding="utf-8")
    print()
    print(f"Detail JSON: {out_path}")

    # Verdict
    print()
    print("=" * 72)
    print("F14b READINESS VERDICT")
    print("=" * 72)
    ok_stability = avg_j is None or avg_j >= 0.5
    ok_diversity = avg_div >= 3
    ok_top1      = n >= len(per_day_plans) * 0.7
    # STRONG WARN coverage is informative not blocking — corpus may
    # honestly lack a deployed-overlap candidate in this window.
    if ok_stability and ok_diversity and ok_top1:
        print("  GREEN  — selection logic is stable + diverse + sensibly ranked.")
        print("           Proceed to manual-run top-1 candidate; if that works,")
        print("           promote to F14b.")
    else:
        print("  RED    — selection logic has issues:")
        if not ok_stability:
            print(f"           - day-over-day churn too high (Jaccard {avg_j:.2f} < 0.5)")
        if not ok_diversity:
            print(f"           - diversity {avg_div:.1f} < 3 distinct cells")
        if not ok_top1:
            print(f"           - top-1 unstable ({n}/{len(per_day_plans)} days "
                   f"on most-common = {100*n/len(per_day_plans):.0f}%)")
        print("           Fix the rule first; do NOT promote to F14b.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
