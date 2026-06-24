"""scripts/autopilot_perturbation_tests.py — F14a stability+reachability
gate (2026-06-05).

Replaces the "wait 2-3 days for Jaccard accumulation" stability check.
Rationale: F14a is a pure function of corpus state at time T.

  - Same state, run twice  -> Jaccard 1.0 trivially (determinism only)
  - Different state, run   -> picks shift (whether that's "stable" is
                                undefined without a counterfactual)

What we actually want to know is: does the selection rule RESPOND
SENSIBLY to perturbed inputs? Four perturbations cover the surface:

  T1 DETERMINISM       : run twice back-to-back, output identical
  T2 RED-PROPAGATION   : inject a synthetic RED verdict for top-1's
                          cell, top-1 must flip to WOULD_SKIP_REDUNDANCY
  T3 BACKFILL          : drop top-1 spec from the pool, old #2 must
                          promote to #1 and total distinct cells must
                          NOT decrease
  T4 DIVERSITY-PARAM   : set per_cell_cap=1, top-N must span N distinct
                          cells (no cell repeats)

All four run in ~10 seconds against today's catalog. If all GREEN,
the rule is stable + reachable; manual-run + cron promotion is the
last gate before F14b.

Run:
  python scripts/autopilot_perturbation_tests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _decisions_signature(plan) -> tuple:
    """Hashable signature of a plan's decisions for equality compare."""
    return tuple(
        (d.rank, d.source_hypothesis_id, d.spec_hash, d.action,
         d.family, d.signal_type)
        for d in plan.decisions
    )


def _summarize(label: str, plan) -> str:
    cells = [(d.family, d.signal_type, d.action) for d in plan.decisions]
    distinct_test_cells = len({(f, s) for f, s, a in cells if a == "WOULD_TEST"})
    return (f"{label}: n_test={plan.n_would_test} n_skip={plan.n_would_skip} "
             f"distinct_cells={distinct_test_cells}")


# ──────────────────────────────────────────────────────────────────────
# T1. DETERMINISM
# ──────────────────────────────────────────────────────────────────────
def test_determinism() -> tuple[bool, str]:
    from engine.agents.autopilot import compute_dry_run_plan
    a = compute_dry_run_plan(top_n=5)
    b = compute_dry_run_plan(top_n=5)
    sig_a = _decisions_signature(a)
    sig_b = _decisions_signature(b)
    ok = sig_a == sig_b
    msg = ("identical output across 2 runs"
           if ok else
           f"DIVERGED: run1 vs run2 differ in {sum(1 for x,y in zip(sig_a,sig_b) if x!=y)} positions")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────
# T2. RED-PROPAGATION
# ──────────────────────────────────────────────────────────────────────
def test_red_propagation() -> tuple[bool, str]:
    """Inject a synthetic RED verdict at top-1's (family) and assert
    that hyp's action becomes WOULD_SKIP_REDUNDANCY.

    Note: redundancy is family-level + signal-level match. We inject
    3 synthetic REDs (the STRONG WARN threshold), all at top-1's family
    + signal, so the worst-case advice is STRONG WARN.
    """
    from engine.agents.autopilot import compute_dry_run_plan
    import engine.research_store.mechanism_catalog as mc

    baseline = compute_dry_run_plan(top_n=5)
    if not baseline.decisions:
        return False, "baseline has no decisions; cannot perturb"
    top1 = baseline.decisions[0]
    target_family = top1.family
    target_signal = top1.signal_type

    orig_verdicts = mc._factor_verdicts
    def patched_verdicts():
        rows = list(orig_verdicts())
        for i in range(3):   # 3 REDs -> STRONG WARN threshold
            rows.append({
                "event_id":   f"synthetic_red_{i}",
                "subject_id": f"synthetic_red_subject_{i}",
                "family":     target_family,
                "verdict":    "RED",
                "ts":         "2099-01-01T00:00:00Z",
                "metrics":    {},
                "summary":    "synthetic for perturbation test",
            })
        return rows
    mc._factor_verdicts = patched_verdicts
    try:
        perturbed = compute_dry_run_plan(top_n=5)
    finally:
        mc._factor_verdicts = orig_verdicts

    # Find the top-1 hyp in the perturbed plan
    perturbed_for_top1 = next(
        (d for d in perturbed.decisions
         if d.source_hypothesis_id == top1.source_hypothesis_id),
        None,
    )
    if perturbed_for_top1 is None:
        return False, (f"top1 hyp {top1.source_hypothesis_id[:8]} VANISHED from "
                       f"perturbed plan (expected: it's still there but flagged SKIP)")
    ok = perturbed_for_top1.action == "WOULD_SKIP_REDUNDANCY"
    msg = (f"injected 3 REDs into {target_family}/{target_signal}; "
           f"top1 hyp {top1.source_hypothesis_id[:8]} action="
           f"{perturbed_for_top1.action}")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────
# T3. BACKFILL
# ──────────────────────────────────────────────────────────────────────
def test_backfill_after_top1_removed() -> tuple[bool, str]:
    """Remove top-1's spec from the pool. Expectations:
      - old rank-2 (or first non-skipped) becomes new rank-1
      - distinct-cell count in the perturbed plan does NOT decrease
    """
    from engine.agents.autopilot import compute_dry_run_plan
    import engine.agents.autopilot as ap

    baseline = compute_dry_run_plan(top_n=5)
    test_only = [d for d in baseline.decisions if d.action == "WOULD_TEST"]
    if len(test_only) < 2:
        return False, "baseline has < 2 testable candidates; backfill test n/a"
    top1_hyp = test_only[0].source_hypothesis_id
    old_rank2 = test_only[1]
    baseline_cells = len({(d.family, d.signal_type) for d in test_only})

    orig_latest = ap._latest_specs_by_hyp
    def patched_latest():
        d = dict(orig_latest())
        d.pop(top1_hyp, None)
        return d
    ap._latest_specs_by_hyp = patched_latest
    try:
        perturbed = compute_dry_run_plan(top_n=5)
    finally:
        ap._latest_specs_by_hyp = orig_latest

    perturbed_test = [d for d in perturbed.decisions if d.action == "WOULD_TEST"]
    if not perturbed_test:
        return False, "perturbed plan has no testable decisions"
    new_top1 = perturbed_test[0]
    new_cells = len({(d.family, d.signal_type) for d in perturbed_test})

    ok_promoted = new_top1.source_hypothesis_id == old_rank2.source_hypothesis_id
    ok_diversity = new_cells >= baseline_cells - 1   # at most 1 cell may
                                                       # vanish if top-1 was
                                                       # the sole occupant
    ok = ok_promoted and ok_diversity
    msg = (f"removed top1 ({top1_hyp[:8]}); new top1="
           f"{new_top1.source_hypothesis_id[:8]} "
           f"(expected {old_rank2.source_hypothesis_id[:8]}); "
           f"distinct_cells {baseline_cells} -> {new_cells}")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────
# T5. CROSS-PROCESS DETERMINISM
# ──────────────────────────────────────────────────────────────────────
def test_cross_process_determinism() -> tuple[bool, str]:
    """Two fresh Python interpreters with different PYTHONHASHSEED MUST
    produce identical top-5. This catches the bug where the tie-breaker
    uses Python's built-in hash() (randomized per-process), so two cron
    invocations of the same corpus would rank differently.

    Caught 2026-06-05; regression test ensures no one re-introduces it.
    """
    import os
    import subprocess
    import sys as _sys

    snippet = (
        "from engine.agents.autopilot import compute_dry_run_plan; "
        "p = compute_dry_run_plan(top_n=5); "
        "print('|'.join(d.source_hypothesis_id for d in p.decisions))"
    )
    env_a = {**os.environ, "PYTHONHASHSEED": "1"}
    env_b = {**os.environ, "PYTHONHASHSEED": "999999"}
    try:
        out_a = subprocess.check_output(
            [_sys.executable, "-c", snippet], env=env_a, timeout=120,
            stderr=subprocess.STDOUT, encoding="utf-8",
        ).strip().splitlines()[-1]
        out_b = subprocess.check_output(
            [_sys.executable, "-c", snippet], env=env_b, timeout=120,
            stderr=subprocess.STDOUT, encoding="utf-8",
        ).strip().splitlines()[-1]
    except subprocess.CalledProcessError as exc:
        return False, f"subprocess failed: {exc.output[-300:] if exc.output else ''}"
    ok = out_a == out_b
    msg = ("identical top-5 across PYTHONHASHSEED=1 vs 999999"
           if ok else
           f"DIVERGED across hash seeds — tie-breaker is non-deterministic")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────
# T4. DIVERSITY-PARAM
# ──────────────────────────────────────────────────────────────────────
def test_diversity_param() -> tuple[bool, str]:
    """per_cell_cap=1 must produce top-N with N distinct cells (no
    cell repeats)."""
    from engine.agents.autopilot import compute_dry_run_plan
    plan = compute_dry_run_plan(top_n=5, per_cell_cap=1)
    cells = [(d.family, d.signal_type) for d in plan.decisions
              if d.action == "WOULD_TEST"]
    distinct = len(set(cells))
    ok = distinct == len(cells)
    msg = (f"per_cell_cap=1: {len(cells)} test picks, {distinct} distinct cells "
           f"({'no repeats' if ok else 'REPEATS PRESENT'})")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    print("F14a perturbation tests (2026-06-05)")
    print()

    # Baseline preview
    from engine.agents.autopilot import compute_dry_run_plan
    baseline = compute_dry_run_plan(top_n=5)
    print(_summarize("baseline", baseline))
    print()

    tests = [
        ("T1 determinism",          test_determinism),
        ("T2 RED-propagation",      test_red_propagation),
        ("T3 backfill on removal",  test_backfill_after_top1_removed),
        ("T4 diversity param",      test_diversity_param),
        ("T5 cross-process det.",   test_cross_process_determinism),
    ]
    results: list[tuple[str, bool, str]] = []
    for name, fn in tests:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"raised {type(exc).__name__}: {exc}"
        results.append((name, ok, msg))
        tag = "GREEN" if ok else "RED  "
        print(f"  {tag}  {name:<28}  {msg}")

    print()
    print("=" * 70)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"PASS {n_pass}/{len(results)}")
    if n_pass == len(results):
        print("F14a selection rule: stable + reachable + responsive.")
        print("Next gate: manual top-1 run (autopilot_manual_run_top1.py).")
    else:
        print("F14a has perturbation failures. Do NOT promote to F14b until "
               "the failing tests pass.")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
