"""scripts/simulate_frontier_validation.py — Path D-lite synthetic
outcome generator.

Synthesizes N (council, pipeline) outcome pairs deterministically so
the calibration_feedback / critic_calibration pipelines can be
exercised + validated WITHOUT a 2-week real-cron wait.

DOCTRINE: simulated outcomes are NOT ground truth. This script
validates the PIPELINE (does critic_calibration's marginal-info-gain
math compute correctly given input outcomes?), NOT the COUNCIL (does
reflection actually improve verdicts in reality?).

Generation model: each (proposal_family) has a true latent P(success),
and each critic has a per-family base-rate accuracy. Outcomes are
sampled from the latent + critic noise. Reproducible via fixed seed.

USAGE:
    python scripts/simulate_frontier_validation.py --n 30 [--seed 42]

After running:
    python -m engine.research.critic_calibration report 365
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from pathlib import Path

# Add repo root to sys.path so the script runs from any cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Synthetic distributions ──────────────────────────────────────────


# True latent P(GREEN) per family — realistic distribution per the
# real labeled dataset.
_FAMILY_TRUE_P_GREEN = {
    "momentum":               0.30,
    "carry":                  0.45,
    "reversal":               0.20,
    "earnings_underreaction": 0.50,
    "tsmom":                  0.40,
    "text_ml":                0.05,
    "vol_carry":              0.10,
}

# Per-critic per-family accuracy — realistic heterogeneity. Theorist
# is better on mechanism families, DA is better on stat-heavy families.
_CRITIC_FAMILY_ACCURACY = {
    "behavioral_theorist": {
        "momentum": 0.70, "carry": 0.75, "reversal": 0.65,
        "earnings_underreaction": 0.80, "tsmom": 0.70,
        "text_ml": 0.55, "vol_carry": 0.60,
    },
    "empirical_devils_advocate": {
        "momentum": 0.65, "carry": 0.60, "reversal": 0.75,
        "earnings_underreaction": 0.70, "tsmom": 0.75,
        "text_ml": 0.85, "vol_carry": 0.80,
    },
}


def _critic_verdict(critic: str, family: str, true_green: bool,
                     rng: random.Random) -> str:
    """Sample a (PASS / WARN / FAIL) verdict given the latent truth."""
    acc = _CRITIC_FAMILY_ACCURACY.get(critic, {}).get(family, 0.65)
    if rng.random() < acc:
        # Critic gets it "right" — high confidence on correct direction
        return "PASS" if true_green else "FAIL"
    # Critic uncertain — emit WARN half the time, wrong-direction other half
    if rng.random() < 0.5:
        return "WARN"
    return "FAIL" if true_green else "PASS"


def _pipeline_decision(true_green: bool, rng: random.Random) -> str:
    """Ground-truth pipeline verdict (with small noise) given latent truth."""
    if true_green:
        return "PROMOTE_TO_GATE" if rng.random() > 0.10 else "BORDERLINE_REVIEW"
    return "HARD_REJECT" if rng.random() > 0.15 else "BORDERLINE_REVIEW"


def simulate_iterations(n: int, *, seed: int = 42) -> list[dict]:
    """Generate N synthetic L4 iterations with critics + outcomes."""
    from engine.research.agent_council import aggregate_verdicts
    from engine.research.agent_council import AgentVerdict

    rng = random.Random(seed)
    families = list(_FAMILY_TRUE_P_GREEN.keys())
    critics  = list(_CRITIC_FAMILY_ACCURACY.keys())

    rows: list[dict] = []
    base_time = _dt.datetime(2026, 5, 1, 12, 0, 0)

    for i in range(n):
        family = rng.choice(families)
        p_green = _FAMILY_TRUE_P_GREEN[family]
        true_green = rng.random() < p_green

        # Each critic emits a verdict
        verdicts = [
            AgentVerdict(
                agent_name=c,
                verdict=_critic_verdict(c, family, true_green, rng),
                confidence=0.7,
                rationale=f"synthetic (true_green={true_green})",
            )
            for c in critics
        ]
        consensus, _rationale = aggregate_verdicts(verdicts)
        pipeline_decision = _pipeline_decision(true_green, rng)

        rows.append({
            "ts":            (base_time + _dt.timedelta(hours=i*6)).isoformat(timespec="seconds") + "Z",
            "iteration_id":  f"sim-{seed}-{i:04d}",
            "workflow_id":   f"wf-sim-{seed}-{i:04d}",
            "proposal":      {
                "family":        family,
                "proposed_role": "alpha_seeker",
                "title":         f"synthetic_{family}_{i}",
            },
            "council":       {
                "consensus": consensus,
                "rationale": f"synthetic seed={seed} idx={i}",
                "verdicts":  [v.to_dict() for v in verdicts],
                "run_id":    f"run-sim-{seed}-{i:04d}",
            },
            "pipeline":      {
                "ran":            True,
                "final_decision": pipeline_decision,
                "rationale":      f"synthetic ground-truth (true_green={true_green})",
                "step_results":   [],
            },
            "_simulator_meta": {
                "true_green": true_green,
                "family":     family,
                "seed":       seed,
                "idx":        i,
            },
        })

    return rows


def _cleanup_synthetic_rows() -> dict:
    """Remove rows with _simulator_meta from both ledgers."""
    import os
    from engine.research.outcome_ledger import L4_LEDGER_PATH
    from engine.research.critic_calibration import CRITIC_CALIBRATION_LEDGER

    removed = {"l4_iterations": 0, "critic_calibration": 0}

    # Filter l4_iterations.jsonl
    if L4_LEDGER_PATH.is_file():
        rows = []
        n_removed = 0
        for line in L4_LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                rows.append(line)  # preserve malformed lines verbatim
                continue
            if "_simulator_meta" in r or (r.get("workflow_id") or "").startswith("wf-sim-") or (r.get("iteration_id") or "").startswith("sim-"):
                n_removed += 1
                continue
            rows.append(json.dumps(r, default=str))
        tmp = L4_LEDGER_PATH.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(rows) + ("\n" if rows else ""),
                        encoding="utf-8")
        os.replace(tmp, L4_LEDGER_PATH)
        removed["l4_iterations"] = n_removed

    # Filter critic_calibration.jsonl
    if CRITIC_CALIBRATION_LEDGER.is_file():
        rows = []
        n_removed = 0
        for line in CRITIC_CALIBRATION_LEDGER.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                rows.append(line)
                continue
            # critic_calibration rows have iteration_id from outcome_ledger
            # which is auto-generated. Match on the synthetic title pattern
            # in proposal_title (carried through from append).
            # NOTE: this only works for rows from THIS run; old leftover
            # rows may not match. The l4_iterations pass above is the
            # authoritative one.
            title = (r.get("proposal_title") or
                     (r.get("proposal") or {}).get("title") or "")
            family = r.get("family") or ""
            if (title.startswith("synthetic_") or
                family in ("text_ml", "vol_carry") and r.get("critic_agent_name") in (
                    "behavioral_theorist", "empirical_devils_advocate"
                ) and r.get("critic_confidence") == 0.7):
                n_removed += 1
                continue
            rows.append(json.dumps(r, default=str))
        tmp = CRITIC_CALIBRATION_LEDGER.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(rows) + ("\n" if rows else ""),
                        encoding="utf-8")
        os.replace(tmp, CRITIC_CALIBRATION_LEDGER)
        removed["critic_calibration"] = n_removed

    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30,
                        help="number of synthetic iterations to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="print rows without writing to ledger")
    parser.add_argument("--cleanup", action="store_true",
                        help="remove all previously-written synthetic rows from real ledgers")
    args = parser.parse_args()

    if args.cleanup:
        removed = _cleanup_synthetic_rows()
        print(f"Removed synthetic rows: {removed}")
        return 0

    print(f"Generating {args.n} synthetic L4 iterations (seed={args.seed})...")
    rows = simulate_iterations(args.n, seed=args.seed)

    if args.dry_run:
        print(json.dumps(rows[:3], indent=2, default=str))
        print(f"...and {len(rows) - 3} more rows")
        return 0

    # Append to the real ledgers (so all downstream tools see the data)
    from engine.research.outcome_ledger import append_l4_iteration
    n_written = 0
    for row in rows:
        try:
            append_l4_iteration(
                workflow_id=row["workflow_id"],
                proposal=row["proposal"],
                council=row["council"],
                pipeline_report=row["pipeline"],
                elapsed_s=1.0,
            )
            n_written += 1
        except Exception as exc:
            print(f"  warn: row {row.get('iteration_id')} failed: {exc}")

    print(f"Wrote {n_written}/{args.n} rows to outcome_ledger + auto-emitted")
    print("       per-critic rows to critic_calibration ledger.")
    print()
    print("Verify with:")
    print("    python -m engine.research.critic_calibration report 365")
    print()
    print("Or open /lab/outcomes in the UI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
