"""engine/agents/eval/runner.py — static (CI) + live (gated) agent-eval runners.

  Tier A  run_static_eval()  — no LLM, runs in CI: behavioral CONTRACT per persona.
  Tier B  run_live_eval()    — calls chat_turn per held-out case, scores deterministically.
                               Gated: only via --live / RUN_AGENT_EVAL=1 (costs tokens).

CLI: python -m engine.agents.eval.runner            # static contract report
     python -m engine.agents.eval.runner --live     # + live behavioral eval (needs keys)
     python -m engine.agents.eval.runner --live --agent decay_sentinel
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

from engine.agents.eval.cases import cases_for
from engine.agents.eval.contract import score_static_contract, score_turn

logger = logging.getLogger(__name__)
OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "validation"


def _personas() -> dict:
    """agent_id -> persona singleton (all built personas)."""
    import engine.agents.persona as p
    out = {}
    for attr in ("RISK_MANAGER", "DQ_INSPECTOR", "DEVILS_ADVOCATE", "ANOMALY_SENTINEL",
                 "ATTRIBUTION_ANALYST", "AUDIT_RECORDER", "CHIEF_OF_STAFF", "DECAY_SENTINEL"):
        per = getattr(p, attr, None)
        if per is not None:
            out[per.agent_id] = per
    return out


# ── Tier A: static contract ──────────────────────────────────────────────────
def run_static_eval(agent_id: str | None = None) -> dict:
    personas = _personas()
    items = {k: v for k, v in personas.items() if agent_id in (None, k)}
    report, all_pass = {}, True
    for aid, persona in sorted(items.items()):
        results = score_static_contract(persona)
        report[aid] = [dataclasses.asdict(r) for r in results]
        all_pass = all_pass and all(r.passed for r in results)
    return {"tier": "static_contract", "all_pass": all_pass, "report": report}


# ── Tier B: live behavioral eval (gated, statistical) ────────────────────────
def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial pass-rate (small-n honest, unlike normal approx)."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def _run_one_turn(persona, prompt: str, expect, max_iterations: int):
    """One live turn with a RECORDING executor that captures FULL tool outputs (not the
    200-char preview) for the grounding check."""
    import dataclasses as _dc
    from engine.agents.persona.base import chat_turn
    records: list[str] = []
    inner = persona.tool_executor

    def recording_executor(name, tool_input):
        out, is_err = inner(name, tool_input)
        records.append(f"{name} {tool_input} {out}")     # FULL output, not truncated
        return out, is_err

    bounded = _dc.replace(persona, tool_executor=recording_executor,
                          max_iterations=min(max_iterations, persona.max_iterations))
    turn = chat_turn(bounded, prompt, history=[])
    scored = score_turn(turn, prompt, list(expect), tool_haystack=" ".join(records))
    return turn, scored


def run_live_eval(agent_id: str | None = None, n_samples: int = 3, max_iterations: int = 4) -> dict:
    """Tier B: each case run n_samples times (outputs are STOCHASTIC); per-case pass-rate +
    Wilson CI. Monitored metric, NOT a flaky CI gate (Tier A is the hard gate)."""
    personas = _personas()
    cases = cases_for(agent_id)
    results = []
    tot_runs = tot_pass_runs = 0          # a "run" passes iff ALL its expectations pass
    total_cost = 0.0
    for c in cases:
        persona = personas.get(c.agent_id)
        if persona is None:
            results.append({"case_id": c.case_id, "error": f"no persona {c.agent_id}"})
            continue
        run_pass = 0; samples = []
        for _ in range(max(1, n_samples)):
            try:
                turn, scored = _run_one_turn(persona, c.prompt, c.expect, max_iterations)
                ok = all(s.passed for s in scored)
                run_pass += int(ok); total_cost += turn.total_cost_usd
                samples.append({"all_pass": ok,
                                "fails": [f"{s.kind}:{s.detail}" for s in scored if not s.passed],
                                "tools": [tc.get("name") for tc in turn.tool_calls_log],
                                "answer": turn.final_text[:300]})
            except Exception as exc:
                logger.exception("live eval %s failed", c.case_id)
                samples.append({"error": str(exc)})
        n = sum(1 for s in samples if "error" not in s)
        tot_runs += n; tot_pass_runs += run_pass
        results.append({"case_id": c.case_id, "agent_id": c.agent_id,
                        "n": n, "pass": run_pass,
                        "pass_rate": round(run_pass / n, 3) if n else None,
                        "wilson_ci": _wilson_ci(run_pass, n), "samples": samples})
    return {"tier": "live_behavioral", "n_cases": len(cases), "n_samples": n_samples,
            "runs": tot_runs, "runs_passed": tot_pass_runs,
            "pass_rate": round(tot_pass_runs / tot_runs, 3) if tot_runs else None,
            "wilson_ci": _wilson_ci(tot_pass_runs, tot_runs),
            "total_cost_usd": round(total_cost, 4), "cases": results}


def main() -> int:
    ap = argparse.ArgumentParser(description="agent behavioral eval harness")
    ap.add_argument("--live", action="store_true", help="run the live LLM eval (needs API keys; costs tokens)")
    ap.add_argument("--agent", default=None, help="filter to one agent_id")
    ap.add_argument("--n-samples", type=int, default=3, help="Tier B samples per case (stochastic outputs)")
    ap.add_argument("--save", action="store_true", help="persist JSON report")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    static = run_static_eval(args.agent)
    print("\n" + "=" * 78)
    print(f"AGENT EVAL — Tier A static contract | all_pass={static['all_pass']}")
    print("=" * 78)
    for aid, checks in static["report"].items():
        bad = [c["check"] for c in checks if not c["passed"]]
        print(f"  {'PASS' if not bad else 'FAIL':4s}  {aid:20s}" + (f"  failed: {bad}" if bad else ""))

    payload = {"static": static}
    run_live = args.live or os.environ.get("RUN_AGENT_EVAL") == "1"
    if run_live:
        print("\n" + "=" * 78)
        print("AGENT EVAL — Tier B live behavioral (calling the model)")
        print("=" * 78)
        live = run_live_eval(args.agent, n_samples=args.n_samples)
        payload["live"] = live
        for c in live["cases"]:
            if "error" in c:
                print(f"  ERR  {c['case_id']}: {c['error']}"); continue
            ci = c["wilson_ci"]
            print(f"  {c['case_id']:34s} {c['pass']}/{c['n']} "
                  f"(rate {c['pass_rate']}, 95% CI {ci})")
            for s in c["samples"]:
                if s.get("fails"):
                    print(f"        MISS {s['fails']}")
        pr, ci = live["pass_rate"], live["wilson_ci"]
        print(f"\n  LIVE all-pass runs: {live['runs_passed']}/{live['runs']} = {pr} "
              f"(95% CI {ci}); cost ${live['total_cost_usd']}")
    else:
        print("\n  (Tier B live eval skipped — pass --live or RUN_AGENT_EVAL=1 to run it; costs tokens.)")

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "agent_eval_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nsaved {out}")
    return 0 if static["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
