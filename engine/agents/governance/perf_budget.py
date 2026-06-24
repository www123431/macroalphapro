"""engine/agents/governance/perf_budget.py — Phase 4 performance / cost governance.

Two deterministic, zero-token controls (blueprint spec id=78 Phase 4):

  A. CACHE-SURFACE report — Anthropic prompt caching is already wired (call(cache_system=
     True) wraps the system prompt in ephemeral cache_control when it clears the model's
     minimum). This report makes the lever VISIBLE: per persona, estimate the system-prompt
     tokens and flag any prompt TOO SHORT to cache (a silent cost leak — every turn re-pays
     full input cost). Caching the long persona prompts saves ~90% of repeat input cost.

  B. SLO targets + compliance — role-aware targets (interactive chat vs batch cron) +
     a checker over the EXISTING engine.agents.observability metrics
     (data/agent_slo_metrics.jsonl: latency_ms, success). Reuses the infra; adds the
     deterministic report/gate layer. (Cost-per-turn lives in the cost ledger; the cost
     lever here is the cache surface, not a fabricated per-turn number.)

0-LLM, deterministic. Token estimate = chars/4 (documented heuristic; no tokenizer dep).
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Anthropic ephemeral-cache minimums (per engine.llm.providers.anthropic_provider docs).
CACHE_MIN_TOKENS = {"claude-haiku-4-5": 1024}
CACHE_MIN_DEFAULT_SONNET = 2048          # Sonnet 4.6
CHARS_PER_TOKEN = 4                       # documented estimate (no tokenizer dependency)

METRICS_PATH = Path("data/agent_slo_metrics.jsonl")


def _est_tokens(text: str) -> int:
    return len(text or "") // CHARS_PER_TOKEN


def _cache_min(model: str) -> int:
    return CACHE_MIN_TOKENS.get(model, CACHE_MIN_DEFAULT_SONNET)


def cache_surface_report() -> dict:
    """Per persona: system-prompt token estimate, the model's cache minimum, eligibility,
    and a flag if the prompt is too short to cache (cost leak)."""
    from engine.agents.eval.runner import _personas
    try:
        from engine.llm.call import _WORKLOAD_ROUTING
    except Exception:
        _WORKLOAD_ROUTING = {}
    rows = {}
    for aid, p in sorted(_personas().items()):
        provider, model = _WORKLOAD_ROUTING.get(p.workload, ("?", "?"))
        est = _est_tokens(p.system_prompt)
        if provider == "anthropic":
            mn = _cache_min(model)
            eligible = est >= mn
            # est is a chars/4 ESTIMATE vs the CONFIGURED min; ground truth = the cost
            # ledger's cache_read_tokens. Frame as verify, not a definitive leak verdict.
            note = ("cache-eligible (est >= configured min)" if eligible else
                    f"est < configured min ({mn} tok) — VERIFY empirically (ledger cache_read)")
        else:
            eligible, mn, note = None, None, f"provider-managed ({provider})"
        rows[aid] = {"provider": provider, "model": model, "est_prompt_tokens": est,
                     "cache_min_tokens": mn, "cache_eligible": eligible, "note": note}
    return rows


# ── SLO targets (role-aware) ─────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class SLOTarget:
    latency_p95_ms: int
    min_success_rate: float
    role: str


SLO_DEFAULT = SLOTarget(latency_p95_ms=60_000, min_success_rate=0.95, role="interactive")
# Crons / batch agents legitimately run minutes, not seconds — don't false-flag them.
SLO_OVERRIDES = {
    "ops_watchdog": SLOTarget(latency_p95_ms=600_000, min_success_rate=0.90, role="batch_cron"),
    "etf_holdings": SLOTarget(latency_p95_ms=600_000, min_success_rate=0.90, role="batch_cron"),
}


def _target_for(agent_id: str) -> SLOTarget:
    return SLO_OVERRIDES.get(agent_id, SLO_DEFAULT)


def _p95(values: list) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, int(round(0.95 * (len(s) - 1))))
    return float(s[k])


def check_slo(metrics_path: Path = METRICS_PATH) -> dict:
    """Group the recorded invocations by agent_id; compute latency p95 + success rate;
    compare to the role-aware target. Returns per-agent compliance."""
    if not metrics_path.exists():
        return {"available": False, "reason": f"no metrics at {metrics_path}", "agents": {}}
    by_agent: dict = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        a = r.get("agent_id", "?")
        d = by_agent.setdefault(a, {"lat": [], "ok": 0, "n": 0})
        if r.get("latency_ms") is not None:
            d["lat"].append(float(r["latency_ms"]))
        d["n"] += 1
        d["ok"] += 1 if r.get("success") else 0
    agents = {}
    for a, d in sorted(by_agent.items()):
        t = _target_for(a)
        p95 = _p95(d["lat"])
        sr = d["ok"] / d["n"] if d["n"] else float("nan")
        lat_ok = (p95 != p95) or p95 <= t.latency_p95_ms          # nan -> insufficient, not a fail
        sr_ok = (sr != sr) or sr >= t.min_success_rate
        low_sample = d["n"] < 20                                   # breach is tentative at small n
        agents[a] = {"role": t.role, "n": d["n"], "latency_p95_ms": None if p95 != p95 else p95,
                     "success_rate": None if sr != sr else round(sr, 3),
                     "target_p95_ms": t.latency_p95_ms, "target_success": t.min_success_rate,
                     "latency_ok": bool(lat_ok), "success_ok": bool(sr_ok), "low_sample": low_sample,
                     "compliant": bool(lat_ok and sr_ok)}
    return {"available": True, "agents": agents,
            "all_compliant": all(v["compliant"] for v in agents.values()) if agents else True}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("\n" + "=" * 78 + "\nPERF/COST — A. prompt-cache surface (cost lever; est=chars/4)\n" + "=" * 78)
    to_verify = []
    for aid, r in cache_surface_report().items():
        mark = "vfy " if r["cache_eligible"] is False else "ok  "
        if r["cache_eligible"] is False:
            to_verify.append(aid)
        print(f"  [{mark}] {aid:20s} ~{r['est_prompt_tokens']:5d} tok  {r['note']}")
    if to_verify:
        print(f"  -> verify caching empirically (ledger cache_read_tokens) for: {to_verify}")
    print("\n" + "=" * 78 + "\nPERF/COST — B. SLO compliance (role-aware, over recorded metrics)\n" + "=" * 78)
    slo = check_slo()
    if not slo["available"]:
        print(f"  {slo['reason']}")
    else:
        for a, v in slo["agents"].items():
            mark = "PASS" if v["compliant"] else "FAIL"
            ls = " [low-sample, tentative]" if v.get("low_sample") else ""
            print(f"  [{mark}] {a:18s} ({v['role']}) n={v['n']} p95={v['latency_p95_ms']}ms "
                  f"(<= {v['target_p95_ms']}) success={v['success_rate']} (>= {v['target_success']}){ls}")
        print(f"\n  all_compliant={slo['all_compliant']}")
    # to_verify is advisory (estimate-based), not a hard failure; SLO breach is the gate.
    return 1 if (slo.get("available") and not slo["all_compliant"]) else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
