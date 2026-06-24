"""engine.agents.autopilot — L4 research-autopilot decision layer (F14).

A+B substrate-first plan locked 2026-06-05 (see memory file):
  Phase F14a (THIS MODULE, read-only) — dry-run preview. NO compose(),
        NO pipeline calls, NO LLM spend. Reads the catalog + redundancy
        reports and writes a markdown of "what cron WOULD run tonight".
        User soaks this output ~1 week to validate selection logic
        BEFORE any code auto-runs.

  Phase F14b (later, gated by 1-week clean soak) — limited live auto-
        run with $5/night ceiling, 2-candidate cap, full audit tag.

  Phase F14c (later, gated by 2-week clean F14b) — scale to 5/night.

Design invariants
-----------------
- READ-ONLY. compute_dry_run_plan() never calls compose(), pipeline,
  or any LLM. Pure metadata transformation over the F13 catalog +
  redundancy reports.
- DETERMINISTIC. Same corpus state → same plan output. No randomness.
- HONEST. Skipped candidates carry the reason in plain English. No
  silent filters.
- BOUNDED. top_n cap on the candidate pool. Auto-skip if redundancy
  STRONG WARN. No "expand selection if cap not met" — better to test
  fewer good candidates than dilute with marginal ones.

Selection logic (the rule cron will eventually follow)
------------------------------------------------------
1. Get all composer-ready cells from build_catalog().
2. Sort by (convergence priority desc) then (n_specs_ready desc):
   convergence priority = n_papers when cell is in convergence_clusters,
                          else 0.
3. For each ready spec in the top cells, run find_redundancy_for_spec.
4. Decisions:
     advice = STRONG WARN  → action = WOULD_SKIP_REDUNDANCY
     advice ∈ (WARN, INFO) → action = WOULD_TEST (with caveat in note)
     no match              → action = WOULD_TEST
5. Cap at top_n (default 5) AFTER skip filter so a STRONG WARN
   doesn't reduce the effective batch.
6. Estimate per-candidate: compose ~60s elapsed (cache miss),
   pipeline ~60-90s (no LLM, mostly stats). Cost $0 (no LLM in compose
   or pipeline today; only DA step in pipeline uses LLM but that's
   guarded behind a flag).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AUTOPILOT_DIR = _REPO_ROOT / "data" / "autopilot"

# Per-candidate cost / latency estimates (used for digest header)
_PER_CANDIDATE_USD = 0.0      # compose + pipeline are no-LLM today;
                              # devils_advocate inside pipeline may use
                              # LLM but is currently behind a flag
_PER_CANDIDATE_SEC = 90       # rough median: ~60s composer + ~30s pipeline


@dataclass(frozen=True)
class CandidateDecision:
    """One candidate spec + the cron's decision about it.

    Persisted in the dry-run markdown so user can audit the selection
    logic each morning before agreeing the rule is right.
    """
    rank:                int
    source_hypothesis_id: str
    spec_hash:           str
    family:              str
    signal_type:         str
    universe_subset:     str
    weighting:           str
    rebalance:           str
    claim_preview:       str
    action:              str          # "WOULD_TEST" | "WOULD_SKIP_REDUNDANCY"
    reason:              str
    redundancy_advice:   Optional[str] = None
    redundancy_n_red:    int = 0
    cell_n_papers:       int = 0
    cell_in_convergence: bool = False


@dataclass(frozen=True)
class DryRunPlan:
    plan_ts:           str
    n_ready_specs:     int
    n_would_test:      int
    n_would_skip:      int
    estimated_cost_usd: float
    estimated_wall_s:  int
    decisions:         tuple[CandidateDecision, ...]


def _latest_specs_by_hyp() -> dict:
    """{source_hypothesis_id -> HypothesisSpec} picking latest by
    extracted_ts. Mirrors F13.1's helper but returns the dataclass,
    not raw dicts."""
    from engine.hypothesis_spec.store import all_specs
    by_id = {}
    by_id_ts: dict = {}
    for s in all_specs():
        hid = s.source_hypothesis_id
        ts = s.extraction.extracted_ts or s.created_ts or ""
        if hid not in by_id_ts or ts > by_id_ts[hid]:
            by_id_ts[hid] = ts
            by_id[hid] = s
    return by_id


def compute_dry_run_plan(top_n: int = 5, per_cell_cap: int = 2) -> DryRunPlan:
    """The F14a core. Read-only. Decides what cron WOULD run tonight
    if it were live, with full reasoning.

    per_cell_cap: max picks per (family, signal_type) cell. Default 2
    forces top-N to span >= N/2 distinct cells. Lowered to 1 in
    perturbation tests (autopilot_perturbation_tests.py) to assert
    diversity behavior responds to the parameter."""
    from engine.hypothesis_spec.enums import ClaimType
    from engine.hypothesis_spec.hash import spec_hash
    from engine.composer.contract import is_spec_covered
    from engine.research_store.mechanism_catalog import (
        build_catalog, find_redundancy_for_spec, convergence_clusters,
    )

    catalog = build_catalog()
    convergence = {(r.family, r.signal_type)
                   for r in convergence_clusters(min_papers=1, catalog=catalog)}
    catalog_by_key = {(r.family, r.signal_type): r for r in catalog}

    # 1. Collect ready FACTOR_HYPOTHESIS specs with their cell membership
    specs_by_hyp = _latest_specs_by_hyp()
    candidates: list[tuple] = []   # (priority_tuple, spec)
    for s in specs_by_hyp.values():
        if s.claim_type != ClaimType.FACTOR_HYPOTHESIS:
            continue
        if not s.legs:
            continue
        try:
            covered, _gaps = is_spec_covered(s)
        except Exception:
            continue
        if not covered:
            continue
        fam = s.family.value
        sig = s.legs[0].signal_type.value
        cell = catalog_by_key.get((fam, sig))
        if cell is None:
            # Cell-less spec — possible when spec was just added and
            # catalog snapshot pre-dates it. Treat as low priority.
            n_papers = 0
            in_conv = False
        else:
            n_papers = cell.n_papers
            in_conv = (fam, sig) in convergence
        # Priority tuple: convergence-first, then by n_papers, then by
        # hyp_id (stable tie-breaker so re-runs produce same ordering).
        # MUST use a hashlib hash, NOT Python's built-in hash() — the
        # latter is randomized per process by PYTHONHASHSEED, so two
        # cron invocations would rank the same corpus differently.
        # Perturbation test T1 caught this on 2026-06-05.
        priority = (
            1 if in_conv else 0,
            n_papers,
            int(hashlib.md5(s.source_hypothesis_id.encode()).hexdigest()[:8], 16),
        )
        candidates.append((priority, s, cell, n_papers, in_conv))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # 2. Per-candidate redundancy check + decision
    # diversity cap (F14a v2 2026-06-05): force at most `per_cell_cap`
    # picks per (family, signal_type) cell so the daily candidate set
    # surfaces cross-cell variety. Without this, a cell with 8 ready
    # specs (like PROFITABILITY/PROFITABILITY_GROSS) would fill the
    # entire top-N. With cap=2, top-N spans ≥ N/2 distinct cells.
    cell_picks: dict[tuple, int] = {}
    decisions: list[CandidateDecision] = []
    n_test = 0
    n_skip = 0
    for prio, s, cell, n_papers, in_conv in candidates:
        if n_test >= top_n:
            break
        fam = s.family.value
        sig = s.legs[0].signal_type.value
        if cell_picks.get((fam, sig), 0) >= per_cell_cap:
            continue   # cap hit on this cell; let next cell get a slot
        spec_dict = s.to_dict()
        matches = find_redundancy_for_spec(spec_dict, catalog=catalog)
        # Worst-case advice
        advice = None
        n_red = 0
        if matches:
            advice = matches[0].advice
            n_red = matches[0].n_red_in_cluster

        if advice and "STRONG WARN" in advice:
            action = "WOULD_SKIP_REDUNDANCY"
            reason = (f"cluster has {n_red} prior RED verdicts; "
                       f"auto-test would re-confirm what's already killed")
            n_skip += 1
        else:
            action = "WOULD_TEST"
            if advice and "WARN" in advice:
                reason = f"cluster has prior REDs but deployed alongside; verify novelty after run"
            elif advice and "INFO" in advice:
                reason = "cluster has REDs alongside deployed; verifying"
            else:
                reason = "clean cluster; first-test of this (family, signal)"
            cell_picks[(fam, sig)] = cell_picks.get((fam, sig), 0) + 1
            n_test += 1

        primary_leg = s.legs[0]
        decisions.append(CandidateDecision(
            rank                = len(decisions) + 1,
            source_hypothesis_id = s.source_hypothesis_id,
            spec_hash            = spec_hash(s),
            family               = s.family.value,
            signal_type          = primary_leg.signal_type.value,
            universe_subset      = f"{s.universe.asset_class.value}/{s.universe.subset.value}",
            weighting            = s.construction.weighting.value,
            rebalance            = s.construction.rebalance.value,
            claim_preview        = (s.claim_text or "")[:160],
            action               = action,
            reason               = reason,
            redundancy_advice    = advice,
            redundancy_n_red     = n_red,
            cell_n_papers        = n_papers,
            cell_in_convergence  = in_conv,
        ))

    plan_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    n_ready_specs = sum(1 for _, s, _, _, _ in candidates)
    return DryRunPlan(
        plan_ts            = plan_ts,
        n_ready_specs      = n_ready_specs,
        n_would_test       = n_test,
        n_would_skip       = n_skip,
        estimated_cost_usd = round(n_test * _PER_CANDIDATE_USD, 2),
        estimated_wall_s   = n_test * _PER_CANDIDATE_SEC,
        decisions          = tuple(decisions),
    )


def render_markdown(plan: DryRunPlan) -> str:
    """Render the plan as user-readable markdown. Goes to
    data/autopilot/<date>.md + the daily directive section."""
    today = plan.plan_ts[:10]
    lines: list[str] = []
    lines.append(f"# Autopilot dry-run · {today}")
    lines.append("")
    lines.append("**This is what cron WOULD run if F14b were live. No code auto-ran.**")
    lines.append("")
    lines.append(f"- Ready FACTOR_HYPOTHESIS specs in corpus: **{plan.n_ready_specs}**")
    lines.append(f"- Would test: **{plan.n_would_test}**")
    lines.append(f"- Would skip (redundancy): **{plan.n_would_skip}**")
    lines.append(f"- Estimated cost: **${plan.estimated_cost_usd:.2f}** (LLM-free pipeline today)")
    lines.append(f"- Estimated wall-clock: **{plan.estimated_wall_s // 60}m {plan.estimated_wall_s % 60}s**")
    lines.append("")
    lines.append("## Candidates")
    lines.append("")
    for d in plan.decisions:
        action_emoji = "▶" if d.action == "WOULD_TEST" else "⊘"
        conv_mark = " *(convergence)*" if d.cell_in_convergence else ""
        lines.append(f"### {action_emoji} {d.rank}. {d.family}/{d.signal_type}{conv_mark}")
        lines.append("")
        lines.append(f"- **action**: `{d.action}`")
        lines.append(f"- **reason**: {d.reason}")
        lines.append(f"- **spec_hash**: `{d.spec_hash}`  hyp_id: `{d.source_hypothesis_id[:8]}`")
        lines.append(f"- **universe**: {d.universe_subset}  weighting: {d.weighting}  rebalance: {d.rebalance}")
        lines.append(f"- **cell**: {d.cell_n_papers} prior papers"
                      + (f", {d.redundancy_n_red} prior REDs" if d.redundancy_n_red else ""))
        if d.redundancy_advice:
            lines.append(f"- **redundancy_advice**: {d.redundancy_advice}")
        lines.append("")
        lines.append(f"> {d.claim_preview}")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Soak instructions**: review for ~1 week. If selection logic "
                  "produces the right candidates (vs your manual judgment), "
                  "advance to F14b (limited live auto-run).")
    return "\n".join(lines)


def write_dry_run_to_disk(plan: DryRunPlan) -> Path:
    """Persist the markdown to data/autopilot/<date>.md. Overwrites
    today's file (idempotent re-runs). Also creates a 'latest.md'
    symlink-equivalent (just copies the bytes; symlinks unreliable
    on Windows)."""
    _AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)
    today = plan.plan_ts[:10]
    out_path = _AUTOPILOT_DIR / f"{today}.md"
    md = render_markdown(plan)
    out_path.write_text(md, encoding="utf-8")
    latest = _AUTOPILOT_DIR / "latest.md"
    latest.write_text(md, encoding="utf-8")
    logger.info("F14a dry-run written: %s (%d candidates, %d would test)",
                 out_path, len(plan.decisions), plan.n_would_test)
    return out_path
