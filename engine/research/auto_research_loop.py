"""engine/research/auto_research_loop.py — autonomous research loop v1.

Phase 1 Task II.A of `docs/decisions/research_agenda_2026-05-29.md`. The
Karpathy AutoResearch pattern, adapted to our strict-gate doctrine.

The loop:
  1. Read a SKILL contract (docs/skills/<name>.skill.yaml).
  2. Identify the parent version (highest in version_history with decision !=
     "rollback").
  3. Propose ONE parameter modification within the SKILL's allowed ranges.
  4. Build the strategy variant by passing modified params to the SKILL's
     generator (via a thin wrapper — public code stays untouched).
  5. Evaluate on the VAL window only (train_end < dates ≤ val_end).
  6. Decide keep / rollback per the SKILL's rollback_triggers.
  7. Append the result to the SKILL yaml + a versioned JSON in
     data/research/skill_versions/<name>/<version>.json.
  8. Halt if N consecutive rollbacks.

Hard constraints (assert at every iteration):
  - TEST WINDOW (val_end+1 onwards) is NEVER passed to the gate.
  - Gate thresholds (HLZ_T, DEFLSR_MIN, MAX_BOOK_CORR) match
    engine.research.pipeline globals — no per-skill override.
  - Proposed params NEVER go out of declared range.
  - locked_parameters never appear in the proposed delta.
  - Public code (engine.portfolio.*, engine.validation.*) is never written
    to by the loop.

Usage:
    from engine.research.auto_research_loop import run_one_iteration
    res = run_one_iteration("equity_book")
    print(res["decision"], res["new_version"], res["val_metrics"]["sharpe"])

This is v1: deterministic proposer (rule-based, samples one param uniformly
within its range/step). v2 will plug in an LLM persona (Devil's Advocate
or Quant Engineer) for proposal generation.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "docs" / "skills"
VERSIONS_DIR = REPO_ROOT / "data" / "research" / "skill_versions"


# ─── Loader / writer ─────────────────────────────────────────────────────────

def load_skill(skill_name: str) -> dict:
    """Read docs/skills/<name>.skill.yaml. Validates basic structure."""
    yaml_path = SKILLS_DIR / f"{skill_name}.skill.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"SKILL not found: {yaml_path}")
    skill = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    for key in ("skill_name", "parameters", "evaluation", "rollback_triggers",
                "version_history"):
        if key not in skill:
            raise ValueError(f"SKILL {skill_name} missing required key: {key}")
    return skill


def save_skill(skill: dict) -> None:
    """Write back, preserving comments NOT possible with yaml.safe_dump.
    For v1: appends version-history entry only (other fields untouched at
    runtime). v2 would use ruamel.yaml for round-trip preservation."""
    yaml_path = SKILLS_DIR / f"{skill['skill_name']}.skill.yaml"
    # We append history without rewriting the entire file. Strategy: rewrite
    # only the version_history block, preserve everything else as text.
    txt = yaml_path.read_text(encoding="utf-8")
    history_yaml = yaml.safe_dump(
        {"version_history": skill["version_history"]},
        default_flow_style=False, sort_keys=False, allow_unicode=True)
    # Find existing version_history block (last section by convention) and
    # replace it. Simple-but-robust: split on the literal "version_history:"
    # line.
    marker = "\nversion_history:"
    idx = txt.find(marker)
    if idx < 0:
        # No existing block (shouldn't happen, but) — append.
        new_txt = txt.rstrip() + "\n\n" + history_yaml
    else:
        new_txt = txt[:idx + 1] + history_yaml
    yaml_path.write_text(new_txt, encoding="utf-8")


# ─── Sample isolation enforcer ───────────────────────────────────────────────

def enforce_sample_isolation(returns: pd.Series, skill: dict) -> pd.Series:
    """Clip a monthly returns series to the VAL window (train_end < date ≤
    val_end). The orchestrator MUST pass this clipped series to the gate.

    Asserts that the returned series does not extend past val_end, even by
    one day. This is the load-bearing guard against test-set leakage.
    """
    train_end = pd.Timestamp(skill["evaluation"]["train_end"])
    val_end = pd.Timestamp(skill["evaluation"]["val_end"])
    if val_end <= train_end:
        raise ValueError(f"val_end {val_end} must be > train_end {train_end}")
    r = returns.dropna()
    r.index = pd.to_datetime(r.index)
    val_slice = r[(r.index > train_end) & (r.index <= val_end)]
    # Hard assertion — no entry past val_end can leak through
    if not val_slice.empty:
        assert val_slice.index.max() <= val_end, \
            f"TEST LEAK: val slice extends to {val_slice.index.max()} > val_end {val_end}"
    return val_slice


# ─── Proposer (v1: rule-based, single-param tweak) ───────────────────────────

def propose_change(skill: dict, rng_seed: int | None = None) -> dict[str, Any]:
    """v1 proposer: pick ONE parameter from the SKILL's tunable list, sample
    a step away from its parent (or baseline) value within the declared range.

    Returns a dict of proposed param values (full param set, not just delta).
    """
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

    # Find the parent version's params (last entry where decision != rollback)
    parent = None
    for ver in reversed(skill["version_history"]):
        if ver.get("decision") in ("baseline", "keep"):
            parent = ver
            break
    if parent is None:
        raise RuntimeError(f"SKILL {skill['skill_name']}: no kept parent version")
    parent_params = dict(parent["params"])

    # Pick one tunable parameter
    tunables = list(skill["parameters"].keys())
    pick = rng.choice(tunables)
    pspec = skill["parameters"][pick]

    proposed_params = dict(parent_params)
    cur = parent_params.get(pick, pspec.get("baseline"))

    if pspec["type"] == "float":
        lo, hi = pspec["range"]
        step = pspec.get("step", (hi - lo) / 10.0)
        # Try a step up OR down from current; if at the boundary, the other way
        direction = rng.choice([-1, 1])
        candidate = round(cur + direction * step, 6)
        if not (lo <= candidate <= hi):
            candidate = round(cur - direction * step, 6)
        if not (lo <= candidate <= hi):
            candidate = cur   # no movement possible; will likely be a no-op
        proposed_params[pick] = candidate
    elif pspec["type"] == "enum":
        opts = [o for o in pspec["options"] if o != cur]
        if not opts:
            proposed_params[pick] = cur
        else:
            proposed_params[pick] = rng.choice(opts)
    elif pspec["type"] == "int":
        lo, hi = pspec["range"]
        step = int(pspec.get("step", 1))
        direction = rng.choice([-1, 1])
        candidate = int(cur) + direction * step
        if not (lo <= candidate <= hi):
            candidate = int(cur) - direction * step
        if not (lo <= candidate <= hi):
            candidate = int(cur)
        proposed_params[pick] = candidate
    else:
        raise ValueError(f"Unsupported param type: {pspec['type']}")

    # Apply explicit cross-param constraint check
    cons = pspec.get("constraint")
    if cons == "q_out > q_in":
        if proposed_params.get("revision_q_out", 0) <= proposed_params.get("revision_q_in", 0):
            # Reject — propose no-op so the loop tries a different param next iter
            proposed_params = dict(parent_params)
            pick = "(constraint-rejected, no-op this iteration)"

    return {"params": proposed_params, "changed": pick,
            "parent_version": parent["version"]}


# ─── Bounds enforcement ──────────────────────────────────────────────────────

def assert_within_bounds(params: dict, skill: dict) -> None:
    """Hard guard against proposer bugs."""
    for k, v in params.items():
        if k not in skill["parameters"]:
            # Param not in tunable surface (shouldn't have been changed)
            continue
        ps = skill["parameters"][k]
        if ps["type"] == "float":
            lo, hi = ps["range"]
            assert lo <= v <= hi, f"PROPOSER BUG: {k}={v} outside [{lo},{hi}]"
        elif ps["type"] == "enum":
            assert v in ps["options"], f"PROPOSER BUG: {k}={v!r} not in {ps['options']}"
        elif ps["type"] == "int":
            lo, hi = ps["range"]
            assert lo <= v <= hi, f"PROPOSER BUG: {k}={v} outside [{lo},{hi}]"


# ─── Build + evaluate ────────────────────────────────────────────────────────

def _build_equity_book_with_params(params: dict) -> pd.Series:
    """Wrapper that calls build_equity_book's components with proposer-supplied
    revision params. We don't mutate engine.portfolio.combined_book; we
    replicate its 4-line composition here with the new params.

    Doing this in the orchestrator (not in public code) keeps the production
    build_equity_book() byte-identical to its commit, and isolates iteration
    risk to this file.
    """
    from engine.portfolio.combined_book import RT_EQ
    from engine.validation.analyst_revision import build_revision_sleeve_buffered

    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet").iloc[:, 0]
    d.index = pd.to_datetime(d.index)
    dp = ((1 + d.clip(-0.2, 0.2)).resample("ME").prod() - 1)
    dp_net = (dp - 5.0 * RT_EQ / 10000.0 / 12).rename("dp")

    rev, rev_turn = build_revision_sleeve_buffered(
        q_in=params["revision_q_in"],
        q_out=params["revision_q_out"],
        weight=params["revision_weight"],
        disp_pctile=params["revision_disp_pctile"],
    )
    rev_net = (rev - rev_turn * RT_EQ / 10000.0 / 12).rename("rev")

    E = pd.concat([dp_net, rev_net], axis=1).dropna()
    vdp = E["dp"].rolling(12).std().shift(1)
    vre = E["rev"].rolling(12).std().shift(1)
    w = (1 / vdp) / (1 / vdp + 1 / vre)
    return (w * E["dp"] + (1 - w) * E["rev"]).dropna().rename("equity_book")


def _val_metrics(returns: pd.Series) -> dict:
    """Compute the metrics the orchestrator decides on (val window only)."""
    r = returns.dropna()
    if len(r) < 12:
        return {"n": int(len(r)), "sharpe": None, "ann_ret": None,
                "ann_vol": None, "maxdd": None}
    sh = float(r.mean() * 12 / (r.std() * np.sqrt(12)))
    cum = (1 + r).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    return {
        "n": int(len(r)),
        "sharpe": round(sh, 4),
        "ann_ret": round(float(r.mean() * 12), 4),
        "ann_vol": round(float(r.std() * np.sqrt(12)), 4),
        "maxdd": round(dd, 4),
    }


# ─── Decision logic ──────────────────────────────────────────────────────────

def _consecutive_rollbacks(skill: dict) -> int:
    n = 0
    for ver in reversed(skill["version_history"]):
        if ver.get("decision") == "rollback":
            n += 1
        else:
            break
    return n


def decide(parent_val: dict | None, new_val: dict, skill: dict) -> tuple[str, str]:
    """Apply the SKILL's rollback_triggers. Returns (decision, reason).

    decision ∈ {"keep", "rollback", "halt"}.
    """
    # Bounds-check that val window had enough data
    min_val = skill["evaluation"].get("min_val_months", 24)
    if (new_val.get("n") or 0) < min_val:
        return "rollback", f"val n={new_val.get('n')} < min {min_val}"

    if parent_val is None or parent_val.get("sharpe") is None:
        # First evaluation — keep by default (this establishes the baseline
        # val metrics for the parent version retroactively)
        return "keep", "first evaluation, establishing val baseline"

    triggers = {t["rule"]: t for t in skill["rollback_triggers"]}
    if (new_val.get("sharpe") is not None and parent_val.get("sharpe") is not None):
        drop = parent_val["sharpe"] - new_val["sharpe"]
        thresh = triggers.get("val_sharpe_drop_pp", {}).get("value", 0.05)
        if drop > thresh:
            return "rollback", (f"val Sharpe dropped {drop:+.3f} > {thresh} "
                                 f"(parent {parent_val['sharpe']:+.3f} -> "
                                 f"new {new_val['sharpe']:+.3f})")

    if (new_val.get("maxdd") is not None and parent_val.get("maxdd") is not None):
        worsening = parent_val["maxdd"] - new_val["maxdd"]   # both negative; parent - new > 0 if new is worse
        thresh = triggers.get("val_maxdd_increase_pp", {}).get("value", 0.02)
        if worsening > thresh:
            return "rollback", (f"val MaxDD worsened by {worsening*100:+.2f}pp > "
                                 f"{thresh*100}pp")

    return "keep", "all rollback triggers cleared"


# ─── Main entry ──────────────────────────────────────────────────────────────

def run_one_iteration(skill_name: str, rng_seed: int | None = None) -> dict:
    """Run a single propose → evaluate → decide → log cycle.

    Returns a result dict (also persisted to disk).
    """
    skill = load_skill(skill_name)

    # Hard guard #1: SKILL gate thresholds must match global pipeline
    from engine.research.pipeline import HLZ_T, DEFLSR_MIN, MAX_BOOK_CORR
    g = skill["evaluation"]["gate"]
    assert g["hlz_t"] == HLZ_T and g["deflsr_min"] == DEFLSR_MIN \
        and g["max_book_corr"] == MAX_BOOK_CORR, \
        f"SKILL gate thresholds differ from global pipeline — refuse to run"

    # Hard guard #2: halt on consecutive rollbacks
    n_consec = _consecutive_rollbacks(skill)
    halt_at = next((t["value"] for t in skill["rollback_triggers"]
                    if t["rule"] == "n_consecutive_rollback_halt"), 3)
    if n_consec >= halt_at:
        return {"decision": "halt",
                "reason": f"{n_consec} consecutive rollbacks; human attention needed",
                "skill": skill_name}

    # 1. Propose
    proposal = propose_change(skill, rng_seed=rng_seed)
    assert_within_bounds(proposal["params"], skill)

    # 2. Find parent for retro val metrics
    parent = next(v for v in reversed(skill["version_history"])
                  if v.get("decision") in ("baseline", "keep"))

    # 3. Build variant on the proposed params
    if skill_name == "equity_book":
        full_returns = _build_equity_book_with_params(proposal["params"])
    else:
        raise NotImplementedError(
            f"SKILL builder not yet wired for {skill_name}. Add to "
            "_build_<skill>_with_params and dispatch here.")

    # 4. Clip to val window (sample-isolation guard)
    val_returns = enforce_sample_isolation(full_returns, skill)

    # 5. Compute val metrics
    new_val = _val_metrics(val_returns)

    # 6. Compute parent val metrics (retro, on parent's params, same val window)
    if parent.get("val_metrics") is None or parent.get("val_metrics", {}).get("sharpe") is None:
        # Retro-evaluate parent so future iterations have a baseline
        parent_returns = _build_equity_book_with_params(parent["params"])
        parent_val = _val_metrics(enforce_sample_isolation(parent_returns, skill))
        parent["val_metrics"] = parent_val
    else:
        parent_val = parent["val_metrics"]

    # 7. Decide
    decision, reason = decide(parent_val, new_val, skill)

    # 8. Build new version entry. Every iteration gets a UNIQUE id (including
    #    rollbacks) so the lineage is fully auditable; parent_version tracks
    #    which KEPT version a proposal was branched from.
    new_version = f"v0.0.{len(skill['version_history'])}"
    entry = {
        "version":      new_version,
        "parent":       proposal["parent_version"],
        "ts":           _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "proposer":     "rule_based_v1",
        "rationale":    f"propose change in {proposal['changed']}",
        "params":       proposal["params"],
        "val_metrics":  new_val,
        "test_metrics": None,
        "decision":     decision,
        "decision_reason": reason,
    }
    skill["version_history"].append(entry)
    save_skill(skill)

    # 9. Persist a versioned snapshot to data/research/skill_versions/
    vdir = VERSIONS_DIR / skill_name
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / f"{new_version}.json").write_text(
        json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "skill":         skill_name,
        "new_version":   new_version,
        "parent_version": proposal["parent_version"],
        "changed":       proposal["changed"],
        "parent_val":    parent_val,
        "val_metrics":   new_val,
        "decision":      decision,
        "decision_reason": reason,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True, help="skill name, e.g. equity_book")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    for i in range(args.iterations):
        seed = args.seed + i if args.seed is not None else None
        res = run_one_iteration(args.skill, rng_seed=seed)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if res.get("decision") == "halt":
            break
