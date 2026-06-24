"""engine/research/calibration_feedback.py — Frontier 3 (2026-06-01):
calibration feedback loop.

When the council is WRONG (verdict_alignment == "council_wrong" — i.e.
council APPROVED but pipeline REJECTED, or vice versa) AND the same
pattern recurs across iterations, synthesize a candidate intuition_rule
and queue it for human review.

DESIGN: human-gated, NOT auto-written. intuition_rules.yaml is the
council's load-bearing knowledge base; a bad auto-generated rule would
corrupt every future verdict. Proposed rules go to
data/research/proposed_intuition_rules.jsonl with status="pending",
and a human explicitly promotes them via accept_proposed_rule().

Anti-pattern blocked: "L4 generates rules → council uses them → L4
generates rules" — that's the kind of feedback loop where the council
agrees with itself forever. Human-in-the-loop on rule promotion breaks
this circuit.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROPOSED_RULES_LEDGER = REPO_ROOT / "data" / "research" / "proposed_intuition_rules.jsonl"

ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_MAX_TOKENS = 1500
ANTHROPIC_TEMPERATURE = 0.2

# Minimum number of council_wrong events in a cluster before we'll
# synthesize a rule. Single one-offs are noise; recurring patterns
# are signal.
DEFAULT_MIN_CLUSTER_SIZE = 2

# Lookback window for the scan — how far back to consider iterations.
DEFAULT_SINCE_DAYS = 30


# ── Cluster definitions ───────────────────────────────────────────────


@dataclass
class WrongCluster:
    """One group of council_wrong iterations sharing a pattern key."""
    pattern_key:        str
    family:             str
    council_consensus:  str
    pipeline_decision:  str
    iteration_ids:      list[str]
    n:                  int
    sample_proposals:   list[dict]  # truncated to 3 for the prompt
    sample_council_rationales:  list[str]
    sample_pipeline_rationales: list[str]


# ── Step 1: find council_wrong iterations ─────────────────────────────


def find_council_wrong_iterations(
    *,
    since_days: int = DEFAULT_SINCE_DAYS,
    limit: int = 500,
) -> list[dict]:
    """Read l4_iterations.jsonl and return iterations where the council
    materially disagreed with the empirical pipeline.

    "council_wrong" = council APPROVED but pipeline rejected, or council
    REJECTED but pipeline promoted. NEEDS_REVISION cases are treated as
    "pipeline_resolved" by the outcome_ledger classifier and are NOT
    council errors — those iterations are signal that the council
    correctly said "uncertain".
    """
    from engine.research.outcome_ledger import read_l4_iterations
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=since_days)
    rows = read_l4_iterations(limit=limit, alignment="council_wrong")
    out: list[dict] = []
    for r in rows:
        ts_str = (r.get("ts") or "").rstrip("Z")
        try:
            ts = _dt.datetime.fromisoformat(ts_str)
        except Exception:
            ts = None
        if ts is None or ts >= cutoff:
            out.append(r)
    return out


# ── Step 2: cluster by pattern ────────────────────────────────────────


def _pattern_key(row: dict) -> str:
    """Stable group-by key: family × council direction × pipeline direction.

    Same family + same disagreement direction → same root cause class.
    Different families with the same direction are NOT clustered because
    the corrective rule needs to mention the family.
    """
    family = (row.get("proposal") or {}).get("family", "unknown")
    cc = (row.get("council") or {}).get("consensus", "unknown")
    pd = (row.get("pipeline") or {}).get("final_decision", "unknown")
    return f"{family}::{cc}::{pd}"


def cluster_council_wrong(rows: list[dict]) -> list[WrongCluster]:
    """Group iterations by pattern_key and return clusters sorted by size."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[_pattern_key(r)].append(r)

    clusters: list[WrongCluster] = []
    for key, members in by_key.items():
        family = (members[0].get("proposal") or {}).get("family", "unknown")
        cc = (members[0].get("council") or {}).get("consensus", "unknown")
        pd = (members[0].get("pipeline") or {}).get("final_decision", "unknown")
        # Sample up to 3 representative entries for the LLM prompt
        sample = members[:3]
        clusters.append(WrongCluster(
            pattern_key=key,
            family=family,
            council_consensus=cc,
            pipeline_decision=pd,
            iteration_ids=[m.get("iteration_id", "?") for m in members],
            n=len(members),
            sample_proposals=[m.get("proposal") or {} for m in sample],
            sample_council_rationales=[
                ((m.get("council") or {}).get("rationale") or "")[:400]
                for m in sample
            ],
            sample_pipeline_rationales=[
                ((m.get("pipeline") or {}).get("rationale") or "")[:400]
                for m in sample
            ],
        ))
    clusters.sort(key=lambda c: -c.n)
    return clusters


# ── Step 3: synthesize a candidate rule via LLM ───────────────────────


def _load_anthropic_key() -> Optional[str]:
    """Reuse the council's key loader for consistency."""
    from engine.research.agent_council import _load_anthropic_key as _load
    return _load()


_SYNTH_SYSTEM_PROMPT = """\
You are RULE_SYNTHESIZER — a senior quant who, when shown a pattern of
council miscalibration, writes a single concise intuition_rule that
would have prevented that pattern.

You MUST produce STRICT JSON matching this schema (no markdown fences):

{
  "id":          "snake_case_unique_identifier",
  "category":   "statistical|structural|data_quality|regime|decay|cross_market|role_interpretation|process|evidence",
  "severity":   "FATAL_BLOCK|HARD_WARN|SOFT_INFO",
  "when":       "Free-text conditions where this rule fires. Be specific enough that an LLM agent can pattern-match the next proposal.",
  "then":       "Free-text consequence + recommended action. Tell the agent what to do, not just what to fear.",
  "evidence_source": "calibration_cluster:{pattern_key} (auto-synthesized {date})",
  "rationale":  "1-2 sentences for the human reviewer: what pattern triggered this rule, why it's worth adopting."
}

GUIDELINES:
  - severity HARD_WARN is the right default — the rule SURFACES the
    concern so the agent acknowledges it, but doesn't auto-reject.
    Reserve FATAL_BLOCK for invariants (no look-ahead, no fabricated
    citations) where ANY violation is invalid.
  - The "when" clause should be matchable: avoid vague phrasings like
    "if the proposal is unusual"; use concrete signals from the
    pattern (family name, signal type, role, etc.).
  - Do NOT propose rules that contradict existing rules — if the
    pattern matches an existing rule's "when", the rule is probably
    not being USED, which is a separate calibration issue.
  - One rule per cluster. Don't bundle multiple concerns.
"""


def synthesize_proposed_rule(
    cluster: WrongCluster,
    *,
    api_key: Optional[str] = None,
    model: str = ANTHROPIC_MODEL,
) -> dict:
    """Call Anthropic to draft a candidate IntuitionRule for this cluster.

    Raises RuntimeError if no API key. Returns the proposed-rule dict
    PLUS metadata (cluster size, pattern key, ts). Does NOT persist —
    caller decides whether to queue.
    """
    key = api_key or _load_anthropic_key()
    if not key:
        raise RuntimeError(
            "no ANTHROPIC_API_KEY found in env or .streamlit/secrets.toml"
        )

    import anthropic
    user_msg = _format_cluster_for_prompt(cluster)

    client = anthropic.Anthropic(api_key=key)
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        temperature=ANTHROPIC_TEMPERATURE,
        system=_SYNTH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    elapsed_s = time.perf_counter() - t0

    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    parsed = _parse_json(text)
    if not parsed or "id" not in parsed:
        raise RuntimeError(
            f"rule synthesizer returned unparseable JSON. "
            f"Raw (first 400 chars): {text[:400]}"
        )

    # Enrich with provenance the LLM can't compute reliably
    parsed["_meta"] = {
        "cluster_pattern_key":  cluster.pattern_key,
        "cluster_size":         cluster.n,
        "cluster_iteration_ids": cluster.iteration_ids,
        "synthesized_at":       _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "synthesized_by":       "calibration_feedback.synthesize_proposed_rule",
        "model":                model,
        "elapsed_s":            round(elapsed_s, 2),
    }
    return parsed


def _format_cluster_for_prompt(cluster: WrongCluster) -> str:
    samples = []
    for i, prop in enumerate(cluster.sample_proposals):
        samples.append(
            f"  Iteration {i+1}: title={prop.get('title')!r} "
            f"role={prop.get('proposed_role')!r}\n"
            f"    Council rationale: "
            f"{cluster.sample_council_rationales[i][:200]}\n"
            f"    Pipeline rationale: "
            f"{cluster.sample_pipeline_rationales[i][:200]}"
        )
    samples_str = "\n".join(samples)
    return (
        f"Synthesize ONE intuition_rule that would have caught this "
        f"recurring miscalibration pattern.\n\n"
        f"PATTERN KEY: {cluster.pattern_key}\n"
        f"CLUSTER SIZE: {cluster.n} iterations\n"
        f"FAMILY: {cluster.family}\n"
        f"DIRECTION: council said {cluster.council_consensus}, "
        f"pipeline said {cluster.pipeline_decision}\n\n"
        f"REPRESENTATIVE ITERATIONS:\n{samples_str}\n\n"
        f"Return strict JSON per the schema in the system prompt."
    )


def _parse_json(raw: str) -> dict:
    """Lenient JSON extraction matching agent_council._parse_verdict_json."""
    if not raw:
        return {}
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(s[start: end + 1])
    except Exception:
        return {}


# ── Step 4: proposed-rule queue (the human review surface) ────────────


def append_proposed_rule(rule_dict: dict) -> str:
    """Persist one proposed rule to the review queue.

    Returns a proposal_id (separate from the rule's "id" because the
    LLM-generated id might collide with existing rules; human reviewer
    arbitrates)."""
    proposal_id = f"prop-{uuid.uuid4().hex[:12]}"
    try:
        PROPOSED_RULES_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "proposal_id": proposal_id,
            "ts":          _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "status":      "pending",
            "rule":        rule_dict,
            "reviewed_at": None,
            "reviewed_by": None,
            "review_note": None,
        }
        with PROPOSED_RULES_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        logger.exception("proposed rules ledger append failed (non-fatal)")
    return proposal_id


def read_proposed_rules(
    *,
    status: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Read proposed rules newest-first. Filterable by status."""
    if not PROPOSED_RULES_LEDGER.is_file():
        return []
    out: list[dict] = []
    with PROPOSED_RULES_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if status and r.get("status") != status:
                continue
            out.append(r)
    out.reverse()
    return out[: max(1, int(limit))]


def review_proposed_rule(
    proposal_id: str,
    *,
    status: str,
    reviewer: str = "unknown",
    note: Optional[str] = None,
) -> dict:
    """Update one proposed rule's status to "accepted" / "rejected".

    Rewrites the ledger file (compact-and-swap) since JSONL doesn't
    natively support in-place edits. Atomic via tempfile + os.replace.
    """
    if status not in ("accepted", "rejected"):
        raise ValueError(
            f"status must be 'accepted' or 'rejected'; got {status!r}"
        )
    if not PROPOSED_RULES_LEDGER.is_file():
        raise FileNotFoundError(
            f"proposed rules ledger does not exist: {PROPOSED_RULES_LEDGER}"
        )

    updated: Optional[dict] = None
    rows: list[dict] = []
    with PROPOSED_RULES_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("proposal_id") == proposal_id:
                r["status"] = status
                r["reviewed_at"] = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
                r["reviewed_by"] = reviewer
                r["review_note"] = note
                updated = r
            rows.append(r)

    if updated is None:
        raise KeyError(
            f"no proposed rule found with proposal_id={proposal_id!r}"
        )

    tmp = PROPOSED_RULES_LEDGER.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, PROPOSED_RULES_LEDGER)

    return updated


# ── Top-level scan ────────────────────────────────────────────────────


def run_calibration_scan(
    *,
    since_days: int = DEFAULT_SINCE_DAYS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    max_synthesize: int = 5,
    api_key: Optional[str] = None,
) -> dict:
    """End-to-end: find council_wrong → cluster → synthesize → queue.

    Returns a summary: n_wrong, n_clusters, clusters_synthesized,
    proposed_ids. Tolerates LLM failure per cluster (one bad synthesis
    doesn't stop the rest)."""
    rows = find_council_wrong_iterations(since_days=since_days)
    clusters = cluster_council_wrong(rows)
    actionable = [c for c in clusters if c.n >= min_cluster_size]

    proposed_ids: list[str] = []
    errors: list[dict] = []

    for c in actionable[:max_synthesize]:
        try:
            rule = synthesize_proposed_rule(c, api_key=api_key)
            pid = append_proposed_rule(rule)
            proposed_ids.append(pid)
        except Exception as exc:
            logger.exception("synthesize failed for cluster %s", c.pattern_key)
            errors.append({"pattern_key": c.pattern_key,
                            "error": str(exc)[:200]})

    return {
        "since_days":              since_days,
        "n_wrong_iterations":      len(rows),
        "n_clusters":              len(clusters),
        "n_actionable_clusters":   len(actionable),
        "n_synthesized":           len(proposed_ids),
        "proposed_ids":            proposed_ids,
        "min_cluster_size":        min_cluster_size,
        "errors":                  errors,
        "actionable_pattern_keys": [c.pattern_key for c in actionable],
    }


# ── CLI ───────────────────────────────────────────────────────────────


def _cli() -> None:
    """python -m engine.research.calibration_feedback <scan|list|accept|reject>"""
    import sys
    args = sys.argv[1:]
    cmd = args[0] if args else "list"

    if cmd == "scan":
        out = run_calibration_scan()
        print(json.dumps(out, indent=2, default=str))
        return

    if cmd == "list":
        status = args[1] if len(args) > 1 else None
        rules = read_proposed_rules(status=status, limit=50)
        print(json.dumps({"n": len(rules), "proposed": rules},
                          indent=2, default=str))
        return

    if cmd == "accept" and len(args) >= 2:
        out = review_proposed_rule(args[1], status="accepted",
                                     reviewer="cli",
                                     note=" ".join(args[2:]) if len(args) > 2 else None)
        print(json.dumps(out, indent=2, default=str))
        return

    if cmd == "reject" and len(args) >= 2:
        out = review_proposed_rule(args[1], status="rejected",
                                     reviewer="cli",
                                     note=" ".join(args[2:]) if len(args) > 2 else None)
        print(json.dumps(out, indent=2, default=str))
        return

    print(f"usage: scan | list [status] | accept <proposal_id> [note] | "
          "reject <proposal_id> [note]", file=__import__("sys").stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    _cli()
