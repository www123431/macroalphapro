"""engine/research/critic_calibration.py — Frontier A (2026-06-01):
per-critic calibration.

Frontier 3 measures council CONSENSUS accuracy vs the pipeline. But
council consensus is just a vote over critics. If theorist and DA are
~80% correlated, we're paying 2x token cost for ~1x information; if
one critic is systematically right and the other is noisy, we should
either drop the noisy one or specialize it. We can't know which is
the case without measuring critics individually.

This module appends ONE row per (iteration, critic) tuple to a
sibling ledger (data/research/critic_calibration.jsonl) every time a
council run lands AND a pipeline outcome exists. Then provides:
  - compute_critic_accuracy(critic_name): naive accuracy + by-family
  - compute_critic_marginal_info(critic_name): council accuracy WITH
    this critic counted vs WITHOUT — the real "what does this critic
    actually add" KPI
  - compute_pairwise_critic_agreement(): are theorist + DA telling us
    the same thing? if so, ensemble is redundant

Doctrine: this module is READ-ONLY for the council — we observe, we
do not retrofit critic prompts based on these stats. Prompt changes
are still human-curated (matches the Frontier 3 / intuition rules
human gate principle: agent doesn't write its own rules).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CRITIC_CALIBRATION_LEDGER = REPO_ROOT / "data" / "research" / "critic_calibration.jsonl"


# ── Per-critic alignment classifier ───────────────────────────────────


# Pipeline decision buckets — mirrors outcome_ledger._classify_alignment.
_PROMOTIVE = {"PROMOTE_TO_GATE", "PROMOTE_AS_REPLACEMENT"}
_REJECTIVE = {"HARD_REJECT", "SOFT_REJECT"}
_BORDERLINE = {"BORDERLINE_REVIEW"}


def classify_critic_alignment(
    critic_verdict: str,
    pipeline_decision: Optional[str],
) -> str:
    """Classify ONE critic's vote vs the empirical pipeline outcome.

    Returns one of:
      agree            — critic was empirically right
      critic_wrong     — critic was empirically wrong
      pipeline_resolved— critic said WARN (uncertain); pipeline answered
      not_runnable     — pipeline didn't run, no calibration possible

    Mapping (per-critic level, not consensus level):
      PASS = critic thinks PROMOTE
      FAIL = critic thinks REJECT
      WARN = critic uncertain
    """
    if not pipeline_decision:
        return "not_runnable"
    cv = (critic_verdict or "").upper().strip()
    pd = pipeline_decision

    if cv == "PASS":
        if pd in _PROMOTIVE:  return "agree"
        if pd in _REJECTIVE:  return "critic_wrong"
        return "agree"        # borderline ≈ optimistic
    if cv == "FAIL":
        if pd in _REJECTIVE:  return "agree"
        if pd in _PROMOTIVE:  return "critic_wrong"
        return "agree"        # borderline ≈ defensive
    if cv == "WARN":
        if pd in _BORDERLINE: return "agree"
        return "pipeline_resolved"
    return "not_runnable"


# ── Append rows ───────────────────────────────────────────────────────


def append_critic_calibration_rows(
    *,
    iteration_id: str,
    council: dict,
    proposal: dict,
    pipeline_report: Optional[dict],
) -> int:
    """Write ONE row per critic in this council run.

    Called by outcome_ledger.append_l4_iteration after the main row
    lands. Best-effort: failures are logged but never raised, so
    calibration ledger trouble can't crash the L4 loop.

    Returns the number of rows written.
    """
    verdicts = council.get("verdicts") or []
    if not verdicts:
        return 0

    pipeline_decision = (pipeline_report or {}).get("final_decision")
    family = (proposal or {}).get("family", "unknown")
    proposed_role = (proposal or {}).get("proposed_role", "unknown")
    ts = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    n_written = 0

    try:
        CRITIC_CALIBRATION_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with CRITIC_CALIBRATION_LEDGER.open("a", encoding="utf-8") as f:
            for v in verdicts:
                row = {
                    "ts":                ts,
                    "iteration_id":      iteration_id,
                    "critic_agent_name": v.get("agent_name", "unknown"),
                    "critic_verdict":    v.get("verdict", "unknown"),
                    "critic_confidence": v.get("confidence"),
                    "round_1_verdict":   v.get("round_1_verdict"),     # None if no reflection
                    "round_1_confidence": v.get("round_1_confidence"),
                    "reflection_action": v.get("reflection_action"),
                    "council_consensus": council.get("consensus"),
                    "pipeline_decision": pipeline_decision,
                    "family":            family,
                    "proposed_role":     proposed_role,
                    "alignment":         classify_critic_alignment(
                        v.get("verdict", ""), pipeline_decision,
                    ),
                }
                f.write(json.dumps(row, default=str) + "\n")
                n_written += 1
    except Exception:
        logger.exception("critic calibration append failed (non-fatal)")
        return n_written
    return n_written


# ── Read + aggregate ──────────────────────────────────────────────────


def _read_all_rows(
    *,
    since_days: Optional[int] = None,
) -> list[dict]:
    if not CRITIC_CALIBRATION_LEDGER.is_file():
        return []
    cutoff = None
    if since_days is not None:
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=since_days)
    out: list[dict] = []
    with CRITIC_CALIBRATION_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if cutoff is not None:
                ts_str = (r.get("ts") or "").rstrip("Z")
                try:
                    ts = _dt.datetime.fromisoformat(ts_str)
                    if ts < cutoff:
                        continue
                except Exception:
                    continue
            out.append(r)
    return out


def compute_critic_accuracy(
    critic_name: str,
    *,
    since_days: int = 90,
    family: Optional[str] = None,
) -> dict:
    """Per-critic accuracy, optionally filtered to one family.

    accuracy = agree / (agree + critic_wrong). pipeline_resolved is
    excluded — when critic said WARN we can't credit/blame them for
    being right or wrong. not_runnable is also excluded.
    """
    rows = _read_all_rows(since_days=since_days)
    rows = [r for r in rows if r.get("critic_agent_name") == critic_name]
    if family:
        rows = [r for r in rows if r.get("family") == family]

    counts = defaultdict(int)
    for r in rows:
        counts[r.get("alignment") or "unknown"] += 1
    decided = counts["agree"] + counts["critic_wrong"]
    accuracy = (counts["agree"] / decided) if decided else None

    # Break out by family for the "is this critic better on some families?" view
    by_family: dict[str, dict] = defaultdict(lambda: {"agree": 0, "wrong": 0})
    for r in rows:
        fam = r.get("family") or "unknown"
        if r.get("alignment") == "agree":
            by_family[fam]["agree"] += 1
        elif r.get("alignment") == "critic_wrong":
            by_family[fam]["wrong"] += 1
    family_acc: dict[str, Optional[float]] = {}
    for fam, c in by_family.items():
        total = c["agree"] + c["wrong"]
        family_acc[fam] = (c["agree"] / total) if total else None

    return {
        "critic_name":   critic_name,
        "since_days":    since_days,
        "family_filter": family,
        "n_total":       len(rows),
        "n_decided":     decided,
        "accuracy":      round(accuracy, 3) if accuracy is not None else None,
        "by_alignment":  dict(counts),
        "by_family":     {
            f: {"n": by_family[f]["agree"] + by_family[f]["wrong"],
                "accuracy": round(a, 3) if a is not None else None}
            for f, a in family_acc.items()
        },
    }


def compute_pairwise_critic_agreement(
    *,
    since_days: int = 90,
) -> dict:
    """Within-iteration: how often do critics give the same verdict?

    High agreement (e.g. >85%) is a red flag — it means the critics
    are likely measuring the same thing and the ensemble is wasted
    LLM cost. We expect ~60-75% agreement: enough overlap to validate
    each other, enough disagreement to add information.
    """
    rows = _read_all_rows(since_days=since_days)
    by_iter: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_iter[r.get("iteration_id") or ""].append(r)

    pair_counts: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"same_verdict": 0, "diff_verdict": 0, "n_iter": 0,
                  "same_alignment": 0, "diff_alignment": 0, "n_decided": 0},
    )
    for iter_id, critics in by_iter.items():
        for i, ci in enumerate(critics):
            for cj in critics[i + 1:]:
                key = tuple(sorted([ci.get("critic_agent_name", "?"),
                                     cj.get("critic_agent_name", "?")]))
                p = pair_counts[key]
                p["n_iter"] += 1
                if ci.get("critic_verdict") == cj.get("critic_verdict"):
                    p["same_verdict"] += 1
                else:
                    p["diff_verdict"] += 1
                ai = ci.get("alignment"); aj = cj.get("alignment")
                if ai in ("agree", "critic_wrong") and aj in ("agree", "critic_wrong"):
                    p["n_decided"] += 1
                    if ai == aj:
                        p["same_alignment"] += 1
                    else:
                        p["diff_alignment"] += 1

    out: list[dict] = []
    for (a, b), p in pair_counts.items():
        verdict_agreement = (p["same_verdict"] / p["n_iter"]) if p["n_iter"] else None
        # Conditional-on-pipeline-decision agreement — strips out the
        # easy "both correctly said PASS on an obvious win" cases by
        # weighting toward decided iterations.
        outcome_alignment_agreement = (
            p["same_alignment"] / p["n_decided"]
        ) if p["n_decided"] else None
        out.append({
            "pair":            list((a, b)),
            "n_iterations":    p["n_iter"],
            "verdict_agreement_pct": (
                round(verdict_agreement * 100, 1)
                if verdict_agreement is not None else None
            ),
            "n_decided":       p["n_decided"],
            "alignment_agreement_pct": (
                round(outcome_alignment_agreement * 100, 1)
                if outcome_alignment_agreement is not None else None
            ),
        })
    return {"since_days": since_days, "pairs": out}


def compute_critic_marginal_info(
    critic_name: str,
    *,
    since_days: int = 90,
) -> dict:
    """How much accuracy does the council LOSE if this critic is dropped?

    Computes counterfactual consensus: for each iteration, recompute
    consensus using OTHER critics only (apply the same aggregator
    rules: ANY FAIL → REJECT, ALL PASS → APPROVE, else NEEDS_REVISION).
    Then compare alignment with pipeline.

    A POSITIVE marginal_info means dropping this critic would hurt
    council accuracy — it's earning its keep. NEAR-ZERO marginal_info
    means it's redundant.

    NEGATIVE marginal_info (rare but possible) means this critic is
    actively hurting council accuracy — typically a sign of a
    miscalibrated persona or a regime where its mandate doesn't apply.
    """
    from engine.research.outcome_ledger import _classify_alignment

    rows = _read_all_rows(since_days=since_days)
    by_iter: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_iter[r.get("iteration_id") or ""].append(r)

    full_correct = full_decided = 0
    without_correct = without_decided = 0

    for iter_id, critics in by_iter.items():
        pipeline = critics[0].get("pipeline_decision")
        if not pipeline:
            continue
        full_verdicts = [c.get("critic_verdict") for c in critics]
        without_verdicts = [
            c.get("critic_verdict") for c in critics
            if c.get("critic_agent_name") != critic_name
        ]
        if not without_verdicts:
            # this critic was the only one — counterfactual undefined
            continue

        full_consensus     = _aggregate(full_verdicts)
        without_consensus  = _aggregate(without_verdicts)

        full_align     = _classify_alignment(full_consensus, pipeline)
        without_align  = _classify_alignment(without_consensus, pipeline)

        if full_align in ("agree", "council_wrong"):
            full_decided += 1
            if full_align == "agree":
                full_correct += 1
        if without_align in ("agree", "council_wrong"):
            without_decided += 1
            if without_align == "agree":
                without_correct += 1

    full_acc    = (full_correct    / full_decided)    if full_decided    else None
    without_acc = (without_correct / without_decided) if without_decided else None
    marginal = (
        round(full_acc - without_acc, 3)
        if (full_acc is not None and without_acc is not None) else None
    )

    return {
        "critic_name":         critic_name,
        "since_days":          since_days,
        "full_council_accuracy":         (
            round(full_acc, 3) if full_acc is not None else None
        ),
        "without_critic_accuracy":       (
            round(without_acc, 3) if without_acc is not None else None
        ),
        "marginal_information_gain":     marginal,
        "n_full_decided":                full_decided,
        "n_without_critic_decided":      without_decided,
        "interpretation": _interpret_marginal(marginal, full_decided),
    }


def _aggregate(verdicts: list[str]) -> str:
    """Replica of agent_council.aggregate_verdicts rules at verdict-string level."""
    vs = [v for v in verdicts if v]
    if not vs:
        return "REJECT"  # no signal → conservative
    if any(v == "FAIL" for v in vs):
        return "REJECT"
    if any(v == "WARN" for v in vs):
        return "NEEDS_REVISION"
    return "APPROVE"


def _interpret_marginal(marginal: Optional[float], n: int) -> str:
    """Human-readable summary the UI / report can display verbatim."""
    if marginal is None:
        return "insufficient data"
    if n < 20:
        return f"low confidence (n={n} decided iterations; need 20+)"
    if marginal >= 0.05:
        return "this critic ADDS material information — keep"
    if marginal >= 0.02:
        return "this critic adds modest information"
    if marginal >= -0.02:
        return "this critic is REDUNDANT with peers — consider dropping or specializing"
    return "this critic HURTS accuracy — review persona prompt"


# ── Top-level report ──────────────────────────────────────────────────


def critic_calibration_report(*, since_days: int = 90) -> dict:
    """One-shot report: per-critic accuracy + pairwise agreement +
    marginal info gain. Backs an UI panel / CLI summary."""
    rows = _read_all_rows(since_days=since_days)
    critics = sorted({r.get("critic_agent_name") for r in rows
                       if r.get("critic_agent_name")})
    return {
        "since_days":         since_days,
        "n_total_rows":       len(rows),
        "n_distinct_critics": len(critics),
        "per_critic": {
            c: {
                "accuracy":     compute_critic_accuracy(c, since_days=since_days),
                "marginal_info": compute_critic_marginal_info(c, since_days=since_days),
            }
            for c in critics
        },
        "pairwise_agreement": compute_pairwise_critic_agreement(since_days=since_days),
    }


# ── CLI ───────────────────────────────────────────────────────────────


def _cli() -> None:
    """python -m engine.research.critic_calibration <report|critic NAME>"""
    import sys
    args = sys.argv[1:]
    cmd = args[0] if args else "report"

    if cmd == "report":
        out = critic_calibration_report(
            since_days=int(args[1]) if len(args) > 1 else 90,
        )
        print(json.dumps(out, indent=2, default=str))
        return
    if cmd == "critic" and len(args) >= 2:
        print(json.dumps({
            "accuracy":     compute_critic_accuracy(args[1]),
            "marginal_info": compute_critic_marginal_info(args[1]),
        }, indent=2, default=str))
        return
    print("usage: report [since_days] | critic NAME",
          file=__import__("sys").stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    _cli()
