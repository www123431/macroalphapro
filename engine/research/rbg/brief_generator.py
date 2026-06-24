"""engine/research/rbg/brief_generator.py — Markdown brief assembly.

Pipeline:
  1. Take a PFH ScoredProposal (the structured evidence + score)
  2. Optionally take the materialized output (real Sharpe, vol, etc.)
  3. Assemble the SKELETON deterministically (headers + evidence cells
     + paste-able commands + predicted critic concerns)
  4. Optionally call Anthropic to fill PROSE sections (LLM never
     touches numerical fields)
  5. Validate that LLM-written prose cites evidence_ids that exist
     in the input ScoredProposal — flag unbacked claims as violations
  6. Return BriefArtifact (markdown + warnings + metadata)
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
BRIEFS_DIR = REPO_ROOT / "data" / "research" / "briefs"

ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_MAX_TOKENS = 1500
ANTHROPIC_TEMPERATURE = 0.2


# ── Artifact ──────────────────────────────────────────────────────────


@dataclass
class BriefArtifact:
    """The output of generate_brief.

    markdown:        the full markdown text ready to write to disk
    used_llm:        True if Anthropic was called for prose; False on
                      structured-only fallback
    validation_warnings:
                     list of post-hoc validations the brief failed
                      (e.g. LLM cited an evidence_id that didn't exist
                      in the input ScoredProposal)
    evidence_ids:    deterministic set of IDs the brief references —
                      used by the validator to enforce no-fabricated-cites
    metadata:        {brief_id, generated_at, spec_id, llm_model, ...}
    """
    markdown:            str
    used_llm:            bool
    validation_warnings: list[str]
    evidence_ids:        set[str]
    metadata:            dict


# ── Evidence extraction ──────────────────────────────────────────────


def _collect_evidence_ids(scored: dict) -> set[str]:
    """Extract all evidence_ids referenced in a ScoredProposal.

    The brief MUST only cite IDs from this set. Anything else is a
    fabricated claim and triggers a validation warning."""
    ids: set[str] = set()
    proposal = scored.get("proposal", {})
    # Family + universe + signal + weighting names are evidence
    for k in ("family_normalized", "universe", "signal_recipe", "weighting"):
        v = proposal.get(k)
        if v:
            ids.add(str(v))
    # derived_from = library mechanism names
    for d in proposal.get("derived_from", []):
        ids.add(str(d))
    # cousin_warnings often reference graveyard family names
    for w in proposal.get("cousin_warnings", []):
        # Extract any quoted names from the warning string
        for m in re.findall(r"\b([a-z_]+(?:_[a-z]+)*)\b", str(w).lower()):
            if len(m) > 3:
                ids.add(m)
    # Score breakdown family
    bd = scored.get("score_breakdown", {})
    if isinstance(bd, dict) and "family" in bd:
        ids.add(str(bd["family"]))
    return ids


# ── Deterministic skeleton sections ──────────────────────────────────


def _section_metrics(
    scored: dict, materialized: Optional[dict],
) -> str:
    """Headline KPI block. All numbers come from input dicts verbatim."""
    post     = scored.get("posterior", {})
    breakdown = scored.get("score_breakdown", {})
    lines = ["## Headline metrics", ""]
    lines.append(f"- **Bayesian posterior P(success):** "
                 f"{post.get('posterior_mean', '?')}")
    ci_lo = post.get("credible_05")
    ci_hi = post.get("credible_95")
    if ci_lo is not None and ci_hi is not None:
        lines.append(f"  - 90% credible interval: [{ci_lo}, {ci_hi}]")
    lines.append(f"- **Cousin penalty:** "
                 f"{scored.get('cousin_penalty', 1.0):.4f}")
    lines.append(f"- **Final ranked score:** "
                 f"{scored.get('final_score', '?'):.4f}")
    if materialized:
        v = materialized.get("validation", {})
        if v.get("observed_ann_sharpe") is not None:
            lines.append(f"- **Materialized annualized Sharpe:** "
                         f"{v['observed_ann_sharpe']:.4f}")
        if v.get("observed_ann_vol") is not None:
            lines.append(f"- **Materialized annualized vol:** "
                         f"{v['observed_ann_vol']:.4f}")
        lines.append(f"- **n_months observed:** {v.get('observed_n_rows', '?')}")
    lines.append("")
    return "\n".join(lines)


def _section_evidence(scored: dict) -> str:
    """Evidence chain dump. Every cited ID lives in this section so
    the validator can verify LLM-written prose against it."""
    p = scored.get("proposal", {})
    breakdown = scored.get("score_breakdown", {})
    lines = ["## Evidence chain", ""]
    lines.append(f"- **Family:** `{p.get('family_normalized', '?')}`")
    lines.append(f"- **Proposal kind:** `{p.get('proposal_kind', '?')}`")
    if p.get("universe"):
        lines.append(f"- **Universe:** `{p['universe']}`")
    if p.get("signal_recipe"):
        lines.append(f"- **Signal recipe:** `{p['signal_recipe']}`")
    if p.get("weighting"):
        lines.append(f"- **Weighting:** `{p['weighting']}`")
    df = p.get("derived_from") or []
    if df:
        lines.append(f"- **Derived from GREEN sleeves:** "
                     + ", ".join(f"`{x}`" for x in df))
    lines.append("")
    lines.append("### Bayesian cell")
    lines.append(f"- n_green: **{breakdown.get('cell_n_green', '?')}** · "
                 f"n_yellow: {breakdown.get('cell_n_yellow', '?')} · "
                 f"n_red: **{breakdown.get('cell_n_red', '?')}**")
    lines.append(f"- Base rate used: {breakdown.get('base_rate', '?')}")
    lines.append(f"- Prior strength: {breakdown.get('prior_strength', '?')}")
    cw = p.get("cousin_warnings") or []
    if cw:
        lines.append("")
        lines.append("### Cousin warnings")
        for w in cw:
            lines.append(f"- {w}")
    nn = p.get("needs_new_axes") or []
    if nn:
        lines.append("")
        lines.append("### Axis components needed")
        for a in nn:
            lines.append(f"- `{a}`")
    lines.append("")
    return "\n".join(lines)


def _section_predicted_concerns(scored: dict) -> str:
    """Deterministic projection of which council critics will flag
    what, based on the evidence chain. Drives the user's pre-flight
    'know what's coming' expectation."""
    p = scored.get("proposal", {})
    bd = scored.get("score_breakdown", {})
    lines = ["## Predicted council concerns", ""]
    concerns: list[str] = []

    # Cousin warnings → DA will flag graveyard match
    cw_count = len(p.get("cousin_warnings") or [])
    if cw_count > 0:
        concerns.append(
            f"**Devil's advocate** will likely flag the {cw_count} cousin "
            "warning(s) and ask why this variation should differ in mechanism "
            "from the graveyarded cousin(s)."
        )

    # Low posterior + wide CI → theorist will ask for mechanism story
    ci_lo = scored.get("posterior", {}).get("credible_05")
    ci_hi = scored.get("posterior", {}).get("credible_95")
    if ci_lo is not None and ci_hi is not None and (ci_hi - ci_lo) > 0.5:
        concerns.append(
            f"**Behavioral theorist** will likely ask for the economic "
            f"mechanism — the credible interval [{ci_lo}, {ci_hi}] is wide, "
            "meaning the Bayesian prior is mostly driven by base rate and a "
            "concrete mechanism story is needed before accepting."
        )

    # Many RED in cell + new variation → DA will ask about deflated Sharpe
    if bd.get("cell_n_red", 0) >= 3:
        concerns.append(
            f"**Devil's advocate** will likely ask for deflated Sharpe + "
            f"family_n_trials_lookup — {bd.get('cell_n_red')} prior failures in "
            "this family means multi-testing budget is already partially "
            "consumed."
        )

    # No GREEN in cell → theorist will scrutinize publication-bias risk
    if bd.get("cell_n_green", 0) == 0:
        concerns.append(
            "**Behavioral theorist** will likely probe publication-bias risk — "
            "no prior GREEN in this family means we're entering on literature "
            "alone, which is subject to Hou-Xue-Zhang 2020 replication concern."
        )

    # Needs new axes → reflection round will likely diverge
    nn = p.get("needs_new_axes") or []
    if nn:
        concerns.append(
            f"**Reflection round** may diverge — the candidate needs "
            f"{len(nn)} new axis component(s) defined, and critics will "
            "likely disagree on whether the proposed axis spec is adequate."
        )

    if not concerns:
        lines.append("*No major concerns flagged by skeleton heuristics. "
                     "Council will still run a full critique.*")
    else:
        for c in concerns:
            lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


def _section_run_commands(scored: dict) -> str:
    """Paste-able commands. Every line is literal — no <FILL_IN>."""
    p = scored.get("proposal", {})
    cid = p.get("candidate_id", "<spec_id>")
    family = p.get("family_normalized", "?")
    universe = p.get("universe", "?")
    signal = p.get("signal_recipe", "?")
    weighting = p.get("weighting", "?")
    lines = [
        "## Run this now",
        "",
        "Re-materialize this factor (uses the cached compose-spec PFH "
        "already wrote):",
        "",
        "```python",
        "from engine.feature_store import materialize_spec",
        f"result = materialize_spec({cid!r}, force=True)",
        "print('Sharpe:', result['validation']['observed_ann_sharpe'])",
        "print('vol:',    result['validation']['observed_ann_vol'])",
        "```",
        "",
        "Send the proposal to the council for verdict:",
        "",
        "```python",
        "import asyncio",
        "from engine.research.agent_council import run_full_council",
        f"seed = '''Candidate factor (PFH-suggested): {cid}.",
        f"Family: {family}.",
        f"Universe x Signal x Weighting = {universe} x {signal} x {weighting}.",
        "'''",
        "proposal, council = asyncio.run(run_full_council(seed, "
        "enable_reflection=True))",
        "print('Consensus:', council.consensus)",
        "print('Rationale:', council.rationale)",
        "```",
        "",
    ]
    return "\n".join(lines)


def _section_stop_criteria(scored: dict, materialized: Optional[dict]) -> str:
    """Pre-registered stop criteria so the user doesn't drift mid-test."""
    lines = ["## Stop criteria (pre-registered)", ""]
    lines.append("- If pipeline cosine vs any deployed sleeve > 0.6: "
                 "STOP (not orthogonal, would just inflate book gross "
                 "without adding alpha)")
    lines.append("- If council REJECT unanimously: STOP (record in "
                 "graveyard with reason)")
    lines.append("- If deflated Sharpe < 0.30 after cost adjustment: STOP")
    if materialized:
        sh = materialized.get("validation", {}).get("observed_ann_sharpe")
        if sh is not None and sh < 0:
            lines.append(f"- **Already failing**: materialized Sharpe "
                         f"{sh:.3f} is negative — log RED reason and stop")
    lines.append("")
    return "\n".join(lines)


# ── LLM prose layer (optional) ───────────────────────────────────────


_PROSE_SYSTEM_PROMPT = """\
You are a SENIOR QUANT RESEARCH ASSISTANT writing a brief 2-paragraph
introduction to a candidate factor proposal. You will be given a
structured evidence chain. Write ONLY the introductory prose; do NOT
recompute or restate the numerical metrics — they live in a separate
section the reader will see.

CONSTRAINTS:
  - Keep total length to ~150 words (2 short paragraphs).
  - Every specific claim (a paper, a family, a mechanism) MUST be
    backed by an evidence_id from the provided list. Do not invent
    citations. If you need to make a claim without a backing ID,
    use generic language ("the literature suggests...") rather than
    fabricating a citation.
  - First paragraph: economic hypothesis being tested.
  - Second paragraph: relationship to our existing book and what
    success/failure would tell us.
  - NO marketing language ("promising", "exciting"). Senior quant tone.
"""


def _load_anthropic_key() -> Optional[str]:
    """Reuse council's key loader."""
    try:
        from engine.research.agent_council import _load_anthropic_key as _f
        return _f()
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def _llm_prose_intro(
    scored: dict,
    evidence_ids: set[str],
    api_key: Optional[str] = None,
) -> tuple[str, bool]:
    """Generate the introductory prose. Returns (prose_text, used_llm)."""
    key = api_key or _load_anthropic_key()
    if not key:
        return (
            "*(Structured-only mode — no ANTHROPIC_API_KEY available; "
            "LLM prose layer skipped. The evidence sections below "
            "are complete and the brief is usable as-is.)*",
            False,
        )

    try:
        import anthropic
    except ImportError:
        return ("*(anthropic SDK not installed; LLM prose layer skipped.)*",
                False)

    user_msg = (
        f"Candidate factor evidence chain:\n\n"
        f"```json\n{json.dumps(scored, indent=2, default=str)}\n```\n\n"
        f"Available evidence_ids you may cite: {sorted(evidence_ids)}\n\n"
        f"Write the 2-paragraph introduction per the system instructions."
    )
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            temperature=ANTHROPIC_TEMPERATURE,
            system=_PROSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in resp.content
                        if getattr(b, "type", "") == "text").strip()
        return text or "*(LLM returned empty prose.)*", True
    except Exception as exc:
        logger.exception("LLM prose generation failed")
        return f"*(LLM call failed: {str(exc)[:200]}.)*", False


# ── Validation: no fabricated evidence IDs ───────────────────────────


def _validate_prose_against_evidence(
    prose: str,
    evidence_ids: set[str],
) -> list[str]:
    """Heuristic: extract snake_case-looking tokens from prose; any
    that look like factor/family/sleeve IDs but aren't in evidence
    set are flagged as potential fabrications.

    NOT bulletproof (LLM can paraphrase a fake citation in english),
    but catches the obvious failure mode of "wrote a snake_case name
    that doesn't exist".
    """
    warnings: list[str] = []
    # Tokens that look like quant IDs: ≥2 segments, alnum + underscore
    candidates = re.findall(r"`([a-z][a-z0-9_]{6,})`", prose.lower())
    candidates += re.findall(r"\b([a-z]+_[a-z][a-z0-9_]{4,})\b", prose.lower())
    suspicious = set(candidates) - evidence_ids
    # Filter out generic English tokens that happen to have underscores
    suspicious = {s for s in suspicious if "_" in s}
    if suspicious:
        warnings.append(
            f"prose references {len(suspicious)} ID-like token(s) not in "
            f"evidence chain: {sorted(suspicious)[:5]}"
        )
    return warnings


# ── Public API ───────────────────────────────────────────────────────


def generate_brief(
    scored: dict,
    *,
    materialized: Optional[dict] = None,
    use_llm: bool = True,
    api_key: Optional[str] = None,
) -> BriefArtifact:
    """Assemble a research brief markdown from a PFH ScoredProposal.

    Args:
      scored:       a ScoredProposal.to_dict() output (or compatible dict)
      materialized: optional materialize_spec() result dict for the
                     same spec — adds real Sharpe/vol/etc. to the brief
      use_llm:      if True, calls Anthropic for the intro prose.
                     False forces structured-only mode.
      api_key:      override; defaults to env / streamlit secrets
    """
    proposal = scored.get("proposal", {})
    spec_id  = proposal.get("candidate_id", "unknown")
    ts       = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    brief_id = f"brief-{uuid.uuid4().hex[:12]}"

    evidence_ids = _collect_evidence_ids(scored)

    # Skeleton (deterministic)
    header = "\n".join([
        f"# Research Brief: `{spec_id}`",
        "",
        f"*Generated {ts} · brief_id `{brief_id}` · "
        f"PFH-originated, pending council review*",
        "",
    ])
    metrics_section   = _section_metrics(scored, materialized)
    evidence_section  = _section_evidence(scored)
    concerns_section  = _section_predicted_concerns(scored)
    commands_section  = _section_run_commands(scored)
    stop_section      = _section_stop_criteria(scored, materialized)

    # Optional LLM prose
    used_llm = False
    intro_section = ""
    if use_llm:
        prose, used_llm = _llm_prose_intro(scored, evidence_ids, api_key)
        intro_section = "## Hypothesis\n\n" + prose + "\n\n"
    else:
        intro_section = ""

    # Validate prose if LLM was used
    warnings: list[str] = []
    if used_llm:
        warnings = _validate_prose_against_evidence(intro_section, evidence_ids)

    markdown = (
        header
        + intro_section
        + metrics_section
        + evidence_section
        + concerns_section
        + commands_section
        + stop_section
        + "\n---\n*This brief is auto-generated by engine.research.rbg. "
          "Numerical fields are deterministic; prose (if present) is "
          "LLM-generated. All claims tied to evidence_ids listed above.*\n"
    )

    metadata = {
        "brief_id":         brief_id,
        "spec_id":          spec_id,
        "generated_at":     ts,
        "used_llm":         used_llm,
        "llm_model":        ANTHROPIC_MODEL if used_llm else None,
        "has_materialized": materialized is not None,
        "n_evidence_ids":   len(evidence_ids),
        "n_warnings":       len(warnings),
    }

    return BriefArtifact(
        markdown=markdown,
        used_llm=used_llm,
        validation_warnings=warnings,
        evidence_ids=evidence_ids,
        metadata=metadata,
    )


def write_brief_to_disk(
    artifact: BriefArtifact,
    *,
    out_dir: Optional[Path] = None,
) -> Path:
    """Persist a brief to data/research/briefs/<spec_id>_<brief_id>.md."""
    target_dir = out_dir if out_dir is not None else BRIEFS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    spec_id  = artifact.metadata.get("spec_id", "unknown")
    brief_id = artifact.metadata.get("brief_id", "?")
    out_path = target_dir / f"{spec_id}_{brief_id}.md"
    out_path.write_text(artifact.markdown, encoding="utf-8")
    # Sidecar metadata
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({
            **artifact.metadata,
            "warnings":    artifact.validation_warnings,
            "evidence_ids": sorted(artifact.evidence_ids),
        }, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────


def _cli() -> None:
    """python -m engine.research.rbg.brief_generator <generate|preview>

    generate <spec_id> [--no-llm]
        Generate a brief from the most-recent PFH run that suggested
        spec_id. Materializes the spec to fill in real Sharpe.
        Writes markdown + sidecar to data/research/briefs/.

    preview <spec_id>
        Generate brief but print to stdout instead of writing to disk.
    """
    import sys
    args = sys.argv[1:]
    cmd = args[0] if args else "preview"

    if cmd not in ("generate", "preview"):
        print("usage: generate <spec_id> [--no-llm] | preview <spec_id>",
              file=sys.stderr)
        raise SystemExit(2)
    if len(args) < 2:
        print("spec_id required", file=sys.stderr)
        raise SystemExit(2)

    spec_id = args[1]
    use_llm = "--no-llm" not in args

    # Find the most-recent PFH suggestion for this spec_id
    from engine.research.pfh.proposer import read_pfh_history
    for run in read_pfh_history(limit=50):
        for s in run.get("top", []):
            if s.get("proposal", {}).get("candidate_id") == spec_id:
                scored = s
                break
        else:
            continue
        break
    else:
        print(f"no PFH suggestion found for spec_id={spec_id!r}", file=sys.stderr)
        raise SystemExit(1)

    # Materialize for real Sharpe
    materialized = None
    try:
        from engine.feature_store import materialize_spec
        materialized = materialize_spec(spec_id, force=True, strict_sanity=False)
    except Exception as exc:
        print(f"# WARN materialize failed: {exc}", file=sys.stderr)

    artifact = generate_brief(scored, materialized=materialized, use_llm=use_llm)

    if cmd == "preview":
        print(artifact.markdown)
        return

    out = write_brief_to_disk(artifact)
    print(f"wrote {out}")
    if artifact.validation_warnings:
        print("warnings:", file=sys.stderr)
        for w in artifact.validation_warnings:
            print(f"  - {w}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
