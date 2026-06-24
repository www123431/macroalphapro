"""engine.inbox.paper_scorer — L2.2 Haiku-4.5 relevance scoring.

⚠️  DEPRECATED 2026-06-04 (PR-A+B). See engine.inbox.paper_fetcher
deprecation note. The /lab/literature reading queue is now sourced
from the T7 paper registry (engine.inbox.composer.source_papers_from_t7);
shelf assignment replaces the Haiku numeric score.

──────────────────────────────────────────────────────────────────


Reads survivors from data/research_ops/papers_pre_filtered.jsonl (written
by L2.1), sends title + abstract + family_match to Haiku-4.5 with a
strict scoring rubric, writes scored papers to
data/research_ops/papers_scored.jsonl.

Cost expectation: ~20 papers/week × ~200 tokens out × $5/M = ~$0.02/wk.

Doctrine — the scoring rubric is RESEARCH-PROCESS not TRADE-INTEL:
  - score 0-10 for relevance to research process
  - novelty: improvement / extension / refutation / methodology / dead
  - explicit `kill_reason` field that fires for trade-intel papers
    ("this paper predicts X will outperform" → killed)
  - the LLM is told this is a 0-LLM-in-DECISION shop and trade-intel
    is excluded by doctrine

Logged to engine.llm_cost_ledger under agent_id="research_ops_paper_scorer".
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
_PAPERS_DIR     = _REPO_ROOT / "data" / "research_ops"
_PRE_FILTERED   = _PAPERS_DIR / "papers_pre_filtered.jsonl"
_SCORED         = _PAPERS_DIR / "papers_scored.jsonl"


_RUBRIC_SYSTEM = """You score academic finance papers for a quantitative research operations inbox.

CRITICAL DOCTRINE: the consuming shop trades by deterministic formulas. They DO NOT trade on news or predictions. Trade-intel papers ("this strategy will work going forward", "X factor is positive now") are USELESS to them and must be killed.

What IS useful (high relevance):
- Methodology improvements: new statistical tests, better cross-validation, deflated Sharpe variants, multi-test correction
- Mechanism research: papers studying the SAME family as deployed sleeves (PEAD, carry, TSMOM, BAB, tail hedge) at any of: improvement / extension / refutation
- Risk-model improvements (Ledoit-Wolf, BARRA, factor model construction)
- Portfolio construction theory (mean-variance with constraints, risk parity, Black-Litterman)
- Honest negatives: papers showing X mechanism DIED out-of-sample (reinforces graveyard)
- Replication studies (validate or invalidate published claims)

What is NOT useful (kill):
- Position-specific predictions / forecasts
- "X stock will go up because Y"
- Macro forecasts / outlook reports
- Generic "we found a factor that works" without rigorous OOS / multi-test
- Trade-actionable signals

Output STRICT JSON only (no preamble):
{
  "score": 0-10 (integer, 0=useless, 5=worth reading, 8+=actionable for research direction),
  "novelty": "improvement" | "extension" | "refutation" | "methodology" | "dead" | "irrelevant",
  "relevant_to_deployed": true | false (true = directly improves a sleeve we run),
  "kill": true | false (true if trade-intel / forecast / not research-process),
  "kill_reason": string ("" if kill=false),
  "lane_hint": "direction" | "methodology" | "graveyard",
  "summary_one_line": string (≤120 chars, the gist)
}"""


def _score_with_haiku(paper: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Send one paper to Haiku-4.5. Returns parsed JSON or None on failure."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic SDK not installed; paper scorer cannot run")
        return None

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try:
            import toml
            sec = toml.load(_REPO_ROOT / ".streamlit" / "secrets.toml")
            key = sec.get("ANTHROPIC_API_KEY") or sec.get("anthropic_api_key")
        except Exception:
            pass
    if not key:
        logger.error("ANTHROPIC_API_KEY not set; paper scorer cannot run")
        return None

    families = ", ".join(sorted({f["family"] for f in paper.get("family_match", [])}))
    user_msg = (
        f"Paper title: {paper.get('title','')}\n\n"
        f"Abstract: {paper.get('abstract','')[:2000]}\n\n"
        f"Source: {paper.get('source','')}\n\n"
        f"Keyword pre-filter matched these mechanism families in our deployed book:\n"
        f"  {families}\n\n"
        f"Score it per the rubric. Return ONLY the JSON."
    )

    import time as _time
    t0 = _time.perf_counter()
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            temperature=0.0,
            system=_RUBRIC_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        logger.exception("paper_scorer LLM call failed: %s", exc)
        return None

    elapsed_s = _time.perf_counter() - t0
    answer = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    # Cost ledger
    try:
        from engine.llm.pricing import compute_cost
        from engine.llm_cost_ledger import record_call
        u = getattr(resp, "usage", None)
        in_tok  = int(getattr(u, "input_tokens", 0) or 0)
        out_tok = int(getattr(u, "output_tokens", 0) or 0)
        cost_usd = compute_cost(
            model="claude-haiku-4-5",
            input_tokens=in_tok, output_tokens=out_tok,
        )
        record_call(
            agent_id="research_ops_paper_scorer",
            provider="anthropic",
            model="claude-haiku-4-5",
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            cost_usd=cost_usd,
            latency_ms=int(elapsed_s * 1000),
            scope=paper.get("source", ""),
            extra={"paper_id": paper.get("id"), "title": paper.get("title","")[:80]},
        )
    except Exception:
        logger.exception("paper_scorer: cost ledger write failed (non-fatal)")

    # Parse JSON. Sometimes Haiku wraps in markdown code-fence; strip.
    answer = answer.strip()
    if answer.startswith("```"):
        # remove first line + last ``` line
        lines = answer.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        answer = "\n".join(lines)

    try:
        out = json.loads(answer)
    except Exception:
        logger.warning("paper_scorer: non-JSON response, skipping: %s", answer[:200])
        return None

    return out


def _iter_unscored(limit: int) -> list[dict[str, Any]]:
    """Read pre-filtered jsonl, return papers where scored=False, up to limit."""
    if not _PRE_FILTERED.is_file():
        return []
    out: list[dict[str, Any]] = []
    with _PRE_FILTERED.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("scored"):
                continue
            out.append(r)
            if len(out) >= limit:
                break
    return out


def _mark_scored(paper_ids: list[str]) -> None:
    """Rewrite pre_filtered.jsonl marking these ids as scored."""
    if not _PRE_FILTERED.is_file():
        return
    rows = []
    with _PRE_FILTERED.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("id") in paper_ids:
                r["scored"] = True
            rows.append(r)
    with _PRE_FILTERED.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _append_scored(paper: dict[str, Any], score: dict[str, Any]) -> None:
    _PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    row = dict(paper)
    row["score"]      = score
    row["scored_ts"]  = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with _SCORED.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(*, limit: int = 30) -> dict[str, Any]:
    """Score up to `limit` unscored papers from pre_filtered.jsonl.

    Returns stats: { n_scored, n_killed, n_relevant_to_deployed }
    """
    unscored = _iter_unscored(limit)
    if not unscored:
        return {"n_scored": 0, "n_killed": 0, "n_relevant_to_deployed": 0}

    n_scored = 0
    n_killed = 0
    n_relevant = 0
    scored_ids: list[str] = []

    for p in unscored:
        score = _score_with_haiku(p)
        if score is None:
            continue
        _append_scored(p, score)
        scored_ids.append(p["id"])
        n_scored += 1
        if score.get("kill"):
            n_killed += 1
        if score.get("relevant_to_deployed"):
            n_relevant += 1

    if scored_ids:
        _mark_scored(scored_ids)

    return {
        "n_scored":              n_scored,
        "n_killed":              n_killed,
        "n_relevant_to_deployed": n_relevant,
        "ts":                    _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="L2.2 paper relevance scorer (Haiku-4.5)")
    ap.add_argument("--limit", type=int, default=20, help="max papers to score this run")
    ap.add_argument("--dry-run", action="store_true",
                    help="score but don't write scored.jsonl")
    args = ap.parse_args()

    if args.dry_run:
        unscored = _iter_unscored(args.limit)
        print(f"would score: {len(unscored)} papers")
        for p in unscored[:3]:
            print(f"  - [{p.get('source')}] {p.get('title','')[:80]}")
    else:
        print(json.dumps(run(limit=args.limit), indent=2))
