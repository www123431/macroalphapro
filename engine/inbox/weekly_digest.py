"""engine.inbox.weekly_digest — L2.3 weekly cross-paper digest.

⚠️  DEPRECATED 2026-06-04 (PR-A+B). Depends on the retired
papers_scored.jsonl pipeline (see engine.inbox.paper_fetcher). Reading
queue is now T7-sourced; a weekly digest over the T7 registry is a
follow-up PR.

──────────────────────────────────────────────────────────────────


Once a week: take the top 5-10 scored papers from the last 7 days, send
to Sonnet-4.6 with a digest rubric, write a single narrative summary
linking the papers to themes that matter for our deployed strategies.

Cost: ~$0.05/week (Sonnet input ~5k tokens + output ~800 tokens).

Output: data/research_ops/weekly_digest.jsonl — newest digest first.
Consumed by composer.py source_weekly_digest() as one item in the
Methodology lane.

Doctrine: the digest is META-RESEARCH — "what themes did papers explore
this week, and how do they map to our deployed strategies' improvement
directions or our graveyard". NOT trading commentary.
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
_SCORED         = _PAPERS_DIR / "papers_scored.jsonl"
_DIGEST         = _PAPERS_DIR / "weekly_digest.jsonl"


_DIGEST_SYSTEM = """You write the weekly research digest for a systematic quant research operations inbox.

CRITICAL: the consuming shop trades by deterministic formulas — they don't trade on news or predictions. Your digest is META-RESEARCH about what the academic literature explored this week, NOT trading commentary.

You will receive:
1. A list of scored papers (5-10) with title, abstract, family_match (mechanism families they touched in our deployed book), score, novelty, summary
2. The deployed sleeve composition: equity_book, cross_asset_carry, cross_asset_tsmom, crisis_hedge_tlt_gld, mom_hedge_overlay
3. Per-sleeve improvement_directions from active_deployment.yaml

Write a digest in this STRICT JSON format:
{
  "headline":          string (≤80 chars, the week's dominant theme),
  "narrative":         string (≤500 chars, integrates the 5-10 papers into 2-3 themes; references specific papers by short title),
  "improvement_directions_hit": [
    {"sleeve": "equity_book", "direction": "PIT FF12 within-sector ranking variants", "papers": [paper_id, ...]}
  ],
  "methodology_advances": [
    {"area": "deflated SR", "advance": "...", "papers": [paper_id, ...]}
  ],
  "graveyard_reinforcement": [
    {"mechanism": "...", "evidence": "...", "papers": [paper_id, ...]}
  ],
  "n_papers_summarized": int
}

Style: terse + technical. Use academic anchors when relevant (e.g., "extends Asness-Frazzini-Pedersen 2014"). NEVER suggest trade actions or predictions."""


def _load_recent_scored(days: int = 7) -> list[dict[str, Any]]:
    if not _SCORED.is_file():
        return []
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
    rows: list[dict[str, Any]] = []
    with _SCORED.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            try:
                ts = _dt.datetime.strptime(r.get("scored_ts", ""), "%Y-%m-%dT%H:%M:%SZ")
                if ts >= cutoff:
                    rows.append(r)
            except Exception:
                continue
    return rows


def _top_papers(rows: list[dict[str, Any]], k: int = 8) -> list[dict[str, Any]]:
    """Pick top-k by score, exclude killed, prefer relevant_to_deployed."""
    candidates = [r for r in rows if not (r.get("score", {}).get("kill"))]
    if not candidates:
        return []
    candidates.sort(key=lambda r: (
        1 if r.get("score", {}).get("relevant_to_deployed") else 0,
        r.get("score", {}).get("score", 0),
    ), reverse=True)
    return candidates[:k]


def _compose_prompt(papers: list[dict[str, Any]]) -> str:
    try:
        from engine.portfolio.deployed_registry import load_active
        cfg = load_active()
        sleeves_lines = []
        for s in cfg.sleeves:
            dirs = "; ".join(s.improvement_directions) if s.improvement_directions else "(none)"
            sleeves_lines.append(f"  - {s.name} ({s.role}, weight {s.base_weight:.2f}) "
                                 f"→ improvement_directions: {dirs}")
        sleeves_block = "\n".join(sleeves_lines)
    except Exception:
        sleeves_block = "(deployed_registry unavailable)"

    papers_block_lines = []
    for p in papers:
        sc = p.get("score", {})
        families = ", ".join({f["family"] for f in p.get("family_match", [])})
        papers_block_lines.append(
            f"  - id={p['id']} | source={p.get('source')} | family={families}\n"
            f"    score={sc.get('score')} novelty={sc.get('novelty')} relevant={sc.get('relevant_to_deployed')}\n"
            f"    title: {p.get('title','')[:160]}\n"
            f"    gist:  {sc.get('summary_one_line','')[:160]}"
        )
    papers_block = "\n".join(papers_block_lines)

    return (
        f"Deployed sleeves:\n{sleeves_block}\n\n"
        f"Top {len(papers)} scored papers this week:\n{papers_block}\n\n"
        f"Write the weekly digest per the rubric. Return ONLY JSON."
    )


def _llm_call(prompt: str) -> Optional[dict[str, Any]]:
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic SDK missing")
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
        logger.error("no ANTHROPIC_API_KEY for weekly digest")
        return None

    import time as _time
    t0 = _time.perf_counter()
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            temperature=0.2,
            system=_DIGEST_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.exception("weekly_digest LLM failed: %s", exc)
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
        cost = compute_cost(model="claude-sonnet-4-6",
                            input_tokens=in_tok, output_tokens=out_tok)
        record_call(
            agent_id="research_ops_weekly_digest",
            provider="anthropic",
            model="claude-sonnet-4-6",
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            cost_usd=cost,
            latency_ms=int(elapsed_s * 1000),
            scope="weekly_digest",
            extra={"n_papers_in_prompt": prompt.count("id=px_")},
        )
    except Exception:
        logger.exception("weekly_digest: cost ledger write failed (non-fatal)")

    # Strip markdown code fences if present
    answer = answer.strip()
    if answer.startswith("```"):
        lines = answer.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        answer = "\n".join(lines)

    try:
        return json.loads(answer)
    except Exception:
        logger.warning("weekly_digest: non-JSON response: %s", answer[:300])
        return None


def run(*, days: int = 7, top_k: int = 8) -> dict[str, Any]:
    """Generate one weekly digest from the last `days` of scored papers."""
    rows = _load_recent_scored(days=days)
    if not rows:
        return {"status": "no_scored_papers"}

    top = _top_papers(rows, k=top_k)
    if not top:
        return {"status": "no_qualifying_papers", "n_total_recent": len(rows)}

    prompt = _compose_prompt(top)
    digest = _llm_call(prompt)
    if digest is None:
        return {"status": "llm_failed", "n_top": len(top)}

    digest_row = {
        "id":           f"digest_{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        "ts":           _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_days":  days,
        "n_top_papers": len(top),
        "n_total_recent": len(rows),
        "paper_ids":    [p["id"] for p in top],
        "digest":       digest,
    }
    _PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    with _DIGEST.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(digest_row, ensure_ascii=False) + "\n")
    return {"status": "ok", "digest_id": digest_row["id"], "n_papers": len(top)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="L2.3 weekly digest")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()
    print(json.dumps(run(days=args.days, top_k=args.top_k), indent=2))
