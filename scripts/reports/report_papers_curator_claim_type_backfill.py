"""Backfill claim_type onto existing papers_curator summaries + write coverage report.

Run:  python scripts/reports/report_papers_curator_claim_type_backfill.py

W4 of six-week-critical-path, 2026-06-21. Stage 0 of the
papers_curator pipeline classifies each paper into one of 8
ClaimType values BEFORE the LLM summarizer runs. The W4-piece-1
deterministic router was just shipped; this script measures its
coverage on the existing 108-summary backlog so we can decide:

  - If UNKNOWN rate is low (< 20%): deterministic router is good
    enough; W4-piece-2 LLM fallback can be DEFERRED.
  - If UNKNOWN rate is medium (20-40%): keyword tables expand in
    W4-piece-2, no LLM yet.
  - If UNKNOWN rate is high (> 40%): W4-piece-2 must add LLM
    fallback (Haiku, ~$0.001 per paper).

Outputs:
  - data/papers_curator/summaries_with_claim_type.jsonl (backfilled
    copy, leaves original untouched)
  - data/papers_curator/claim_type_coverage_report.md (audit report)
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import os

# Load API keys from .streamlit/secrets.toml so the deepseek fallback
# can call out. The engine.llm.providers expect either env vars OR
# streamlit runtime; this script is plain Python so env-var is needed.
_SECRETS = REPO_ROOT / ".streamlit" / "secrets.toml"
if _SECRETS.is_file():
    for _ln in _SECRETS.read_text(encoding="utf-8").splitlines():
        if "=" in _ln and not _ln.strip().startswith("#"):
            _k, _, _v = _ln.partition("=")
            _v = _v.strip().strip('"').strip("'")
            os.environ.setdefault(_k.strip(), _v)

from engine.agents.papers_curator.claim_type_router import (
    classify,
    classify_hybrid,
)
from engine.hypothesis_spec.enums import ClaimType


# 2026-06-21 revision: classify on cache.jsonl (original paper title +
# abstract). summaries.jsonl is LLM-rewritten thesis — keyword router
# tuned to academic-formal language hits worse on rewrites than on
# originals. Production router will see cache.jsonl inputs, so test on
# that.
IN_PATH       = REPO_ROOT / "data" / "papers_curator" / "cache.jsonl"
OUT_JSONL     = REPO_ROOT / "data" / "papers_curator" / "cache_with_claim_type.jsonl"
OUT_REPORT_MD = REPO_ROOT / "data" / "papers_curator" / "claim_type_coverage_report.md"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-llm", action="store_true",
                       help="Enable LLM fallback for UNKNOWN cases (W4-piece-2). "
                            "Costs ~$0.0003/UNKNOWN paper via Deepseek-v4-pro.")
    ap.add_argument("--limit-llm", type=int, default=None,
                       help="Cap LLM calls (for cost-controlled sample runs).")
    args = ap.parse_args()

    if not IN_PATH.is_file():
        print(f"[error] {IN_PATH} not found")
        return

    print(f"[1/3] reading {IN_PATH.relative_to(REPO_ROOT)}...")
    rows: list[dict] = []
    with IN_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"      n_papers = {len(rows)}")

    mode = ("hybrid (det + LLM fallback)" if args.use_llm
              else "deterministic-only")
    print(f"[2/3] classifying via {mode}...")
    counts: collections.Counter = collections.Counter()
    confidence_buckets: dict[str, list[float]] = collections.defaultdict(list)
    backfilled: list[dict] = []
    unknown_samples: list[dict] = []
    llm_calls_made = 0

    for i, r in enumerate(rows):
        title    = r.get("title") or ""
        abstract = r.get("abstract") or ""
        # Always run deterministic first
        v_det = classify(title, abstract)
        # If UNKNOWN + LLM enabled + under cap → escalate
        if (v_det.claim_type == ClaimType.UNKNOWN
                and args.use_llm
                and (args.limit_llm is None or llm_calls_made < args.limit_llm)):
            v = classify_hybrid(title, abstract, llm_fallback=True)
            llm_calls_made += 1
            if i % 25 == 0:
                print(f"      [{i+1}/{len(rows)}] LLM calls so far: {llm_calls_made}")
        else:
            v = v_det
        ct = v.claim_type.value
        counts[ct] += 1
        confidence_buckets[ct].append(v.confidence)
        backfilled.append({
            **r,
            "claim_type":             ct,
            "claim_type_confidence":  v.confidence,
            "claim_type_router":      "deterministic_v1_2026-06-21",
        })
        if v.claim_type == ClaimType.UNKNOWN and len(unknown_samples) < 10:
            unknown_samples.append({
                "title":    (title or "")[:150],
                "abstract": (abstract or "")[:200],
            })
    # Re-render UNKNOWN sample dicts: caller below expects 'thesis'/'mechanism'
    # keys but cache.jsonl uses 'title'/'abstract' — keep both schemas consistent
    # for the markdown writer below.

    print(f"      counts: {dict(counts)}")

    print(f"[3/3] writing {OUT_JSONL.relative_to(REPO_ROOT)} + report...")
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for r in backfilled:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(rows)
    unknown_rate = counts.get("UNKNOWN", 0) / total if total else 0.0
    if unknown_rate < 0.20:
        verdict = "GOOD - deterministic router sufficient; defer LLM fallback"
    elif unknown_rate < 0.40:
        verdict = "MEDIUM - expand keyword tables in W4-piece-2; no LLM yet"
    else:
        verdict = "HIGH - W4-piece-2 must add LLM (Haiku) fallback"

    lines = [
        "# papers_curator ClaimType Coverage Report",
        "",
        f"_Backfilled from `data/papers_curator/cache.jsonl`. "
        f"n = {total} papers. UNKNOWN = router failure rate (not "
        f"a valid class); OTHER = LLM says \"fits no specific class\"._",
        "",
        f"## Coverage verdict: **{verdict}**",
        f"UNKNOWN rate (router failure): {unknown_rate:.1%}",
        f"OTHER rate (LLM says no fit):  "
        f"{counts.get('OTHER', 0) / total:.1%}" if total else "",
        "",
        "## Per-class breakdown",
        "",
        "| claim_type | n | fraction | mean confidence |",
        "|---|---|---|---|",
    ]
    for ct_name, n in counts.most_common():
        frac = n / total
        confs = confidence_buckets.get(ct_name, [])
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        lines.append(f"| {ct_name} | {n} | {frac:.1%} | {mean_conf:.3f} |")

    if unknown_samples:
        lines += [
            "",
            "## Sample UNKNOWN rows (first 10)",
            "",
        ]
        for i, s in enumerate(unknown_samples, 1):
            lines.append(f"{i}. title: {s['title']}")
            if s["abstract"]:
                lines.append(f"   abstract: {s['abstract']}")
            lines.append("")

    lines += [
        "## Next steps",
        "",
        "Based on the verdict above:",
        "- If GOOD: wire router into `engine/agents/papers_curator/summarizer.py` (W4-piece-3) so new summaries get `claim_type` tagged at write time. No LLM cost added.",
        "- If MEDIUM: inspect UNKNOWN samples above + extend keyword tables in `claim_type_router.py`. Re-run this report.",
        "- If HIGH: build LLM fallback in `claim_type_router.py` using Haiku at ~$0.001/paper. Re-run this report.",
        "",
        "## How to re-run",
        "",
        "```bash",
        "python scripts/reports/report_papers_curator_claim_type_backfill.py",
        "```",
        "",
        "## Why this report exists",
        "",
        "The papers_curator architecture (spec_papers_curator_full_architecture_2026-06-05) places ClaimType routing as Stage 0 — before the LLM summarizer. The router is the substrate; FACTOR_HYPOTHESIS-vs-METHODOLOGY-vs-DECAY_STUDY routing determines what downstream gates apply, what extractor schema is invoked, and what audit trail is expected. Without classification, every paper goes through the same generic schema (the pre-W4 state, where all 108 summaries have `claim_type = None`).",
        "",
    ]

    OUT_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"      report: {OUT_REPORT_MD.relative_to(REPO_ROOT)}")
    print()
    print(f"=== summary ===")
    print(f"  total summaries: {total}")
    print(f"  UNKNOWN rate:    {unknown_rate:.1%}")
    print(f"  verdict:         {verdict}")
    if args.use_llm:
        print(f"  LLM calls made:  {llm_calls_made}")


if __name__ == "__main__":
    main()
