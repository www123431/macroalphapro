"""engine/research_store/hypothesis/dedup.py — per-paper hypothesis dedup.

Empirical audit (2026-06-15): A's extractor produces 6-12 hyps per paper
(avg 12 pre-prompt-tune, 6 post-tune). B's strengthener REJECTs 96%
because most are robustness checks / sub-group analyses / patches on
the CORE claim — not independent testable predictions.

This module collapses near-duplicate hyps within the same paper to one
"canonical" hyp per cluster. NO LLM call — token-Jaccard similarity is
fast and good enough for this task (we're catching paraphrases of the
same claim, not detecting semantic novelty).

Algorithm:
  1. Group hyps by source_paper_id
  2. Within each paper, compute pairwise claim_text token-Jaccard
  3. Cluster hyps where Jaccard >= TOKEN_JACCARD_THRESHOLD (0.55)
  4. From each cluster, keep the HIGHEST-QUALITY representative:
       - max(len(verbatim_quotes)) — more evidence-backed
       - tie-break: longer claim_text — more specific
       - tie-break: more required_data fields — more concrete
  5. Drop the rest

Target: 3-5 hyps per paper (vs current 6-12).
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Token-Jaccard threshold above which two hyps are considered duplicate.
# Empirically: 0.4 = aggressive (merges related-but-distinct claims),
# 0.55 = balanced (catches paraphrases without merging distinct hyps),
# 0.70 = conservative (only catches near-identical claims).
TOKEN_JACCARD_THRESHOLD = 0.55

# Per-paper hard cap. After token-Jaccard dedup, if a paper still has
# > PER_PAPER_CAP hyps, keep only the highest-quality PER_PAPER_CAP.
# Empirical (2026-06-15): a typical paper has 2-5 PRIMARY testable
# predictions. Anything beyond cap is robustness check / sub-group /
# methodology variant — those are NOT independent testable hyps and
# bloat downstream B's review queue.
PER_PAPER_CAP = 5

# Stop-words that should not contribute to similarity scoring.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "has", "have", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "us", "i", "my", "me", "you", "your",
    # quant common words — too generic to distinguish hyps
    "returns", "stock", "stocks", "return", "factor", "factors",
    "alpha", "premium", "premia", "predict", "predicts",
    "show", "shows", "find", "finds", "results",
})

_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{2,}")


def _tokens(text: str) -> set:
    """Tokenize + lowercase + drop stopwords."""
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _quality_score(hyp: dict) -> tuple:
    """Tuple for max-quality picking. Higher tuple wins."""
    claim_text = ""
    claim = hyp.get("claim")
    if isinstance(claim, str):
        claim_text = claim
    elif isinstance(claim, dict):
        claim_text = claim.get("one_line") or claim.get("text") or ""
    quotes = hyp.get("verbatim_quotes") or []
    req_data = hyp.get("required_data") or []
    # 1) more evidence quotes — more grounded
    # 2) longer claim — more specific
    # 3) more data fields — more concrete
    # 4) has predicted_magnitude — testable
    has_mag = 1 if hyp.get("predicted_magnitude") else 0
    return (len(quotes), len(claim_text), len(req_data), has_mag)


def _claim_text(hyp: dict) -> str:
    c = hyp.get("claim")
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return c.get("one_line") or c.get("text") or ""
    return ""


def dedup_paper_hyps(hyps: list[dict],
                      *,
                      threshold: float = TOKEN_JACCARD_THRESHOLD,
                      per_paper_cap: int = PER_PAPER_CAP,
                      preserve_ids: Optional[set] = None,
                      ) -> tuple[list[dict], list[dict]]:
    """Cluster + keep best per cluster.
    Returns (kept, dropped).

    Single-linkage clustering: if A~B and B~C then {A,B,C} cluster
    even if A and C are below threshold. Conservative for false-merge."""
    if len(hyps) <= 1:
        return list(hyps), []
    # Compute token sets once
    tokens = [_tokens(_claim_text(h)) for h in hyps]
    # Union-Find for single-linkage clustering
    n = len(hyps)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(tokens[i], tokens[j]) >= threshold:
                union(i, j)
    # Group by root
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    kept: list[dict] = []
    dropped: list[dict] = []
    preserve_ids = preserve_ids or set()

    def _is_preserved(h: dict) -> bool:
        return h.get("hypothesis_id") in preserve_ids

    for cluster_idxs in clusters.values():
        if len(cluster_idxs) == 1:
            kept.append(hyps[cluster_idxs[0]])
            continue
        # Pick best: prefer preserved if any, else highest quality
        cluster_preserved = [i for i in cluster_idxs if _is_preserved(hyps[i])]
        if cluster_preserved:
            # Keep ALL preserved + best non-preserved? Keep all preserved
            for i in cluster_preserved:
                kept.append(hyps[i])
            # Drop non-preserved in this cluster
            for i in cluster_idxs:
                if i not in cluster_preserved:
                    dropped.append(hyps[i])
        else:
            best_i = max(cluster_idxs, key=lambda i: _quality_score(hyps[i]))
            kept.append(hyps[best_i])
            for i in cluster_idxs:
                if i != best_i:
                    dropped.append(hyps[i])

    # Per-paper cap: if still over cap, drop lowest-quality
    # PRESERVED hyps are exempt from cap.
    if per_paper_cap and len(kept) > per_paper_cap:
        preserved_kept = [h for h in kept if _is_preserved(h)]
        non_preserved = [h for h in kept if not _is_preserved(h)]
        slots_left = max(0, per_paper_cap - len(preserved_kept))
        # Sort non-preserved by quality DESC, keep top `slots_left`
        ranked = sorted(non_preserved, key=_quality_score, reverse=True)
        kept_non_preserved = ranked[:slots_left]
        capped_dropped = ranked[slots_left:]
        # Reconstruct keep list (preserved + capped non-preserved), keep order
        keep_set = {id(h) for h in (preserved_kept + kept_non_preserved)}
        kept = [h for h in kept if id(h) in keep_set]
        dropped.extend(capped_dropped)
    return kept, dropped


def _auto_preserve_ids(repo_root: Path) -> set:
    """Hyps with downstream artifacts (α pre_mortem / γ replication /
    autopsies / verdicts) must NEVER be dropped — would orphan
    references. Reads:
      data/research/pre_mortems.jsonl
      data/research/replication_checks.jsonl
      data/research/autopsies.jsonl
      data/research_store/events.jsonl (factor_verdict_filed.metrics
                                         .source_hypothesis_id)
    """
    preserve: set = set()
    for rel in ("data/research/pre_mortems.jsonl",
                "data/research/replication_checks.jsonl",
                "data/research/autopsies.jsonl"):
        p = repo_root / rel
        if not p.is_file():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            hid = r.get("hypothesis_id")
            if hid:
                preserve.add(hid)
    # Verdict events
    ev_path = repo_root / "data" / "research_store" / "events.jsonl"
    if ev_path.is_file():
        for ln in ev_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("event_type") != "factor_verdict_filed":
                continue
            m = e.get("metrics") or {}
            hid = m.get("source_hypothesis_id") or m.get("hypothesis_id")
            if hid:
                preserve.add(hid)
    return preserve


def dedup_jsonl_in_place(path: Path,
                          *,
                          threshold: float = TOKEN_JACCARD_THRESHOLD,
                          dry_run: bool = False,
                          preserve_ids: Optional[set] = None,
                          ) -> dict:
    """Run dedup over hypotheses.jsonl in place. Returns stats."""
    rows: list[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue

    # Auto-collect preserve set if not given
    if preserve_ids is None:
        # Walk up from this module to find repo root
        repo_root = Path(__file__).resolve().parents[3]
        preserve_ids = _auto_preserve_ids(repo_root)

    # Group by source_paper_id
    by_paper: dict[str, list[dict]] = defaultdict(list)
    orphans: list[dict] = []   # no source_paper_id (e.g. brainstorm-derived)
    for r in rows:
        pid = r.get("source_paper_id")
        if pid:
            by_paper[pid].append(r)
        else:
            orphans.append(r)

    kept_all: list[dict] = list(orphans)   # orphans never dedup'd
    dropped_all: list[dict] = []
    paper_stats: list[dict] = []
    for pid, hyps in by_paper.items():
        kept, dropped = dedup_paper_hyps(hyps, threshold=threshold,
                                          preserve_ids=preserve_ids)
        kept_all.extend(kept)
        dropped_all.extend(dropped)
        if dropped:
            paper_stats.append({
                "paper_id":   pid,
                "before":     len(hyps),
                "after":      len(kept),
                "dropped":    len(dropped),
            })

    # Stable order: by created_ts ascending (preserve history)
    kept_all.sort(key=lambda r: r.get("created_ts") or "")

    if not dry_run:
        backup = path.with_suffix(path.suffix + ".pre_dedup_bak")
        backup.write_bytes(path.read_bytes())
        with path.open("w", encoding="utf-8") as f:
            for r in kept_all:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "n_total":        len(rows),
        "n_orphans":      len(orphans),
        "n_papers":       len(by_paper),
        "n_kept":         len(kept_all),
        "n_dropped":      len(dropped_all),
        "drop_pct":       round(len(dropped_all) / max(1, len(rows)) * 100, 1),
        "n_preserved":    len(preserve_ids),
        "paper_stats":    paper_stats,
        "threshold":      threshold,
        "dry_run":        dry_run,
    }
