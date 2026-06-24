"""scripts/compact_papers_registry.py — papers_registry catalog dedup.

papers_registry is a CATALOG (state of each paper), not an event log.
Past ingestion code created NEW paper_ids when ingesting a paper that
already had a metadata-only stub, leaving duplicate paper_ids per DOI.

This script:
  1. Loads the registry
  2. Deduplicates: for each (paper_id), keeps latest version
  3. Deduplicates again: for each (DOI), keeps the canonical row
     (highest n_chunks → highest version → most recent updated_ts)
  4. UNIONS shelf labels from dropped rows into the canonical row
     (preserves the "why each version was created" info)
  5. Adds `dropped_duplicates: [pid1, pid2, ...]` to canonical note
  6. Backs up old registry to papers_registry.jsonl.bak_TIMESTAMP
  7. Atomically rewrites papers_registry.jsonl with canonical rows only

The event-log stores (events / hypotheses / verdicts) remain strictly
append-only — they're different beasts.

Safety:
  - Default DRY-RUN: prints plan, no writes
  - --write requires explicit flag
  - Cross-reference check: if any hypotheses/verdicts/events reference
    a to-be-dropped paper_id, ABORT (would create dangling refs)

Usage:
  python scripts/compact_papers_registry.py            # dry-run
  python scripts/compact_papers_registry.py --write    # apply
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


REGISTRY_PATH = (_REPO_ROOT / "data" / "research_store"
                  / "papers_registry.jsonl")

# Stores to scan for cross-references — if any line contains a
# to-be-dropped paper_id we abort
REF_FILES = [
    _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl",
    _REPO_ROOT / "data" / "strengthener"   / "verdicts.jsonl",
    _REPO_ROOT / "data" / "research_store" / "events.jsonl",
]


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_all_rows(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception as exc:
                print(f"WARN: malformed line skipped: {exc}",
                      file=sys.stderr)
    return out


def _latest_per_paper_id(rows: list[dict]) -> dict[str, dict]:
    by_pid: dict[str, dict] = {}
    for r in rows:
        pid = r.get("paper_id", "")
        if not pid:
            continue
        prior = by_pid.get(pid)
        if prior is None or r.get("version", 1) > prior.get("version", 1):
            by_pid[pid] = r
    return by_pid


def _collapse_parent_chains(by_pid: dict[str, dict]) -> dict[str, dict]:
    """Pre-2026-06-07 amend_entry minted a NEW paper_id per amendment
    + set parent_paper_id pointing back. That left "orphan chains":
    a root paper_id (v1, original) plus 1-N child paper_ids each
    holding amendment state.

    Critical: external stores (hypotheses.jsonl etc.) reference the
    ROOT paper_id, NOT the leaves. So we must KEEP the root's
    paper_id but MERGE the leaf's state (tier, version, updated_ts,
    etc.) into it. Drop all descendants.

    Returns {root_pid → merged_row}.
    """
    # Build child lookup: parent_pid → [child_row, ...]
    children: dict[str, list[dict]] = {}
    for row in by_pid.values():
        parent = row.get("parent_paper_id")
        if parent:
            children.setdefault(parent, []).append(row)

    # A row is a "root" if it has no parent_paper_id set
    roots: list[dict] = [
        row for row in by_pid.values()
        if not row.get("parent_paper_id")
    ]

    out: dict[str, dict] = {}
    for root in roots:
        # Walk chain: root → child1 → child2 → ... (DFS deepest leaf)
        chain = [root]
        cur = root
        while True:
            kids = children.get(cur["paper_id"], [])
            if not kids:
                break
            # Pick the highest-version child (in case of fork, take
            # the most-amended branch)
            cur = sorted(kids, key=lambda r: -r.get("version", 0))[0]
            chain.append(cur)
        leaf = chain[-1]

        # Merge: keep root's paper_id + parent_paper_id (None for root)
        # but adopt leaf's state for everything else. UNION shelves +
        # other list fields across the chain.
        merged = dict(leaf)
        merged["paper_id"] = root["paper_id"]
        merged["parent_paper_id"] = None
        # Union shelves across chain
        seen_shelves: set = set()
        union_shelves: list = []
        for r in chain:
            for s in (r.get("shelves") or []):
                if s not in seen_shelves:
                    union_shelves.append(s)
                    seen_shelves.add(s)
        merged["shelves"] = union_shelves
        out[root["paper_id"]] = merged
    return out


def _pick_canonical(rows: list[dict]) -> dict:
    """Among rows sharing a DOI: highest n_chunks → highest version
    → most recent updated_ts."""
    return sorted(rows, key=lambda r: (
        -(r.get("n_chunks") or 0),
        -(r.get("version") or 0),
        r.get("updated_ts", ""),
    ))[0]


def _union_shelves(canonical: dict, dropped: list[dict]) -> list[str]:
    """Union shelf labels — keep canonical's order, append dropped's
    additions."""
    out = list(canonical.get("shelves") or [])
    seen = set(out)
    for d in dropped:
        for s in (d.get("shelves") or []):
            if s not in seen:
                out.append(s)
                seen.add(s)
    return out


def _scan_cross_refs(dropped_pids: set[str]) -> dict[str, int]:
    """Returns {file_path: n_lines_referencing_dropped} — non-zero
    anywhere means abort."""
    hits = {}
    for fp in REF_FILES:
        if not fp.is_file():
            hits[str(fp)] = 0
            continue
        n = 0
        with fp.open("r", encoding="utf-8") as f:
            for ln in f:
                # Cheap substring check — false positives are tolerable
                # (just makes safety check stricter, not weaker)
                if any(pid in ln for pid in dropped_pids):
                    n += 1
        hits[str(fp)] = n
    return hits


def build_compaction_plan() -> dict:
    """Pure-function plan builder. Returns:
      {
        rows_before:       int (raw),
        rows_after:        int (canonical),
        kept:              list[dict] canonical rows (already UNION'd),
        dropped_pids:      list[str],
        merges:            list[{doi, canonical_pid, dropped_pids, ...}],
      }
    """
    raw = _load_all_rows(REGISTRY_PATH)
    by_pid = _latest_per_paper_id(raw)

    # Collapse parent chains from pre-2026-06-07 amend_entry pattern
    # (which minted new paper_id per amendment). Keep only LEAVES.
    n_before_chain_collapse = len(by_pid)
    by_pid = _collapse_parent_chains(by_pid)
    n_chain_collapsed = n_before_chain_collapse - len(by_pid)

    # Group by DOI for dedup
    by_doi: dict[str, list[dict]] = defaultdict(list)
    no_doi_rows: list[dict] = []
    for p in by_pid.values():
        d = (p.get("doi") or "").strip().lower()
        if d:
            by_doi[d].append(p)
        else:
            no_doi_rows.append(p)

    kept: list[dict] = []
    dropped_pids: list[str] = []
    merges: list[dict] = []

    # DOI-keyed: pick canonical + merge metadata
    for doi, rows in by_doi.items():
        if len(rows) == 1:
            kept.append(rows[0])
            continue
        canonical = _pick_canonical(rows)
        dropped = [r for r in rows if r["paper_id"] != canonical["paper_id"]]

        # UNION shelves
        merged_shelves = _union_shelves(canonical, dropped)
        # Annotate note
        dropped_pids_short = [d["paper_id"][:8] for d in dropped]
        merge_note = (f"merged from {len(dropped)} stub paper_id(s) "
                       f"sharing DOI: "
                       f"{', '.join(dropped_pids_short)}")
        new_note = canonical.get("note", "") or ""
        if new_note:
            new_note = f"{new_note} | {merge_note}"
        else:
            new_note = merge_note

        canonical_out = dict(canonical)
        canonical_out["shelves"] = merged_shelves
        canonical_out["note"] = new_note
        # Bump version + stamp
        canonical_out["version"] = (canonical.get("version", 1) + 1)
        canonical_out["updated_ts"] = _utc_iso()

        kept.append(canonical_out)
        for d in dropped:
            dropped_pids.append(d["paper_id"])
        merges.append({
            "doi":           doi,
            "title":         (canonical.get("title") or "")[:60],
            "canonical_pid": canonical["paper_id"][:8],
            "dropped_pids":  [d["paper_id"][:8] for d in dropped],
            "n_dropped":     len(dropped),
        })

    # No-DOI rows: keep all (can't dedup without DOI)
    kept.extend(no_doi_rows)

    return {
        "rows_before":  len(raw),
        "rows_after":   len(kept),
        "n_distinct_pids_before":  n_before_chain_collapse,
        "n_chain_collapsed":       n_chain_collapsed,
        "n_distinct_after_chain":  len(by_pid),
        "kept":         kept,
        "dropped_pids": dropped_pids,
        "merges":       merges,
        "no_doi_kept":  len(no_doi_rows),
    }


def _backup_then_write(kept: list[dict]) -> Path:
    """Atomic-ish: write to .tmp, fsync-rename. Backup original first."""
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    bak = REGISTRY_PATH.with_suffix(f".jsonl.bak_{stamp}")
    shutil.copyfile(REGISTRY_PATH, bak)

    tmp = REGISTRY_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, REGISTRY_PATH)
    return bak


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true",
                    help="Actually rewrite registry (default dry-run)")
    args = p.parse_args()

    plan = build_compaction_plan()

    print(f"Registry compaction plan:")
    print(f"  raw rows:                {plan['rows_before']}")
    print(f"  distinct paper_ids:      {plan['n_distinct_pids_before']}")
    print(f"  after chain collapse:    {plan['n_distinct_after_chain']}"
          f"  ({plan['n_chain_collapsed']} orphan amendments dropped)")
    print(f"  → rows after compaction: {plan['rows_after']}")
    print(f"  no-DOI rows kept as-is:  {plan['no_doi_kept']}")
    print(f"  paper_ids to drop (DOI dup): {len(plan['dropped_pids'])}")
    print()

    if plan["merges"]:
        print("Merges (canonical kept, stubs dropped):")
        for m in plan["merges"]:
            print(f"  doi: {m['doi']}")
            print(f"    title:    {m['title']}")
            print(f"    canonical:{m['canonical_pid']}")
            print(f"    dropping: {m['dropped_pids']}")
        print()

    # Any change at all? (chain collapse OR DOI dedup)
    if (plan["rows_after"] == plan["rows_before"]
        and plan["n_chain_collapsed"] == 0):
        print("Nothing to compact. Exit.")
        return 0

    # Safety: cross-ref check. Must include BOTH chain-collapsed
    # orphans AND DOI-dedup drops.
    raw_pids = {r.get("paper_id") for r in _load_all_rows(REGISTRY_PATH)
                  if r.get("paper_id")}
    kept_pids = {r["paper_id"] for r in plan["kept"]}
    dropped_set = raw_pids - kept_pids
    print(f"Cross-reference safety check (all {len(dropped_set)} "
          f"to-be-dropped pids referenced in audit logs)...")
    refs = _scan_cross_refs(dropped_set)
    any_refs = False
    for fp, n in refs.items():
        print(f"  {fp}: {n} lines reference dropped pids")
        if n > 0:
            any_refs = True
    if any_refs:
        print()
        print("ABORT: cross-references exist. Compaction would create "
              "dangling refs. Manual review needed.", file=sys.stderr)
        return 1
    print("  → clean. No dangling refs would result.")
    print()

    if not args.write:
        print("DRY RUN — re-run with --write to compact.")
        return 0

    bak = _backup_then_write(plan["kept"])
    print(f"WROTE: {REGISTRY_PATH.name} ({plan['rows_after']} rows)")
    print(f"BACKUP: {bak.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
