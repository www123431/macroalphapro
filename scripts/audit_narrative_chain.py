"""
Narrative hash chain integrity audit (NARRATIVE-C v2, 2026-05-04).

For every PendingApproval row with review_narrative_snapshot present:
  1. recompute SHA-256 of stored snapshot text
  2. compare to stored review_narrative_hash → flag tamper if mismatch
  3. verify prev_narrative_hash points to an actual existing prior hash
     in the chain (None for the first row); flag broken chain otherwise

Run as part of compliance audit. Designed for institutional record-keeping
review (CFA GIPS / SEC 17a-4 / MiFID II reference).

Returns:
    {
      "n_with_snapshot":  int,
      "n_hash_match":     int,
      "n_hash_mismatch":  int,    # tampered / corrupted
      "n_chain_broken":   int,    # prev_narrative_hash doesn't exist in DB
      "chain_root_count": int,    # rows with prev=None (should be 1 for full chain)
      "issues":           list[{id, kind, detail}]
    }
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_audit() -> dict:
    from engine.memory import PendingApproval, SessionFactory

    n_with_snapshot = 0
    n_hash_match = 0
    n_hash_mismatch = 0
    n_chain_broken = 0
    chain_root_count = 0
    issues: list[dict] = []

    with SessionFactory() as s:
        rows = (
            s.query(PendingApproval)
             .filter(PendingApproval.review_narrative_snapshot.isnot(None))
             .all()
        )
        all_hashes = {r.review_narrative_hash for r in rows if r.review_narrative_hash}

        for r in rows:
            n_with_snapshot += 1
            recomputed = hashlib.sha256(
                (r.review_narrative_snapshot or "").encode("utf-8")
            ).hexdigest()
            if recomputed == r.review_narrative_hash:
                n_hash_match += 1
            else:
                n_hash_mismatch += 1
                issues.append({
                    "id":     int(r.id),
                    "kind":   "hash_mismatch",
                    "detail": f"stored={r.review_narrative_hash} vs recomputed={recomputed} (TAMPER)",
                })

            if r.prev_narrative_hash is None:
                chain_root_count += 1
            else:
                if r.prev_narrative_hash not in all_hashes:
                    n_chain_broken += 1
                    issues.append({
                        "id":     int(r.id),
                        "kind":   "chain_broken",
                        "detail": f"prev_narrative_hash={r.prev_narrative_hash} not found in DB",
                    })

    return {
        "n_with_snapshot":  n_with_snapshot,
        "n_hash_match":     n_hash_match,
        "n_hash_mismatch":  n_hash_mismatch,
        "n_chain_broken":   n_chain_broken,
        "chain_root_count": chain_root_count,
        "issues":           issues,
    }


def _print_report(out: dict) -> bool:
    print("=" * 78)
    print("Narrative Hash Chain Integrity Audit")
    print("=" * 78)
    print(f"  rows with snapshot:  {out['n_with_snapshot']}")
    print(f"  hash match:          {out['n_hash_match']}")
    print(f"  hash mismatch:       {out['n_hash_mismatch']}  (TAMPER if > 0)")
    print(f"  chain broken:        {out['n_chain_broken']}  (broken if > 0)")
    print(f"  chain root count:    {out['chain_root_count']}  (≤1 expected; >1 = chain forked)")
    if out["issues"]:
        print("\n  ISSUES:")
        for i in out["issues"]:
            print(f"    #{i['id']} {i['kind']}: {i['detail']}")
    print("=" * 78)
    ok = out["n_hash_mismatch"] == 0 and out["n_chain_broken"] == 0
    print("VERDICT: " + ("CHAIN INTACT" if ok else "INTEGRITY ISSUES PRESENT"))
    print("=" * 78)
    return ok


if __name__ == "__main__":
    out = run_audit()
    ok = _print_report(out)
    sys.exit(0 if ok else 1)
