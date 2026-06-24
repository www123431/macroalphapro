"""scripts/slm_verify_ledger.py — verify the Merkle chain integrity of
the SLM state_transitions ledger.

Runs against the live production state store
(data/strategy_lifecycle.db) unless --db is overridden.

USAGE:
  python scripts/slm_verify_ledger.py
  python scripts/slm_verify_ledger.py --db custom/path.db
  python scripts/slm_verify_ledger.py --publish-head  # publish head for attestation

If --publish-head is set, the script appends the verified head_chain_hash
+ timestamp + git_sha to data/research/slm_ledger_attestations.jsonl —
the local equivalent of publishing to an immutable external store.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.research.merkle_ledger import (
    get_head_chain_hash, verify_ledger_integrity,
)
from engine.research.strategy_state_store import (
    DEFAULT_DB_PATH, _connect, init_db,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                     help="path to state_lifecycle.db (default: production)")
    p.add_argument("--publish-head", action="store_true",
                     help="append head_chain_hash to attestations log")
    args = p.parse_args()

    print("=" * 88)
    print(f" SLM Merkle ledger integrity check — {args.db.name}")
    print("=" * 88)

    init_db(args.db)
    conn = _connect(args.db)
    result = verify_ledger_integrity(conn)

    print(f"\n  total_rows:       {result.total_rows}")
    print(f"  chain_intact:     {result.chain_intact}")
    if result.head_chain_hash:
        print(f"  head_chain_hash:  {result.head_chain_hash}")
    print(f"  breaks_found:     {len(result.breaks)}")

    if result.breaks:
        print(f"\n  ── DETECTED BREAKS ────────────────────────────────────────────────")
        for b in result.breaks:
            print(f"    row id={b.row_id}  strategy={b.strategy_id}  at={b.transition_at}")
            print(f"      severity:           {b.severity}")
            print(f"      expected_chain_hash: {b.expected_chain_hash}")
            print(f"      stored_chain_hash:   {b.stored_chain_hash}")
        return 1

    print(f"\n  [OK] Ledger chain INTACT across {result.total_rows} transition rows.")

    if args.publish_head and result.head_chain_hash:
        attestation_log = Path(__file__).resolve().parent.parent / \
                          "data" / "research" / "slm_ledger_attestations.jsonl"
        attestation_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "db_path": str(args.db),
            "total_rows": result.total_rows,
            "head_chain_hash": result.head_chain_hash,
            "git_sha": _git_sha(),
        }
        with attestation_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  [PUBLISHED] head_chain_hash appended to "
              f"{attestation_log.relative_to(Path.cwd())}")
        print(f"             For external attestation: publish this hash to an")
        print(f"             immutable store (git tag / public gist / blockchain).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
