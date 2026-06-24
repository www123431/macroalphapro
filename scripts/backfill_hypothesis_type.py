"""scripts/backfill_hypothesis_type.py — one-time classification of
existing hypotheses.jsonl rows.

Reads data/research_store/hypotheses.jsonl, runs classify_hypothesis_type
on each row, writes the type into row["hypothesis_type"], rewrites the
file atomically (tmp + rename).

Idempotent: re-running on an already-classified row produces the same
type. Safe to run multiple times.

Usage:
  # Dry-run: print breakdown without modifying the file
  python scripts/backfill_hypothesis_type.py --dry-run

  # Real: rewrite hypotheses.jsonl with hypothesis_type populated
  python scripts/backfill_hypothesis_type.py

Safety
======
- The original file is BACKED UP to hypotheses.jsonl.bak.<ts> before
  rewrite. Restore by `mv hypotheses.jsonl.bak.<ts> hypotheses.jsonl`.
- Tmp-then-rename is atomic on Windows + POSIX.
- If the classifier raises on any row, the script aborts before
  writing — file is unchanged.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.research_store.hypothesis.classifier import (  # noqa: E402
    classify_hypothesis_type,
    hypothesis_type_breakdown,
)

HYPOTHESES_PATH = REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"


def _read_rows(p: Path) -> list[dict]:
    rows: list[dict] = []
    if not p.is_file():
        return rows
    with p.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARN: line {ln_no} malformed; skipping")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="backfill hypothesis_type")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not HYPOTHESES_PATH.is_file():
        print(f"[backfill] no hypotheses.jsonl at {HYPOTHESES_PATH}")
        return 0

    rows = _read_rows(HYPOTHESES_PATH)
    if not rows:
        print("[backfill] no rows to classify")
        return 0

    # Pre-classify pass — abort BEFORE writing if any classifier error
    classifications: list[str] = []
    for h in rows:
        try:
            classifications.append(classify_hypothesis_type(h))
        except Exception as exc:
            print(f"[backfill] classifier raised on hypothesis_id="
                  f"{h.get('hypothesis_id','?')}: {exc}")
            return 1

    print(f"[backfill] classified {len(rows)} hypotheses.")
    breakdown = hypothesis_type_breakdown(rows)
    print(f"[backfill] breakdown: {breakdown}")

    # Change detection — diff against existing field
    changed = sum(
        1 for h, c in zip(rows, classifications)
        if h.get("hypothesis_type") != c
    )
    print(f"[backfill] {changed} rows would change hypothesis_type.")

    if args.dry_run:
        print("[backfill] dry-run — file unchanged.")
        return 0

    # Backup
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    backup = HYPOTHESES_PATH.with_suffix(f".jsonl.bak.{ts}")
    backup.write_bytes(HYPOTHESES_PATH.read_bytes())
    print(f"[backfill] backup written: {backup.name}")

    # Atomic write via tmp
    tmp_path = HYPOTHESES_PATH.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for h, c in zip(rows, classifications):
            h["hypothesis_type"] = c
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    os.replace(tmp_path, HYPOTHESES_PATH)
    print(f"[backfill] {HYPOTHESES_PATH.relative_to(REPO_ROOT)} updated "
          f"({len(rows)} rows, {changed} changed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
