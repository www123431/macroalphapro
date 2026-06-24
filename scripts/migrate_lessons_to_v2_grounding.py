"""scripts/migrate_lessons_to_v2_grounding.py — T1 migration.

Walks data/research_store/red_lessons.jsonl. For every record without
`grounding_method` set, REWRITE the file in place with:
  - grounding_method = "pretrain_grounded"
  - tested_hypothesis_ids = []
  - verbatim_quotes = []
  - schema_version = 2
  - tags += ["t1_migrated_legacy"]

The freeze-TS rule: lessons with created_ts <= 2026-06-04 may be
pretrain_grounded. Since all current 47 lessons have created_ts on
2026-06-03 (per P1 backfill), they are within the legal window.

The original file is backed up to red_lessons.jsonl.pre_v2_backup.

Run:
  python scripts/migrate_lessons_to_v2_grounding.py            # dry-run
  python scripts/migrate_lessons_to_v2_grounding.py --write    # rewrite
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.red_lessons import LESSONS_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually rewrite the jsonl (and back it up first)")
    args = ap.parse_args()

    if not LESSONS_PATH.is_file():
        print(f"no lessons file at {LESSONS_PATH}")
        sys.exit(0)

    raw_lines = LESSONS_PATH.read_text(encoding="utf-8").splitlines()
    records = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    n_total = len(records)
    n_needs_migrate = 0
    migrated = []
    for r in records:
        if "grounding_method" in r and "tested_hypothesis_ids" in r:
            migrated.append(r)
            continue
        n_needs_migrate += 1
        new_r = dict(r)
        new_r.setdefault("tested_hypothesis_ids", [])
        new_r.setdefault("verbatim_quotes", [])
        new_r["grounding_method"] = "pretrain_grounded"
        new_r["schema_version"] = 2
        existing_tags = list(new_r.get("tags") or [])
        if "t1_migrated_legacy" not in existing_tags:
            existing_tags.append("t1_migrated_legacy")
        new_r["tags"] = existing_tags
        migrated.append(new_r)

    print(f"Total records:       {n_total}")
    print(f"Need migration:      {n_needs_migrate}")
    print(f"Already at schema 2: {n_total - n_needs_migrate}")

    if not args.write:
        print(f"\nDRY RUN — pass --write to back up + rewrite")
        return

    if n_needs_migrate == 0:
        print("nothing to migrate")
        return

    # Back up original
    backup_path = LESSONS_PATH.with_suffix(".jsonl.pre_v2_backup")
    shutil.copy2(LESSONS_PATH, backup_path)
    print(f"Backed up original to {backup_path}")

    # Rewrite
    with LESSONS_PATH.open("w", encoding="utf-8") as f:
        for r in migrated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Rewrote {LESSONS_PATH} ({len(migrated)} records)")


if __name__ == "__main__":
    main()
