"""scripts/probe_wrds_catalog.py — one-shot WRDS catalog probe.

User 2026-05-30: "拿一根探针把这里面的数据类目探全然后存储下来"

Connects to WRDS, lists all tables in finance-relevant schemas,
saves to data/cache/wrds_catalog.json. Runs ~30s on a healthy
WRDS connection. Should be re-run weekly (catalog rarely changes
but new tables get added).

USAGE:
  python scripts/probe_wrds_catalog.py                     # standard probe
  python scripts/probe_wrds_catalog.py --row-counts        # also COUNT(*) (slow!)
  python scripts/probe_wrds_catalog.py --schemas crsp,ibes # subset
  python scripts/probe_wrds_catalog.py --account ${WRDS_USER_2}   # alternate account

After probing, look at data/cache/wrds_catalog.json + decide which
DATA_INVENTORY tokens to wire as fetchers next.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--account", default="${WRDS_USER_1}",
                        choices=["${WRDS_USER_1}", "${WRDS_USER_2}"])
    parser.add_argument("--schemas", default=None,
                        help="comma-separated subset (default: all TARGET_SCHEMAS)")
    parser.add_argument("--row-counts", action="store_true",
                        help="also compute row counts (slow, WRDS-quota-heavy)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", default=None,
                        help="catalog output path (default: data/cache/wrds_catalog.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from engine.data.fetchers.wrds_catalog import (
        probe_wrds, save_catalog, summarize_catalog, CATALOG_PATH,
    )

    schemas = args.schemas.split(",") if args.schemas else None
    print(f"Probing WRDS account={args.account}, "
          f"schemas={schemas or 'TARGET_SCHEMAS default'}, "
          f"row_counts={args.row_counts}")
    catalog = probe_wrds(
        account=args.account,
        target_schemas=schemas,
        include_row_counts=args.row_counts,
        timeout_sec=args.timeout,
    )

    out_path = Path(args.out) if args.out else CATALOG_PATH
    save_catalog(catalog, out_path)

    summary = summarize_catalog(catalog)
    print()
    print("=" * 56)
    print("WRDS CATALOG PROBE SUMMARY")
    print("=" * 56)
    print(json.dumps(summary, indent=2, default=str))
    print()
    print(f"Saved to: {out_path}")
    if catalog.get("errors"):
        print()
        print(f"Errors ({len(catalog['errors'])}):")
        for err in catalog["errors"][:5]:
            print(f"  - {err}")
        if len(catalog["errors"]) > 5:
            print(f"  (... {len(catalog['errors']) - 5} more)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
