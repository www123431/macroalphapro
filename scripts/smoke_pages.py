"""
scripts/smoke_pages.py — Reusable Streamlit AppTest smoke runner.

Loads each pages/*.py via Streamlit's AppTest and reports PASS / FAIL / TIMEOUT.

Why a custom runner (not pytest):
  AppTest cold-starts a fresh script every invocation, so st.cache_data is
  always empty. Pages that do yfinance downloads + MSM regime fit on first
  load (live_dashboard, signal_board) take ~40-80s cold even though they
  hit cache <1s in production. The default 30s budget mis-flagged these
  as "broken" in 2026-05-06 verification — they actually work, the cache
  just isn't warm in test.

Budget: 120s (covers worst observed cold-start, signal_board ~70s).

Usage:
  python scripts/smoke_pages.py                   # all pages
  python scripts/smoke_pages.py --fail-fast       # stop on first failure
  python scripts/smoke_pages.py --pages live_dashboard signal_board
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def smoke_one(page_path: str, timeout: int) -> dict:
    from streamlit.testing.v1 import AppTest

    t0 = time.time()
    try:
        at = AppTest.from_file(page_path, default_timeout=timeout).run()
        elapsed = time.time() - t0
        if at.exception:
            err = str(at.exception[0])[:300]
            return {"page": page_path, "status": "FAIL", "elapsed": elapsed, "error": err}
        return {"page": page_path, "status": "PASS", "elapsed": elapsed, "error": None}
    except Exception as e:
        elapsed = time.time() - t0
        msg = str(e)[:300]
        status = "TIMEOUT" if "timed out" in msg else "ERROR"
        return {"page": page_path, "status": status, "elapsed": elapsed, "error": msg}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-page timeout in seconds (default: 120)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first non-PASS result")
    parser.add_argument("--pages", nargs="*", default=None,
                        help="Page basenames to run (default: all in pages/)")
    args = parser.parse_args()

    pages = sorted(glob.glob(os.path.join(ROOT, "pages", "*.py")))
    if args.pages:
        wanted = set(args.pages)
        pages = [p for p in pages
                 if os.path.splitext(os.path.basename(p))[0] in wanted
                 or os.path.basename(p) in wanted]

    print(f"Smoke runner — {len(pages)} page(s), timeout={args.timeout}s")
    print("=" * 78)

    n_pass = n_fail = 0
    failures: list[dict] = []
    for p in pages:
        rel = os.path.relpath(p, ROOT)
        result = smoke_one(p, args.timeout)
        marker = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "TIMEOUT": "[TMO] ",
                  "ERROR": "[ERR] "}[result["status"]]
        print(f"  {marker} {rel:<45} {result['elapsed']:>6.1f}s")
        if result["status"] == "PASS":
            n_pass += 1
        else:
            n_fail += 1
            failures.append(result)
            print(f"         {result['error']}")
            if args.fail_fast:
                break

    print("=" * 78)
    print(f"SUMMARY: {n_pass} PASS / {n_fail} FAIL  ({len(pages)} total)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
