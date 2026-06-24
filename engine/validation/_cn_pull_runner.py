"""engine/validation/_cn_pull_runner.py — resumable background runner for the
China A-share PEAD data pull. Logs to data/cache/cn/_pull.log (line-flushed) so
progress is monitorable; both pull_eps and fetch_cn_prices are skip-cached, so
re-running after an interruption resumes where it left off.

Run: python -u engine/validation/_cn_pull_runner.py
"""
from __future__ import annotations

import logging
import sys

from engine.validation import cn_pead_data as cn

LOG_PATH = "data/cache/cn/_pull.log"


def _setup_logging() -> None:
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    for h in (fh, sh):
        h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [fh, sh]


def main() -> None:
    _setup_logging()
    log = logging.getLogger("cn_runner")

    log.info("=== EPS pull (resumable, 7 missing quarters expected) ===")
    cn.pull_eps()
    p = cn.load_eps_panel()
    log.info("EPS panel now: %d rows, %d stocks, %d quarter-files",
             len(p), p["code"].nunique(),
             p["report_date"].nunique() if "report_date" in p else -1)

    log.info("=== price pull (resumable, target = CSI300+500 universe) ===")
    px = cn.fetch_cn_prices()
    log.info("DONE prices: %d rows, %d codes, %s..%s",
             len(px), px["code"].nunique(), px["date"].min(), px["date"].max())


if __name__ == "__main__":
    main()
