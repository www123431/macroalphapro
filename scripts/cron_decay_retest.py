"""scripts/cron_decay_retest.py — daily auto-trigger for decay re-tests.

Reactive subscriber #4 of the deferred queue (2026-06-14). Runs every
morning right after cron_decay_audit (which produces the fresh decay
sentinel report). For each sleeve in WATCH/ACTION state:

  1. enqueue a retest (dedup 24h handles repeated alerts)
  2. process the queue (Chow + bootstrap → verdict)

Verdicts land in data/research/decay_retest_results.jsonl and the UI
surfaces them on /research/decay. Without this, every WATCH/ACTION
alert relies on the principal manually scheduling the retest; with it,
the verdict ("is this real decay?") is waiting when they open the app.

Cron registration: scripts/install_agentic_cron.py — daily 06:45 SGT
(immediately after the 06:30 daily memo / 06:40 workflow executor so
the daily memo can reference the freshest retest verdicts).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
HEALTH_PATH = REPO_ROOT / "data" / "agents" / "_health" / "decay_retest.jsonl"


def _record(status: str, *, elapsed_s: float, n_enqueued: int = 0,
            n_processed: int = 0, error: str | None = None) -> None:
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "agent_id":     "decay_retest",
        "ts":           _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status":       status,
        "elapsed_s":    round(elapsed_s, 2),
        "date_key":     _dt.date.today().isoformat(),
        "n_enqueued":   n_enqueued,
        "n_processed":  n_processed,
    }
    if error:
        row["error"] = error[:500]
    with HEALTH_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = _dt.datetime.utcnow()
    try:
        # 1) Read the latest decay sentinel artifact for WATCH/ACTION sleeves.
        from engine.agents.persona.tools import read_decay_sentinel_report
        from engine.research.decay_retest import enqueue_retest, process_queue
        rep = json.loads(read_decay_sentinel_report())
        mechs = rep.get("mechanisms") or {}
        flagged = [
            (name, m) for name, m in mechs.items()
            if m.get("structural_decay")
            or (m.get("rolling_sharpe") is not None
                and m.get("decay_ratio") is not None
                and m.get("decay_ratio") < 0.5)
        ]

        # 2) Enqueue each flagged sleeve (dedup handles repeats).
        n_enqueued = 0
        for name, _m in flagged:
            try:
                enqueue_retest(name, triggered_by="cron")
                n_enqueued += 1
            except Exception:
                logger.exception("enqueue failed for %s", name)

        # 3) Drain the queue.
        results = process_queue(limit=20)

        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("ok", elapsed_s=elapsed,
                n_enqueued=n_enqueued, n_processed=len(results))
        confirmed = sum(1 for r in results if r.verdict == "CONFIRMED_DECAY")
        print(f"[cron_decay_retest] ok — {n_enqueued} enqueued / "
              f"{len(results)} processed / {confirmed} CONFIRMED_DECAY "
              f"in {elapsed:.1f}s")
        return 0
    except Exception as exc:
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("error", elapsed_s=elapsed,
                error=f"{type(exc).__name__}: {exc}")
        logger.exception("cron_decay_retest failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
