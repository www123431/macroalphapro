"""scripts/start_l4_worker.py — Phase 4c: long-running worker
process for the L4 Temporal task queue.

Picks tasks off the `l4-discovery` task queue and runs the
L4DiscoveryWorkflow + activities (propose / critique) in this
process. Connects to the Temporal server started by
scripts/start_temporal_local.py.

Run with:
    python -m scripts.start_l4_worker

Requires the Temporal dev server to already be running at
TEMPORAL_ADDRESS (default localhost:7233).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("l4-worker")


async def main() -> None:
    from engine.research.l4_workflow import run_worker
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    logger.info("Connecting L4 worker to %s ...", address)
    try:
        await run_worker(address=address)
    except KeyboardInterrupt:
        logger.info("Worker shutdown.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
