"""scripts/start_temporal_local.py — Phase 4c: long-running dev
Temporal server for the L4 outer-ring workflow.

Boots a local Temporalite server (downloaded automatically by the
temporalio Python SDK on first run; ~30MB; cached afterward) bound
to a fixed gRPC port + Web UI port — so the FastAPI shim can reach
it predictably across restarts.

Run with:
    python -m scripts.start_temporal_local

Default ports:
  gRPC: 7233   (TEMPORAL_ADDRESS used by l4_temporal_client)
  Web:  8233   (browse: http://localhost:8233)

The server is in-memory by default — restart wipes state. For
persistence, use a real `temporal` CLI install. For the lab scale
this lightweight setup is enough.

This script blocks forever; ctrl-c to stop.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

from temporalio.testing import WorkflowEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("temporal-local")


async def main() -> None:
    logger.info("Starting local Temporal dev server (may download ~30MB on first run)...")
    env = await WorkflowEnvironment.start_local(
        ip="127.0.0.1",
        port=7233,
        ui=True,
    )
    logger.info("Temporal up.")
    logger.info("  gRPC:    127.0.0.1:7233  (TEMPORAL_ADDRESS)")
    logger.info("  Web UI:  http://127.0.0.1:8233")
    logger.info("Ctrl-C to stop.")

    stop_event = asyncio.Event()

    def _request_stop(*_: object) -> None:
        logger.info("Shutdown requested.")
        stop_event.set()

    # Best-effort signal hookup (Windows + Unix)
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler — fall back to
                # the default KeyboardInterrupt path
                pass
    except Exception:
        pass

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down Temporal...")
        await env.shutdown()
        logger.info("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
