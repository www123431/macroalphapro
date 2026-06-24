"""engine.research_store.manifest — reproducibility manifest helpers.

A research event without a git_sha is worth half as much. This module
provides the minimal helpers to capture the current git HEAD at emit
time, with a graceful fallback if not inside a git repo.

We intentionally do NOT bundle lib versions / data mtimes here — those
go in artifact-specific metadata (see e.g. PipelineReport.repro_manifest
already in the codebase). Keeping this lean.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def current_git_sha() -> str:
    """Return short HEAD SHA, or 'unknown' if git unavailable / not a repo.
    Never raises — a failure here must not block event emission."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("utf-8").strip()
    except Exception:
        logger.debug("git sha lookup failed", exc_info=True)
        return "unknown"
