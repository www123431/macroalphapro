"""engine/data/source_health.py — persistent source-availability tracker.

Records per-source failures (especially 429 rate limits + auth denials)
with cooldown windows. Orchestrator/fetchers query this BEFORE attempting
real fetches, skipping known-unhealthy sources to avoid quota burn /
IP-block escalation.

Per [[feedback-wrds-care-and-probe-pattern-2026-05-30]] and the arxiv 429
incident — repeated requests to a throttled API trigger longer bans, so
SOURCE HEALTH MUST PERSIST ACROSS PROCESS RUNS.

State: data/research/source_health.json (read-only at orchestrator runtime;
       updated when fetcher hits a failure).

Cooldown policy per error class (configurable):
  access_denied: 60 minutes
  rate_limited:  24 hours (especially aggressive for IP-attached APIs like arxiv)
  network:       10 minutes
  schema_unknown: 24 hours (probably code bug; needs human attention)
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HEALTH_FILE = REPO_ROOT / "data" / "research" / "source_health.json"

# Cooldown durations per error class (minutes)
COOLDOWN_MINUTES = {
    "access_denied":  60,
    "rate_limited":   24 * 60,    # 24h for rate-limit (avoid IP block)
    "network":        10,
    "schema_unknown": 24 * 60,    # 24h — needs human attention
    "auth_missing":   0,           # don't cooldown auth-missing; user can fix immediately
}


def _now() -> datetime.datetime:
    return datetime.datetime.utcnow()


def _read_state() -> dict:
    if not HEALTH_FILE.exists():
        return {}
    try:
        return json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def mark_failure(source: str, error_class: str, error_message: str = "") -> None:
    """Record a failure with cooldown duration based on error_class."""
    cooldown_min = COOLDOWN_MINUTES.get(error_class, 60)
    if cooldown_min == 0:
        return    # don't track no-cooldown error classes
    state = _read_state()
    cooldown_until = _now() + datetime.timedelta(minutes=cooldown_min)
    entry = state.get(source, {})
    consecutive = entry.get("consecutive_failures", 0) + 1
    # Exponential backoff for repeated failures of same source
    if consecutive > 1:
        cooldown_min *= 2 ** min(consecutive - 1, 4)    # cap at 16x
        cooldown_until = _now() + datetime.timedelta(minutes=cooldown_min)
    state[source] = {
        "last_error_class":     error_class,
        "last_error_message":   error_message[:500],
        "last_failed_ts":       _now().isoformat(timespec="seconds") + "Z",
        "cooldown_until_ts":    cooldown_until.isoformat(timespec="seconds") + "Z",
        "cooldown_minutes":     cooldown_min,
        "consecutive_failures": consecutive,
    }
    _write_state(state)
    logger.warning("source %s marked unhealthy (%s); cooldown %d min",
                    source, error_class, cooldown_min)


def mark_success(source: str) -> None:
    """Clear failure tracking for a source after a successful fetch."""
    state = _read_state()
    if source in state:
        del state[source]
        _write_state(state)


def is_healthy(source: str) -> tuple[bool, str | None]:
    """Returns (is_healthy, reason_if_not).

    Sources in cooldown return (False, '<error_class> until <ts>').
    Sources never failed return (True, None).
    Sources whose cooldown has expired return (True, None) AND the
    record is auto-cleared.
    """
    state = _read_state()
    entry = state.get(source)
    if not entry:
        return True, None
    try:
        cooldown_until = datetime.datetime.fromisoformat(
            entry["cooldown_until_ts"].rstrip("Z")
        )
    except (KeyError, ValueError):
        return True, None
    if _now() >= cooldown_until:
        # Cooldown expired — clear and return healthy
        del state[source]
        _write_state(state)
        return True, None
    return False, (
        f"in cooldown ({entry.get('last_error_class', 'unknown')} "
        f"until {entry['cooldown_until_ts']}, "
        f"{entry.get('consecutive_failures', 1)} consecutive failures)"
    )


def list_unhealthy() -> dict:
    """Return current unhealthy sources for monitoring."""
    state = _read_state()
    out = {}
    for src, entry in state.items():
        ok, reason = is_healthy(src)
        if not ok:
            out[src] = {**entry, "current_reason": reason}
    return out


def clear_all() -> None:
    """Reset all health tracking (operator override)."""
    if HEALTH_FILE.exists():
        HEALTH_FILE.unlink()
