"""
engine/universe_singlename/wrds_retry.py — WRDS query retry decorator.

Wraps WRDS-touching functions with exponential-backoff retry to survive:
  - Transient PostgreSQL connection drops (psycopg2.OperationalError)
  - WRDS server-side rate limits / connection resets
  - Network blips during long walk-forward runs

Built 2026-05-11 alongside checkpoint module for Wave B publishable verdict
run resilience (per session findings memory §Finding 5).
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)


# Default retry policy — conservative; tunable per-call via decorator args
DEFAULT_MAX_ATTEMPTS:  int   = 3
DEFAULT_BASE_DELAY:    float = 5.0    # seconds; 5 → 15 → 45 backoff
DEFAULT_BACKOFF_MULT:  float = 3.0


def _is_retryable_exception(exc: Exception) -> bool:
    """Identify transient errors worth retrying.

    Retries:
      - psycopg2.OperationalError (connection dropped, server gone away)
      - psycopg2.errors.AdminShutdown (WRDS maintenance)
      - psycopg2.InterfaceError (connection-state corrupt)
      - ConnectionError / TimeoutError (Python stdlib)
      - **RuntimeError("WRDS connection failed: EOF...")** —
        wrapped by `crsp_loader._open_wrds_connection` when wrds.Connection()
        prompts for username/password in non-interactive context due to
        pgpass parse failure or transient credentials lookup. 2026-05-11
        fix: extend retryable to wrapped pattern.

    Does NOT retry:
      - psycopg2.ProgrammingError (bad SQL — won't fix itself)
      - DataError (bad input — same)
      - KeyError / ValueError (caller bug)
      - RuntimeError("WRDS not configured") — fatal config error
    """
    # Lazy import — psycopg2 only available if wrds is installed
    try:
        import psycopg2
        if isinstance(exc, psycopg2.OperationalError):
            return True
        if isinstance(exc, psycopg2.InterfaceError):
            return True
        # AdminShutdown is in psycopg2.errors subclass tree
        if "AdminShutdown" in type(exc).__name__:
            return True
    except ImportError:
        pass

    if isinstance(exc, (ConnectionError, TimeoutError, EOFError)):
        return True

    # Specific WRDS lib errors
    exc_name = type(exc).__name__
    if "ConnectionClosed" in exc_name or "ConnectionRefused" in exc_name:
        return True

    # Wrapped RuntimeError from crsp_loader._open_wrds_connection — retry only
    # if message indicates transient connection issue, NOT fatal config error.
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        # Transient (worth retry): EOF on stdin (wrds lib lost pgpass mid-call)
        # or "connection failed" / "could not connect"
        retryable_patterns = (
            "EOF when reading",          # wrds prompted in non-interactive
            "connection failed",          # wrapped connect error
            "could not connect",          # libpq connect error
            "server closed the connection",
            "Connection terminated",
        )
        if any(p in msg for p in retryable_patterns):
            return True
        # Specifically NOT retryable: pre-config check failure
        if "WRDS not configured" in msg:
            return False

    return False


def with_wrds_retry(
    max_attempts: int   = DEFAULT_MAX_ATTEMPTS,
    base_delay:   float = DEFAULT_BASE_DELAY,
    backoff_mult: float = DEFAULT_BACKOFF_MULT,
) -> Callable:
    """Decorator: retry decorated function on transient WRDS errors.

    Usage:
        @with_wrds_retry(max_attempts=3, base_delay=5)
        def _real_crsp_panel(...):
            ...

    Backoff: attempt 1 fails → wait `base_delay` → attempt 2 fails → wait
    `base_delay × backoff_mult` → attempt 3 fails → wait
    `base_delay × backoff_mult^2` → attempt 4 NOT taken (max=3) → raise last
    exception.

    Note: this wraps the entire function call. Inner connection state is
    NOT retried mid-call; the caller's `_open_wrds_connection()` will be
    invoked fresh on each retry attempt.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if not _is_retryable_exception(exc):
                        # Non-transient — raise immediately (don't waste time)
                        raise
                    last_exc = exc
                    if attempt >= max_attempts:
                        logger.error(
                            "with_wrds_retry: %s exhausted %d attempts; last error: %s",
                            fn.__name__, max_attempts, exc,
                        )
                        raise
                    delay = base_delay * (backoff_mult ** (attempt - 1))
                    logger.warning(
                        "with_wrds_retry: %s attempt %d/%d failed (%s); "
                        "backing off %.0fs",
                        fn.__name__, attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
            # Unreachable, but defensive
            if last_exc:
                raise last_exc
            raise RuntimeError("with_wrds_retry: no attempts made")
        return wrapper
    return decorator
