"""
Circuit Breaker
===============
Three-level anomaly guard for the TradingCycleOrchestrator.

Level   Trigger                               Response
------  ------------------------------------  ----------------------------------------
LIGHT   Single data source failed             Log warning; caller falls back to backup
MEDIUM  LLM quota consumed > 80% today        Pause non-core LLM calls; keep risk patrol
SEVERE  VIX single-day spike > 30%            Halt all auto signal generation;
                                              write persistent flag; require manual reset

Design notes
------------
- LIGHT is stateless — it is returned by check_data_source() inline, never persisted.
- MEDIUM resets automatically when quota pressure drops below threshold on next check.
- SEVERE persists to .streamlit/circuit_breaker.json; requires explicit manual_reset().
- All checks are non-blocking: yfinance fetch has a 6-second timeout; quota check is
  a local dict read; any exception degrades gracefully to LEVEL_NONE.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LEVEL_NONE   = "none"
LEVEL_LIGHT  = "light"
LEVEL_MEDIUM = "medium"
LEVEL_SEVERE = "severe"

_LEVEL_RANK = {LEVEL_NONE: 0, LEVEL_LIGHT: 1, LEVEL_MEDIUM: 2, LEVEL_SEVERE: 3}

VIX_SPIKE_THRESHOLD  = 0.30   # 30% single-day VIX rise → SEVERE
QUOTA_MEDIUM_FRAC    = 0.80   # >80% RPD consumed → MEDIUM
_STATE_FILE = Path(__file__).parent / "state" / "circuit_breaker.json"
_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()


@dataclass
class CircuitBreakerState:
    level:         str                      # none | light | medium | severe
    reason:        str   = ""
    triggered_at:  Optional[str] = None    # ISO datetime string
    auto_reset:    bool  = True            # SEVERE=False; requires manual reset
    vix_today:     Optional[float] = None
    vix_prev:      Optional[float] = None
    quota_frac:    Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self.level != LEVEL_NONE

    @property
    def rank(self) -> int:
        return _LEVEL_RANK.get(self.level, 0)


# ── Persistent SEVERE state ────────────────────────────────────────────────────

def _load_persistent() -> Optional[CircuitBreakerState]:
    try:
        if _STATE_FILE.exists():
            data = json.loads(_STATE_FILE.read_text())
            if data.get("level") == LEVEL_SEVERE:
                return CircuitBreakerState(**data)
    except Exception:
        pass
    return None


def _save_persistent(state: CircuitBreakerState) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.warning("CircuitBreaker: failed to persist state: %s", exc)


def _clear_persistent() -> None:
    try:
        if _STATE_FILE.exists():
            _STATE_FILE.unlink()
    except Exception:
        pass


# ── Individual checks ──────────────────────────────────────────────────────────

def check_vix_spike(as_of: Optional[datetime.date] = None) -> CircuitBreakerState:
    """
    Return SEVERE if VIX rose > 30% in a single day; NONE otherwise.
    Fetches last 5 trading days of ^VIX via yfinance (6s timeout).
    """
    try:
        import yfinance as yf
        _end   = (as_of or datetime.date.today()) + datetime.timedelta(days=1)
        _start = _end - datetime.timedelta(days=10)
        vix_df = yf.download("^VIX", start=str(_start), end=str(_end),
                             progress=False, auto_adjust=True, timeout=6)
        if vix_df.empty or len(vix_df) < 2:
            return CircuitBreakerState(level=LEVEL_NONE)
        closes = vix_df["Close"]
        # yfinance auto_adjust=True with single ticker can return a 1-col
        # DataFrame; squeeze to Series so .iloc[-1] returns a scalar (else
        # pandas FutureWarning about float(single-element Series))
        if hasattr(closes, "columns"):
            closes = closes.iloc[:, 0]
        closes = closes.dropna()
        if len(closes) < 2:
            return CircuitBreakerState(level=LEVEL_NONE)
        v_today = float(closes.iloc[-1])
        v_prev  = float(closes.iloc[-2])
        if v_prev > 0:
            spike = (v_today - v_prev) / v_prev
            if spike > VIX_SPIKE_THRESHOLD:
                state = CircuitBreakerState(
                    level=LEVEL_SEVERE,
                    reason=f"VIX 单日涨幅 {spike:.0%}（{v_prev:.1f}→{v_today:.1f}），超过 {VIX_SPIKE_THRESHOLD:.0%} 阈值",
                    triggered_at=datetime.datetime.utcnow().isoformat(timespec="seconds"),
                    auto_reset=False,
                    vix_today=round(v_today, 2),
                    vix_prev=round(v_prev, 2),
                )
                return state
    except Exception as exc:
        logger.debug("CircuitBreaker VIX check failed: %s", exc)
    return CircuitBreakerState(level=LEVEL_NONE)


def check_quota_pressure() -> CircuitBreakerState:
    """
    Return MEDIUM if today's API calls consumed > 80% of RPD_HARD_LIMIT; NONE otherwise.
    """
    try:
        from engine.key_pool import get_pool, RPD_HARD_LIMIT
        if not RPD_HARD_LIMIT:
            return CircuitBreakerState(level=LEVEL_NONE)
        summary = get_pool().pool_summary()
        today_calls = summary.get("today_calls", 0)
        frac = today_calls / RPD_HARD_LIMIT
        if frac > QUOTA_MEDIUM_FRAC:
            return CircuitBreakerState(
                level=LEVEL_MEDIUM,
                reason=(
                    f"LLM 配额已消耗 {frac:.0%}（{today_calls}/{RPD_HARD_LIMIT} RPD），"
                    f"余量 < {(1 - QUOTA_MEDIUM_FRAC):.0%}；暂停非核心 LLM 调用"
                ),
                triggered_at=datetime.datetime.utcnow().isoformat(timespec="seconds"),
                auto_reset=True,
                quota_frac=round(frac, 3),
            )
    except Exception as exc:
        logger.debug("CircuitBreaker quota check failed: %s", exc)
    return CircuitBreakerState(level=LEVEL_NONE)


def check_data_source(name: str, exc: Exception) -> CircuitBreakerState:
    """
    Inline LIGHT-level check: call after catching a data source exception.
    Always returns LIGHT (caller decides whether to log / fallback).
    """
    return CircuitBreakerState(
        level=LEVEL_LIGHT,
        reason=f"数据源 [{name}] 不可用：{exc}",
        triggered_at=datetime.datetime.utcnow().isoformat(timespec="seconds"),
        auto_reset=True,
    )


# ── Master evaluate ───────────────────────────────────────────────────────────

def evaluate(as_of: Optional[datetime.date] = None) -> CircuitBreakerState:
    """
    Run all checks and return the highest-severity active state.

    Priority: persistent SEVERE > new SEVERE (VIX) > MEDIUM (quota) > NONE.
    LIGHT is not included here (it is checked inline per data source call).
    """
    with _lock:
        # 1. Persistent SEVERE from previous session (requires manual_reset())
        persisted = _load_persistent()
        if persisted and persisted.level == LEVEL_SEVERE:
            return persisted

        # 2. Live VIX spike check
        vix_state = check_vix_spike(as_of)
        if vix_state.level == LEVEL_SEVERE:
            _save_persistent(vix_state)
            logger.error(
                "CIRCUIT BREAKER SEVERE: %s — 所有自动信号生成已暂停，等待人工恢复",
                vix_state.reason,
            )
            return vix_state

        # 3. Quota pressure check
        quota_state = check_quota_pressure()
        if quota_state.level == LEVEL_MEDIUM:
            logger.warning("CIRCUIT BREAKER MEDIUM: %s", quota_state.reason)
            return quota_state

        return CircuitBreakerState(level=LEVEL_NONE)


def set_external_halt_flag(reason: str, source: str = "ops_watchdog") -> CircuitBreakerState:
    """
    Public API for external SEVERE halt-flag trigger (added 2026-05-13 for
    Ops Watchdog Agent spec id=63 §2.6). Until now external callers had to
    use the `_save_persistent` private function; this is the supported entry.

    Args:
        reason:  human-readable explanation (kept ≤400 chars, prefixed with
                 `<source>:` so meta-monitors can distinguish trigger sources)
        source:  trigger source identifier (default 'ops_watchdog' matches
                 the Watchdog notifications dispatcher; future sources can
                 add their own prefix)

    Returns:
        CircuitBreakerState that was persisted (level=SEVERE, auto_reset=False).

    INVARIANT (spec id=63 §六): only SEVERE severity can SET; only human can
    CLEAR via `manual_reset()` (called from Streamlit dashboard
    'Acknowledge Halt' button at pages/ops_watchdog.py). External code MUST
    NOT call manual_reset.
    """
    with _lock:
        prefixed = f"{source}: {reason}"[:400] if not reason.startswith(
            f"{source}:"
        ) else reason[:400]
        state = CircuitBreakerState(
            level        = LEVEL_SEVERE,
            reason       = prefixed,
            triggered_at = datetime.datetime.utcnow().isoformat() + "Z",
            auto_reset   = False,
        )
        _save_persistent(state)
        logger.error("CircuitBreaker SEVERE set externally (source=%s): %s",
                     source, reason)
        return state


def manual_reset(reason: str = "") -> None:
    """
    Clear a SEVERE circuit breaker state. Must be called explicitly from Admin UI.
    Logs the reset event for audit trail.
    """
    with _lock:
        _clear_persistent()
        logger.warning(
            "CircuitBreaker SEVERE manually reset. Reason: %s",
            reason or "(no reason provided)",
        )


def get_status() -> CircuitBreakerState:
    """Return current state without triggering new checks (for UI display)."""
    persisted = _load_persistent()
    if persisted and persisted.level == LEVEL_SEVERE:
        return persisted
    return check_quota_pressure()
