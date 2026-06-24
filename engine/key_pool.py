"""
engine/key_pool.py
──────────────────
Gemini API Key Pool Manager with two-layer safety:

  Layer 1 — Validity Gate (enforced in history.py before AI call)
    Skip AI call entirely when both news context AND quant data are empty.
    Zero tokens consumed, record logged as "skipped_no_data".

  Layer 2 — Circuit Breaker (enforced here)
    • Consecutive quota errors  ≥ QUOTA_FAILS_BEFORE_SWITCH  → switch to next key
    • Consecutive empty outputs ≥ EMPTY_OUTPUT_LIMIT         → halt entire pool
      (distinguishes "real quota hit" from "something is broken")
    • All keys exhausted → raise AllKeysExhausted

Key pool is read from secrets.toml [GEMINI_POOL] section:
    [GEMINI_POOL]
    "Account1/Proj1" = "AIza..."
    "Account1/Proj2" = "AIza..."

Falls back to single GEMINI_KEY / GEMINI_API_KEY if no pool defined.
Runtime stats are persisted to .streamlit/key_pool_stats.json (survives restarts).
"""

from __future__ import annotations

import json
import logging
import os
import datetime
import threading
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

# ── Safety thresholds ─────────────────────────────────────────────────────────
QUOTA_FAILS_BEFORE_SWITCH = 3   # consecutive quota errors before rotating key
EMPTY_OUTPUT_LIMIT        = 5   # consecutive calls with no useful AI output before halt
RETRY_BASE_DELAY          = 15  # seconds to wait after first quota/RPM error before retrying same key

# ── Hard billing-protection limits (applied per-pool, not per-key) ────────────
# Cost-control guardrails for paid tier (billing enabled, 1K RPM / 10K RPD quota).
# RPM_HARD_LIMIT : max requests per 60-second sliding window across ALL keys combined
# RPD_HARD_LIMIT : max requests per Gemini quota day across ALL keys combined
# Set both to None to disable.
RPM_HARD_LIMIT: int = 60          # well below 1K paid-tier limit; single report won't trigger
RPD_HARD_LIMIT: int = 500         # daily cap for cost control; abnormal if exceeded in research use

STATS_FILE = Path(__file__).parent.parent / ".streamlit" / "key_pool_stats.json"


class AllKeysExhausted(Exception):
    """Raised when every key in the pool has been exhausted."""


class EmptyOutputCircuitBreaker(Exception):
    """Raised when too many consecutive calls produce no useful output."""


class BillingProtectionError(Exception):
    """Raised when a hard rate/daily limit is about to be breached to prevent charges."""


# ── Stats schema per key ──────────────────────────────────────────────────────
def _empty_stats(label: str) -> dict:
    return {
        "label":               label,
        "status":              "active",   # active | exhausted | halted
        "today_calls":         0,
        "today_errors":        0,
        "today_skips":         0,          # validity-gate skips (no data)
        "today_empty":         0,          # AI called but produced empty/stub output
        "consecutive_quota":   0,          # resets on success or key switch
        "consecutive_empty":   0,          # resets on any valid output
        "total_calls":         0,
        "total_errors":        0,
        "last_used":           None,
        "exhausted_at":        None,
        "anomaly_log":         [],         # list of {ts, event} dicts (last 20)
    }


class KeyPoolManager:
    """
    Thread-safe Gemini API key pool with circuit-breaker and validity gate support.

    Usage in backtest loop:
        pool = KeyPoolManager.from_secrets()
        model = pool.get_model()
        try:
            result = model.generate_content(prompt)
            pool.report_success(has_content=bool(result.text.strip()))
        except Exception as e:
            if pool.is_quota_error(e):
                pool.report_quota_error()   # may raise AllKeysExhausted
            else:
                raise
    """

    _lock = threading.Lock()

    def __init__(self, keys: dict[str, str]):
        """
        keys: {label: api_key_value}
        For Vertex AI mode, from_secrets() sets _vertex_mode=True after construction.
        """
        if not keys:
            raise ValueError("Key pool is empty — add at least one Gemini API key.")
        self._keys   = list(keys.items())   # [(label, key), ...]
        self._idx    = 0
        self._stats  = self._load_stats()
        self._today  = datetime.date.today().isoformat()

        # Vertex AI mode flags — set by from_secrets() when [VERTEX] config found
        self._vertex_mode:     bool = False
        self._vertex_project:  str  = ""
        self._vertex_location: str  = "us-central1"

        # ── Billing-protection rate limiter ──────────────────────────────────
        # Tracks timestamps of recent calls for RPM sliding window.
        # _rpd_count mirrors sum(today_calls) but is reset on PT quota day rollover.
        self._rpm_window: list[float] = []   # timestamps of last N calls (epoch seconds)
        self._rpd_count:  int         = sum(
            s.get("today_calls", 0) for s in self._stats.values()
        )
        self._reset_daily_counters_if_needed()

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_secrets(cls) -> "KeyPoolManager":
        """
        Build pool from Streamlit secrets.

        Priority:
          1. st.secrets["VERTEX"]  → Vertex AI mode (ADC, no API key needed)
          2. st.secrets["GEMINI_POOL"]  → AI Studio key pool
          3. st.secrets["GEMINI_KEY"] / ["GEMINI_API_KEY"]  → single AI Studio key

        Vertex AI mode: uses Application Default Credentials (ADC).
        Run `gcloud auth application-default login` once before starting the app.
        """
        # ── Priority 1: Vertex AI ────────────────────────────────────────────
        try:
            vertex_section = st.secrets.get("VERTEX")
            if vertex_section:
                project  = vertex_section.get("project", "")
                location = vertex_section.get("location", "us-central1")
                if project:
                    logger.info(
                        "KeyPoolManager: Vertex AI mode — project=%s location=%s",
                        project, location,
                    )
                    inst = cls({"vertex": f"__vertex__{project}__{location}"})
                    inst._vertex_project  = project
                    inst._vertex_location = location
                    inst._vertex_mode     = True
                    return inst
        except Exception as e:
            logger.warning("KeyPoolManager: failed to read VERTEX config: %s", e)

        # ── Priority 2: AI Studio key pool ───────────────────────────────────
        try:
            pool_section = st.secrets.get("GEMINI_POOL")
            if pool_section:
                keys = dict(pool_section)
                if keys:
                    logger.info("KeyPoolManager: loaded %d keys from GEMINI_POOL", len(keys))
                    return cls(keys)
        except Exception as e:
            logger.warning("KeyPoolManager: failed to read GEMINI_POOL: %s", e)

        # ── Priority 3: Single AI Studio key ─────────────────────────────────
        single = (
            st.secrets.get("GEMINI_API_KEY")
            or st.secrets.get("GEMINI_KEY", "")
        )
        if single:
            logger.info("KeyPoolManager: using single GEMINI_KEY fallback")
            return cls({"default": single})

        raise ValueError(
            "未找到 Gemini 认证配置。\n"
            "选项 A（推荐）：在 secrets.toml 添加 [VERTEX] 并运行 gcloud auth application-default login\n"
            "选项 B：在 secrets.toml 添加 GEMINI_KEY 或 [GEMINI_POOL]"
        )

    # ── Key access ───────────────────────────────────────────────────────────

    @property
    def current_label(self) -> str:
        return self._keys[self._idx][0]

    @property
    def current_key(self) -> str:
        return self._keys[self._idx][1]

    def get_model(self, model_name: str = "gemini-2.5-flash", response_schema=None):
        """
        Return a model wrapper.

        response_schema : optional JSON schema dict (or TypedDict / Pydantic class).
            When provided, the API call uses:
                response_mime_type = "application/json"
                response_schema    = <schema>
            This enforces structured JSON output regardless of which underlying
            model is used — the primary defence against fragile regex parsing.

        Vertex AI mode  : ADC, no API key.
        AI Studio mode  : API key from secrets.toml.
        """
        from google import genai as _genai
        from google.genai import types as _types

        _mname  = model_name
        _schema = response_schema

        def _make_config():
            if not _schema:
                return None
            return _types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_schema,
            )

        if self._vertex_mode:
            _project  = self._vertex_project
            _location = self._vertex_location

            class _ModelWrapper:
                def generate_content(self, prompt: str):
                    client = _genai.Client(
                        vertexai=True,
                        project=_project,
                        location=_location,
                    )
                    return client.models.generate_content(
                        model=_mname,
                        contents=prompt,
                        config=_make_config(),
                    )
        else:
            _key = self.current_key

            class _ModelWrapper:
                def generate_content(self, prompt: str):
                    client = _genai.Client(api_key=_key)
                    return client.models.generate_content(
                        model=_mname,
                        contents=prompt,
                        config=_make_config(),
                    )

        return _ModelWrapper()

    # ── Reporting ────────────────────────────────────────────────────────────

    def report_success(self, has_content: bool = True) -> None:
        """Call after a successful AI response."""
        with self._lock:
            s = self._get_stats(self.current_label)
            s["today_calls"]       += 1
            s["total_calls"]       += 1
            s["consecutive_quota"]  = 0
            s["last_used"]          = datetime.datetime.now().isoformat(timespec="seconds")
            if has_content:
                s["consecutive_empty"] = 0
            else:
                s["consecutive_empty"] += 1
                s["today_empty"]       += 1
                self._log_anomaly(s, "empty_output",
                                  f"AI returned empty/stub output "
                                  f"(consecutive={s['consecutive_empty']})")
                if s["consecutive_empty"] >= EMPTY_OUTPUT_LIMIT:
                    s["status"] = "halted"
                    self._log_anomaly(s, "circuit_breaker",
                                      f"Halted after {EMPTY_OUTPUT_LIMIT} consecutive empty outputs")
                    self._save_stats()
                    raise EmptyOutputCircuitBreaker(
                        f"Key '{self.current_label}' halted: "
                        f"{EMPTY_OUTPUT_LIMIT} consecutive calls produced no useful output. "
                        "Check data pipeline before resuming."
                    )
            self._save_stats()

    def report_quota_error(self) -> None:
        """
        Call when a quota / 429 / ResourceExhausted error is caught.

        Vertex AI mode: no key rotation (single GCP project). Records the error
        and raises AllKeysExhausted after QUOTA_FAILS_BEFORE_SWITCH consecutive
        failures so callers can handle gracefully (e.g. pause backtest).

        AI Studio mode: rotates to the next key in the pool.
        """
        with self._lock:
            s = self._get_stats(self.current_label)
            s["consecutive_quota"] += 1
            s["today_errors"]      += 1
            s["total_errors"]      += 1
            self._log_anomaly(s, "quota_error",
                              f"Quota error #{s['consecutive_quota']} on key '{self.current_label}'")

            if s["consecutive_quota"] >= QUOTA_FAILS_BEFORE_SWITCH:
                s["status"]       = "exhausted"
                s["exhausted_at"] = datetime.datetime.now().isoformat(timespec="seconds")
                self._log_anomaly(s, "key_exhausted",
                                  f"Key '{self.current_label}' marked exhausted after "
                                  f"{QUOTA_FAILS_BEFORE_SWITCH} consecutive quota errors")
                self._save_stats()
                if self._vertex_mode:
                    raise AllKeysExhausted(
                        f"Vertex AI 配额超限（连续 {QUOTA_FAILS_BEFORE_SWITCH} 次错误）。"
                        f"项目：{self._vertex_project}  区域：{self._vertex_location}\n"
                        "请检查 GCP 控制台的配额使用情况，或等待配额重置。"
                    )
                self._rotate_key()   # AI Studio: rotate to next key
            else:
                self._save_stats()

    def report_skip(self) -> None:
        """Call when validity gate fires (no news + no quant data)."""
        with self._lock:
            s = self._get_stats(self.current_label)
            s["today_skips"] += 1
            self._save_stats()

    def check_billing_limits(self) -> None:
        """
        Hard billing-protection gate. Call BEFORE every API request.
        Raises BillingProtectionError if either limit would be breached.

        RPM : sliding 60-second window across all keys combined.
        RPD : cumulative today_calls across all keys combined.

        Both limits are set conservatively below the free-tier ceiling so that
        a billing charge is structurally impossible as long as this gate is used.
        """
        import time as _time
        with self._lock:
            now = _time.monotonic()

            # ── RPD hard block ────────────────────────────────────────────────
            if RPD_HARD_LIMIT is not None and self._rpd_count >= RPD_HARD_LIMIT:
                raise BillingProtectionError(
                    f"每日请求上限已达 {self._rpd_count}/{RPD_HARD_LIMIT}（成本控制封锁）。"
                    "今日 API 调用已暂停，如需提高上限请修改 RPD_HARD_LIMIT。配额将在太平洋时间午夜后重置。"
                )

            # ── RPM sliding window ────────────────────────────────────────────
            if RPM_HARD_LIMIT is not None:
                # Evict timestamps older than 60 seconds
                self._rpm_window = [t for t in self._rpm_window if now - t < 60.0]
                if len(self._rpm_window) >= RPM_HARD_LIMIT:
                    oldest = self._rpm_window[0]
                    wait   = 60.0 - (now - oldest)
                    raise BillingProtectionError(
                        f"每分钟请求已达 {len(self._rpm_window)}/{RPM_HARD_LIMIT}（成本控制封锁）。"
                        f"请等待 {wait:.0f} 秒后重试。"
                    )

            # Record this request
            self._rpm_window.append(now)
            self._rpd_count += 1

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def is_quota_error(exc: Exception) -> bool:
        err = str(exc).lower()
        logger.warning("KeyPool raw error (full): %s", str(exc)[:300])
        return (
            "429"                in err
            or "quota"           in err
            or "resource_exhausted" in err
            or "rate_limit"      in err
            or "rateLimitExceeded" in str(exc)
        )

    def get_all_stats(self) -> list[dict]:
        """Return stats for all keys (for UI display), after syncing Gemini quota day."""
        with self._lock:
            self._reset_daily_counters_if_needed()
        result = []
        for label, key in self._keys:
            s   = dict(self._get_stats(label))
            idx = self._keys.index((label, key))
            s["is_current"] = (idx == self._idx)
            s["key_masked"] = key[:8] + "..." + key[-4:] if len(key) > 12 else "****"
            result.append(s)
        return result

    def force_switch_to(self, label: str) -> None:
        """Force the pool to use a specific key by label, regardless of its status."""
        with self._lock:
            for idx, (lbl, _) in enumerate(self._keys):
                if lbl == label:
                    self._idx = idx
                    logger.info("KeyPoolManager: force-switched to key '%s'", label)
                    return
        raise ValueError(f"Key label '{label}' not found in pool")

    def pool_summary(self) -> dict:
        stats = [self._get_stats(lbl) for lbl, _ in self._keys]
        return {
            "total":     len(self._keys),
            "active":    sum(1 for s in stats if s["status"] == "active"),
            "exhausted": sum(1 for s in stats if s["status"] == "exhausted"),
            "halted":    sum(1 for s in stats if s["status"] == "halted"),
            "today_calls": sum(s["today_calls"] for s in stats),
            "today_skips": sum(s["today_skips"] for s in stats),
            "today_errors": sum(s["today_errors"] for s in stats),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rotate_key(self) -> None:
        """Switch to the next non-exhausted key. Raises AllKeysExhausted if none."""
        start = self._idx
        for _ in range(len(self._keys)):
            self._idx = (self._idx + 1) % len(self._keys)
            label = self._keys[self._idx][0]
            if self._get_stats(label)["status"] == "active":
                logger.info("KeyPoolManager: rotated to key '%s'", label)
                return
            if self._idx == start:
                break
        raise AllKeysExhausted(
            "All Gemini API keys in the pool are exhausted or halted. "
            "Backtest stopped to protect remaining quota."
        )

    def _get_stats(self, label: str) -> dict:
        if label not in self._stats:
            self._stats[label] = _empty_stats(label)
        return self._stats[label]

    def _log_anomaly(self, stats_dict: dict, event: str, msg: str) -> None:
        log = stats_dict.setdefault("anomaly_log", [])
        log.append({
            "ts":    datetime.datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "msg":   msg,
        })
        # Keep only last 20 entries
        stats_dict["anomaly_log"] = log[-20:]
        logger.warning("KeyPool anomaly [%s] %s: %s", stats_dict["label"], event, msg)

    @staticmethod
    def _gemini_quota_date() -> str:
        """
        Gemini free-tier quota resets at midnight Pacific Time (UTC-8 / UTC-7 DST).
        We use the conservative fixed offset UTC-8 (no DST adjustment needed for
        quota purposes — being 1 hour early is fine, being late is not).
        Returns the current "Gemini day" as an ISO date string.
        """
        pt_now = datetime.datetime.utcnow() - datetime.timedelta(hours=8)
        return pt_now.date().isoformat()

    def _reset_daily_counters_if_needed(self) -> None:
        # Use Gemini's Pacific-Time quota day, not local calendar date.
        today = self._gemini_quota_date()
        for label, _ in self._keys:
            s = self._get_stats(label)

            # Determine the reference PT date for this key.
            # Priority: last_used (successful call) → exhausted_at (failed key that never succeeded)
            # Without this fallback, keys that were exhausted on first call (last_used=null)
            # would never be reset because the old condition `if _lu_pt and ...` short-circuits.
            ref_ts = s.get("last_used") or s.get("exhausted_at") or ""
            if ref_ts:
                try:
                    _ref_utc = datetime.datetime.fromisoformat(ref_ts)
                    _ref_pt  = (_ref_utc - datetime.timedelta(hours=8)).date().isoformat()
                except ValueError:
                    _ref_pt = ref_ts[:10]
            else:
                _ref_pt = ""

            # Reset daily counters if reference timestamp is from a previous Gemini quota day
            if _ref_pt and _ref_pt < today:
                s["today_calls"]  = 0
                s["today_errors"] = 0
                s["today_skips"]  = 0
                s["today_empty"]  = 0
                # Also restore exhausted keys — Gemini quota has refreshed
                if s["status"] == "exhausted":
                    s["status"]            = "active"
                    s["consecutive_quota"] = 0
                    s["exhausted_at"]      = None
                    self._log_anomaly(s, "daily_reset",
                                      f"Key restored to active — Gemini PT quota day rolled over ({today})")
        self._save_stats()

    def _load_stats(self) -> dict:
        try:
            if STATS_FILE.exists():
                return json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("KeyPoolManager: could not load stats file: %s", e)
        return {}

    def _save_stats(self) -> None:
        try:
            STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATS_FILE.write_text(
                json.dumps(self._stats, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("KeyPoolManager: could not save stats file: %s", e)


# ── Module-level singleton (shared across tabs in same Streamlit process) ─────
_pool_instance: Optional[KeyPoolManager] = None
_pool_lock = threading.Lock()


def get_pool() -> KeyPoolManager:
    """Return the module-level singleton KeyPoolManager, creating it if needed."""
    global _pool_instance
    with _pool_lock:
        if _pool_instance is None:
            _pool_instance = KeyPoolManager.from_secrets()
        return _pool_instance


def reset_pool() -> None:
    """Force re-initialise the pool (e.g. after adding new keys via UI)."""
    global _pool_instance
    with _pool_lock:
        _pool_instance = None
