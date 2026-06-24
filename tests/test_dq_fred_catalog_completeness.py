"""tests/test_dq_fred_catalog_completeness.py — preventive catalog audit.

Failure mode this test guards against: an engine.* module starts
fetching a new FRED series_id but the developer forgets to add it to
engine.agents.dq_inspector.thresholds.FRED_MAX_STALENESS_BDAYS.
Without this test, the new series silently goes un-monitored — exactly
the kind of audit miss that triggered the 2026-05-19 senior re-review
(9 missing series surfaced by manual grep).

How it works:
  1. Static-grep ALL `*.py` files under engine/ for two FRED API call
     patterns: `_fetch_observations("SID", ...)` and
     `_fetch_fred("SID", ...)`. Extract every literal SID string.
  2. Filter out known-dead series (NAPM — FRED returns empty via free
     API; engine.macro_fetcher:419 should be retired separately).
  3. Assert every extracted series is in FRED_MAX_STALENESS_BDAYS.

If a consumer adds a new series, this test fails until the catalog
is updated. If a consumer is removed/dead, the catalog can shrink.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from engine.agents.dq_inspector.thresholds import FRED_MAX_STALENESS_BDAYS


# Series that engine.* code references but FRED no longer publishes
# data for via free API; adding them to FRED_MAX_STALENESS_BDAYS would
# generate chronic false-positive alerts. Triage list — periodically
# revisit to remove the dead consumer.
_KNOWN_DEAD_OR_RESTRICTED: frozenset[str] = frozenset({
    "NAPM",   # engine.macro_fetcher:419; FRED returns no data via free API
})


# Common UPPERCASE_WITH_UNDERSCORES tokens that look like FRED series
# IDs to a static grep but are actually Python module constants
# (configuration keys / URL bases / variable names). Filtered out
# before catalog comparison.
_REGEX_FALSE_POSITIVES: frozenset[str] = frozenset({
    "FRED_API_KEY", "FRED_BASE", "FRED_DEFAULT_FALLBACK_BDAYS",
    "FRED_MAX_STALENESS_BDAYS",
    "MODE_7_CAP_BY_TICKER_CLASS", "SLEEVE_CLASS_INTRA_CAPS",
    "BOOK_SINGLE_TICKER_ABS_CAP",
})

# Series consumed by backtest-only modules (not in production daily
# critical path). DQ Inspector scope is production daily; these are
# acceptable to leave outside the catalog as long as we're explicit.
_BACKTEST_ONLY_CONSUMERS: frozenset[str] = frozenset({
    "BAMLH0A0HYM2",   # engine.data_snapshot — high yield spread, used
                       # by scripts/freeze_backtest_data.py NOT daily
                       # production path
})


# Two patterns extract a FRED series_id literal from engine source:
#   _fetch_observations("CPIAUCSL", ...)
#   _fetch_fred("DGS10", ...)
# Series IDs are uppercase + digits + occasional underscores (matched
# generously; filter to ALL-CAPS to avoid false positives like the
# string "Fed Funds" inside a label).
_FRED_CALL_PATTERN = re.compile(
    r'(?:_fetch_observations|_fetch_fred)\s*\(\s*["\']([A-Z][A-Z0-9_]{2,})["\']'
)

# Series referenced as tuples inside _YC_TENORS / _POLICY_SERIES style
# constants — they look like `("DGS10", "10Y", 10),`. These are also
# real consumers (downstream code iterates the constant + calls
# _fetch_observations on each id).
_FRED_TUPLE_PATTERN = re.compile(
    r'\(\s*["\']([A-Z][A-Z0-9_]{4,})["\']\s*,'
)


def _grep_engine_for_fred_series_ids() -> set[str]:
    """Walk engine/**/*.py, extract every FRED series_id literal."""
    engine_root = Path(__file__).resolve().parent.parent / "engine"
    found: set[str] = set()
    for path in engine_root.rglob("*.py"):
        # Skip archived / dead-code directories
        if "_archive" in path.parts or "deadcode" in path.name:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in (_FRED_CALL_PATTERN, _FRED_TUPLE_PATTERN):
            for m in pattern.finditer(text):
                sid = m.group(1)
                # Tuple pattern is greedy — filter to plausible FRED IDs:
                # uppercase + digits + underscores, 3-15 chars, must
                # appear NEAR a _fetch_* call (proximity scoped to module).
                if 3 <= len(sid) <= 15:
                    found.add(sid)
    return found


def _filter_to_fred_series_consumers() -> set[str]:
    """Static-scan result intersected with files that actually call
    `_fetch_observations` or `_fetch_fred`. This kills false positives
    from tuple-pattern (e.g. "SCRIPT_TIMEOUT" string constants)."""
    engine_root = Path(__file__).resolve().parent.parent / "engine"
    fred_consumer_paths: list[Path] = []
    for path in engine_root.rglob("*.py"):
        if "_archive" in path.parts or "deadcode" in path.name:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "_fetch_observations" in text or "_fetch_fred" in text:
            fred_consumer_paths.append(path)

    series_ids: set[str] = set()
    for path in fred_consumer_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in (_FRED_CALL_PATTERN, _FRED_TUPLE_PATTERN):
            for m in pattern.finditer(text):
                sid = m.group(1)
                if 3 <= len(sid) <= 15:
                    series_ids.add(sid)
    return series_ids


def test_every_consumed_fred_series_is_in_catalog():
    """Every FRED series_id literal referenced by an engine.* module
    that also calls _fetch_observations / _fetch_fred must appear in
    FRED_MAX_STALENESS_BDAYS, unless explicitly whitelisted as
    dead/restricted or backtest-only.
    """
    consumed = _filter_to_fred_series_consumers() - _REGEX_FALSE_POSITIVES
    catalog = set(FRED_MAX_STALENESS_BDAYS)
    whitelisted = _KNOWN_DEAD_OR_RESTRICTED | _BACKTEST_ONLY_CONSUMERS

    missing = consumed - catalog - whitelisted
    assert not missing, (
        f"FRED_MAX_STALENESS_BDAYS missing {len(missing)} consumed series: "
        f"{sorted(missing)!r}. Add each to thresholds.py with an "
        f"appropriate staleness threshold OR whitelist it in this test "
        f"as dead/restricted or backtest-only."
    )


def test_no_dead_series_in_catalog():
    """Series in _KNOWN_DEAD_OR_RESTRICTED must NOT be in the catalog;
    they'd generate chronic false-positive alerts on every daily run."""
    catalog = set(FRED_MAX_STALENESS_BDAYS)
    dead_in_catalog = _KNOWN_DEAD_OR_RESTRICTED & catalog
    assert not dead_in_catalog, (
        f"Dead/restricted series found in catalog: {sorted(dead_in_catalog)!r}. "
        f"Either restore upstream data dependency or remove from catalog."
    )


def test_catalog_thresholds_in_reasonable_range():
    """Sanity: daily series ≤ 5bd, monthly series ≤ 70bd. Catches
    accidental threshold inflation that would mask real staleness."""
    for sid, threshold in FRED_MAX_STALENESS_BDAYS.items():
        # Heuristic: daily series IDs start with 'D' (DGS*, DFEDTAR*, DFF)
        # or are short rate IDs (SOFR, T*Y2Y, T*YIE, BAMLC*).
        is_daily = (
            sid.startswith("D")
            or sid in {"SOFR"}
            or sid.startswith("T") and ("Y" in sid)
            or sid.startswith("BAMLC")
        )
        if is_daily:
            assert threshold <= 5, (
                f"Daily series {sid!r} has threshold {threshold}bd > 5bd; "
                f"either reclassify as monthly or investigate why daily "
                f"data is so stale."
            )
        else:
            assert threshold <= 70, (
                f"Monthly/quarterly series {sid!r} has threshold {threshold}bd "
                f"> 70bd; FRED publish cycles rarely exceed 65bd."
            )
