"""engine/data/orchestrator.py — Layer 2: intelligent data orchestration.

Bridges required_data tokens → DataFrame data_kwargs. Handles 5 scenarios
autonomously:

  1. Primary source + cache hit  → return cache; record source
  2. Primary source + cache miss → fetch primary; write cache; return
  3. Primary unavailable (auth missing / network / rate-limited)
     → fall through to next fetcher in chain; flag downgrade
  4. NO source produced data    → structured AcquisitionResult with
                                     success=False; engine MUST handle
                                     (NEVER silent substitute synth data)
  5. Partial date coverage      → returned with partial_coverage=True;
                                     protocol designer can downgrade

Doctrine:
- AUTONOMOUS source selection: prefer paid (higher quality), skip if
  auth missing without raising; try next fetcher.
- NO SILENT DOWNGRADE: every source decision logged to acquisition_log.jsonl
- NO SILENT FAILURE: failure surfaces to caller with structured detail.
- NEVER substitute synthetic data for missing real data.

Flexibility ↔ Rigor balance:
- FLEX: autoselect across paid/free/scraped without user configuration
- RIGOR: source_used + quality_caveats propagate to protocol designer
  (which can downgrade bars when serving fallback data)
"""
from __future__ import annotations

import dataclasses
import datetime
import importlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from engine.data import cache_manager as cache

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "data" / "research" / "data_inventory.yaml"
ACQUISITION_LOG = REPO_ROOT / "data" / "research" / "data_acquisition_log.jsonl"


# ── Result types ────────────────────────────────────────────────────────

@dataclasses.dataclass
class ProbeResult:
    """Lightweight availability check — returned by each fetcher's probe()."""
    available:        bool
    error:            str | None
    error_class:      str | None       # access_denied | auth_missing | network | schema_unknown | rate_limited
    elapsed_secs:     float
    estimated_rows:   int | None = None


@dataclasses.dataclass
class FetcherAttempt:
    """Records ONE fetcher attempt (probe + fetch combined)."""
    source:        str
    tier:          str          # paid | free | scraped
    success:       bool
    error:         str | None
    rows:          int          # 0 if failed
    elapsed_secs:  float
    probe_only:    bool = False         # True if probe failed and fetch skipped
    error_class:   str | None = None    # propagated from probe


@dataclasses.dataclass
class AcquisitionResult:
    """Structured outcome of fetching ONE token.

    PIT (Point-In-Time) fields support senior-quant data integrity
    per project_senior_quant_data_pitfalls_2026-05-30:
      pit_vintage_date: as-of date the data represents (None for non-PIT-aware
                         sources like FRED daily macro)
      latest_data_date: latest date in returned df (for cross-token alignment;
                         e.g. Compustat ~90d lag vs CRSP daily)
      universe_as_of:   for universe-filtered data, the snapshot date used
                         (None for universe-free queries)
    """
    token:             str
    success:           bool
    df:                pd.DataFrame | None
    source_used:       str | None      # which fetcher served
    source_tier:       str | None      # paid | free | scraped
    quality_caveats:   list[str]
    partial_coverage:  bool             # did we get full requested range?
    coverage_start:    str | None
    coverage_end:      str | None
    attempts:          list[FetcherAttempt]
    cache_hit:         bool
    pit_vintage_date:  str | None = None
    latest_data_date:  str | None = None
    universe_as_of:    str | None = None

    def to_dict(self) -> dict:
        return {
            "token":             self.token,
            "success":           self.success,
            "source_used":       self.source_used,
            "source_tier":       self.source_tier,
            "quality_caveats":   self.quality_caveats,
            "partial_coverage":  self.partial_coverage,
            "coverage_start":    self.coverage_start,
            "coverage_end":      self.coverage_end,
            "cache_hit":         self.cache_hit,
            "pit_vintage_date":  self.pit_vintage_date,
            "latest_data_date":  self.latest_data_date,
            "universe_as_of":    self.universe_as_of,
            "n_rows":            len(self.df) if self.df is not None else 0,
            "n_attempts":        len(self.attempts),
            "attempt_summary":   [
                {"source": a.source, "tier": a.tier, "success": a.success,
                  "error": a.error, "rows": a.rows}
                for a in self.attempts
            ],
        }


# ── Inventory loading ───────────────────────────────────────────────────

def _load_inventory() -> dict:
    if not INVENTORY_PATH.exists():
        return {"inventory": {}}
    return yaml.safe_load(INVENTORY_PATH.read_text(encoding="utf-8")) or {"inventory": {}}


def list_tokens() -> list[str]:
    return sorted(_load_inventory().get("inventory", {}).keys())


def get_token_spec(token: str) -> dict | None:
    return _load_inventory().get("inventory", {}).get(token)


# ── Auth check (autonomous source decision) ─────────────────────────────

def _auth_available(auth_key: str | None) -> bool:
    """Check if an auth credential is configured. Used to autoselect among
    fetchers — skip paid sources without configured auth WITHOUT raising."""
    if not auth_key:
        return True    # no auth required
    try:
        import streamlit as st
        return bool(st.secrets.get(auth_key))
    except Exception:
        pass
    import os
    return bool(os.environ.get(auth_key.upper()))


# ── Fetcher dispatch ────────────────────────────────────────────────────

def _call_fetcher(source: str, function: str,
                    start: str, end: str, **kw) -> pd.DataFrame:
    """Dynamically import and call a fetcher function."""
    module_name = f"engine.data.fetchers.{source}"
    mod = importlib.import_module(module_name)
    fn = getattr(mod, function)
    return fn(start=start, end=end, **kw)


def _call_probe(source: str, function: str,
                  start: str, end: str, **kw) -> ProbeResult:
    """Call a fetcher's probe() function. Returns ProbeResult."""
    module_name = f"engine.data.fetchers.{source}"
    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        return ProbeResult(
            available=False, error=f"module not implemented: {exc}",
            error_class="schema_unknown", elapsed_secs=time.time() - t0,
        )
    probe_fn = getattr(mod, "probe", None)
    if probe_fn is None:
        # No probe → conservatively assume available (legacy fetchers)
        return ProbeResult(
            available=True, error=None, error_class=None,
            elapsed_secs=time.time() - t0,
        )
    fetch_target = function    # parameterize probe with target function name
    try:
        result = probe_fn(start=start, end=end,
                            target_function=fetch_target, **kw)
        if isinstance(result, ProbeResult):
            return result
        # Fetcher returned plain bool — convert
        if isinstance(result, bool):
            return ProbeResult(
                available=result, error=None, error_class=None,
                elapsed_secs=time.time() - t0,
            )
        return ProbeResult(
            available=True, error=None, error_class=None,
            elapsed_secs=time.time() - t0,
        )
    except Exception as exc:
        return ProbeResult(
            available=False, error=f"{type(exc).__name__}: {exc}",
            error_class="network", elapsed_secs=time.time() - t0,
        )


def _try_fetcher(spec: dict, start: str, end: str,
                   *, skip_probe: bool = False, **kw):
    """Try one fetcher in the chain. PROBE-FIRST.

    Returns FetcherAttempt on failure; (FetcherAttempt, DataFrame) on success.
    """
    source = spec["source"]
    function = spec["function"]
    tier = spec.get("tier", "unknown")
    auth = spec.get("auth")

    # Auth-check (no need to probe if auth missing — known failure)
    if not _auth_available(auth):
        return FetcherAttempt(
            source=source, tier=tier, success=False,
            error=f"auth credential {auth!r} not configured",
            error_class="auth_missing",
            rows=0, elapsed_secs=0.0, probe_only=True,
        )

    # Probe (unless explicitly skipped — e.g. test mode)
    if not skip_probe:
        probe = _call_probe(source, function, start, end, **kw)
        if not probe.available:
            return FetcherAttempt(
                source=source, tier=tier, success=False,
                error=f"probe failed: {probe.error}",
                error_class=probe.error_class,
                rows=0, elapsed_secs=probe.elapsed_secs, probe_only=True,
            )

    # Probe OK → call fetch
    t0 = time.time()
    try:
        df = _call_fetcher(source, function, start=start, end=end, **kw)
    except Exception as exc:
        return FetcherAttempt(
            source=source, tier=tier, success=False,
            error=f"{type(exc).__name__}: {exc}",
            rows=0, elapsed_secs=time.time() - t0,
        )
    elapsed = time.time() - t0

    if df is None or (hasattr(df, "empty") and df.empty):
        return FetcherAttempt(
            source=source, tier=tier, success=False,
            error="fetcher returned empty DataFrame",
            rows=0, elapsed_secs=elapsed,
        )

    return (
        FetcherAttempt(source=source, tier=tier, success=True, error=None,
                        rows=len(df), elapsed_secs=elapsed),
        df,
    )


# ── Audit log ───────────────────────────────────────────────────────────

def _append_log(result: AcquisitionResult, start: str, end: str, kw: dict) -> None:
    ACQUISITION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts":               datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "token":            result.token,
        "start":            start,
        "end":              end,
        "kw":               kw,
        **result.to_dict(),
    }
    with ACQUISITION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ── Coverage check ──────────────────────────────────────────────────────

def _coverage(df: pd.DataFrame, requested_start: str,
                requested_end: str) -> tuple[str | None, str | None, bool]:
    """Check whether df's date column covers the requested range fully.

    Returns (actual_start, actual_end, partial_coverage_bool).
    """
    date_col = None
    for cand in ("date", "datadate", "filing_date", "report_date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        return None, None, False
    actual_start = str(pd.to_datetime(df[date_col]).min().date())
    actual_end = str(pd.to_datetime(df[date_col]).max().date())
    partial = (actual_start > requested_start) or (actual_end < requested_end)
    return actual_start, actual_end, partial


# ── Public API ──────────────────────────────────────────────────────────

def fetch_token(
    token: str,
    *,
    start: str,
    end: str,
    use_cache: bool = True,
    max_cache_age_days: float | None = 30.0,
    log: bool = True,
    **kw,
) -> AcquisitionResult:
    """Fetch one token from its fetcher chain.

    Scenarios handled:
      1-2. Cache hit / cache miss + first fetcher works
      3.   First fetcher unavailable → try next (downgrade flagged)
      4.   ALL fetchers failed → success=False with attempt detail
      5.   Partial coverage → success=True with partial_coverage=True

    Args:
      token:               inventory token (e.g. crsp_dsf)
      start, end:          YYYY-MM-DD range
      use_cache:           consult cache first
      max_cache_age_days:  cache TTL; None = no expiry
      log:                 if True, append to data_acquisition_log.jsonl
      **kw:                additional query params (e.g. universe, fred_series_id)
    """
    spec = get_token_spec(token)
    if not spec:
        result = AcquisitionResult(
            token=token, success=False, df=None,
            source_used=None, source_tier=None,
            quality_caveats=[f"token {token!r} not in data_inventory.yaml"],
            partial_coverage=False, coverage_start=None, coverage_end=None,
            attempts=[], cache_hit=False,
        )
        if log:
            _append_log(result, start, end, kw)
        return result

    # 1. Cache check
    if use_cache:
        df, meta = cache.get(token, start, end,
                                max_age_days=max_cache_age_days, **kw)
        if df is not None:
            actual_s, actual_e, partial = _coverage(df, start, end)
            result = AcquisitionResult(
                token=token, success=True, df=df,
                source_used=meta.get("source"), source_tier=meta.get("source_tier"),
                quality_caveats=[],
                partial_coverage=partial,
                coverage_start=actual_s, coverage_end=actual_e,
                attempts=[], cache_hit=True,
            )
            if log:
                _append_log(result, start, end, kw)
            return result

    # 2-3. Fetcher chain
    attempts: list[FetcherAttempt] = []
    df_result: pd.DataFrame | None = None
    chain = spec.get("fetcher_chain") or []
    served_spec: dict | None = None

    for fetcher_spec in chain:
        attempt = _try_fetcher(fetcher_spec, start, end, **kw)
        if isinstance(attempt, tuple):
            attempts.append(attempt[0])
            df_result = attempt[1]
            served_spec = fetcher_spec
            break
        attempts.append(attempt)
        logger.info(
            "token %s fetcher %s failed (%s); trying next",
            token, fetcher_spec.get("source"), attempt.error,
        )

    # 4. All failed
    if df_result is None:
        result = AcquisitionResult(
            token=token, success=False, df=None,
            source_used=None, source_tier=None,
            quality_caveats=[
                f"all {len(attempts)} fetchers failed for token {token!r}"
            ],
            partial_coverage=False, coverage_start=None, coverage_end=None,
            attempts=attempts, cache_hit=False,
        )
        if log:
            _append_log(result, start, end, kw)
        return result

    # 5. Success — write cache + check coverage
    cache.put(
        token, start, end, df_result,
        source=served_spec["source"], source_tier=served_spec.get("tier", "unknown"),
        wallclock_seconds=attempts[-1].elapsed_secs,
        **kw,
    )
    actual_s, actual_e, partial = _coverage(df_result, start, end)

    quality_caveats = []
    if "quality_caveat" in served_spec:
        quality_caveats.append(served_spec["quality_caveat"])
    if served_spec.get("tier") != "paid" and any(
        s.get("tier") == "paid" for s in chain
    ):
        quality_caveats.append(
            f"served by {served_spec['source']!r} (tier={served_spec.get('tier')}); "
            f"paid source(s) unavailable — protocol designer should consider stricter bars"
        )

    # PIT fields: orchestrator at v1 forwards what the fetcher provided
    # via spec (pit_vintage_date in inventory). Phase 6b will have real
    # fetchers populate dynamically.
    pit_vintage = served_spec.get("pit_vintage_date")
    universe_as_of = kw.get("as_of_date") or kw.get("universe_as_of")

    result = AcquisitionResult(
        token=token, success=True, df=df_result,
        source_used=served_spec["source"],
        source_tier=served_spec.get("tier", "unknown"),
        quality_caveats=quality_caveats,
        partial_coverage=partial,
        coverage_start=actual_s, coverage_end=actual_e,
        attempts=attempts, cache_hit=False,
        pit_vintage_date=pit_vintage,
        latest_data_date=actual_e,
        universe_as_of=universe_as_of,
    )
    if log:
        _append_log(result, start, end, kw)
    return result


def assemble_data_kwargs(
    required_data: list[str],
    *,
    start: str,
    end: str,
    universe: str | None = None,
    log: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[AcquisitionResult]]:
    """Fetch all required tokens and assemble into data_kwargs dict.

    Returns:
      (data_kwargs, results)
        data_kwargs: {token: DataFrame} for successful fetches
        results:     list of AcquisitionResult for ALL tokens (incl. failures)

    Caller can inspect results to detect:
      - Any failure (success=False) → may need to abort or downgrade protocol
      - Any partial_coverage → protocol designer should restrict sample window
      - Any quality_caveats → protocol designer should add bars

    The caller is responsible for deciding what to do with failures — this
    function NEVER silently substitutes data or raises (caller decides).
    """
    data_kwargs: dict[str, pd.DataFrame] = {}
    results: list[AcquisitionResult] = []
    for token in (required_data or []):
        result = fetch_token(token, start=start, end=end,
                               log=log, universe=universe)
        results.append(result)
        if result.success and result.df is not None:
            data_kwargs[token] = result.df
    return data_kwargs, results
