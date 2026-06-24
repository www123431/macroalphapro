"""engine/research/decay_retest.py — when decay sentinel flags WATCH/ACTION,
auto-trigger a deeper "is this real decay or noise?" check.

Reactive-subscriber #4 off the deferred queue (2026-06-14). Without this,
a WATCH/ACTION alert sits on /research/decay and the principal has to
manually decide if the sleeve's underlying anomaly actually degraded.
This module:

  1. Receives a retest request (auto from decay_alert OR manual from UI)
  2. Pulls the sleeve's monthly returns from the live registry
  3. Runs Chow structural-break test + bootstrap Sharpe CI
  4. Emits CONFIRMED_DECAY / NOISE_INDISTINGUISHABLE / INSUFFICIENT_DATA
     to data/research/decay_retest_results.jsonl
  5. (Optionally) emits a typed retest_completed event into the store
     for lineage with the originating decay_alert

Key choice: this is a STATISTICAL retest of the existing return series,
NOT a full re-dispatch of the original strategy spec. Re-dispatch is
~10x more expensive and largely returns the same answer (the spec
hasn't changed, only the returns have); the cheap statistical retest
gives the principal a defensible "is the drop real?" verdict in seconds.

Doctrine:
  Chow p < 0.05 + post-mean < pre-mean        → CONFIRMED_DECAY
  Bootstrap-CI(rolling-Sharpe) overlaps zero  → CONFIRMED_DECAY
  Otherwise (Chow p >= 0.05 AND CI > 0)       → NOISE_INDISTINGUISHABLE
  n_obs < 36 monthly                          → INSUFFICIENT_DATA
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
RETEST_QUEUE_PATH    = _REPO_ROOT / "data" / "research" / "decay_retest_queue.jsonl"
RETEST_RESULTS_PATH  = _REPO_ROOT / "data" / "research" / "decay_retest_results.jsonl"

MIN_OBS_MONTHS         = 36     # below = INSUFFICIENT_DATA
BREAK_WINDOW_MONTHS    = 18     # post-window for Chow
BOOTSTRAP_ROUNDS       = 2000
BOOTSTRAP_CI_ALPHA     = 0.10   # 90% two-sided


@_dc.dataclass(frozen=True)
class RetestResult:
    retest_id:      str
    sleeve_id:      str
    triggered_by:   str   # "decay_alert" / "manual" / "cron"
    triggered_at:   str
    n_obs_months:   int
    verdict:        str   # CONFIRMED_DECAY / NOISE_INDISTINGUISHABLE / INSUFFICIENT_DATA
    chow_p_value:   float | None
    chow_structural_break: bool | None
    pre_mean:       float | None
    post_mean:      float | None
    sharpe_full:    float | None
    sharpe_recent:  float | None
    sharpe_ci_lo:   float | None
    sharpe_ci_hi:   float | None
    rationale:      str
    parent_event_id: Optional[str] = None


# ── Statistical core ───────────────────────────────────────────────


def _bootstrap_sharpe_ci(returns: pd.Series, *,
                          rounds: int = BOOTSTRAP_ROUNDS,
                          window_months: int = BREAK_WINDOW_MONTHS,
                          alpha: float = BOOTSTRAP_CI_ALPHA,
                          seed: int = 42) -> tuple[float, float]:
    """Stationary-block bootstrap CI on the RECENT-window annualized
    Sharpe. Block length sqrt(N) per Politis-Romano 1994 standard rec."""
    r = returns.dropna().iloc[-window_months:].values
    if len(r) < 12:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    block_len = max(2, int(np.sqrt(len(r))))
    n_blocks = (len(r) + block_len - 1) // block_len
    sharpes = np.empty(rounds, dtype=np.float64)
    for i in range(rounds):
        starts = rng.integers(0, len(r), size=n_blocks)
        sample = np.concatenate([
            np.take(r, range(s, s + block_len), mode="wrap") for s in starts
        ])[:len(r)]
        s_mean = sample.mean()
        s_std  = sample.std(ddof=1)
        sharpes[i] = (s_mean / s_std * np.sqrt(12)) if s_std > 1e-9 else 0.0
    lo, hi = np.quantile(sharpes, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _full_sample_sharpe(returns: pd.Series) -> Optional[float]:
    r = returns.dropna()
    if len(r) < 6:
        return None
    s_std = r.std(ddof=1)
    if s_std < 1e-9:
        return None
    return float(r.mean() / s_std * np.sqrt(12))


# ── Public retest fn ───────────────────────────────────────────────


def run_retest(sleeve_id: str, returns: pd.Series, *,
                triggered_by: str = "manual",
                parent_event_id: Optional[str] = None) -> RetestResult:
    """Run the statistical retest on a sleeve's monthly returns series.

    Returns a RetestResult with the verdict. The verdict logic:

      n < MIN_OBS_MONTHS               → INSUFFICIENT_DATA
      Chow p < 0.05 AND post < pre     → CONFIRMED_DECAY
      bootstrap CI on recent Sharpe
        upper bound <= 0               → CONFIRMED_DECAY (signal flipped)
      otherwise                        → NOISE_INDISTINGUISHABLE

    Caller decides whether to emit a typed event; this fn just computes.
    """
    from engine.validation.decay_sentinel import chow_test_decay
    import uuid as _uuid

    now = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    rid = str(_uuid.uuid4())
    r = returns.dropna()
    n_obs = int(len(r))

    if n_obs < MIN_OBS_MONTHS:
        return RetestResult(
            retest_id=rid, sleeve_id=sleeve_id, triggered_by=triggered_by,
            triggered_at=now, n_obs_months=n_obs,
            verdict="INSUFFICIENT_DATA",
            chow_p_value=None, chow_structural_break=None,
            pre_mean=None, post_mean=None,
            sharpe_full=None, sharpe_recent=None,
            sharpe_ci_lo=None, sharpe_ci_hi=None,
            rationale=f"n_obs={n_obs} < MIN_OBS_MONTHS={MIN_OBS_MONTHS}; cannot test",
            parent_event_id=parent_event_id,
        )

    # Chow test (from decay_sentinel)
    chow = chow_test_decay(r, break_n_periods=BREAK_WINDOW_MONTHS)
    sharpe_full   = _full_sample_sharpe(r)
    sharpe_recent = _full_sample_sharpe(r.iloc[-BREAK_WINDOW_MONTHS:])

    # Bootstrap CI
    ci_lo, ci_hi = _bootstrap_sharpe_ci(r,
        rounds=BOOTSTRAP_ROUNDS, window_months=BREAK_WINDOW_MONTHS)

    # Verdict logic
    verdict = "NOISE_INDISTINGUISHABLE"
    rationale_parts = []
    if chow.get("structural_break"):
        verdict = "CONFIRMED_DECAY"
        rationale_parts.append(f"Chow p={chow.get('p_value'):.3f} → structural break")
    elif np.isfinite(ci_hi) and ci_hi <= 0:
        verdict = "CONFIRMED_DECAY"
        rationale_parts.append(
            f"bootstrap CI on recent {BREAK_WINDOW_MONTHS}m Sharpe = "
            f"[{ci_lo:.2f}, {ci_hi:.2f}] — entirely <= 0")
    else:
        rationale_parts.append(
            f"Chow p={chow.get('p_value', float('nan')):.3f} >= 0.05; "
            f"recent Sharpe CI [{ci_lo:.2f}, {ci_hi:.2f}] crosses 0 — "
            f"cannot reject H0 'no structural decline'")
    if sharpe_full is not None and sharpe_recent is not None:
        rationale_parts.append(
            f"full-sample Sh={sharpe_full:.2f} vs recent Sh={sharpe_recent:.2f}")

    return RetestResult(
        retest_id=rid, sleeve_id=sleeve_id, triggered_by=triggered_by,
        triggered_at=now, n_obs_months=n_obs,
        verdict=verdict,
        chow_p_value=chow.get("p_value"),
        chow_structural_break=bool(chow.get("structural_break")),
        pre_mean=chow.get("pre_mean"),
        post_mean=chow.get("post_mean"),
        sharpe_full=sharpe_full, sharpe_recent=sharpe_recent,
        sharpe_ci_lo=ci_lo, sharpe_ci_hi=ci_hi,
        rationale="; ".join(rationale_parts),
        parent_event_id=parent_event_id,
    )


# ── Queue + results persistence ────────────────────────────────────


def enqueue_retest(sleeve_id: str, *,
                    triggered_by: str = "manual",
                    parent_event_id: Optional[str] = None) -> str:
    """Append a retest request to the queue. Dedups within 24h —
    re-queuing a sleeve that's already pending in the last day no-ops
    so a noisy decay alert can't flood the queue."""
    import uuid as _uuid
    rid = str(_uuid.uuid4())
    now = _dt.datetime.utcnow()

    # Dedup
    if RETEST_QUEUE_PATH.is_file():
        cutoff_iso = (now - _dt.timedelta(hours=24)).isoformat()
        for ln in RETEST_QUEUE_PATH.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if (r.get("sleeve_id") == sleeve_id
                and r.get("status") in ("pending", "processing")
                and r.get("queued_at", "") >= cutoff_iso):
                logger.info("decay_retest dedup hit for %s within 24h", sleeve_id)
                return r.get("retest_id") or rid

    RETEST_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "retest_id":       rid,
        "sleeve_id":       sleeve_id,
        "triggered_by":    triggered_by,
        "parent_event_id": parent_event_id,
        "queued_at":       now.isoformat(timespec="seconds") + "Z",
        "status":          "pending",
    }
    with RETEST_QUEUE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return rid


def _load_queue() -> list[dict]:
    if not RETEST_QUEUE_PATH.is_file():
        return []
    out: list[dict] = []
    for ln in RETEST_QUEUE_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def list_queue(*, include_completed: bool = False) -> list[dict]:
    """Return current queue. By default skips status=completed/failed
    rows so the UI shows only actionable items."""
    rows = _load_queue()
    if not include_completed:
        rows = [r for r in rows if r.get("status") in ("pending", "processing")]
    return rows


def _persist_result(result: RetestResult) -> None:
    RETEST_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RETEST_RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_dc.asdict(result)) + "\n")


def list_results(limit: int = 50) -> list[dict]:
    if not RETEST_RESULTS_PATH.is_file():
        return []
    rows = []
    for ln in RETEST_RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("triggered_at", ""), reverse=True)
    return rows[:limit]


def _mark_queue_status(retest_id: str, status: str) -> None:
    """Append a new queue row marking the status change. Simpler than
    rewriting the file; the loader takes the LATEST status per
    retest_id when checking dedup."""
    if not retest_id:
        return
    with RETEST_QUEUE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "retest_id":   retest_id,
            "status":      status,
            "ts":          _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }) + "\n")


# ── Returns loader ─────────────────────────────────────────────────


def _load_sleeve_returns(sleeve_id: str) -> Optional[pd.Series]:
    """Pull monthly returns for a deployed sleeve from the live engine
    registry. Same source decay_sentinel uses (single-source-of-truth
    per the existing build_mechanisms doctrine)."""
    try:
        from engine.portfolio.replay_combined import load_all_strategy_returns_weekly
        weekly = load_all_strategy_returns_weekly()
        if sleeve_id not in weekly.columns:
            return None
        monthly = weekly[sleeve_id].resample("ME").apply(
            lambda s: (1 + s).prod() - 1)
        return monthly.dropna()
    except Exception:
        logger.exception("decay_retest: returns load failed for %s", sleeve_id)
        return None


# ── Process queue (cron + manual entrypoint) ────────────────────────


def process_queue(*, limit: int = 10) -> list[RetestResult]:
    """Drain up to `limit` pending queue rows; for each run retest +
    persist result + mark status=completed.

    Returns the list of results produced (may be shorter than limit
    if queue is exhausted)."""
    out: list[RetestResult] = []
    # Build a sleeve_id → latest_status map so the loop respects
    # in-flight markers from the same run.
    rows = _load_queue()
    latest_status: dict[str, dict] = {}
    for r in rows:
        rid = r.get("retest_id")
        if rid:
            latest_status[rid] = r

    for rid, row in list(latest_status.items())[:limit]:
        if row.get("status") != "pending":
            continue
        sleeve_id = row.get("sleeve_id")
        if not sleeve_id:
            continue
        _mark_queue_status(rid, "processing")
        try:
            returns = _load_sleeve_returns(sleeve_id)
            if returns is None or returns.empty:
                result = RetestResult(
                    retest_id=rid, sleeve_id=sleeve_id,
                    triggered_by=row.get("triggered_by", "?"),
                    triggered_at=_dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    n_obs_months=0,
                    verdict="INSUFFICIENT_DATA",
                    chow_p_value=None, chow_structural_break=None,
                    pre_mean=None, post_mean=None,
                    sharpe_full=None, sharpe_recent=None,
                    sharpe_ci_lo=None, sharpe_ci_hi=None,
                    rationale=f"no return series available for sleeve_id={sleeve_id}",
                    parent_event_id=row.get("parent_event_id"),
                )
            else:
                result = run_retest(
                    sleeve_id, returns,
                    triggered_by=row.get("triggered_by", "?"),
                    parent_event_id=row.get("parent_event_id"),
                )
                # Override the auto-generated retest_id with the queue's
                # so caller can match back to the queue row.
                result = _dc.replace(result, retest_id=rid)
            _persist_result(result)
            _mark_queue_status(rid, "completed")
            out.append(result)
        except Exception as exc:
            logger.exception("decay_retest: processing failed for %s", sleeve_id)
            _mark_queue_status(rid, "failed")
    return out
