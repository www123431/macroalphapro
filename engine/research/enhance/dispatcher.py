"""engine.research.enhance.dispatcher — Phase 2 enhance entry point.

Public API: dispatch_enhance_hypothesis(sleeve_id, variant_returns,
                                        hypothesis_id, ...)

What this does
==============
1. Resolves deployed sleeve PnL via SleeveProtocol.returns()
2. Pairs against caller-supplied variant returns
3. Runs paired block bootstrap (Politis-Romano 1994) on Sharpe diff
4. Classifies verdict (IMPROVEMENT / NOISE / DEGRADATION)
5. Writes verdict + summary metrics to data/research/enhance_verdicts.jsonl
6. Returns EnhanceDispatchResult for caller audit

What this DOES NOT do
=====================
- Generate the variant from the hypothesis claim (Phase 2.2 LLM-driven
  variant builder lives in a separate module; substrate caller supplies
  pd.Series directly)
- Emit canonical research_store events (Phase 2.2 adds the
  'enhancement_evaluated' EventType)
- Modify library yaml / deployed config — IMPROVEMENT verdicts route to
  /approvals for capital decision (kept HUMAN per standing doctrine)
- Burn forward cron quota — enhance is a parallel pipeline with its own
  weekly Sunday 04:00 cadence (Phase 3 installer)
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from engine.research.enhance.paired_bootstrap import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_N_BOOTSTRAP,
    PairedBootstrapResult,
    paired_block_bootstrap_sharpe_diff,
)
from engine.research.enhance.verdict import (
    EnhanceVerdict,
    classify_enhance_verdict,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VERDICT_LOG_PATH = _REPO_ROOT / "data" / "research" / "enhance_verdicts.jsonl"


@_dc.dataclass(frozen=True)
class EnhanceDispatchResult:
    dispatch_event_id:    str
    ts:                   str
    hypothesis_id:        str
    sleeve_id:            str
    cron_run_id:          Optional[str]
    verdict:              str          # EnhanceVerdict.value
    bootstrap_result:     Optional[dict]   # PairedBootstrapResult.to_dict()
    refusal_reason:       Optional[str]
    refusal_detail:       Optional[str]
    n_obs_baseline:       int
    n_obs_variant:        int
    summary:              str

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_sleeve_returns(sleeve_id: str) -> Optional[pd.Series]:
    """Read deployed-sleeve monthly returns via SleeveProtocol.

    Returns None when sleeve isn't registered or returns() raises.
    """
    try:
        from engine.research.sleeve_registry import get_sleeve
    except ImportError:
        logger.warning("enhance.dispatcher: sleeve_registry import failed")
        return None
    try:
        sleeve = get_sleeve(sleeve_id)
    except KeyError:
        logger.warning("enhance.dispatcher: sleeve %r not registered",
                        sleeve_id)
        return None
    try:
        rets = sleeve.returns()
    except Exception as exc:
        logger.warning("enhance.dispatcher: %r.returns() raised: %s",
                        sleeve_id, exc)
        return None
    if not isinstance(rets, pd.Series):
        logger.warning("enhance.dispatcher: %r.returns() returned non-Series",
                        sleeve_id)
        return None
    return rets.dropna()


def _append_verdict_log(record: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Public API ─────────────────────────────────────────────────────


def dispatch_enhance_hypothesis(
    *,
    hypothesis_id:     str,
    sleeve_id:         str,
    variant_returns:   pd.Series,
    baseline_returns:  Optional[pd.Series] = None,
    cron_run_id:       Optional[str] = None,
    cron_source:       Optional[str] = None,
    n_iterations:      int = DEFAULT_N_BOOTSTRAP,
    block_size:        int = DEFAULT_BLOCK_SIZE,
    log_path:          Optional[Path] = None,
    seed:              int = 42,
) -> EnhanceDispatchResult:
    """Enhance pipeline entry. Returns EnhanceDispatchResult.

    baseline_returns: optional override; defaults to
        SleeveProtocol.returns() for sleeve_id. Useful for tests where
        the registry isn't populated.
    variant_returns: caller-supplied monthly returns of the proposed
        modification, aligned to baseline by datetime index.

    Refusal reasons:
      SLEEVE_NOT_RESOLVED      : sleeve_id unknown / returns() failed
      INSUFFICIENT_OVERLAP     : <24 paired months after alignment
      DEGENERATE_VARIANCE      : either series has zero variance
      LOW_CORRELATION_NEW_FACTOR_ROUTE :
                                  corr < 0.50 → caller should route
                                  this through forward, not enhance
    """
    ts = _utc_iso()
    dispatch_event_id = str(uuid.uuid4())

    # 1. Resolve baseline
    if baseline_returns is None:
        baseline_returns = _resolve_sleeve_returns(sleeve_id)
    if baseline_returns is None:
        return _record_refusal(
            dispatch_event_id, ts, hypothesis_id, sleeve_id,
            cron_run_id, cron_source,
            "SLEEVE_NOT_RESOLVED",
            f"sleeve {sleeve_id!r} returns() unavailable",
            log_path=log_path,
        )

    n_obs_baseline = len(baseline_returns.dropna())
    n_obs_variant  = len(variant_returns.dropna())

    # 2. Run paired bootstrap (handles alignment + min-obs + variance check)
    result = paired_block_bootstrap_sharpe_diff(
        baseline_returns, variant_returns,
        n_iterations=n_iterations, block_size=block_size, seed=seed,
    )
    if result is None:
        return _record_refusal(
            dispatch_event_id, ts, hypothesis_id, sleeve_id,
            cron_run_id, cron_source,
            "INSUFFICIENT_OVERLAP_OR_DEGENERATE",
            (f"paired_block_bootstrap returned None — "
              f"n_base={n_obs_baseline}, n_var={n_obs_variant}, "
              f"check min 24mo overlap + non-zero std"),
            log_path=log_path,
            n_obs_baseline=n_obs_baseline,
            n_obs_variant=n_obs_variant,
        )

    # 3. Correlation routing check
    if abs(result.correlation) < 0.50:
        return _record_refusal(
            dispatch_event_id, ts, hypothesis_id, sleeve_id,
            cron_run_id, cron_source,
            "LOW_CORRELATION_NEW_FACTOR_ROUTE",
            (f"corr(baseline,variant)={result.correlation:.3f} < 0.50; "
              f"this is a new strategy not an enhancement — route through "
              f"forward pipeline instead."),
            log_path=log_path,
            bootstrap_result=result.to_dict(),
            n_obs_baseline=n_obs_baseline,
            n_obs_variant=n_obs_variant,
        )

    # 4. Classify
    verdict = classify_enhance_verdict(result)

    from engine.research.enhance.paired_bootstrap import paired_block_bootstrap_summary
    summary = (
        f"{sleeve_id}: variant vs deployed → {verdict.value}. "
        + paired_block_bootstrap_summary(result)
    )

    record = EnhanceDispatchResult(
        dispatch_event_id = dispatch_event_id,
        ts                = ts,
        hypothesis_id     = hypothesis_id,
        sleeve_id         = sleeve_id,
        cron_run_id       = cron_run_id,
        verdict           = verdict.value,
        bootstrap_result  = result.to_dict(),
        refusal_reason    = None,
        refusal_detail    = None,
        n_obs_baseline    = n_obs_baseline,
        n_obs_variant     = n_obs_variant,
        summary           = summary,
    )
    try:
        _append_verdict_log(_to_log_row(record, cron_source),
                              log_path or DEFAULT_VERDICT_LOG_PATH)
    except OSError as exc:
        logger.error("enhance.dispatcher: verdict_log write failed: %s", exc)
    return record


def _to_log_row(r: EnhanceDispatchResult, cron_source: Optional[str]) -> dict:
    d = r.to_dict()
    d["cron_source"] = cron_source
    return d


def _record_refusal(
    dispatch_event_id, ts, hypothesis_id, sleeve_id,
    cron_run_id, cron_source,
    refusal_reason, refusal_detail,
    *,
    log_path: Optional[Path] = None,
    bootstrap_result: Optional[dict] = None,
    n_obs_baseline: int = 0,
    n_obs_variant: int = 0,
) -> EnhanceDispatchResult:
    record = EnhanceDispatchResult(
        dispatch_event_id = dispatch_event_id,
        ts                = ts,
        hypothesis_id     = hypothesis_id,
        sleeve_id         = sleeve_id,
        cron_run_id       = cron_run_id,
        verdict           = "REFUSED",
        bootstrap_result  = bootstrap_result,
        refusal_reason    = refusal_reason,
        refusal_detail    = refusal_detail,
        n_obs_baseline    = n_obs_baseline,
        n_obs_variant     = n_obs_variant,
        summary           = f"enhance dispatch refused for {sleeve_id}: {refusal_reason}",
    )
    try:
        _append_verdict_log(_to_log_row(record, cron_source),
                              log_path or DEFAULT_VERDICT_LOG_PATH)
    except OSError as exc:
        logger.error("enhance.dispatcher: refusal log write failed: %s", exc)
    return record
