"""engine.research.decay_watch_trigger — C of A→B→C senior施工建议.

Closes the audit→capital-decision loop: when a deployed sleeve's
subsample stability lens shows the McLean-Pontiff decay pattern
(worst/best Sharpe collapse / monotone decay / latest-vs-prior
drop), AUTO-emit a `decay_alert` event with a SUGGESTION to review
capital allocation.

CRITICAL discipline preserved per
[[feedback-research-auto-capital-human-2026-06-05]]:
  - Event carries SUGGESTION ("review allocation"), NOT a command.
  - Verdict is MARGINAL by default (NEUTRAL when only mild signal,
    RED only for catastrophic decay).
  - Principal decides whether to reduce / re-allocate. The system
    surfaces; the human acts.

EVALUATION CRITERIA (locked per施工建议 plan 2026-06-09)
========================================================
For a deployed-sleeve subsample_stability lens output, fire when ANY of:

  TRIGGER A: worst_best_sharpe_ratio < 0.20
    → most of the headline Sharpe came from one window; rest is dead.
    Carry sleeve audit pattern: W1 2002-2007 Sharpe +1.11, W2-W4 ~0.
    Empirical bar 0.20 chosen because McLean-Pontiff 2016 average
    post-pub decay is 32-58% — anything more concentrated than
    "best window is 5× worst" is a stronger signal than the paper's
    average.

  TRIGGER B: monotone_decay == True
    → each split's Sharpe strictly less than the prior. Classical
    publication-decay pattern; not always present in noisy series
    but DEFINITIVE when it is.

  TRIGGER C: latest_window_sharpe < 0.5 × prior_window_sharpe
    → fresh evidence of regime break in the most recent split.
    Cocooned in current sample; future likely worse.

SEVERITY MAPPING
================
Counts how many triggers fired:
  3 of 3 → "RED"  (HARD — extreme decay; review URGENT)
  2 of 3 → "MARGINAL" (SOFT — clear decay; review recommended)
  1 of 3 → "NEUTRAL" (INFO — one signal only; no auto-emit by default)
  0 of 3 → no event emitted

The NEUTRAL boundary is opt-in via `min_triggers_for_emit=1`. Default
is `min_triggers_for_emit=2` so we don't flood the inbox on weak
signals.

PURE FUNCTIONS
==============
`evaluate_subsample_for_decay` is IO-free — takes a subsample dict
and returns an evaluation dict. Wiring to emit + persistence is in
the separate `emit_decay_alert_from_subsample` helper. Tests can
exercise evaluation logic without touching the event store.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Trigger thresholds — locked per senior施工建议 2026-06-09
# ────────────────────────────────────────────────────────────────────
WORST_BEST_DECAY_BAR    = 0.20   # < this → trigger A
LATEST_VS_PRIOR_BAR     = 0.50   # latest < this × prior → trigger C
MIN_TRIGGERS_FOR_EMIT_DEFAULT = 2   # 2-of-3 → MARGINAL emit; 1 → NEUTRAL log


# ────────────────────────────────────────────────────────────────────
# Pure evaluator
# ────────────────────────────────────────────────────────────────────
def evaluate_subsample_for_decay(
    subsample_output: dict,
) -> dict:
    """Evaluate a subsample_stability lens output for decay signals.

    Args:
      subsample_output: the dict returned by
        engine.research.subsample_stability.compute_subsample_stability
        (or its Tier C wiring helper). MUST contain `windows`,
        `worst_best_sharpe_ratio`, `monotone_decay`.

    Returns a dict:
      {
        "triggers_hit":  ["A", "B", "C"] subset,
        "n_triggers":    int,
        "severity":      "NEUTRAL" | "MARGINAL" | "RED",
        "worst_best_sharpe_ratio": float | None,
        "monotone_decay":          bool,
        "latest_vs_prior_ratio":   float | None,
        "latest_window_sharpe":    float | None,
        "prior_window_sharpe":     float | None,
        "summary": str,            # one-line human-readable
      }
    """
    triggers: list[str] = []

    wbr = subsample_output.get("worst_best_sharpe_ratio")
    monotone = bool(subsample_output.get("monotone_decay"))

    # TRIGGER A — worst/best collapse
    if wbr is not None and wbr < WORST_BEST_DECAY_BAR:
        triggers.append("A")

    # TRIGGER B — strict monotone decay
    if monotone:
        triggers.append("B")

    # TRIGGER C — latest vs prior window
    windows = subsample_output.get("windows") or []
    latest_sharpe: Optional[float] = None
    prior_sharpe:  Optional[float] = None
    ratio: Optional[float] = None
    if len(windows) >= 2:
        latest = windows[-1].get("sharpe_ann")
        prior  = windows[-2].get("sharpe_ann")
        if (latest is not None and prior is not None
                and prior > 0):
            latest_sharpe = float(latest)
            prior_sharpe  = float(prior)
            ratio = latest_sharpe / prior_sharpe
            if ratio < LATEST_VS_PRIOR_BAR:
                triggers.append("C")

    # Severity from trigger count
    n = len(triggers)
    if n >= 3:
        severity = "RED"
    elif n >= 2:
        severity = "MARGINAL"
    elif n >= 1:
        severity = "NEUTRAL"
    else:
        severity = "NEUTRAL"

    # Human-readable summary
    parts = []
    if "A" in triggers:
        parts.append(f"worst/best Sharpe={wbr:.2f} < {WORST_BEST_DECAY_BAR}")
    if "B" in triggers:
        parts.append("strict monotone decay")
    if "C" in triggers and ratio is not None:
        parts.append(f"latest Sharpe={latest_sharpe:.2f} < "
                       f"{LATEST_VS_PRIOR_BAR}× prior {prior_sharpe:.2f}")
    summary = ("Decay signals: " + "; ".join(parts)) if parts else (
        "No decay signal triggered.")

    return {
        "triggers_hit":            triggers,
        "n_triggers":              n,
        "severity":                severity,
        "worst_best_sharpe_ratio": wbr,
        "monotone_decay":          monotone,
        "latest_vs_prior_ratio":   ratio,
        "latest_window_sharpe":    latest_sharpe,
        "prior_window_sharpe":     prior_sharpe,
        "summary":                 summary,
    }


# ────────────────────────────────────────────────────────────────────
# K (2026-06-10): cron-safe dedup
# ────────────────────────────────────────────────────────────────────
def _alert_signature(triggers: list, severity: str) -> str:
    """Stable signature of a decay finding. Two alerts with the same
    signature describe the same fact — re-emitting is spam."""
    return f"{','.join(sorted(triggers))}|{severity}"


def should_emit_for_subject(
    subject_id: str,
    evaluation: dict,
) -> tuple[bool, str]:
    """K cron-dedup semantics. Returns (should_emit, reason).

      1. No prior canonical alert for subject       → emit
      2. Latest original alert is OPEN (un-acked)   → skip
         (the principal hasn't reviewed the existing one; re-emitting
         the same sleeve daily is alert fatigue, the #1 way
         institutional alert systems die)
      3. Latest original is ACKED, same signature   → skip
         (principal already reviewed THIS exact finding)
      4. Latest original is ACKED, signature DIFFERS → emit
         (the facts changed — e.g. escalated A,B→A,B,C — a past ack
         does not cover new evidence)

    Reads ack state via api.main._decay_ack_chain (the same walker
    the UI uses, so cron and UI can never disagree about open-ness).
    """
    try:
        from engine.research_store import store
        from api.main import _decay_ack_chain
    except Exception:
        # Defensive: if state can't be read, emit (loud beats silent)
        return True, "state_unreadable_fail_open"

    try:
        events = store.filter_events(
            event_type="decay_alert", subject_id=subject_id, limit=500,
        )
    except Exception:
        return True, "store_unreadable_fail_open"

    canonical = [e for e in events if "decay_watch" in (e.tags or ())]
    originals = [
        e for e in canonical
        if "acknowledged" not in (e.tags or ())
        and "unacknowledged" not in (e.tags or ())
    ]
    if not originals:
        return True, "no_prior_alert"

    latest = originals[0]   # filter_events is newest-first
    ack_state = _decay_ack_chain(canonical)
    latest_state = ack_state.get(latest.event_id)
    is_acked = bool(latest_state and latest_state.get("is_acknowledged"))

    if not is_acked:
        return False, "open_alert_exists"

    prior_sig = _alert_signature(
        (latest.metrics or {}).get("triggers_hit") or [],
        (latest.metrics or {}).get("severity") or "",
    )
    new_sig = _alert_signature(
        evaluation["triggers_hit"], evaluation["severity"],
    )
    if prior_sig == new_sig:
        return False, "acked_same_signature"
    return True, f"signature_changed:{prior_sig}->{new_sig}"


# ────────────────────────────────────────────────────────────────────
# Emit helper — calls engine.research_store.emit.decay_alert
# ────────────────────────────────────────────────────────────────────
def emit_decay_alert_from_subsample(
    *,
    subject_id:           str,
    subsample_output:     dict,
    parent_event_ids:     tuple = (),
    min_triggers_for_emit: int = MIN_TRIGGERS_FOR_EMIT_DEFAULT,
    actor:                str = "engine.decay_watch_trigger",
    extra_tags:           tuple = (),
    dedup:                bool = False,
) -> Optional[str]:
    """Evaluate the subsample output; if enough triggers fired, emit
    a `decay_alert` event tagged for principal review.

    Args:
      subject_id: registered sleeve subject (e.g., "equity_book",
                  "cross_asset_carry"). MUST exist in the registry
                  per the research-store doctrine; emit.decay_alert
                  validates this and raises on typo.
      subsample_output: subsample_stability lens output dict.
      parent_event_ids: lineage to upstream event(s) (e.g., the
                        factor_verdict_filed event from the audit).
      min_triggers_for_emit: 2 (default) → MARGINAL or RED only.
                             Set to 1 to also emit on NEUTRAL.
      extra_tags: appended to the standard tags ("decay_watch",
                  "review_recommended").

    Returns the new event_id on emit, or None when:
      * no triggers fired
      * fewer than `min_triggers_for_emit` triggers
      * emit raises (logged, swallowed)
    """
    evaluation = evaluate_subsample_for_decay(subsample_output)
    if evaluation["n_triggers"] < min_triggers_for_emit:
        return None

    # K (2026-06-10): cron dedup — skip when an open alert exists or
    # the same finding was already acked. See should_emit_for_subject.
    if dedup:
        ok, reason = should_emit_for_subject(subject_id, evaluation)
        if not ok:
            logger.info("decay_watch_trigger: dedup skip for %s (%s)",
                           subject_id, reason)
            return None

    # Map severity → verdict accepted by emit.decay_alert
    severity = evaluation["severity"]
    if severity == "RED":
        verdict_str = "RED"
    elif severity == "MARGINAL":
        verdict_str = "MARGINAL"
    else:
        # NEUTRAL emits skipped unless caller forced via
        # min_triggers_for_emit=1. Even then we route as MARGINAL
        # since "NEUTRAL" isn't an emit.decay_alert verdict.
        verdict_str = "MARGINAL"

    # Build metrics + artifacts payload
    metrics = {
        "triggers_hit":            evaluation["triggers_hit"],
        "n_triggers":              evaluation["n_triggers"],
        "severity":                severity,
        "worst_best_sharpe_ratio": evaluation["worst_best_sharpe_ratio"],
        "monotone_decay":          evaluation["monotone_decay"],
        "latest_vs_prior_ratio":   evaluation["latest_vs_prior_ratio"],
        "latest_window_sharpe":    evaluation["latest_window_sharpe"],
        "prior_window_sharpe":     evaluation["prior_window_sharpe"],
        # Provenance: capture the subsample's window breakdown
        "windows":                 list(subsample_output.get("windows", [])),
        "n_splits":                subsample_output.get("n_splits"),
        "n_total_months":          subsample_output.get("n_total_months"),
    }
    artifacts: dict[str, str] = {}

    tags = ("decay_watch", "review_recommended") + tuple(extra_tags)

    # Suggestion language — never a command. Per
    # [[feedback-research-auto-capital-human-2026-06-05]] capital
    # decisions remain HUMAN.
    summary = (
        f"Decay watch [{severity}] on {subject_id}: "
        + evaluation["summary"]
        + " — SUGGESTION: review capital allocation."
    )

    try:
        from engine.research_store import emit
        event_id = emit.decay_alert(
            subject_id       = subject_id,
            verdict          = verdict_str,
            metrics          = metrics,
            artifacts        = artifacts,
            summary          = summary,
            parent_event_ids = parent_event_ids,
            tags             = tags,
            actor            = actor,
        )
        logger.info("decay_watch_trigger: emitted decay_alert "
                       "event_id=%s for subject=%s (severity=%s)",
                       event_id, subject_id, severity)
        return event_id
    except Exception as exc:
        logger.exception("decay_watch_trigger: emit failed for %s: %s",
                            subject_id, exc)
        return None
