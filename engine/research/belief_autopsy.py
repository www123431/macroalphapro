"""engine.research.belief_autopsy — Belief Layer Phase 2.

Reads (prediction, verdict) pairs and produces surprise diagnostics +
pattern flags. Output is persistent autopsy records that belief-3
(calibration dashboard) and belief-4 (closed-loop prior update) consume.

Doctrine
========
- belief-1 predicts BEFORE verdict computation (air-gapped from lens
  code; see test_belief.py structural invariant)
- belief-2 (this module) reads BOTH predictions and verdicts, joins
  by subject_id, computes surprise — runs AFTER verdict emit, never
  during template/lens computation, so the air-gap is preserved
- belief-2 does NOT update predictions; it produces a parallel
  autopsies.jsonl file. Predictions remain frozen + auditable.

Inputs
======
- data/research/predictions.jsonl    (belief-1 output)
- data/research_store/events.jsonl   (factor_verdict_filed events)

Output
======
- data/research/autopsies.jsonl      (one row per matched pair)

Each autopsy row:
  autopsy_id                uuid4
  ts                        ISO-8601
  prediction_id             linked
  verdict_event_id          linked
  subject_id                shared key
  strategy_family           from verdict event (canonical, post-Option B)
  claim_family              from prediction.inputs.claim_family if available
  predicted_verdict_dist    {GREEN, MARGINAL, RED} - sums to 1
  actual_verdict            GREEN / MARGINAL / RED
  brier_component           (1 - p(actual_verdict))^2 — lower = better
                              calibrated. Range [0, 1].
  surprise_direction        "over_predicted_green" / "over_predicted_red" /
                              "well_calibrated" / "neutral"
  surprise_magnitude        |p(actual_verdict) - max_p(other_verdicts)| where
                              actual not in argmax(predicted). 0 if actual
                              was modal predicted.
  load_bearing_realized     subset of predicted_load_bearing that actually
                              materialized (judged by metrics, e.g.
                              spanning_risk realized iff jk_vs_a_t <= 0)
  prediction_basis_echo     echo from belief-1 prediction for audit

Pattern detection
=================
After accumulating N >= 10 autopsies, compute:
  - Brier mean / median / stddev (overall calibration)
  - Per-family Brier mean (which families are mispredicted)
  - Direction bias (consistent over_predicted_green?)
  - N=10 sliding-window flag: if last 10 all over_predicted_green,
    return PATTERN_GREEN_OVERCONFIDENCE flag

These flags feed belief-3 dashboard + eventually belief-4 closed-loop.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = _REPO_ROOT / "data" / "research" / "predictions.jsonl"
EVENTS_PATH      = _REPO_ROOT / "data" / "research_store" / "events.jsonl"
AUTOPSIES_PATH   = _REPO_ROOT / "data" / "research" / "autopsies.jsonl"

# Pattern detection thresholds
MIN_AUTOPSIES_FOR_PATTERN = 10
GREEN_OVERCONFIDENCE_WINDOW = 10
GREEN_OVERCONFIDENCE_THRESHOLD = 0.7   # 7+/10 over-predicted GREEN


@_dc.dataclass(frozen=True)
class Autopsy:
    autopsy_id:             str
    ts:                     str
    prediction_id:          str
    verdict_event_id:       str
    subject_id:             str
    strategy_family:        Optional[str]
    claim_family:            Optional[str]
    predicted_verdict_dist: dict[str, float]
    actual_verdict:         str
    brier_component:        float
    surprise_direction:     str
    surprise_magnitude:     float
    load_bearing_realized:  list[str]
    prediction_basis_echo:  str
    # BUG-1 correction support (2026-06-13):
    # - superseded_by: autopsy_id of the correction row that replaces this
    # - bug1_correction: True iff this row IS a correction of an earlier
    #   autopsy made obsolete by the BUG-1 spanning fix
    superseded_by:          Optional[str] = None
    bug1_correction:        bool = False
    # BUG-4 (2026-06-13): sample size for belief-4 precision weighting
    n_obs_months:           int = 0

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


# ── IO helpers ─────────────────────────────────────────────────────


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("belief_autopsy: %s line %d malformed",
                                 path.name, ln_no)


def _load_predictions(path: Optional[Path] = None) -> list[dict]:
    return list(_iter_jsonl(path or PREDICTIONS_PATH))


def _load_factor_verdict_events(path: Optional[Path] = None) -> list[dict]:
    out = []
    for ev in _iter_jsonl(path or EVENTS_PATH):
        if ev.get("event_type") == "factor_verdict_filed":
            out.append(ev)
    return out


def _existing_autopsy_pairs(path: Optional[Path] = None) -> set[tuple[str, str]]:
    """Return set of (prediction_id, verdict_event_id) pairs already autopsied.
    Used to dedup: same pair appended only once."""
    out: set[tuple[str, str]] = set()
    for row in _iter_jsonl(path or AUTOPSIES_PATH):
        pid = row.get("prediction_id") or ""
        vid = row.get("verdict_event_id") or ""
        if pid and vid:
            out.add((pid, vid))
    return out


def _append_autopsy(autopsy: Autopsy, path: Optional[Path] = None) -> None:
    p = path or AUTOPSIES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(autopsy.to_dict(), ensure_ascii=False) + "\n")


# ── Surprise math ──────────────────────────────────────────────────


def _brier_component(predicted_dist: dict[str, float],
                       actual_verdict: str) -> float:
    """(1 - p(actual))^2. Standard Brier component per category.
    Lower is better calibrated. Range [0, 1]."""
    p_actual = float(predicted_dist.get(actual_verdict, 0.0))
    return (1.0 - p_actual) ** 2


def _surprise_direction(predicted_dist: dict[str, float],
                          actual_verdict: str) -> str:
    """Categorize the surprise direction.

    over_predicted_green: predicted GREEN as modal, got MARGINAL or RED
    over_predicted_red:   predicted RED as modal, got MARGINAL or GREEN
    well_calibrated:      actual_verdict matches argmax(predicted)
    neutral:              fall-through for ambiguous (e.g. modal MARGINAL)
    """
    if not predicted_dist:
        return "neutral"
    modal = max(predicted_dist.items(), key=lambda kv: kv[1])[0]
    if modal == actual_verdict:
        return "well_calibrated"
    if modal == "GREEN":
        return "over_predicted_green"
    if modal == "RED":
        return "over_predicted_red"
    return "neutral"


def _surprise_magnitude(predicted_dist: dict[str, float],
                          actual_verdict: str) -> float:
    """Distance from p(actual) to the max p over other categories. 0 if
    actual was modal. Range [0, 1]."""
    if not predicted_dist:
        return 0.0
    modal_verdict = max(predicted_dist.items(), key=lambda kv: kv[1])[0]
    if modal_verdict == actual_verdict:
        return 0.0
    p_actual = float(predicted_dist.get(actual_verdict, 0.0))
    p_modal = float(predicted_dist[modal_verdict])
    return max(0.0, p_modal - p_actual)


def _load_bearing_realized(
    load_bearing: list[str], event_metrics: dict,
) -> list[str]:
    """For each load_bearing item belief-1 flagged, check whether the
    realized verdict metrics support it. Conservative — only marks
    realized when we have a clear metric signal.

    spanning_risk     realized iff CAPM α-t close to 0 OR jk vs ANY
                       component t < 1.96 (combo doesn't strictly beat
                       components → real spanning)
    family_trials     realized iff event.metrics has 'n_trials_family'
                       above 10 (BLdP pressure active at verdict time)
    post_publication_decay  realized iff oos_triple severity is severe
                              / broken
    """
    realized: list[str] = []
    if "spanning_risk" in load_bearing:
        capm_t = event_metrics.get("capm_alpha_t")
        jk_vs_a_t = event_metrics.get("jk_vs_a_t")
        jk_vs_b_t = event_metrics.get("jk_vs_b_t")
        if capm_t is not None and abs(capm_t) < 1.65:
            realized.append("spanning_risk")
        elif jk_vs_a_t is not None and abs(jk_vs_a_t) < 1.96:
            realized.append("spanning_risk")
        elif jk_vs_b_t is not None and abs(jk_vs_b_t) < 1.96:
            realized.append("spanning_risk")
    if "family_trials" in load_bearing:
        # n_trials info lives in pnl_diagnostics or directly on event
        if event_metrics.get("n_trials_family", 0) >= 10:
            realized.append("family_trials")
    if "post_publication_decay" in load_bearing:
        oos = event_metrics.get("oos_triple")
        if isinstance(oos, dict) and oos.get("severity") in ("severe", "broken"):
            realized.append("post_publication_decay")
    return realized


# ── Joining + autopsy build ────────────────────────────────────────


def _hyp_short_key(subject_id: str) -> str:
    """Extract the canonical 8-char prefix from a hypothesis_id-like
    subject_id. Mirrors factor_verdict_emit.auto_subject_id which uses
    `(spec.hypothesis_id or "unknown")[:8]` — NO hyphen-stripping.

    Bug fix 2026-06-14: prior version stripped hyphens, which worked
    for UUIDs by coincidence (hyphen at position 9 isn't in first 8)
    but BROKE for non-UUID hypothesis_ids like 'max-effect-prod' where
    auto_subject_id stored 'tier_c_auto_max-effe_X' while the matcher
    looked for 'maxeffec' (stripped). → matcher returned None → no
    autopsy emitted → belief layer silently undercounted.

    UUID '1c258025-1acf-...' → '1c258025' (same as before — hyphen at
    position 9 so [:8] is identical regardless of strip).
    'max-effect-prod' → 'max-effe' (now matches auto_subject_id).
    """
    return (subject_id or "")[:8]


def _match_prediction_to_event(
    prediction: dict, events_by_subject: dict[str, list[dict]],
) -> Optional[dict]:
    """Match prediction → factor_verdict_filed event. Joins by:
      1. subject_id == hypothesis_id (prediction's subject_id is the
         original hypothesis_id; events' subject_id is the registered
         factor subject like `tier_c_auto_<hyp_short>_<sk>`)
      2. So: scan events whose subject_id CONTAINS the prediction's
         8-char hex-prefix (hyphens stripped) — same key used by
         emit's auto_subject_id derivation.

    Returns latest matching event (by ts) or None.
    """
    pred_subject = prediction.get("subject_id", "")
    if not pred_subject:
        return None
    short = _hyp_short_key(pred_subject)
    if not short:
        return None
    candidates: list[dict] = []
    for sid, evs in events_by_subject.items():
        if short in sid:
            candidates.extend(evs)
    if not candidates:
        return None
    # Prefer events emitted AFTER the prediction (forward-look causality)
    pred_ts = prediction.get("ts", "")
    after = [e for e in candidates if (e.get("ts") or "") >= pred_ts]
    pool = after or candidates
    return max(pool, key=lambda e: e.get("ts") or "")


def build_autopsy(
    prediction: dict, event: dict,
) -> Autopsy:
    """Compute autopsy record from prediction + verdict event."""
    pred_dist = dict(prediction.get("predicted_verdict_dist") or {})
    verdict = event.get("verdict", "")
    metrics = event.get("metrics") or {}

    # BUG-4 (2026-06-13): record sample size (n_obs_months) on the
    # autopsy so belief-4 can weight observations by precision
    # (1/SE^2 ∝ N). Without this, a 60mo verdict counts the same as
    # a 360mo verdict — wrong by strict Bayesian standards.
    n_obs_months = (
        metrics.get("n_obs_months")
        or metrics.get("n_obs")   # fallback older templates
        or 0
    )

    return Autopsy(
        autopsy_id             = str(uuid.uuid4()),
        ts                     = _utc_iso(),
        prediction_id          = prediction.get("prediction_id", ""),
        verdict_event_id       = event.get("event_id", ""),
        subject_id             = prediction.get("subject_id", ""),
        strategy_family        = metrics.get("strategy_family") or event.get("family"),
        claim_family           = (
            (prediction.get("inputs") or {}).get("claim_family")
            or metrics.get("claim_family")
            or prediction.get("family")
        ),
        predicted_verdict_dist = pred_dist,
        actual_verdict         = verdict,
        brier_component        = _brier_component(pred_dist, verdict),
        surprise_direction     = _surprise_direction(pred_dist, verdict),
        surprise_magnitude     = _surprise_magnitude(pred_dist, verdict),
        load_bearing_realized  = _load_bearing_realized(
            list(prediction.get("predicted_load_bearing") or ()), metrics,
        ),
        prediction_basis_echo  = prediction.get("prediction_basis", ""),
        n_obs_months           = int(n_obs_months or 0),
    )


# ── Public API ─────────────────────────────────────────────────────


def run_autopsy_for_verdict_event(
    verdict_event_id: str,
    *,
    predictions_path: Optional[Path] = None,
    events_path:      Optional[Path] = None,
    autopsies_path:   Optional[Path] = None,
) -> Optional[Autopsy]:
    """Find the prediction paired with this verdict event + write autopsy.

    Returns None when no matching prediction OR autopsy already exists
    for this pair (idempotent).
    """
    events = _load_factor_verdict_events(events_path)
    target_event = next(
        (e for e in events if e.get("event_id") == verdict_event_id),
        None,
    )
    if target_event is None:
        return None

    target_sid = target_event.get("subject_id") or ""
    predictions = _load_predictions(predictions_path)

    # Walk predictions; find the latest one whose subject_id 8-char
    # hex prefix appears in this event's subject_id (same join key
    # logic as _match_prediction_to_event).
    candidates: list[dict] = []
    for p in predictions:
        psid = p.get("subject_id", "")
        short = _hyp_short_key(psid)
        if short and short in target_sid:
            candidates.append(p)
    if not candidates:
        return None
    # Take the latest prediction that's AT OR BEFORE the verdict time
    verdict_ts = target_event.get("ts", "")
    before = [p for p in candidates if (p.get("ts") or "") <= verdict_ts]
    pool = before or candidates
    prediction = max(pool, key=lambda p: p.get("ts") or "")

    # Dedup
    existing = _existing_autopsy_pairs(autopsies_path)
    pair = (prediction.get("prediction_id") or "",
              verdict_event_id)
    if pair in existing:
        return None

    autopsy = build_autopsy(prediction, target_event)
    try:
        _append_autopsy(autopsy, autopsies_path)
    except OSError as exc:
        logger.error("belief_autopsy: write failed: %s", exc)
    return autopsy


def backfill_all(
    *,
    predictions_path: Optional[Path] = None,
    events_path:      Optional[Path] = None,
    autopsies_path:   Optional[Path] = None,
) -> list[Autopsy]:
    """Walk all predictions, find matching events, produce autopsies for
    pairs not yet autopsied. Idempotent — re-running adds 0 rows after
    convergence."""
    predictions = _load_predictions(predictions_path)
    events = _load_factor_verdict_events(events_path)
    if not predictions or not events:
        return []

    events_by_sid: dict[str, list[dict]] = {}
    for ev in events:
        sid = ev.get("subject_id") or ""
        events_by_sid.setdefault(sid, []).append(ev)

    existing = _existing_autopsy_pairs(autopsies_path)
    produced: list[Autopsy] = []
    for p in predictions:
        ev = _match_prediction_to_event(p, events_by_sid)
        if ev is None:
            continue
        pair = (p.get("prediction_id") or "",
                  ev.get("event_id") or "")
        if pair in existing or not pair[0] or not pair[1]:
            continue
        autopsy = build_autopsy(p, ev)
        try:
            _append_autopsy(autopsy, autopsies_path)
            existing.add(pair)
            produced.append(autopsy)
        except OSError as exc:
            logger.error("belief_autopsy: write failed: %s", exc)
    return produced


# ── Pattern detection ─────────────────────────────────────────────


def detect_patterns(
    autopsies_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Aggregate Brier + family stats + sliding-window calibration flags
    from autopsies.jsonl. Returns a diagnostics dict.

    Suitable for belief-3 calibration dashboard once it ships.
    """
    all_rows = list(_iter_jsonl(autopsies_path or AUTOPSIES_PATH))
    # Exclude superseded rows (BUG-1 / future corrections) from pattern
    # detection so the calibration narrative reflects current truth.
    rows = [r for r in all_rows if not r.get("superseded_by")]
    n = len(rows)
    if n == 0:
        return {"n_autopsies": 0, "patterns": []}

    briers = [float(r.get("brier_component", 0.0)) for r in rows]
    mean_brier = sum(briers) / n

    # Per-family aggregation
    by_family: dict[str, list[float]] = {}
    for r in rows:
        fam = r.get("strategy_family") or "UNKNOWN"
        by_family.setdefault(fam, []).append(float(r.get("brier_component", 0.0)))
    family_brier = {
        fam: sum(bs) / len(bs) for fam, bs in by_family.items()
    }

    # Direction breakdown
    dir_counts: dict[str, int] = {}
    for r in rows:
        d = r.get("surprise_direction") or "neutral"
        dir_counts[d] = dir_counts.get(d, 0) + 1

    # Sliding-window calibration flag
    patterns: list[dict] = []
    if n >= MIN_AUTOPSIES_FOR_PATTERN:
        window = rows[-GREEN_OVERCONFIDENCE_WINDOW:]
        green_over = sum(
            1 for r in window
            if r.get("surprise_direction") == "over_predicted_green"
        )
        ratio = green_over / len(window)
        if ratio >= GREEN_OVERCONFIDENCE_THRESHOLD:
            patterns.append({
                "pattern":   "GREEN_OVERCONFIDENCE",
                "window":    len(window),
                "occurrences": green_over,
                "ratio":     ratio,
                "advice":    ("Recent dispatches over-predicted GREEN. "
                                "Tighten belief-1 family prior OR investigate "
                                "extractor drift (Sonnet stretching claims to "
                                "fit favorable templates)."),
            })

    return {
        "n_autopsies":         n,
        "mean_brier":          mean_brier,
        "family_brier":        family_brier,
        "direction_counts":    dir_counts,
        "patterns":            patterns,
    }
