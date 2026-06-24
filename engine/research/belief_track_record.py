"""engine.research.belief_track_record — calibration aggregator over autopsies.

Phase 3 (calibration surface) of the 5-phase belief layer, built 2026-06-18.
Pure aggregation; no LLM, no I/O beyond loading autopsies.jsonl.

What this answers (the 3 senior questions):
  1. Is the system calibrated?  → overall Brier + direction breakdown +
                                    comparison vs naive baselines
  2. Is it improving over time? → sliding-window Brier trend
  3. Where does it fail?        → per-family Brier rank +
                                    per-prediction-confidence reliability bins

Baselines for context (3-class GREEN/MARGINAL/RED Brier component):
  - Random uniform (1/3, 1/3, 1/3):  Brier = (1 - 1/3)^2 = 0.444
  - Always-MARGINAL (modal class):     Brier = depends on actual mix
  - Perfect calibration:               Brier = 0
  Lower is better.

Reliability diagram bins answer: "When the predictor says 60% GREEN, does
the actual GREEN rate match?" Standard 10 bins on the modal-class probability.

Anchors:
  - Brier 1950 — Brier score original definition
  - Tetlock 2015 "Superforecasting" Ch.5 — calibration vs resolution
  - López de Prado AFML Ch.13 — track record interpretation
"""
from __future__ import annotations

import collections
import dataclasses as _dc
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any, Optional

from engine.research.belief_autopsy import (
    AUTOPSIES_PATH,
    _iter_jsonl,
)

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────

BASELINE_RANDOM_3CLASS = 4.0 / 9.0  # (1 - 1/3)^2 exactly; ≈0.4444
BASELINE_PERFECT = 0.0
WINDOW_DAYS_SHORT = 30
WINDOW_DAYS_LONG = 90
RELIABILITY_BINS = 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0


def _parse_ts(ts: str) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return None


def _load_autopsies(path: Optional[Path] = None) -> list[dict]:
    p = path or AUTOPSIES_PATH
    out = []
    for row in _iter_jsonl(p):
        if row.get("superseded_by"):
            continue  # exclude superseded corrections from track record
        out.append(row)
    return out


def _window_brier(
    autopsies: list[dict],
    *,
    now: _dt.datetime,
    window_days: int,
) -> dict[str, Any]:
    cutoff = now - _dt.timedelta(days=window_days)
    rows = [
        r for r in autopsies
        if (ts := _parse_ts(r.get("ts", ""))) is not None and ts >= cutoff
    ]
    if not rows:
        return {"n": 0, "mean_brier": None, "window_days": window_days}
    briers = [float(r.get("brier_component", 0.0)) for r in rows]
    return {
        "n":           len(rows),
        "mean_brier":  sum(briers) / len(briers),
        "window_days": window_days,
    }


def _family_breakdown(autopsies: list[dict]) -> list[dict]:
    by_family: dict[str, list[float]] = collections.defaultdict(list)
    for r in autopsies:
        fam = r.get("strategy_family") or "UNKNOWN"
        by_family[fam].append(float(r.get("brier_component", 0.0)))
    rows = [
        {
            "family":     fam,
            "n":          len(bs),
            "mean_brier": sum(bs) / len(bs),
        }
        for fam, bs in by_family.items()
    ]
    rows.sort(key=lambda r: (-r["n"], r["mean_brier"]))
    return rows


def _direction_breakdown(autopsies: list[dict]) -> dict[str, Any]:
    counts: dict[str, int] = collections.Counter()
    for r in autopsies:
        counts[r.get("surprise_direction") or "neutral"] += 1
    total = sum(counts.values())
    return {
        "counts":   dict(counts),
        "fractions": {k: round(v / total, 4) for k, v in counts.items()} if total else {},
        "total":    total,
    }


def _reliability_diagram(autopsies: list[dict]) -> list[dict]:
    """Standard reliability bins on the predicted-modal-class probability.

    For each autopsy, take p = max(predicted_verdict_dist) — the predictor's
    confidence in its own modal prediction. Bin by p in 10 buckets.
    For each bin, compute observed fraction-correct (= fraction where
    predicted modal == actual_verdict).

    Perfectly calibrated → observed_fraction in bin [p_lo, p_hi] sits
    near the bin midpoint. Above midpoint = under-confident; below =
    over-confident.
    """
    buckets: list[list[dict]] = [[] for _ in range(RELIABILITY_BINS)]
    for r in autopsies:
        dist = r.get("predicted_verdict_dist") or {}
        if not dist:
            continue
        modal_label = max(dist.items(), key=lambda kv: kv[1])[0]
        p_modal = float(dist[modal_label])
        idx = min(int(p_modal * RELIABILITY_BINS), RELIABILITY_BINS - 1)
        buckets[idx].append({
            "p_modal":    p_modal,
            "correct":    1 if modal_label == r.get("actual_verdict") else 0,
        })
    out = []
    for i, bucket in enumerate(buckets):
        lo = i / RELIABILITY_BINS
        hi = (i + 1) / RELIABILITY_BINS
        if not bucket:
            out.append({
                "bin_lo":             round(lo, 2),
                "bin_hi":             round(hi, 2),
                "n":                  0,
                "mean_p_modal":       None,
                "observed_correct":    None,
            })
            continue
        ps = [b["p_modal"] for b in bucket]
        cs = [b["correct"] for b in bucket]
        out.append({
            "bin_lo":            round(lo, 2),
            "bin_hi":            round(hi, 2),
            "n":                 len(bucket),
            "mean_p_modal":      round(sum(ps) / len(ps), 4),
            "observed_correct":  round(sum(cs) / len(cs), 4),
        })
    return out


def build_track_record(
    autopsies_path: Optional[Path] = None,
    now: Optional[_dt.datetime] = None,
) -> dict[str, Any]:
    """Compute the full track-record dict from autopsies.jsonl.

    Returns:
      {
        as_of:              ISO timestamp,
        n_autopsies:        int,
        mean_brier_overall: float,
        baselines:          {random_3class: 0.444, perfect: 0.0},
        windows:            [{window_days, n, mean_brier}, ...],
        family_breakdown:   [{family, n, mean_brier}, ...] sorted desc by n,
        direction:          {counts, fractions, total},
        reliability:        [{bin_lo, bin_hi, n, mean_p_modal, observed_correct}, ...],
      }
    """
    autopsies = _load_autopsies(autopsies_path)
    if now is None:
        now = _dt.datetime.utcnow()
    if not autopsies:
        return {
            "as_of":              now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_autopsies":        0,
            "mean_brier_overall": None,
            "baselines":          {"random_3class": BASELINE_RANDOM_3CLASS,
                                    "perfect":       BASELINE_PERFECT},
            "windows":            [],
            "family_breakdown":   [],
            "direction":          {"counts": {}, "fractions": {}, "total": 0},
            "reliability":        [],
        }
    briers = [float(r.get("brier_component", 0.0)) for r in autopsies]
    return {
        "as_of":              now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_autopsies":        len(autopsies),
        "mean_brier_overall": round(sum(briers) / len(briers), 6),
        "baselines":          {"random_3class": BASELINE_RANDOM_3CLASS,
                                "perfect":       BASELINE_PERFECT},
        "windows":            [
            _window_brier(autopsies, now=now, window_days=WINDOW_DAYS_SHORT),
            _window_brier(autopsies, now=now, window_days=WINDOW_DAYS_LONG),
        ],
        "family_breakdown":   _family_breakdown(autopsies),
        "direction":          _direction_breakdown(autopsies),
        "reliability":        _reliability_diagram(autopsies),
    }
