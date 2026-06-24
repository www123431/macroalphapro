"""engine.research.belief_ensemble_sweep — per-family LLM × family-prior ensemble.

Built 2026-06-22 (W7-arxiv-v05). $0 LLM. Tests the architectural improvement
implied by W6-rigor's Section 4.3 finding (LLM-driven predictor loses to fair
family-prior baseline by 0.149 Brier): can a per-family weighted mix beat
both standalone?

Ensemble: predicted_dist = w_fam × family_prior_dist + (1 - w_fam) × llm_prior_dist
where family_prior_dist is the time-aware leave-one-out empirical distribution
(only verdicts realized BEFORE this prediction's pred_ts are included).

Sweep:
  - w_fam ∈ {0.0, 0.1, ..., 1.0}: 11 values
  - For each family with n >= 5: find argmin mean-Brier over family members
  - For families with n < 5: report only n; recommend a global w
  - Output: optimal w per family + new ensemble Brier estimate

Anchors:
  - López de Prado AFML Ch.4 — ensemble prediction
  - Tetlock 2015 — when to defer to base-rate (family-empirical = base-rate
    for quant verdicts)
"""
from __future__ import annotations

import collections
import dataclasses as _dc
import datetime as _dt
from pathlib import Path
from typing import Optional

from engine.research.belief_autopsy import AUTOPSIES_PATH, _iter_jsonl
from engine.research.belief_track_record_rigor import (
    BASELINE_RANDOM_3CLASS,
    _load_autopsies,
    _load_predictions_by_id,
    _load_verdict_events_by_id,
)

MIN_FAMILY_N_FOR_OPTIMUM = 5  # below this, use global w_fam
W_GRID = tuple(round(0.1 * i, 1) for i in range(11))  # 0.0, 0.1, ..., 1.0
MIN_FAMILY_N_FOR_LOOCV = 3    # n>=3 needed to leave-one-out on family-prior side too


def _parse_ts(ts: Optional[str]) -> Optional[_dt.datetime]:
    if not ts:
        return None
    try:
        return _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return None


def _build_enriched(
    autopsies: list[dict],
    predictions_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
) -> list[dict]:
    """For each autopsy, attach pred_ts + verdict_ts + LLM-prior dist.

    Returns a list of dicts with keys:
      family, actual, pred_ts, verdict_ts, llm_dist (dict over GREEN/MARG/RED).
    """
    preds = _load_predictions_by_id(predictions_path)
    events = _load_verdict_events_by_id(events_path)
    out = []
    for a in autopsies:
        pred = preds.get(a.get("prediction_id") or "")
        ev = events.get(a.get("verdict_event_id") or "")
        llm_dist = a.get("predicted_verdict_dist") or {}
        # Normalize keys
        llm_dist = {
            "GREEN":    float(llm_dist.get("GREEN", 0.0)),
            "MARGINAL": float(llm_dist.get("MARGINAL", 0.0)),
            "RED":      float(llm_dist.get("RED", 0.0)),
        }
        out.append({
            "fam":        a.get("strategy_family") or "UNKNOWN",
            "actual":     a.get("actual_verdict") or "",
            "pred_ts":    _parse_ts((pred or {}).get("ts")),
            "verdict_ts": _parse_ts((ev or {}).get("ts")),
            "llm_dist":   llm_dist,
        })
    return out


def _time_aware_family_dist(
    enriched: list[dict],
    target_idx: int,
) -> Optional[dict[str, float]]:
    """Time-aware leave-one-out family-empirical dist for the target autopsy.

    Returns None if no eligible neighbors (caller falls back to uniform).
    """
    target = enriched[target_idx]
    if target["pred_ts"] is None:
        return None
    counts = {"GREEN": 0, "MARGINAL": 0, "RED": 0}
    n_elig = 0
    for j, other in enumerate(enriched):
        if j == target_idx or other["fam"] != target["fam"]:
            continue
        if other["verdict_ts"] is None:
            continue
        if other["verdict_ts"] < target["pred_ts"]:
            if other["actual"] in counts:
                counts[other["actual"]] += 1
                n_elig += 1
    if n_elig == 0:
        return None
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def _brier_component(dist: dict[str, float], actual: str) -> float:
    p_actual = dist.get(actual, 0.0)
    return (1.0 - p_actual) ** 2


def sweep_ensemble_per_family(
    autopsies_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
) -> dict:
    """Find per-family optimal w_fam minimizing mean Brier.

    Returns a structured dict with:
      n_total, by_family: {family: {n, n_eligible, optimal_w, optimal_brier,
                                      llm_only_brier, family_only_brier}},
      global_summary
    """
    autopsies = _load_autopsies(autopsies_path)
    enriched = _build_enriched(autopsies, predictions_path, events_path)
    n_total = len(enriched)

    # Compute family_dist + per-w Brier table per autopsy
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for i, h in enumerate(enriched):
        fam_dist = _time_aware_family_dist(enriched, i)
        actual = h["actual"]
        llm_dist = h["llm_dist"]
        # Brier per w in grid
        brier_by_w = {}
        for w in W_GRID:
            if fam_dist is None:
                # No eligible family history → ensemble degenerates to pure LLM
                # (use that for all w to keep accounting consistent;
                # caller separates these cases)
                mixed = llm_dist
            else:
                mixed = {
                    k: w * fam_dist[k] + (1 - w) * llm_dist[k]
                    for k in ("GREEN", "MARGINAL", "RED")
                }
            brier_by_w[w] = _brier_component(mixed, actual)
        by_family[h["fam"]].append({
            "idx":         i,
            "actual":      actual,
            "has_family":  fam_dist is not None,
            "brier_by_w":  brier_by_w,
        })

    # Per-family summary
    family_results = {}
    for fam, rows in by_family.items():
        n_fam = len(rows)
        n_eligible = sum(1 for r in rows if r["has_family"])
        # Mean Brier as function of w (averaged over all rows in family)
        mean_brier_by_w = {}
        for w in W_GRID:
            mean_brier_by_w[w] = (
                sum(r["brier_by_w"][w] for r in rows) / n_fam if n_fam else 0
            )
        # Optimal w
        w_star = min(mean_brier_by_w, key=mean_brier_by_w.get)
        optimal_brier = mean_brier_by_w[w_star]
        llm_only_brier = mean_brier_by_w[0.0]
        family_only_brier = mean_brier_by_w[1.0]

        family_results[fam] = {
            "n":                  n_fam,
            "n_eligible":         n_eligible,
            "optimal_w":          w_star,
            "optimal_brier":      round(optimal_brier, 6),
            "llm_only_brier":     round(llm_only_brier, 6),
            "family_only_brier":  round(family_only_brier, 6),
            "improvement_vs_llm": round(llm_only_brier - optimal_brier, 6),
            "use_family_specific": n_fam >= MIN_FAMILY_N_FOR_OPTIMUM,
            "mean_brier_curve":    {str(w): round(b, 6)
                                      for w, b in mean_brier_by_w.items()},
        }

    # Global: per-autopsy use family-specific w if available, else global w
    # Find global w (over all autopsies) for fallback
    global_brier_by_w = {}
    for w in W_GRID:
        total = sum(r["brier_by_w"][w]
                      for rows in by_family.values() for r in rows)
        global_brier_by_w[w] = round(total / n_total if n_total else 0, 6)
    global_w = min(global_brier_by_w, key=global_brier_by_w.get)

    # Final ensemble Brier: each autopsy uses family-specific w if family
    # has n >= MIN_FAMILY_N_FOR_OPTIMUM, else global_w
    final_brier_sum = 0.0
    for fam, rows in by_family.items():
        use_specific = family_results[fam]["use_family_specific"]
        w_to_use = family_results[fam]["optimal_w"] if use_specific else global_w
        for r in rows:
            final_brier_sum += r["brier_by_w"][w_to_use]
    final_brier = round(final_brier_sum / n_total, 6) if n_total else 0

    # Baselines for comparison
    llm_only_global = global_brier_by_w[0.0]
    family_only_global = global_brier_by_w[1.0]

    return {
        "n_total":              n_total,
        "n_families_total":     len(family_results),
        "n_families_eligible":  sum(1 for f in family_results.values()
                                      if f["use_family_specific"]),
        "global_w_fallback":    global_w,
        "global_brier_curve":   global_brier_by_w,
        "final_ensemble_brier": final_brier,
        "llm_only_brier":       round(llm_only_global, 6),
        "family_only_brier":    round(family_only_global, 6),
        "improvement_vs_llm":   round(llm_only_global - final_brier, 6),
        "improvement_vs_fam":   round(family_only_global - final_brier, 6),
        "by_family":            family_results,
    }


def loocv_ensemble_brier(
    autopsies_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
) -> dict:
    """Leave-one-out cross-validation on per-family w_fam optimization.

    For each autopsy h:
      1. Drop h from the dataset.
      2. For h's family, find optimal w_fam on the OTHER members
         (in-sample on the remaining, OOS for h).
      3. Compute h's Brier using that w_fam blend.
    Aggregate: mean Brier across all 92 holdouts.

    Compares to: the in-sample sweep's predicted 0.246.

    Families with n < MIN_FAMILY_N_FOR_LOOCV (3) skip LOOCV and use
    the GLOBAL_W_FALLBACK from the in-sample sweep (single fallback
    so caller can compare consistently).

    Returns: {n_total, loocv_brier, in_sample_brier (sweep),
              llm_only_brier, family_only_brier, by_family}
    """
    autopsies = _load_autopsies(autopsies_path)
    enriched = _build_enriched(autopsies, predictions_path, events_path)

    # First do the full in-sample sweep to get FAMILY_OPTIMAL_W as
    # ground-truth reference + the global_w fallback
    in_sample = sweep_ensemble_per_family(
        autopsies_path, predictions_path, events_path,
    )
    in_sample_global_w = in_sample["global_w_fallback"]

    # Pre-compute family-prior dist for every autopsy (cached, $0)
    fam_dists = []
    for i in range(len(enriched)):
        fam_dists.append(_time_aware_family_dist(enriched, i))

    # For each autopsy, find leave-one-out optimal w_fam on its family
    # and apply that to score the held-out
    loocv_briers = []
    family_loocv_summary: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "mean_brier": 0.0, "briers": []}
    )

    # Index autopsies by family
    by_family_idx: dict[str, list[int]] = collections.defaultdict(list)
    for i, h in enumerate(enriched):
        by_family_idx[h["fam"]].append(i)

    for i, h in enumerate(enriched):
        fam = h["fam"]
        actual = h["actual"]
        llm_dist = h["llm_dist"]
        same_family_others = [j for j in by_family_idx[fam] if j != i]

        # Decide w to use for this holdout
        if len(same_family_others) < MIN_FAMILY_N_FOR_LOOCV - 1:
            # Not enough family neighbors to compute LOO w; use global fallback
            w_used = in_sample_global_w
        else:
            # Find w_fam minimizing Brier on the OTHER same-family members
            best_w, best_score = 0.0, float("inf")
            for w in W_GRID:
                tot = 0.0
                cnt = 0
                for j in same_family_others:
                    other = enriched[j]
                    fd = fam_dists[j]
                    if fd is None:
                        # treat as pure LLM contribution
                        mixed_j = other["llm_dist"]
                    else:
                        mixed_j = {
                            k: w * fd[k] + (1 - w) * other["llm_dist"][k]
                            for k in ("GREEN", "MARGINAL", "RED")
                        }
                    tot += _brier_component(mixed_j, other["actual"])
                    cnt += 1
                if cnt == 0:
                    continue
                m = tot / cnt
                if m < best_score:
                    best_score = m
                    best_w = w
            w_used = best_w

        # Score the held-out h with w_used
        fd_h = fam_dists[i]
        if fd_h is None:
            mixed_h = llm_dist
        else:
            mixed_h = {
                k: w_used * fd_h[k] + (1 - w_used) * llm_dist[k]
                for k in ("GREEN", "MARGINAL", "RED")
            }
        b = _brier_component(mixed_h, actual)
        loocv_briers.append({
            "idx":      i,
            "fam":      fam,
            "actual":   actual,
            "w_used":   w_used,
            "brier":    b,
        })
        family_loocv_summary[fam]["n"] += 1
        family_loocv_summary[fam]["briers"].append(b)

    # Aggregate
    for fam in family_loocv_summary:
        bs = family_loocv_summary[fam]["briers"]
        family_loocv_summary[fam]["mean_brier"] = (
            round(sum(bs) / len(bs), 6) if bs else None
        )
        del family_loocv_summary[fam]["briers"]

    loocv_mean = round(sum(b["brier"] for b in loocv_briers) /
                          len(loocv_briers), 6) if loocv_briers else None

    return {
        "n_total":            len(loocv_briers),
        "loocv_brier":        loocv_mean,
        "in_sample_brier":    in_sample["final_ensemble_brier"],
        "llm_only_brier":     in_sample["llm_only_brier"],
        "family_only_brier":  in_sample["family_only_brier"],
        "overfit_gap":        round(loocv_mean - in_sample["final_ensemble_brier"], 6),
        "by_family":          dict(family_loocv_summary),
        "improvement_vs_llm": round(in_sample["llm_only_brier"] - loocv_mean, 6),
    }
