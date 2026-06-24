"""engine.research.belief_track_record_rigor — statistical rigor pass on the belief track record.

Built 2026-06-22 (W6-rigor of six-week-critical-path, replacing
arxiv-draft after senior reframe). The Phase 3 track record
publishes raw aggregates (mean Brier 0.373, etc.). This module
adds the 6 statistical tests a senior reviewer (academic OR HF MD)
would ask for before accepting the calibration claim.

The 6 tests:

  T1. Bootstrap CI on overall Brier
      Q: is the observed 0.373 significantly < random baseline 4/9?
      method: 10000 resamples with replacement, percentile CI

  T2. Baseline comparison
      Q: does the LLM-driven predictor beat dumber baselines?
      baselines: always-MARGINAL / family-prior-no-LLM / uniform-random
      method: paired bootstrap on per-autopsy Brier deltas

  T3. Sign test on optimism bias
      Q: is the 7% over-predicted-green / 1% over-predicted-red split
         statistically systematic, or sample noise?
      method: binomial test on directional surprise counts

  T4. Per-family CI + Benjamini-Hochberg FDR correction
      Q: which family Brier differences are real vs multi-test artifact?
      method: bootstrap CI per family + BH-FDR across ~20 families at q=0.10

  T5. Time-series stability (Mann-Kendall trend test)
      Q: is calibration trending (learning / degrading) over time?
      method: non-parametric trend test on autopsy.ts-binned mean Brier

  T6. Reliability bin calibration (Hosmer-Lemeshow goodness-of-fit)
      Q: do predicted probabilities match observed frequencies?
      method: H-L chi-square over the 10-bin reliability diagram

All tests run on existing data/research/autopsies.jsonl. Zero LLM
calls; pure numpy/scipy. Anchors: Politis-Romano 1994 (bootstrap),
Benjamini-Hochberg 1995 (FDR), Mann 1945 / Kendall 1948 (trend),
Hosmer-Lemeshow 1980 (calibration GoF), Brier 1950 (the score itself).
"""
from __future__ import annotations

import collections
import dataclasses as _dc
import logging
import math
import random
from pathlib import Path
from typing import Any, Optional

from engine.research.belief_autopsy import AUTOPSIES_PATH, _iter_jsonl

logger = logging.getLogger(__name__)

# Try numpy/scipy; degrade gracefully if missing.
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Constants ────────────────────────────────────────────────────────

BASELINE_RANDOM_3CLASS = 4.0 / 9.0  # (1 - 1/3)^2
BOOTSTRAP_B = 10000
RNG_SEED = 42
BH_FDR_Q = 0.10
HL_BINS = 10
MIN_BIN_FOR_HL = 5    # H-L bins need ≥5 obs to be valid
MIN_FAMILY_N = 3      # don't bootstrap families with n<3 (CI uninformative)


def _load_autopsies(path: Optional[Path] = None) -> list[dict]:
    p = path or AUTOPSIES_PATH
    rows = []
    for row in _iter_jsonl(p):
        if row.get("superseded_by"):
            continue
        rows.append(row)
    return rows


def _load_predictions_by_id(
    predictions_path: Optional[Path] = None,
) -> dict[str, dict]:
    """Load predictions.jsonl, keyed by prediction_id."""
    from engine.research.belief_autopsy import PREDICTIONS_PATH
    p = predictions_path or PREDICTIONS_PATH
    out: dict[str, dict] = {}
    for row in _iter_jsonl(p):
        pid = row.get("prediction_id")
        if pid:
            out[pid] = row
    return out


def _load_verdict_events_by_id(
    events_path: Optional[Path] = None,
) -> dict[str, dict]:
    """Load events.jsonl, keep factor_verdict_filed events keyed by event_id."""
    from engine.research.belief_autopsy import EVENTS_PATH
    p = events_path or EVENTS_PATH
    out: dict[str, dict] = {}
    for row in _iter_jsonl(p):
        if row.get("event_type") != "factor_verdict_filed":
            continue
        eid = row.get("event_id")
        if eid:
            out[eid] = row
    return out


def _brier_components(autopsies: list[dict]) -> list[float]:
    return [float(r.get("brier_component", 0.0)) for r in autopsies]


# ── T1. Bootstrap CI on overall Brier ──────────────────────────────


def t1_bootstrap_overall_brier(
    autopsies: list[dict],
    *,
    B: int = BOOTSTRAP_B,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Bootstrap CI on mean Brier + significance vs random baseline."""
    briers = _brier_components(autopsies)
    n = len(briers)
    if n == 0:
        return {"n": 0}
    observed_mean = sum(briers) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(B):
        sample = [briers[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2)) - 1]
    # One-sided test: how often is bootstrap mean ≥ baseline?
    # If that fraction is < 0.05, observed mean is significantly below baseline.
    n_ge_baseline = sum(1 for m in means if m >= BASELINE_RANDOM_3CLASS)
    p_one_sided = n_ge_baseline / B
    return {
        "n":                       n,
        "observed_mean":           round(observed_mean, 6),
        "ci_95_lo":                round(lo, 6),
        "ci_95_hi":                round(hi, 6),
        "baseline_random_3class":  BASELINE_RANDOM_3CLASS,
        "p_one_sided_vs_baseline": round(p_one_sided, 6),
        "significantly_better":    p_one_sided < 0.05,
        "improvement_pct":         round((BASELINE_RANDOM_3CLASS - observed_mean)
                                         / BASELINE_RANDOM_3CLASS, 4),
    }


# ── T2. Baseline comparison ────────────────────────────────────────


def _baseline_always_marginal_brier(actual: str) -> float:
    """If we always predicted (0, 1, 0), Brier component = (1-1)^2 if MARGINAL else 1."""
    return 0.0 if actual == "MARGINAL" else 1.0


def _baseline_uniform_brier(_actual: str) -> float:
    """Uniform predictor (1/3,1/3,1/3) — Brier always (1 - 1/3)^2 = 4/9."""
    return BASELINE_RANDOM_3CLASS


def _baseline_family_prior_brier(
    autopsies: list[dict],
    target_idx: int,
) -> float:
    """Leave-one-out family-prior baseline: predict the family's empirical
    distribution (excluding the target autopsy) and score on the target."""
    target = autopsies[target_idx]
    fam = target.get("strategy_family") or "UNKNOWN"
    others = [
        a for i, a in enumerate(autopsies)
        if i != target_idx and a.get("strategy_family") == fam
    ]
    if not others:
        # No family history → fall back to uniform
        return _baseline_uniform_brier(target.get("actual_verdict") or "")
    # Empirical distribution from others
    counts: dict[str, int] = collections.Counter()
    for a in others:
        counts[a.get("actual_verdict") or ""] += 1
    total = sum(counts.values())
    dist = {
        "GREEN":    counts.get("GREEN", 0) / total,
        "MARGINAL": counts.get("MARGINAL", 0) / total,
        "RED":      counts.get("RED", 0) / total,
    }
    actual = target.get("actual_verdict") or ""
    p_actual = dist.get(actual, 0.0)
    return (1.0 - p_actual) ** 2


def t2_baseline_comparison(
    autopsies: list[dict],
    *,
    B: int = BOOTSTRAP_B,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Compare predictor Brier vs 3 baselines, paired bootstrap on deltas."""
    briers_predictor = _brier_components(autopsies)
    briers_marg     = [_baseline_always_marginal_brier(a.get("actual_verdict") or "")
                        for a in autopsies]
    briers_unif     = [_baseline_uniform_brier("") for _ in autopsies]
    briers_family   = [_baseline_family_prior_brier(autopsies, i)
                        for i in range(len(autopsies))]

    n = len(briers_predictor)
    out = {"n": n, "comparisons": []}
    if n == 0:
        return out

    rng = random.Random(seed)
    for name, baseline in [("always_marginal",  briers_marg),
                              ("uniform_random",   briers_unif),
                              ("family_prior_loo", briers_family)]:
        deltas = [briers_predictor[i] - baseline[i] for i in range(n)]
        mean_delta = sum(deltas) / n
        # Paired bootstrap on deltas
        boot_means = []
        for _ in range(B):
            sample = [deltas[rng.randrange(n)] for _ in range(n)]
            boot_means.append(sum(sample) / n)
        boot_means.sort()
        lo = boot_means[int(B * alpha / 2)]
        hi = boot_means[int(B * (1 - alpha / 2)) - 1]
        # H0: predictor no better than baseline → mean_delta ≥ 0.
        # Reject if upper CI < 0 (predictor strictly better).
        p_one_sided = sum(1 for m in boot_means if m >= 0) / B
        out["comparisons"].append({
            "baseline":                name,
            "predictor_mean_brier":    round(sum(briers_predictor) / n, 6),
            "baseline_mean_brier":     round(sum(baseline) / n, 6),
            "mean_delta":              round(mean_delta, 6),
            "delta_ci_95_lo":          round(lo, 6),
            "delta_ci_95_hi":          round(hi, 6),
            "p_one_sided":             round(p_one_sided, 6),
            "predictor_significantly_better": p_one_sided < 0.05,
        })
    return out


# ── T2-time-aware. Fair time-aware family-prior baseline ──────────


def t2_time_aware_family_prior(
    autopsies: list[dict],
    *,
    predictions_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
    seed: int = RNG_SEED,
    B: int = BOOTSTRAP_B,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Time-aware family-prior baseline.

    For each autopsy h:
      pred_ts_h = predictions_by_id[h.prediction_id].ts
      eligible_others = {h' in same family as h:
                            verdict_ts_of_h' < pred_ts_h}
      If eligible_others non-empty: build family prior from their actuals.
      Else: fall back to uniform (1/3, 1/3, 1/3).
      brier_h = (1 - p_eligible[actual_h])^2

    Compared (paired bootstrap on per-autopsy deltas) to the LLM
    predictor Brier. This is the FAIR version of T2 family-prior-LOO,
    which used the FULL sample (future info leakage).
    """
    import datetime as _dt

    preds_by_id = _load_predictions_by_id(predictions_path)
    events_by_id = _load_verdict_events_by_id(events_path)

    def _parse(ts: Optional[str]) -> Optional[_dt.datetime]:
        if not ts:
            return None
        try:
            return _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            return None

    n = len(autopsies)
    if n == 0:
        return {"n": 0}

    # Pre-resolve (pred_ts, verdict_ts) for every autopsy so we can
    # do family-LOO efficiently.
    enriched = []
    n_missing_pred_ts = 0
    n_missing_verdict_ts = 0
    for a in autopsies:
        pred = preds_by_id.get(a.get("prediction_id") or "")
        pred_ts = _parse((pred or {}).get("ts"))
        ev = events_by_id.get(a.get("verdict_event_id") or "")
        verdict_ts = _parse((ev or {}).get("ts"))
        if pred_ts is None:
            n_missing_pred_ts += 1
        if verdict_ts is None:
            n_missing_verdict_ts += 1
        enriched.append({
            "fam":        a.get("strategy_family") or "UNKNOWN",
            "actual":     a.get("actual_verdict") or "",
            "pred_ts":    pred_ts,
            "verdict_ts": verdict_ts,
            "llm_brier":  float(a.get("brier_component", 0.0)),
        })

    # Compute time-aware family-prior Brier per autopsy
    briers_time_aware: list[float] = []
    fallback_uniform_count = 0
    n_eligible_zero = 0
    for i, h in enumerate(enriched):
        if h["pred_ts"] is None:
            briers_time_aware.append(BASELINE_RANDOM_3CLASS)
            fallback_uniform_count += 1
            continue
        # Eligible: others in same family with verdict_ts strictly < pred_ts
        counts: dict[str, int] = {"GREEN": 0, "MARGINAL": 0, "RED": 0}
        n_elig = 0
        for j, other in enumerate(enriched):
            if j == i:
                continue
            if other["fam"] != h["fam"]:
                continue
            if other["verdict_ts"] is None:
                continue
            if other["verdict_ts"] < h["pred_ts"]:
                counts[other["actual"]] = counts.get(other["actual"], 0) + 1
                n_elig += 1
        if n_elig == 0:
            briers_time_aware.append(BASELINE_RANDOM_3CLASS)
            fallback_uniform_count += 1
            n_eligible_zero += 1
            continue
        # Empirical distribution
        total = sum(counts.values())
        dist = {k: v / total for k, v in counts.items()}
        p_actual = dist.get(h["actual"], 0.0)
        briers_time_aware.append((1.0 - p_actual) ** 2)

    briers_predictor = [h["llm_brier"] for h in enriched]
    deltas = [briers_predictor[i] - briers_time_aware[i] for i in range(n)]
    mean_delta = sum(deltas) / n

    rng = random.Random(seed)
    boot_means = []
    for _ in range(B):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(B * alpha / 2)]
    hi = boot_means[int(B * (1 - alpha / 2)) - 1]
    p_one_sided = sum(1 for m in boot_means if m >= 0) / B  # predictor better if delta<0

    return {
        "n":                              n,
        "predictor_mean_brier":           round(sum(briers_predictor) / n, 6),
        "time_aware_family_prior_brier":  round(sum(briers_time_aware) / n, 6),
        "mean_delta_predictor_minus_fp":  round(mean_delta, 6),
        "delta_ci_95_lo":                 round(lo, 6),
        "delta_ci_95_hi":                 round(hi, 6),
        "p_one_sided_predictor_better":   round(p_one_sided, 6),
        "predictor_significantly_better": p_one_sided < 0.05,
        "n_eligible_zero_fallback":       n_eligible_zero,
        "n_missing_pred_ts":              n_missing_pred_ts,
        "n_missing_verdict_ts":           n_missing_verdict_ts,
        "fallback_uniform_used":          fallback_uniform_count,
        "note": (
            "FAIR comparison: family-prior baseline uses only verdicts "
            "with verdict_ts < prediction_ts (no future info leakage). "
            "Compare to T2's family_prior_loo which used the full "
            "sample (optimistic baseline)."
        ),
    }


# ── A. Threshold/smoothing sweep (W6-rigor-A, 2026-06-22) ──────────


def sweep_threshold_alpha(
    autopsies: list[dict],
    *,
    thresholds: tuple[int, ...] = (1, 3, 5, 7, 9),
    alphas:     tuple[float, ...] = (0.5, 1.0, 3.0, 5.0),
    predictions_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Sweep (threshold, alpha) for the family-observed-posterior step
    of `belief.predict_verdict`. For each (N*, alpha) cell:

      For each autopsy h at prediction time T_h:
        eligible = {h' in same family: verdict_ts(h') < T_h}
        n_elig = |eligible|
        if n_elig >= N*:
          counts[GREEN/MARGINAL/RED] from eligible.actual_verdict
          dist[k] = (counts[k] + alpha) / (n_elig + 3*alpha)  # Dirichlet smoothing
        else:
          dist = uniform (1/3, 1/3, 1/3)  # conservative fallback
        Brier_h = (1 - dist[actual_h])^2
      mean_brier = mean(Brier_h over h)

    Returns a grid mean_brier[(N*, alpha)] + identifies argmin.

    NOTE: skips the n_trials penalty and publication-age decay steps —
    those apply uniformly across sweep cells so don't affect ordering.
    """
    import datetime as _dt

    preds_by_id = _load_predictions_by_id(predictions_path)
    events_by_id = _load_verdict_events_by_id(events_path)

    def _parse(ts: Optional[str]) -> Optional[_dt.datetime]:
        if not ts:
            return None
        try:
            return _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            return None

    enriched = []
    for a in autopsies:
        pred = preds_by_id.get(a.get("prediction_id") or "")
        pred_ts = _parse((pred or {}).get("ts"))
        ev = events_by_id.get(a.get("verdict_event_id") or "")
        verdict_ts = _parse((ev or {}).get("ts"))
        enriched.append({
            "fam":        a.get("strategy_family") or "UNKNOWN",
            "actual":     a.get("actual_verdict") or "",
            "pred_ts":    pred_ts,
            "verdict_ts": verdict_ts,
        })

    n = len(enriched)
    if n == 0:
        return {"n": 0}

    grid: list[dict] = []
    best: dict[str, Any] = {"mean_brier": float("inf")}
    for N_star in thresholds:
        for alpha in alphas:
            briers: list[float] = []
            n_fallback = 0
            for i, h in enumerate(enriched):
                if h["pred_ts"] is None:
                    briers.append(BASELINE_RANDOM_3CLASS)
                    n_fallback += 1
                    continue
                counts = {"GREEN": 0, "MARGINAL": 0, "RED": 0}
                n_elig = 0
                for j, other in enumerate(enriched):
                    if j == i or other["fam"] != h["fam"]:
                        continue
                    if other["verdict_ts"] is None:
                        continue
                    if other["verdict_ts"] < h["pred_ts"]:
                        if other["actual"] in counts:
                            counts[other["actual"]] += 1
                            n_elig += 1
                if n_elig < N_star:
                    briers.append(BASELINE_RANDOM_3CLASS)
                    n_fallback += 1
                    continue
                denom = n_elig + 3 * alpha
                dist = {k: (counts[k] + alpha) / denom for k in counts}
                p_actual = dist.get(h["actual"], 0.0)
                briers.append((1.0 - p_actual) ** 2)
            mean_b = sum(briers) / n
            cell = {
                "threshold_N":    N_star,
                "alpha":          alpha,
                "mean_brier":     round(mean_b, 6),
                "n_fallback":     n_fallback,
                "n_total":        n,
            }
            grid.append(cell)
            if mean_b < best["mean_brier"]:
                best = dict(cell)

    return {
        "n":              n,
        "grid":           grid,
        "best":           best,
        "current_production": {
            "threshold_N":    5,
            "alpha":          3.0,
            "comment":        "current belief.predict_verdict values",
        },
    }


# ── T3. Sign test on optimism bias ─────────────────────────────────


def t3_optimism_bias_sign_test(autopsies: list[dict]) -> dict[str, Any]:
    """Binomial sign test on over_predicted_green vs over_predicted_red."""
    counts = collections.Counter(a.get("surprise_direction") or "" for a in autopsies)
    over_green = counts.get("over_predicted_green", 0)
    over_red   = counts.get("over_predicted_red", 0)
    n_directional = over_green + over_red
    if n_directional == 0:
        return {"n_directional": 0, "test": "skipped"}

    # H0: P(over_green | directional) = 0.5
    # Two-sided binomial test
    if _HAS_SCIPY:
        try:
            res = _scipy_stats.binomtest(over_green, n_directional, p=0.5)
            p_two_sided = float(res.pvalue)
        except AttributeError:
            res = _scipy_stats.binom_test(over_green, n_directional, p=0.5)
            p_two_sided = float(res)
    else:
        # Manual exact binomial two-sided p-value
        from math import comb
        k_extreme = max(over_green, over_red)
        p_two_sided = 2 * sum(
            comb(n_directional, k) * (0.5 ** n_directional)
            for k in range(k_extreme, n_directional + 1)
        )
        p_two_sided = min(p_two_sided, 1.0)

    direction = ("optimistic"  if over_green > over_red else
                  "pessimistic" if over_red > over_green else "neutral")
    return {
        "n_directional":         n_directional,
        "over_predicted_green":  over_green,
        "over_predicted_red":    over_red,
        "p_two_sided":           round(p_two_sided, 6),
        "significant_at_0_05":   p_two_sided < 0.05,
        "direction":             direction,
    }


# ── T4. Per-family CI + Benjamini-Hochberg FDR ─────────────────────


def t4_per_family_with_fdr(
    autopsies: list[dict],
    *,
    B: int = 2000,  # smaller B per family to keep total runtime sane
    alpha: float = 0.05,
    fdr_q: float = BH_FDR_Q,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Per-family bootstrap CI vs baseline + BH-FDR across families."""
    by_family: dict[str, list[float]] = collections.defaultdict(list)
    for a in autopsies:
        fam = a.get("strategy_family") or "UNKNOWN"
        by_family[fam].append(float(a.get("brier_component", 0.0)))

    rng = random.Random(seed)
    family_results: list[dict] = []
    for fam, briers in by_family.items():
        n_fam = len(briers)
        if n_fam < MIN_FAMILY_N:
            family_results.append({
                "family":         fam,
                "n":              n_fam,
                "mean_brier":     round(sum(briers) / n_fam, 6),
                "ci_95_lo":       None,
                "ci_95_hi":       None,
                "p_one_sided":    None,
                "skipped_reason": f"n<{MIN_FAMILY_N}",
            })
            continue
        obs_mean = sum(briers) / n_fam
        boot_means = []
        for _ in range(B):
            sample = [briers[rng.randrange(n_fam)] for _ in range(n_fam)]
            boot_means.append(sum(sample) / n_fam)
        boot_means.sort()
        lo = boot_means[int(B * alpha / 2)]
        hi = boot_means[int(B * (1 - alpha / 2)) - 1]
        # H0: family Brier ≤ baseline (family is well-calibrated)
        # Reject if observed AND CI lower bound > baseline
        p = sum(1 for m in boot_means if m <= BASELINE_RANDOM_3CLASS) / B
        family_results.append({
            "family":      fam,
            "n":           n_fam,
            "mean_brier":  round(obs_mean, 6),
            "ci_95_lo":    round(lo, 6),
            "ci_95_hi":    round(hi, 6),
            # p = P(bootstrap mean ≤ baseline | data); small p = significantly WORSE
            "p_one_sided": round(p, 6),
        })

    # BH-FDR across families with valid p-values
    valid = [r for r in family_results if r["p_one_sided"] is not None]
    valid_sorted = sorted(valid, key=lambda r: r["p_one_sided"])
    m = len(valid_sorted)
    # BH: largest k such that p_(k) ≤ k/m * q
    significant_set: set[str] = set()
    if m > 0:
        for k, r in enumerate(valid_sorted, 1):
            if r["p_one_sided"] <= (k / m) * fdr_q:
                significant_set.add(r["family"])
    for r in family_results:
        r["fdr_significant_worse_than_baseline"] = (r["family"] in significant_set)

    family_results.sort(key=lambda r: (-r["n"], r["mean_brier"]))
    return {
        "n_families_total": len(family_results),
        "n_families_valid": m,
        "fdr_q":            fdr_q,
        "families":         family_results,
    }


# ── T5. Time-series stability (Mann-Kendall) ───────────────────────


def _mann_kendall(values: list[float]) -> dict[str, Any]:
    """Mann-Kendall trend test on a sequence. Returns S statistic + p-value."""
    n = len(values)
    if n < 4:
        return {"n": n, "test": "skipped_n_too_small"}
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if values[j] > values[i]:
                s += 1
            elif values[j] < values[i]:
                s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    # Two-sided normal-approx p-value
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {
        "n":       n,
        "s":       s,
        "z":       round(z, 4),
        "p_two_sided": round(p, 6),
        "trend":   ("increasing" if s > 0 and p < 0.05 else
                     "decreasing" if s < 0 and p < 0.05 else
                     "no_trend"),
    }


def t5_time_series_stability(autopsies: list[dict]) -> dict[str, Any]:
    """Bin autopsies into weekly buckets by ts; run Mann-Kendall on mean Brier."""
    import datetime as _dt
    by_week: dict[str, list[float]] = collections.defaultdict(list)
    for a in autopsies:
        ts = a.get("ts", "")
        try:
            d = _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            continue
        iso_year, iso_week, _ = d.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        by_week[key].append(float(a.get("brier_component", 0.0)))
    weeks_sorted = sorted(by_week.keys())
    weekly_means = [sum(by_week[w]) / len(by_week[w]) for w in weeks_sorted]
    mk = _mann_kendall(weekly_means)
    return {
        "n_weeks":      len(weeks_sorted),
        "weekly_means": [
            {"week": w, "n": len(by_week[w]),
             "mean_brier": round(sum(by_week[w]) / len(by_week[w]), 6)}
            for w in weeks_sorted
        ],
        "mann_kendall": mk,
    }


# ── T6. Hosmer-Lemeshow calibration GoF ────────────────────────────


def t6_hosmer_lemeshow(autopsies: list[dict],
                          *, bins: int = HL_BINS) -> dict[str, Any]:
    """H-L chi-square on modal-class confidence vs observed correct-rate."""
    rows = []
    for a in autopsies:
        dist = a.get("predicted_verdict_dist") or {}
        if not dist:
            continue
        modal_label = max(dist.items(), key=lambda kv: kv[1])[0]
        p_modal = float(dist[modal_label])
        correct = 1 if modal_label == a.get("actual_verdict") else 0
        rows.append((p_modal, correct))
    if not rows:
        return {"n": 0, "test": "skipped"}
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    # Equal-frequency bins
    bin_size = max(1, n // bins)
    chi2 = 0.0
    df = 0
    bin_details: list[dict] = []
    for i in range(bins):
        lo = i * bin_size
        hi = n if i == bins - 1 else (i + 1) * bin_size
        bucket = rows[lo:hi]
        if len(bucket) < MIN_BIN_FOR_HL:
            bin_details.append({
                "bin": i + 1, "n": len(bucket), "skipped": True,
            })
            continue
        mean_p   = sum(r[0] for r in bucket) / len(bucket)
        obs_pos  = sum(r[1] for r in bucket)
        exp_pos  = mean_p * len(bucket)
        obs_neg  = len(bucket) - obs_pos
        exp_neg  = (1 - mean_p) * len(bucket)
        if exp_pos > 0:
            chi2 += (obs_pos - exp_pos) ** 2 / exp_pos
        if exp_neg > 0:
            chi2 += (obs_neg - exp_neg) ** 2 / exp_neg
        df += 1
        bin_details.append({
            "bin":       i + 1,
            "n":         len(bucket),
            "mean_p":    round(mean_p, 4),
            "observed_correct": obs_pos,
            "expected_correct": round(exp_pos, 2),
        })
    # H-L df = bins - 2 typically; here use df = #bins_used - 2
    hl_df = max(1, df - 2)
    if _HAS_SCIPY:
        p_value = 1 - _scipy_stats.chi2.cdf(chi2, hl_df)
    else:
        p_value = None
    return {
        "n":         n,
        "bins_used": df,
        "chi2":      round(chi2, 4),
        "df":        hl_df,
        "p_value":   round(p_value, 6) if p_value is not None else None,
        # H0: model is calibrated → fail-to-reject if p > 0.05
        "calibrated_fail_to_reject_h0_at_0_05": (
            p_value > 0.05 if p_value is not None else None
        ),
        "bin_details": bin_details,
    }


# ── Orchestrator ─────────────────────────────────────────────────────


def run_all_rigor_tests(
    autopsies_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run T1..T6 and return a single dict suitable for JSON dump + report."""
    autopsies = _load_autopsies(autopsies_path)
    return {
        "n_autopsies": len(autopsies),
        "T1_overall_brier_bootstrap":     t1_bootstrap_overall_brier(autopsies),
        "T2_baseline_comparison":         t2_baseline_comparison(autopsies),
        "T2_time_aware_family_prior":     t2_time_aware_family_prior(autopsies),
        "A_threshold_alpha_sweep":        sweep_threshold_alpha(autopsies),
        "T3_optimism_bias_sign_test":     t3_optimism_bias_sign_test(autopsies),
        "T4_per_family_fdr":              t4_per_family_with_fdr(autopsies),
        "T5_time_series_stability":       t5_time_series_stability(autopsies),
        "T6_hosmer_lemeshow":             t6_hosmer_lemeshow(autopsies),
    }
