"""
engine/stats.py — Deterministic Statistical Helpers
====================================================
Created 2026-05-05 (A-1 thesis-grade polish): extracted from engine/lcs.py
to decouple statistical helpers from the deprecated LCS module.

Contents:
  • _MIN_N_PERMUTATION — sample-size gate for block-bootstrap permutation
  • PermutationResult  — structured result dataclass
  • compute_permutation_p_value() — block-bootstrap permutation test
  • bonferroni_adjusted_threshold() — multi-comparison threshold adjustment

These functions are deterministic (Layer 2 evaluation, no LLM per
docs/decisions/llm_3layer_architecture_2026-05-05.md §3 Invariant 1).

Used by:
  - engine/memory.py — verify_pending_decisions historical aggregation
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Sample-size gate ──────────────────────────────────────────────────────────
_MIN_N_PERMUTATION: int = 100


# ── Statistical Gate: Block Bootstrap Permutation ────────────────────────────

@dataclass
class PermutationResult:
    """
    Structured result of a block-bootstrap permutation test for one sector × regime cell.

    Three mutually exclusive status values:
      "insufficient_data" — n < _MIN_N_PERMUTATION. p_value is None.
                            Do NOT interpret this as "not significant".
                            The test was not run, not failed.
      "not_significant"   — n >= threshold, p >= adjusted_threshold.
                            Signal is indistinguishable from noise at this sample size.
      "significant"       — n >= threshold, p < adjusted_threshold.
                            Observed accuracy beats the null distribution after
                            multiple-test correction.
    """
    status:            str              # "insufficient_data" | "not_significant" | "significant"
    p_value:           Optional[float]  # None when status == "insufficient_data"
    observed_accuracy: float
    n_samples:         int
    n_needed:          int              # = _MIN_N_PERMUTATION
    adjusted_threshold: float
    sector:            str = ""
    regime:            str = ""

    @property
    def passed(self) -> bool:
        return self.status == "significant"

    @property
    def progress_pct(self) -> float:
        """How far to the minimum sample threshold (0.0 → 1.0)."""
        return min(1.0, self.n_samples / max(1, self.n_needed))


def compute_permutation_p_value(
    accuracy_scores:    list[float],
    n_permutations:     int   = 10_000,
    block_size:         int   = 4,
    adjusted_threshold: float = 0.003,
    sector:             str   = "",
    regime:             str   = "",
) -> PermutationResult:
    """
    Block-bootstrap permutation test for signal significance.

    Null hypothesis: observed accuracy is no better than random shuffling of
    outcomes within contiguous time-blocks (preserves autocorrelation structure).

    Parameters
    ----------
    accuracy_scores    : 0.0 / 0.5 / 0.75 / 1.0 scores, chronologically ordered
    n_permutations     : permutation draws (≥10_000 for stable p near 0.003)
    block_size         : contiguous block length; 4 = quarterly (recommended)
    adjusted_threshold : per-test p-value threshold after multiple-test correction
                         Use bonferroni_adjusted_threshold() to compute this.
    sector / regime    : labels carried through for display purposes

    Returns
    -------
    PermutationResult with status in {"insufficient_data", "not_significant", "significant"}
    """
    n = len(accuracy_scores)
    observed_mean = sum(accuracy_scores) / n if n > 0 else 0.0

    if n < _MIN_N_PERMUTATION:
        return PermutationResult(
            status             = "insufficient_data",
            p_value            = None,
            observed_accuracy  = observed_mean,
            n_samples          = n,
            n_needed           = _MIN_N_PERMUTATION,
            adjusted_threshold = adjusted_threshold,
            sector             = sector,
            regime             = regime,
        )

    beats = 0
    for _ in range(n_permutations):
        permuted: list[float] = []
        while len(permuted) < n:
            start = random.randint(0, max(0, n - block_size))
            permuted.extend(accuracy_scores[start : start + block_size])
        perm_mean = sum(permuted[:n]) / n
        if perm_mean >= observed_mean:
            beats += 1

    p_value = beats / n_permutations
    status  = "significant" if p_value < adjusted_threshold else "not_significant"

    logger.debug(
        "Permutation test: sector=%s regime=%s n=%d obs=%.3f p=%.4f status=%s",
        sector, regime, n, observed_mean, p_value, status,
    )
    return PermutationResult(
        status             = status,
        p_value            = p_value,
        observed_accuracy  = observed_mean,
        n_samples          = n,
        n_needed           = _MIN_N_PERMUTATION,
        adjusted_threshold = adjusted_threshold,
        sector             = sector,
        regime             = regime,
    )


def bonferroni_adjusted_threshold(
    base_alpha: float = 0.05,
    n_tests:    int   = 16,
    method:     str   = "romano_wolf",
) -> float:
    """
    Return the adjusted per-test significance threshold for multiple comparisons.

    Romano-Wolf stepdown is recommended for correlated sector tests: it is more
    powerful than Bonferroni when hypotheses share common factor exposure.

    Parameters
    ----------
    base_alpha : family-wise error rate (default 0.05)
    n_tests    : number of simultaneous tests (default 16 sectors)
    method     : "bonferroni" | "romano_wolf" (approximate first-step)

    Returns
    -------
    threshold : per-test p-value cutoff
    """
    if method == "bonferroni":
        return base_alpha / n_tests
    return base_alpha / (n_tests * 0.85)
