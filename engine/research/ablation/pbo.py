"""engine.research.ablation.pbo — Probability of Backtest Overfitting.

Bailey-Borwein-Lopez de Prado-Zhu 2014 "The Probability of Backtest
Overfitting" (Journal of Computational Finance).

Method:
  Given N backtest paths (one per CPCV split) × K strategies (variants),
  compute PBO as the probability that the IN-SAMPLE best strategy ranks
  in the bottom half OUT-OF-SAMPLE.

  PBO = (1/N) × Σ I[rank_OOS(s*_IS_path_i) ≤ K/2]

  where s*_IS_path_i is the strategy with best IS Sharpe on the
  complement-of-path-i, evaluated OOS on path_i.

  Low PBO (< 0.5) → IS performance generalizes;
  High PBO (> 0.5) → IS performance is noise, doesn't generalize.

Output also includes the "logit-PBO" used in plots: logit(PBO) = log(PBO/(1-PBO))

References:
  - Bailey et al 2014 JCF (the canonical paper)
  - Lopez de Prado 2018 §11
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_pbo(
    is_sharpes_matrix:  pd.DataFrame,   # rows = paths, cols = strategies
    oos_sharpes_matrix: pd.DataFrame,   # same shape, same labels
) -> dict:
    """Compute PBO + logit-PBO + per-path winners.

    For each row (path):
      - find argmax IS strategy
      - look up its OOS Sharpe
      - rank that OOS Sharpe among all strategies' OOS for this path
      - record fraction of rows where rank ≤ K/2 (= worse than median OOS)

    PBO = fraction of "IS winner ends up below median OOS".

    Returns dict with:
      pbo:                  the PBO value [0, 1]
      logit_pbo:            log(pbo / (1 - pbo)) for plotting
      n_paths:              number of paths used
      n_strategies:         K
      paths_winner_oos_rank: list of OOS ranks of IS-winners (0-indexed)
      is_winner_indices:    list of which strategy was IS-winner per path
    """
    assert is_sharpes_matrix.shape == oos_sharpes_matrix.shape, \
        "IS and OOS matrices must have same shape"
    if is_sharpes_matrix.empty:
        return {"pbo": float("nan"), "logit_pbo": float("nan"),
                "n_paths": 0, "n_strategies": 0,
                "paths_winner_oos_rank": [], "is_winner_indices": []}

    K = is_sharpes_matrix.shape[1]
    N = is_sharpes_matrix.shape[0]

    oos_ranks_of_is_winners = []
    is_winner_indices = []
    for i in range(N):
        is_row = is_sharpes_matrix.iloc[i].values
        oos_row = oos_sharpes_matrix.iloc[i].values
        if np.all(np.isnan(is_row)) or np.all(np.isnan(oos_row)):
            continue
        is_winner_idx = int(np.nanargmax(is_row))
        is_winner_indices.append(is_winner_idx)
        # OOS rank of this winner: 0 = worst, K-1 = best
        oos_ranks = pd.Series(oos_row).rank().values - 1
        winner_oos_rank = oos_ranks[is_winner_idx]
        oos_ranks_of_is_winners.append(winner_oos_rank)

    if not oos_ranks_of_is_winners:
        return {"pbo": float("nan"), "logit_pbo": float("nan"),
                "n_paths": 0, "n_strategies": K,
                "paths_winner_oos_rank": [], "is_winner_indices": []}

    arr = np.array(oos_ranks_of_is_winners)
    median = (K - 1) / 2.0
    pbo = float((arr <= median).mean())
    pbo_clamped = min(max(pbo, 1e-6), 1 - 1e-6)
    logit_pbo = math.log(pbo_clamped / (1 - pbo_clamped))

    return {
        "pbo":                   pbo,
        "logit_pbo":              logit_pbo,
        "n_paths":                len(arr),
        "n_strategies":           K,
        "paths_winner_oos_rank":  arr.tolist(),
        "is_winner_indices":      is_winner_indices,
    }
