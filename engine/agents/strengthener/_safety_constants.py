"""engine.agents.strengthener._safety_constants — Tier C L2-1 Phase 2.5.

LOCKED safety boundaries for Tier C dispatcher + templates. These
are the A-class constants identified in the 2026-06-08 hardcoding
audit (commit eaed51c8 doc): statistical/theoretical truths that
must NEVER be auto-relaxed because relaxing them enables p-hacking
or sample-mining.

ARCHITECTURAL DOCTRINE
======================
Every constant in this module is "stay hardcoded forever" by design.
The 4-class taxonomy (docs/spec_pit_data_accessor.md §A.2):
  A-class: SAFETY (this module — locked)
  B-class: design choices (parameterized on FactorSpec v2)
  C-class: controlled vocabularies (locked enums on FactorSpec)
  D-class: implementation details (per-template private constants)

If you find yourself wanting to modify a value here, you are
probably wrong. The right move is:
  1. Read the literature reference in the docstring
  2. Confirm the new value has equal-or-stronger theoretical
     backing
  3. Get human-audit sign-off
  4. Update the literature reference
A-class values are NEVER changed via spec parameters.

USAGE
=====
Templates and dispatcher import from here:
  from engine.agents.strengthener._safety_constants import (
      T_GREEN, T_MARGINAL, N_TRIALS_HARD, MIN_STOCKS_PER_BUCKET,
      MAX_AUTO_DISPATCHES_PER_WEEK, REPLICATION_T_TOLERANCE,
  )
NEVER:
  T_GREEN = 1.5  # ← DOCTRINE VIOLATION
"""
from __future__ import annotations


# ────────────────────────────────────────────────────────────────────
# Statistical significance thresholds (verdict ladder)
# ────────────────────────────────────────────────────────────────────
T_GREEN: float = 1.96
"""Two-sided 5% significance threshold for NW-adjusted Sharpe t-stat.

LITERATURE: Newey-West 1987 + Lo 2002 + Bailey-Lopez de Prado 2014.
This is the CONVENTIONAL threshold. Harvey-Liu-Zhu 2016 argue 1.96
is too lenient for factor zoo discovery and recommend |t| >= 3.0.

Tier C dispatcher uses 1.96 as the WEAK floor (GREEN). Stricter
gates (post-pub robust t, DSR-deflated t, anchor-orthogonal t) are
layered on top via L2-2/L2-4/L3-* — not by raising this threshold.

DO NOT change this value. It is the basis of the GREEN verdict
contract across the entire Tier C ecosystem.
"""


T_MARGINAL: float = 1.65
"""Two-sided 10% significance threshold (= one-sided 5%).

LITERATURE: same as T_GREEN. Standard convention.
"""


REPLICATION_T_TOLERANCE: float = 0.5
"""L2-2 replication mode: if |our_t - paper_t| > tolerance →
MISMATCH flag, headline verdict downgrades GREEN → MARGINAL.

LITERATURE: Sharpe SE ~ sqrt((1+SR²/2)/n_years). For SR=0.6, 30y:
SE ≈ 0.13 (in Sharpe space) ≈ 0.5 in t-stat space. So 0.5 t-stat
tolerance ≈ 1 standard error — a reasonable replication threshold
per Bailey-LdP guidance.

DO NOT loosen — too-loose tolerance hides implementation bugs.
"""


# ────────────────────────────────────────────────────────────────────
# Multi-testing penalty (Bailey-Lopez de Prado §3)
# ────────────────────────────────────────────────────────────────────
N_TRIALS_CAUTION: int = 7
"""Per-family/mechanism-class n_trials at which the system emits
inbox CAUTION (not gate block). Wakes the principal up to think
about DSR penalty before next dispatch.

LITERATURE: Bailey-Lopez de Prado 2014, Hou-Xue-Zhang 2020.
"""


N_TRIALS_HARD: int = 15
"""Per-family/mechanism-class n_trials at which dispatcher REFUSES
to dispatch (DispatchRefusal.N_TRIALS_HARD). Principal must
explicitly override + acknowledge inflated DSR threshold for that
family.

LITERATURE: Bailey-Lopez de Prado §3 — DSR penalty grows roughly
log(N_trials); at N=15 deflation factor ≈ 0.7, beyond which
"significance" claims require unusually strong evidence.

DO NOT raise — silent multi-test inflation = factor zoo at scale.
"""


# ────────────────────────────────────────────────────────────────────
# Cost-gate (operational safety)
# ────────────────────────────────────────────────────────────────────
MAX_AUTO_DISPATCHES_PER_WEEK: int = 5
"""Rolling 7-day cap on dispatch count across ALL families.

NOT a statistical threshold — operational: prevents an LLM-extractor
regression from triggering 50 spurious dispatches in a day (runaway
cron / multi-test inflation). Human override required above this.

If you genuinely need higher throughput, the right answer is to
RUN MULTIPLE PRINCIPALS, not raise this cap — at 5/week per
principal you already exceed 200/year, more than any individual
researcher should generate without retrospective.
"""


# ────────────────────────────────────────────────────────────────────
# Bucket / portfolio formation lower bounds
# ────────────────────────────────────────────────────────────────────
MIN_STOCKS_PER_BUCKET: int = 30
"""Cross-sectional L/S: minimum stocks per quintile/decile bucket
before the L-S spread is considered statistically meaningful.

LITERATURE: bootstrap CI requires n >= 30 for asymptotic
normality (CLT). Stronger: Lo 2002 / Politis-Romano block
bootstrap CI shrinks slowly below n=30.

Cross-sec template skips months below this threshold (no PnL
recorded — see _quintile_long_short_pnl). DO NOT lower.
"""
