"""engine/research/family_trial_counter.py — count candidate trials
within the same mechanism family, for Bailey-LdP Deflated Sharpe
multiple-testing correction.

Why this matters (per Bailey-LdP 2014 §3):

  "The number of trials N entering the DSR calculation should be the
  number of INDEPENDENT model configurations tried for the SAME family
  of strategies, NOT the total number of strategies tested across the
  organization."

  Conflating cross-family trials inflates N, depresses DeflSR, and
  rejects genuinely-different mechanisms as if they competed.

Phase 2.5 fix: when validating a candidate, count N = (library entries
in same family) + EXPLORATION_BUFFER (failed experiments not in library).

Default buffer = 3, configurable per family. Per the project graveyard
audit (2026-05-28), our exploration ratio is ~3 failed / 1 promoted.

Module API:

  family = "earnings_underreaction"
  n = count_trials_in_family(family)
  # → 2 library entries (post_earnings_drift + PIT SN) + 3 buffer = 5

  result = evaluate_three_layer(
      ...
      n_trials_across_research=count_trials_in_family(candidate_family),
  )
"""
from __future__ import annotations

from pathlib import Path

import yaml as _pyyaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"

# Exploration buffer per family — how many failed experiments we tried
# that didn't make it into the library (variants tried in candidate_
# pipeline that were REJECTED before promotion). Calibrated against
# the 2026-05-28 graveyard audit ratio of ~3 failed / 1 promoted.
DEFAULT_EXPLORATION_BUFFER = 3

# Family-specific overrides — set when we KNOW we tried many or few
# configurations. Higher buffer = more conservative DeflSR.
FAMILY_BUFFER_OVERRIDES: dict[str, int] = {
    # We tried many earnings_underreaction variants this session
    # (parent + PIT SN + IBES booster + combo + sector variants)
    "earnings_underreaction": 5,
    # carry: 4-leg + cousins + bond_carry_slope (RED) + carry_equity_div (RED)
    "carry": 4,
    # tsmom: 5-leg + bond_xsmom (RED in our book)
    "tsmom": 2,
    # momentum: jegadeesh + residual + xsmom_jt + cousins
    "momentum": 3,
    # vol_carry: VRP + variance-of-variance variants
    "vol_carry": 2,
    # quality: gross profitability + QMJ + BAB
    "quality": 2,
    # cross_asset_hedge: TLT/GLD + crisis_alpha variants
    "cross_asset_hedge": 2,
    # factor_hedge: MTUM-short + variants
    "factor_hedge": 1,
}

# Hard floor — never go below this when family is fully unknown
ABSOLUTE_MIN_TRIALS = 3


def _load_library_families() -> dict[str, list[str]]:
    """Group library strategy_ids by family. Returns {family → [ids]}."""
    out: dict[str, list[str]] = {}
    for fp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            d = _pyyaml.safe_load(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d or "family" not in d:
            continue
        fam = d["family"]
        sid = d.get("id", fp.stem)
        out.setdefault(fam, []).append(sid)
    return out


def count_library_entries_in_family(family: str) -> int:
    """Count library YAMLs tagged with `family`. Includes both DEPLOYED
    and REJECTED entries — they all count as trials per Bailey-LdP."""
    families = _load_library_families()
    return len(families.get(family, []))


def count_trials_in_family(
    family: str,
    *,
    exploration_buffer: int | None = None,
) -> int:
    """N to pass to deflated_sharpe_ratio for a candidate in `family`.

    Default buffer per family is taken from FAMILY_BUFFER_OVERRIDES;
    override via the exploration_buffer parameter when the caller has
    a more accurate count (e.g. has logged the actual experiments).

    Returns max(library_count + buffer, ABSOLUTE_MIN_TRIALS).
    """
    library_count = count_library_entries_in_family(family)
    if exploration_buffer is None:
        exploration_buffer = FAMILY_BUFFER_OVERRIDES.get(
            family, DEFAULT_EXPLORATION_BUFFER,
        )
    return max(library_count + exploration_buffer, ABSOLUTE_MIN_TRIALS)


def explain_count(family: str) -> dict:
    """For UI / debugging: return the components that go into the count."""
    library_count = count_library_entries_in_family(family)
    buffer = FAMILY_BUFFER_OVERRIDES.get(family, DEFAULT_EXPLORATION_BUFFER)
    total = max(library_count + buffer, ABSOLUTE_MIN_TRIALS)
    return {
        "family": family,
        "library_entries": library_count,
        "exploration_buffer": buffer,
        "buffer_source": ("FAMILY_BUFFER_OVERRIDES" if family in FAMILY_BUFFER_OVERRIDES
                          else "DEFAULT_EXPLORATION_BUFFER"),
        "absolute_min": ABSOLUTE_MIN_TRIALS,
        "computed_n_trials": total,
    }
