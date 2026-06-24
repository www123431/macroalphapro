"""
engine/portfolio_sleeves.py — MS-2 cross-sleeve capital allocation layer.

Status: NEW 2026-05-10 (MS-2). Multi-sleeve commit per
`project_final_vision_hybrid_2026-05-10.md`.

Purpose
-------
Combines per-sleeve portfolio weights into a single capital-weighted final
portfolio. Production `engine/portfolio.py::construct_portfolio` is the
ETF-sleeve engine (unchanged); this module is the sleeve-aware aggregator.

Capital allocation governance
-----------------------------
Per memory `feedback_factor_research_3_tier_framework.md` Tier 3 + project
final vision V2:
  - Initial: 100% etf_l1 / 0% ss_sp500
    (only ETF universe production-ready; single-stock sleeve gated on Wave B
    verdict + supervisor PendingApproval)
  - Reallocation triggers (locked):
    * Wave B PASS  (BHY ≤ 0.05 + Sharpe ≥ 0.4): 30-40% / 60-70%
    * Wave B MARGINAL (directional positive)  : 10-20% / 80-90%
    * Wave B REJECT / FAIL_UNDERPOWERED        : 0% / 100% (unchanged)
    * Wave B PRELIMINARY                       : 0% real capital (paper only)
  - Reallocation requires Tier 3 governance: supervisor PendingApproval +
    spec amendment (no automatic capital re-route).

Architecture invariants
-----------------------
- 0 LLM imports (capital allocation is deterministic, not AI-mediated)
- `construct_portfolio` (ETF) and any future single-stock portfolio engine
  remain side-by-side; this module owns ONLY the cross-sleeve capital math
- Allowlist enforcement: known sleeve_id set is locked (etf_l1, ss_sp500;
  cta_defensive to be added after Path O spec lock — see ALLOWED_SLEEVES note)
  to prevent typo-induced silent data drift
"""
from __future__ import annotations

import dataclasses
import json
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Locked sleeve allowlist (must match db_models.py sleeve_id values) ─────
# 2026-05-13 evening: crypto_btc_eth REMOVED. Path N v1 alpha FAIL (spec id=71)
# + Path N-SAA v1 DEPLOYABLE_BUT_MARGINAL (spec id=72; ρ=0.32 to equity, not
# real crisis hedge — "self-deceiving diversification" per user). Crypto sleeve
# never received production trades (0 rows in simulated_trades). Specs id=71/72
# preserved in spec_metadata for falsification chain audit but sleeve removed
# from runtime allowlist.
#
# 2026-05-13 evening: cta_defensive ADDED. Path O CTA Defensive Overlay v1
# (spec id=73 hash 9630c2bb) SAA_DEPLOYABLE 5/5 gates. PQTIX 100% sleeve, 10%
# portfolio allocation per Faber 2007 institutional floor. Empirically validated:
# max DD 34%→30% (-3.6pp), Sharpe-neutral (+0.003), ρ=-0.19, all 3 crisis
# windows positive (2018+6.34%, 2020+6.47%, 2022+11.61%). Tier 3 PendingApproval
# required for actual capital deployment per multi-sleeve governance.
#
# 2026-05-15: rms_crisis_hedge ADDED. Path AC TLT/GLD Crisis Hedge Sleeve v1
# (spec id=77 hash 4db40176) v3 INSURANCE class PASS 4/4 on extended 2005-23 +
# 60/40 SPY/AGG baseline. G7 portfolio max DD reduction +7.42pp at 15% (2008 GFC
# +18pp dominant). Tier 3 APPROVED 2026-05-15 at 10% allocation per Asness-
# Israelov 2017 RMS insurance-budget framework (existing 4 sleeves reduced 10%
# proportionally). See docs/decisions/saa_path_ac_addition_review_2026-05-15.md.
# Single source of truth lives in engine.strategies.registry. Re-exported
# here so existing consumers (engine.db_models, etc.) keep their import paths
# unchanged. Adding a new sleeve requires editing only registry.ALLOWED_SLEEVES
# + adapters.py — this re-export propagates automatically.
from engine.strategies.registry import ALLOWED_SLEEVES

# Default initial capital allocation (project final vision V2 lock + Path O SAA)
DEFAULT_INITIAL_ALLOCATION: dict[str, float] = {
    "etf_l1":         0.90,    # ETF tier 1/2 production (QL01 BAB + Multivariate v3)
    "ss_sp500":       0.00,    # gated on Wave B verdict
    "cta_defensive":  0.10,    # Path O SAA_DEPLOYABLE (PQTIX crisis-positive overlay)
}

# SystemConfig key for runtime override (supervisor-tunable via PendingApproval)
_SYSCONFIG_KEY_SLEEVE_ALLOC = "portfolio_sleeves.capital_allocation"


# ── Configuration dataclass ────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class SleeveCapitalConfig:
    """Capital allocation across portfolio sleeves.

    Sums to 1.0 (validated at construction). Each sleeve_id must be in
    ALLOWED_SLEEVES. Negative weights and weights > 1.0 are rejected.

    Audit trail: when persisted to SystemConfig, includes timestamp + actor
    so reallocation history is queryable.
    """
    allocations: dict[str, float]   # {sleeve_id: weight}; sums to 1.0

    def __post_init__(self) -> None:
        unknown = set(self.allocations) - ALLOWED_SLEEVES
        if unknown:
            raise ValueError(
                f"unknown sleeve_id(s) {sorted(unknown)}; ALLOWED={sorted(ALLOWED_SLEEVES)}"
            )
        for sid, w in self.allocations.items():
            if not isinstance(w, (int, float)):
                raise TypeError(f"weight for {sid} must be numeric, got {type(w).__name__}")
            if w < 0 or w > 1:
                raise ValueError(
                    f"weight for {sid} = {w} outside [0, 1]; capital allocation "
                    f"must be a long-only proportion"
                )
        total = sum(float(w) for w in self.allocations.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"capital allocation must sum to 1.0, got {total:.6f} "
                f"(allocations={self.allocations})"
            )

    def to_dict(self) -> dict[str, float]:
        return dict(self.allocations)

    @classmethod
    def initial(cls) -> "SleeveCapitalConfig":
        """Project initial state: 100% etf_l1 / 0% ss_sp500."""
        return cls(allocations=dict(DEFAULT_INITIAL_ALLOCATION))


# ── Public API ──────────────────────────────────────────────────────────────
def combine_sleeve_weights(
    sleeve_weights:  dict[str, pd.Series],
    config:          Optional[SleeveCapitalConfig] = None,
    leverage_factor: float = 1.0,
) -> pd.Series:
    """Combine per-sleeve portfolio weights into a single capital-weighted
    final portfolio, optionally scaled by leverage factor.

    Args:
        sleeve_weights: {sleeve_id: pd.Series indexed by ticker}
                        Per-sleeve weights as produced by each sleeve's own
                        portfolio construction engine.
        config:         SleeveCapitalConfig; defaults to initial 100/0.
        leverage_factor: Gross exposure multiplier (default 1.0 = unlevered).
                         1.5 = 150% gross / 50% borrowed at RFR. Used by
                         Path B (Tier 3 mandate amendment 2026-05-15).
                         Per Modigliani-Miller 1958: preserves Sharpe under
                         RFR-borrow-cost assumption.

    Returns:
        pd.Series indexed by union of all sleeve tickers, values =
        Σ_sleeve allocation[sleeve] × sleeve_weight[ticker] × leverage_factor

    Behavior:
        - Sleeves with allocation=0 are skipped (return 0 contribution)
        - Tickers appearing in multiple sleeves are summed
        - Empty input → empty output Series
        - leverage_factor=1.0 (default) preserves backward compat
    """
    if config is None:
        config = SleeveCapitalConfig.initial()

    if not sleeve_weights:
        return pd.Series(dtype=float)

    if leverage_factor <= 0:
        raise ValueError(f"leverage_factor must be > 0, got {leverage_factor}")
    if leverage_factor > 3.0:
        raise ValueError(
            f"leverage_factor {leverage_factor} > 3.0 disallowed for safety; "
            "extreme leverage requires explicit doctrine amendment"
        )

    contributions: list[pd.Series] = []
    for sleeve_id, weights in sleeve_weights.items():
        if sleeve_id not in ALLOWED_SLEEVES:
            logger.warning(
                "combine_sleeve_weights: sleeve_id %r not in ALLOWED_SLEEVES; skipping",
                sleeve_id,
            )
            continue
        alloc = config.allocations.get(sleeve_id, 0.0)
        if alloc == 0.0:
            continue
        if weights is None or weights.empty:
            continue
        contributions.append(weights.astype(float) * float(alloc))

    if not contributions:
        return pd.Series(dtype=float)

    combined = pd.concat(contributions, axis=1).fillna(0.0).sum(axis=1)
    # Apply leverage factor at the combination step
    combined = combined * float(leverage_factor)
    # Drop near-zero entries to avoid numeric noise in downstream
    combined = combined[combined.abs() > 1e-12]
    return combined.astype(float)


# ── Persistence (SystemConfig-backed) ──────────────────────────────────────
def get_active_config() -> SleeveCapitalConfig:
    """Read current capital allocation from SystemConfig; falls back to initial.

    Wraps engine.memory.get_system_config so allocation can be tuned at
    runtime by supervisor (via PendingApproval) without code edit.
    """
    try:
        from engine.memory import get_system_config
        raw = get_system_config(_SYSCONFIG_KEY_SLEEVE_ALLOC, "")
    except Exception as exc:
        logger.warning("get_active_config: SystemConfig unreachable (%s); falling back to initial", exc)
        return SleeveCapitalConfig.initial()

    if not raw:
        return SleeveCapitalConfig.initial()
    try:
        parsed = json.loads(raw)
        return SleeveCapitalConfig(allocations={
            str(k): float(v) for k, v in parsed.items()
        })
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning(
            "get_active_config: invalid SystemConfig value %r (%s); falling back to initial",
            raw, exc,
        )
        return SleeveCapitalConfig.initial()


def set_active_config(
    config: SleeveCapitalConfig,
    *,
    actor:  str = "supervisor",
) -> None:
    """Persist new capital allocation to SystemConfig (Tier 3 governance).

    NOTE: This does NOT bypass the PendingApproval gate — the caller is
    expected to have completed Tier 3 governance (supervisor signoff) before
    invoking this. The actor parameter records which approval flow
    triggered the change for audit.
    """
    from engine.memory import set_system_config
    payload = json.dumps(config.to_dict(), ensure_ascii=False)
    set_system_config(_SYSCONFIG_KEY_SLEEVE_ALLOC, payload)
    logger.info(
        "Sleeve capital allocation updated by %s: %s",
        actor, config.to_dict(),
    )


# ── Utility ────────────────────────────────────────────────────────────────
def is_sleeve_active(sleeve_id: str, config: Optional[SleeveCapitalConfig] = None) -> bool:
    """True iff sleeve has > 0 capital allocation in active config."""
    if sleeve_id not in ALLOWED_SLEEVES:
        return False
    cfg = config or get_active_config()
    return cfg.allocations.get(sleeve_id, 0.0) > 0.0
