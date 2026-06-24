"""engine/research/capacity_model.py — strategy capacity analysis.

Senior gap ④ per [[project-end-to-end-vision-2026-05-30]]: "这信号能装
多少钱?" Without an answer, a Sharpe-0.8 strategy that only works at
$5M AUM gets confused with a Sharpe-0.8 strategy that scales to $1B.
For deploy decisions these are utterly different.

KEY METRICS:

  hard_capacity_usd
    Maximum AUM where you can hold the strategy's target weights
    without exceeding max_participation × ADV in ANY position.
    "If I had $X, would I be a 5%+ ADV trade in any name?" If yes,
    that name caps capacity at adv_usd × max_participation / weight.

  alpha_decay_aum
    AUM at which expected NET Sharpe drops to half of gross Sharpe.
    Per Korajczyk-Sadka 2004 + Chen-Da-Gao 2010 stylized fact:
    net_sharpe ≈ gross_sharpe × sqrt(capacity / AUM) once impact
    binds. Decay starts well before hard capacity.

  half_life_aum
    The AUM at which alpha decays to half — usable as a "deploy ceiling"
    for the strategy.

  monthly_turnover_dollars
    At test AUM, how many $ are traded per period? Useful for
    sanity-checking against broker capacity / TCA estimates.

DESIGN: pure-math module, no fetcher deps. Takes weights panel +
ADV panel as inputs. The runner / forward_oos_observer can compute
capacity metrics post-hoc from real-data simulation outputs.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CapacityParams:
    """Tunable capacity-analysis params."""
    max_participation: float = 0.05    # cap each trade at 5% ADV
    half_decay_target: float = 0.5     # net Sharpe / gross = 0.5 at half-life AUM
    impact_coef:       float = 0.5     # Almgren-Chriss sqrt-impact multiplier
    default_adv_usd:   float = 200_000_000   # fallback per asset
    default_sigma:     float = 0.015   # daily vol fallback


@dataclasses.dataclass
class CapacityReport:
    """Output of capacity_report() — comprehensive view."""
    hard_capacity_usd:        float
    half_life_aum_usd:        float | None    # None if cannot estimate
    monthly_turnover_usd:     float           # at the test AUM
    binding_constraint_asset: str | None      # which name caps hardest
    binding_adv_usd:          float | None
    binding_weight:           float | None
    n_periods_analyzed:       int
    test_aum_usd:             float           # input AUM used for analysis
    n_names_average:          float           # avg portfolio breadth

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ── Hard capacity (participation-rate constraint) ─────────────────────────

def hard_capacity_usd(
    weights_panel:   pd.DataFrame,
    adv_panel:       pd.DataFrame | None,
    *,
    max_participation: float = 0.05,
    default_adv_usd:   float = 200_000_000,
) -> tuple[float, str | None, float | None, float | None]:
    """Maximum AUM where no position exceeds max_participation × ADV.

    For asset i, capacity_i = adv_i × max_participation / |w_i|
    (the AUM at which weight w_i becomes a max_participation × ADV trade).
    Strategy capacity = min over assets and time.

    Returns: (capacity_usd, binding_asset, binding_adv_usd, binding_weight)
    """
    if weights_panel is None or weights_panel.empty:
        return float("inf"), None, None, None

    if adv_panel is None or adv_panel.empty:
        adv = pd.DataFrame(
            default_adv_usd,
            index=weights_panel.index, columns=weights_panel.columns,
        )
    else:
        adv = adv_panel.reindex_like(weights_panel).fillna(default_adv_usd)
        adv = adv.where(adv > 0, default_adv_usd)

    abs_w = weights_panel.abs()
    # Avoid divide-by-zero where weight is tiny / zero
    safe_w = abs_w.where(abs_w > 1e-6)
    capacity_per_pos = adv * max_participation / safe_w
    capacity_per_pos = capacity_per_pos.where(np.isfinite(capacity_per_pos))

    if capacity_per_pos.empty or capacity_per_pos.isna().all().all():
        return float("inf"), None, None, None

    # The binding constraint = the smallest capacity_per_pos
    min_val = capacity_per_pos.min(skipna=True).min(skipna=True)
    if not np.isfinite(min_val):
        return float("inf"), None, None, None
    # Locate it
    flat = capacity_per_pos.stack(dropna=True)
    if flat.empty:
        return float(min_val), None, None, None
    idx = flat.idxmin()    # (date, asset)
    binding_date, binding_asset = idx
    binding_adv = float(adv.loc[binding_date, binding_asset])
    binding_weight = float(abs_w.loc[binding_date, binding_asset])
    return float(min_val), str(binding_asset), binding_adv, binding_weight


# ── Alpha decay estimate (soft capacity via sqrt impact) ─────────────────

def estimate_half_life_aum(
    gross_sharpe:      float,
    monthly_turnover_usd_at_test_aum: float,
    test_aum_usd:      float,
    universe_adv_usd:  float,
    *,
    daily_sigma:       float = 0.015,
    impact_coef:       float = 0.5,
    half_target:       float = 0.5,
) -> float | None:
    """Estimate AUM at which net Sharpe drops to (half_target × gross).

    MODEL: alpha lost to impact per dollar traded ≈
    σ × √(notional / ADV) × impact_coef (Almgren-Chriss linearized).

    Scaling: if turnover and AUM both grow with AUM, impact loss
    scales like sqrt(AUM). So net_sharpe ≈ gross_sharpe × (1 - k × √AUM/something).

    Approximation: half_life_aum is the AUM where impact loss equals
    (1 - half_target) × gross alpha per turnover dollar.

    Returns None if inputs make the estimate undefined.
    """
    if gross_sharpe <= 0:
        return None
    if monthly_turnover_usd_at_test_aum <= 0 or test_aum_usd <= 0:
        return None
    if universe_adv_usd <= 0:
        return None

    # Stylized closed-form: solve for the AUM where impact_cost_bps
    # = (1 - half_target) × gross_alpha_bps
    # gross_alpha_bps ≈ gross_sharpe × σ × √12 × 100 (annualized in bps)
    annualized_gross_alpha_bps = gross_sharpe * daily_sigma * (252.0 ** 0.5) * 10_000.0

    # Impact bps at test AUM ≈
    #   σ × √(turnover_dollars / universe_ADV) × impact_coef × 10000
    test_participation = monthly_turnover_usd_at_test_aum / universe_adv_usd
    impact_bps_at_test = (
        daily_sigma * np.sqrt(test_participation) * impact_coef * 10_000.0
    )

    if impact_bps_at_test <= 0:
        return None

    # Target impact = (1 - half_target) × gross
    target_impact_bps = (1.0 - half_target) * annualized_gross_alpha_bps
    # Impact scales with √AUM (since turnover ∝ AUM at scale)
    # impact(AUM) = impact_at_test × √(AUM/test_AUM)
    # So AUM_target = test_AUM × (target_impact / impact_at_test) ** 2
    ratio = target_impact_bps / impact_bps_at_test
    half_life_aum = test_aum_usd * (ratio ** 2)
    return float(half_life_aum)


# ── Comprehensive report ──────────────────────────────────────────────────

def capacity_report(
    weights_panel:  pd.DataFrame,
    adv_panel:      pd.DataFrame | None,
    *,
    test_aum_usd:   float = 100_000_000,
    gross_sharpe:   float | None = None,
    params:         CapacityParams | None = None,
) -> CapacityReport:
    """Build a CapacityReport from a strategy's weights time-series.

    test_aum_usd: the AUM for which to compute monthly-turnover and
      half_life_aum estimates. Defaults to $100M.
    gross_sharpe: needed for the soft alpha-decay estimate. Pass the
      run_gate-reported Sharpe. If None, half_life_aum returned as None.
    """
    p = params or CapacityParams()
    if weights_panel is None or weights_panel.empty:
        return CapacityReport(
            hard_capacity_usd=0.0,
            half_life_aum_usd=None,
            monthly_turnover_usd=0.0,
            binding_constraint_asset=None,
            binding_adv_usd=None,
            binding_weight=None,
            n_periods_analyzed=0,
            test_aum_usd=test_aum_usd,
            n_names_average=0.0,
        )

    # Hard capacity
    hc, asset, adv_b, w_b = hard_capacity_usd(
        weights_panel, adv_panel,
        max_participation=p.max_participation,
        default_adv_usd=p.default_adv_usd,
    )

    # Turnover at test AUM
    abs_dw = weights_panel.diff().abs().fillna(0)
    period_turnover_fraction = abs_dw.sum(axis=1)
    avg_period_turnover_usd = (period_turnover_fraction.mean() * test_aum_usd)

    # Universe ADV (average across observed assets)
    if adv_panel is not None and not adv_panel.empty:
        universe_adv = (
            adv_panel.reindex_like(weights_panel)
            .fillna(p.default_adv_usd)
            .sum(axis=1).mean()
        )
    else:
        universe_adv = p.default_adv_usd * weights_panel.shape[1]

    # Soft half-life estimate
    half_life = None
    if gross_sharpe is not None and gross_sharpe > 0 and avg_period_turnover_usd > 0:
        half_life = estimate_half_life_aum(
            gross_sharpe=gross_sharpe,
            monthly_turnover_usd_at_test_aum=avg_period_turnover_usd,
            test_aum_usd=test_aum_usd,
            universe_adv_usd=universe_adv,
            daily_sigma=p.default_sigma,
            impact_coef=p.impact_coef,
            half_target=p.half_decay_target,
        )

    n_names_avg = float((weights_panel.abs() > 1e-6).sum(axis=1).mean())

    return CapacityReport(
        hard_capacity_usd=hc,
        half_life_aum_usd=half_life,
        monthly_turnover_usd=float(avg_period_turnover_usd),
        binding_constraint_asset=asset,
        binding_adv_usd=adv_b,
        binding_weight=w_b,
        n_periods_analyzed=int(weights_panel.shape[0]),
        test_aum_usd=test_aum_usd,
        n_names_average=n_names_avg,
    )
