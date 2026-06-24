"""engine/validation/cost_stress.py — does the edge survive realistic costs?

The single most common way a backtested edge dies in production: the
gross return is real, but it evaporates once you subtract bid-ask +
market impact at realistic turnover. Path N (5-day index-reconstitution
drift, 24 events/yr) is the prime suspect — and worse, you trade INTO
the exact names everyone else rebalances into, so effective cost is
HIGHER than a normal trade (you face the crowding you're trying to
harvest).

Two framings:
  net_sharpe_curve  — net annualized Sharpe at a grid of round-trip
                      cost levels (bps), given turnover.
  breakeven_cost    — the round-trip cost (bps) at which net mean
                      return hits zero. Compare to a realistic estimate
                      for the instrument to judge survival.

Event-based mode (for Path N): pass per-event returns + events/year +
round-trips/event. Period-based mode (for weekly strategies): pass the
weekly return series + estimated annual turnover (× full book).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostStressResult:
    strategy:           str
    gross_ann_return:   float
    gross_ann_sharpe:   float
    turnover_per_year:  float        # round-trips/yr (event mode) or book-turnover (period mode)
    breakeven_cost_bps: float        # round-trip cost that zeroes mean return
    net_sharpe_at:      dict         # {cost_bps: net_ann_sharpe}
    realistic_cost_bps: float        # the cost we judge realistic for this instrument
    survives:           bool
    verdict:            str


def _ann_sharpe(per_period_mean, per_period_std, periods_per_year):
    if per_period_std <= 0:
        return float("nan")
    return per_period_mean / per_period_std * math.sqrt(periods_per_year)


def cost_stress_event(
    event_returns:     np.ndarray,
    events_per_year:   float,
    realistic_cost_bps: float,
    roundtrips_per_event: float = 1.0,
    cost_grid_bps:     tuple = (0, 5, 10, 20, 30, 50),
) -> CostStressResult:
    """Cost stress for an EVENT strategy (e.g. Path N).

    Each event = one round-trip position (enter + exit). Subtracting
    C bps round-trip cost reduces each event return by
    roundtrips_per_event * C / 10000.

    Args:
      event_returns:        per-event gross returns (decimal).
      events_per_year:      event frequency (Path N ≈ 24).
      realistic_cost_bps:   round-trip cost we judge realistic for the
                            traded instrument. For S&P reconstitution
                            names with crowding-adverse-selection, this
                            is higher than a normal large-cap trade.
      roundtrips_per_event: round-trips per event (1.0 = enter+exit once).
      cost_grid_bps:        cost levels to report net Sharpe at.
    """
    er = np.asarray(event_returns, dtype=float)
    er = er[~np.isnan(er)]
    n = len(er)
    if n < 5:
        return CostStressResult("event", float("nan"), float("nan"),
                                events_per_year, float("nan"), {},
                                realistic_cost_bps, False, "UNDEFINED")

    gross_mean = er.mean()
    gross_std  = er.std(ddof=1)
    gross_ann_ret = events_per_year * gross_mean
    gross_ann_sr  = _ann_sharpe(gross_mean, gross_std, events_per_year)

    def _net_sr(cost_bps):
        drag = roundtrips_per_event * cost_bps / 10000.0
        net_mean = gross_mean - drag
        return _ann_sharpe(net_mean, gross_std, events_per_year)

    net_at = {c: _net_sr(c) for c in cost_grid_bps}

    # Break-even round-trip cost (mean → 0):
    #   gross_mean - roundtrips*C/10000 = 0  →  C = gross_mean*10000/roundtrips
    breakeven = gross_mean * 10000.0 / roundtrips_per_event if roundtrips_per_event else float("nan")

    net_sr_realistic = _net_sr(realistic_cost_bps)
    survives = bool((not math.isnan(net_sr_realistic)) and net_sr_realistic > 0.3)

    if math.isnan(net_sr_realistic):
        verdict = "UNDEFINED"
    elif net_sr_realistic <= 0:
        verdict = f"DIES at realistic {realistic_cost_bps:.0f}bp (net SR <= 0)"
    elif net_sr_realistic < 0.3:
        verdict = f"MARGINAL at {realistic_cost_bps:.0f}bp (net SR {net_sr_realistic:.2f})"
    else:
        verdict = f"SURVIVES {realistic_cost_bps:.0f}bp (net SR {net_sr_realistic:.2f})"

    return CostStressResult(
        strategy="event", gross_ann_return=gross_ann_ret,
        gross_ann_sharpe=gross_ann_sr, turnover_per_year=events_per_year,
        breakeven_cost_bps=breakeven, net_sharpe_at=net_at,
        realistic_cost_bps=realistic_cost_bps, survives=survives,
        verdict=verdict,
    )


def cost_stress_period(
    returns:             np.ndarray,
    annual_turnover:     float,
    realistic_cost_bps:  float,
    periods_per_year:    int = 52,
    cost_grid_bps:       tuple = (0, 5, 10, 20, 30, 50),
) -> CostStressResult:
    """Cost stress for a PERIOD strategy (weekly returns).

    annual_turnover is the fraction of the book turned over per year
    (e.g. 6.0 = 600% = full turnover at each of 6 rebalances). Annual
    cost drag = annual_turnover * cost_bps / 10000, spread evenly across
    periods.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10:
        return CostStressResult("period", float("nan"), float("nan"),
                                annual_turnover, float("nan"), {},
                                realistic_cost_bps, False, "UNDEFINED")

    pp_mean = r.mean()
    pp_std  = r.std(ddof=1)
    gross_ann_ret = pp_mean * periods_per_year
    gross_ann_sr  = _ann_sharpe(pp_mean, pp_std, periods_per_year)

    def _net_sr(cost_bps):
        annual_drag = annual_turnover * cost_bps / 10000.0
        pp_drag = annual_drag / periods_per_year
        return _ann_sharpe(pp_mean - pp_drag, pp_std, periods_per_year)

    net_at = {c: _net_sr(c) for c in cost_grid_bps}

    # Break-even: annual_turnover*C/10000 = gross_ann_ret → C
    breakeven = (gross_ann_ret * 10000.0 / annual_turnover
                 if annual_turnover else float("nan"))

    net_sr_realistic = _net_sr(realistic_cost_bps)
    survives = bool((not math.isnan(net_sr_realistic)) and net_sr_realistic > 0.3)

    if math.isnan(net_sr_realistic):
        verdict = "UNDEFINED"
    elif net_sr_realistic <= 0:
        verdict = f"DIES at realistic {realistic_cost_bps:.0f}bp"
    elif net_sr_realistic < 0.3:
        verdict = f"MARGINAL at {realistic_cost_bps:.0f}bp (net SR {net_sr_realistic:.2f})"
    else:
        verdict = f"SURVIVES {realistic_cost_bps:.0f}bp (net SR {net_sr_realistic:.2f})"

    return CostStressResult(
        strategy="period", gross_ann_return=gross_ann_ret,
        gross_ann_sharpe=gross_ann_sr, turnover_per_year=annual_turnover,
        breakeven_cost_bps=breakeven, net_sharpe_at=net_at,
        realistic_cost_bps=realistic_cost_bps, survives=survives,
        verdict=verdict,
    )
