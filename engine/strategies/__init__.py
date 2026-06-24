"""
engine/strategies/ — Strategy/Sleeve abstraction layer (Agent Constellation Week 1).

Foundation refactor per memory/project_agent_constellation_2026-05-17.md.

Replaces the ad-hoc patterns scattered across engine/portfolio/paper_trade_combined.py
(STRATEGY_DISPLAY_META dict + get_X_signal/is_rebalance_day_X free functions +
duplicate STRATEGY_SPEC_MAP in attribution_logger.py) with:

  - StrategyModule  : ABC each concrete strategy implements
  - StrategyMeta    : frozen dataclass holding the per-strategy locked attributes
  - Sleeve          : frozen dataclass holding sleeve_id / target_weight / members
  - StrategyRegistry: single source of truth UI pages + orchestrator iterate over

Build sequence is intentionally additive — Slice 1-3 introduce the new layer
without touching existing call sites. Slice 4+ migrate consumers.
"""
from engine.strategies.base import StrategyMeta, StrategyModule
from engine.strategies.registry import (
    ALLOWED_SLEEVES,
    Sleeve,
    SleeveClass,
    StrategyRegistry,
    get_registry,
)

__all__ = [
    "StrategyMeta",
    "StrategyModule",
    "Sleeve",
    "SleeveClass",
    "StrategyRegistry",
    "ALLOWED_SLEEVES",
    "get_registry",
]
