"""engine/research/sleeves/tail_hedge_put_spread_sleeve.py — SLM
registered sleeve for the Path C put-spread tail hedge.

Strategy: SPX 30d put spread (delta -25 long / delta -10 short),
monthly roll, hold to expiry, 5% of book NAV notional.

Library YAML: data/research/mechanism_library/tail_hedge_put_spread.yaml
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.research.sleeve_registry import (
    SleeveProtocol, load_audit_blocks_from_yaml, register_sleeve,
)
from engine.research.strategy_lifecycle import AuditBlocks

REPO_ROOT = Path(__file__).resolve().parents[3]
_LIBRARY_YAML = REPO_ROOT / "data" / "research" / "mechanism_library" / \
                "tail_hedge_put_spread.yaml"
_RETURNS_PARQUET = REPO_ROOT / "data" / "cache" / \
                   "_tail_hedge_put_spread_monthly.parquet"


@register_sleeve("tail_hedge_put_spread")
class TailHedgePutSpreadSleeve:
    """SPX put-spread tail hedge sleeve — role=insurance."""

    strategy_id = "tail_hedge_put_spread"
    library_yaml_path = _LIBRARY_YAML

    _audit_cache: AuditBlocks | None = None

    def returns(self) -> pd.Series:
        s = pd.read_parquet(_RETURNS_PARQUET).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s.dropna().rename(self.strategy_id)

    def audit_blocks(self) -> AuditBlocks:
        if type(self)._audit_cache is None:
            type(self)._audit_cache = load_audit_blocks_from_yaml(_LIBRARY_YAML)
        return type(self)._audit_cache
