"""engine/research/sleeves/post_earnings_drift_parent.py — Sleeve
implementation for parent D_PEAD (cousin_anchor in library).

Mirrors PitSnDpeadSleeve pattern as proof that the same Protocol works
for a second concrete sleeve — establishes the migration template for
remaining sleeves (carry / TSMOM / crisis_hedge / mom_hedge).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.research.sleeve_registry import (
    SleeveProtocol,
    load_audit_blocks_from_yaml,
    register_sleeve,
)
from engine.research.strategy_lifecycle import AuditBlocks

REPO_ROOT = Path(__file__).resolve().parents[3]
_LIBRARY_YAML = REPO_ROOT / "data" / "research" / "mechanism_library" / \
                "post_earnings_drift.yaml"
_RETURNS_PARQUET = REPO_ROOT / "data" / "cache" / "_dpead_recon_base.parquet"


@register_sleeve("post_earnings_drift")
class ParentDpeadSleeve:
    """Parent universe-wide D_PEAD sleeve (cousin_anchor)."""

    strategy_id = "post_earnings_drift"
    library_yaml_path = _LIBRARY_YAML

    _audit_cache: AuditBlocks | None = None

    def returns(self) -> pd.Series:
        s = pd.read_parquet(_RETURNS_PARQUET).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        # Parent returns are daily — resample to monthly to match SleeveProtocol contract
        monthly = ((1 + s.clip(-0.2, 0.2)).resample("ME").prod() - 1)
        return monthly.dropna().rename(self.strategy_id)

    def audit_blocks(self) -> AuditBlocks:
        if type(self)._audit_cache is None:
            type(self)._audit_cache = load_audit_blocks_from_yaml(_LIBRARY_YAML)
        return type(self)._audit_cache
