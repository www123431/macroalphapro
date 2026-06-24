"""engine/research/sleeves/post_earnings_drift_pit_sn.py — Sleeve
implementation for PIT FF12 within-sector D_PEAD (the deploy variant).

Registers under strategy_id="post_earnings_drift_pit_sn" matching the
library YAML at data/research/mechanism_library/post_earnings_drift_pit_sn.yaml.

Audit lineage (per the YAML):
  cost_model.audit_commit:        844d401
  factor_exposure.audit_commit:   844d401  (Phase 3, 5 styles + 11 GICS)
  factor_exposure.alpha_t_hac:    +9.65
  factor_exposure.proposed_role:  alpha_seeker
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
                "post_earnings_drift_pit_sn.yaml"
_RETURNS_PARQUET = REPO_ROOT / "data" / "cache" / "_dpead_sn_pit_monthly.parquet"


@register_sleeve("post_earnings_drift_pit_sn")
class PitSnDpeadSleeve:
    """PIT FF12 within-sector D_PEAD sleeve.

    Construction is zero-arg; YAML + parquet paths are module-level constants.
    Caching: audit_blocks() parses the YAML once; returns() reloads each call
    (Pandas caches the parquet via OS page cache so this is cheap).
    """

    strategy_id = "post_earnings_drift_pit_sn"
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
