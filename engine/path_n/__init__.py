"""
engine/path_n/ — Path N S&P 500 Index Reconstitution Drift v1.

Pre-registration: docs/spec_path_n_index_reconstitution_drift_v1.md (id=70 hash c92d2c36)

Chen-Noronha-Singal 2004 *JF* — S&P 500 add events create predictable
pre-effective-date drift due to anticipated forced index-fund rebalancing.

Strategy: Long S&P 500 adds T-5 to T-1 trading days before effective date.
TC: 30bp roundtrip per single-stock trade (standing rule).
Gates: single-stock 0.5/2.0 PASS, 0.3/1.5 MARGINAL.

Scout (2026-05-13 night) Add-only long: gross Sharpe 0.81 NW t 2.57;
all sub-period > 0 (Pre 0.42 / COVID 1.59 / Post 0.99).

This is project's first STRUCTURAL mechanism test since K1 BAB — distinct
from flow-contrarian (Ben-David / Coval-Stafford both dead/reversed),
calendar (Path E dead), momentum/value (Path H/L dead).
"""
from __future__ import annotations
