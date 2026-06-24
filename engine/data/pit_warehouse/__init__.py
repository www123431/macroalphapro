"""engine.data.pit_warehouse — Tier C L2-1 PIT data layer.

Per docs/spec_pit_data_accessor.md, this package implements:

  L1 PIT Data Warehouse
    - All cached parquets in this layer are PIT-clean by construction
    - Pull scripts in scripts/extend_*_pit_history.py keep them fresh
    - Module-level lru_cache(1) on each loader for process lifetime

  L2 Simulation Clock
    - SimClock advances through backtest time
    - Provides knows_about(as_of) → bool for PIT filtering

  L3 PIT Data Accessor
    - PITDataAccessor wraps L1 + L2
    - All template data access goes through this interface
    - Templates CANNOT read parquet directly (architectural rule)

  L4 Template Contract enforcement is in
  engine.agents.strengthener.templates._template_contract (separate
  module to avoid circular deps).

Cached files this layer reads (all PIT-clean by construction):

  data/cache/_crsp_msf_long_history.parquet     (CRSP monthly, prices/ret/mktcap)
  data/cache/_crsp_dsedelist.parquet            (delisting events + dlret)
  data/cache/_crsp_ccm_link.parquet             (CRSP-Compustat link)
  data/cache/_compustat_funda_pit.parquet       (PIT first-reported funda; NEW)
  data/cache/_sp500_constituents_pit.parquet    (PIT SP500 membership; NEW)

Each loader:
  - Reads parquet exactly once per process (lru_cache)
  - Returns immutable view (callers must .copy() if mutating)
  - Raises FileNotFoundError if cache missing (caller maps to DATA_ERROR
    verdict in template; user must run the appropriate pull script)
"""
from engine.data.pit_warehouse.simulation_clock import SimClock
from engine.data.pit_warehouse.accessor import PITDataAccessor

__all__ = ["SimClock", "PITDataAccessor"]
