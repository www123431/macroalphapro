"""
Deployment-level trading configuration.
========================================
Reads the [trading] section from .streamlit/secrets.toml.
Engine code imports from here — never imports streamlit directly.

Priority: secrets.toml [trading] > hardcoded defaults
Changes require app restart (deployment-level config, not runtime preference).
"""
from __future__ import annotations

# ─── Tier 4 hard-block portfolio limits (restored 2026-05-14) ────────────────
# Centralized 2026-05-06 per pages/risk_console.py header comment. Defaults are
# the industry-standard long-short equity limits. Tuned for a multi-sleeve
# system (etf_l1 36% / ss_sp500 54% / cta_defensive 10%); not direct
# parameters of any spec — just hard caps that the Limits tab + Tier 4 audit
# checks against to flag excess concentration / leverage.
LIMIT_SINGLE_POS:   float = 0.10   # max abs weight per single ticker (10%)
LIMIT_HHI:          float = 0.25   # Herfindahl alarm (0.25 = effective ~4 names)
LIMIT_SECTOR_CAP:   float = 0.30   # max combined weight per sector (30%)
LIMIT_GROSS_LEV:    float = 1.50   # max gross exposure (150% = 75L + 75S)
LIMIT_NET_LEV:      float = 1.00   # max abs(net) exposure (100% — long-only equiv at upper bound)
LIMIT_SHORT_TOTAL:  float = 0.50   # max total short side (50% of gross)

# Common position-construction constants used by orchestrator + UI
MAX_WEIGHT:     float = 0.25
MIN_NET:        float = -0.20
MAX_NET:        float =  1.00
MAX_LEVERAGE:   float =  1.50
TARGET_VOL:     float =  0.10


_DEFAULTS: dict = {
    # Existing
    "auto_execute_stops":     True,   # ATR/drawdown stops auto-execute (Layer 2)
    "auto_execute_entries":   False,  # entry triggers require human approval (Layer 3)
    "monthly_rebalance_auto": False,  # monthly rebalance requires human approval (Layer 3)
    "stop_max_weight_auto":   0.25,   # positions above this weight still need human confirmation
    # P4-2: Tactical patrol config
    "auto_execute_regime_compress": True,   # regime jump → auto-compress all longs (Layer 2)
    "auto_execute_high_conf_entry": False,  # high-confidence entry auto-execute (Layer 2; off during debug)
    "tactical_entry_max_weight":    0.05,   # max weight per tactical entry
    "tactical_entry_daily_limit":   2,      # max Layer-2 entries per day
    "regime_jump_threshold_ppt":    30.0,   # P(risk-off) single-day change threshold (ppt)
    "fast_signal_lookback":         3,      # TSMOM-Fast formation window (months)
    "fast_signal_skip":             1,      # TSMOM-Fast skip period (months)
    "entry_composite_score_min":    60,     # minimum composite score for high-conf entry
    "entry_momentum_zscore_min":    1.5,    # minimum 5-day momentum z-score for high-conf entry
}


def get_trading_config() -> dict:
    """
    Return merged trading config dict.
    Safe to call from engine code — catches all streamlit import errors.
    """
    try:
        import streamlit as st
        cfg = dict(st.secrets.get("trading", {}))
        merged = {**_DEFAULTS, **cfg}
        # Coerce types in case secrets.toml returns strings
        merged["auto_execute_stops"]     = _bool(merged["auto_execute_stops"])
        merged["auto_execute_entries"]   = _bool(merged["auto_execute_entries"])
        merged["monthly_rebalance_auto"] = _bool(merged["monthly_rebalance_auto"])
        merged["stop_max_weight_auto"]   = float(merged["stop_max_weight_auto"])
        # P4-2 tactical
        merged["auto_execute_regime_compress"] = _bool(merged["auto_execute_regime_compress"])
        merged["auto_execute_high_conf_entry"] = _bool(merged["auto_execute_high_conf_entry"])
        merged["tactical_entry_max_weight"]    = float(merged["tactical_entry_max_weight"])
        merged["tactical_entry_daily_limit"]   = int(merged["tactical_entry_daily_limit"])
        merged["regime_jump_threshold_ppt"]    = float(merged["regime_jump_threshold_ppt"])
        merged["fast_signal_lookback"]         = int(merged["fast_signal_lookback"])
        merged["fast_signal_skip"]             = int(merged["fast_signal_skip"])
        merged["entry_composite_score_min"]    = int(merged["entry_composite_score_min"])
        merged["entry_momentum_zscore_min"]    = float(merged["entry_momentum_zscore_min"])
        return merged
    except Exception:
        return dict(_DEFAULTS)


def _bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)
