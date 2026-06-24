"""
engine/factor_ensemble_crypto — Path N Crypto TSMOM v1 (spec id=71 hash 48db143d).

Pre-registration: docs/spec_path_n_crypto_tsmom_v1.md
Sleeve: crypto_btc_eth (0% initial capital, gated on Path N verdict + Tier 3)
TC: 25 bp roundtrip per event (Binance retail baseline, locked)
Signal: TSMOM 12-1 per Moskowitz-Ooi-Pedersen 2012 / Liu-Tsyvinski 2021
Universe: {BTC-USD, ETH-USD} via yfinance daily close at UTC 00:00
Window: 2018-01-01 to 2026-05-13 (8.4y, BTC+ETH co-active sample)

0 LLM imports — pure deterministic alpha framework.
"""
from engine.factor_ensemble_crypto.tc import TC_BPS_PER_EVENT_LOCKED
from engine.factor_ensemble_crypto.data_loader import (
    UNIVERSE_LOCKED,
    WINDOW_START_LOCKED,
    WINDOW_END_LOCKED,
    load_crypto_panel,
)
from engine.factor_ensemble_crypto.signal import (
    LOOKBACK_MONTHS_LOCKED,
    SKIP_MONTHS_LOCKED,
    compute_tsmom_signal_panel,
)
from engine.factor_ensemble_crypto.walk_forward import (
    SPEC_ID,
    SLEEVE_ID,
    run_walk_forward,
)
from engine.factor_ensemble_crypto.verdict import (
    GATE_PASS_SHARPE_LOCKED,
    GATE_PASS_NW_T_LOCKED,
    GATE_MARGINAL_SHARPE_LOCKED,
    GATE_MARGINAL_NW_T_LOCKED,
    evaluate_verdict,
)

__all__ = [
    "GATE_MARGINAL_NW_T_LOCKED",
    "GATE_MARGINAL_SHARPE_LOCKED",
    "GATE_PASS_NW_T_LOCKED",
    "GATE_PASS_SHARPE_LOCKED",
    "LOOKBACK_MONTHS_LOCKED",
    "SKIP_MONTHS_LOCKED",
    "SLEEVE_ID",
    "SPEC_ID",
    "TC_BPS_PER_EVENT_LOCKED",
    "UNIVERSE_LOCKED",
    "WINDOW_END_LOCKED",
    "WINDOW_START_LOCKED",
    "compute_tsmom_signal_panel",
    "evaluate_verdict",
    "load_crypto_panel",
    "run_walk_forward",
]
