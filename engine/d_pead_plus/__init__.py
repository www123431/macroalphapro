"""
engine/d_pead_plus/ — Sprint I D-PEAD-Plus LLM sentiment supplement (spec id=74).

Pre-registered: docs/spec_d_pead_plus_llm_sentiment_supplement_v1.md
Hash:           d0532f8fba32c94b677a9a704e164d8aa0ea4cdd  (short: d0532f8f)

Test of LLM-as-INPUT-FEATURE (Pattern 1, Two Sigma class) for D-PEAD baseline
(spec id=62 hash c5d9cd09) augmentation. Adds 5 LLM-extracted features from
earnings call transcripts to deterministic OLS rank, retaining 0-LLM-in-DECISION
doctrine.

Doctrine amendment 2026-05-13 (§〇 of spec): LLM-as-INPUT-FEATURE allowed
in alpha pipeline iff all 6 conditions:
  1. Determinism (temp=0)
  2. Information-isolation (no returns/portfolio/future)
  3. Pure decision layer (no LLM call in decision path)
  4. Hash-lock (prompt + model + temp at registration)
  5. Pre-registration rigor (single-run, falsification chain)
  6. Code-level enforcement (this module's `doctrine` submodule)

Module map:
  doctrine.py            — Invariant enforcement (assert NO_LLM_IN_DECISION_LAYER)
  universe.py             — Top-1500 mcap CRSP point-in-time
  transcripts_loader.py   — WRDS ciq_transcripts fetch + ±5-day rdq linkage
  llm_extractor.py        — Gemini 2.5 Flash adapter; hash-locked prompt; parquet
  feature_combiner.py     — OLS fit on dev; freeze coefficients; OOS predict
  backtest.py             — Walk-forward signal → P&L (mirror Path D structure)
  verdict.py              — 5-gate evaluation; bootstrap CI; NW-t; decision matrix
"""
from engine.d_pead_plus.doctrine import (
    NO_LLM_IN_DECISION_LAYER,
    assert_no_llm_in_decision_layer,
)

SPEC_ID:       int = 74
SPEC_HASH:     str = "6d8e614ebd68ec42d071949bfd4299b0e4a7a363"   # post-amendment 1
SPEC_HASH_SHORT: str = "6d8e614e"
SLEEVE_ID:     str = "ss_sp500"
DOCTRINE:      str = "0-LLM-in-DECISION (amendment 2026-05-13)"

__all__ = [
    "SPEC_ID",
    "SPEC_HASH",
    "SPEC_HASH_SHORT",
    "SLEEVE_ID",
    "DOCTRINE",
    "NO_LLM_IN_DECISION_LAYER",
    "assert_no_llm_in_decision_layer",
]
