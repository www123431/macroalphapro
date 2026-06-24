"""
engine/llm/pricing.py — per-model cost computation.

Per-1M-token rates sourced from claude-api skill cache 2026-04-29 +
DeepSeek public pricing (post-promo 2026-05-31). Update when providers
publish new rates.

Cost formula:
  cost = (input_uncached × rate_in
          + cache_read    × rate_cache_read
          + cache_write   × rate_cache_write
          + output        × rate_out) / 1_000_000

Anthropic ephemeral cache: read ≈ 10% of input rate, write ≈ 125%.
DeepSeek has no prompt caching (treat cache_read/write as 0).
"""
from __future__ import annotations


# Per 1M tokens, USD
_PRICING: dict[str, dict[str, float]] = {
    # ── Anthropic (skill cache 2026-04-29) ─────────────────────────────────
    "claude-haiku-4-5":          {"in": 1.0,  "out": 5.0,
                                  "cache_read": 0.1,  "cache_write": 1.25},
    "claude-sonnet-4-6":         {"in": 3.0,  "out": 15.0,
                                  "cache_read": 0.3,  "cache_write": 3.75},
    "claude-opus-4-6":           {"in": 5.0,  "out": 25.0,
                                  "cache_read": 0.5,  "cache_write": 6.25},
    "claude-opus-4-7":           {"in": 5.0,  "out": 25.0,
                                  "cache_read": 0.5,  "cache_write": 6.25},
    # ── DeepSeek (post-promo 2026-05-31 list rates; verified live 2026-05-19) ──
    # V4 Pro promo until 2026-05-31: $0.435 in / $0.87 out — when using promo,
    # caller overrides via use_promo=True (see compute_cost kwargs).
    # DeepSeek DOES support prompt caching: usage.prompt_cache_hit_tokens
    # billed at ~10% of input rate per public docs (no separate cache_write
    # charge — writes are implicit on first request).
    "deepseek-v4-pro":           {"in": 1.74, "out": 3.48,
                                  "cache_read": 0.174, "cache_write": 0.0},
    "deepseek-v4-flash":         {"in": 0.14, "out": 0.28,
                                  "cache_read": 0.014, "cache_write": 0.0},
}

# Promo overrides (currently active)
_PROMO_PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-pro":           {"in": 0.435, "out": 0.87,
                                  "cache_read": 0.0435, "cache_write": 0.0},
}


def compute_cost(
    *,
    model:             str,
    input_tokens:      int,        # uncached input tokens
    output_tokens:     int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    use_promo:         bool = False,
) -> float:
    """Compute USD cost for one LLM call.

    Args:
      model:               full model id (e.g. "claude-haiku-4-5")
      input_tokens:        the UNCACHED portion (Anthropic SDK exposes this as
                           usage.input_tokens — distinct from cache_read/write)
      output_tokens:       completion tokens
      cache_read_tokens:   served from cache at ~10% rate
      cache_write_tokens:  written to cache at ~125% rate
      use_promo:           use promotional rates if available
    """
    if use_promo and model in _PROMO_PRICING:
        rates = _PROMO_PRICING[model]
    elif model in _PRICING:
        rates = _PRICING[model]
    else:
        # Unknown model — return 0 so we don't crash the call, but the cost
        # ledger entry will reflect 0 (caller can audit). Better than guessing.
        return 0.0

    cost = (
          input_tokens       * rates["in"]
        + output_tokens      * rates["out"]
        + cache_read_tokens  * rates["cache_read"]
        + cache_write_tokens * rates["cache_write"]
    ) / 1_000_000.0
    return cost


def supported_models() -> list[str]:
    """Return list of models with known pricing (sanity check helper)."""
    return sorted(_PRICING)
