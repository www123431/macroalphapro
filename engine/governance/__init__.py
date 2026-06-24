"""engine.governance — promotion gateway + approval ledger.

Doctrine (2026-06-02): institutional quant books REQUIRE a human-in-the-loop
gate between "research promotes X" and "X actually runs in production".
Without it, the 0-LLM-in-DECISION boundary is a sieve — research/LLM
proposals leak straight into the live config because nobody enforced
"two-eye + cooling-off".

This package owns:
  - approval_ledger.py  — append-only jsonl ledger of approval requests
  - (future)             promotion_gateway.py — orchestrates SLM promote →
                         create approval → human decision → execution

See project_approval_gateway_2026-06-02 for the architectural memo.
"""
