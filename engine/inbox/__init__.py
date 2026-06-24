"""engine.inbox — composite "research inspiration inbox" aggregator.

Doctrine (2026-06-02): traditional "inbox" surfaces become decorative
in systematic books because the engine bypasses them. Repurpose the
mailbox metaphor for what's ACTUALLY useful — daily-relevant signal
aggregated from sources we already have:

  - Daily brief snapshot (engine.daily_batch outputs)
  - Decay sentinel diff vs yesterday
  - New PFH suggestions this week
  - Recent council critique verdicts
  - Capability evidence: new PASS / RED
  - Memory entries written this session
  - DQ Inspector flags

This is the L1 INTERNAL aggregator. L2 (curated external — FRED, EDGAR,
ArXiv RSS) and L3 (LLM curation) live in later modules.

The composer lives in engine.inbox.composer; this package exists so
import paths are clean.
"""
