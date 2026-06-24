"""engine/research/rbg — Research Brief Generator.

Converts a PFH ScoredProposal + (optionally) its materialized output
into a human-readable markdown research brief. The brief's STRUCTURE
is deterministic; LLM is called only to write the prose connecting
evidence cells.

DESIGN DOCTRINES (mirroring engine/research/pfh):

  1. LLM NEVER SCORES. The brief's headline metrics (Sharpe / Bayesian
     prior / posterior CI / cousin warnings) are filled from the
     PFH ScoredProposal verbatim. The LLM's job is to write the
     PROSE that explains those numbers — never to recompute them.

  2. EVERY CLAIM HAS AN EVIDENCE ID. The skeleton enforces that the
     LLM-written prose sections cite specific evidence_ids that
     appear in the PFH evidence chain (paper IDs, library YAMLs,
     graveyard entries). The brief writer validates this post-hoc
     and flags unbacked claims as a sanity violation.

  3. STRUCTURED FALLBACK. If no API key is available, emit the
     evidence skeleton without LLM prose. The brief is still useful
     because the numbers + IDs + paste-able commands are there.

  4. PASTE-ABLE COMMANDS. Every brief ends with a Python block the
     user can copy directly into a REPL to (a) re-materialize the
     factor, (b) check sanity, (c) feed it into council. No
     instructions like "you'll need to set this up" — the commands
     are literal and runnable.

  5. PREDICTED CRITIC CONCERNS. The brief preemptively lists what
     the council's theorist + DA + reflection round will most likely
     ask, derived from the candidate's evidence chain (e.g. high
     cousin_penalty → "DA will ask about graveyard match"; long
     post_pub age → "theorist will probe McLean-Pontiff decay").
"""
from engine.research.rbg.brief_generator import (
    BriefArtifact,
    generate_brief,
    write_brief_to_disk,
)

__all__ = ["BriefArtifact", "generate_brief", "write_brief_to_disk"]
