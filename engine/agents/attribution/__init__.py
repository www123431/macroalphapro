"""engine.agents.attribution — Layer 4 outcome tracking.

JOIN layer (not a new store) over events.jsonl + hypotheses.jsonl +
verdicts.jsonl + resolutions.jsonl + lessons.jsonl + papers cache.

Per project_anti_rut_doctrine_2026-06-07.md and the senior-quant
review on 2026-06-07: outcome tracking is the institutional answer to
'self-evolving system'. Weight-level learning is not viable at this
scale (RAG > fine-tuning empirically for knowledge-intensive tasks);
SYSTEM-level learning via attribution rollups + workflow reweighting
is what D.E. Shaw / Renaissance / Two Sigma actually do.

Modules:
  helpers       — paper_source lookup + author resolution joins
  lifecycle     — get_candidate_lifecycle + aggregate rollups
                  (piece 3b, separate commit)
"""
