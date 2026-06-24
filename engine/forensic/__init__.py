"""
engine.forensic — Forensic analysis tools for DD investigation.

DOCTRINE: This package is OUTSIDE the alpha decision layer. Modules here read
from `engine.portfolio.PaperTradeTradeLog` (Sprint H attribution log) and
may use LLM for narrative summarization, but their outputs NEVER feed back
into strategy / portfolio / backtest decisions. 0-LLM-in-DECISION preserved.

Modules (full pipeline, top to bottom):
  - anomaly_detector:     Cohen-Polk-Vuolteenaho z-score outlier flag
                          (entry point for forensic queue, replaces hand-typed UI)
  - residual_attribution: Brinson-Hood-Beebower decomposition
                          (beta-F + TC + epsilon) — LLM scoped to epsilon
  - news_context:         PRIMARY Gemini 2.5 Flash news summarization
                          (consumed by devils_advocate as the primary leg)
  - devils_advocate:      Dual-LLM (Gemini + DeepSeek) cross-validation
                          for narrative-coherence bias mitigation
                          (Tetlock-Gardner 2015 anchor)
  - audit_chain:          Provenance-locked markdown brief writer with
                          YAML frontmatter (Donoho 2010 reproducibility)
"""
