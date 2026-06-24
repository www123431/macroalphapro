"""engine/line_c — Deep-history earnings-call transcript text-feature alpha line.

Line C goal: extract transcript-derived features (Loughran-McDonald dictionary
scalars + FinBERT tone/embeddings) over the DEEP CIQ history (2011-2024, top-1500
PIT) and test their INCREMENTAL value over SUE through the project gate
(strict OOS + deflated Sharpe + audit battery), with two evaluation lenses:
  Lens A — Fama-MacBeth incremental regressions (statistical power)
  Lens B — decile L/S + deflated Sharpe (economic / deployable significance)

Account split discovered 2026-05-21:
  - ${WRDS_USER_1}  → CIQ transcripts (ciq_transcripts.*) + CRSP + Compustat
  - ${WRDS_USER_2}  → I/B/E/S + CRSP + Compustat, but NO ciq_transcripts (sample only)
secrets.toml says ${WRDS_USER_1} but active pgpass is ${WRDS_USER_2} (stale). Transcript pulls
MUST use ${WRDS_USER_1}; see engine.line_c.wrds_direct.
"""
