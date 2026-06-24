# Spec — News Curator Agent (Wave 8, 2026-05-07) — v2 post-deep-audit

| Field | Value |
|---|---|
| Status | 🟡 SPEC v2 — post-deep-audit, pre-implementation review |
| Author | Wave 8 follow-up to Wave 7 cap-fix restart |
| Reviewer | Supervisor |
| Effort estimate | ~5-6 hours (added Tier R integration + failure-mode tests) |
| Cost projection | ~$2.5/year LLM (1 daily call × 252 trading days × $0.01) |

## v2 changelog (deep-audit findings 2026-05-07)
- **Discovered** `_generate_macro_brief_llm` (daily_batch.py:1622) already exists,
  emitting StructuredMacroBrief (regime narrative paragraph). Clarified that
  news_curator is **complementary not duplicate** — different output format
  (cards) and different consumer (supervisor surface).
- **Adjusted UI** from "tab" to flat **section** in `executive_brief.py`
  (current page is flat-section pattern, not tabbed; tabs would be invasive).
- **Added** Tier R integration: PRODUCTION_CODE_FILES + KNOWN_AGENTS +
  CAPABILITY_REGISTRY entries (§14 new section).
- **Added** 9-item failure mode matrix (§15 new section).
- **Added** 90-day purge job (§16 new section).
- **Reuse** existing `_pool.get_model(response_schema=...)` pattern from
  `engine.key_pool` (proven path used by `anomaly_llm_detector`,
  `auto_audit_proposer`, `debate`, `_generate_macro_brief_llm`).

## 0. Cap-fix-style honesty header

**This is a CONTEXT agent, NOT a FORECAST agent.**

- No `alpha_memory` writes
- No Brier verification
- No "predicted vs realized" scoring loop
- Output is informational metadata for supervisor situational awareness
  + supplementary context for `sector_pipeline` LLM debate
- Cost $2.5/yr supports supervisor news visibility — not predictive alpha

This addresses the meta-audit kill rationale that retired the old
`macro_research weekly` agent on 2026-05-05 ("evaluation theater").
The new agent never produces forecasts to be verified, so the kill
criteria (low-signal forecast accumulation) are structurally inapplicable.

## 1. Problem statement

Current "news vision" of project is broken at L2:

| Layer | Function | State |
|---|---|---|
| L1 ingestion | `engine.news.NewsPerceiver` fetches AV / GNews / spillover, per-sector, live | works but ephemeral (no cache, no DB persistence) |
| L2 curation | LLM-summarised top-N ranked by importance | **missing** |
| L3 consumption | `sector_pipeline.news_context` reads raw fetcher output | works but consumes raw text not structured |

User-facing impact:
- Supervisor has NO single page showing "what's the project paying attention
  to today?" — must drill into individual decisions to see embedded news_context
- `sector_pipeline` LLM debate gets unprocessed text, easy for LLM to miss
  cross-sector themes or lose a 50-headline blob in its context window

## 2. Architecture

```
        ┌─────────────────────────────────────────────────────────┐
        │  L1 (existing): NewsPerceiver fetches per-sector live   │
        │  AV / GNews / spillover_map                             │
        └──────────────────┬──────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────────────────┐
        │  L1.5 (new): aggregate raw headlines across universe    │
        │  call NewsPerceiver for top sectors only (5-7 dominant) │
        │  cap raw headlines at ~80                               │
        └──────────────────┬──────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────────────────┐
        │  L2 (new): NewsCurator LLM agent                        │
        │  inputs: 80 raw headlines + portfolio holdings + regime │
        │  prompt: rank top 10-15, summarise 50w, sector_tag,     │
        │          importance 0-100, macro_themes                 │
        │  output: JSON via response_schema (Gemini structured)   │
        │  persist: curated_news_items rows for the day           │
        └──────────────────┬──────────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
       ┌───────────────────┐  ┌──────────────────┐
       │ Consumer 3a       │  │ Consumer 3b      │
       │ sector_pipeline   │  │ News tab UI      │
       │ news_context      │  │ supervisor       │
       │ (only if available;│  │ (Bloomberg-style │
       │  fallback = old   │  │  card list)      │
       │  per-sector live) │  │                  │
       └───────────────────┘  └──────────────────┘
```

## 3. Schema

```sql
CREATE TABLE curated_news_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    curated_at      DATETIME    NOT NULL,            -- LLM completion time
    curation_run_id VARCHAR(36) NOT NULL,            -- UUID per daily curate
    headline        TEXT        NOT NULL,
    summary_50w     TEXT        NOT NULL,            -- LLM ≤80-word summary
    source          VARCHAR(40) NOT NULL,            -- "GNews" / "Alpha Vantage" / etc.
    source_url      VARCHAR(800),                    -- original story link
    sector_tag      VARCHAR(50),                     -- must ∈ known universe sectors OR "macro"
    ticker_tag      VARCHAR(20),                     -- optional, ETF-specific
    importance      INTEGER     NOT NULL,            -- 0-100 LLM-assigned
    macro_themes    TEXT,                            -- JSON array, e.g. ["fed_pivot", "earnings"]
    sentiment_score REAL,                            -- LM-dictionary derived [-1, +1]
    raw_headline_id INTEGER                          -- optional FK if we cache raw too
);
CREATE INDEX ix_curated_news_run        ON curated_news_items (curation_run_id);
CREATE INDEX ix_curated_news_sector_at  ON curated_news_items (sector_tag, curated_at DESC);
CREATE INDEX ix_curated_news_imp_at     ON curated_news_items (importance DESC, curated_at DESC);
```

## 4. Curator agent contract

### Entry point
`engine/agents/news_curator.py`:

```python
def run_news_curator(
    model,                        # Gemini client (None → degrade to no-LLM placeholder run)
    universe_sectors: list[str],  # current 16 holdings' sectors
    regime_label: str,            # "risk-on" / "transition" / "risk-off"
    max_raw_headlines: int = 80,
    max_curated_items: int = 15,
) -> dict:
    """
    1. For each top-N sector (by current weight), call NewsPerceiver.fetch.
    2. Aggregate up to max_raw_headlines raw items.
    3. LLM curates: structured JSON via Gemini response_schema.
    4. Persist curated rows; return {curation_run_id, n_curated, ...}.
    Idempotent: if a run with today's date already exists, skip + return
    existing run_id.
    """
```

### LLM prompt (schema-enforced JSON)

```
You are a news curator for a quantitative macro portfolio.
Current holdings: {sectors}.
Current regime: {regime}.

Below are {N} raw headlines fetched in the last 48 hours.
Rank by impact on the holdings, output the top {K} items.

For each item provide:
- headline: original headline (verbatim)
- summary_50w: ≤80 words; cite source domain inline
- sector_tag: MUST be one of {sector_list} or "macro" if cross-sector
- importance: 0-100; calibrate so distribution is roughly tertile
  (≤4 above 70, ≤8 between 40-70, rest below 40)
- macro_themes: 1-3 short tags (e.g. "fed_pivot", "opec_cut",
  "earnings_szn", "geopolitical")

Strict rules:
- Never invent details not in the source headline / snippet.
- If sector_tag candidate not in {sector_list} ∪ {"macro"}, drop the item.
- summary_50w must be specific to the headline, not generic prose.
```

### Output JSON schema (Gemini response_schema)

```json
{
  "type": "object",
  "properties": {
    "curated": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "headline":     {"type": "string"},
          "summary_50w":  {"type": "string"},
          "sector_tag":   {"type": "string"},
          "ticker_tag":   {"type": "string", "nullable": true},
          "importance":   {"type": "integer"},
          "macro_themes": {"type": "array", "items": {"type": "string"}},
          "source":       {"type": "string"},
          "source_url":   {"type": "string", "nullable": true}
        },
        "required": ["headline", "summary_50w", "sector_tag", "importance", "source"]
      }
    }
  },
  "required": ["curated"]
}
```

### Validation gate (defensive — Wave 6 8-facet rule 7)

After LLM response, drop any item where:
- `sector_tag ∉ universe_sectors ∪ {"macro"}` (anti-hallucination)
- `importance < 0 or > 100` (out-of-range)
- `len(summary_50w.split()) > 100` (too verbose; LLM dragged on)
- `source_url` present but malformed (regex-validated http(s)://)

If after validation `len(curated) < 3`, mark run `degraded` and skip persist.

## 5. Daily trigger

In `engine.daily_batch.ensure_daily_batch_completed`, near top of the
post-batch agent block:

```python
# Wave 8 (2026-05-07): news curator runs FIRST in agent block so
# downstream sector_pipeline can read today's curated rows.
if model is not None:
    try:
        from engine.agents.news_curator import run_news_curator
        from engine.portfolio_tracker import get_current_positions
        _pos = get_current_positions()
        _sectors = list(_pos["sector"].dropna().unique()) if not _pos.empty else []
        _regime = regime_result.regime if regime_result else "transition"
        run_news_curator(model=model, universe_sectors=_sectors,
                         regime_label=_regime)
    except Exception as exc:
        logger.warning("news_curator non-fatal: %s", exc)
```

Idempotency: agent's own dedup checks if run_id for today exists; safe re-run.

## 6. UI — Brief page News section (v2: flat section, NOT tabs)

`pages/executive_brief.py` is flat-section layout (`st.subheader` +
`st.markdown("---")` dividers between HEADLINE / RISK PULSE / ATTENTION).
Adding tabs would be an invasive structural refactor.  **Add a 4th flat
section** at end of page (after ATTENTION):

```python
# ── existing 3 sections render above ─────────────────────────────────────
# st.subheader("Headline")  ...
# st.markdown("---") ; st.subheader("Risk Pulse")  ...
# st.markdown("---") ; st.subheader("What needs your attention") ...

# ── Wave 8 new section ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("News Digest")
st.caption("LLM-curated headlines · ranked by importance to current portfolio")
_render_news_digest()       # function in same file or imported helper
```

Render contract:
- Read latest `curation_run_id` from `curated_news_items` (single SQL query;
  cache for 60s via `@st.cache_data`)
- Filter widget: sector_tag multiselect (default = all) + importance min
  slider (default = 50)
- Each item rendered as Bloomberg-style card (mono font / right-aligned
  score badge / source URL `[link]` clickable, opens new tab via
  `st.markdown(...unsafe_allow_html=True)`)
- Sort by `importance DESC, curated_at DESC`
- Empty state: "no curated news yet for today; daily batch runs at first
  trading-hour"
- Failure-mode display: if curated count < 3 (degraded run), show banner
  "curator degraded today; check daily_batch logs"

## 7. sector_pipeline consumer upgrade (deferred)

**Out of Wave 8 scope.** Current `_build_news_context` does per-sector live fetch
which still works. After 1-2 weeks of curated_news_items accumulation, replace
with curated reads (saves 14 NewsPerceiver fetches per sector_pipeline run).

Reasoning: ship UI value first; optimize sector_pipeline second.

## 8. Test plan (cascade audit T2 step 4)

| Test | What | Expected |
|---|---|---|
| 1 | Mock-LLM unit test: prompt structure | JSON schema valid; required fields present |
| 2 | Mock-LLM with hallucinated sector_tag | validation drops the row |
| 3 | Mock-LLM with importance > 100 | validation drops the row |
| 4 | Idempotency: run twice in same day | second call skips, returns existing run_id |
| 5 | Live LLM smoke (manual, ~$0.01) | inspect actual JSON for cleanliness |
| 6 | UI smoke | Brief page News tab renders with mock data; no exception |
| 7 | Tier R critical sweep | 11 rules / 0 findings unchanged |
| 8 | 22-page smoke | 22/22 PASS with new tab |

## 9. Migration & rollback

- Schema migration: idempotent ALTER (skip if `curated_news_items` exists)
- Rollback: drop the table + revert daily_batch hook + remove tab
- Cost recovery if killed: ~$0 (no expensive backfill)
- Hash chain: unaffected (table not in chain payload)

## 10. Spec amendment ledger

After implementation, register via:
```
amend_spec(
    path="engine/agents/news_curator.py",
    kind="capability_extension",
    reason="Wave 8: NewsCurator agent — context-only, not forecast. "
           "Daily LLM curation of universe headlines for supervisor News tab "
           "+ sector_pipeline upstream context. Architecturally distinct from "
           "killed macro_research weekly (no alpha_memory writes, no Brier "
           "verification loop). Cost ~$2.5/year. Spec doc: "
           "docs/decisions/spec_news_curator_2026-05-07.md."
)
```

## 11. Reference (academic anchors)

- Tetlock 2007 — "Giving Content to Investor Sentiment" (negative news → return)
- Loughran & McDonald 2011 — financial sentiment lexicon (sentiment_score derivation)
- Engelberg, Reed, Ringgenberg 2018 — structured news features
- Ke, Kelly, Xiu 2019 — JFE "Predicting Returns with Text Data"

## 12. Open questions for supervisor review

1. **Tab placement**: Brief sub-tab (recommended) vs standalone `pages/news_digest.py` in 今日 group?
2. **Importance distribution**: enforce tertile (4/8/3 ratio above/between/below 50) or let LLM decide?
3. **History retention**: keep all curated_news_items forever, or auto-purge after 90 days?
4. **API keys**: AV_KEY / GNEWS_KEY currently set in `.streamlit/secrets.toml`? If absent, fall back to GNews-only?

## 13. What we're explicitly NOT doing (scope discipline)

- ❌ Per-article BERT sentiment (out of scope; LM dictionary lite version OK)
- ❌ Real-time stream (daily refresh sufficient for monthly-rebal portfolio)
- ❌ Sector_pipeline rewrite (deferred; Wave 8 only adds Brief section)
- ❌ Forecast verification loop (this is the kill-criteria firewall)
- ❌ Alternative data integration (twitter / satellite / etc.)
- ❌ Replace `_generate_macro_brief_llm` (different purpose; both coexist)


## 14. Tier R / capability infrastructure integration

| Registry | Add | Reason |
|---|---|---|
| `engine/auto_audit_rules.py` `PRODUCTION_CODE_FILES` | `engine/agents/news_curator.py` | drift detection on agent code changes |
| `engine/auto_audit_rules.py` `CAPABILITY_REGISTRY` | `news_curator_daily` (table=curated_news_items, cadence=1 day, skip_until_age_d=7) | Wave 5 rule flags if curator goes silent |
| `scripts/audit_agent_liveness.py` `KNOWN_AGENTS` | `{agent_id: 'news_curator', downstream_table: 'curated_news_items', expected_cadence_days: 1, no_agent_class: True}` | liveness probe |
| `engine.preregistration` amendment ledger | `kind='capability_extension', reason='Wave 8 spec_news_curator_2026-05-07'` | scientific discipline |


## 15. Failure mode matrix (9 cases tested in §8 test plan)

| # | Failure | Detection | Behaviour |
|---|---|---|---|
| F1 | AV_KEY / GNEWS_KEY both missing | NewsPerceiver returns empty list | log warning; skip LLM call; persist no rows; UI shows empty state |
| F2 | One of two news APIs down (5xx) | NewsPerceiver internal try/except | use other; partial fetch OK |
| F3 | All sectors return 0 headlines | aggregated raw_headlines=[] | skip LLM; mark run degraded |
| F4 | LLM JSON malformed | `json.loads` raises | log + skip persist; do not crash batch |
| F5 | LLM returns hallucinated sector_tag | validation gate (§4) | drop the row; if curated count < 3 mark degraded |
| F6 | Empty portfolio (0 positions) | `get_current_positions()` empty | fallback `universe_sectors` from UniverseETF |
| F7 | Same-day rerun (idempotency) | check existing `curation_run_id` for today | return existing; do not re-LLM |
| F8 | LLM timeout (>30s) | wrap `generate_content` with timeout | log + skip persist |
| F9 | Old data accumulation (>90 days) | monthly purge job (§16) | DELETE WHERE curated_at < cutoff |


## 16. 90-day purge job

In `engine.daily_batch` near the end (after sector_pipeline), idempotent:

```python
# Wave 8: prune old curated_news_items (90-day retention)
if t_day.day == 1:   # first day of month
    try:
        from engine.memory import SessionFactory, CuratedNewsItem
        cutoff = t_day - datetime.timedelta(days=90)
        with SessionFactory() as s:
            n = (s.query(CuratedNewsItem)
                  .filter(CuratedNewsItem.curated_at < cutoff)
                  .delete())
            s.commit()
            logger.info("news_curator purge: removed %d rows older than %s", n, cutoff)
    except Exception as exc:
        logger.warning("news_curator purge non-fatal: %s", exc)
```

Cost: O(N) DELETE on indexed column; ~0 DB impact.


## 17. Coexistence with `_generate_macro_brief_llm`

| | macro_brief_llm (existing) | news_curator (Wave 8 new) |
|---|---|---|
| Output format | 2-3 sentence Chinese paragraph | list of 10-15 cards |
| Schema | `StructuredMacroBrief` (regime_assessment / key_driver / tail_risk / brief_text) | `curated_news_items` rows |
| Storage | `daily_brief_snapshots.macro_brief_llm` (TEXT column) | new table `curated_news_items` |
| Consumer | Brief page HEADLINE area | Brief page NEWS DIGEST section + sector_pipeline (deferred) |
| LLM cost / day | ~$0.01 | ~$0.01 |
| Cadence | daily | daily |

Both coexist. They serve different supervisor needs:
- macro_brief_llm = "what's the regime story today (one paragraph)"
- news_curator = "what specific news items matter today (cards)"


## 18. 8-facet self-audit gating (post-implementation, before merge)

| Facet | Test |
|---|---|
| 1. Edit verification | grep new agent file + new table + new section + Tier R additions, all present |
| 2. Sibling pattern hunt | check no other LLM-call site got accidentally duplicated; verify `_pool.get_model` is the ONLY Gemini access path used |
| 3. Regex residue | `news_curator` import location uses `engine.agents.news_curator`, not stale path |
| 4. Helper edge cases | each F1-F9 failure mode tested with mock |
| 5. Sibling write paths | only news_curator writes to curated_news_items (no leak from other paths) |
| 6. Stored data migration | empty start state OK; no migration needed |
| 7. Downstream consumer | Brief page renders with 0 rows / 1 row / 15 rows / degraded all OK |
| 8. Reverse derivation | given a curated row, can identify its raw NewsPerceiver source via `source` + `source_url` |
