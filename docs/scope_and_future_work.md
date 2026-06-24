# Scope & Future Work

> **What this document is**: An honest accounting of what this project is and isn't,
> what it can and cannot demonstrate, and what would have to change for the architecture
> to scale into production-grade institutional alpha generation. Written per
> [`feedback_quant_perspective.md`](../memory/feedback_quant_perspective.md) — boundaries
> disclosed proactively, not after reviewer challenge.

---

## What this project is

**An architecture prototype** demonstrating production-grade agentic AI engineering
patterns applied to a high-stakes sequential decision domain (quantitative trading).
Its scientific contribution is:

1. A **complete falsification chain** documenting how to honestly reject alpha hypotheses
   without retroactive parameter rescue
2. **Methodology rules** (spec power analysis, layer boundary discipline, lookahead
   bias 4-tier protocol) emerged from concrete project incidents and encoded as
   standing rules
3. **Production-ready infrastructure** for multi-agent coordination, memory persistence,
   automatic factor mining, and self-learning loops — verified end-to-end in real runs

## What this project is NOT

- **Not a profitable trading strategy**: production baseline (TSMOM Sharpe 0.39–0.51)
  is classical 1990s CTA alpha; agentic-AI overlay was rigorously falsified
- **Not an attempt to compete with institutional quant**: explicitly scope-bounded to
  retail-accessible data and a small ETF universe
- **Not novel methodology research**: methodology builds on well-cited published work
  (López de Prado, Bybee-Kelly-Manela-Xiu, Schölkopf, Zheng et al.); this project's
  contribution is *integration* and *enforcement discipline*, not new statistical methods
- **Not battle-tested in live trading**: paper-trading infrastructure exists but has
  not yet accumulated the calendar-bound 6-12 months of forward verification data

---

## Documented limitations

Per the falsification chain, the rejected hypotheses share a common physical limit.
Each constraint is recorded here so future readers / reviewers / maintainers know
where the architecture's claim ends.

### L1 — Universe size

**Constraint**: ~35 ETF (sector + cross-asset, US-listed; dynamic point-in-time filtering via
[`engine/universe_audit.py`](../engine/universe_audit.py))

**Effect**:
- Cross-section is small relative to institutional standard (Russell 3000 = 3,000)
- Sectors are highly correlated (tech-cluster, defensive-cluster, rate-sensitive-cluster)
- Effective independent rank dimensions ~ 5-7
- Cross-sectional alpha mining (factor IC, ICIR detection) requires ICIR ≥ 0.30 for
  acceptable power on 25-30 monthly cross-section observations — unreachable for
  realistic factor effect sizes (literature: ICIR 0.05-0.20)

**Why this constraint exists**:
- yfinance is the only free reliable data source; ETF data is well-maintained,
  individual stock data has corporate-action complexity (splits, dividends, mergers,
  spinoffs, bankruptcies) that requires paid CRSP/Refinitiv/FactSet for PIT-correct
  handling
- Master-thesis scope: stock universe migration estimated 13-21 weeks (3-5 months)
  excluding data subscription cost, exceeding deliverable timeline

### L2 — Rebalance frequency

**Constraint**: Monthly rebalance only

**Effect**:
- HFT and institutional algos absorb narrative shocks within seconds-to-days; monthly
  strategies receive only residual drift
- Bybee-Kelly-Manela-Xiu (2023, RFS) strict OOS narrative alpha is 2-4% annual
  (ΔSharpe 0.05-0.10) — exactly the regime where monthly NW HAC SE is too inflated
  for confirmatory detection
- ATR-based monthly transaction cost modeling does not capture intraday market impact

**Why this constraint exists**:
- Daily/intraday rebalance requires ETF intraday data (Polygon / IB) and full bar/tick
  feed for proper market impact modeling
- Daily strategies need different signal families (1-3 day mean reversion, event-driven,
  microstructure) that don't generalize from monthly TSMOM
- Engineering rewrite for daily would touch signal layer, vol estimation, portfolio
  construction, transaction cost model, and orchestrator — estimated 4-8 weeks

### L3 — Data sources

**Constraint**:
- Price: yfinance (free, monthly close, no point-in-time survivorship)
- Macro: FRED public CSV (no API key required, daily updates)
- News: Alpha Vantage (free tier) + GNews (free tier) + RSS fallback
- Shocks: GPR/EPU/NVIX-proxy (NVIX original Manela source 404; VIX month-end as proxy)
- LLM: Gemini 2.5 Flash via project key pool

**Effect**:
- No fundamental data (P/E, BV, EPS, ROE, accruals, etc.) → cannot test value/quality
  factor families
- No PIT survivorship-free historical universe → backtests inherit survivorship bias
  (mitigated for ETF via `universe_audit.ETF_INCEPTION` but cannot eliminate)
- News quality skewed toward English-language US-centric coverage
- NVIX-VIX proxy substitution introduces residual collinearity with VIX baseline
  (degrades narrative-NVIX independent information)

**Why this constraint exists**:
- All paid alternatives ($5K-50K/year for CRSP / Refinitiv / FactSet) exceed student
  project budget
- Free alternatives have inherent limits documented above

### L4 — Sample size

**Constraint**: 88-192 month OOS windows depending on test

**Effect** (from D1.1 power analysis):
- For ΔSharpe = 0.10 detection at threshold NW t = 1.5 with HAC inflation 1.3 on
  n=192: **power = 9%**
- For ΔSharpe = 0.10 at threshold NW t = 1.0: **power = 24%**
- 80% power at ΔSharpe = 0.10 requires n ≈ 250-300 months (21-25 years)

**Why this constraint exists**:
- Most ETFs in universe inception ≥ 2002 (XLE/XLF/XLI/XLV/XLP/XLY 1998-12; SHY/TLT
  2002-07; LQD 2002-07; QQQ 1999-03; SMH 2000-05). 192-month window already nearly
  maximal for current ETF coverage
- Going to 252+ months requires falling back to mutual fund proxies for ETFs that
  didn't exist pre-2003, introducing data-quality issues

### L5 — Cost / risk model

**Constraint**:
- Transaction cost: ATR-based turnover × 5 bps fixed half-spread
- No market impact (Almgren-Chriss / linear)
- No borrow cost for short positions
- No tax modeling (short/long term distinctions)
- No liquidity constraints (capacity assumed unlimited at ETF level)

**Why this constraint exists**:
- ETF universe has high liquidity and low cost on retail scale; for the architecture
  prototype, simple cost model adequate
- Production trading infrastructure (FIX connectivity, order routing, position
  management) is out of scope

### L6 — Hardware / deployment

**Constraint**: Single-machine, Streamlit UI, SQLite persistence

**Effect**:
- Not designed for HFT-scale latency requirements
- EventBus is in-process (not Kafka / Redis distributed)
- Memory + spec system is file/SQLite-based (not enterprise-grade)
- Concurrent user support limited

**Why this constraint exists**:
- Architecture prototype scope; demonstrating patterns, not deployment infrastructure

---

### L7 — HITL governance is single-supervisor (2026-05-05)

**Constraint**: One supervisor (project author) makes all governance approval
decisions across the 6 approval categories (cash_flow / spec_amendment /
risk_control kill switch / strategy_arm_toggle / universe_change /
anomaly_screener post-S6).

**Effect**:
- No model committee / risk committee / compliance committee separation
- No quorum / voting / dissent-recording mechanism
- Self-approval on spec_amendment is theoretically vulnerable to HARKing
  bypass (mitigated by the 4-rule HARKing detector and append-only amendment
  ledger, but not eliminated)

**Why this constraint exists**:
- Master's-project scope; production-grade fund operation requires a 3-5
  person committee per industry practice (Renaissance / Two Sigma / Citadel
  publicly documented model + risk committee structures)

**What would relax it**:
- Production deployment with a real fund / family office / institutional
  client mandates committee setup; Tier 5 future work item

### L9 — LCS legacy module deprecated (2026-05-05)

**Status**: `engine/lcs.py` is deprecated as of 2026-05-05 (B-pragmatic-v2
Tier 1 retroactive audit, B-revised remediation). The single production
entry point `engine/memory.py:_run_lcs_on_decision()` is a no-op stub with
deprecation docstring; the lcs.py file is preserved for historical audit
trail with a top-level deprecation banner.

**Why deprecated**:
- LCS v1 used `model.generate_content()` (LLM) for Mirror / Noise /
  Cross-Cycle consistency tests, which violates the 0-LLM-in-evaluation
  invariant codified in
  [docs/decisions/llm_3layer_architecture_2026-05-05.md](decisions/llm_3layer_architecture_2026-05-05.md) §3 Invariant 1
- A deterministic v2 refactor was specced
  ([docs/spec_lcs_deterministic_v2.md](spec_lcs_deterministic_v2.md),
  now SUPERSEDED) but inspection of `engine/portfolio.py` revealed the
  project's actual ranking rule is `direction = sign(raw_return_12m_skip_1m)`
  — a deterministic monotonic sign function. Mirror property holds by
  construction (mathematically trivial); noise stability is trivial away
  from zero (project upstream filters `tsmom != 0`). The C-refactor would
  add zero marginal information.
- SkillLibrary write-back, LCS's historical downstream consumer, is now
  gated by `engine.memory_curator.run_monthly()` with Benjamini-Hochberg
  FDR (deterministic, independent of LCS).

**Effect**:
- `DecisionLog.lcs_passed` column preserved (historical audit trail). New
  decisions verified after 2026-05-05 have `lcs_passed = NULL`. Downstream
  semantics treat NULL as pass (existing convention from L9 v1 deployment).
- Tier 1 audit Claim 2 (0-LLM-in-evaluation red line) verified PASS.
- 0 production code path now contains LLM in Layer 2 evaluation.

**What would un-defer it**: a future research need for an LLM
consistency-audit gate would require formally specifying it in the v2 LLM
3-layer architecture (Layer 1 only), pre-registering via SpecRegistry,
and demonstrating marginal information beyond memory_curator BH FDR.
Current project does not have such a need.

### L8 — Pre-registration hash chain is local-only (2026-05-05)

**Constraint**: SHA-256 narrative hash chain for approved decisions (P-AUDIT
v1 NARRATIVE-C v2) is stored only in the local SQLite `pending_approvals`
table. No external time-stamp deposit (OSF.io / SSRN preprint / blockchain
anchor).

**Effect**:
- Third-party verifiability is weaker than Lakatos-grade pre-registration
  standard; theoretically the hash chain could be reconstructed by a
  motivated adversary with database write access
- Not blocking thesis defense (academic standard for student project), but
  cannot be cited as fully tamper-proof for institutional review

**Why this constraint exists**:
- B-pragmatic-v2 sprint scope (HITL slim refactor) chose to defer external
  deposit rather than block this sprint on additional infrastructure

**What would relax it**:
- 1-2 hour task: register OSF.io project, push spec_hash + amendment ledger
  to it, store OSF DOI in each amendment row. Not done in B-pragmatic-v2 to
  preserve sprint focus.

---

## Future work roadmap

The architecture is designed to scale. Each future-work item below is a *concrete*
engineering project with estimated cost; this is not vague aspiration but realistic
plan if resources become available.

### Tier 1 — Architecture validation (no new alpha hypothesis required)

| Task | Cost | Value |
|---|---|---|
| **F: Forward-only paper trading** — accumulate 6-12 months of forward live data on production baseline + (cancelled) narrative gate; validate calendar-bound clean zone | 1 week setup + 6-12 months wait | Real Clean Zone evidence; could shift D1.1 verdict via forward augmentation |
| **SSRN methodology note** — "Spec Power Analysis as a First-Class Discipline in Quant Strategy Validation: A Case Study" | 1-2 weeks | Officially "Published" credential; SSRN is non-peer-reviewed but Google-Scholar indexed |
| **Demonstration video** (10 min walkthrough of falsification chain + architecture) | 2-3 days | Resume / portfolio asset |
| **Standalone Streamlit cloud deployment** with read-only public access | 1 week | Reviewer / interview demo |

### Tier 2 — Universe migration (architecture stress test)

| Task | Cost | Value |
|---|---|---|
| **Stock universe minimal prototype**: S&P 100, monthly, single TSMOM signal, LedoitWolf covariance | 1-2 weeks | Single test of whether universe-size alone changes alpha detection (do not extrapolate to full Russell 3000 system) |
| **Full Russell 3000 + monthly migration**: PIT historical universe, corporate action handling, fundamental data integration, factor library expansion, beta/sector neutral portfolio construction, transaction cost model upgrade | 13-21 weeks (3-5 months) + $5K-50K data subscription | Production-grade scope; could realistically detect ICIR 0.05-0.10 alpha; requires either funding or institutional sponsorship |
| **Cross-asset extension** (commodities, currencies, fixed income beyond ETF wrappers) | 4-6 weeks per asset class | Diversification of alpha sources; validates architecture across regime types |

### Tier 3 — Frequency / signal expansion

| Task | Cost | Value |
|---|---|---|
| **Daily rebalance pipeline** (signal layer rewrite + intraday market impact model + fast-TSMOM signals) | 4-8 weeks + intraday data feed | Different signal universe (mean reversion, event-driven); high engineering risk |
| **Event-driven trading**: earnings calls / FOMC announcements / SEC filings as triggers, with LLM extraction of forward guidance / risk factor changes | 6-10 weeks | Could be where LLM contributes real alpha (Bianchi 2024, Chen 2025 forward guidance work) |
| **Tick-level micro-strategies** (rebalance at minute frequency, statistical arbitrage between ETFs) | 12-20 weeks + tick feed | Out of architecture's current design intent; not recommended |

### Tier 4 — LLM capability expansion

| Task | Cost | Value |
|---|---|---|
| **D2 — LLM shock-type extraction** (Phase 1 in original spec hierarchy); LLM extracts {war, election, pandemic, monetary_surprise, ...} categorical labels from raw macro news; feeds into deterministic risk gate | 4-7 weeks (must include all 4 lookahead-bias mitigation per [`feedback_llm_lookahead_bias.md`](../memory/feedback_llm_lookahead_bias.md)) | Validates whether LLM categorization adds incremental signal beyond GPR/EPU/NVIX z-scores; conditional on D1 alpha existing (currently rejected) |
| **Cross-cutoff LLM lookahead empirical study**: same prompt across GPT-4 / Claude 4.5 / Claude 4.7 / Gemini 2.5; measure alpha as function of training cutoff date | 2-3 weeks | Significant academic contribution; very few papers do this rigorously |
| **Earnings transcript / 10-K risk-section LLM extraction** for stock-level signals | 4-6 weeks per signal family | Real LLM edge per Lopez-Lira 2023 critiques |

### Tier 5 — Production deployment

| Task | Cost | Value |
|---|---|---|
| **Distributed EventBus** (Kafka/Redis migration from SQLite) | 2-4 weeks | Multi-machine deployment; latency improvement |
| **Postgres migration** from SQLite (concurrent users, larger schemas) | 2-3 weeks | Enterprise-grade persistence |
| **Real broker integration** (Interactive Brokers / Alpaca FIX or REST API) | 4-6 weeks | Live trading capability |
| **Risk monitoring dashboard** with real-time VaR / ES / position limits | 3-4 weeks | Operations-grade observability |
| **Backup, audit trail, regulatory compliance** infrastructure | 4-8 weeks | Required for any commercial deployment |

### Tier 6 — Deferred refactor candidates (lazy evaluation)

These are improvements considered and **deliberately deferred** until a concrete v2 use case justifies the schema-level cost. Following YAGNI / Open-Closed Principle / Postel's Law, we keep the core enum stable and handle the surface-level need at the UI layer first.

| Deferred refactor | Why deferred | What would un-defer it |
|---|---|---|
| **`PendingApproval.approval_type` enum split**: current 5 types (`entry / risk_control / rebalance / universe_change / cash_flow`); proposed split of `risk_control` into `position_close / position_reduce / risk_alert / regime_derisk` (~3h optimistic, 5-6h realistic with collateral fixes). | Currently solved at UI layer (P-AUDIT v1 amendment 2026-05-04 §F-pre M1+B): `triggered_condition` substring match → display sub-label "止损待批 / 制度避险 / 波动告警 / 信号衰减". Achieves 80% of supervisor disambiguation benefit, zero schema risk, zero backfill cost on 11 historical `risk_control` rows. Project codebase has demonstrated raw-query inconsistency (sentinel + global-MAX bug, 2026-05-04, 5 sites repro'd same fix); each schema migration carries non-trivial collateral risk. Iyengar-Lepper 2000 *choice overload* also argues against expanding enum from 5 → 8 without driving use case. | A concrete v2 analysis use case requiring SQL `WHERE approval_type = 'position_close'` granularity (e.g., per-sub-type hit-rate analysis, or per-sub-type reviewer routing). Until then, sub-type signal lives in `triggered_condition` text and can be `regex`-extracted on demand for ad-hoc analysis. |

---

## Why this scope is right for what this project is

Per [`project_positioning.md`](../memory/project_positioning.md) **dual goal**
(2026-05-02 Update):

### Agentic AI engineering goal

ETF universe + monthly rebalance + free data is **scope-appropriate** because:

1. The architecture demonstration is the value, not the alpha number
2. Smaller universe = clearer agent traces, cleaner evidence for capability claims
3. Free data + ETF means anyone can reproduce the system without subscription cost
4. Single-machine SQLite means setup-friendly for reviewer/interviewer evaluation
5. Documented limitations + future-work roadmap show understanding of what *would*
   change for production scale

### Alpha generation goal

The reject chain (Phase 0 → D1 → D1.1 → FactorMAD Q1) **is the alpha-generation
contribution**. Per [`feedback_alpha_hard_polish_easy_drift.md`](../memory/feedback_alpha_hard_polish_easy_drift.md):
when alpha is hard, the discipline is to falsify cleanly, not to drift into engineering
escalation.

The honest answer to "did you produce alpha?" is:
- TSMOM baseline produces classical CTA alpha (Sharpe ~0.4-0.5 monthly), not novelty
- Agentic-AI narrative overlays were tested, all rejected
- Future work tier 2 migration could realistically extend this; current scope cannot

This honest framing is **valuable in itself** for academic and recruitment contexts
that distinguish careful researchers from those who manufacture results.

---

## Decision authority for future work

Per [`memory/MEMORY.md`](../memory/MEMORY.md) cross-session continuity:

- **Tier 1**: Can be initiated by author within current project timeline; no external
  resources required
- **Tier 2 minimal prototype**: Within timeline if author allocates 1-2 weeks
- **Tier 2 full migration / Tier 3 / Tier 5**: Require external resources (data
  subscription, funding, team); not pursued within current project scope
- **Tier 4 LLM capability expansion**: Conditional on Tier 1 F (paper trading) producing
  any indication that narrative-driven alpha exists in expectation; otherwise rejected

This is a hard scope boundary. Reviewers should evaluate the project against what is
delivered (architecture + falsification chain + methodology rules + production-ready
infrastructure), not against future-work aspirations.

---

## Versioning

This document is updated when:
- New limitations are discovered
- New future work items become viable
- Existing future work is started, completed, or abandoned

Cross-reference [`memory/MEMORY.md`](../memory/MEMORY.md) for live project state.

Last update: 2026-05-03
