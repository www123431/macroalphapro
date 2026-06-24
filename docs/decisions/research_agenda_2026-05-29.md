# Research Agenda — Forward Roadmap (2026-05-29)

**Author**: Claude Code session, post 3-mechanism book deployment
**Audience**: future sessions (human + AI), to provide a north-star without
forcing replay of this session's reasoning chain
**Status**: living document; update at each session boundary

---

## 0. Honest current state

**Book**: Sharpe ~1.03 (IS, 99-month overlap 2016-2024) / equivalent OOS unknown
(forward live = 15 days only); MaxDD -6.49%; Calmar 1.27. 3 mechanism families:

| Sleeve | Weight | Sharpe (sleeve) | Mechanism |
|---|---|---|---|
| Equity (D-PEAD + analyst revision) | 70% | ~1.03 | Earnings information underreaction |
| Carry (4-leg cmdty/fx/rates_us/rates_xc) | 25% | 1.10 IS / 0.83 OOS | Cross-asset roll yield |
| TSMOM (5-leg + eqidx) | 5% | 0.62 net | Time-series momentum |

**Infrastructure shipped this session**:
- MSM regime overlay deprecated (ablation -0.26 Sharpe penalty)
- TSMOM added as 3rd mechanism with strict-gate evidence
- Alpaca paper-trade live (114 orders queued); SimAdapter multi-venue fallback
- vol-target tuning verified (10% is optimal)
- strict-gate doctrine codified in [[feedback-strict-gate-no-lowering-2026-05-28]]

**Where the book is in the institutional landscape**:
- Sharpe 1.0 ≈ institutional "fund-worthy" floor. 1.5+ is exceptional.
- 3 mechanisms is *light*. AQR, Bridgewater, Renaissance run 10-50+ orthogonal sleeves.
- The book is a "smart-beta + futures alpha" config, NOT yet a true multi-mechanism shop.
- 99-month sample is short — real conviction requires 10-20+ year evidence.
- Alpaca live = 0 day historical observation (just submitted today).

**The honest read**: we have a deployable v1, NOT a finished product. There is
substantial defensible work remaining on both the quant alpha side AND the
infrastructure side.

---

## I. Quant alpha gaps (real, non-cargo-cult)

### I.A — Equity factor diversity (PRIORITY: HIGH)

D-PEAD + revision are both in the **earnings-information family**. They share
mechanism (information arriving slowly into prices). When that family decays
(and it will, partially), 70% of the book degrades simultaneously.

**Genuinely orthogonal equity mechanisms still untested in this codebase**:

| Mechanism | Status here | Why it matters |
|---|---|---|
| Quality (Asness-Frazzini-Pedersen "QMJ") | NOT tested | Persistent, non-arbitraged across decades |
| Value (multi-metric: EBITDA/EV + FCF/EV + B/M) | partial (HML in factor regression only, not as own sleeve) | Anchor anomaly, post-1992 sample still works |
| Residual momentum (Blitz-Huij-Martens 2011) | NOT tested | Decorrelated from raw momentum; FF5+UMD-orthogonal |
| Short-term reversal (1m, Jegadeesh 1990) | NOT tested | High Sharpe, high turnover; needs cost-aware sizing |
| Idiosyncratic skewness (Boyer-Mitton-Vorkink 2010) | NOT tested | Genuinely orthogonal lottery-preference anomaly |
| Investment / asset growth (CMA in FF5) | partial | Already in FF5 controls but not as sleeve |

These are **institutional table stakes**, not exotic. The fact that all 6 are
absent is a real gap.

### I.B — Variance Risk Premium / vol carry (PRIORITY: HIGH)

The single most documented robust anomaly in modern asset pricing, and we have
**zero exposure to it**. Mechanism: realized vol < implied vol systematically
(insurance premium). Three accessible expressions:

1. **VIX term structure carry** (Karagozoglu-Lin 2010): short front-month VIX
   futures, long back-month, when curve in contango. Easy to backtest from
   CBOE VIX futures (free data via CFE). Net Sharpe ~0.8-1.2 historical, but
   crashes hard in vol spikes (Aug 2015, Feb 2018). Needs hedging.

2. **Delta-hedged short straddles** (Carr-Wu 2009): sell ATM straddles on SPX
   monthly, delta-hedge daily. Mechanism = volatility risk premium. Requires
   options data + Greeks. Carry sleeve cousin but mechanism-orthogonal.

3. **Dispersion trade**: long index vol, short component vol. Captures
   correlation risk premium.

Recommended start: VIX term structure carry. Same data infrastructure
philosophy as our existing carry sleeves. Strict gate ablation needed.

### I.C — Lead-lag / cross-stock information transmission (PRIORITY: MEDIUM)

The DGNSDE paper (WWW 2026) and earlier work (Cohen-Frazzini 2008 "Economic
Links") document that information flows with delay between linked stocks.
Industry leaders → suppliers, customer disclosures → suppliers, etc.

**Honest disposition** (correcting my earlier blanket-skip):
- Different mechanism class from Line C's text-feature NN
- POC-worthy WITH STRICT PROTOCOL: realistic costs, 36+ month OOS, FF5+UMD orthogonality
- Methodology can be simplified for POC (don't need full SDE/Hermite; cubic
  spline + DTW + simple GNN suffices for first-pass feasibility)
- If passes strict gate → genuine 4th mechanism. If RED → honest graveyard entry.

POC scope: single sector (e.g., semiconductors NVDA/AMD/TSM/AVGO/MU/INTC),
60-month formation window for DTW lag estimation, monthly rebalance, full
transaction cost.

### I.D — Credit risk premium (PRIORITY: LOW)

High-yield carry, distress momentum, capital structure arb. All legitimate
alpha sources but require:
- Credit data (HYG/JNK + individual bonds)
- Different execution venue (IB or specialized)
- Capacity concerns at our scale

Defer until book scales or until other higher-priority work completes.

### I.E — Tail risk / explicit convexity (PRIORITY: MEDIUM)

Current crisis hedge = Spec 80 sleeve (10.37% = 75% TLT-GLD + 25% trend). This
is a STRUCTURAL hedge, not explicit convexity. We have:
- No long put protection (would be expensive — usually negative-EV)
- No long volatility expression (VIX call ladders, similar to VRP but inverse)
- No CVaR-optimized portfolio construction (we use vol-target, not tail-aware)

A senior allocator would push for at least *measured* tail-risk metrics
(MES, ES@95%, conditional drawdown). Building a tail-risk dashboard is cheap;
implementing convexity overlays is expensive and often negative-EV.

### I.F — Risk model upgrade (PRIORITY: MEDIUM)

Current cov matrix: Ledoit-Wolf shrinkage on 252-day daily returns. This is
good enough for vol target but suboptimal for:
- Sector/factor exposure neutralization
- True risk decomposition
- Capacity / liquidity-aware sizing

Upgrade path: Barra-style multifactor risk model (8-15 factors: market, size,
value, momentum, quality, volatility, sectors). Open-source implementations
exist (e.g., empyrical, pyfolio extensions). 1-2 day infrastructure work.

Effect: probably +0.05-0.10 Sharpe via better exposure control, not new alpha.

### I.G — Higher-frequency rebalancing variants (PRIORITY: LOW)

Weekly TSMOM has shorter-horizon trend signal. Daily mean reversion is well-
known but turnover-constrained. Most of these are NOT new mechanisms,
just different sampling rates of existing ones. Defer unless specific
gap surfaces.

---

## II. Agentic AI gaps (where infrastructure is designed but not built)

### II.A — Autonomous research loop (PRIORITY: HIGHEST)

This is the **single highest-leverage build remaining**. Designed twice
(Research Co-Pilot, Quant Engineer), built zero times. Each session of work
re-derives context from memory + git log. A persistent loop solves this.

Design (Karpathy AutoResearch + our strict-gate doctrine):

```
loop {
  1. propose: agent (Devil's Advocate or proposer-persona) suggests
     ONE parameter modification or new candidate within the SKILL
     constraints (pre-committed scope)
  2. test: engine.research.pipeline.run_gate evaluates on TRAIN+VAL
  3. log: append to data/research/gate_runs.jsonl with timestamp,
     proposer, change, result, decision
  4. decide: if VAL Sharpe > prior version's VAL Sharpe AND no
     constraint violation → keep; else rollback
  5. test set ONLY for final manual review, never in the auto-loop
  6. on N rollbacks in a row → halt, request human attention
}
```

Components we already have:
- `engine/research/pipeline.run_gate` (the evaluator)
- `engine/agents/*` (6 persona definitions)
- `data/research/gate_runs.jsonl` (the ledger)
- `data/validation/factory_ledger.jsonl`

Components we lack:
- The orchestrator (~300-500 lines)
- The SKILL.md template per sleeve (defines what proposers can/cannot change)
- The sample-isolation enforcement (train/val/test gating)
- The auto-rollback logic

Effort: ~10-15 hours for v1. Pays compound interest for every future POC.

### II.B — Decay Sentinel with LLM reasoning layer (PRIORITY: HIGH)

Current Decay Sentinel: rule-based (IC threshold, drawdown threshold). It
detects WHEN a sleeve is decaying but not WHY.

Upgrade: Attribution Analyst persona + signal-history reads + NAV history →
LLM-generated reasoning: "Sleeve X declined in Sept because Y (concentration
drift / regime shift / signal stale / cost increase). Recommended action: Z."

Output: actionable downgrade/halt recommendations with evidence.

Effort: ~6-8 hours.

### II.C — Pattern 6 cross-agent DD orchestration (PRIORITY: MEDIUM)

Per [[project-agent-collaboration-patterns-2026-05-18]] (already approved):
when a new strategy candidate appears, automatically run:
- Devil's Advocate (try to kill it)
- Attribution Analyst (decompose P&L)
- Audit Recorder (lineage)
in parallel, then merge into a single decision packet.

Reduces single-evaluator (currently me) bias. 6 personas exist; orchestrator
missing.

Effort: ~4-6 hours.

### II.D — Live anomaly detection that takes action (PRIORITY: MEDIUM)

Anomaly Sentinel persona currently reports findings. Wire it to:
- Pause new orders when sleeve breaches risk threshold
- Demand human review before resume
- Snapshot full state for forensic analysis

Effort: ~4 hours, mostly plumbing.

### II.E — Cost-prediction model from real fills (PRIORITY: HIGH, but
delayed until 30 days of Alpaca fills accumulated)

Once 30+ days of real Alpaca fills exist:
- Train per-ticker slippage model (regression of realized slippage on
  features: bid-ask, ADV, time-of-day, volatility)
- Pre-trade cost estimation per order
- Better order sizing → real Sharpe improvement

Effort: ~6-8 hours but blocked on data accumulation.

### II.F — Multi-model decision ensemble (PRIORITY: LOW)

Right now I (Claude) am the single decision-maker for gate verdicts. Ensemble
of 2-3 LLMs voting would reduce single-model bias but:
- API cost per gate run multiplies
- Latency increases
- For our scale, marginal benefit < marginal cost

Defer.

### II.G — Continuous attribution feedback loop (PRIORITY: MEDIUM)

Daily attribution report → automatic priority adjustment for next day's
research. E.g., if equity sleeve underperformed for 3 weeks running, agent
priority-bumps "investigate equity decay" task.

Effort: ~6 hours.

---

## III. Operational gaps (low priority but important)

| Item | Effort | Blocker |
|---|---|---|
| PQTIX → KMLM substitution in CTA sleeve config | 1-2 hours | none |
| AMRK/TPX/JBT universe cleanup (corporate actions check) | 1 hour | none |
| Real Alpaca slippage measurement | 2 hours | needs 30 days fills |
| Daily Alpaca schedule integration | 2 hours | none |

---

## IV. What to do NEXT — recommended sequence

A senior CIO / head of quant would order this as:

**Phase 1 (next 1-2 weeks): Infrastructure leverage**
1. **II.A — Autonomous research loop** (10-15h). Highest compound interest.
   Every subsequent POC benefits.
2. **II.B — Decay Sentinel reasoning** (6-8h). Risk management improvement,
   prevents disasters.
3. **Operational cleanup** (4-6h total). Get the live paper book truly clean.

**Phase 2 (weeks 2-4): Real new alpha**
4. **I.B — Vol carry / VRP** (8-10h). Most well-documented missing mechanism,
   self-contained data, clean gate eval.
5. **I.A.1 — Quality factor sleeve** (6-8h). Equity diversification anchor.
6. **I.C — Lead-lag POC** (12-15h). Genuine new mechanism class POC with
   STRICT protocol (real costs, 36mo OOS).

**Phase 3 (weeks 4-8): Polish**
7. **I.F — Multifactor risk model** (1-2d). Sharpe improvement via noise
   reduction.
8. **I.A.2-5 — Other equity factors as 1-3 mini sleeves** (each 4-6h).
9. **II.C — Pattern 6 orchestration** (4-6h, can run alongside Phase 2 work).

**Phase 4 (when fills accumulated)**:
10. **II.E — Cost prediction model** (6-8h, depends on real Alpaca data).
11. **Real-money pilot** (small fraction of paper NAV → real, IF Phase 1-3
    show 3-6 months of stable forward-OOS performance).

**Never do**:
- Any equity single-name signal without first checking the graveyard ledger
- Regime classifier overlays (ablation evidence already accumulated)
- Parameter searches on already-deployed sleeves (violates strict-gate doctrine)
- Anything based purely on backtest without real cost modeling

---

## V. Honest meta-observations

1. **The "harvest mode" framing I used earlier in this session was wrong.**
   Mature institutions never enter pure harvest mode; they continuously
   research while live-trading. The right framing is "live the v1 while
   building v2 in parallel with strict isolation".

2. **The agentic AI side is the biggest underused asset.** Six personas
   built, sophisticated research-gate ledger system, three+ specs designed
   for autonomous orchestration — and none of it is doing real work. This is
   leaving the largest single compound-interest opportunity on the table.

3. **The strict-gate doctrine is a tool, not a wall.** Using it to refuse
   evaluation is anti-doctrine. Using it to evaluate honestly and accept the
   outcome (GREEN or RED) is doctrine. POC + gate + verdict is the correct
   workflow for every candidate including ones I've previously deflected.

4. **3 mechanisms is a starting point, not an endpoint.** Genuine
   diversification needs 5-10 orthogonal mechanism families for real
   robustness against any single mechanism decaying.

5. **Real-money observation is mandatory.** Until 6-12 months of
   forward-OOS, the headline Sharpe 1.03 is a model estimate, not an
   established track record. Every public quant fund discount their model
   Sharpe by 30-50% when communicating with allocators for exactly this reason.

---

## VI. First concrete next step

Pick ONE Phase-1 item, complete it, commit, move on. Recommended starting
point:

**II.A — Autonomous research loop v1**

Why this first:
- Largest compound interest (every later POC benefits)
- Uses existing infrastructure (personas + gate + ledger)
- Concrete deliverable (orchestrator + SKILL.md template + sample-isolation
  enforcement)
- 10-15h is manageable in 1-2 sessions
- Output is a tool we use, not a number we report

Concrete v1 scope:
- `engine/research/auto_research_loop.py` (new) — orchestrator
- `docs/skills/equity_book.skill.md` (new) — SKILL definition for equity
- `docs/skills/carry_sleeve.skill.md` (new)
- `docs/skills/tsmom_sleeve.skill.md` (new)
- Unit tests for sample-isolation enforcement
- First end-to-end run: propose ONE D-PEAD parameter modification, run
  through full loop, verify ledger updates and rollback logic

Alternative starting point if user prefers a quick-win: **I.B — VIX term
structure carry POC** (8-10h, self-contained, immediate alpha relevance).
This is the easier-but-still-real path.

User choice — agenda is recorded either way.
