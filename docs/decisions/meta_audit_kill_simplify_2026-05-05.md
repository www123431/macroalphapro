# Meta-Audit Kill & Simplify (2026-05-05)

| Field | Value |
|---|---|
| Status | 🟢 ACTIVE — meta-audit cleanup post-Tier 1 audit + B-PLUS-PROD migration |
| Date | 2026-05-05 |
| Trigger | Supervisor 2026-05-05 challenge: "严肃检查其他部分有没有类似不是 ship TSMOM 假装严谨的问题" |
| Sibling | [tier1_retroactive_audit_2026-05-05.md](tier1_retroactive_audit_2026-05-05.md) (audit framework) · [b_plus_prod_migration_2026-05-05.md](b_plus_prod_migration_2026-05-05.md) (production switch) |

---

## 1. The Pattern Caught

The B-PLUS-PROD migration revealed a recurring pattern: **defending stale or
underpowered components as "capability demonstrations" instead of actually
removing them**. Same pattern as TSMOM-vs-QL01: ship FAIL/MARGINAL pretending
to be rigorous via documentation alone.

This audit applies the same Lakatos research-programme self-correction to
6 candidate components found by supervisor challenge.

## 2. Findings + Action

### 2.1 KILL — macro_research weekly pipeline

**Finding**: macro_research_agent ran weekly via `engine/orchestrator.run_weekly()`,
producing LLM regime forecasts → Brier-scored → reflection-written. But
since 2026-05-02 (REGIME_SCALE=1.0 disable), forecasts have **zero impact on
production trades**. After 2026-05-05 B-PLUS-PROD migration to QL01 BAB,
the production decision path is fully orthogonal to macro forecast.

**Verdict**: Evaluation theater. $50-100/yr LLM cost + supervisor attention
+ infra maintenance for no production benefit.

**Action**: Removed from `orchestrator.run_weekly()` weekly cycle (4 steps:
macro_research / macro_verification / macro_reflection / supporting). Code
files preserved for historical reference. `pages/macro_brief.py` displays
deprecation banner; historical AlphaMemory rows are read-only audit trail.

```diff
- engine/orchestrator.py: 4 steps removed from run_weekly() (macro_research +
                          MACRO-V + MACRO-R + supporting)
+ engine/orchestrator.py: deprecation comment block explains rationale
+ pages/macro_brief.py:   st.warning() deprecation banner at top
+ engine/agents/macro_research/agent.py: file preserved, import path intact
                          (manual /api invocation still works for ad-hoc tests)
```

**EFFECTIVE_N_TRIALS impact**: 0 (this is removal, not amendment of pre-reg).

### 2.2 KILL — paper trading E

**Finding**: paper_trading_e ran via `engine/orchestrator.run_monthly()`
month-end hook with 3-arm forward ablation. Pre-registered statistical power
was **22-41%** on realistic LLM-debate effect sizes (5-10% Sharpe lift),
making INCONCLUSIVE the modal expected verdict. The 24-month commitment
required ~30 min/month supervisor review = 12h total for a test we already
knew would not conclude.

**Verdict**: Underpowered evaluation theater. S6 anomaly_screener (built
2026-05-05) provides equivalent forward-only LLM ablation infrastructure
with faster accumulation (90-day window with daily flag accumulation vs
monthly snapshot) and tighter design (rule_baseline_a/b vs LLM, M1/M2/M3
metrics vs paper E's binary 3-arm comparison).

**Action**: Removed monthly hook from `orchestrator.run_monthly()`. The
`run_paper_trading=True` parameter is preserved for back-compat but is now
a no-op. `pages/paper_trading.py` shows deprecation banner; historical
PaperTradingRun rows are read-only audit trail (signal_baseline tag from
B-PLUS-PROD already present).

```diff
- engine/orchestrator.py: snapshot_paper_trading_arms() block removed from run_monthly()
+ engine/orchestrator.py: deprecation comment block explains rationale
+ pages/paper_trading.py: st.error() deprecation banner + st.stop()
+ engine/paper_trading.py: file preserved (PaperTradingRun ORM stays for data)
```

**EFFECTIVE_N_TRIALS impact**: 0.

### 2.3 DOCUMENT — SkillLibrary count = 0

**Finding**: `SELECT COUNT(*) FROM skill_library` returns **0**. The "self-
learning loops" capability claim in README references SkillLibrary but the
table is currently empty (BH FDR α=0.05 over multiple factors makes
confirmed pattern injection rare; project hasn't accumulated enough data
for confirmed patterns to emerge).

**Verdict**: Capability infrastructure built, accumulation calendar-bound.
Honest disclosure required.

**Action**: Already documented in this audit; capability_evidence + README
will note "infrastructure ready, accumulation calendar-bound" pattern
(consistent with S2 reflection memory disclosure).

### 2.4 FALSE POSITIVE — FRED data integration

**Finding**: README listed FRED as data source. Initial concern was that
FRED was claimed but unused. Grep verification:

```
engine/regime.py:149:  def _fetch_fred(series_id, ...)
engine/regime.py:161:  url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
engine/macro_fetcher.py:60:  _FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
```

**Verdict**: FRED is genuinely used for regime detection (10y-2y spread,
unemployment, etc) and macro_fetcher (was used by macro_research before
deprecation). README claim is correct.

**Action**: No change needed. Logging here for audit completeness.

### 2.5 DOCUMENT — Regime overlay code path is dead branch

**Finding**: `engine/portfolio.py` Step 5 contains `if multiplier < 1.0:`
branch that scales positions in risk-off regime. With production
`REGIME_SCALE = 1.0`, multiplier is always ≥ 1.0, making this branch
UNREACHABLE in production. The branch was preserved for "future
A/B test of dual-signal agreement-gated revival" per existing comment
(2026-05-02 baseline switch), but has been dormant since.

**Verdict**: Acceptable preservation, but the dead-branch state should be
explicitly flagged in code so future maintainers do not assume regime
overlay is active.

**Action**: Added `_REGIME_OVERLAY_LIVE_BRANCH` constant + explicit
"DEAD BRANCH in production" inline comment at the `if multiplier < 1.0`
line. Code path preserved; flag explicit.

```diff
+ _REGIME_OVERLAY_LIVE_BRANCH = (REGIME_SCALE < 1.0 - 1e-9)  # False in production
+
  if multiplier < 1.0:   # DEAD BRANCH in production (REGIME_SCALE=1.0)
```

### 2.6 DOCUMENT — Regime-conditional position limits binding under QL01 BAB

**Finding**: With B-PLUS-PROD QL01 BAB, current universe (45 ETF Tier 2)
in `transition` regime produces:

```
QL01 BAB signals:        15 long / 15 short
After construct_portfolio: 7 long / 7 short = 14 positions
MAX_LONG=8 (transition regime), MAX_SHORT_EQUITY=6
Long limit hit:  False (7 < 8, ~12% slack)
Short limit hit: True  (7 ≥ 6, BAB wants more shorts than limit allows)
```

The MAX_SHORT_EQUITY=6 cap, calibrated during P6-1a (TSMOM era), is binding
under QL01 BAB in transition regime. BAB wants more high-β shorts than the
limit allows, so the strategy is slightly under-utilized.

**Verdict**: This is a risk-control / alpha trade-off, not a bug. Keeping
the cap tight is conservative; BAB performance might improve marginally if
the cap were relaxed but at the cost of more concentrated short exposure.

**Action**: Document as known constraint; do not amend without prior
forward observation (e.g. if 90-day forward shows BAB underperformance,
amend MAX_SHORT_EQUITY upward as `threshold_tweak` amendment +1 trial).

## 3. Aggregate Outcomes

| Component | Before | After |
|---|---|---|
| macro_research weekly pipeline | Active in `run_weekly()` | Deprecated; weekly cycle no longer invokes |
| paper trading E monthly hook | Active in `run_monthly()` | Deprecated; monthly cycle no longer invokes; UI shows banner |
| SkillLibrary | Claimed capability | 0 rows; documented as "infra ready, accumulation calendar-bound" |
| FRED integration | Claim verified | True; no change needed |
| Regime overlay dead branch | Comment-only | Explicit `_REGIME_OVERLAY_LIVE_BRANCH = False` flag |
| Regime limits under QL01 BAB | Unknown | Documented: MAX_SHORT_EQUITY=6 binding in transition regime |

**Total LLM cost saved (forward-looking, annualized)**: ~$50-100/yr
(macro_research weekly Gemini calls).

**Total supervisor attention saved**: ~12h over 24 months (paper E
removed) + ongoing dashboard simplification (macro_brief now archive-only).

## 4. Lakatos Self-Correction Pattern

This audit, taken together with B-PLUS-PROD migration, demonstrates the
project's research-programme self-correction in action:

```
Pattern: Component X was defensible at time of creation
            ↓
         Later evidence changed value proposition
            ↓
         Initial response: defend with "capability demo" framing
            ↓
         Supervisor adversarial check: "is it really useful or just theater?"
            ↓
         Honest action: KILL or DOCUMENT, not just reframe
```

This is exactly Lakatos 1970 *protective belt* discipline: amend the
protective hypotheses (auxiliary components) rather than the hard core
(production strategy + 0-LLM-in-evaluation invariant + falsification
chain).

## 5. References

**Project-internal**:
- [tier1_retroactive_audit_2026-05-05.md](tier1_retroactive_audit_2026-05-05.md) — audit framework
- [b_plus_prod_migration_2026-05-05.md](b_plus_prod_migration_2026-05-05.md) — production strategy switch
- [llm_3layer_architecture_2026-05-05.md](llm_3layer_architecture_2026-05-05.md) §3 — invariants this audit defends

**Academic**:
- Lakatos 1970, *The Methodology of Scientific Research Programmes* — protective belt amendment

## 6. Amendment Ledger

| Date | Change | Author | Notes |
|---|---|---|---|
| 2026-05-05 | Initial Kill & Simplify audit; macro_research + paper E deprecated; SkillLibrary count documented; regime overlay dead-flag explicit; regime limit binding noted | zhangxizhe | Triggered by supervisor meta-audit challenge post-B-PLUS-PROD |
