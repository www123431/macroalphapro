# P-AUDIT v1 — Amendment M3-corrected-ext-full Evidence

**Status**: SHIPPED 2026-05-04 (clarification +0 trials)
**Spec**: [docs/spec_supervisor_approval_panel_v1.md](../spec_supervisor_approval_panel_v1.md) (Amendment Ledger §)
**Original P-AUDIT v1 evidence**: [p_audit_supervisor_panel_evidence.md](p_audit_supervisor_panel_evidence.md)
**Verify**: [scripts/verify_p_audit_v1.py](../../scripts/verify_p_audit_v1.py) — **22 / 22 facets PASS**
**Verdict**: **CAPABILITY PASS** — 7 layers + 5 EXT extensions all deterministic / 0 LLM.

---

## 1. Why this amendment

User reviewed the panel rendered on a real entry alert (USO 战术入场触发) and identified that **P-AUDIT v1 base scope answered the audit-trail question (spec hash / replay / category enum) but did NOT answer the supervisor's actual decision-time questions**:

> "为什么是这个标的 / 为什么是现在 / 当下量化全景如何 / 历史能撑得住吗 / 批了之后会怎样"

This was a **task-framing error**. P-AUDIT v1 was built as backward-looking audit (post-decision compliance) when the original user need was forward-looking decision support (pre-decision deliberation). Cross-reference: [feedback_engineering_change_requires_ui.md](../../${REPO_ROOT}/.claude/projects/c--Users-${USER}-Desktop-intern/memory/feedback_engineering_change_requires_ui.md).

## 2. Scope (M3-corrected-ext-full)

### 2.1 Layers L1-L7a (M3-corrected base)

| Layer | Function | Academic anchor |
|---|---|---|
| L1 | `decision_context.get_watchlist_origin` | Brinson-Hood-Beebower 1986 decision attribution |
| L2 | `decision_context.get_quant_posture` | Moskowitz-Ooi-Pedersen 2012 TSMOM; Asness-Moskowitz-Pedersen 2013 CSMOM |
| L3 | `decision_context.get_regime_context` | Hamilton 1989; Ang-Bekaert 2002; Diebold-Lee-Weinbach 1994 (ex-ante caveat) |
| L4 | `decision_context.get_portfolio_posture` | Markowitz; Pedersen 2015 *Efficiently Inefficient* §4 |
| L5 | `decision_context.get_conditional_history` | Cochrane 2011 conditional posterior |
| L6 | `decision_context.compose_thesis` | LLM-or-rule, Klein 1999 pre-mortem framing |
| L7a | `decision_context.get_forward_preview` | Deterministic only; **L7b Monte Carlo EXCLUDED** (D1) |

### 2.2 EXT-1/2/3/4/5 extensions

| EXT | Content | Anchor | Loc |
|---|---|---|---|
| EXT-1 | Macro snapshot (yield curve / credit spread / VIX / dollar) | Estrella-Mishkin 1998; Bakshi-Madan 2000; Gilchrist-Zakrajšek 2012 | folded into L3 |
| EXT-2 | Cross-sectional sector league table (18 rows ranked by composite) | Asness-Moskowitz-Pedersen 2013 | folded into L2 |
| EXT-3 | Calendar effects flag (FOMC blackout / pre-FOMC drift / TOM / earnings) | Lucca-Moench 2015; Lakonishok-Smidt 1988; Ariel 1987 | folded into L7a |
| EXT-4 | HHI concentration (current + post-approve + interpretation) | Hirschman 1964; standard MV theory | folded into L4 |
| EXT-5 | Underwater duration (current DD / underwater days / 90d / 1y max DD) | Magdon-Ismail-Atiya 2004; Pedersen 2015 §10 | folded into L4 |

### 2.3 Hard exclusions

- **L7b Monte Carlo / probabilistic forward sim**: NOT included. Stationarity (S1 multi-window self-falsification) / fat-tail handling / behavioral anchoring / model-as-judge backdoor — same 5 reasons as spec §1.2 D1. Verify harness Facet 21 actively asserts `monte_carlo / mc_paths / probability / outcome_distribution / forward_pnl_pdf` keys are absent from L7a output.
- **LLM in evaluation layer**: Maintained. L6 rule-based thesis is a deterministic string formatter. League table, FOMC calendar, HHI, drawdown — all SQL / arithmetic.

## 3. Files shipped

| Layer | File | Lines |
|---|---|---|
| Spec amendment | [docs/spec_supervisor_approval_panel_v1.md](../spec_supervisor_approval_panel_v1.md) Amendment Ledger § | +180 lines |
| Backend modules | [engine/decision_context.py](../../engine/decision_context.py) | 700 lines, 7 public + helpers |
| Aggregator | [engine/approval_context.py](../../engine/approval_context.py) `_build_decision_context` | +60 lines |
| UI | [pages/orchestrator.py](../../pages/orchestrator.py) `_render_decision_context_section` + `_render_audit_trail_section` | +220 lines |
| Verify | [scripts/verify_p_audit_v1.py](../../scripts/verify_p_audit_v1.py) Facets 13-22 | +120 lines |

## 4. Verification (verbatim from harness)

```
Facet 13 amendment_log: kind=clarification n_added=0 OK
Facet 14 L1: available=False (or True with origin fields)  OK
Facet 15 L2 + EXT-2: league_n=18..40 OK
Facet 16 L3 + EXT-1: p_sum≈1.0 caveat=True macro keys OK
Facet 17 L4 + EXT-4: hhi 0.0-1.0 + interpretation OK
Facet 18 EXT-5 drawdown: shape OK
Facet 19 L5: n_obs + insufficient_data flag OK
Facet 20 L6: rule_based + decision_log paths both OK
Facet 21 L7a + EXT-3: calendar flags + NO MC keys OK
Facet 22 aggregator: 7-layer dict in get_approval_context OK

P-AUDIT v1 + M3-corrected-ext-full verification: 22 / 22 facets PASS
```

## 5. Capability axis update (capability_evidence.md)

The axis text from spec § 7 evolves:

> **From**: "Supervisor Decision Provenance & Audit (deterministic)"
> **To**: "Supervisor Decision Support & Audit (deterministic, 7-layer + 5 EXT)"

The deterministic 7-layer DECISION CONTEXT panel surfaces (L1) WatchlistEntry origin chain with quant baseline / LLM adjustment / suggested weight / source agent / days-in-watchlist; (L2) current TSMOM/CSMOM/composite + 5-day trend + 18-sector cross-section league table (EXT-2); (L3) current regime label with **filtered probability decomposition** (P(risk_on/off/transition)) — not hard label — plus macro snapshot of yield curve / credit spread / VIX / dollar (EXT-1) and ex-ante caveat per Diebold-Lee-Weinbach 1994; (L4) sector exposure pre-/post-approve, simplified MCR estimate, **HHI concentration** with interpretation (EXT-4) and **underwater duration** + 90d/1y max drawdown (EXT-5); (L5) sector × direction × regime conditional historical hit rate with n_obs ≥ 5 gate; (L6) thesis dual-source (LLM-debate-payload OR deterministic rule-based composer) with risk pre-mortem; (L7a) deterministic forward preview (position $ delta / cost bps / watchlist revert state) plus FOMC blackout / pre-FOMC drift / turn-of-month calendar flags (EXT-3) per Lucca-Moench 2015. **L7b Monte Carlo simulator explicitly excluded** per the project's own falsification-chain ethos (spec §1.2 D1, S1 multi-window self-falsification).

## 6. What changed in the amendment

This is a **clarification**, not a hypothesis amendment:
- `n_trials_added = 0`: no new statistical claim, no new threshold
- New columns or tables: **none**
- New schema migrations: **none**
- New ORM models: **none**
- spec_hash: 86484504bfef0842 → 0dac977bcd7c895a (git-blob hash recompute on file edit)
- SpecRegistry.amendment_log: 1 entry appended with full reasoning

## 7. References

- CFA Institute (2020). *GIPS 2020 Standards*, §Ⅰ.6 audit trail.
- Brinson, Hood, Beebower (1986). "Determinants of Portfolio Performance." *FAJ* 42(4).
- Moskowitz, Ooi, Pedersen (2012). "Time Series Momentum." *JFE* 104.
- Asness, Moskowitz, Pedersen (2013). "Value and Momentum Everywhere." *JF* 68(3).
- Hamilton (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica* 57(2).
- Ang, Bekaert (2002). "Regime Switches in Interest Rates." *JBES* 20(2).
- Diebold, Lee, Weinbach (1994). "Regime Switching with Time-Varying Transition Probabilities." in *Nonstationary Time Series Analysis*.
- Estrella, Mishkin (1998). "Predicting U.S. Recessions." *RES* 80(1).
- Bakshi, Madan (2000). "Spanning and Derivative Security Valuation." *JFE* 55.
- Gilchrist, Zakrajšek (2012). "Credit Spreads and Business Cycle Fluctuations." *AER* 102(4).
- Lucca, Moench (2015). "The Pre-FOMC Announcement Drift." *JF* 70(1).
- Lakonishok, Smidt (1988). "Are Seasonal Anomalies Real? A Ninety-Year Perspective." *RFS* 1(4).
- Ariel (1987). "A Monthly Effect in Stock Returns." *JFE* 18.
- Hirschman (1964). "The Paternity of an Index." *AER* 54.
- Magdon-Ismail, Atiya (2004). "Maximum Drawdown." *Risk Magazine*.
- Pedersen (2015). *Efficiently Inefficient*, Princeton Univ Press.
- Cochrane (2011). "Discount Rates." *JF* 66(4).
- Klein (1999). *Sources of Power*, Ch. on pre-mortem.
- Tetlock (2015). *Superforecasting*.
- Kahneman (2011). *Thinking, Fast and Slow*.
