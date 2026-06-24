# HITL Architecture Audit + Slim Refactor Spec (2026-05-05)

| Field | Value |
|---|---|
| Status | 🟢 ACTIVE — D2 of B-pragmatic-v2 sprint |
| Date | 2026-05-05 |
| Sibling | [llm_3layer_architecture_2026-05-05.md](llm_3layer_architecture_2026-05-05.md) (D1) |
| Implements | A (HITL slim refactor) — D3 implementation |
| Pre-registration | SpecRegistry entry `arch.hitl.governance_split.v1` (added in D3) |

---

## 1. Trigger and Premise

Supervisor critique 2026-05-05:

> "全部都是机械流程，也没见量化机构需要笔笔交易都审批啊，我们只是频率相较于他们更慢了，但这也不是我们需要审批的理由。"

Honest verdict: **70% of current HITL is ceremony**.

Industry reference (publicly documented):
- Renaissance / Two Sigma / Citadel / DE Shaw: **no trade-level approval**. Signal → risk filter → automatic execution.
- Approval exists only at four levels:
  1. Model committee (strategy on/off, parameter changes)
  2. Risk committee (kill switch, anomaly, limit breach)
  3. Compliance (cash flow, regulatory events)
  4. Spec amendments (HARKing prevention)

Our system over-engineered trade-level approval with no client capital and no regulator. Slim refactor is overdue.

---

## 2. Current State (Pre-Refactor) — 9-Issue Audit

### Sound (4 issues — keep)

#### Issue S-1 — 4 类 governance approval academically grounded

Approvals that **should** require supervisor:

| Type | Reference | Reason |
|---|---|---|
| `cash_flow` | GIPS 2020 §III.A.18 | Custodian/supervisor verification of fund flows |
| `spec_amendment` | Hansen 2005 *Backtesting* + Lakatos 1970 | Pre-registration discipline; HARKing prevention |
| `risk_control` (kill switch) | Knight Capital 2012 case + Artzner et al. 1999 coherent risk | Cannot let algorithm decide to halt itself |
| `strategy_arm_toggle` | Paper trading E v0.2 spec | Forward-test integrity (24-mo lock) |

#### Issue S-2 — Hash chain narrative snapshot
SEC 17a-4(b) electronic record + GIPS audit trail + López de Prado §10. Current `review_narrative_snapshot` + `review_narrative_hash` + `prev_narrative_hash` chain is sound (P-AUDIT v1 NARRATIVE-C v2).

#### Issue S-3 — SpecRegistry + amendment ledger + 4-rule HARKing detection
Hansen 2005 / S3 Pre-Registration sprint complete (31/31 facets). EFFECTIVE_N_TRIALS dynamic integration sound.

#### Issue S-4 — EFFECTIVE_N_TRIALS dynamic count
BHY FDR over multiple comparisons enforced. Reference: Benjamini-Hochberg-Yekutieli 2006.

---

### Problems (5 issues — fix)

#### Issue P-1 — Trade-level approval is 70% ceremony **(BLOCKING)**

Current `approval_type` enum: `entry / risk_control / rebalance / cash_flow`.

`entry / rebalance` at trade-level:
- Algorithm decides per pre-registered rule
- Supervisor cannot meaningfully override (would break pre-registration)
- Approval click adds **zero information value**
- Supervisor fatigue → rubber-stamping → M2 (acceptance) data poisoned

Production quant funds do not approve these. **Removing this approval reduces noise without weakening governance.**

#### Issue P-2 — HITL has no LLM input channel **(architectural void)**

Current state:
- All approvals deterministic (algorithm-generated signals)
- LLM (macro_research / paper E arm B) does not flow into HITL queue
- Result: HITL is purely "supervisor stamps deterministic output" = **ceremony**

This void is the structural reason supervisor critique landed. HITL gains substantive value only when supervisor judgment adds information **beyond what algorithm already determined**.

S6 fills this void — anomaly cases enter HITL queue as 5th category (`anomaly_screener`). Supervisor judgment on anomaly cases adds genuine signal because:
- Anomaly is by definition not pre-determined by primary algorithm
- LLM and rule-based detector outputs may disagree
- Forward-event verification (M1) is calendar-bound, only supervisor can act in real time

Without S6, HITL is governance + ceremony. With S6, HITL is governance + LLM-output gating.

#### Issue P-3 — Single supervisor; no model committee equivalent

Real funds: 3-5 person model + risk + compliance committees with explicit voting / quorum.
Our system: 1 supervisor (you).

Acceptable for master's project; **must be documented in scope_and_future_work as known limitation**. Real production requires multi-person committee (deferred indefinitely).

#### Issue P-4 — Local hash chain only; no third-party deposit

Current: SHA-256 chain stored locally in `pending_approvals.review_narrative_hash`.
Standard for Lakatos-grade pre-registration: external time-stamp deposit (OSF.io / SSRN preprint / Anchor protocol).

**Decision**: defer external deposit to future work. Local chain is sufficient for thesis defense if accompanied by explicit limitation note in scope_and_future_work. Not blocking B-pragmatic-v2 scope.

#### Issue P-5 — Approval ergonomics metrics absent

Current `pending_approvals` does not track:
- `approval_latency` (created_at → resolved_at duration)
- Rejection-reason categorical distribution
- Weekly self-review pattern

Without these, supervisor rubber-stamping cannot be detected. M2 (acceptance rate) data may be poisoned.

**Fix**: add `approval_latency_seconds` (computed at resolve) + weekly review dashboard panel. Cost ~2h.

---

## 3. Slim Refactor Spec — Approval Type Re-Classification

### New approval_type taxonomy (supervisor-confirmed 2026-05-05)

```
PendingApproval.approval_type ∈ {
  ┌──────────────────────────────────────────┐
  │ Governance (REQUIRES supervisor approval) │
  ├──────────────────────────────────────────┤
  │  cash_flow              │  P-FUND        │
  │  spec_amendment         │  S3 SpecReg    │
  │  risk_control           │  Kill switch   │
  │  strategy_arm_toggle    │  Paper E lock  │
  │  universe_change        │  Pool add/drop │ ← added 2026-05-05
  ├──────────────────────────────────────────┤
  │ LLM-output (REQUIRES, post-S6)           │
  ├──────────────────────────────────────────┤
  │  anomaly_screener       │  S6, D4        │
  ├──────────────────────────────────────────┤
  │ Routine review (NO approval; audit only) │
  ├──────────────────────────────────────────┤
  │  routine_review         │  entry/exit/   │
  │                         │  rebalance     │
  │                         │  post-hoc trace│
  └──────────────────────────────────────────┘
}
```

**Total**: 5 governance + 1 LLM-output + 1 routine_review = **6 approval-required + 1 audit-only**.

### Renamed and migrated mappings

| Old type | New type | Behavior change |
|---|---|---|
| `entry` | `routine_review` | Auto-execute; post-hoc trace written; **no supervisor click** |
| `rebalance` | `routine_review` | Same |
| `risk_control` (kill switch class) | `risk_control` (unchanged) | Continues governance approval |
| `cash_flow` | `cash_flow` (unchanged) | Continues governance approval |
| (new) | `spec_amendment` | New row added when SpecRegistry amendment requested |
| (new) | `strategy_arm_toggle` | New row added when paper E arm enable/disable |
| (new) | `anomaly_screener` | New row from S6 D4 onwards |

### New schema fields (D3 implementation)

```python
class PendingApproval(Base):
    # ... (existing fields preserved) ...

    # 2026-05-05 HITL slim refactor
    approval_class    = Column(String(16), nullable=False, default="governance")
    # governance | routine_review | llm_output
    approval_latency_seconds = Column(Integer, nullable=True)
    # Computed at resolve time; null while pending
    rejection_category = Column(String(32), nullable=True)
    # Enum: insufficient_evidence / contradicts_quant / risk_breach /
    #       harking_flag / cash_compliance / arm_lock / other
    # Categorical, distinct from review_category (which is approval-side)
```

### routine_review behavior

- Created by `daily_batch` for entry / exit / rebalance signals
- `approval_class = "routine_review"`
- `status = "auto_executed"` immediately on creation
- Not displayed in Governance Queue
- Displayed in **Routine Timeline** (read-only) on Operations page
- Supervisor cannot approve/reject; can only add `review_rationale` post-hoc note (optional)
- Counted in audit_agent_liveness as evidence of agent activity

### Migration plan (D3)

1. Add columns `approval_class`, `approval_latency_seconds`, `rejection_category`
2. Backfill `approval_class`:
   - existing `entry` / `rebalance` rows → `approval_class = "routine_review"`, `status = "approved"` left as-is for historical (only new ones become `auto_executed`)
   - existing `risk_control` / `cash_flow` rows → `approval_class = "governance"`
3. New row creation:
   - `daily_batch` entry/rebalance generators write `approval_type = "routine_review"`, `status = "auto_executed"` directly (not "pending")
   - existing `cash_flow` / `risk_control` paths unchanged
4. Operations page UI rewrite (split queue vs timeline)
5. README + scope_and_future_work language reframe

---

## 4. Operations Page Restructure

### Before
Single queue mixing entry / rebalance / risk_control / cash_flow → all required supervisor click → high cognitive load + ceremony.

### After
```
┌─────────────────────────────────────────────────────────────┐
│  GOVERNANCE QUEUE  (requires supervisor decision)            │
├─────────────────────────────────────────────────────────────┤
│  Pending: 4 (max ~5/day expected)                            │
│  [#42] cash_flow      USD  ↑$50k   deadline 2d              │
│  [#43] risk_control   XLE  kill   immediate                 │
│  [#44] anomaly        XLE  flag   90d window  (S6, post-D4) │
│  [#45] spec_amendment STRAT_X param tweak                    │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  ROUTINE TIMELINE  (read-only audit trail)                   │
├─────────────────────────────────────────────────────────────┤
│  [auto] 14:32 entry      AAPL  +1.5%   per spec.entry.v3    │
│  [auto] 14:31 rebalance  XLE   -0.8%   per spec.rebalance   │
│  [auto] 14:30 entry      MSFT  +1.2%   per spec.entry.v3    │
│  Supervisor may add post-hoc review note (optional).         │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  ERGONOMICS  (rubber-stamp detection)                        │
├─────────────────────────────────────────────────────────────┤
│  This week: 3 governance approvals; mean latency 2m18s       │
│  Rejection breakdown: 1 risk_breach / 0 harking / 0 other    │
│  ⚠ approvals < 30s: 0 (no rubber-stamp pattern detected)     │
└─────────────────────────────────────────────────────────────┘
```

**Implementation in D3**:
- `pages/orchestrator.py` split into 3 sections via `st.tabs(["Governance", "Routine Timeline", "Ergonomics"])`
- Existing Governance render logic preserved (already 4 类 working)
- Routine Timeline = read-only `st.dataframe` filtered on `approval_class = "routine_review"` last **30 days** (supervisor-confirmed 2026-05-05; 2-year window for trend visibility)
- Ergonomics panel computed from `approval_latency_seconds` + `rejection_category` aggregates

---

## 5. Compliance Mapping

| Regulator / Standard | Requirement | Our Compliance Path |
|---|---|---|
| **GIPS 2020 §III.A.18** | Cash flow custodian/supervisor verification | `cash_flow` governance approval; `review_rationale` ≥10 chars; hash chain frozen at resolve |
| **SEC 17a-4(b)** | Electronic record retention; deterministic reconstruction | Hash chain narrative snapshot; SpecRegistry deterministic replay |
| **CFA Code of Ethics III(C)** | Pre-registered methodology disclosure | SpecRegistry + amendment ledger; `spec_amendment` governance approval |
| **MiFID II Art. 16(7)** | Order record pre-trade risk check | (Future work — currently routine_review post-hoc only; documented limitation) |
| **Knight Capital 2012 case** | Kill-switch human oversight | `risk_control` governance approval |
| **Hansen 2005 / Lakatos 1970** | Pre-registration discipline | `spec_amendment` route + HARKing 4-rule detection |

**Coverage assessment**: 5 of 6 cited standards have direct mappings. MiFID II pre-trade is documented as future-work limitation. This is acceptable for master's-project scope.

---

## 6. What This Refactor Removes

### Removed claim
> "Supervisor approves every trade decision."

### Replacement claim
> "Supervisor approves 4 governance categories (cash flow / spec amendment / risk kill switch / strategy arm toggle) plus 1 LLM-output category (anomaly screener, post-S6). Trade-level entry/exit/rebalance is auto-executed per pre-registered spec; supervisor reviews timeline post-hoc as audit trail."

### Removed UI
- 70% of current Operations queue items (entry / rebalance pending) → moved to read-only timeline

### Preserved evidence
- All historical pending_approvals rows preserved; data structure additive only (no destructive migration)
- Existing audit infrastructure (hash chain / spec_hash / amendment ledger / HARKing detection) unchanged
- 100+ verification facets across S2/S3/P-FUND/P-AUDIT/Macro untouched

### Reframe language for thesis / SSRN / interview
- Old: "Multi-agent quant trading with LLM-augmented HITL"
- New: "Quant alpha (with 7 LLM-as-X falsifications documented) plus selective HITL governance on regulator-relevant events plus optional LLM-augmented anomaly screening with rule-based baseline (S6 forward test in progress)"

The new framing is longer but defensible. It accurately reflects what the system does and what it does not claim. **Honesty over brevity.**

---

## 7. Implementation Plan (D3)

| Step | Files | Lines | Estimated time |
|---|---|---|---|
| Schema migration | `engine/memory.py` (PendingApproval) + `_migrate_db()` | ~20 | 30 min |
| daily_batch routine_review write path | `engine/daily_batch.py` (entry / rebalance generators) | ~30 | 45 min |
| Operations UI split | `pages/orchestrator.py` (3-tab layout) | ~120 | 90 min |
| Approval ergonomics panel | `pages/orchestrator.py` + helper | ~50 | 30 min |
| README reframe | `README.md` | ~30 | 30 min |
| scope_and_future_work update | `docs/scope_and_future_work.md` | ~20 | 15 min |
| Smoke verification | n/a | n/a | 30 min |
| **Total D3** | | | **~4 hours** |

---

## 8. Acceptance Criteria for D3

- [ ] PendingApproval schema migrated; new fields populated on insert
- [ ] daily_batch entry/rebalance writes `approval_class = "routine_review"`, `status = "auto_executed"` directly
- [ ] cash_flow / risk_control / spec_amendment / strategy_arm_toggle paths preserved unchanged
- [ ] Operations page renders 3 tabs (Governance / Timeline / Ergonomics)
- [ ] Governance tab shows only `approval_class = "governance"` rows
- [ ] Timeline tab shows last 7 days `routine_review` rows, read-only
- [ ] Ergonomics tab shows weekly latency + rejection_category breakdown
- [ ] No existing approval row mutated (additive migration only)
- [ ] README and scope_and_future_work language updated
- [ ] Smoke: 6 pages exception-free; existing 100+ facets still PASS
- [ ] Hash chain integrity: 0 broken links pre/post

---

## 9. References

**Compliance**:
- GIPS 2020 §III.A.18 — Composite custodian verification
- SEC 17a-4(b) — Electronic record retention
- CFA Code of Ethics III(C) — Methodology disclosure
- MiFID II Art. 16(7) — Order record requirements
- Hansen 2005 *A Test for Superior Predictive Ability*

**Academic**:
- Lakatos 1970, *The Methodology of Scientific Research Programmes*
- Knight Capital 2012 case (SEC 34-70694)
- Artzner, Delbaen, Eber, Heath 1999, *Coherent Measures of Risk*
- Benjamini-Hochberg-Yekutieli 2006, *Adaptive linear step-up procedures*
- López de Prado 2018, *Advances in Financial Machine Learning* §10

**Project internal**:
- D1: [llm_3layer_architecture_2026-05-05.md](llm_3layer_architecture_2026-05-05.md)
- [s3_pre_registration_enforcement_evidence.md](s3_pre_registration_enforcement_evidence.md)
- [p_audit_supervisor_panel_evidence.md](p_audit_supervisor_panel_evidence.md)
- [p_fund_performance_reporting_evidence.md](p_fund_performance_reporting_evidence.md)

---

## 10. Amendment Ledger

| Date | Change | Author | Notes |
|---|---|---|---|
| 2026-05-05 | Initial commit; 9-issue audit + 4 governance + 1 LLM-output + routine_review | zhangxizhe | Triggers D3 implementation |
| 2026-05-05 | Add `universe_change` as 5th governance category; Timeline default = 30 days | zhangxizhe | Supervisor confirmation 2026-05-05; D3 implementation reflects |
