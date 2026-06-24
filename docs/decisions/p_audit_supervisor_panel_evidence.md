# Decision Evidence — Supervisor Approval Audit Panel (P-AUDIT v1)

**Status**: SHIPPED 2026-05-04
**Spec**: [docs/spec_supervisor_approval_panel_v1.md](../spec_supervisor_approval_panel_v1.md) (forward-registered)
**Verify harness**: [scripts/verify_p_audit_v1.py](../../scripts/verify_p_audit_v1.py)
**Verdict**: **CAPABILITY PASS** — 12 / 12 facets green; project's 2nd S3 forward registration after P-FUND.

---

## 1. Problem framed

Supervisor confronts the approval queue with one-line `triggered_condition` text and a 2-button gate. When a question arises ("why this sector now? has this fired before? what would reject break?"), supervisor must hop across 4–5 pages or guess. This is the same problem the user originally proposed solving with an LLM chatbot — and the same problem the project ruled out via [feedback_no_llm_as_judge.md](../../${REPO_ROOT}/.claude/projects/c--Users-${USER}-Desktop-intern/memory/feedback_no_llm_as_judge.md).

P-AUDIT v1 is the structured-context replacement: 8 deterministic modules per approval card, top-K rule + RAG hybrid retrieval of similar past decisions, event-chain replay reconstructed from `AgentRun + AgentEventRow + DecisionLog` rows, and mandatory categorized rationale capture before resolve. 0 LLM in the evaluation layer.

## 2. What ships

| Layer | File | Lines |
|---|---|---|
| Spec (frozen + forward registered) | [docs/spec_supervisor_approval_panel_v1.md](../spec_supervisor_approval_panel_v1.md) | 9 sections, 8 D-records |
| ORM | [engine/memory.py](../../engine/memory.py) PendingApproval +2 cols + migrate | +13 lines |
| Backend (Tier 2 + 3a + 3b) | [engine/approval_context.py](../../engine/approval_context.py) | 460 lines, 3 public + helpers |
| Backend (3d analytics) | [engine/approval_analytics.py](../../engine/approval_analytics.py) | 200 lines, 3 public |
| UI inline panel | [pages/orchestrator.py](../../pages/orchestrator.py) `_render_audit_panel` | +130 lines |
| UI analytics page | [pages/approval_analytics.py](../../pages/approval_analytics.py) | 165 lines, 4 sections |
| Verify harness | [scripts/verify_p_audit_v1.py](../../scripts/verify_p_audit_v1.py) | 12 facets |

## 3. Verification (verbatim from harness)

```
Facet 1 + 2  — S3 forward registration + EFFECTIVE_N_TRIALS
  spec_hash[:16] = 86484504bfef0842
  retro_registered = False
  n_trials_contributed = 1
  forward registrations total = 2     # P-FUND (1st) + P-AUDIT (2nd)
  EFFECTIVE_N_TRIALS = 45              # 43 grid + 2 forward
  pre_registration axis = 2

Facet 3   — pending_approvals columns OK: review_rationale + review_category present
Facet 4-7 — approval_context.py contracts: 4 functions × shape OK
Facet 8   — approval_analytics.py contracts: 3 functions × shape OK
Facet 9   — AppTest pages/orchestrator.py cold + seeded: 0 exception
Facet 10  — AppTest pages/approval_analytics.py: 0 exception
Facet 11  — resolve_pending_approval persists rationale + category
Facet 12  — 6 / 6 deliverable files exist
```

12 / 12 PASS. Run `D:/python/python.exe scripts/verify_p_audit_v1.py` to reproduce.

## 4. Decision records (mirrored from spec)

| ID | Decision | Why |
|---|---|---|
| **D1** | Excludes Tier 3.c What-if Monte Carlo simulator | Stationarity assumption falsified by S1 multi-window, fat-tail handling fragile, behavioral anchoring re-imports model-as-judge through supervisor channel, polish-drift away from capability claim, conflict with project's 7-test falsification chain ethos |
| **D2** | `review_category` enum limited to 6 values | Deterministic; group-by analytics need stable buckets, free-text would invite LLM categorizer drift |
| **D3** | `review_rationale` ≥ 10 chars, supervisor-typed only | [feedback_no_llm_as_judge.md](../../${REPO_ROOT}/.claude/projects/c--Users-${USER}-Desktop-intern/memory/feedback_no_llm_as_judge.md); LLM pre-fill is the LLM-as-judge backdoor |
| **D4** | 3b similarity = rule filter ∩ S2 RAG hybrid | RAG cold-start fallback to deterministic rule (sector × type × regime); `retrieval_method` field marks origin |
| **D5** | 3a Replay strictly from AgentRun + AgentEventRow + DecisionLog | No new event tables; `reconstructed=True` flag tells UI when timestamp is reverse-inferred |
| **D6** | Inline expander on Operations page (not a new approval-only page) | Adds ≤30s context to existing workflow; doesn't fragment the approval gesture |
| **D7** | 3d Analytics on its own `pages/approval_analytics.py` | Cross-time analytics ≠ per-approval workflow; separating reduces operations page weight |
| **D8** | 0 LLM in the evaluation layer | Hard project red line; rationale text is supervisor-typed, category is enum, replay is event-bus rows, similarity is sentence-transformer cosine (Layer 1 generation only — Layer 2 ranking is deterministic dot product, same as S2) |

## 5. Capability axis (capability_evidence.md update P-AUDIT-6)

Spec §7 supplies the verbatim text to add to [docs/capability_evidence.md](../capability_evidence.md). The axis name is **"Supervisor Decision Provenance & Audit (deterministic)"**.

## 6. Cross-references

- **S2 RAG retriever** ([engine/agents/reflection.py:retrieve_relevant_reflections](../../engine/agents/reflection.py)) — reused for 3b
- **S3 SpecRegistry** ([engine/preregistration.py:register_spec](../../engine/preregistration.py)) — registered 2026-05-04 as project's 2nd forward spec
- **P-FUND PendingApproval** ([memory/project_p_fund_performance_reporting_2026-05-04.md](../../${REPO_ROOT}/.claude/projects/c--Users-${USER}-Desktop-intern/memory/project_p_fund_performance_reporting_2026-05-04.md)) — supervisor cash-flow approve path now flows through this same audit panel
- **Paper trading E v0.2** Arm B decisions — review_category data feeds future per-sector outcome analysis

## 7. What this is NOT (deliberate exclusions)

- Not a chatbot. There is no LLM Q&A interface.
- Not a What-if simulator. Stationarity & anchoring concerns documented in D1.
- Not a recommender. Panel surfaces context; supervisor decides.
- Not multi-supervisor. Single-supervisor scope, conflicts deferred to v2.
- Not adaptive. Approval rate does not auto-adjust thresholds (would be self-judge).

## 8. References

- CFA Institute (2020). *GIPS 2020 Standards*, §Ⅰ.6 (audit trail).
- Brinson, Hood, Beebower (1986). "Determinants of Portfolio Performance." *FAJ* 42(4).
- Kahneman (2011). *Thinking, Fast and Slow.* (System-2 deliberation under structured information.)
- Bridgewater Associates (2010). *Principles* — machine-believable decision-framework principle.
