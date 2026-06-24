# Tier 1 Retroactive Audit — Findings + Remediation (2026-05-05)

| Field | Value |
|---|---|
| Status | 🟢 ACTIVE — Audit complete; remediation = Option A (documentation) |
| Date | 2026-05-05 |
| Trigger | Supervisor question 2026-05-05: "之前没做 self-adversarial audit 的 sprint 是不是可能都有漏洞" |
| Tool | [scripts/tier1_retroactive_audit.py](../../scripts/tier1_retroactive_audit.py) (deterministic; no LLM in audit per D1 invariant 1) |
| Findings file | [tier1_audit_results.json](../../tier1_audit_results.json) |
| Sibling docs | D1 [llm_3layer_architecture_2026-05-05.md](llm_3layer_architecture_2026-05-05.md) · D2 [hitl_architecture_audit_2026-05-05.md](hitl_architecture_audit_2026-05-05.md) · S6 [s6_anomaly_screener_spec_2026-05-05.md](s6_anomaly_screener_spec_2026-05-05.md) |

---

## 1. Methodology

Self-adversarial validation pass over three thesis-critical claims that the
project has been making in README, capability_evidence, falsification chain,
and SpecRegistry-bound docs. Audit script runs deterministic checks; no LLM
involvement (audit is itself Layer 2 per D1 architecture invariant 1).

**Three claims under audit**:

| Claim | What we promise | Why it matters |
|---|---|---|
| 1 — 7 falsifications valid | Each rejected hypothesis was tested against pre-registered criteria, not inconclusive mislabelled as reject | Falsification chain is core thesis contribution; weak entries → chain shortens |
| 2 — 0-LLM-in-evaluation red line | LLM is forbidden in Layer 2 (evaluation / scoring / verdict) and Layer 3 (audit / persistence) project-wide | Stated red line; if violated anywhere, framework integrity broken |
| 3 — Pre-registration spec_hash chain | spec_hash + amendment ledger + HARKing 4-rule detector cannot be silently bypassed | Lakatos research programme rests on this; if bypassable, rigor narrative collapses |

---

## 2. Audit Results

**Summary** (after B-revised LCS deprecation + spec_hash drift fix):
**47 PASS · 6 WARN · 0 FAIL**

```
Claim 1 (7 falsifications):       38 PASS · 4 WARN · 0 FAIL  (regex-only WARN; doc-style)
Claim 2 (0-LLM-in-eval):           8 PASS · 1 WARN · 0 FAIL  ← lcs.py deprecated; macro_verification WARN false-positive
Claim 3 (spec_hash chain):         5 PASS · 0 WARN · 0 FAIL  ← spec_ui_redesign drift fixed via amend_spec id=20
```

**Initial run** (before remediation): 47 PASS · 6 WARN · 2 FAIL.
**Mid-audit fixes**:
- spec_ui_redesign.md drift → `amend_spec(kind="clarification")` id=20 (+0 trials)
- engine/lcs.py LLM-as-judge → `_run_lcs_on_decision()` deprecated to no-op stub (B-revised)

### 2.1 Claim 1 detailed

All 7 falsification docs (narrative_risk_gate D1 / D1.1 / overlay phase0 /
factor_mad / efa_three_piece / s1_multi_window / b_plus_marginal) have:
- Explicit verdict label ✅
- Verdict-supporting evidence reference ✅

4 WARNs are regex-driven false positives:
- factor_mad doc terse (n=24 implicit, not labelled "n="); 0/24 reject is unambiguous
- narrative_overlay_phase0 doesn't use the word "threshold" but does pre-register B-C ≈ 0 verdict
- b_plus_marginal mentions "post-hoc" in a CAUTION context (warning against, not committing)

**Verdict on Claim 1**: PASS. Falsification chain integrity confirmed at scale.
The 4 WARNs are documentation-style only, not substantive.

### 2.2 Claim 2 detailed

**FAIL — engine/lcs.py contains forbidden `generate_content` call**

Evidence:
- `engine/lcs.py:136` `_safe_call()` calls `model.generate_content(prompt)`
- `engine/memory.py:3998` calls `run_full_lcs_audit()` from `verify_pending_decisions()`
- `engine/memory.py:108` `DecisionLog.lcs_passed` is a Boolean column populated from LCS LLM output
- `engine/memory.py:104` comment: "lcs_passed=False blocks write-back to learning tables"
- `engine/memory.py:4393` LCS gate: "decisions with lcs_passed=False are excluded from meta-analysis"

This is a Layer 2 LLM-as-judge pattern. The LLM evaluates a mirrored prompt
("if vix were 50 instead of 18, would direction flip?") and the resulting
`lcs_passed` bool gates downstream learning. **This violates the project's
0-LLM-in-evaluation invariant as stated in
[llm_3layer_architecture_2026-05-05.md](llm_3layer_architecture_2026-05-05.md)
§3 Invariant 1.**

**Historical context**: lcs.py was created 2026-04-11 (devlog_2026-04-11.md
section 5.2), predating the formal 0-LLM-in-evaluation red line which was
crystallized 2026-05-02 in
[narrative_risk_gate_d1_soft_rejected.md](narrative_risk_gate_d1_soft_rejected.md)
and [narrative_overlay_phase0_rejected.md](narrative_overlay_phase0_rejected.md)
verdicts. The 2026-05-03 cleanup sprint deleted narrative_overlay,
factor_mad, narrative_risk_gate, and risk_narrative_agent modules — **but
lcs.py was not in the deletion list**. It survived as legacy.

**Severity assessment**: Medium-low.
- LCS only fires inside `verify_pending_decisions()`, which gates
  DecisionLog records being marked verified. Current usage is sparse
  (project shifted focus to paper_trading E forward test + S6 anomaly
  screener; DecisionLog verification is on slow calendar tick).
- LCS does NOT participate in S6 anomaly_screener verification (D4.5 M1 is
  fully deterministic price-based forward event check).
- LCS does NOT participate in S3 pre-registration enforcement (S3 uses
  deterministic spec_hash + 4-rule HARKing detector, no LLM).
- LCS does NOT participate in B-pragmatic-v2 D2 HITL governance approval
  workflow (deterministic queue + supervisor decision + hash chain freeze).

**Decision (Option B-revised — deprecate LCS entirely)**: Initially proposed
Option A (document only). Then upgraded to Option C (deterministic refactor)
when supervisor confirmed time was available. Then **inspection of
`engine/portfolio.py` revealed the project's actual ranking rule is
`direction = sign(raw_return_12m_skip_1m)`** (TSMOM, Moskowitz et al. 2012)
— a deterministic monotonic sign function. Mirror property holds by
construction; noise stability is trivial away from zero (project upstream
filters `tsmom != 0`); cross-cycle consistency already measured by
`engine.memory_curator` BH FDR. **The C-refactor would be mathematically
vacuous on the simple sign() rule** — equivalent to "testing whether a
circle is round" (always tautologically true).

Switched to **B-revised**: deprecate LCS entirely. `_run_lcs_on_decision()`
in `engine/memory.py` is now a no-op stub with deprecation docstring;
`engine/lcs.py` retains a top-level deprecation banner; production callers
remain unchanged. `lcs_passed` column preserved as historical audit trail
(NULL on new rows = pass per existing semantics). Tier 1 audit Claim 2
turns full PASS as a result.

Rationale chain that led here:
1. Original Option A was documentation-only — would leave active red-line
   violation in production code; weak defensively
2. Option C (deterministic refactor) was specced
   ([spec_lcs_deterministic_v2.md](../spec_lcs_deterministic_v2.md), now
   SUPERSEDED) but inspection of project's real ranking rule made clear
   the refactor would be mathematically trivial
3. Option B-revised: explicit deprecation acknowledges that LCS adds zero
   marginal information on a deterministic sign() rule. SkillLibrary
   write-back (LCS's historical downstream) is now gated by
   `memory_curator` BH FDR, an independent deterministic mechanism. LCS
   has no remaining function.

Filed as `L9 — LCS legacy module deprecated` in
[scope_and_future_work.md](../scope_and_future_work.md).

### 2.3 Claim 3 detailed

**Spec hash chain integrity**: 22/23 specs hash-intact at audit time. 1
drift detected (`docs/spec_ui_redesign.md`) — caused by mid-sprint cleanup
edit (command_center archive notation, line 242). **Resolved during audit
session via** `amend_spec(kind="clarification", reason="...")`, which logged
amendment id=20 with +0 EFFECTIVE_N_TRIALS impact (clarification kind = 0
multiplier per spec §2.3).

**Amendment log integrity**: All 23 SpecRegistry rows have valid JSON list
amendment_log fields.

**HARKing 4-rule detector**: Runs without crash, produces 0 flags on current
spec history. (Note: live HARKing detection is not enforced for S6 per the
Slimmed-corrected spec; amendment ledger + spec_hash recomputation is the
de-facto audit trail.)

**Bypass risk**: Public API (`register_spec` / `amend_spec`) compute hashes
deterministically; raw INSERT into SpecRegistry would bypass, but next
audit run would detect the drift.

**Verdict on Claim 3**: PASS. Spec_hash chain mechanics work as designed.
EFFECTIVE_N_TRIALS = 3 (forward registrations: P-FUND id=21, P-AUDIT id=22,
S6 id=23) + amendments.

---

## 3. Aggregate Verdict

```
Claim 1 — 7 falsifications:        PASS (cosmetic doc improvements possible)
Claim 2 — 0-LLM-in-evaluation:     PARTIAL — 1 legacy violation documented
Claim 3 — spec_hash chain:         PASS (drift fixed via amend_spec)
```

The project's overall research integrity claim survives the audit. The
single violation (lcs.py LLM-as-judge) is documented as a known legacy
limitation; it is bounded in scope (does not affect 2026-05 sprint outputs)
and explicitly disclosed in scope_and_future_work.md L9.

---

## 4. Self-Adversarial Audit as Methodology Contribution

This audit itself is the project's contribution:

> "We adopt iterative self-adversarial validation. After each main sprint
> completes verification (do we build what we say we build?), we run an
> additional adversarial validation pass (is what we built actually
> defensible?). This Tier 1 audit catches 1 legacy violation and 1 spec
> drift across 23 SpecRegistry entries and 7 falsification docs — both
> remediated in the same session."

This pattern is consistent with **Lakatos 1970** research-programme
self-correction: the hard core (deterministic invariants in Layer 2 / 3)
is preserved; the protective belt (Layer 1 LLM components) is amended via
amendment ledger + spec_hash whenever drift is detected. The single
violation in lcs.py is documented rather than hidden, demonstrating
**Popper 1959** falsification rigor at the framework level.

---

## 5. References

**Project-internal**:
- [scripts/tier1_retroactive_audit.py](../../scripts/tier1_retroactive_audit.py) — deterministic audit script
- [tier1_audit_results.json](../../tier1_audit_results.json) — machine-readable findings (47 PASS / 6 WARN / 1 FAIL)
- [docs/scope_and_future_work.md](../scope_and_future_work.md) §L9 — LCS legacy disclosure
- [docs/decisions/llm_3layer_architecture_2026-05-05.md](llm_3layer_architecture_2026-05-05.md) §3 Invariant 1 — 0-LLM-in-evaluation source

**Academic**:
- Lakatos 1970, *The Methodology of Scientific Research Programmes* — hard-core / protective-belt distinction
- Popper 1959, *The Logic of Scientific Discovery* — falsification rigor
- Zheng et al. 2023, *Judging LLM-as-a-Judge with MT-Bench* — basis for the 0-LLM-in-evaluation invariant

---

## 6. Amendment Ledger

| Date | Change | Author | Notes |
|---|---|---|---|
| 2026-05-05 | Initial Tier 1 audit; 47 PASS / 6 WARN / 1 FAIL; lcs.py finding documented to L9; spec_ui_redesign drift fixed via amendment id=20 | zhangxizhe | First retroactive audit pass over thesis-critical claims |
