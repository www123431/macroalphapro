# MacroAlphaPro

**The system measured its own LLM predictor against a fair family-prior
baseline, found it LOSES by +0.114 Brier, and published the loss
instead of hiding it.** That epistemic posture — and the air-gapped
calibration infra that makes it falsifiable — is the project's headline.

The wrapper around it: a solo-built AI-augmented quant research
workbench. 4-sleeve canonical-replay strategy book (Sharpe 1.32
backtest, paper-trade live since 2026-05-13). 661-paper ingestion
pipeline. End-to-end autonomous research demonstration on cross-asset
bond-VRP. 9-station UI-triggerable Operator Console with import-time
capital-decision doctrine enforcement. arxiv preprint v0.9 ready for
submission.

**Author**: [Zhang Xizhe (Terrence)](https://www.linkedin.com/in/zhangxizhe) (NUS MSBA 2026)
**Status**: paper-trade live since 2026-05-13 · Operator Console v1
(9 of 9 Pipeline Stations) · arxiv preprint v0.9 ready for submission

---

### Most-impressive single artifact — bond-VRP autonomous demo

The system was given a real empirical question — *"does the variance
risk premium extend to cross-asset (sovereign bonds) instead of just
equities?"* — with no hint of an answer. It independently retrieved
the relevant literature (Carr-Wu 2009 + follow-ups), generated a
falsifiable FactorSpec, ran the strict-gate FORWARD pipeline (FF5+MOM
spanning, NW-HAC, Bailey-LdP DSR, Politis-Romano bootstrap), and
returned **RED** — bond-VRP does not survive — consistent with the
published finding that VRP is equity-specific. **Zero human input
in the prediction loop.** Full record: [paper §5](docs/arxiv_preprint_draft_2026-06-22.md)
· prediction air-gap evidence: `data/research/predictions/` · verdict
event: `engine/research_store/events.jsonl` (event_type=factor_verdict_filed).

---

## TL;DR for the recruiter / professor reading this

| What | Number | Where to verify |
|---|---|---|
| Backtest Sharpe (4-sleeve canonical replay, 486 weeks) | **1.32** (Lo-2002 SE 0.33, 95% CI [0.67, 1.97], t=4.0) | [paper §A.1](docs/arxiv_preprint_draft_2026-06-22.md), `scripts/reports/report_sharpe_se.py` |
| **Honest negative finding** — predictor loses to fair baseline | Brier **0.374** vs FAIR family-prior **0.260** (Δ +0.114, 95% CI [+0.054, +0.173] excludes zero, 8 rigor tests, n=94 at paper v0.9; live count refreshes daily) | paper §3, `engine/research/belief_track_record_rigor.py` |
| End-to-end autonomous demonstration | bond-VRP RED verdict, no human in the prediction loop (consistent with Carr-Wu 2009) | paper §5 |
| **Operator Console (2026-06-23 → 9/9 stations 2026-06-25)** | **9 of 9 Pipeline Stations live** — UI-triggerable end-to-end research pipeline + typed sessions + per-session cost cap + SSE streaming + audit trail | `engine/operator_console/` |
| Papers ingested + tagged | **661** via Stage-0 ClaimType router | `engine/agents/papers_curator/` |
| Engineering surface | 117 research modules · 247 scripts · ~20 agents · FastAPI + Next.js · **357 test files / ~5,800 assertions** | this repo |

(More context: ~$19/mo operating cost · sequential-specialist multi-agent architecture (not free debate) · 3 STANDING doctrines in [CLAUDE.md](CLAUDE.md). Full table in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).)

> **Number reconciliation (read once, then it's clear):** Two
> deployed-book definitions co-exist. (a) **4-sleeve canonical
> replay** = K1_BAB / D_PEAD / PATH_N / CTA_PQTIX over 486 weekly
> obs ~9.4 years; this is the **Sharpe 1.32** figure cited in the
> paper. (b) **5-sleeve current operating book** = equity_book /
> cross_asset_carry / cross_asset_tsmom / crisis_hedge_tlt_gld /
> mom_hedge_overlay; this is what `engine.portfolio.deployed_registry.load_active()`
> returns and is the surface S8 Rollback acts on. Both are real;
> both ship in the docs. The 4-sleeve replay is the audited backtest
> (clean, no look-ahead) while the 5-sleeve book is the present
> operating state (post-decisions made after the replay window
> closed). The earlier arxiv draft conflated the two; v0.9 §A.8
> discloses + resolves the inconsistency.

---

## AI-engineering surface (for AI / LLM / agent infra recruiters)

| Component | What it is | Where |
|---|---|---|
| **Air-gapped predictor + calibration ledger** | Belief layer commits a typed `PredictedVerdictDist` BEFORE each verdict runs; a structural test walks the AST and forbids any module under `engine/research/` from importing `engine.research.belief` outside a whitelist. **Calibration is testable because the architecture makes it falsifiable.** | `engine/research/belief.py` + `tests/test_belief.py:test_air_gap_lens_strict_gate_template_must_not_import_belief` |
| **8-test rigor harness** | Bootstrap CI / paired delta / time-aware FAIR baseline / sign test / Benjamini-Hochberg FDR (q=0.10) / Mann-Kendall / Hosmer-Lemeshow / LOOCV. `BOOTSTRAP_B = 10000, RNG_SEED = 42` — reproducibility is real. | `engine/research/belief_track_record_rigor.py` |
| **Multi-model routing (workload → provider)** | LLM workloads dispatch to provider+model via a single table. Sonnet for synthesis / extraction, Gemini Flash for high-volume tagging. ~89% volume on Flash, ~46% spend on Sonnet. R1/Deepseek A/B was audited + rejected (kept Claude for spec drafting). | `engine/llm/call.py` workload table + `data/llm_cost_ledger.jsonl` |
| **Sequential-specialist multi-agent (Pattern 1, not Pattern 5)** | α pre-mortem / β cross-domain / γ replication / DA pre+post / strengthener / decay sentinel — each agent owns ONE distinct epistemic lens, fans out via `asyncio.gather`, NO turn-taking debate. Tetlock 2017 fake-diversity literature explicitly cited as the reason; the doctrine is enforced by code review, not just docs. | `engine/research/agent_council.py:5-6` + memory `feedback_anti_n_persona_brainstorm_2026-06-14.md` |
| **MCP server (`intern-research`)** | Internal MCP server exposing the project's typed research toolkit (intuition-rules query, graveyard lookup, mechanism library, Sharpe-SE estimator, family n_trials, L4 outcome ledger). Used by Claude Code sessions to ground recommendations in the actual research state — no LLM hallucinations about which factor families have how many trials. | MCP server registered as `intern-research` in `.claude/settings.local.json` |
| **Operator Console with capital-doctrine code enforcement** | 9 Pipeline Stations. `StationSpec.mutates_capital` flag is enforced at import time: any station declaring `mutates_capital=True` whose source doesn't reference `_proposals.jsonl` (proof of routing to /approvals) raises `CapitalDoctrineViolation` on register. **The doctrine is the type system, not a comment.** | `engine/operator_console/registry.py` + `engine/operator_console/stations/` |
| **Typed event store + session protocol** | Every research state change emits a typed event (`factor_verdict_filed`, `capability_evidence_filed`, `memory_doctrine_locked`, ...). Session lifecycle (research_new / audit / ops / doctrine / exploration) is first-class with pre-flight + exit conditions. Ad-hoc state is structurally excluded. | `engine/research_store/` + CLAUDE.md "Session Protocol Doctrine" |

---

## What this project IS (one paragraph)

A working **applied AI-quant engineering MVP** built solo. The system
combines a deployed 4-sleeve diversified backtested book with an
LLM-augmented research pipeline that ingests papers, generates
hypotheses, and runs them through a 3-pipeline rigor stack
(FORWARD/ENHANCE/PROMOTE). The differentiator is the **belief layer**:
every verdict generated by the system is preceded by an air-gapped
prediction, and every prediction is later joined to the realized
verdict and published in a calibration track record. That's how I
caught my own LLM predictor underperforming a naive family-prior
baseline — and published that finding in the arxiv preprint instead
of hiding it. The full self-tuning loop (measure → tune → re-measure
honestly → revise) is reconstructable from the git history.

## What this project IS NOT (honest framing)

- **Not a hedge fund** — simulated capital only, no real money
- **Not Renaissance / Citadel** — Sharpe 1.32 is upper-tier
  multi-strategy hedge fund range, NOT prop-fund tier (Sharpe 2-3+)
- **Not a novel theoretical paper** — production-applied use of
  established literature (Fama-French, López de Prado, Politis-Romano,
  Hosmer-Lemeshow, Bailey-LdP, Frazzini-Pedersen, etc.)
- **Not AGI / autonomous AI** — LLM is a *bounded tool* in the
  pipeline; capital decisions stay human; predictions are air-gapped
  from verdict logic at the code level
- **Not battle-tested at scale** — single user, single workstation,
  Windows + Task Scheduler crons. WRDS data dependency means external
  reviewers cannot reproduce the live data path without their own
  credentials

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  4 AUTONOMOUS AGENTS                                         │
│  papers_curator → 50+ papers/day → tag → summarize →        │
│                   synthesize hypothesis candidates           │
│  strengthener   → review → FactorSpec extract → dispatch    │
│  persona α/β/γ  → pre-mortem / cross-domain / replication   │
│  decay_sentinel → monitor deployed sleeves → emit alerts    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  3-PIPELINE SPLIT                                            │
│  FORWARD  "is X a real alpha?"  → GREEN/MARGINAL/RED         │
│           (FF5+MOM spanning, NW-t HAC, Bailey-LdP DSR)       │
│  ENHANCE  "does X' improve deployed X?" → IMPROVE/NOISE/DEG  │
│           (Politis-Romano paired bootstrap, JK Sharpe-diff)  │
│  PROMOTE  "deploy as new sleeve?" → 9 gates + human          │
└─────────────────────────────────────────────────────────────┘
                          │ verdict events
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  BELIEF LAYER (5 phases, all LIVE)                           │
│  Phase 1: predict-commit BEFORE each verdict (air-gapped)    │
│  Phase 2: autopsy joins prediction ↔ verdict                 │
│  Phase 3: track record markdown (daily refresh 06:35)        │
│  Phase 4: closed-loop prior calibration from autopsies       │
│  Phase 5: track-record-aware synthesizer context             │
└─────────────────────────────────────────────────────────────┘
                          │ feedback
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  RISK + DEPLOYMENT (12-gate Risk Manager + HUMAN gate)       │
│  Position caps (2-tier Basel-III), HHI, VaR-95, ES-95,       │
│  kill switch. Capital decisions: HUMAN, never auto.          │
└─────────────────────────────────────────────────────────────┘
```

Full architecture detail: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
Internal design memory: [INTERNAL_DESIGN_INDEX.md](INTERNAL_DESIGN_INDEX.md)

---

## Operator Console (UI-triggerable end-to-end pipeline)

9 Pipeline Stations let an external operator drive the full research
chain through clicks alone — no Claude conversation required.
Sessions are typed (per CLAUDE.md doctrine), every action emits typed
events, capital decisions stay human (S7+S8 route to /approvals
without auto-deploying; the `mutates_capital` flag on StationSpec is
import-time enforced).

```
Session (research_new / audit / ops / doctrine / exploration)
   │
   ├── S1  Paper Ingest         arxiv URL → ClaimType → registry          user_data     $0.00
   ├── S2  Hypothesize          recent papers → new hypothesis (LLM)      snapshot      $0.10
   ├── S3  FactorSpec Extract   hypothesis → factor_specs.jsonl           user_data     $0.05
   ├── S4  FORWARD Dispatch     spec → strict-gate verdict (8 stat tests) wrds_required $0.10
   ├── S5  ENHANCE Dispatch     variant CSV → paired bootstrap verdict    snapshot      $0.00
   ├── S6  Verdict View         verdict_event → prediction+autopsy drill  snapshot      $0.00
   ├── S7  PROMOTE 9-gate (MVP) GREEN verdict → /approvals (HUMAN gate)   snapshot      $0.00
   ├── S8  Rollback             deployed sleeve → /approvals (HUMAN gate) snapshot      $0.00
   └── S8b Doctrine Lock        form → memory/*.md + MEMORY.md + event    user_data     $0.00
```

Each station has 5 elements: (1) preflight checks, (2) JSON-schema
config form, (3) cost-gated trigger, (4) SSE progress stream,
(5) result + lineage hints to next station. Architecture doc:
[docs/architecture/operator_console.md](docs/architecture/operator_console.md)

Phase 2.2 / 3 deferred: S5 LLM-driven variant builder (today the
operator supplies variant returns as a CSV path), S7 Gates 2-8 full
statistical implementations (Gates 1+9 wired in MVP).

---

## The self-tuning loop (load-bearing differentiator)

```
W6-rigor    measure → Brier 0.374 (LLM-only)
              ↓
W6-rigor-A  tune    → 0.353  (N=5→3, α=3→1; -6%)
              ↓
W7-v05      sweep   → 0.254  (in-sample per-family ensemble)
              ↓
W7-v06      wire    → flag-gated OFF
              ↓
W7-v07      activate→ 0.246  (per-family w_fam)
              ↓
W7-v08      LOOCV   → 0.278  (revealed overfit gap +0.018)
              ↓
W7-v09      CORRECT → 0.260  (pure family-empirical w=1.0)
              ↓
OOS evidence        → 2/2 ensemble-active Brier 0.000 (perfect, n=2)
```

The system **measured → tuned → re-measured honestly → revised
publicly**. That's the entire research methodology, end-to-end, in
one weekend. Git log: `git log --oneline e1478826^..HEAD`.

---

## Key artifacts

| Artifact | What |
|---|---|
| `docs/arxiv_preprint_draft_2026-06-22.md` | arxiv preprint v0.9 (6141 words, markdown source) |
| `docs/arxiv_preprint_2026-06-22.tex` | LaTeX submittable (830 lines) |
| `docs/figures/` | 3 figures: reliability diagram + per-family CI + baseline comparison |
| `engine/research/belief.py` | Belief layer Phase 1 predictor (air-gapped) |
| `engine/research/belief_track_record_rigor.py` | 8 statistical rigor tests |
| `engine/research/belief_ensemble_sweep.py` | Per-family ensemble sweep + LOOCV |
| `engine/research/burndown_ranker.py` | FORWARD vs ENHANCE classifier |
| `engine/research_store/` | Typed event store (canonical record of all research state) |
| `engine/agents/papers_curator/` | Paper ingestion pipeline + ClaimType router |
| `engine/risk_manager/` | 12-gate Risk Manager MVP |
| `CLAUDE.md` | Project-wide doctrine (3 STANDING doctrines) |
| `docs/architecture/SKILL_three_layers.md` | Three-layer architecture (substrate / experience / self-correction) — directly comparable to and contrasted with concurrent agentic factor-mining work (QuantML / Kimi+WorldQuant BRAIN, 2026-06-23). Six load-bearing differentiators documented. |

---

## Reproducibility

Each headline number reproduces from a single command. None requires
WRDS / Bloomberg / paid API access for the core belief-layer claims:

```bash
# 1. Belief layer rigor — 8 statistical tests
python scripts/reports/report_belief_track_record_rigor.py
# Reads: data/research/autopsies.jsonl (sample fixture in
# tests/fixtures/ exercises the same code path)
# Reproduces: Section 3 of arxiv paper

# 2. Per-family ensemble sweep + LOOCV
python scripts/reports/report_belief_ensemble_sweep.py
# Reproduces: Section 4 of arxiv paper, including the LOOCV honesty pass

# 3. ClaimType router accuracy on labeled fixture
python -m pytest tests/test_papers_curator_claim_type_router.py
# Reproduces: Section 2.4 (router v2 false-positive rate ~0%)
```

### What you can actually run from this snapshot

| Path | Needs | Time | What you get |
|---|---|---|---|
| **Read the code + paper** | nothing | 0 | The methodology + statistical anchors are all in `docs/` + inline; this is the path most readers should take |
| **Reproduce the belief-layer headline numbers** | Python 3.10+ | ~10 min | Brier ~0.37, the honest-negative finding, ClaimType router accuracy. Fixture at `data/_samples/autopsies_sample.jsonl` ships with the snapshot so the 3 numbered scripts in *Reproducibility* above run end-to-end without WRDS / API keys. |
| **Run the test suite** | Python 3.10+ | ~5 min | 357 test files / ~5,800 assertions; 5,744 of 5,805 collect-time IDs run without WRDS (61 deselected probe live WRDS) |
| **Run the Operator Console UI** | Python 3.10+ · Node 18+ | ~20 min | Local FastAPI backend + Next.js frontend; ingest a paper, browse stations, see SSE progress live |
| **Reproduce the live backtest** | WRDS subscription + credentials | several hours | The Sharpe 1.32 4-sleeve canonical replay; needs CRSP / Compustat / IBES / OptionMetrics |

### Quick start — minimal path (no WRDS, no Node, ~10 min)

```bash
git clone https://github.com/falsifiable-t/macroalphapro.git
cd macroalphapro

# Python 3.10 or 3.11 (3.12 untested, 3.13 will likely break torch deps)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Sanity: 30 fast tests (router + track-record) pass clean, no external data
python -m pytest tests/test_papers_curator_claim_type_router.py tests/test_belief_track_record.py -q

# Reproduce the headline Brier number from the arxiv paper
python scripts/reports/report_belief_track_record_rigor.py
```

### Quick start — full stack with UI (~20 min)

```bash
# 1. Backend (same as minimal path above, then:)
uvicorn api.main:app --reload --port 8000

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev          # http://localhost:3000

# Open http://localhost:3000/console — Operator Console launchpad
# Open a research_new session, trigger S1 Paper Ingest with an arxiv URL
```

### Installation troubleshooting

- **`win10toast` fails on Mac/Linux**: it's Windows-only, fail-soft at runtime. Remove the line from `requirements.txt` before installing on non-Windows.
- **`wrds` install hangs / errors**: this dep requires a WRDS account configured in `~/.pgpass`. If you don't have WRDS access, comment out the `wrds` line — every belief-layer demo runs without it.
- **`sentence-transformers` pulls ~2GB of PyTorch**: required for the doctrine retrieval layer; if you only want to reproduce belief-layer numbers, you can `pip install --no-deps -r requirements.txt` then install just `numpy pandas scipy scikit-learn arch`.

---

## Honest caveats (read before judging)

1. **Public snapshot ≠ private dev** — this repo is a sanitized public
   mirror of a private monorepo. Per-session research artifacts
   (predictions, autopsies, paper cache, capability evidence) are
   excluded. The methodology + code is fully here; the live data
   stream is not.

2. **Windows-tested** — primary dev is on Windows + Task Scheduler.
   Cross-platform cron registration (Linux cron / macOS launchd) is
   not in this snapshot. The Python code itself is platform-neutral.

3. **WRDS dependency for real data** — the deployed book uses WRDS
   (Compustat / CRSP / IBES / OptionMetrics). External reviewers must
   BYO credentials for live runs. Sample fixtures in
   `tests/fixtures/` exercise the same code paths.

4. **Sharpe 1.32 is backtest, not live** — paper-trade started
   2026-05-13. Live forward NAV is too short for annual Sharpe
   inference (n=21 days as of 2026-06-22). Per-design CPCV expectation
   is Sharpe 0.85-1.15 (forward typically degrades from backtest).

5. **Sample size on belief layer** — n=94 autopsies at paper v0.9
   (live count grows daily via cron; current in
   `data/research/belief_track_record_rigor.json`). Bootstrap CIs are
   wide. The "0.000 Brier on 2 OOS pairs" headline is directional
   evidence, not validation. Validation requires ~30+ OOS pairs.

6. **AI-augmented, not autonomous** — LLMs are bounded tools (paper
   summarization, hypothesis synthesis, FactorSpec extraction). All
   verdicts are computed by deterministic Python statistics. All
   capital decisions are human. The "self-tuning loop" tuned a
   *predictor weight*, not a *capital allocation*.

---

## Statistical anchors (academic literature relied on)

| Anchor | Year | Used for |
|---|---|---|
| Fama-French | 1993, 2015 | FF3/FF5 spanning regression |
| Carhart | 1997 | MOM factor |
| Hou-Xue-Zhang | 2015 | q-factor model spanning |
| Frazzini-Pedersen | 2014, 2018 | BAB factor + institutional alpha = 70% enhance |
| Asness-Pedersen | 2013 | QMJ factor |
| Novy-Marx | 2013 | GP/A profitability |
| Newey-West | 1987 | HAC standard errors |
| Bailey-López de Prado | 2014 | DSR multi-test correction |
| Politis-Romano | 1994 | Circular block bootstrap (paired Sharpe-diff) |
| Jobson-Korkie / Memmel | 1981, 2003 | Sharpe-diff t-stat (paired) |
| Hosmer-Lemeshow | 1980 | Calibration goodness-of-fit |
| Benjamini-Hochberg | 1995 | FDR correction multi-family |
| Mann-Kendall | 1948 | Trend test on per-period Brier |
| McLean-Pontiff | 2016 | 32-58% Sharpe drop post-publication |
| Carr-Wu | 2009 | VRP equity-only finding (consistent with our bond-VRP RED) |

---

## License

[MIT](LICENSE).

## Contact

[LinkedIn](https://www.linkedin.com/in/zhangxizhe) · [GitHub](https://github.com/falsifiable-t/macroalphapro) · e1521244 @ u.nus.edu

Open to applied-AI research-engineer roles (LLM systems + calibration / agent orchestration) and quant research-engineer positions at research-driven firms, mid-2026 onward.

---

_Public mirror of a private development monorepo. Snapshot built via [`scripts/publish/build_public_snapshot.py`](scripts/publish/build_public_snapshot.py); operator guide at [`docs/PUBLISH_PIPELINE.md`](docs/PUBLISH_PIPELINE.md)._
