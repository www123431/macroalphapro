# Three-Agent Public-Repo Audit — 2026-06-24

**Context.** After the Operator Console build (Sessions 14 + 15;
7 of 9 Pipeline Stations shipped) and the first public push to
GitHub, we dispatched three parallel specialist agents to audit the
public snapshot before broader sharing. This doc captures their
findings, what got fixed, and what's **deliberately deferred** so a
future session can pick up the backlog without re-running the audit.

This is the load-bearing doc for "what's left on the Operator
Console + portfolio repo." Read first if continuing this work line.

---

## TL;DR

| Agent | Verdict | Action taken |
|---|---|---|
| **Recruiter 90-second simulator** | "Interview yes, forward-to-team NO until 3 fixes ship" | 9 P0 fixes shipped in v3 commit `31f6186` |
| **Senior quant engineer code review** | "3 must-fix before merge" + ~15 secondary findings | 1 docstring fix (`registry.py:10`) shipped; 17 items deferred |
| **Paranoid security scan** | **CLEAN** (no secrets, no PII, no internal paths leaked) | 2 cosmetic URL inconsistencies fixed |

Snapshot v3 stats: 2,165 files / ~31 MB / 0 forbidden-pattern hits.

---

## What got fixed (P0, shipped in v3)

1. README `git clone` block was broken (cloned `macro-intern-agent`
   then `cd macroalphapro` — dir mismatch crash on first copy-paste).
2. PROJECT_OVERVIEW.md line 48 still said `n=101 autopsies`; current
   rigor JSON says `n=94`. Unified across all files.
3. Papers cache count: README said 661, PROJECT_OVERVIEW said 633
   (lines 69, 286, 376). Unified to 661 (latest count).
4. PROJECT_OVERVIEW § numbering `§2.6` / `§2.6a` / `§2.6b` looked
   like a merge conflict. Renumbered `§2.7` / `§2.8` / `§2.9`.
5. README claimed "13 standing rules in CLAUDE.md" — actual file
   has 3 STANDING doctrines (Research Event Emission / Session
   Protocol / FORWARD vs ENHANCE Statistical Separation). Now
   accurate.
6. "arxiv preprint draft v0.9 in submission" was self-contradictory
   ("draft" + "in submission"). Now "v0.9 ready for submission".
7. Repo URL inconsistency across docs (`zhangxizhe/macroalphapro`,
   `www123431/macro-intern-agent`, `www123431/macroalphapro`).
   Unified to `github.com/www123431/macroalphapro`.
8. INTERNAL_DESIGN_INDEX:769-771 mentioned the deleted
   `www123431-patch-1` branch. Scrubbed.
9. `engine/operator_console/registry.py:10` docstring claimed
   `__init_subclass__` auto-registration — actually manual
   `registry.register(...)`. Docstring fixed + flagged the
   move-to-init_stations() polish item.

---

## Deferred — P1 engineering debt (code-side; from Agent 2)

These are real bugs / smells that didn't ship in v3 because each
needs careful work and the v3 push needed to land first. Estimated
effort + file:line references included so a future session can pick
up cold.

### P1.1 — `JOB_QUEUES` memory leak + orphan-queue bug
**Files**: `engine/operator_console/worker.py:44,47,50,79,219`
**Problem**: Late SSE subscriber (connects after `cleanup_job` ran)
calls `get_or_create_queue` which **re-creates a queue that no
producer will ever fill**. That queue leaks forever. The 100ms
`asyncio.sleep` window at line 219 is way too short to guarantee
all subscribers have drained.
**Fix**: subscriber refcounting + periodic TTL sweep for queues
older than N seconds with no consumer.
**Effort**: 1-2h.
**Severity**: Real bug, but only bites at scale (many concurrent
users). For the current solo-user portfolio repo, won't show in
testing — but **must fix before opening Operator Console to
external users**.

### P1.2 — Silent `JSONDecodeError` swallow (D4 cost-cap risk)
**Files**: `engine/operator_console/store.py:124,151,241` +
`engine/operator_console/cost_ledger.py:84`.
**Problem**: A corrupt line in `jobs.jsonl` / `events.jsonl` /
`session_cost_ledger.jsonl` is silently skipped with `continue`.
For the cost ledger this is **load-bearing for D4**: a corrupted
line means the user effectively gets *more* budget than the
declared cap. Cost cap enforcement was the whole point of D4 —
silent skip breaks the invariant.
**Fix**: at minimum `logger.warning(...)` on parse failure; ideally
emit an `operator_console_corruption_detected` event so the UI
surfaces it.
**Effort**: ~30 min.
**Severity**: Doctrine-violation under sustained adverse
conditions. Low probability today, but a known unknown.

### P1.3 — Capital-decision doctrine enforced by **convention, not code**
**Files**: `engine/operator_console/pipeline_station.py` +
`schema.py` + `worker.py`.
**Problem**: S7 (PROMOTE) and S8 (Rollback) both honor the doctrine
by writing pending-proposal rows instead of mutating
`engine/portfolio/deployed_registry`. But **nothing in the base
class or worker forbids a future S9 from calling
`deployed_registry.save_active()` directly**. Today's safety net is
`git grep deployed_registry`. A junior engineer adding a new
station has no scaffolding to stop them.
**Fix**:
  - Add `StationSpec.mutates_capital: bool = False`
  - Refuse to register any station with `mutates_capital=True` that
    doesn't route through `/approvals`
  - Optionally: import-time lint that scans station modules for
    forbidden imports (`from engine.portfolio.deployed_registry import save_active`).
**Effort**: 1h.
**Severity**: Doctrine load-bearing. Low risk today (solo dev), but
the whole point of CLAUDE.md doctrine is that it should outlive any
single contributor's memory.

### P1.4 — `pipeline_station.py:60` STATION_SPEC has no enforcement hook
Subclasses that forget `STATION_SPEC: ClassVar[StationSpec] = ...`
crash at first `cls.STATION_SPEC` access at *runtime*, not at
import. Add `__init_subclass__` check that raises
`TypeError("Station X must declare STATION_SPEC")` immediately on
class creation. **Effort**: 15 min.

### P1.5 — `worker.py:201,216` calls `emitter._put(...)` from outside
`_put` is private-by-convention (leading underscore) but used as
public API from `run_job`. Either promote to `emitter.terminal(...)`
or rename without the underscore. **Effort**: 5 min.

### P1.6 — `datetime.utcnow()` deprecated in Py 3.12
Multiple files. Replace with `datetime.now(timezone.utc)`. Trivial
sweep. **Effort**: 10 min.

### P1.7 — Type-hint gaps
`config: dict` → `dict[str, Any]` across `pipeline_station.py:65,79,88` +
every station's `preflight/execute/render_config_form`.
`worker.py:44` `dict[str, asyncio.Queue]` missing element type
(should be `asyncio.Queue[dict[str, Any]]`).
`schema.py:14` unused `Callable` import.
**Effort**: 15 min global sweep.

---

## Deferred — P2 polish (docs + UX + portfolio)

### P2.1 — README TL;DR table is 13 rows, recruiter eyes glaze at row 6
**Source**: Recruiter audit. Cut to 6 rows. Keep the most senior-
signal items: backtest Sharpe, honest negative finding, Operator
Console headline, multi-agent framing, doctrine count. Move the
rest to PROJECT_OVERVIEW. **Effort**: 20 min.

### P2.2 — Bury-the-lead: bond-VRP autonomous demo deserves README top
Currently buried in PROJECT_OVERVIEW §2.5. **Most impressive
concrete thing** — paper-to-verdict with no human in the prediction
loop. Move a 3-sentence summary + link to README hero section.
**Effort**: 15 min.

### P2.3 — Zero screenshots / GIFs of Operator Console
Operator Console is the visual differentiator. **At least 1 PNG**
(workflow trace OR calibration page OR an in-flight station) would
5× README dwell time per the recruiter audit. A 30-second GIF of
clicking through one Pipeline Station would be golden.
**Effort**: 30-45 min (capture + crop + commit + reference in
README).

### P2.4 — Missing "why me / why now" sentence
NUS MSBA framing only appears in byline. One sentence on what's
sought next (PhD? Quant role? Research engineer?) lets the reader
self-route. Currently the reader has to guess what to do with the
work. **Effort**: 5 min (1 sentence after the LinkedIn link).

### P2.5 — Replace `render_config_form` JSON-Schema dict with pydantic
Stations could declare a `ConfigModel(BaseModel)` and the base
class emits `.model_json_schema()` automatically. Saves ~30 LOC per
station + removes `x-ui-*` typo risk (currently each station hand-
rolls the JSON Schema). **Effort**: 1.5h to refactor all 7 stations.

### P2.6 — Missing PipelineStation hooks: `cleanup()` / `dry_run()` / `idempotency_key`
Today there's no:
- `cleanup()` for failed stations that wrote partial artifacts
- `dry_run()` to preview without side effects
- `idempotency_key` on `create_job` — double-click on the trigger
  button creates 2 jobs.
**Effort**: ~45 min each hook.

### P2.7 — Stations hard to unit-test as written
- Stations import `from engine.portfolio.deployed_registry import load_active` inline (`s8_rollback.py:71,94`) — non-injectable.
- Module-level path constants read from real repo layout (no env-var override).
- `registry.register(...)` at import time is a global side effect.
**Fix**: introduce `Settings` object passed into stations + move
registration into explicit `init_stations()` entry point.
**Effort**: 2-3h (touches every station + worker + tests).

---

## Deferred — remaining stations (Operator Console)

| Station | Status | Why deferred |
|---|---|---|
| **S2 Synthesize** | Not started | LLM-heavy ($0.10/call); papers → new hypothesis; design doc §5 has full spec |
| **S5 ENHANCE Dispatch** | Not started | Needs `variant_returns: pd.Series` input wiring; non-trivial because the time-series can't easily be passed via JSON form. Probable approach: accept hypothesis_id + auto-build variant via template (similar to how S4 inputs work). |
| **S7 Gates 2-8** | MVP-only (Gate 1 + Gate 9 wired; 2-8 deferred YELLOW in UI) | Each gate is a separate statistical computation (Almgren-Chriss cost-robust / PIT audit / γ replication / Mann-Kendall multi-period / FF5+MOM anchor-residual / cross-sleeve correlation / Pastor-Stambaugh capacity). Estimate ~3-4h per gate. |

---

## Known nuances (don't second-guess these as bugs)

### 4-sleeve vs 5-sleeve
- `data/portfolio_replay/v1_combined_replay_verdict.json` is the
  **canonical replay** = **4 sleeves**: K1_BAB / D_PEAD / PATH_N /
  CTA_PQTIX. This is the Sharpe 1.32 / MaxDD -5.8% claim's source.
- `engine.portfolio.deployed_registry.load_active()` returns the
  **current operating book** = **5 sleeves**: equity_book /
  cross_asset_carry / cross_asset_tsmom / crisis_hedge_tlt_gld /
  mom_hedge_overlay.
- Both are real. README + arxiv §A.8 disclose both.
- S8 Rollback reads the **5-sleeve** registry (current operating
  state), which is correct.

### n=94 autopsies vs n=101 raw rows
- `data/research/autopsies.jsonl` line count varies as cron processes
  new autopsies daily.
- `data/research/belief_track_record_rigor.json` reports the **n
  used by the bootstrap CI** (after filtering autopsies with missing
  data / not-yet-eligible / etc.). Current value: 94.
- Historical session narratives in INTERNAL_DESIGN_INDEX may show
  older snapshots (101 etc.) — left as historical accuracy, not
  retroactively edited.

### Operating cost $19/mo vs earlier "<$5/mo" claims
- The "<$5/mo" was a stale target from an early design memory
  before backfill + cron crons were all running.
- Real measured run-rate is ~$19/mo (verified from
  `data/llm_cost_ledger.jsonl`: 16,910 calls / $26.75 over 41 days).
- Steady-state excluding initial backfill spike (week of
  2026-05-12: 15,094 calls from one-time PDF backfill): ~$16/mo.
- All portfolio docs now reflect $19/mo run-rate.

---

## Repo URL evolution (for future-Claude debugging)

```
First push: github.com/www123431/macro-intern-agent  (March 2026
  — old Streamlit-era; got forced-overwritten with v1/v2 snapshots
  during Session 15)

Renamed:    github.com/www123431/macroalphapro  (2026-06-24 — user
  did GitHub UI rename to drop the "intern" suffix that read as
  junior + drop ~1 GB of unreachable git-cruft accumulated from
  force-pushes; this is the canonical URL referenced throughout
  all docs from v3 onward)
```

Old branch `www123431-patch-1` was deleted before the rename
(6 commits of `Update app.py` edits to the deprecated Streamlit
UI from March — orphan WIP that didn't belong on a portfolio
repo).

---

## Push history (public snapshot)

| ver | commit | what shipped |
|---|---|---|
| v1 | `60edef0` | Initial public release (Sessions 14 + 15 cumulative; 2,311 files) |
| v2 | `7839240` | Contact section + headline number cleanup (6 portfolio P0 issues) |
| v3 | `31f6186` | 3-agent senior audit fixes (9 P0; this commit's referenced doc) |

Each new push uses `git push --force origin main` from a fresh
local `git init` of the rebuilt snapshot dir (per the dual-repo
B1 architecture in `docs/PUBLISH_PIPELINE.md`).

---

## What a future Claude session should do first

1. Read this doc (5 min).
2. Check `git log --oneline -10` on both the private repo
   (c:${REPO_ROOT}/Desktop/intern) and the public mirror dir
   (${REPO_ROOT}/Desktop/macroalphapro-public).
3. If continuing operator-console build: pick from P1.1-P1.7
   (engineering debt) or S2/S5 (remaining stations). Do NOT start
   with P2.1-P2.7 polish unless the engineering debt is closed —
   polish on top of known bugs is malpractice.
4. If updating portfolio docs: cross-check the "Known nuances"
   section above so you don't accidentally "fix" things that are
   intentional (4-vs-5 sleeves, n=94 vs n=101, $19/mo vs $5/mo).
5. After any code or doc change: re-run
   `python scripts/publish/build_public_snapshot.py` then push the
   public mirror per the standard flow in `docs/PUBLISH_PIPELINE.md`.
