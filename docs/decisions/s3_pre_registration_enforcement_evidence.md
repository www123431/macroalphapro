# S3 Pre-Registration Enforcement — VERDICT: CAPABILITY PASS (2026-05-04)

**Spec**: [../spec_pre_registration_enforcement.md](../spec_pre_registration_enforcement.md) v1.0
**spec_hash[:16]**: `292fdd6039f90d05`

**Memory**:
- `project_2026_summer_roadmap.md` §3 — S3 是 4-month roadmap 第三/可选项，置换原 "S3 optional"
- `project_b_plus_marginal_2026-05-04.md` — B++ MARGINAL 触发"forward FDR 必须严格 deflate" 的论证
- `project_s2_reflection_complete_2026-05-04.md` — S2 capability 主线第一项；S3 是第二项
- `feedback_no_llm_as_judge.md` — Layer 1 / Layer 2 separation；S3 完全零 LLM
- `feedback_verify_each_step.md` — verify-then-go 纪律
- `feedback_spec_power_analysis.md` — pre-registration 纪律学术依据

**状态**: ✅ **Capability PASS** — 4 sub-sprints (S3.0-S3.4) + 全部 verification facets PASS。

---

## TL;DR

把"科学纪律"从依赖人记忆升级为 **harness 自动强制执行**。具体三层：

1. **Spec 不可静默修改**：每份 spec 在 `SpecRegistry` 表注册时锁 git-blob hash；任何修改必须走 `amend_spec` workflow；`current_hash != git_blob_hash` 自动检测。
2. **n_trials 自动累加**：每次 amendment 按 kind 加权（clarification +0 / scope_narrow +0 / threshold_tweak +1 / hypothesis_amend +3 / endpoint_swap +5），实时反馈进 `EFFECTIVE_N_TRIALS` → 任何后续 backtest / paper trading 阳性结果的 deflated p-value 都自动收紧。
3. **HARKing 自动检测**：4 条规则 R1-R4 (silent edit / threshold drift / unannounced trial / predictions rewrite)，零 LLM，每日 cron 自动跑，UI 红黄橙牌展示。

```
Verification matrix (全部 0 LLM, 全部 deterministic):
  S3.1 SpecRegistry + CLI + retro-backfill          7/7  PASS
  S3.2 backtest n_trials + DecisionLog spec_hash    7/7  PASS
  S3.3 HARKing R1-R4 trigger + idempotency          6/6  PASS
  S3.3 UI panel + cron hook                         5/5  PASS
                                                    -----------
                                                    25/25  PASS
```

**实战已用**：S3.1 上线后立即被 SSRN AI 披露要求倒逼调用——SSRN paper spec 走完整 amendment workflow（kind=clarification, n_trials_added=0, ledger 持久化）。这是"工具上线即被使用而非演示"的最强信号。

---

## 1. Why（first principle 链路）

> 项目 7 个 hypothesis test 已 6 reject + 1 marginal，发表 SSRN 需要 deflated p-value 真实反映 actual trial budget；spec 文件 15+ 份且每份生命周期内会改 3-5 次，这些修改若静默就是未声明的 hyperparameter trial → 6 个月项目剩余周期预计 30-50 个静默 trial → BHY-FDR 阈值会被人为放松 5-10% → 任何后续阳性都不可信。

S3 不是 polish，是 **paper trading E + B++ 后续可发表性的硬前置**——没有 forward integrity 框架，未来阳性结果都得自己加 caveat 说"trial budget 估计偏低"，对 reviewer 就是软肋。

**学术依据**：
- @benjamini2001control BHY FDR — 多重检验下 dependency 容忍下的 conservative 阈值
- @harvey2016cross — t > 3 标准 + N_TRIALS audit 框架
- @nosek2018preregistration — 预登记革命
- @simmons2011false — HARKing / undisclosed flexibility 的 foundational 警告
- López de Prado (2018) §11 — power analysis 与 deflation

---

## 2. 工程交付清单

### 2.1 Schema (engine/memory.py)

```python
class SpecRegistry(Base):  # spec_path / git_blob_hash / current_hash /
                            # amendment_log (JSON) / status / retro_registered /
                            # first_referenced_at / n_trials_contributed
class HARKingFlag(Base):    # rule (R1-R4) / spec_path / severity / detected_at /
                            # resolved_at / notes
DecisionLog.spec_hash = Column(String(64), nullable=True)  # S3 column
```

Migration: `_migrate_db()` 加新表的 PRAGMA 检查 + DecisionLog `spec_hash` ALTER。

### 2.2 Core API (engine/preregistration.py, ~600 lines)

```
register_spec(path, retro=False)         首次注册（或 idempotent re-validation）
amend_spec(path, kind, reason)            按 6 种 kind 累加 n_trials + ledger
validate_reference(spec_path)             每次 backtest 引用前检 hash + 标 first_referenced_at
list_specs()                              UI / CLI consumption
detect_harking(as_of=None)                R1-R4 4 条规则，写 HARKingFlag
compute_pre_registration_n_trials()       sum forward (non-retro) contributions
_compute_git_blob_hash(path)              `git hash-object` 等价
CLI: python -m engine.preregistration {register, amend, list, validate, n_trials, harking}
```

### 2.3 Backtest 集成 (engine/backtest.py)

```python
EFFECTIVE_N_TRIALS = sqrt(grid_raw) + pre_registration
                                       ^                  ← 从 SpecRegistry 实时拉
_N_TRIALS_AUDIT["pre_registration"] = compute_pre_registration_n_trials()
refresh_effective_n_trials()           ← 公开 API for amendment 后即时刷新
```

### 2.4 SectorPipelineAgent 集成 (engine/agents/sector_pipeline/agent.py)

每条 sector_pipeline 决策 save_decision 时自动注入 `docs/spec_sector_pipeline_unification.md` 的 git-blob hash 到 DecisionLog.spec_hash → R3 (unannounced trial) 检测有完整素材。

### 2.5 Cron Hook (engine/orchestrator.py)

`run_daily` 末尾 try/except 包裹 detect_harking 调用——HARKing 检测每日 nightly 跑，失败不阻塞 daily cycle 主流。

### 2.6 UI (pages/agent_observability.py Section H)

3 子区：
- **H.1 EFFECTIVE_N_TRIALS breakdown**：4 metric (grid_raw / sqrt / pre_reg / effective) + audit 表
- **H.2 SpecRegistry**：18 行 spec 表 + amendment timeline drill-down
- **H.3 HARKing flags**：3 严重度 metric (CRITICAL/HIGH/MEDIUM)、open flags 表、resolved 折叠

### 2.7 Verification (scripts/)

| Script | Facets | 覆盖 |
|---|---|---|
| `verify_s3_1_specregistry.py` | 7 | ORM + register/amend CLI + validate + n_trials + 18-spec backfill |
| `verify_s3_2_backtest_decisionlog.py` | 7 | DecisionLog column + audit dict + arithmetic + liveness + spec_hash round-trip + auto-inject |
| `verify_s3_3_harking.py` | 6 | R1-R4 trigger + idempotency + persistence |
| `verify_s3_3_ui_cron.py` | 5 | UI cold + seeded + cron wiring + cleanup |
| `verify_ssrn_ai_disclosure.py` | 6 | 实战 amendment 走 workflow（SSRN AI 披露） |
| **总计** | **31** | **31/31 PASS** |

---

## 3. 实战首战（2026-05-04）

S3.1 上线**当天**就被 SSRN policy 倒逼调用：

```
$ python -m engine.preregistration amend docs/spec_ssrn_paper_v1.md \
    --kind clarification \
    --reason "SSRN policy 2026 requires AI-use disclosure in abstract +
             standalone section; spec amended to add Section 8 contract,
             no hypothesis or threshold change"
amend_spec: docs/spec_ssrn_paper_v1.md kind=clarification n_trials_added=0
            cumulative=0
OK: amendment recorded id=17
```

为何 `n_trials_added=0`：clarification kind 不影响假设或阈值；SSRN paper spec 是 retro_registered，amendment 不污染 forward EFFECTIVE_N_TRIALS。但 ledger 永久 audit trail 留下，reviewer 可追溯。

paper.md 同步更新：摘要加 AI 披露 preamble + §8 standalone "AI-Use Disclosure (per SSRN policy)" 三桶分类（research subject / research assistant / evaluation judge）。验证 6/6 PASS。

---

## 4. Verdict tiers (Spec §0)

按 spec 没有正式 verdict tier 闸门（capability spec 不直接产 alpha）。改用**功能 + 验证完整度**衡量：

| Tier | 条件 | 实际 | Pass? |
|---|---|---|---|
| **CAPABILITY PASS** | (a) Schema + CLI + register/amend workflow + retro-backfill ≥10 specs ≥1 forward; (b) backtest n_trials 实时联动; (c) 4 条 HARKing 规则全部 trigger; (d) UI 渲染 cold + seeded; (e) cron 接入 daily; (f) 实战至少 1 次真实 amendment | (a) 18 retro + 1 forward + 1 实战 amendment ✅; (b) refresh 后 43→45 ✅; (c) R1-R4 all fire ✅; (d) AppTest 0 exceptions ✅; (e) orchestrator.run_daily 接入 ✅; (f) SSRN 披露 amendment ✅ | ✅ **TRIGGERED** |
| **PARTIAL** | (a-c) met, (d-f) partial | n/a | n/a |
| **FAIL** | (a) 或 (b) 缺失 | n/a | n/a |

**Verdict**: ✅ **CAPABILITY PASS**

---

## 5. 与 S2 Reflection / S4 SSRN paper 的耦合

- **S2 Reflection**: AgentReflection 表共享 `decision_ref_id`；未来扩展可让 reflection 文本 cite spec_hash → 反思自己依赖了哪个版本的 spec
- **S4 SSRN paper**: paper §2 Methodology 明确描述本 framework，§Appendix B 列出 8 个 spec_hash registry → SSRN reviewer 可 reproducibility 复现 SHA1 / SHA256
- **B++ marginal evidence**: 后续若想 promote QL01 从 MARGINAL → DISCOVERY (例如 retest on 2008-2017 OOS)，新增 spec amendment 自动 +n_trials → 必须用更高门槛的 deflated p-value，避免误导

---

## 6. 学术 framing（reviewer / 答辩用）

**不能 claim**：
- ❌ "S3 让我们项目赚钱" — capability 不是 alpha
- ❌ "Pre-registration 改善 hit rate" — 跟 alpha 完全独立维度
- ❌ "我们发明了 spec_hash" — git blob hash 已是行业标准；本框架是工程合成，不是密码学新原语

**可以 claim**：
- ✅ "Self-enforced scientific discipline 工程化层落地，零 LLM、deterministic、可审计"
- ✅ "Olken 2015 / Nosek 2018 经济学预登记范式 + Harvey-Liu 多重检验框架的应用层移植"
- ✅ "Pre-registration discipline + agentic AI 工程合成的 case study；HARKing attack surface 全面工程封堵"
- ✅ "回应 López de Prado / Harvey / Bailey 近期 keynote 的 frontier direction"

**项目独特 extension**（区别于纯学术 OSF / AsPredicted 注册）：
1. spec_hash 锚到 git blob → 与 git 工作流原生集成，不需要外部 platform commitment
2. 与既有 EFFECTIVE_N_TRIALS dynamic 联动 → DSR / p-value 自动 deflate
3. HARKing 检测 R1-R4 是规则化的、针对 spec markdown 形态定制（不是通用 OSF 模板）

---

## 7. 后续 work（不在 S3 scope）

- **S3.5 Pre-commit hook**：git pre-commit detect spec edit 自动 prompt amend，进一步降低人工 friction（当前 v1 不做，纪律靠 transparency）
- **S3.6 Config-dict hash**：扩展到非 spec 文件（如 backtest config dict）— spec §1.3 "v2" 范围
- **S3.7 Cross-project registry**：如果以后做 wealth_manager 项目，registry 跨项目 sync — 远期
- **S3.8 PR-blocking integration**：CI flag CRITICAL HARKing on PR open（与 GitHub Actions 集成）— 远期

---

## 8. Cross-references

- 项目主线: [executive_summary.md](../executive_summary.md) — 加入 S3 capability claim "Methodological Integrity Automation"
- Roadmap: [project_2026_summer_roadmap.md](memory:project_2026_summer_roadmap.md) — S3 ✅ 完成；S2 + S4 + S3 三大 capability 主线齐
- Cleanup 状态: spec_registry 18 行 (1 forward + 17 retro)，harking_flags 表干净，DecisionLog spec_hash 列上线
