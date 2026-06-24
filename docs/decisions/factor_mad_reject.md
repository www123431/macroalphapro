# FactorMAD — REJECTED (2026-05-02 → 2026-05-03 删除)

**前置 Spec**：[../decisions/rejected/spec_factor_mad_redesign.md](rejected/spec_factor_mad_redesign.md)
**Memory**：
- `project_factor_mad_redesign.md` (2026-05-02 决策) — 已被 cleanup 跨域 supersede
- `project_factor_mad_s4_dependency_note.md` (2026-05-02 复核) — 已失效
- `project_cleanup_2026-05-03.md` (2026-05-03 删除事实)

---

## TL;DR

**FactorMAD 严格 reject** —— LLM-driven sector factor mining 在 Q1 2026（首个 mining quarter）promoted **0 out of 24** candidates。Critic gate 拒掉 100% 提案。后续判定为 master's-project scope 下不可行，2026-05-03 cleanup sprint 整体删除。

不是 ICIR 闸门设错，不是 BH FDR 设错——是 **LLM proposer 在 Gemini-1.5 + 当前 prompt + 公开 macro/news context 下产生不出能过 critic 的 sector factor 候选**。结构性证伪。

---

## 1. 设计假设（事前）

按 spec_factor_mad_redesign.md (2026-05-02)：

> "升级为机构级 alpha mining agent + 接入 operations 架构"，9 项决策锁定，5 Sprint 路径。

- **Stage 1-2 Proposer**：LLM 接 macro context + news 输出候选 factor 公式（DSL 化）
- **Stage 3 Search**：候选过 universe ICIR + IC 稳定性预筛
- **Stage 4 Critic**：第二个 LLM 角色 challenge 生成 hypothesis（multiple testing aware）
- **Stage 5 BH FDR**：跨季度候选 family-wise correction
- **Stage 6-7 Lifecycle**：promotion / dormancy / sunset

**Hypothesis**：在 4 个 seed factor (mom_3m, rev_1m, vol_adj_mom_6m, trend_strength) 之外，LLM 能 mine 出 ≥ 3 个新 factor 通过 ICIR > 0.05 + critic gate + BH FDR。

---

## 2. Q1 2026 实证结果

**Mining cycle**：2026-04-24 触发（factor_definitions 表 created_at）

| 指标 | 数值 |
|---|---|
| Raw proposals from Proposer | **24** |
| Accepted by Search (pre-screen) | 不详（未持久化） |
| Promoted past Critic | **0** |
| Promoted past BH FDR | **0** |
| Net new factors entering production | **0** |

24 个候选全部死在 Critic 那一关。

---

## 3. 失败模式分析

### 3.1 Proposer 输出质量

LLM 候选普遍是已知 anomaly 的轻微变体（动量延长/反转扁平/vol-adj rebrand）。Critic 用 multiple-testing-aware prompt + economic logic gate 直接拒——**因为这些变体跟 4 个 seed factor 高度相关**，没有 incremental information value。

### 3.2 Critic 过严还是 Proposer 过弱？

按 spec §S4 critic 设计是模拟 institutional review board 标准（multiple testing + economic prior + IS overfit risk）。

- **如果调松 critic**：会过更多伪 factor → BH FDR 后归 0，BH 在做 critic 的事
- **如果调强 proposer**：需要换更强模型（GPT-4-class）+ 更多 context（专属 financial corpus）+ chain-of-thought scaffolding —— 超过 master's-project 工程边界

### 3.3 LLM 在金融因子挖掘的文献先验

- BloombergGPT、FinGPT 公开实验未给出"LLM 挖掘出新 factor 通过严格 OOS"的案例
- López de Prado AFML §11：mass factor screening 在 multiple-testing 下 effective alpha = 0
- Harvey-Liu-Zhu (2016)：316 个 published factor 大多数 NW + multiple testing 后失效

LLM 不创造新数学规律，只 recombine 已知 statistical anomaly。当 critic gate 是 multiple-testing-aware 时，recombine 通不过。

---

## 4. 学术诚实声明

### 4.1 Critic 严格度选择是 trade-off
本 reject 的 critic 校准到 institutional 严格档（multiple-testing + economic prior + IS overfit）。如调到 retail / academic-publication 档，可能有 1-3 个 factor 过 critic gate。**但这等于把 critic 调到 BH FDR 之前的位置——失去 critic 的意义**。

### 4.2 Proposer 模型与 budget
仅在 Gemini-1.5 (free-tier) 下测试。如换 GPT-4-class 或专属 financial-pretrained 模型，proposer 输出**可能**改善——但成本与 master's-project scope 不兼容。

### 4.3 1 quarter 样本
Q1 0/24 是单 quarter 数据。多次 quarterly mining + 调 prompt 可能 marginal 改善。但**这违反 spec §13 in-sample tuning 禁令**。

### 4.4 不是 spec 设计 bug
spec_factor_mad_redesign.md 9 决策事前锁定，包括 critic 校准 + BH FDR 阈值 + ICIR 阈值。0/24 不是阈值过严的 artifact——是 proposer 输出质量与阈值不匹配。

---

## 5. 学术 + 工程教训

1. **LLM 挖掘金融 factor 在公开模型 + 公开 context 下结构性失败**：与 Bybee 2023 / FinGPT 公开实验一致，文献先验是对的
2. **multiple-testing-aware critic 是 LLM factor mining 的硬约束**：不存在"绕过 critic + 守 BH"这种 cheap path
3. **5-Sprint redesign 投入是 sunk cost 但有学术档案价值**：spec 移到 `docs/decisions/rejected/spec_factor_mad_redesign.md`，方法学（DSL + Stage 流程）可作为同类项目模板
4. **agentic AI 的 alpha 来源不在 macro factor mining**：本 reject 后 cleanup sprint 把 factor_mad 模块整体删除（参 [project_cleanup_2026-05-03.md](../../memory/project_cleanup_2026-05-03.md)）

---

## 6. 删除的代码（2026-05-03 cleanup）

| 路径 | 行数 |
|---|---|
| `engine/factor_mad.py` | ~600 |
| `engine/factor_mad_methodology.py` | ~120 |
| `engine/agents/factor_mad/` (5 文件) | ~1200 |
| `pages/factor_mad_approvals.py` | ~280 |
| `pages/factor_dashboard.py` | ~340 |
| `pages/_factor_mad_components.py` | ~495 |

**Total**: ~3000 LOC removed; database tables `factor_definitions` / `factor_icir` / `factor_failure_memory` 保留作历史档案。

---

## 7. 与其他决策的关系

- [narrative_overlay_phase0_rejected.md](narrative_overlay_phase0_rejected.md) (2026-05-02) — 同期 LLM-as-alpha reject
- [paper_trading_e_v0_2_redesign.md](paper_trading_e_v0_2_redesign.md) (2026-05-03) — 第三个 LLM-as-alpha 方向（sector_pipeline debate），尚在 forward demo 阶段
- 综合：4 个 LLM-as-alpha 方向（D1 risk gate, narrative phase 0, factor_mad, track_b）已 reject；sector debate 唯一仍 active 但定位为 demo 而非 verdict
