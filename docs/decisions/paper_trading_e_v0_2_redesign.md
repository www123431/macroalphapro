# Paper Trading E — Path 1 Redesign (v0.1 → v0.2 → demo pivot)

**Spec**：[../spec_paper_trading_three_arm_e.md](../spec_paper_trading_three_arm_e.md) v0.2
**Memory**：
- `project_paper_trading_e_setup.md` (2026-05-03 早段, v0.1 setup)
- `project_paper_trading_e_power.md` (2026-05-03, power analysis 触发 redesign)
- `project_paper_trading_e_path1_2026-05-03.md` (2026-05-03 晚段, v0.2 锁定)
- `project_cleanup_2026-05-03.md` (2026-05-03 末, demo pivot)

---

## TL;DR

Sector_pipeline LLM debate 是项目里**唯一仍 active 的 LLM-as-alpha 方向**。用 forward-only 三 arm paper trading（A baseline / B production / C placebo）拷问"LLM debate 是否产生 incremental net P&L"。

经历两次设计迭代：
- **v0.1**（早段 setup）：NW HAC Sharpe t-stat + n=24 month verdict
- **v0.2**（晚段 Path 1 redesign）：plain t-test + 经济闸门 + n=36 first / n=48 hardcap + bootstrap CI + 三分区决策
- **2026-05-03 demo pivot**：horizon 缩短为 9-12 月 forward demo（不再是 alpha verdict 实验）

每次迭代都是诚实地修正前一次的统计或 product-fit 缺陷。

---

## 1. v0.1 设计与缺陷

### 1.1 v0.1 概要

- 3 arm 同时 month-end snapshot：A (baseline TSMOM + vol-target, 无 LLM) / B (sector_pipeline LLM debate adjustments, ±20pp/sector) / C (random N(0, σ) placebo on same flip sectors)
- 5 闸门 D1.1-style：ΔSharpe(B-A)≥0.10 / NW HAC t≥1.0 / Sharpe(B-C)≥0.05 / subperiod stability ≥-0.20 / PBO≤50%
- n=24 first hard verdict, n=36+ 高 power verdict
- 60 月累积后 first hard verdict

### 1.2 Power Analysis 揭示的两个缺陷

2026-05-03 中段 power analysis（详见 [project_paper_trading_e_power.md](../../memory/project_paper_trading_e_power.md)）：

**缺陷 A — NW HAC type-I error inflation**：
- NW HAC Sharpe SE 在 h ∈ [12, 36] 区间 type-I 实证 7-9%（target α=5%）
- Lo (2002) eq. 12 的 (1+0.5SR²) 校正在小样本 liberal
- 含义：v0.1 的"5% 显著"实际是 7-9% 显著，inflate-by-statistics

**缺陷 B — n=24 power 不足**：
- 在最 plausible LLM hit-rate p∈[0.55, 0.60]（学术先验：Lopez-Lira & Tang 2023 ChatGPT 头条 hit rate 52-55%）上 24-month power **22-41%**
- < `feedback_spec_power_analysis.md` 设的 30% 闸门
- 含义：实验**判不出**——花 24 月得到 inconclusive

按 first principle "辅助但不赚钱 = 0"：power < 30% 的实验等于花 24 月得不到 verdict。

---

## 2. v0.2 Path 1 Redesign

### 2.1 8 项变更

按 evaluate-before-implement，soundness audit 后定 8 项：

| # | 改动 | Why |
|---|---|---|
| 1 | NW HAC → plain monthly t-test on (B-C) | type-I 修正 7-9% → 5% |
| 2 | 经济闸门：ann_diff ≥ 1.5% AND ΔSharpe ≥ 0.15（必须同时过） | Novy-Marx & Velikov (2016) TC-aware floor + half López de Prado 制度标准；防"统计显著但赚不到钱" |
| 3 | 三分区 verdict (insufficient_n / reject / accept / inconclusive) | pre-registration enforce + 防 HARKing |
| 4 | n=36 first / n=48 force-resolve hardcap | Lakatos research programme 硬上限；inconclusive 默认 → reject (default-skeptic) |
| 5 | 持久化 per-sector placebo_adj 到 PaperTradingRun.placebo_adjustments | 准备未来 cluster-by-month T2 panel test (no RNG replay risk) |
| 6 | Stationary block bootstrap CI on Sharpe diff | t-test 假设 sanity check + robustness |
| 7 | 中期 conviction-stratified checkpoints (n=12, n=24) | pre-registered Lever 1 trigger（top-15 event 扩展），防 mid-experiment HARKing |
| 8 | Spec doc bumped v0.2 + multiple testing 声明（joint test 不 inflate α）+ 学术诚实段 | 学术 rigor |

### 2.2 v0.2 Power 表（h=36 first verdict）

| LLM hit-rate (p_flip / p_trend) | n=24 | **n=36** | n=48 |
|---|---|---|---|
| 0.55 / 0.55 | 14% | 18% | 20% |
| 0.60 / 0.55 | 32% | **41%** | 46% |
| 0.65 / 0.55 | 55% | **68%** | **79%** |

**非对称解读**：
- Reject 那侧（H0 真）：α=5% 校正后 sanity ~95% 可靠 — 项目"否决 dead infrastructure"高把握
- Accept 那侧：在 p=0.60 plausible 区域 h=36 power 41% / h=48 power 46% — 仅在 LLM 真 ≥ 0.65 时可信 accept

---

## 3. Demo Pivot（2026-05-03 末段）

### 3.1 触发

用户 reframe：项目 horizon 从 4 年学术研究 → 9-12 月毕业 + 求职 demo。v0.2 的 36/48 月 verdict 与新 horizon **不兼容**。

### 3.2 Pivot 内容

- v0.2 的 verdict zone 决策规则**保留在代码** (`compute_test_statistics`) 但 dashboard 不显示 hardcap 警告
- horizon 缩短为 9-12 月 NAV curve + agent decision audit trail
- 不声称 "LLM has incremental alpha"；声称 "live agentic system with placebo control trail"
- 工程框架（3 arm + per-sector placebo persistence + cluster bootstrap CI）全部保留作 portfolio piece evidence

### 3.3 学术 honest framing

> "9-12 月 forward paper trading demonstrating an agentic decision pipeline with built-in placebo control. Sample size below the v0.2 spec's pre-registered verdict horizon (n≥36); results presented as an audit trail of system behaviour, not an alpha verdict."

---

## 4. 学术 + 工程教训

1. **NW HAC Sharpe stat 在 small-n 下 liberal**：任何 master's-project 量级 (n < 60 月) 的 ablation，应用 plain t-test 或 cluster-bootstrap，而不是 NW HAC（Lo 2002 校正在大 n 才稳）
2. **detection-power-vs-horizon 是 LLM ablation 的硬约束**：plausible LLM hit-rate (0.55-0.60) 下 80% power 需 n≥48-60 month，超出 master's project horizon
3. **Pre-registration enforcement 是 agentic alpha test 的核心 capability**：v0.2 的 spec_hash + amendment ledger + 三分区 + force-resolve 是 generic infrastructure，下个 LLM-alpha 项目可直接复用
4. **Power analysis must precede spec lock**：v0.1 是反例（fixed thresholds 没做 power 分析），v0.2 是正例
5. **Placebo control (Arm C) 是 agentic AI alpha verification 的 sine qua non**：无 placebo 的 forward paper trading 无法分离"alpha"与"baseline market drift"

---

## 5. 与其他决策的关系

- [narrative_overlay_phase0_rejected.md](narrative_overlay_phase0_rejected.md): 同 sprint LLM-as-alpha reject（同样 Arm C placebo 设计 inspired 本 spec）
- [factor_mad_reject.md](factor_mad_reject.md): 同 sprint LLM-as-alpha reject
- [project_cleanup_2026-05-03.md](../../memory/project_cleanup_2026-05-03.md): 整体 cleanup + demo pivot

**唯一仍 active 的 LLM 方向**。前 3 个 LLM 方向（D1 risk gate, narrative phase 0, factor_mad）已 reject；sector debate 在 demo 中持续累积证据。

---

## 6. 工程产出（保留）

| 文件 | 价值 |
|---|---|
| `engine/paper_trading.py` | 3-arm runner + compute_test_statistics + bootstrap CI + mid_checkpoint_conviction，支持 v0.2 完整 verdict 流程 |
| `engine/memory.py:PaperTradingRun` | per-arm + per-sector placebo persistence schema |
| `pages/paper_trading.py` | 三分区 verdict UI + bootstrap CI + 经济闸门表 + agent decision drilldown |
| `docs/spec_paper_trading_three_arm_e.md` v0.2 | pre-registered spec, 锁定后冻结 |

整套 infrastructure 是 generic "agentic AI alpha verification harness"，超越本项目 scope。
