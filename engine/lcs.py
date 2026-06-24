"""
LCS (Logical Consistency Score) Suite
=======================================
Behavioral-consistency audit for AI-generated decisions.

WHAT LCS MEASURES
-----------------
LCS tests whether the model's reasoning is *logically responsive* to input
changes.  A model that always outputs "超配" regardless of inputs has a
low LCS — not because it's wrong, but because its logic is degenerate.

WHAT LCS CANNOT MEASURE
------------------------
LCS cannot detect high-quality historical memorisation.  If the model
correctly remembers that 2008 ended in a crash *and* correctly flips its
answer on the mirror prompt, it passes LCS with flying colours — yet the
signal is still contaminated by foreknowledge.

For true foreknowledge isolation we rely on CLEAN_ZONE_START (2025-04-01):
only decisions made on data after that date can be considered free of LLM
training-set contamination.

COMPONENTS
----------
1. Mirror Test      — flip bullish ↔ bearish inputs; expect conclusion to flip
2. Noise Test       — add 5 % Gaussian noise to numeric inputs; expect stable direction
3. Cross-Cycle Anchor — strip key reasoning vectors; check conclusion survives
4. LCS Score        — weighted combination; threshold = 0.70

ARCHITECTURE
------------
Quality Gate (LCS score ≥ 0.70):
    Blocks degenerate logic from writing back to LearningLog / QuantPatternLog.
Statistical Gate (Block Bootstrap Permutation, see notes below):
    Validates that the signal is statistically better than random.
    Full implementation deferred; placeholder hook provided here.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LCS_THRESHOLD: float = 0.70   # quality gate pass/fail boundary

# Component weights in the combined LCS score
_W_MIRROR      = 0.50   # most important: does logic flip with inputs?
_W_NOISE       = 0.30   # stability: is decision robust to small numeric perturbations?
_W_CROSS_CYCLE = 0.20   # depth: does logic survive without clichéd macro scripts?

# ── Statistical Gate: minimum sample threshold ────────────────────────────────
# Derivation (Bonferroni-corrected α ≈ 0.003, one-sided z = 2.75):
#
#   SE(n) = √(0.5 × 0.5 / n)
#   Required accuracy to pass = 0.50 + 2.75 × SE(n)
#
#   n =  50 → need ≥ 69.4% accuracy to pass  ← too strict, real signal invisible
#   n = 100 → need ≥ 63.8% accuracy to pass  ← detects 65%+ signal  ✓
#   n = 200 → need ≥ 59.7% accuracy to pass  ← most robust, takes longer
#
# 100 is the lowest n where a genuine 65%-accuracy signal becomes detectable.
# Below 100, the test produces "not_significant" for real signals — which is
# MORE misleading than showing "insufficient data".
_MIN_N_PERMUTATION: int = 100


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class LCSResult:
    """Structured outcome of a full LCS audit on one AI decision."""

    # Component results
    mirror_passed:      bool
    noise_passed:       bool
    cross_cycle_passed: bool

    # Scores (0.0 – 1.0 per component, weighted combination in lcs_score)
    mirror_score:      float
    noise_score:       float
    cross_cycle_score: float
    lcs_score:         float      # weighted combination

    # Gate outcome
    lcs_passed: bool              # lcs_score >= LCS_THRESHOLD

    # Diagnostics
    mirror_original:  str = ""   # original direction
    mirror_response:  str = ""   # mirror direction
    noise_response:   str = ""   # direction under noisy inputs
    cross_cycle_resp: str = ""   # direction without clichéd vectors
    notes:            str = ""   # human-readable summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_direction(text: str) -> str:
    """Extract a single direction keyword from a free-text AI response."""
    text = text or ""
    for kw in ["超配", "看多", "强烈买入", "做多"]:
        if kw in text:
            return "超配"
    for kw in ["低配", "看空", "减仓", "做空"]:
        if kw in text:
            return "低配"
    for kw in ["标配", "中性", "观望", "持有"]:
        if kw in text:
            return "标配"
    for kw in ["🚨", "拦截", "否决"]:
        if kw in text:
            return "拦截"
    return "中性"


def _directions_are_opposite(d1: str, d2: str) -> bool:
    """True when d1 and d2 are logically opposing directions."""
    pairs = {("超配", "低配"), ("低配", "超配"), ("拦截", "通过"), ("通过", "拦截")}
    return (d1, d2) in pairs


def _flip_direction(direction: str) -> str:
    """Return the logical opposite of a direction label."""
    mapping = {"超配": "低配", "低配": "超配", "拦截": "通过", "通过": "拦截"}
    return mapping.get(direction, "标配")


def _safe_call(model, prompt: str) -> str:
    """Call model.generate_content safely; return empty string on failure."""
    try:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as exc:
        logger.warning("LCS model call failed: %s", exc)
        return ""


# ── Component 1: Mirror Test ──────────────────────────────────────────────────

def run_mirror_test(
    model,
    sector: str,
    original_direction: str,
    vix: float,
    macro_regime: str,
    quant_metrics: dict,
) -> tuple[bool, float, str, str]:
    """
    Construct a semantically reversed scenario and check if the conclusion flips.

    Returns
    -------
    passed : bool   — True when the mirror conclusion is the logical opposite
    score  : float  — 1.0 if perfectly flipped, 0.5 if neutral, 0.0 if same
    mirror_dir : str  — the direction the model gave for the mirrored inputs
    notes  : str    — diagnostic text
    """
    if original_direction in ("中性", "标配", "通过"):
        # Neutral / pass directions are hard to mirror meaningfully — skip
        return True, 0.8, "标配", "标配/通过方向跳过镜像测试（中性场景无明确镜像）"

    expected_opposite = _flip_direction(original_direction)

    # Build mirrored numeric context (all signals reversed)
    m_vix     = round(max(10.0, min(80.0, 50.0 - vix + 20.0)), 1)  # ~inverse
    mom_1m    = quant_metrics.get("mom_1m") or 0.0
    mom_3m    = quant_metrics.get("mom_3m") or 0.0
    d_var     = quant_metrics.get("d_var")   or 0.02
    a_ret     = quant_metrics.get("a_ret")   or 0.0
    a_vol     = quant_metrics.get("a_vol")   or 0.15

    m_mom_1m  = -mom_1m
    m_mom_3m  = -mom_3m
    m_ret     = -a_ret
    m_vol     = a_vol * 1.5   # amplified uncertainty

    # Construct regime mirror: same regime name but described adversarially
    if original_direction == "超配":
        signal_narrative = (
            "所有动量指标已转为负值，市场持续流出资金。"
            "技术面呈现死叉信号，基本面预期持续下调。"
            "板块受到监管压力和供应链双重冲击。"
        )
    else:
        signal_narrative = (
            "动量全面转正，资金持续流入该板块。"
            "技术面出现黄金交叉，基本面预期超预期上调。"
            "板块受益于政策红利与周期底部反弹共振。"
        )

    prompt = (
        f"你是一名量化分析师。请仅基于以下【镜像情景数据】，输出对 {sector} 的配置方向判断。\n\n"
        f"宏观制度：{macro_regime}\n"
        f"VIX 水平：{m_vix:.1f}\n"
        f"年化收益：{m_ret:+.1%}  年化波动：{m_vol:.1%}\n"
        f"动量信号 — 1M: {m_mom_1m:+.1%} | 3M: {m_mom_3m:+.1%}\n"
        f"风险信号：{signal_narrative}\n\n"
        "请仅回答一个关键词：超配 / 标配 / 低配 / 拦截\n"
        "（不要解释原因，只输出方向关键词）"
    )

    mirror_resp  = _safe_call(model, prompt)
    mirror_dir   = _extract_direction(mirror_resp) if mirror_resp else "中性"

    if _directions_are_opposite(original_direction, mirror_dir):
        score = 1.0
        passed = True
        notes = f"镜像测试通过：原始={original_direction} → 镜像={mirror_dir}（逻辑一致性良好）"
    elif mirror_dir in ("标配", "中性"):
        score = 0.5
        passed = True  # moving to neutral on reversed inputs is acceptable
        notes = f"镜像测试部分通过：原始={original_direction} → 镜像={mirror_dir}（保守中性）"
    else:
        score = 0.0
        passed = False
        notes = (
            f"镜像测试失败：原始={original_direction} → 镜像={mirror_dir}（逻辑退化偏见）。"
            "模型可能无论输入如何都倾向给出相同结论，存在系统性偏差。"
        )

    return passed, score, mirror_dir, notes


# ── Component 2: Noise Injection Test ────────────────────────────────────────

def run_noise_test(
    model,
    sector: str,
    original_direction: str,
    vix: float,
    macro_regime: str,
    quant_metrics: dict,
    noise_level: float = 0.05,  # 5% Gaussian noise
    n_trials:    int   = 1,
) -> tuple[bool, float, str, str]:
    """
    Add Gaussian noise to numeric inputs and check if the direction remains stable.

    A robust model should give the same direction on slightly perturbed inputs.
    Flipping direction on 5% noise suggests over-fitting to specific data points.

    Returns
    -------
    passed   : bool   — True when noisy direction matches original
    score    : float
    noisy_dir: str    — direction under noisy inputs
    notes    : str
    """
    def _jitter(val: float) -> float:
        import random as _rnd
        return val * (1.0 + _rnd.gauss(0, noise_level))

    m_vix   = max(10.0, _jitter(vix))
    mom_1m  = quant_metrics.get("mom_1m") or 0.0
    mom_3m  = quant_metrics.get("mom_3m") or 0.0
    d_var   = quant_metrics.get("d_var")  or 0.02
    a_ret   = quant_metrics.get("a_ret")  or 0.0
    a_vol   = quant_metrics.get("a_vol")  or 0.15

    m_mom_1m = _jitter(mom_1m) if abs(mom_1m) > 0.001 else mom_1m
    m_mom_3m = _jitter(mom_3m) if abs(mom_3m) > 0.001 else mom_3m
    m_var    = max(0.005, _jitter(abs(d_var)))
    m_ret    = _jitter(a_ret)
    m_vol    = max(0.01, _jitter(a_vol))

    prompt = (
        f"你是一名量化分析师。请仅基于以下量化指标，输出对 {sector} 的配置方向判断。\n\n"
        f"宏观制度：{macro_regime}\n"
        f"VIX 水平：{m_vix:.2f}\n"
        f"年化收益：{m_ret:+.2%}  年化波动：{m_vol:.2%}\n"
        f"VaR（日度）：{m_var:.3%}\n"
        f"动量信号 — 1M: {m_mom_1m:+.2%} | 3M: {m_mom_3m:+.2%}\n\n"
        "请仅回答一个关键词：超配 / 标配 / 低配 / 拦截\n"
        "（不要解释原因，只输出方向关键词）"
    )

    noisy_resp = _safe_call(model, prompt)
    noisy_dir  = _extract_direction(noisy_resp) if noisy_resp else "中性"

    if noisy_dir == original_direction:
        score  = 1.0
        passed = True
        notes  = f"噪音稳健性测试通过：±{noise_level:.0%} 噪音下方向不变（{original_direction}）"
    elif noisy_dir in ("标配", "中性") or original_direction in ("标配", "中性"):
        score  = 0.5
        passed = True
        notes  = f"噪音稳健性测试部分通过：{original_direction} → {noisy_dir}（中性漂移在可接受范围内）"
    else:
        score  = 0.0
        passed = False
        notes  = (
            f"噪音稳健性测试失败：±{noise_level:.0%} 噪音后方向从 {original_direction} 变为 {noisy_dir}。"
            "分析结论可能过度拟合特定数值，对输入扰动不稳健。"
        )

    return passed, score, noisy_dir, notes


# ── Component 3: Cross-Cycle Anchoring ───────────────────────────────────────

def run_cross_cycle_test(
    model,
    sector: str,
    original_direction: str,
    macro_regime: str,
    conclusion_text: str,
) -> tuple[bool, float, str, str]:
    """
    Strip standard macro-script reasoning and check if the conclusion survives.

    Certain reasoning patterns are "universal scripts" that apply in every cycle
    (e.g., "加息利空成长股", "避险资金流入黄金"). If the model's conclusion
    depends entirely on these scripts rather than sector-specific analysis,
    it is not generating genuine alpha.

    Method: inject a hard constraint forbidding clichéd macro vectors, rerun.

    Returns
    -------
    passed   : bool
    score    : float
    resp_dir : str
    notes    : str
    """
    # Extract key phrase candidates from the original conclusion (first 200 chars)
    snippet = (conclusion_text or "")[:200]

    prompt = (
        f"你是一名量化分析师。请仅基于【板块基本面与量化技术信号】，"
        f"输出对 {sector} 板块的配置方向判断。\n\n"
        f"宏观制度背景：{macro_regime}\n\n"
        "【强制约束】本次分析禁止使用以下通用宏观叙事：\n"
        "- 加息/降息利好/利空的通用描述\n"
        "- 避险资金流向黄金/债券的标准表述\n"
        "- 经济周期顶部/底部的简单类比\n"
        "- 流动性收紧/宽松的一般性推论\n\n"
        f"仅依赖该板块自身的基本面驱动因子和技术量化信号作出判断。\n"
        f"参考背景（原始分析摘要）：{snippet}\n\n"
        "请仅回答一个关键词：超配 / 标配 / 低配 / 拦截\n"
        "（不要解释原因，只输出方向关键词）"
    )

    resp     = _safe_call(model, prompt)
    resp_dir = _extract_direction(resp) if resp else "中性"

    if resp_dir == original_direction:
        score  = 1.0
        passed = True
        notes  = (
            f"跨周期锚定测试通过：移除通用宏观叙事后结论不变（{original_direction}），"
            "表明分析基于板块特异性因子，而非宏观脚本套用。"
        )
    elif resp_dir in ("标配", "中性"):
        score  = 0.5
        passed = True
        notes  = (
            f"跨周期锚定测试部分通过：{original_direction} → {resp_dir}（不确定性增加，非逆转）。"
            "结论在移除宏观叙事后变得保守，但未出现逻辑倒置。"
        )
    else:
        score  = 0.0
        passed = False
        notes  = (
            f"跨周期锚定测试失败：移除通用宏观叙事后结论从 {original_direction} 倒转为 {resp_dir}。"
            "分析结论高度依赖通用宏观脚本，缺乏板块特异性 alpha 驱动，建议降低置信度。"
        )

    return passed, resp_dir, score, notes


# ── Full LCS Audit ────────────────────────────────────────────────────────────

def run_full_lcs_audit(
    model,
    sector: str,
    original_direction: str,
    vix: float,
    macro_regime: str,
    quant_metrics: dict,
    conclusion_text: str = "",
    run_cross_cycle: bool = True,
) -> LCSResult:
    """
    Run the full 3-component LCS audit on a single AI decision.

    Parameters
    ----------
    model             : Gemini model instance (must have .generate_content())
    sector            : sector name, e.g. "AI算力/半导体"
    original_direction: stored decision direction, e.g. "超配"
    vix               : VIX level used in the analysis
    macro_regime      : macro regime label, e.g. "加息周期"
    quant_metrics     : dict with keys: d_var, a_ret, a_vol, mom_1m, mom_3m, mom_6m
    conclusion_text   : first N chars of the AI conclusion (for cross-cycle test)
    run_cross_cycle   : set False to skip cross-cycle test (saves one API call)

    Returns
    -------
    LCSResult with all component scores and the combined gate verdict.
    """
    logger.info(
        "LCS audit: sector=%s direction=%s regime=%s",
        sector, original_direction, macro_regime,
    )

    # ── Component 1: Mirror Test ──────────────────────────────────────────────
    m_passed, m_score, m_dir, m_notes = run_mirror_test(
        model, sector, original_direction, vix, macro_regime, quant_metrics,
    )

    # ── Component 2: Noise Injection ──────────────────────────────────────────
    n_passed, n_score, n_dir, n_notes = run_noise_test(
        model, sector, original_direction, vix, macro_regime, quant_metrics,
    )

    # ── Component 3: Cross-Cycle Anchoring ────────────────────────────────────
    if run_cross_cycle and conclusion_text:
        cc_passed, cc_dir, cc_score, cc_notes = run_cross_cycle_test(
            model, sector, original_direction, macro_regime, conclusion_text,
        )
    else:
        cc_passed, cc_dir, cc_score, cc_notes = True, original_direction, 0.8, "跨周期测试已跳过"

    # ── Weighted LCS score ────────────────────────────────────────────────────
    lcs_score = (
        _W_MIRROR      * m_score  +
        _W_NOISE       * n_score  +
        _W_CROSS_CYCLE * cc_score
    )
    lcs_passed = lcs_score >= LCS_THRESHOLD

    # ── Summary notes ─────────────────────────────────────────────────────────
    parts = []
    if not m_passed:
        parts.append(f"[镜像失败] {m_notes}")
    if not n_passed:
        parts.append(f"[噪音不稳定] {n_notes}")
    if not cc_passed:
        parts.append(f"[跨周期依赖] {cc_notes}")
    if not parts:
        parts.append(
            f"LCS={lcs_score:.2f} — 三项测试均通过，逻辑一致性良好。"
            f"注意：LCS通过不代表无历史记忆污染，仍需参照 CLEAN_ZONE 边界评估。"
        )
    combined_notes = " | ".join(parts)

    result = LCSResult(
        mirror_passed=m_passed,
        noise_passed=n_passed,
        cross_cycle_passed=cc_passed,
        mirror_score=m_score,
        noise_score=n_score,
        cross_cycle_score=cc_score,
        lcs_score=round(lcs_score, 4),
        lcs_passed=lcs_passed,
        mirror_original=original_direction,
        mirror_response=m_dir,
        noise_response=n_dir,
        cross_cycle_resp=cc_dir,
        notes=combined_notes,
    )

    logger.info(
        "LCS result: score=%.2f passed=%s mirror=%s noise=%s cc=%s",
        lcs_score, lcs_passed, m_passed, n_passed, cc_passed,
    )
    return result


# ── Statistical Gate: Block Bootstrap Permutation ────────────────────────────

@dataclass
class PermutationResult:
    """
    Structured result of a block-bootstrap permutation test for one sector × regime cell.

    Three mutually exclusive status values:
      "insufficient_data" — n < _MIN_N_PERMUTATION. p_value is None.
                            Do NOT interpret this as "not significant".
                            The test was not run, not failed.
      "not_significant"   — n >= threshold, p >= adjusted_threshold.
                            Signal is indistinguishable from noise at this sample size.
      "significant"       — n >= threshold, p < adjusted_threshold.
                            Observed accuracy beats the null distribution after
                            multiple-test correction.
    """
    status:            str              # "insufficient_data" | "not_significant" | "significant"
    p_value:           Optional[float]  # None when status == "insufficient_data"
    observed_accuracy: float
    n_samples:         int
    n_needed:          int              # = _MIN_N_PERMUTATION
    adjusted_threshold: float
    sector:            str = ""
    regime:            str = ""

    @property
    def passed(self) -> bool:
        return self.status == "significant"

    @property
    def progress_pct(self) -> float:
        """How far to the minimum sample threshold (0.0 → 1.0)."""
        return min(1.0, self.n_samples / max(1, self.n_needed))


def compute_permutation_p_value(
    accuracy_scores:    list[float],
    n_permutations:     int   = 10_000,
    block_size:         int   = 4,
    adjusted_threshold: float = 0.003,
    sector:             str   = "",
    regime:             str   = "",
) -> PermutationResult:
    """
    Block-bootstrap permutation test for signal significance.

    Null hypothesis: observed accuracy is no better than random shuffling of
    outcomes within contiguous time-blocks (preserves autocorrelation structure).

    This is a STATISTICAL GATE — separate from the LCS quality gate.
      LCS asks  "is the logic responsive?"       (behavioral quality)
      This asks "is accuracy > chance?"           (statistical significance)

    Parameters
    ----------
    accuracy_scores    : 0.0 / 0.5 / 0.75 / 1.0 scores, chronologically ordered
    n_permutations     : permutation draws (≥10_000 for stable p near 0.003)
    block_size         : contiguous block length; 4 = quarterly (recommended)
    adjusted_threshold : per-test p-value threshold after multiple-test correction
                         Use bonferroni_adjusted_threshold() to compute this.
    sector / regime    : labels carried through for display purposes

    Returns
    -------
    PermutationResult with status in {"insufficient_data", "not_significant", "significant"}

    IMPORTANT: "insufficient_data" ≠ "not_significant".
    If n < _MIN_N_PERMUTATION, the test is not run and no p-value is produced.
    Displaying "not significant" when the true state is "not enough data" would
    mislead users into under-valuing a system that simply hasn't accumulated
    enough Clean Zone decisions yet.
    """
    n = len(accuracy_scores)
    observed_mean = sum(accuracy_scores) / n if n > 0 else 0.0

    # ── Sample-size gate ─────────────────────────────────────────────────────
    if n < _MIN_N_PERMUTATION:
        return PermutationResult(
            status             = "insufficient_data",
            p_value            = None,
            observed_accuracy  = observed_mean,
            n_samples          = n,
            n_needed           = _MIN_N_PERMUTATION,
            adjusted_threshold = adjusted_threshold,
            sector             = sector,
            regime             = regime,
        )

    # ── Block-bootstrap permutation ──────────────────────────────────────────
    beats = 0
    for _ in range(n_permutations):
        permuted: list[float] = []
        while len(permuted) < n:
            start = random.randint(0, max(0, n - block_size))
            permuted.extend(accuracy_scores[start : start + block_size])
        perm_mean = sum(permuted[:n]) / n
        if perm_mean >= observed_mean:
            beats += 1

    p_value = beats / n_permutations
    status  = "significant" if p_value < adjusted_threshold else "not_significant"

    logger.debug(
        "Permutation test: sector=%s regime=%s n=%d obs=%.3f p=%.4f status=%s",
        sector, regime, n, observed_mean, p_value, status,
    )
    return PermutationResult(
        status             = status,
        p_value            = p_value,
        observed_accuracy  = observed_mean,
        n_samples          = n,
        n_needed           = _MIN_N_PERMUTATION,
        adjusted_threshold = adjusted_threshold,
        sector             = sector,
        regime             = regime,
    )


def bonferroni_adjusted_threshold(
    base_alpha: float = 0.05,
    n_tests:    int   = 16,
    method:     str   = "romano_wolf",
) -> float:
    """
    Return the adjusted per-test significance threshold for multiple comparisons.

    Romano-Wolf stepdown is recommended for correlated sector tests: it is more
    powerful than Bonferroni when hypotheses share common factor exposure
    (e.g., all sectors react to the same macro regime shifts).

    Parameters
    ----------
    base_alpha : family-wise error rate (default 0.05)
    n_tests    : number of simultaneous tests (default 16 sectors)
    method     : "bonferroni" | "romano_wolf" (approximate first-step)

    Returns
    -------
    threshold : per-test p-value cutoff
    """
    if method == "bonferroni":
        return base_alpha / n_tests
    # Romano-Wolf first-step approximation (0.85 factor vs pure Bonferroni)
    return base_alpha / (n_tests * 0.85)
