"""engine.research_store.red_lessons.failure_modes — F1..F9 controlled vocabulary.

The failure-mode taxonomy is the load-bearing classification axis for the
RED Lesson system. New failure modes require:

  1. An academic anchor (paper or methodological reference)
  2. A diagnostic recipe (how do we DECIDE a RED falls under this mode?)
  3. A forward-direction implication (what does this failure mode suggest
     about what to try next?)

Don't widen by reflex. 9 modes intentionally covers ~95% of factor failures
in the post-1990 academic literature. If a 10th seems needed, first try
to express it as a combination of existing modes.

Each lesson can carry MULTIPLE failure modes (e.g. F3 + F4 = subsumed AND
cost-fragile). Order of modes in the lesson list = ranked by importance.

References:
  - F1: McLean & Pontiff 2016, "Does Academic Research Destroy Stock Return
        Predictability?" JF 71(1).
  - F1: Linnainmaa & Roberts 2018, "The History of the Cross-Section of
        Stock Returns." RFS 31(7).
  - F2: Asness-Moskowitz-Pedersen 2013, "Value and Momentum Everywhere." JF.
  - F3: Hou-Xue-Zhang 2020, "Replicating Anomalies." RFS 33(5).
  - F3: Fama & French 2018, "Choosing factors." JFE 128(2).
  - F4: Frazzini-Israel-Moskowitz 2018, "Trading Costs." WP.
  - F4: Korajczyk-Sadka 2004, "Are Momentum Profits Robust to Trading Costs?" JF.
  - F5: Asness-Moskowitz-Pedersen 2013 (regime-conditional correlation).
  - F6: Shumway 1997, "The Delisting Bias in CRSP Data." JF.
  - F7: Bailey-Lopez de Prado 2014, "The Deflated Sharpe Ratio." JPM.
  - F8: Bailey-Lopez de Prado 2014; Harvey-Liu-Zhu 2016.
  - F9: Harvey-Liu-Zhu 2016, "...and the Cross-Section of Expected Returns." JF.
"""
from __future__ import annotations

from enum import Enum


class FailureMode(str, Enum):
    """Controlled failure-mode taxonomy. F1..F9; do not invent new codes."""

    F1_PUBLICATION_DECAY        = "F1_PUBLICATION_DECAY"
    F2_MECHANISM_MISMATCH       = "F2_MECHANISM_MISMATCH"
    F3_SUBSUMED_BY_EXISTING     = "F3_SUBSUMED_BY_EXISTING"
    F4_IMPLEMENTATION_COST      = "F4_IMPLEMENTATION_COST"
    F5_REGIME_DEPENDENT         = "F5_REGIME_DEPENDENT"
    F6_DATA_QUALITY             = "F6_DATA_QUALITY"
    F7_POWER_INSUFFICIENT       = "F7_POWER_INSUFFICIENT"
    F8_OVERFIT_INDUCED          = "F8_OVERFIT_INDUCED"
    F9_RESIDUAL_NULL            = "F9_RESIDUAL_NULL"


FAILURE_MODE_DOCS: dict[FailureMode, dict[str, str]] = {
    FailureMode.F1_PUBLICATION_DECAY: {
        "label": "Publication decay",
        "definition":
            "Factor was real ex-ante (pre-publication sample) but the post-"
            "publication out-of-sample period shows 30-100% alpha decay. "
            "The mechanism is real but already arbitraged.",
        "diagnostic":
            "Compare pre-publication-date Sharpe / alpha-t with post-publication. "
            "A drop of > 50% in alpha point estimate (or alpha-t falling below "
            "HLZ bar in post-pub sample) is strong evidence. Reference: "
            "McLean-Pontiff 2016 reports average 26-58% decay; LR 2018 reports "
            "~70% post-2000.",
        "forward_implication":
            "Don't re-test the SAME mechanism in the SAME data window. Worth "
            "revisiting only if (a) new pre-publication-window data becomes "
            "available, (b) the mechanism is testable in a market where it was "
            "NOT published (e.g. emerging markets, non-equity asset classes), "
            "or (c) a structural break creates a new regime where the published "
            "arbitrage strategy stops working.",
    },
    FailureMode.F2_MECHANISM_MISMATCH: {
        "label": "Mechanism mismatch",
        "definition":
            "Signal is theoretically valid but the market structure / participant "
            "mix / instrument doesn't support the trading mechanism the original "
            "paper assumed. Example: PEAD in US (institutional) vs PEAD in China "
            "(retail-dominated, reverses on overreaction).",
        "diagnostic":
            "Look for a 'works in market X, fails in market Y, same signal' "
            "pattern. The economic mechanism in the original paper must be "
            "INCOMPATIBLE with the actual microstructure / participant base "
            "of the test market.",
        "forward_implication":
            "Either find a market that DOES match the mechanism, or reframe the "
            "candidate as a DIFFERENT mechanism appropriate to the test market "
            "(e.g. reversal not drift in China-PEAD). Don't try to patch the "
            "original signal in the mismatched market — root cause is structural.",
    },
    FailureMode.F3_SUBSUMED_BY_EXISTING: {
        "label": "Subsumed by existing factor",
        "definition":
            "Signal is real and stand-alone significant, but residual alpha "
            "vs. an existing deployed / well-known factor is statistically zero. "
            "Adds no marginal information beyond what we already have.",
        "diagnostic":
            "Run Fama-MacBeth or spanning regression of candidate on the suspected "
            "existing factor + FF5. Residual alpha-t < 2 AND correlation > 0.5 "
            "with existing factor = subsumed. HXZ 2020 catalog this as the "
            "single biggest 'anomaly' kill mode.",
        "forward_implication":
            "Either find a sub-universe where the candidate is NOT subsumed "
            "(time-varying subsumption is real), or accept that this is a "
            "redundant signal and drop it. Don't try to engineer-around "
            "subsumption with feature engineering — that's overfitting.",
    },
    FailureMode.F4_IMPLEMENTATION_COST: {
        "label": "Implementation cost",
        "definition":
            "Gross alpha is real and significant, but net of realistic transaction "
            "cost + market impact, the strategy delivers Sharpe < 0.5 or negative "
            "net alpha. The signal exists but cannot be harvested in scale.",
        "diagnostic":
            "Per-asset-class round-trip cost basis: US large-cap equity 3-8 bps, "
            "small-cap 8-20, futures 1-3, FX 2-5. Run TC ablation across realistic "
            "range; if Sharpe falls below 0.5 at the relevant TC level, this is F4. "
            "Reference: Frazzini-Israel-Moskowitz 2018 for the canonical cost "
            "calibration methodology.",
        "forward_implication":
            "Reduce signal frequency (less rebalance → less turnover → lower cost), "
            "or move the signal to a lower-cost asset class (e.g. futures rather "
            "than single-name equities), or aggregate across a basket to amortize "
            "fixed costs. Don't try to lever up to compensate — leverage doesn't "
            "scale alpha but does scale costs linearly.",
    },
    FailureMode.F5_REGIME_DEPENDENT: {
        "label": "Regime-dependent failure",
        "definition":
            "Signal works in normal markets but fails (often catastrophically) "
            "in crisis / high-vol regimes. The classical 'correlations go to 1 "
            "in a crash' failure mode. Diversification claim breaks exactly "
            "when needed.",
        "diagnostic":
            "Run regime decomposition (calendar-anchored crisis windows: 2008 GFC, "
            "Q4 2018, Q1 2020 COVID, 2022 hikes). If Sharpe in any of these is "
            "negative AND magnitude exceeds full-sample Sharpe, F5 is present.",
        "forward_implication":
            "If the signal is ONLY usable in normal regime, deploy it ONLY under "
            "an explicit regime classifier with hard cutoff (vol-target / "
            "drawdown-target / VIX-conditioning). Or pair it with an explicit "
            "crisis hedge sleeve that dominates in the failing regime. Don't try "
            "to find 'a smarter regime classifier' — the signal-failure is "
            "mechanism-level, not measurement-level.",
    },
    FailureMode.F6_DATA_QUALITY: {
        "label": "Data quality artifact",
        "definition":
            "Signal exists in the data as recorded but the data itself is "
            "compromised: survivorship bias, look-ahead, accounting standard "
            "mismatch, delisting bias, point-in-time violations.",
        "diagnostic":
            "Compare against a known-clean PIT panel (e.g. CRSP delisting-adjusted, "
            "IBES with original-date stamps, Compustat with point-in-time vintages). "
            "If alpha disappears on clean data, F6. Shumway 1997 documents delisting "
            "bias inflates anomaly returns by ~50 bps/month.",
        "forward_implication":
            "Acquire / build the clean PIT version of the data. Don't try to "
            "'control for' survivorship bias in the regression — the bias is "
            "selection at the panel level, not an additive nuisance variable. "
            "If clean data is unavailable, the signal is unverifiable; shelf it.",
    },
    FailureMode.F7_POWER_INSUFFICIENT: {
        "label": "Insufficient statistical power",
        "definition":
            "Sample is too small (few events, short window) to distinguish the "
            "signal from zero at HLZ bar even if the signal is real. The verdict "
            "is null-result, not RED-because-failed; cannot reject H0.",
        "diagnostic":
            "Compute Bailey-LdP power: at observed effect size, what sample size "
            "would be needed for 80% power at HLZ |t|>=3? If required N >> our N, "
            "F7. Often co-occurs with F1 (publication-decay leaves too little "
            "post-pub sample).",
        "forward_implication":
            "Wait for more data (if signal is forward-tradeable) or expand sample "
            "via cross-market / cross-asset extension. Do NOT lower the HLZ bar "
            "to 'accept' the signal — that's classic p-hacking. If genuinely no "
            "more data will accrue (e.g. one-time historical event), the signal "
            "is unverifiable; shelf with F7 tag.",
    },
    FailureMode.F8_OVERFIT_INDUCED: {
        "label": "Overfit-induced false-positive",
        "definition":
            "Naive in-sample Sharpe / t-stat looks promising, but after deflated "
            "Sharpe / Bailey-LdP multiple-testing correction, signal is "
            "statistically zero. The original 'positive' result was a multiple-"
            "testing artifact.",
        "diagnostic":
            "Compute deflated SR with realistic n_trials (per Bailey-LdP §3, "
            "use family-aware n_trials — count related signals tried in the "
            "same mechanism family). If DSR < 0.9 (per our codebase doctrine), "
            "F8. Often co-occurs with F7 in small samples.",
        "forward_implication":
            "If many similar signals have been tried in the same family, the "
            "true effective n_trials is much larger than the obvious one. Either "
            "pre-register a SINGLE hypothesis on FRESH data, or accept that "
            "the family is exhausted. Adding more features will NOT help.",
    },
    FailureMode.F9_RESIDUAL_NULL: {
        "label": "Residual alpha null vs FF5+UMD",
        "definition":
            "Signal IS stand-alone significant and NOT subsumed by a specific "
            "factor, BUT after controlling for the full FF5+UMD model, residual "
            "alpha-t falls below the HLZ |t|>=3 bar. The signal is 'real' but "
            "fully explained by the canonical factor zoo.",
        "diagnostic":
            "Run FF5+UMD regression with Newey-West HAC SE (6 lags weekly / "
            "12 monthly). If alpha-t_NW < 3 despite stand-alone significance, "
            "F9. This is the strictest residual-significance bar.",
        "forward_implication":
            "Unless we can identify a missing factor (i.e. our factor zoo is "
            "incomplete), this signal isn't an alpha source — it's a beta "
            "exposure to known factors. May still be useful as a risk-management "
            "/ portfolio construction input, but NOT as a stand-alone alpha "
            "sleeve.",
    },
}


# Sanity check at import time: every FailureMode has a doc entry.
assert set(FAILURE_MODE_DOCS.keys()) == set(FailureMode), (
    f"FAILURE_MODE_DOCS missing entries for: "
    f"{set(FailureMode) - set(FAILURE_MODE_DOCS.keys())}"
)
