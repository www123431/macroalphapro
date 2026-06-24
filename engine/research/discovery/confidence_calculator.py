"""engine/research/discovery/confidence_calculator.py — deterministic
"is this a tradable factor mechanism" confidence score.

REPLACES THE LLM-JUDGED `confidence` FIELD per [[project-senior-pipeline-
roadmap-2026-05-30]] redesign 2026-05-30: production-oriented agent cannot
depend on stochastic LLM self-rating for gating decisions.

DESIGN PRINCIPLE (per [[feedback-no-brittle-hardcoding-2026-05-30]]):
  Deterministic features primary + LLM only for what only LLM can do
  (mechanism prose, family naming, data-token extraction). The
  gating decision (does this paper merit further pipeline cost?) is
  pure regex/keyword on observable text — fully reproducible.

FORMULA:
  confidence = clip(pos_score - neg_score, 0, 1)
    pos_score = Σ w_i × feature_i for positive features
    neg_score = Σ w_j × feature_j for negative features

POSITIVE FEATURES (cumulative weight 1.00):
  return_prediction_claim    0.20   "predicts returns" / "long-short" / "decile"
  sharpe_or_alpha_number     0.15   "Sharpe X.X" / "alpha = Y%"
  tstat_pattern              0.10   "t = X.X" / "t-stat X.X"
  holding_period             0.10   "monthly rebal" / "12-month hold"
  universe_specifier         0.10   "S&P 500" / "CRSP" / "Russell"
  sample_window              0.10   "1990-2020" (≥5 year span)
  required_data_extracted    0.10   LLM extracted ≥1 data token
  family_recognized          0.05   LLM mapped to known family

NEGATIVE FEATURES (cumulative weight 0.85):
  pure_theory                0.30   "we derive" / "general equilibrium"
  survey_or_review           0.20   "literature review" / "we survey"
  no_data_source_mentioned   0.15   No CRSP/Compustat/etc reference
  behavioral_lab             0.20   "lab experiment" / "subjects"

CALIBRATION TARGETS (verified by test_confidence_calculator):
  Known factor papers (Asness/FF/HXZ-style abstracts) → confidence ≥ 0.55
  Pure theory papers (general equilibrium derivations)  → confidence ≤ 0.30
  Survey papers                                          → confidence ≤ 0.30
  Pure NLP / non-finance methodology                      → confidence ≤ 0.20
"""
from __future__ import annotations

import dataclasses
import re


# ── Regex patterns (compiled once) ────────────────────────────────────────

_RETURN_PRED_KW = (
    "predicts return", "predicts returns", "predicts future return",
    "abnormal return", "abnormal returns",
    "long-short", "long short", "long minus short",
    "decile", "quintile", "tercile", "quartile",
    "factor return", "cross-section", "cross sectional",
    "post-earnings", "post earnings",
    "earns alpha", "generates alpha", "generates return",
    "predicting returns", "predicting future returns",
)

_SHARPE_ALPHA_RX = re.compile(
    r"(?:sharpe(?:\s+ratio)?\s*(?:of)?\s*[:=]?\s*\d+\.\d+|"
    r"\balpha\s*(?:of|=)\s*-?\d+(?:\.\d+)?\s*%|"
    r"sharpe\s+ratio\s+(?:of\s+)?\d+|"
    r"information\s+ratio\s+(?:of\s+)?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_TSTAT_RX = re.compile(
    r"(?:\bt\s*[-=]?\s*(?:stat|statistic)?\s*[:=]?\s*\d+\.\d+|"
    r"t-stat(?:istic)?\s+(?:of\s+)?\d+\.\d+|"
    r"t-values?\s+(?:of\s+)?\d+\.\d+)",
    re.IGNORECASE,
)

_HOLDING_KW = (
    "monthly rebal", "quarterly rebal", "annual rebal", "weekly rebal",
    "rebalanc",
    "month hold", "months hold", "month holding",
    "year hold", "years hold", "year holding",
    "12-1", "11-1", "6-1",
    "month-end", "end-of-month",
    "buy-and-hold", "hold for",
)

_UNIVERSE_KW = (
    "s&p 500", "sp500", "sp 500", "s&p500",
    "nyse", "nasdaq", "amex",
    "crsp", "compustat", "russell", "wrds",
    "ibes", "i/b/e/s", "optionmetrics", "option metrics",
    "all u.s. stocks", "all us stocks", "u.s. equity", "us equity",
    "msci", "ftse",
    "futures", "bond futures", "currency futures", "commodity futures",
    "g10 currencies", "g10 fx",
    "developed markets", "emerging markets",
)

_SAMPLE_RX = re.compile(
    r"((?:18|19|20)\d{2})\s*[-–to to]+\s*((?:18|19|20)\d{2})",
    re.IGNORECASE,
)

_DATA_SOURCE_KW = (
    "crsp", "compustat", "wrds", "ibes", "i/b/e/s",
    "optionmetrics", "option metrics",
    "bloomberg", "refinitiv", "datastream", "factset", "msci",
    "tr-ds", "tr_ds_fut", "fred",
    "yfinance", "yahoo finance",
    "sec edgar", "edgar", "13f",
    "ssrn",  # implies finance research even if no specific db
    "thomson reuters", "morningstar",
)

_THEORY_KW = (
    # Strict theory-only markers (avoid generic "asset pricing model"
    # which empirical factor papers also use):
    "we derive", "we prove", "we show theoretically",
    "general equilibrium", "rational expectations equilibrium",
    "axiomatic",
    "this paper presents a theoretical", "purely theoretical",
    "no empirical", "without empirical",
    # Closed-form mathematical economics
    "closed-form solution", "fixed-point theorem",
    "we derive closed-form", "we obtain closed-form",
)

_SURVEY_KW = (
    "we survey", "this paper surveys",
    "literature review", "literature survey",
    "we review", "this paper reviews", "this review",
    "meta-analysis", "meta analysis",
    "we summarize", "we provide an overview",
    "systematic review",
)

_LAB_EXP_KW = (
    "lab experiment", "laboratory experiment",
    "subjects were", "we recruited", "participants completed",
    "experimental subjects",
    "incentivized participants", "we ran an experiment",
    "control group", "treatment group",     # ambiguous but often experimental
    "behavioral experiment",
)


@dataclasses.dataclass
class ConfidenceResult:
    confidence:       float
    pos_score:        float
    neg_score:        float
    positives_hit:    list[str]
    negatives_hit:    list[str]
    feature_weights:  dict[str, float]

    def to_dict(self) -> dict:
        return {
            "confidence":      round(self.confidence, 4),
            "pos_score":       round(self.pos_score, 4),
            "neg_score":       round(self.neg_score, 4),
            "positives_hit":   self.positives_hit,
            "negatives_hit":   self.negatives_hit,
            "feature_weights": self.feature_weights,
        }


# ── Feature detectors (each returns bool) ─────────────────────────────────

def _has_return_prediction(text: str) -> bool:
    return any(kw in text for kw in _RETURN_PRED_KW)


def _has_sharpe_or_alpha(text: str) -> bool:
    return bool(_SHARPE_ALPHA_RX.search(text))


def _has_tstat(text: str) -> bool:
    return bool(_TSTAT_RX.search(text))


def _has_holding_period(text: str) -> bool:
    return any(kw in text for kw in _HOLDING_KW)


def _has_universe(text: str) -> bool:
    return any(kw in text for kw in _UNIVERSE_KW)


def _has_sample_window(text: str) -> bool:
    """At least one (YYYY)-(YYYY) pattern spanning ≥5 years."""
    for m in _SAMPLE_RX.finditer(text):
        try:
            start, end = int(m.group(1)), int(m.group(2))
            if end - start >= 5:
                return True
        except (ValueError, IndexError):
            continue
    return False


def _has_data_source(text: str) -> bool:
    return any(kw in text for kw in _DATA_SOURCE_KW)


def _is_pure_theory(text: str) -> bool:
    return any(kw in text for kw in _THEORY_KW)


def _is_survey(text: str) -> bool:
    return any(kw in text for kw in _SURVEY_KW)


def _is_lab_experiment(text: str) -> bool:
    return any(kw in text for kw in _LAB_EXP_KW)


# ── Public API ────────────────────────────────────────────────────────────

POSITIVE_WEIGHTS = {
    "return_prediction_claim":  0.20,
    "sharpe_or_alpha_number":   0.15,
    "tstat_pattern":            0.10,
    "holding_period":           0.10,
    "universe_specifier":       0.10,
    "sample_window":            0.10,
    "required_data_extracted":  0.10,
    "family_recognized":        0.05,
}

NEGATIVE_WEIGHTS = {
    "pure_theory":              0.30,
    "survey_or_review":         0.20,
    "no_data_source_mentioned": 0.15,
    "behavioral_lab":           0.20,
}


def compute_confidence(
    title: str,
    abstract: str,
    *,
    required_data_tokens: list[str] | None = None,
    family_guess: str | None = None,
) -> ConfidenceResult:
    """Deterministic factor-mechanism-paper confidence score in [0, 1].

    Combines positive (tradability signals) + negative (non-tradable
    signals like theory/survey/lab) features. Result is fully
    reproducible — same inputs always yield same score.

    Args:
      title / abstract: paper text (case-insensitive matching)
      required_data_tokens: optional list from LLM extraction
      family_guess: optional family ID from LLM extraction

    Returns: ConfidenceResult with score + per-feature audit trail.
    """
    text = f"{title or ''}\n\n{abstract or ''}".lower()

    pos_detectors = {
        "return_prediction_claim":  _has_return_prediction(text),
        "sharpe_or_alpha_number":   _has_sharpe_or_alpha(text),
        "tstat_pattern":            _has_tstat(text),
        "holding_period":           _has_holding_period(text),
        "universe_specifier":       _has_universe(text),
        "sample_window":            _has_sample_window(text),
        "required_data_extracted":  bool(required_data_tokens),
        "family_recognized": (family_guess not in
                                  (None, "", "unknown", "Unknown")),
    }
    neg_detectors = {
        "pure_theory":              _is_pure_theory(text),
        "survey_or_review":         _is_survey(text),
        "no_data_source_mentioned": not _has_data_source(text),
        "behavioral_lab":           _is_lab_experiment(text),
    }

    pos_score = sum(POSITIVE_WEIGHTS[k] for k, hit in pos_detectors.items() if hit)
    neg_score = sum(NEGATIVE_WEIGHTS[k] for k, hit in neg_detectors.items() if hit)
    confidence = max(0.0, min(1.0, pos_score - neg_score))

    feature_weights = {
        **{f"+{k}": w for k, w in POSITIVE_WEIGHTS.items()},
        **{f"-{k}": w for k, w in NEGATIVE_WEIGHTS.items()},
    }

    return ConfidenceResult(
        confidence=confidence,
        pos_score=pos_score,
        neg_score=neg_score,
        positives_hit=sorted(k for k, hit in pos_detectors.items() if hit),
        negatives_hit=sorted(k for k, hit in neg_detectors.items() if hit),
        feature_weights=feature_weights,
    )


def explain_confidence(
    title: str, abstract: str, **kwargs,
) -> str:
    """Pretty-printed explanation of which features fired + weights."""
    res = compute_confidence(title, abstract, **kwargs)
    lines = [
        f"Title: {(title or '')[:60]}",
        f"  ─────────────────────────────────────────",
        f"  Positives hit ({len(res.positives_hit)}):",
    ]
    for f in res.positives_hit:
        lines.append(f"    + {f:<26} +{POSITIVE_WEIGHTS[f]:.2f}")
    lines.append(f"  Negatives hit ({len(res.negatives_hit)}):")
    for f in res.negatives_hit:
        lines.append(f"    − {f:<26} −{NEGATIVE_WEIGHTS[f]:.2f}")
    lines.append(f"  ─────────────────────────────────────────")
    lines.append(f"  pos_score: {res.pos_score:.3f}")
    lines.append(f"  neg_score: {res.neg_score:.3f}")
    lines.append(f"  CONFIDENCE: {res.confidence:.3f}")
    return "\n".join(lines)
