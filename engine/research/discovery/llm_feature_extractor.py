"""engine/research/discovery/llm_feature_extractor.py — LLM-structured
BOOLEAN feature extraction for confidence calculator.

Senior B per [[project-senior-pipeline-roadmap-2026-05-30]]: replace
brittle regex matching for some confidence features with LLM-extracted
booleans. The LLM is asked structured yes/no questions like:
  - Did this paper estimate a Sharpe ratio?
  - Does it specify a sample window of 5+ years?
  - Does it use a long-short portfolio construction?
  - Does it specify a tradable universe (e.g. CRSP / futures / FX)?

This complements (does NOT replace) the regex calculator:
  - Regex still primary (fast, free, deterministic)
  - LLM boolean extraction is OPT-IN second pass for borderline papers
    (e.g. when regex conf is 0.10-0.30 with known family)
  - Combined: regex conf merged with LLM-bool features → final conf

Why this works:
  - LLM doing structured Y/N is HIGH accuracy (its strength)
  - LLM doing subjective "rate this 0.0-1.0" is LOW accuracy (its weakness)
  - We replaced subjective rating with structured extraction
  - Robust to phrasing variation: "Sharpe ratio of 1.5" vs "outperformed
    by 1.5 on a risk-adjusted basis" both trigger Sharpe-extracted=True

NOT auto-called everywhere — opt-in by caller (e.g. nominate UI when
abstract is non-empty but regex conf is low).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

LLM_MODEL = "claude-haiku-4-5-20251001"   # cheap for structured Y/N
MAX_TOKENS = 256


SYSTEM_PROMPT = """You are a strict feature-extraction assistant for a
production quant research pipeline. Given a paper's title and abstract,
return a JSON object with the following BOOLEAN fields. Each must be
true ONLY IF the abstract clearly indicates the feature; default to
false if uncertain.

Fields:
  estimates_sharpe_or_alpha:  Does the paper report a Sharpe ratio,
    information ratio, alpha estimate, or risk-adjusted performance
    number? (true even if mechanism is described qualitatively,
    e.g. "outperforms benchmark on risk-adjusted basis")
  reports_tstatistic:  Does the paper report any t-statistic, F-stat,
    or significance level for a return/factor claim?
  specifies_long_short:  Does the construction involve a long-short
    portfolio, decile/quintile sort, or zero-cost portfolio?
  specifies_holding_period:  Does the paper specify a holding period,
    rebalance frequency, or formation window? (monthly, 12-1, 6-month,
    daily, etc.)
  specifies_universe:  Does the paper specify a tradable universe
    (CRSP/NYSE/Russell/futures/FX/options/specific country/etc.)?
  specifies_sample_window:  Does the paper specify a sample period
    spanning at least 5 years?
  proposes_tradable_mechanism:  Is the paper proposing or studying
    a tradable cross-sectional/time-series factor mechanism (as
    opposed to a pure theory paper, survey, methodology paper, or
    macro economic narrative)?

Return STRICTLY valid JSON with these 7 boolean fields. No prose."""


@dataclasses.dataclass
class LLMFeatureExtraction:
    """LLM-extracted boolean features. All default False if extraction failed."""
    estimates_sharpe_or_alpha:   bool = False
    reports_tstatistic:           bool = False
    specifies_long_short:         bool = False
    specifies_holding_period:    bool = False
    specifies_universe:          bool = False
    specifies_sample_window:     bool = False
    proposes_tradable_mechanism: bool = False
    extraction_ok:                bool = False
    cost_usd:                    float = 0.0
    raw_response:                 str = ""

    def to_dict(self) -> dict:
        return {
            "estimates_sharpe_or_alpha":   self.estimates_sharpe_or_alpha,
            "reports_tstatistic":          self.reports_tstatistic,
            "specifies_long_short":        self.specifies_long_short,
            "specifies_holding_period":    self.specifies_holding_period,
            "specifies_universe":          self.specifies_universe,
            "specifies_sample_window":     self.specifies_sample_window,
            "proposes_tradable_mechanism": self.proposes_tradable_mechanism,
            "extraction_ok":               self.extraction_ok,
            "cost_usd":                    round(self.cost_usd, 6),
        }

    def feature_count(self) -> int:
        """How many of the 7 feature booleans are True."""
        return sum([
            self.estimates_sharpe_or_alpha,
            self.reports_tstatistic,
            self.specifies_long_short,
            self.specifies_holding_period,
            self.specifies_universe,
            self.specifies_sample_window,
            self.proposes_tradable_mechanism,
        ])


def _read_anthropic_key() -> str | None:
    """Mirror of paper_extractor._read_anthropic_key — env + TOML parse."""
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        secrets_path = Path(".streamlit/secrets.toml")
        if not secrets_path.exists():
            return None
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                import re
                text = secrets_path.read_text(encoding="utf-8")
                m = re.search(
                    r'^ANTHROPIC_API_KEY\s*=\s*["\']([^"\']+)["\']',
                    text, re.MULTILINE,
                )
                return m.group(1) if m else None
        with secrets_path.open("rb") as f:
            data = tomllib.load(f)
        return data.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def extract_boolean_features(
    title: str, abstract: str,
) -> LLMFeatureExtraction:
    """Ask LLM 7 structured Y/N questions. Returns LLMFeatureExtraction.

    Falls back gracefully to all-False if LLM unavailable / extraction
    fails. Caller can check extraction_ok.
    """
    if not title and not abstract:
        return LLMFeatureExtraction()

    key = _read_anthropic_key()
    if not key:
        return LLMFeatureExtraction()

    try:
        from anthropic import Anthropic
    except ImportError:
        return LLMFeatureExtraction()

    try:
        client = Anthropic(api_key=key, timeout=30.0)
        user_msg = (
            f"Title: {title}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Extract the 7 boolean features per system prompt. "
            f"Return strict JSON only."
        )
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        usage = response.usage
        # Haiku-4.5 pricing approx (input $0.80/M, output $4/M)
        cost = (usage.input_tokens * 0.80 / 1_000_000
                  + usage.output_tokens * 4.00 / 1_000_000)
        text = "\n".join(b.text for b in response.content if b.type == "text")
        parsed = _parse_json(text)
        if not parsed:
            return LLMFeatureExtraction(extraction_ok=False, cost_usd=cost,
                                            raw_response=text[:200])

        return LLMFeatureExtraction(
            estimates_sharpe_or_alpha=bool(parsed.get("estimates_sharpe_or_alpha", False)),
            reports_tstatistic=bool(parsed.get("reports_tstatistic", False)),
            specifies_long_short=bool(parsed.get("specifies_long_short", False)),
            specifies_holding_period=bool(parsed.get("specifies_holding_period", False)),
            specifies_universe=bool(parsed.get("specifies_universe", False)),
            specifies_sample_window=bool(parsed.get("specifies_sample_window", False)),
            proposes_tradable_mechanism=bool(parsed.get("proposes_tradable_mechanism", False)),
            extraction_ok=True,
            cost_usd=cost,
            raw_response="",     # don't store raw on success
        )
    except Exception as exc:
        logger.warning("LLM feature extraction failed: %s", exc)
        return LLMFeatureExtraction()


def _parse_json(text: str) -> dict | None:
    """Find first balanced {...} in LLM response and parse."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ── Hybrid confidence combining regex + LLM-bool ─────────────────────────

# Mapping: which LLM feature corresponds to which positive feature in
# the deterministic calculator. When the LLM bool fires but the regex
# missed it, we credit the regex weight too.
LLM_TO_REGEX_FEATURE = {
    "estimates_sharpe_or_alpha":   "sharpe_or_alpha_number",
    "reports_tstatistic":          "tstat_pattern",
    "specifies_long_short":        "return_prediction_claim",
    "specifies_holding_period":    "holding_period",
    "specifies_universe":          "universe_specifier",
    "specifies_sample_window":     "sample_window",
}


def compute_hybrid_confidence(
    title: str, abstract: str,
    *,
    required_data_tokens: list[str] | None = None,
    family_guess: str | None = None,
    llm_features: LLMFeatureExtraction | None = None,
    enable_llm: bool = False,
) -> dict:
    """Combine deterministic regex confidence with LLM bool features.

    If enable_llm=True and llm_features=None → call extract_boolean_features.
    LLM bool TRUE but regex feature MISSED → credit the regex weight.
    Returns dict with: base_confidence (regex only), hybrid_confidence,
    llm_features_dict, deltas (which features were rescued by LLM).
    """
    from engine.research.discovery.confidence_calculator import (
        POSITIVE_WEIGHTS, compute_confidence,
    )

    base = compute_confidence(
        title, abstract,
        required_data_tokens=required_data_tokens,
        family_guess=family_guess,
    )

    if enable_llm and llm_features is None:
        llm_features = extract_boolean_features(title, abstract)

    if llm_features is None or not llm_features.extraction_ok:
        return {
            "base_confidence":     base.confidence,
            "hybrid_confidence":   base.confidence,
            "rescued_features":     [],
            "llm_extraction_ok":   False,
            "llm_cost_usd":         0.0,
        }

    # For each LLM bool that's True, if regex missed it, credit the weight
    rescued = []
    bonus_score = 0.0
    base_pos_hit = set(base.positives_hit)
    for llm_feat, regex_feat in LLM_TO_REGEX_FEATURE.items():
        if getattr(llm_features, llm_feat, False) and regex_feat not in base_pos_hit:
            bonus_score += POSITIVE_WEIGHTS.get(regex_feat, 0.0)
            rescued.append({
                "llm_feature":   llm_feat,
                "regex_feature": regex_feat,
                "weight":        POSITIVE_WEIGHTS.get(regex_feat, 0.0),
            })

    hybrid = max(0.0, min(1.0, base.confidence + bonus_score))
    return {
        "base_confidence":     base.confidence,
        "hybrid_confidence":   hybrid,
        "rescued_features":    rescued,
        "llm_features":        llm_features.to_dict(),
        "llm_extraction_ok":   llm_features.extraction_ok,
        "llm_cost_usd":         llm_features.cost_usd,
    }
