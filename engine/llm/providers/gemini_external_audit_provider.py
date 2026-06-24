"""engine.llm.providers.gemini_external_audit_provider — Gemini 2.5
adversarial-audit provider for the external_audit pipeline.

Why Gemini (over DeepSeek) for audit (2026-06-14)
==================================================
Three-vendor independence maximizes blind-spot coverage:
  Verdict generator  : Sonnet (Anthropic)        — spec extractor + router
  Paper substrate    : DeepSeek v4-pro            — filter + summarize
  Adversarial audit  : Gemini 2.5 (Google)        — this module

Each stage uses a different model family with different training corpus,
RLHF objective, and architectural inductive bias. A bug or hallucination
pattern that all three families share would have to be deeper than any
single-family weakness — much rarer than single-family blind spots.

DeepSeek-flash was the original choice (cheap), then DeepSeek-pro
(deeper). Gemini gives us cross-vendor diversity at flash-tier cost.

Gemini 2.5 Pro specifics (default tier for audit)
==================================================
- Extended thinking by default; deeper multi-step methodology critique
- Strong literature recall: cites finance papers correctly in audit
- Structured JSON output supported but not needed for free-text severity

Cost tiers (Gemini 2.5, 2026):
- Pro   (default)  : ~$1.25 input / $10 output per M tokens
                      → ~$0.005-0.020 per audit incl. thinking_tokens
                      → ~$0.05-0.18/wk at 9 verdicts/wk
- Flash (cheap)    : ~$0.30 input / $2.50 output per M tokens
                      → ~$0.001-0.002 per audit
                      → ~$0.01-0.02/wk
                      Flash is for high-volume classification, NOT
                      adversarial audit — too shallow for depth catch.

Falls back to DeepSeek-pro if Gemini unreachable (network / quota).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Gemini 2.5 pricing per M tokens (2026)
_GEMINI_PRICING = {
    "gemini-2.5-flash":  {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro":    {"input": 1.25, "output": 10.00},
}
_GEMINI_DEFAULT_PRICING = {"input": 0.30, "output": 2.50}


_SEVERITY_KEYWORDS = {
    "critical": ["critical", "wrong", "invalid", "false positive",
                  "should not trust", "fundamental flaw", "definitely incorrect"],
    "concern":  ["concern", "caveat", "weak", "should surface",
                  "questionable", "suspect", "warrants review"],
    "noted":    ["minor", "noted", "small issue", "edge case"],
    "no_issue": ["no issue", "sound", "appropriate", "looks fine",
                  "no concerns", "methodology appears correct"],
}


def _parse_severity_from_response(text: str) -> str:
    """Extract one severity tag from response text via keyword precedence.
    Defaults to 'concern' if model declined to be definite."""
    if not text:
        return "skipped"
    t = text.lower()
    if "severity: critical" in t or "severity: 'critical'" in t:
        return "critical"
    if "severity: concern" in t or "severity: 'concern'" in t:
        return "concern"
    if "severity: noted" in t or "severity: 'noted'" in t:
        return "noted"
    if "severity: no_issue" in t or "severity: 'no_issue'" in t:
        return "no_issue"
    # Fallback: keyword voting
    for sev in ("critical", "concern", "noted", "no_issue"):
        kws = _SEVERITY_KEYWORDS[sev]
        if any(kw in t for kw in kws):
            return sev
    return "concern"


def _parse_flagged_categories(text: str) -> list[str]:
    """Extract list of category tags from 'CATEGORIES: [...]' line."""
    import re
    if not text:
        return []
    m = re.search(r'CATEGORIES?:\s*\[?\s*([^\]\n]+)\]?', text, re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1)
    out: list[str] = []
    for chunk in raw.split(","):
        tag = chunk.strip().strip('"\'').lower()
        if tag and len(tag) < 40:
            out.append(tag)
    return out


class GeminiExternalAuditProvider:
    """Implements ExternalAuditProvider Protocol via Gemini 2.5."""
    name = "gemini"

    def __init__(self, *, model: str = "gemini-2.5-pro"):
        # 2026-06-14: default is pro (not flash). For adversarial audit
        # the reasoning depth matters more than $0.005 per call savings.
        # 2.5-pro pricing: ~$1.25 input / $10 output per M tokens, with
        # built-in extended thinking. At our verdict volume (Bailey-LdP
        # cap ~9/wk) → ~$0.05-0.15/wk, similar to DeepSeek pro tier.
        # Flash is the budget tier — appropriate for filter/classification
        # tasks but too shallow for catching deep methodology flaws.
        self._model = model

    def adversarial_audit(
        self,
        *,
        subject_payload: dict,
        prompt:          str,
    ) -> tuple[str, str, list[str], float]:
        # Direct google-genai client (no key_pool — key_pool imports
        # streamlit, breaks in headless cron context). API key from
        # GEMINI_KEY env or secrets.toml.
        import os
        api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            # Last-resort: try secrets.toml directly (cron passes through env)
            try:
                from pathlib import Path
                try:
                    import tomllib as tom
                    with (Path(__file__).resolve().parents[3] /
                            ".streamlit" / "secrets.toml").open("rb") as fh:
                        api_key = tom.load(fh).get("GEMINI_KEY", "")
                except ModuleNotFoundError:
                    import tomli as tom
                    with (Path(__file__).resolve().parents[3] /
                            ".streamlit" / "secrets.toml").open("rb") as fh:
                        api_key = tom.load(fh).get("GEMINI_KEY", "")
            except Exception:
                pass
        if not api_key:
            return ("gemini init_failed: no GEMINI_KEY in env or secrets",
                     "skipped", [], 0.0)

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
        except Exception as exc:
            logger.exception("gemini client init failed")
            return (f"gemini init_failed: {type(exc).__name__}: {exc}",
                     "skipped", [], 0.0)

        # Add system instruction prefix (Gemini doesn't have separate
        # system message — prepend to user prompt)
        full_prompt = (
            "You are an independent quantitative research reviewer. The "
            "user will provide a factor research verdict produced by another "
            "LLM-driven system. Identify methodological failure modes the "
            "originating system likely missed. Cite statistical literature "
            "where applicable (López de Prado, Bailey, Harvey-Liu-Zhu, "
            "Fama-French). Output structured:\n"
            "SEVERITY: <one of critical/concern/noted/no_issue>\n"
            "CATEGORIES: <comma-separated short tags>\n"
            "EXPLANATION: <2-4 sentences, concrete>\n\n"
            + prompt
        )

        try:
            resp = client.models.generate_content(
                model=self._model,
                contents=full_prompt,
            )
        except Exception as exc:
            logger.exception("gemini call failed")
            return (f"gemini call_failed: {type(exc).__name__}: {exc}",
                     "skipped", [], 0.0)

        response_text = getattr(resp, "text", None) or str(resp)
        usage = getattr(resp, "usage_metadata", None)
        in_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        # Include thoughts tokens (Gemini 2.5 reasoning surcharge)
        out_tokens = int(
            (getattr(usage, "candidates_token_count", 0) or 0)
            + (getattr(usage, "thoughts_token_count", 0) or 0)
        )

        pricing = _GEMINI_PRICING.get(self._model, _GEMINI_DEFAULT_PRICING)
        cost = (
            out_tokens * pricing["output"] / 1_000_000.0
            + in_tokens * pricing["input"]  / 1_000_000.0
        )

        severity = _parse_severity_from_response(response_text)
        categories = _parse_flagged_categories(response_text)

        return response_text, severity, categories, float(cost)


def install() -> None:
    """Register the Gemini provider with engine.research.external_audit."""
    from engine.research.external_audit import register_provider
    register_provider(GeminiExternalAuditProvider())


# Auto-install on import — caller activates via EXTERNAL_AUDIT_PROVIDER=gemini
install()
