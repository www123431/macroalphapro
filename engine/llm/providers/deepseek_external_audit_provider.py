"""engine.llm.providers.deepseek_external_audit_provider — adapter that
plugs DeepSeek v4 into engine.research.external_audit.

Why DeepSeek
============
- Different vendor than Anthropic (the system's primary LLM) — satisfies
  Mitigation #1 of self-audit blind-spots doctrine (need DIFFERENT
  reasoning process, not same provider second-opinion)
- Existing DEEPSEEK_API_KEY in .streamlit/secrets.toml (per standing
  memory) — no new credential setup needed
- Cheapest production-grade option: ~$0.001-0.003/1K output tokens
  → ~$0.005/audit at typical 1500-output-token verdict review

To activate
===========
1. This module's import triggers register_provider() — happens on first
   `from engine.llm.providers import deepseek_external_audit_provider`
2. Set EXTERNAL_AUDIT_PROVIDER=deepseek in env (or via .streamlit
   secrets if wired)
3. Next emit_tier_c_verdict triggers a real audit; inbox digest
   surfaces breakdown

Cost tracking
=============
DeepSeek API doesn't return per-call USD cost — we estimate based on
output_token count × pricing (deepseek-v4-flash $0.20/1M output as of
2025). Real billing reconciles via DeepSeek dashboard.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# DeepSeek v4 pricing — model-aware; reconciles via DeepSeek dashboard
# 2026-06-14: pro is the new default (see DeepSeekExternalAuditProvider).
# Pro ~10x flash but has reasoning_tokens for adversarial depth.
_DEEPSEEK_PRICING_PER_M_TOKENS = {
    "deepseek-v4-flash":  {"input": 0.05, "output": 0.20},
    "deepseek-v4-pro":    {"input": 0.27, "output": 1.10},
}
_DEEPSEEK_DEFAULT_PRICING = {"input": 0.27, "output": 1.10}   # safest fallback (pro)


_SEVERITY_KEYWORDS = {
    "critical": ["critical", "wrong", "invalid", "false positive", "should not trust"],
    "concern":  ["concern", "caveat", "weak", "should surface", "questionable"],
    "noted":    ["minor", "noted", "small issue", "edge case"],
    "no_issue": ["no issue", "sound", "appropriate", "looks fine", "no concerns"],
}


def _parse_severity_from_response(response: str) -> str:
    """Best-effort severity extraction from free-form DeepSeek response.
    DeepSeek doesn't always emit structured JSON; we keyword-match the
    severity rubric. Multiple matches → most severe wins."""
    lower = (response or "").lower()
    # Priority: critical > concern > noted > no_issue
    for severity in ("critical", "concern", "noted", "no_issue"):
        if any(kw in lower for kw in _SEVERITY_KEYWORDS[severity]):
            return severity
    return "noted"   # fallback when nothing matches


def _parse_flagged_categories(response: str) -> list[str]:
    """Extract category tags. Looks for known failure-mode keywords."""
    lower = (response or "").lower()
    categories = {
        "spanning":          ["span", "anchor", "factor model"],
        "multi_testing":     ["multi", "fdr", "bonferroni", "dsr", "deflated"],
        "PIT":               ["pit ", "look-ahead", "lookahead", "restate"],
        "survivorship":      ["surviv"],
        "cost":              ["cost model", "transaction cost", "implementation"],
        "regime":            ["regime", "pre-2000", "post-2000", "structural break"],
        "sharpe_SE":         ["sharpe se", "newey-west", "autocorrel"],
        "replication":       ["replicat", "sample period mismatch"],
        "capacity":          ["capacity", "fund size", "market impact"],
    }
    out = []
    for tag, kws in categories.items():
        if any(kw in lower for kw in kws):
            out.append(tag)
    return out


class DeepSeekExternalAuditProvider:
    """Implements ExternalAuditProvider Protocol via DeepSeek v4."""
    name = "deepseek"

    def __init__(self, *, model: str = "deepseek-v4-pro",
                 max_tokens: int = 4096):
        # 2026-06-14: switched default flash → pro for adversarial audit.
        # Audit needs deep critical reasoning (multi-step methodology
        # critique + statistical literature recall), not fast pattern
        # matching. Flash was a cost-optimization default that traded
        # depth for $0.10/wk savings — false economy at our verdict volume
        # (Bailey-LdP cap ~9/wk × $0.005-0.01 pro = $0.05-0.10/wk
        # total, well under audit budget cap).
        self._model = model
        self._max_tokens = max_tokens

    def adversarial_audit(
        self,
        *,
        subject_payload: dict,
        prompt:          str,
    ) -> tuple[str, str, list[str], float]:
        try:
            from engine.llm.providers.deepseek_provider import call_deepseek
        except Exception as exc:
            logger.exception("deepseek import failed; audit skipped")
            return (f"deepseek import failed: {exc}", "skipped", [], 0.0)

        system_msg = (
            "You are an independent quantitative research reviewer. The user "
            "will provide a factor research verdict produced by another LLM. "
            "Your job: identify any methodological failure modes the original "
            "system likely missed. Be specific. Cite relevant literature where "
            "useful (López de Prado, Bailey, Harvey-Liu-Zhu, Fama-French, etc.). "
            "Output structured: SEVERITY: <one of critical/concern/noted/no_issue>. "
            "CATEGORIES: <comma-separated short tags>. EXPLANATION: <2-4 sentences>."
        )

        try:
            result = call_deepseek(
                model      = self._model,
                system     = system_msg,
                user       = prompt,
                max_tokens = self._max_tokens,
            )
        except Exception as exc:
            logger.exception("deepseek call failed")
            return (f"deepseek call_failed: {type(exc).__name__}: {exc}",
                     "skipped", [], 0.0)

        # _RawCallResult exposes input_tokens / output_tokens as direct
        # attributes (per deepseek_provider.py), NOT via a usage dict.
        # Fallback to raw_usage dict if attrs missing.
        response_text = getattr(result, "text", "") or ""
        out_tokens = getattr(result, "output_tokens", 0) or 0
        in_tokens  = getattr(result, "input_tokens",  0) or 0
        if not out_tokens or not in_tokens:
            raw = getattr(result, "raw_usage", None) or {}
            out_tokens = out_tokens or raw.get("completion_tokens") or raw.get("output_tokens") or 0
            in_tokens  = in_tokens  or raw.get("prompt_tokens")     or raw.get("input_tokens")  or 0
        pricing = _DEEPSEEK_PRICING_PER_M_TOKENS.get(self._model, _DEEPSEEK_DEFAULT_PRICING)
        cost = (
            out_tokens * pricing["output"] / 1_000_000.0
            + in_tokens * pricing["input"]  / 1_000_000.0
        )

        severity = _parse_severity_from_response(response_text)
        categories = _parse_flagged_categories(response_text)

        return response_text, severity, categories, float(cost)


def install() -> None:
    """Register the DeepSeek provider with engine.research.external_audit."""
    from engine.research.external_audit import register_provider
    register_provider(DeepSeekExternalAuditProvider())


# Auto-install on import — caller activates via env var
install()
