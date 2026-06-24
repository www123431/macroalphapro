"""engine/research/discovery/paper_extractor.py — LLM extracts mechanism
proposal from a paper abstract.

For each discovered paper:
1. LLM reads title + abstract
2. Returns STRICT JSON with:
   - proposed mechanism (would it become a library entry?)
   - mechanism family (mapped to KG taxonomy)
   - required data tokens
   - economic intuition
   - decay-resilience claim from abstract
   - novelty assessment
3. Output schema-validated before returning

LLM is ADVISORY here — humans (or hygiene_gate.py) decide whether the
proposal becomes a library candidate.

Doctrine:
- ALWAYS deterministic schema validation on LLM output
- NEVER auto-add to library
- Cost-capped per paper (~$0.05)
- Falls back to None on parse failure (caller handles)
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # cheap for bulk paper triage


@dataclasses.dataclass
class PaperExtraction:
    arxiv_id:                str
    title:                   str
    mechanism_proposal:      str          # 1-2 sentence summary
    family_guess:            str          # mapped to KG taxonomy
    parent_family_guess:     str
    required_data_tokens:    list[str]    # token IDs from our inventory
    economic_intuition:      str
    decay_resilience_claim:  str          # what the paper says about decay
    novelty_assessment:      str          # 'novel' | 'extension' | 'rebrand'
    confidence:              float
    cost_usd:                float
    mode:                    str          # 'llm' | 'deterministic_fallback'

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


SYSTEM_PROMPT = """You are a senior quant factor researcher reading an arXiv abstract. Triage whether the paper proposes a NEW alpha mechanism worth library inclusion.

# Doctrine
- Be skeptical. Most arXiv q-fin papers are NOT novel mechanisms — many are extensions, ML applications without economic theory, or rebrandings.
- Map to OUR KG taxonomy when possible: earnings_underreaction, quality, momentum, residual_momentum, carry, vol_carry, tsmom, lead_lag, regime_overlay, breadth_expansion, news_sentiment, credit_risk.
- Map data needs to OUR inventory tokens: crsp_dsf, crsp_msf, compustat_quarterly, compustat_annual, ibes_summary, fred_macro, vix_index, tr_ds_fut_settle, edgar_8k_meta, sp500_constituents, cftc_cot.
- If the paper requires data we don't have, list it but flag.

# Output strict JSON (no preamble, no markdown):
{
  "mechanism_proposal": "<1-2 sentence summary of proposed alpha mechanism>",
  "family_guess": "<one of our taxonomy values or 'unknown'>",
  "parent_family_guess": "<equity_factor | cross_asset_carry | cross_asset_trend | network_effects | alt_data | regime_management | credit | unknown>",
  "required_data_tokens": ["<token>", ...],
  "economic_intuition": "<1-3 sentence economic theory>",
  "decay_resilience_claim": "<what the paper says about post-pub decay, robustness, OR 'not addressed'>",
  "novelty_assessment": "<novel | extension | rebrand>",
  "confidence": <float 0-1 — how confident YOU are about this triage>
}

Score guide for confidence:
  0.8-1.0: clear novel mechanism with explicit economic theory
  0.5-0.8: extension of known mechanism with refinement
  0.2-0.5: unclear / ML methodology with weak economic story
  0.0-0.2: not a factor mechanism (e.g. pricing model, derivative analytics)
"""


def _read_anthropic_key() -> str | None:
    """Env var → direct TOML parse of .streamlit/secrets.toml.

    We're no longer running under Streamlit (user 2026-05-30: "我们现在
    不用streamlit了"), but the secrets.toml file is still the project's
    credential store. Pure TOML parse — no streamlit import.
    """
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        from pathlib import Path
        secrets_path = Path(".streamlit/secrets.toml")
        if not secrets_path.exists():
            return None
        try:
            import tomllib    # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                # Last resort: regex parse (single-line string values)
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


def extract_from_paper(
    arxiv_id: str, title: str, abstract: str,
    *, use_llm: bool = True,
) -> PaperExtraction | None:
    """Extract structured proposal from one paper. Returns None on failure."""
    if not use_llm:
        return _deterministic_fallback(arxiv_id, title, abstract=abstract)

    key = _read_anthropic_key()
    if not key:
        return _deterministic_fallback(arxiv_id, title, abstract=abstract)

    try:
        from anthropic import Anthropic
    except ImportError:
        return _deterministic_fallback(arxiv_id, title, abstract=abstract)

    try:
        client = Anthropic(api_key=key, timeout=60.0)
        user_msg = (
            f"Paper: {title}\n\n"
            f"arXiv ID: {arxiv_id}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Triage per system prompt. Return strict JSON."
        )
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        usage = response.usage
        cost = (usage.input_tokens * 0.25 / 1_000_000
                  + usage.output_tokens * 1.25 / 1_000_000)
        text = "\n".join(b.text for b in response.content if b.type == "text")
        parsed = _parse_json(text)
        if not parsed:
            return _deterministic_fallback(arxiv_id, title)
        family_guess = str(parsed.get("family_guess", "unknown"))
        required_data_tokens = list(parsed.get("required_data_tokens") or [])
        # PRODUCTION semantics per [[project-senior-pipeline-roadmap-
        # 2026-05-30]]: confidence comes from the DETERMINISTIC
        # calculator (reproducible + auditable), NOT LLM self-rating.
        # LLM still drives prose: mechanism_proposal / family_guess /
        # required_data_tokens / economic_intuition.
        from engine.research.discovery.confidence_calculator import (
            compute_confidence,
        )
        det_conf = compute_confidence(
            title, abstract,
            required_data_tokens=required_data_tokens,
            family_guess=family_guess,
        )
        return PaperExtraction(
            arxiv_id=               arxiv_id,
            title=                  title,
            mechanism_proposal=     str(parsed.get("mechanism_proposal", "")),
            family_guess=           family_guess,
            parent_family_guess=    str(parsed.get("parent_family_guess", "unknown")),
            required_data_tokens=   required_data_tokens,
            economic_intuition=     str(parsed.get("economic_intuition", "")),
            decay_resilience_claim= str(parsed.get("decay_resilience_claim", "not addressed")),
            novelty_assessment=     str(parsed.get("novelty_assessment", "unclear")),
            confidence=             det_conf.confidence,
            cost_usd=               round(cost, 5),
            mode=                   "llm",
        )
    except Exception as exc:
        logger.warning("LLM extraction failed for %s: %s", arxiv_id, exc)
        return _deterministic_fallback(arxiv_id, title, abstract=abstract)


def _deterministic_fallback(
    arxiv_id: str, title: str, *, abstract: str = "",
) -> PaperExtraction:
    """When LLM is unavailable, still compute deterministic confidence
    from observable text features. Empty mechanism_proposal signals
    that LLM prose extraction did not run."""
    from engine.research.discovery.confidence_calculator import (
        compute_confidence,
    )
    det_conf = compute_confidence(title, abstract or "")
    return PaperExtraction(
        arxiv_id=arxiv_id, title=title,
        mechanism_proposal="(LLM unavailable; manual triage needed)",
        family_guess="unknown", parent_family_guess="unknown",
        required_data_tokens=[],
        economic_intuition="", decay_resilience_claim="not addressed",
        novelty_assessment="unclear", confidence=det_conf.confidence,
        cost_usd=0.0, mode="deterministic_fallback",
    )


def _parse_json(text: str) -> dict | None:
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
