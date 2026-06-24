"""engine/research/discovery/binding_proposer.py — LLM auto-propose
execution_template + binding + required_data for a promoted candidate.

Senior throughput unlocker per [[feedback-confirm-meaningful-before-borrowing-2026-05-30]]:
manual binding writing was the 1-2/year throughput bottleneck for a
single power user. With LLM proposal + 30-sec human review, target
goes to 10-30/year — same person, 5-10× more.

DESIGN:
  Input: paper metadata (title / abstract / family / required_data_tokens
         from earlier LLM extraction)
  + signatures of the 6 currently-registered templates (template_id,
    GATE_PROFILE, expected binding param names)
  Output: BindingProposal {template_id, binding dict, required_data,
                            reasoning, cost_usd}
  Validation: the LLM-proposed template_id MUST be in TEMPLATES registry
              (anti-hallucination guard).
              The binding keys are filtered against
              auto_gate.DEFAULT_AUTOGATE_BINDINGS keys + per-template
              GATE_PROFILE-known keys to drop hallucinated params.
              required_data filtered against IMPLEMENTED_DATA so we
              don't propose data that has no fetcher.

NOT auto-committed: returns the proposal for human review.
queue_actions.promote stores it under stub.proposed_binding;
human approves via UI → stub.execution_template is filled.

COST: ~$0.002/paper (Sonnet output 200 tokens × $15/M = $0.003).
At 30 promotes/year = ~$0.06/year. Trivial.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

LLM_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for structured proposals
MAX_TOKENS = 600


@dataclasses.dataclass
class BindingProposal:
    template_id:    str
    binding:        dict
    required_data:  list[str]
    reasoning:      str
    cost_usd:       float = 0.0
    valid:          bool = True
    validation_warnings: list[str] = dataclasses.field(default_factory=list)
    error:          str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _read_anthropic_key() -> str | None:
    """env var → direct TOML parse. Mirror of paper_extractor."""
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
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


def _template_signatures() -> dict:
    """Snapshot of each registered template's identity + GATE_PROFILE +
    known binding param names. The LLM sees this catalog and picks one."""
    import importlib
    from engine.research.templates import TEMPLATES

    sigs = {}
    for tpl_id in TEMPLATES.keys():
        try:
            mod = importlib.import_module(
                f"engine.research.templates.{tpl_id}",
            )
            profile = getattr(mod, "GATE_PROFILE", None)
            run_fn = getattr(mod, f"run_{tpl_id}", None)
            param_names: list[str] = []
            if run_fn is not None:
                import inspect
                try:
                    sig = inspect.signature(run_fn)
                    for name, p in sig.parameters.items():
                        # Skip data panels (caller supplies these)
                        if name.endswith("_panel") or name == "returns":
                            continue
                        if p.kind in (inspect.Parameter.KEYWORD_ONLY,
                                        inspect.Parameter.POSITIONAL_OR_KEYWORD):
                            param_names.append(name)
                except (ValueError, TypeError):
                    pass
            doc = (run_fn.__doc__ or "").strip().split("\n")[0] if run_fn else ""
            sigs[tpl_id] = {
                "param_names": param_names,
                "gate_profile": profile,
                "doc": doc[:200],
            }
        except Exception as exc:
            logger.warning("template signature read failed %s: %s",
                              tpl_id, exc)
    return sigs


SYSTEM_PROMPT = """You are proposing a strategy binding for a freshly-
promoted quant research paper. Choose ONE of the registered templates,
fill in its binding parameters with reasonable defaults from the paper,
and list the required_data tokens needed.

REGISTERED TEMPLATES:
{template_catalog}

IMPLEMENTED DATA tokens you can request:
{data_tokens}

Return STRICT JSON:
{{
  "template_id":   "<one of the registered ids>",
  "binding":       {{...binding params dict — only use params known to the template...}},
  "required_data": [...subset of implemented tokens needed by this paper's logic...],
  "reasoning":     "<one-paragraph explanation linking paper's claim to template + binding choices>"
}}

CRITICAL CONSTRAINTS:
- template_id must EXACTLY match one of the registered ids
- binding keys must be drawn from the template's known param list
- required_data must be drawn from the implemented tokens list
- Default conservatively (e.g. top_frac=0.1 for decile L/S; cost_bps=12;
  vol_target=0.10) when paper doesn't specify
- reasoning must be concrete (cite the paper's mechanism + parameter choices)"""


def propose_binding(
    title: str,
    abstract: str,
    *,
    family_guess: str = "unknown",
    economic_intuition: str = "",
    existing_required_data: list[str] | None = None,
    use_llm: bool = True,
) -> BindingProposal:
    """Call LLM to propose execution_template + binding for a paper.

    If use_llm=False, returns a default-template proposal based on family
    (factor_quartile for known families, empty for unknown).
    """
    from engine.research.templates import TEMPLATES
    from engine.research.hygiene_tools import IMPLEMENTED_DATA

    signatures = _template_signatures()
    registered_ids = set(TEMPLATES.keys())

    if not use_llm:
        # Deterministic fallback: pick factor_quartile if family is
        # in the auto_gate family map; otherwise no proposal.
        from engine.research.discovery.auto_gate import _FAMILY_TEMPLATE
        tpl = _FAMILY_TEMPLATE.get(family_guess.lower())
        if tpl and tpl in registered_ids:
            return BindingProposal(
                template_id=tpl, binding={"top_frac": 0.1,
                                              "bottom_frac": 0.1,
                                              "cost_bps_per_side": 12.0},
                required_data=list(existing_required_data or []),
                reasoning="Deterministic fallback (no LLM available); "
                            "default decile L/S for known family.",
                valid=True,
            )
        return BindingProposal(
            template_id="", binding={}, required_data=[],
            reasoning="No LLM and no template inferable from family.",
            valid=False, error="no_llm_and_no_inference",
        )

    key = _read_anthropic_key()
    if not key:
        return BindingProposal(
            template_id="", binding={}, required_data=[],
            reasoning="", valid=False, error="ANTHROPIC_API_KEY missing",
        )
    try:
        from anthropic import Anthropic
    except ImportError:
        return BindingProposal(
            template_id="", binding={}, required_data=[],
            reasoning="", valid=False, error="anthropic SDK missing",
        )

    catalog_str = "\n".join(
        f"  - {tid}: {sig['doc']}\n      params: {sig['param_names'][:12]}"
        for tid, sig in signatures.items()
    )
    data_tokens_str = ", ".join(sorted(IMPLEMENTED_DATA))
    sys_prompt = SYSTEM_PROMPT.format(
        template_catalog=catalog_str,
        data_tokens=data_tokens_str,
    )

    user_msg = (
        f"Title: {title}\n\n"
        f"Family guess: {family_guess}\n"
        f"Economic intuition: {economic_intuition or '(not provided)'}\n"
        f"Existing required_data tokens: {existing_required_data or []}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Propose binding per system prompt. Return strict JSON only."
    )

    try:
        client = Anthropic(api_key=key, timeout=45.0)
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            system=sys_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        usage = response.usage
        cost = (usage.input_tokens * 0.80 / 1_000_000
                  + usage.output_tokens * 4.00 / 1_000_000)
        text = "\n".join(b.text for b in response.content if b.type == "text")
        parsed = _parse_json(text)
        if not parsed:
            return BindingProposal(
                template_id="", binding={}, required_data=[],
                reasoning=text[:300], cost_usd=cost,
                valid=False, error="LLM response not parseable as JSON",
            )

        proposal = BindingProposal(
            template_id=str(parsed.get("template_id", "")).strip(),
            binding=dict(parsed.get("binding") or {}),
            required_data=list(parsed.get("required_data") or []),
            reasoning=str(parsed.get("reasoning", "")).strip(),
            cost_usd=cost,
            valid=True,
        )
        # Validation pass
        warnings = []
        if proposal.template_id not in registered_ids:
            warnings.append(
                f"template_id {proposal.template_id!r} not in registered "
                f"{sorted(registered_ids)} — DROPPING proposal"
            )
            proposal.valid = False
        else:
            known_params = set(signatures[proposal.template_id]["param_names"])
            bad_keys = [k for k in proposal.binding if k not in known_params]
            if bad_keys:
                warnings.append(
                    f"binding keys {bad_keys} not in template params; "
                    f"FILTERED OUT (anti-hallucination)"
                )
                proposal.binding = {k: v for k, v in proposal.binding.items()
                                       if k in known_params}
        unknown_data = [t for t in proposal.required_data
                          if t not in IMPLEMENTED_DATA]
        if unknown_data:
            warnings.append(
                f"required_data tokens {unknown_data} not in IMPLEMENTED_DATA "
                f"— FILTERED OUT; if these tokens are needed, wire the "
                f"fetcher first"
            )
            proposal.required_data = [t for t in proposal.required_data
                                          if t in IMPLEMENTED_DATA]
        proposal.validation_warnings = warnings
        return proposal

    except Exception as exc:
        logger.warning("propose_binding LLM call failed: %s", exc)
        return BindingProposal(
            template_id="", binding={}, required_data=[],
            reasoning="", valid=False, error=str(exc)[:300],
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
