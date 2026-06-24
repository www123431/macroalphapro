"""engine/research/cross_domain_transfer.py — β Cross-Domain Transfer.

Phase β of the multi-agent brainstorm rebuild (2026-06-14). NOT a
brainstorm — a single specialized cross-asset thinker persona that
proposes 1-2 mechanism transfers per deployed GREEN sleeve. Output
routes to the ENHANCE pipeline (Politis-Romano paired bootstrap,
NOT the forward pipeline / strict-gate).

ACADEMIC ANCHOR
================
  - Frazzini-Pedersen 2018 "Trading Costs": 70% of institutional alpha
    comes from ENHANCING existing positions (better timing / vol scaling
    / cross-asset extension), NOT from net-new factor discovery
  - Cochrane "asset pricing is a triangle": same economic mechanism
    often manifests across asset classes (carry, momentum, value all
    appear in equities + bonds + commodities + FX)
  - Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere":
    canonical example of cross-asset mechanism transfer

WHY NOT N-PERSONA BRAINSTORM
============================
Same reason as α (pre_mortem.py): substrate-bound, not idea-bound.
But unlike α, this is GENERATIVE — produces new test candidates.
Crucially: each candidate routes to ENHANCE pipeline where the
denominator is paired bootstrap (much tighter SE than forward strict
gate per [[feedback-forward-vs-enhance-statistical-separation
-2026-06-11]]), so a marginal transfer can still be IMPROVEMENT.

OUTPUT SCHEMA
=============
TransferProposal
  source_sleeve_id:   the deployed sleeve we're transferring FROM
  target_asset_class: e.g. "us_bonds_treasury_futures", "fx_g10",
                      "options_vol_surface_spx" (controlled enum?)
  mechanism_carry:    1-2 sentence what carries from source to target
  testable_spec_hint: 1-2 sentence concrete data + dispatcher hint
  precedent_paper:    paper that's done something similar (LLM cites)
  confidence:         0.0-0.99 LLM self-rating
  expected_correlation_with_source: 0.0-1.0 estimate
                      (relevant to enhance vs new-factor classifier
                      per the forward-vs-enhance doctrine)

PERSISTENCE
===========
data/research/transfer_proposals.jsonl — one row per proposal.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFER_PROPOSALS_PATH = _REPO_ROOT / "data" / "research" / "transfer_proposals.jsonl"
MECHANISM_LIBRARY_DIR   = _REPO_ROOT / "data" / "research" / "mechanism_library"

_CONFIDENCE_MAX = 0.99


@_dc.dataclass(frozen=True)
class TransferProposal:
    proposal_id:                     str
    source_sleeve_id:                str
    source_family:                   str
    target_asset_class:              str
    mechanism_carry:                 str
    testable_spec_hint:              str
    precedent_paper:                 str
    confidence:                      float
    expected_correlation_with_source: float
    rationale:                       str
    proposed_ts:                     str
    model:                           str
    cost_usd:                        float


_SYSTEM_PROMPT = """\
You are the CROSS-ASSET THINKER. Your job is to propose 1-2 testable
mechanism transfers from a deployed GREEN sleeve to OTHER asset
classes. You are NOT proposing new factors — you are transferring the
SAME economic mechanism that already works.

ACADEMIC GROUNDING
==================
- Frazzini-Pedersen 2018: 70% of institutional alpha = ENHANCE deployed,
  NOT new factor. Cross-asset transfer is the biggest enhance type.
- Cochrane: asset pricing is a triangle — same risk premium often
  appears across asset classes (carry, momentum, value).
- Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere":
  canonical cross-asset transfer (cite when transferring value/MOM).

INPUT YOU GET
=============
- Source sleeve: family, canonical paper, mechanism, currently-used
  asset class
- Source sleeve's deployed KPIs (Sharpe, t-stat) — proof of GREEN

WHAT TO PRODUCE
===============
1-2 TransferProposal items via emit_transfers tool. Each:
  - target_asset_class: pick ONE specific other asset class. Prefer:
      us_bonds_treasury_futures / fx_g10 / commodity_futures_top_10 /
      cdx_ig_hy_spreads / options_vol_surface_spx / em_equity_etfs /
      muni_curve / convertible_bonds — but if you have a precise one
      not in this list, use it (free-text)
  - mechanism_carry: 1-2 sentences naming the SPECIFIC economic
    mechanism that carries (NOT "momentum is universal" but
    "MOM-12m signal works in equities via underreaction to news;
    same underreaction documented in bond markets by Asness 2013
    when computed on 10y yield changes")
  - testable_spec_hint: 1-2 sentences naming the data + the
    dispatcher signal_kind. Be concrete enough that a quant could
    spec this in <30min. Cite a parquet path or data source IF
    you know one in this codebase.
  - precedent_paper: name + year of paper that's done something
    SIMILAR. Empty string OK if genuinely no precedent (rare).
  - confidence: 0.0-0.99 (NEVER 1.0). Drop below 0.5 if the
    transfer involves regime changes / data we don't have.
  - expected_correlation_with_source: realistic guess on monthly
    return correlation. > 0.5 → likely an ENHANCE candidate
    (paired bootstrap SE is tighter); < 0.5 → likely a NEW
    factor (forward gate with full Bailey-LdP n_trials penalty).
  - rationale: 1-2 sentences why this transfer is worth the gate
    budget (e.g. "Frazzini-Pedersen alpha-on-deployed-positions
    play; the source sleeve already has Sh=1.2, transferring to
    bonds doubles capacity without doubling factor exposure").

DO NOT
======
- Propose transfers that need data we obviously don't have (e.g.
  HFT tick data, China A-share intraday, options on individual
  small-caps)
- Propose transfers with confidence < 0.3 (just don't propose them)
- Generate 3+ proposals — quality over quantity, 1-2 is the spec
- Propose transfers within the SAME asset class (that's a different
  workflow — sleeve variant proposal, not cross-asset transfer)
"""


_TOOL_SCHEMA = {
    "name": "emit_transfers",
    "description": "Emit cross-asset transfer proposals as structured JSON.",
    "input_schema": {
        "type": "object",
        "required": ["proposals", "rationale"],
        "properties": {
            "proposals": {
                "type": "array",
                "minItems": 0,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "required": ["target_asset_class", "mechanism_carry",
                                 "testable_spec_hint", "precedent_paper",
                                 "confidence",
                                 "expected_correlation_with_source"],
                    "properties": {
                        "target_asset_class":               {"type": "string", "maxLength": 100},
                        "mechanism_carry":                  {"type": "string", "maxLength": 500},
                        "testable_spec_hint":               {"type": "string", "maxLength": 500},
                        "precedent_paper":                  {"type": "string", "maxLength": 200},
                        "confidence":                       {"type": "number", "minimum": 0.0, "maximum": _CONFIDENCE_MAX},
                        "expected_correlation_with_source": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
            "rationale": {"type": "string", "maxLength": 600},
        },
    },
}


def _load_sleeve_spec(sleeve_id: str) -> Optional[dict]:
    """Read the sleeve's YAML spec from mechanism library."""
    try:
        import yaml as _pyyaml
    except Exception:
        logger.warning("yaml not installed")
        return None
    for fp in MECHANISM_LIBRARY_DIR.glob("*.yaml"):
        if fp.name.startswith("_"):
            continue
        try:
            d = _pyyaml.safe_load(fp.read_text(encoding="utf-8"))
            if isinstance(d, dict) and d.get("id") == sleeve_id:
                return d
        except Exception:
            continue
    return None


def _sleeve_context_block(sleeve_id: str, spec: Optional[dict]) -> str:
    if not spec:
        return f"sleeve_id: {sleeve_id}  (spec not found in mechanism_library)\n"
    canonical = spec.get("canonical_paper_id") or "?"
    family = spec.get("family") or spec.get("parent_family") or "?"
    purpose = spec.get("purpose") or "?"
    required_data = spec.get("required_data") or []
    observed = (spec.get("post_pub_decay") or {}).get("our_observed") or {}
    obs_sharpe = observed.get("summary_sharpe_observed")
    return (
        f"DEPLOYED SLEEVE\n"
        f"===============\n"
        f"id:                {sleeve_id}\n"
        f"family:            {family}\n"
        f"purpose:           {purpose}\n"
        f"canonical_paper:   {canonical}\n"
        f"required_data:     {required_data[:8]}\n"
        f"observed_sharpe:   {obs_sharpe}\n"
    )


def propose_transfers(
    sleeve_id: str,
    *,
    persist: bool = True,
) -> Optional[tuple[TransferProposal, ...]]:
    """Generate cross-asset transfer proposals for a deployed GREEN
    sleeve. Returns tuple of TransferProposal or None on failure."""
    spec = _load_sleeve_spec(sleeve_id)
    user_msg = "\n".join([
        _sleeve_context_block(sleeve_id, spec),
        "",
        "Now propose 1-2 cross-asset transfers. Be specific. Cite a "
        "precedent paper. NEVER 1.0 confidence. If you can't find a "
        "good transfer, return 0 proposals + PROCEED_NORMAL rationale "
        "rather than inflating count.",
    ])

    try:
        result = llm_call(
            workload   = "cross_domain_transfer",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "cross_domain_transfer",
            tools      = [_TOOL_SCHEMA],
            max_tokens = 1536,
            scope      = "beta_cross_domain_transfer",
        )
    except Exception as exc:
        logger.warning("transfer: llm_call failed for %s: %s",
                        sleeve_id, exc)
        return None

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_transfers":
            payload = tc.input
            break
    if payload is None:
        logger.warning("transfer: %s did not call emit_transfers", sleeve_id)
        return None

    rationale = str(payload.get("rationale") or "")[:600]
    now_iso = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    proposals: list[TransferProposal] = []
    for p in (payload.get("proposals") or []):
        try:
            conf = float(p.get("confidence"))
            exp_corr = float(p.get("expected_correlation_with_source"))
            if not (0.0 <= conf <= _CONFIDENCE_MAX):
                continue
            if not (0.0 <= exp_corr <= 1.0):
                continue
            proposals.append(TransferProposal(
                proposal_id      = str(uuid.uuid4()),
                source_sleeve_id = sleeve_id,
                source_family    = (spec or {}).get("family") or "?",
                target_asset_class = str(p.get("target_asset_class"))[:100],
                mechanism_carry  = str(p.get("mechanism_carry"))[:500],
                testable_spec_hint = str(p.get("testable_spec_hint"))[:500],
                precedent_paper  = str(p.get("precedent_paper"))[:200],
                confidence       = conf,
                expected_correlation_with_source = exp_corr,
                rationale        = rationale,
                proposed_ts      = now_iso,
                model            = result.model,
                cost_usd         = float(result.cost_usd),
            ))
        except Exception:
            continue

    if persist and proposals:
        try:
            TRANSFER_PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TRANSFER_PROPOSALS_PATH.open("a", encoding="utf-8") as f:
                for tp in proposals:
                    f.write(json.dumps(_dc.asdict(tp), ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("transfer: persist failed", exc_info=True)

    return tuple(proposals)


def list_for_sleeve(sleeve_id: str) -> list[dict]:
    if not TRANSFER_PROPOSALS_PATH.is_file():
        return []
    out: list[dict] = []
    for ln in TRANSFER_PROPOSALS_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("source_sleeve_id") == sleeve_id:
            out.append(r)
    out.sort(key=lambda r: r.get("proposed_ts", ""), reverse=True)
    return out


def list_all_recent(limit: int = 50) -> list[dict]:
    if not TRANSFER_PROPOSALS_PATH.is_file():
        return []
    out: list[dict] = []
    for ln in TRANSFER_PROPOSALS_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    out.sort(key=lambda r: r.get("proposed_ts", ""), reverse=True)
    return out[:limit]
