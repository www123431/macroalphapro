"""tests/test_da_verdict_schema.py — T2 DAVerdict + DAClaim + JSON schema tests.

Covers:
  - DAClaim self-validate
  - DAVerdict self-validate (including the key invariant: empty claims
    requires insufficient_evidence stance)
  - REJECT requires ≥ 1 refute claim
  - JSON schema structural validity
  - Tool definition format (Anthropic SDK ready)
"""
from __future__ import annotations

import json

import pytest

from engine.research_store.da_briefing import (
    DAClaim, DAStance, DAVerdict,
    DA_VERDICT_JSON_SCHEMA, DA_VERDICT_TOOL_DEFINITION,
    OverallStance,
)


# ─────────────────────── DAClaim ──────────────────────────────────────


def _claim(stance: DAStance = DAStance.REFUTES,
           chunk_id: str = "doi/abc::p0001",
           paper_id: str = "paper-uuid-001",
           quote_text: str = "Post-publication factor returns decay 26-58 percent on average.",
           argument: str = "Directly contradicts the candidate's claim of stable alpha.") -> DAClaim:
    return DAClaim(
        stance      = stance,
        chunk_id    = chunk_id,
        paper_id    = paper_id,
        quote_text  = quote_text,
        section_ref = "p.10, Table 4",
        argument    = argument,
    )


def test_claim_round_trip():
    c = _claim()
    c2 = DAClaim.from_dict(c.to_dict())
    assert c == c2


def test_claim_rejects_empty_chunk_id():
    c = _claim(chunk_id="")
    assert any("chunk_id" in e for e in c.validate())


def test_claim_rejects_empty_paper_id():
    c = _claim(paper_id="")
    assert any("paper_id" in e for e in c.validate())


def test_claim_rejects_quote_too_short():
    c = _claim(quote_text="too short")
    errs = c.validate()
    assert any("too short" in e for e in errs)


def test_claim_rejects_empty_argument():
    c = _claim(argument="")
    assert any("argument" in e for e in c.validate())


def test_claim_minimal_valid_passes():
    assert _claim().validate() == []


# ─────────────────────── DAVerdict ────────────────────────────────────


def _verdict(refutes: tuple[DAClaim, ...] = (),
             supports: tuple[DAClaim, ...] = (),
             conditional: tuple[DAClaim, ...] = (),
             overall_stance: OverallStance = OverallStance.REJECT,
             overall_rationale: str = "The retrieved chunks contain explicit evidence that the proposed mechanism has been refuted.") -> DAVerdict:
    return DAVerdict(
        verdict_id           = DAVerdict.new_id(),
        candidate_name       = "test_candidate",
        target_hypothesis_id = "hyp-uuid-001",
        version              = 1,
        parent_verdict_id    = None,
        refutes              = refutes,
        supports             = supports,
        conditional          = conditional,
        overall_stance       = overall_stance,
        overall_rationale    = overall_rationale,
        n_chunks_retrieved   = 5,
        papers_consulted     = ("paper-uuid-001",),
        created_ts           = "2026-06-04T15:00:00Z",
        created_by           = "test",
        tags                 = ("test",),
    )


def test_verdict_round_trip():
    v = _verdict(refutes=(_claim(),))
    v2 = DAVerdict.from_dict(v.to_dict())
    assert v == v2


def test_verdict_round_trip_via_json():
    v = _verdict(refutes=(_claim(),))
    s = json.dumps(v.to_dict(), ensure_ascii=False)
    v2 = DAVerdict.from_dict(json.loads(s))
    assert v == v2


def test_verdict_rejects_empty_candidate_name():
    v = _verdict(refutes=(_claim(),))
    bad = DAVerdict(**{**v.__dict__, "candidate_name": "  "})
    assert any("candidate_name" in e for e in bad.validate())


def test_verdict_rejects_short_rationale():
    v = _verdict(refutes=(_claim(),),
                 overall_rationale="too short")
    assert any("rationale too short" in e for e in v.validate())


def test_verdict_INVARIANT_empty_claims_requires_insufficient_evidence():
    """KEY invariant — DA verdict with 0 claims AND non-insufficient
    stance is rejected. Otherwise empty pass masquerades as decision."""
    bad = _verdict(
        refutes=(), supports=(), conditional=(),
        overall_stance=OverallStance.PROCEED_WITH_CAVEATS,
        overall_rationale="Long enough rationale text to clear the min length requirement.",
    )
    errs = bad.validate()
    assert any("0 claims" in e for e in errs)


def test_verdict_INVARIANT_empty_claims_ok_with_insufficient_evidence():
    """The legal way to have no claims: explicit insufficient_evidence."""
    ok = _verdict(
        refutes=(), supports=(), conditional=(),
        overall_stance=OverallStance.INSUFFICIENT_EVIDENCE,
        overall_rationale="The retrieved chunks do not speak to the proposed candidate; cannot critique.",
    )
    assert ok.validate() == []


def test_verdict_INVARIANT_reject_requires_refute_claim():
    """overall_stance=reject without any refutes claim is invalid."""
    bad = _verdict(
        refutes=(),
        supports=(_claim(stance=DAStance.SUPPORTS),),
        overall_stance=OverallStance.REJECT,
        overall_rationale="A long enough rationale here for the min length check to be satisfied.",
    )
    assert any("reject requires ≥ 1 refutes" in e for e in bad.validate())


def test_verdict_proceed_with_caveats_with_only_supports_passes():
    v = _verdict(
        refutes=(),
        supports=(_claim(stance=DAStance.SUPPORTS),),
        overall_stance=OverallStance.PROCEED_WITH_CAVEATS,
        overall_rationale="Evidence retrieved supports the candidate; no refutation found; proceed cautiously.",
    )
    assert v.validate() == []


def test_verdict_propagates_claim_errors():
    """Each claim's self-validate errors should appear in verdict errors."""
    bad_claim = DAClaim(
        stance=DAStance.REFUTES, chunk_id="", paper_id="p1",
        quote_text="long enough quote text for the substantive bar to be cleared", section_ref="",
        argument="long enough argument text to satisfy the argument requirement.",
    )
    v = _verdict(refutes=(bad_claim,),
                 overall_rationale="A long enough rationale string for the validate min length requirement.")
    errs = v.validate()
    assert any("refutes[0]" in e and "chunk_id" in e for e in errs)


def test_verdict_all_claims_concatenates_groups():
    v = _verdict(
        refutes=(_claim(stance=DAStance.REFUTES),),
        supports=(_claim(stance=DAStance.SUPPORTS),),
        conditional=(_claim(stance=DAStance.CONDITIONAL),),
    )
    assert len(v.all_claims()) == 3


# ─────────────────────── JSON schema structural ───────────────────────


def test_json_schema_top_level_required_fields():
    required = set(DA_VERDICT_JSON_SCHEMA["required"])
    assert {"candidate_name", "target_hypothesis_id", "refutes",
            "supports", "conditional", "overall_stance",
            "overall_rationale", "n_chunks_retrieved",
            "papers_consulted"}.issubset(required)


def test_json_schema_overall_stance_enum_complete():
    props = DA_VERDICT_JSON_SCHEMA["properties"]
    enum_values = set(props["overall_stance"]["enum"])
    # Must match Python OverallStance enum 1:1
    expected = {o.value for o in OverallStance}
    assert enum_values == expected


def test_json_schema_quote_text_min_length():
    """quote_text minLength must be at least 20 (matches dataclass validate)."""
    claim_schema = DA_VERDICT_JSON_SCHEMA["properties"]["refutes"]["items"]
    assert claim_schema["properties"]["quote_text"]["minLength"] == 20


def test_tool_definition_anthropic_format():
    """The tool definition has the keys Anthropic SDK messages.create
    tools= param expects."""
    td = DA_VERDICT_TOOL_DEFINITION
    assert "name" in td
    assert "description" in td
    assert "input_schema" in td
    assert td["input_schema"] is DA_VERDICT_JSON_SCHEMA


def test_tool_definition_name_is_submit_da_verdict():
    """The tool name is what the prompt instructs the model to call."""
    assert DA_VERDICT_TOOL_DEFINITION["name"] == "submit_da_verdict"
