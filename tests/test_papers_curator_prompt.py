"""tests/test_papers_curator_prompt.py — L1 immutable system prompt guards.

These tests freeze the load-bearing structure of the immutable system
prompt. Each assertion below maps to a doctrine point the prompt MUST
carry; an accidental edit that drops one of these is a doctrine
regression and must be caught at commit time, not at runtime.
"""
from __future__ import annotations


def test_prompt_returns_string():
    from engine.agents.papers_curator.prompt import get_prompt
    p = get_prompt()
    assert isinstance(p, str)
    assert len(p) > 0


def test_prompt_is_deterministic():
    """Same value every call — no template substitution, no clock."""
    from engine.agents.papers_curator.prompt import get_prompt
    assert get_prompt() == get_prompt()


def test_prompt_token_budget():
    """Soft budget: ~2-3k tokens ≈ 8-13k characters. Hard ceiling 14k
    chars to keep cached-input cost bounded."""
    from engine.agents.papers_curator.prompt import get_prompt
    chars = len(get_prompt())
    assert 5000 < chars < 14_000, f"prompt is {chars} chars; out of budget"


def test_load_bearing_academic_priors():
    """The 4 academic prior anchors that set our default skepticism MUST
    be named explicitly. A junior analyst would skip these; the prompt
    must not let the LLM forget."""
    from engine.agents.papers_curator.prompt import get_prompt
    p = get_prompt()
    assert "Harvey-Liu-Zhu" in p, "HLZ multiple-testing prior must be named"
    assert "McLean-Pontiff" in p, "post-publication decay prior must be named"
    assert "Hou-Xue-Zhang" in p, "65% non-replication prior must be named"
    assert "Cochrane" in p, "discount-rate framing prior must be named"
    assert "Bailey" in p, "DSR/within-family n_trials must be named"


def test_load_bearing_methodology_thresholds():
    """The strict-gate numeric thresholds are doctrine. Drift in any of
    these without an explicit doctrine-change commit is a regression."""
    from engine.agents.papers_curator.prompt import get_prompt
    p = get_prompt()
    assert "|t|>3" in p, "HLZ |t|>3 multiple-testing bar required"
    assert "0.90" in p or "0.95" in p, "DSR threshold required"
    assert "0.3" in p, "OOS Sharpe threshold required"
    assert "0.5" in p, "book correlation ceiling required"
    assert "180-day" in p, "Compustat PIT lag required"


def test_role_and_employee_model_present():
    """Employee A's role + 4-employee mental model must be in the prompt
    so the LLM keeps the org structure straight."""
    from engine.agents.papers_curator.prompt import get_prompt
    p = get_prompt()
    assert "Employee A" in p
    assert "Employee B" in p
    assert "Employee C" in p
    assert "Employee D" in p


def test_seven_tools_listed():
    """The L2 tool layer is named in the prompt so the LLM knows what
    to call. (Tool DEFINITIONS are passed separately in the API call;
    this is just the routing guide.)"""
    from engine.agents.papers_curator.prompt import get_prompt
    p = get_prompt()
    for tool_name in [
        "list_deployed_sleeves",
        "query_graveyard",
        "query_doctrine",
        "is_composer_covered",
        "data_inventory_check",
        "query_recent_emits",
        "shadow_eval",
    ]:
        assert tool_name in p, f"tool {tool_name} must be named in prompt"


def test_categorical_bans_present():
    """The 5 standing categorical bans must surface so the LLM doesn't
    propose graveyard-banned mechanisms as fresh ideas."""
    from engine.agents.papers_curator.prompt import get_prompt
    p = get_prompt()
    p_low = p.lower()
    assert "equity single-name" in p_low
    assert "regime detection" in p_low
    assert "hft" in p_low or "latency" in p_low
    assert "pattern 5" in p_low
    assert "streamlit" in p_low


def test_intent_match_fields_documented():
    """Every verdict must include 3 intent-match fields when user_reason
    is provided. Documented in prompt so the LLM produces them."""
    from engine.agents.papers_curator.prompt import get_prompt
    p = get_prompt()
    assert "on_intent" in p
    assert "intent_plus" in p
    assert "intent_gap" in p


def test_does_not_contain_volatile_state():
    """Volatile state (deployed sleeve names, RED counts, current
    doctrine specifics) MUST NOT appear in the prompt — they go in
    tool results. If the prompt mentions a specific sleeve name like
    'carry' or a memory file name, this is a doctrine-vs-state leak.
    """
    from engine.agents.papers_curator.prompt import get_prompt
    p = get_prompt()
    # Specific sleeve names should not appear as IF DEPLOYED — they
    # can appear as examples but not as "current state". We grep for
    # patterns that imply the LLM should treat them as facts.
    forbidden = [
        "currently deployed sleeves are",
        "the deployed sleeves are:",
        "as of",                      # any "as of X date" leaks staleness
        "we currently deploy",
    ]
    p_low = p.lower()
    for phrase in forbidden:
        assert phrase not in p_low, f"prompt leaks volatile state: '{phrase}'"
