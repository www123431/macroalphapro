"""tests/test_data_egress.py — LLM data-egress / residency governance.

The guard must CATCH position-level data heading to a CN provider, and must NOT block
benign aggregate stats. Deterministic, no LLM.
"""
import pytest

from engine.agents.governance.data_egress import (
    classify_sensitivity, evaluate_egress, guard_egress, egress_matrix,
    EgressViolation, EGRESS_POLICY, PROVIDER_RESIDENCY,
)

# representative payloads mirroring REAL tool outputs
POSITION = '{"strategy":"K1_BAB","n_positions":12,"top_holdings":[{"ticker":"AGG","weight":0.082},{"ticker":"TLT","weight":0.061}]}'
AGGREGATE = "Book health: rolling Sharpe 1.26, VaR -2.3%, max drawdown -9.8%. No structural decay."
PUBLIC = "Is the post-earnings drift anomaly still worth testing?"
PII = "Please email the report to john.doe@example.com or call about SSN 123-45-6789."


# ── classifier ───────────────────────────────────────────────────────────────
def test_classifier_levels():
    assert classify_sensitivity(POSITION) == "POSITION"
    assert classify_sensitivity(AGGREGATE) == "AGGREGATE"
    assert classify_sensitivity(PUBLIC) == "PUBLIC"
    assert classify_sensitivity(PII) == "PII"


# ── policy evaluation ────────────────────────────────────────────────────────
def test_position_to_cn_provider_is_a_violation():
    d = evaluate_egress("deepseek", POSITION)        # DeepSeek = CN
    assert d.residency == "CN" and d.allowed is False and d.sensitivity == "POSITION"


def test_position_to_us_provider_allowed():
    assert evaluate_egress("anthropic", POSITION).allowed is True


def test_aggregate_to_cn_provider_allowed():
    assert evaluate_egress("deepseek", AGGREGATE).allowed is True


def test_pii_blocked_everywhere():
    assert evaluate_egress("anthropic", PII).allowed is False
    assert evaluate_egress("deepseek", PII).allowed is False


# ── guard modes ──────────────────────────────────────────────────────────────
def test_enforce_raises_on_violation():
    with pytest.raises(EgressViolation):
        guard_egress("deepseek", POSITION, mode="enforce")


def test_warn_does_not_raise_but_flags():
    d = guard_egress("deepseek", POSITION, mode="warn")
    assert d.allowed is False                         # flagged, not raised


def test_allowed_call_never_raises():
    assert guard_egress("deepseek", AGGREGATE, mode="enforce").allowed is True
    assert guard_egress("anthropic", POSITION, mode="enforce").allowed is True


def test_off_mode_is_passthrough():
    assert guard_egress("deepseek", POSITION, mode="off").allowed is False  # decision returned, not enforced


# ── matrix surfaces the real finding ─────────────────────────────────────────
def test_matrix_marks_deepseek_cn_aggregate_max():
    m = egress_matrix()
    assert m["deepseek"] == {"residency": "CN", "max_sensitivity": "AGGREGATE"}
    assert m["anthropic"]["max_sensitivity"] == "POSITION"
