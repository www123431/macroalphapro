"""tests/test_strategy_lifecycle.py — Strategy Lifecycle Manager Phase 0
unit tests.

Covers:
  1. StrategyState enum + terminal-state semantics
  2. SleeveRole enum + cross-module alignment guard
  3. enforce_transition() — base gates + role-specific gates
  4. AuditBlocks (Pydantic) — schema validation + completeness checks
  5. strategy_state_store — ACID transitions, history, allocation updates
  6. sleeve_registry — decorator, get/list, audit-block loading

Run with:
  python -m pytest tests/test_strategy_lifecycle.py -v
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
import tempfile
from pathlib import Path

import pytest

from engine.research.strategy_lifecycle import (
    ALLOWED_TRANSITIONS,
    AuditBlocks,
    CapacityBlock,
    CostModelAudit,
    EXPECTED_ROLE_STRINGS_FROZEN,
    FactorExposureAudit,
    GateNotMetError,
    InvalidTransitionError,
    MultiAumSharpe,
    ROLE_SPECIFIC_GATES,
    SleeveRole,
    StrategyState,
    allowed_next_states,
    enforce_transition,
    is_terminal,
    lookup_role_specific_gate,
)
from engine.research.strategy_state_store import (
    create_strategy,
    get_strategy,
    get_transition_history,
    list_strategies,
    reset_db_for_test,
    transition,
    update_allocation,
)
from engine.research.sleeve_registry import (
    SleeveAlreadyRegisteredError,
    SleeveProtocol,
    clear_registry_for_test,
    get_sleeve,
    get_sleeve_class,
    list_registered_sleeves,
    register_sleeve,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """A fresh SQLite DB per test, isolated from production DEFAULT_DB_PATH."""
    db = tmp_path / "test_lifecycle.db"
    yield db
    reset_db_for_test(db)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Wipe the in-memory registry before every test to avoid bleed."""
    clear_registry_for_test()
    yield
    clear_registry_for_test()


# ── 1. State enum ───────────────────────────────────────────────────────


class TestStrategyState:
    def test_all_states_have_unique_values(self):
        values = [s.value for s in StrategyState]
        assert len(values) == len(set(values))

    def test_terminal_states(self):
        assert is_terminal(StrategyState.ARCHIVED)
        assert is_terminal(StrategyState.REJECTED)
        for s in StrategyState:
            if s not in {StrategyState.ARCHIVED, StrategyState.REJECTED}:
                assert not is_terminal(s)

    def test_terminal_states_have_no_outbound(self):
        for terminal in (StrategyState.ARCHIVED, StrategyState.REJECTED):
            assert allowed_next_states(terminal) == []


# ── 2. Role enum + cross-module alignment ──────────────────────────────


class TestSleeveRole:
    def test_role_strings_match_expected_frozenset(self):
        actual = frozenset(r.value for r in SleeveRole)
        assert actual == EXPECTED_ROLE_STRINGS_FROZEN, (
            "SleeveRole enum drift: SLM, library_factor_exposure_audit, "
            "and candidate_pipeline must share IDENTICAL role strings."
        )

    def test_role_strings_match_library_factor_exposure_audit(self):
        """Guard against drift between SLM and the existing library audit."""
        from engine.research.library_factor_exposure_audit import VALID_ROLES
        assert frozenset(r.value for r in SleeveRole) == VALID_ROLES

    def test_from_yaml_value_round_trip(self):
        for role in SleeveRole:
            assert SleeveRole.from_yaml_value(role.value) == role

    def test_from_yaml_value_rejects_unknown(self):
        with pytest.raises(ValueError, match="not a valid SleeveRole"):
            SleeveRole.from_yaml_value("bogus_role")


# ── 3. Base transition gates ───────────────────────────────────────────


class TestBaseTransitionGates:
    def test_proposed_to_audited_requires_pipeline_run(self):
        with pytest.raises(GateNotMetError, match="candidate_pipeline"):
            enforce_transition(
                from_state=StrategyState.PROPOSED,
                to_state=StrategyState.AUDITED,
                has_candidate_pipeline_run=False,
            )

    def test_proposed_to_audited_passes_with_evidence(self):
        gate, role_gate = enforce_transition(
            from_state=StrategyState.PROPOSED,
            to_state=StrategyState.AUDITED,
            has_candidate_pipeline_run=True,
        )
        assert gate.requires_candidate_pipeline_run is True
        assert role_gate is None  # no role passed

    def test_audited_to_approved_requires_human(self):
        with pytest.raises(GateNotMetError, match="human approval"):
            enforce_transition(
                from_state=StrategyState.AUDITED,
                to_state=StrategyState.APPROVED,
                has_human_approval=False,
            )

    def test_paper_trade_to_shadow_requires_six_months(self):
        with pytest.raises(GateNotMetError, match="≥6 months"):
            enforce_transition(
                from_state=StrategyState.PAPER_TRADE,
                to_state=StrategyState.SHADOW,
                paper_trade_months=3,
                sequential_test_pass=True,
                ramp_protocol_step=1,
            )

    def test_paper_trade_to_shadow_requires_sequential_test(self):
        with pytest.raises(GateNotMetError, match="sequential-test"):
            enforce_transition(
                from_state=StrategyState.PAPER_TRADE,
                to_state=StrategyState.SHADOW,
                paper_trade_months=6,
                sequential_test_pass=False,
                ramp_protocol_step=1,
            )

    def test_invalid_transition_raises(self):
        with pytest.raises(InvalidTransitionError, match="not allowed"):
            enforce_transition(
                from_state=StrategyState.PROPOSED,
                to_state=StrategyState.LIVE,  # cannot jump
            )

    def test_decay_watch_to_live_requires_explicit_override(self):
        # Without override → silently passes only if no extra evidence required
        with pytest.raises(GateNotMetError):
            enforce_transition(
                from_state=StrategyState.DECAY_WATCH,
                to_state=StrategyState.LIVE,
                explicit_override=False,
            )
        # With override → ok (backward / repair transition)
        gate, _ = enforce_transition(
            from_state=StrategyState.DECAY_WATCH,
            to_state=StrategyState.LIVE,
            explicit_override=True,
        )
        assert gate.requires_explicit_override is True


# ── 4. Role-specific gates ─────────────────────────────────────────────


class TestRoleSpecificGates:
    def test_alpha_seeker_paper_to_shadow_role_gate_exists(self):
        gate = lookup_role_specific_gate(
            SleeveRole.ALPHA_SEEKER,
            StrategyState.PAPER_TRADE, StrategyState.SHADOW,
        )
        assert gate is not None
        assert gate.metric_name == "trailing_sharpe"

    def test_insurance_uses_hedge_correlation_not_sharpe(self):
        gate = lookup_role_specific_gate(
            SleeveRole.INSURANCE,
            StrategyState.PAPER_TRADE, StrategyState.SHADOW,
        )
        assert gate is not None
        assert "hedge" in gate.metric_name
        assert "sharpe" not in gate.metric_name.lower()

    def test_diversifier_uses_cosine_not_sharpe(self):
        gate = lookup_role_specific_gate(
            SleeveRole.DIVERSIFIER,
            StrategyState.PAPER_TRADE, StrategyState.SHADOW,
        )
        assert gate is not None
        assert "cosine" in gate.metric_name
        assert "sharpe" not in gate.metric_name.lower()

    def test_role_evidence_not_passed_raises(self):
        """alpha_seeker base gates met but role evidence missing → reject."""
        with pytest.raises(GateNotMetError, match="alpha_seeker role-specific"):
            enforce_transition(
                from_state=StrategyState.PAPER_TRADE,
                to_state=StrategyState.SHADOW,
                role=SleeveRole.ALPHA_SEEKER,
                paper_trade_months=6,
                sequential_test_pass=True,
                ramp_protocol_step=1,
                role_specific_evidence_passed=False,
            )

    def test_role_evidence_passed_accepts(self):
        gate, role_gate = enforce_transition(
            from_state=StrategyState.PAPER_TRADE,
            to_state=StrategyState.SHADOW,
            role=SleeveRole.ALPHA_SEEKER,
            paper_trade_months=6,
            sequential_test_pass=True,
            ramp_protocol_step=1,
            role_specific_evidence_passed=True,
        )
        assert role_gate is not None
        assert role_gate.metric_name == "trailing_sharpe"

    def test_role_none_skips_role_layer(self):
        """System-internal transitions can skip the role layer."""
        gate, role_gate = enforce_transition(
            from_state=StrategyState.PAPER_TRADE,
            to_state=StrategyState.SHADOW,
            role=None,
            paper_trade_months=6,
            sequential_test_pass=True,
            ramp_protocol_step=1,
        )
        assert role_gate is None


# ── 5. AuditBlocks Pydantic schema ──────────────────────────────────────


class TestAuditBlocks:
    def _make_full_cost_model(self) -> CostModelAudit:
        return CostModelAudit(
            audit_status="audited",
            audit_date=_dt.date(2026, 5, 31),
            audit_script="scripts/test.py",
            audit_commit="abc1234",
            type="almgren_chriss",
            half_spread_bps=5.0, impact_coef=0.5, daily_sigma_estimate=0.015,
            universe_median_adv_usd=50_000_000,
            n_positions_typical=110,
            monthly_turnover_estimate=0.50,
            stress_multiplier=2.5,
            rationale="X" * 60,  # ≥ 50 chars required
            multi_aum_sharpe_sleeve=MultiAumSharpe(at_10M=2.0, at_100M=1.9, at_1B=1.7),
            capacity=CapacityBlock(
                hard_capacity_usd=550_000_000,
                binding_constraint="participation",
                safe_deploy_band_usd=(10_000_000, 165_000_000),
                max_participation_assumed=0.05,
            ),
        )

    def _make_full_factor_exposure(self) -> FactorExposureAudit:
        return FactorExposureAudit(
            audit_status="audited",
            audit_date=_dt.date(2026, 5, 31),
            audit_script="scripts/test.py",
            audit_commit="abc1234",
            phase=3,
            proposed_role="alpha_seeker",
            n_months=114,
            alpha_annualized=0.1371,
            alpha_t_hac=9.65,
            betas={"MKT": -0.09, "SMB": 0.19, "MOM": 0.38},
            t_stats_hac={"alpha": 9.65, "MKT": -1.48, "SMB": 1.05, "MOM": 4.44},
            r_squared=0.41,
            verdict="X" * 50,
        )

    def test_audited_cost_model_full_passes(self):
        c = self._make_full_cost_model()
        assert c.audit_status == "audited"

    def test_audited_cost_model_missing_field_rejected(self):
        with pytest.raises(ValueError, match="missing"):
            CostModelAudit(
                audit_status="audited",
                audit_date=_dt.date(2026, 5, 31),
                # missing audit_script + rest
            )

    def test_audited_cost_model_short_rationale_rejected(self):
        with pytest.raises(ValueError, match="≥ 50 chars"):
            CostModelAudit(
                audit_status="audited",
                audit_date=_dt.date(2026, 5, 31),
                audit_script="x.py", audit_commit="abc",
                type="almgren_chriss",
                half_spread_bps=5.0, impact_coef=0.5,
                daily_sigma_estimate=0.015,
                universe_median_adv_usd=50_000_000,
                monthly_turnover_estimate=0.5, stress_multiplier=2.5,
                rationale="too short",
                multi_aum_sharpe_sleeve=MultiAumSharpe(at_10M=2.0, at_100M=1.9, at_1B=1.7),
                capacity=CapacityBlock(
                    hard_capacity_usd=1, binding_constraint="x",
                    safe_deploy_band_usd=(1.0, 2.0),
                    max_participation_assumed=0.05,
                ),
            )

    def test_pending_cost_model_requires_priority(self):
        with pytest.raises(ValueError, match="audit_priority"):
            CostModelAudit(audit_status="pending")

    def test_pending_cost_model_with_priority_passes(self):
        c = CostModelAudit(audit_status="pending", audit_priority="high")
        assert c.audit_status == "pending"

    def test_factor_exposure_requires_phase_1_factors(self):
        with pytest.raises(ValueError, match=r"\{.*MKT.*\}|must include"):
            FactorExposureAudit(
                audit_status="audited",
                audit_date=_dt.date(2026, 5, 31),
                audit_script="x.py", audit_commit="abc",
                phase=3, proposed_role="alpha_seeker",
                n_months=114, alpha_annualized=0.13,
                alpha_t_hac=9.6,
                betas={"SMB": 0.1},  # missing MKT, MOM
                t_stats_hac={"alpha": 9.6},
                r_squared=0.4, verdict="X" * 50,
            )

    def test_capacity_band_validation(self):
        with pytest.raises(ValueError, match=r"lo < hi"):
            CapacityBlock(
                hard_capacity_usd=100, binding_constraint="x",
                safe_deploy_band_usd=(100.0, 50.0),  # backwards
                max_participation_assumed=0.05,
            )

    def test_audit_blocks_compose(self):
        ab = AuditBlocks(
            cost_model=self._make_full_cost_model(),
            factor_exposure=self._make_full_factor_exposure(),
        )
        assert ab.cost_model.audit_status == "audited"
        assert ab.factor_exposure.proposed_role == "alpha_seeker"


# ── 6. State store ACID semantics ──────────────────────────────────────


class TestStateStore:
    def test_create_strategy_records_initial_transition(self, tmp_db):
        rec = create_strategy(
            strategy_id="test1",
            actor="test-runner",
            db_path=tmp_db,
        )
        assert rec.current_state == StrategyState.PROPOSED
        assert rec.proposed_at is not None
        history = get_transition_history("test1", db_path=tmp_db)
        assert len(history) == 1
        assert history[0].from_state is None
        assert history[0].to_state == StrategyState.PROPOSED

    def test_duplicate_strategy_id_raises(self, tmp_db):
        create_strategy(strategy_id="dup", actor="t", db_path=tmp_db)
        with pytest.raises(sqlite3.IntegrityError):
            create_strategy(strategy_id="dup", actor="t", db_path=tmp_db)

    def test_get_strategy_unknown_raises(self, tmp_db):
        with pytest.raises(KeyError):
            get_strategy("nonexistent", db_path=tmp_db)

    def test_transition_with_evidence_succeeds(self, tmp_db):
        create_strategy(strategy_id="s1", actor="t", db_path=tmp_db)
        rec = transition(
            strategy_id="s1",
            to_state=StrategyState.AUDITED,
            actor="pipeline-runner",
            reason="candidate_pipeline run abc123 PROMOTE_AS_REPLACEMENT",
            has_candidate_pipeline_run=True,
            db_path=tmp_db,
        )
        assert rec.current_state == StrategyState.AUDITED
        assert rec.audited_at is not None

    def test_transition_invalid_rolls_back(self, tmp_db):
        create_strategy(strategy_id="s1", actor="t", db_path=tmp_db)
        with pytest.raises(InvalidTransitionError):
            transition(
                strategy_id="s1",
                to_state=StrategyState.LIVE,  # invalid jump
                actor="t",
                db_path=tmp_db,
            )
        # state remains PROPOSED
        assert get_strategy("s1", db_path=tmp_db).current_state == StrategyState.PROPOSED
        # no extra transition row written
        assert len(get_transition_history("s1", db_path=tmp_db)) == 1

    def test_transition_gate_fail_rolls_back(self, tmp_db):
        create_strategy(strategy_id="s1", actor="t", db_path=tmp_db)
        with pytest.raises(GateNotMetError):
            transition(
                strategy_id="s1",
                to_state=StrategyState.AUDITED,
                actor="t",
                has_candidate_pipeline_run=False,  # gate fail
                db_path=tmp_db,
            )
        assert get_strategy("s1", db_path=tmp_db).current_state == StrategyState.PROPOSED
        assert len(get_transition_history("s1", db_path=tmp_db)) == 1

    def test_full_lifecycle_walk(self, tmp_db):
        create_strategy(strategy_id="s1", actor="t", db_path=tmp_db)
        transition(strategy_id="s1", to_state=StrategyState.AUDITED,
                   actor="pipeline", has_candidate_pipeline_run=True,
                   db_path=tmp_db)
        transition(strategy_id="s1", to_state=StrategyState.APPROVED,
                   actor="zhangxizhe", has_human_approval=True,
                   db_path=tmp_db)
        transition(strategy_id="s1", to_state=StrategyState.PAPER_TRADE,
                   actor="deploy-script", db_path=tmp_db)
        rec = get_strategy("s1", db_path=tmp_db)
        assert rec.current_state == StrategyState.PAPER_TRADE
        assert rec.approved_by == "zhangxizhe"
        assert rec.paper_trade_started is not None
        history = get_transition_history("s1", db_path=tmp_db)
        assert len(history) == 4  # create + 3 transitions

    def test_update_allocation(self, tmp_db):
        create_strategy(strategy_id="s1", actor="t", db_path=tmp_db)
        rec = update_allocation(
            strategy_id="s1",
            current_allocation_pct=0.01,
            target_allocation_pct=0.15,
            db_path=tmp_db,
        )
        assert rec.current_allocation_pct == 0.01
        assert rec.target_allocation_pct == 0.15

    def test_allocation_validation(self, tmp_db):
        create_strategy(strategy_id="s1", actor="t", db_path=tmp_db)
        with pytest.raises((sqlite3.IntegrityError, ValueError)):
            update_allocation(
                strategy_id="s1",
                current_allocation_pct=1.5,  # invalid > 1
                db_path=tmp_db,
            )

    def test_list_strategies_by_state(self, tmp_db):
        create_strategy(strategy_id="s_a", actor="t", db_path=tmp_db)
        create_strategy(strategy_id="s_b", actor="t", db_path=tmp_db)
        transition(strategy_id="s_a", to_state=StrategyState.AUDITED,
                   actor="t", has_candidate_pipeline_run=True, db_path=tmp_db)
        proposed = list_strategies(StrategyState.PROPOSED, db_path=tmp_db)
        audited = list_strategies(StrategyState.AUDITED, db_path=tmp_db)
        assert {s.strategy_id for s in proposed} == {"s_b"}
        assert {s.strategy_id for s in audited} == {"s_a"}


# ── 7. Sleeve registry ──────────────────────────────────────────────────


class _DummySleeve:
    """Minimal valid SleeveProtocol implementation for registry tests."""
    strategy_id = "dummy_test"
    library_yaml_path = Path("data/research/mechanism_library/_dummy.yaml")

    def returns(self):
        import pandas as pd
        return pd.Series([0.01, -0.02, 0.03],
                         index=pd.date_range("2024-01-31", periods=3, freq="ME"))

    def audit_blocks(self) -> AuditBlocks:
        return AuditBlocks(
            cost_model=CostModelAudit(audit_status="pending", audit_priority="low"),
            factor_exposure=FactorExposureAudit(audit_status="pending", audit_priority="low"),
        )


class TestSleeveRegistry:
    def test_register_and_get(self):
        register_sleeve("dummy_test")(_DummySleeve)
        cls = get_sleeve_class("dummy_test")
        assert cls is _DummySleeve
        instance = get_sleeve("dummy_test")
        assert instance.strategy_id == "dummy_test"

    def test_duplicate_strict_raises(self):
        register_sleeve("dummy_test")(_DummySleeve)
        with pytest.raises(SleeveAlreadyRegisteredError):
            register_sleeve("dummy_test", strict=True)(_DummySleeve)

    def test_duplicate_nonstrict_replaces(self):
        register_sleeve("dummy_test")(_DummySleeve)

        class _Alt(_DummySleeve):
            pass

        register_sleeve("dummy_test", strict=False)(_Alt)
        assert get_sleeve_class("dummy_test") is _Alt

    def test_missing_attribute_rejected(self):
        class _Incomplete:
            strategy_id = "x"
            # missing library_yaml_path, returns, audit_blocks

        with pytest.raises(TypeError, match="missing required attribute"):
            register_sleeve("incomplete")(_Incomplete)

    def test_unknown_get_raises(self):
        with pytest.raises(KeyError):
            get_sleeve("nonexistent_sleeve")

    def test_list_registered(self):
        register_sleeve("dummy_test")(_DummySleeve)
        names = list_registered_sleeves()
        assert "dummy_test" in names
