"""tests/test_promote_candidate.py — SLM Phase 1 unit tests.

Tests the library_yaml_renderer + promote_candidate orchestrator:
  1. Pydantic AuditBlocks → YAML round-trip preserves values
  2. Scaffold has all schema-required fields filled
  3. Human-curated fields contain TODO placeholders
  4. Dry-run mode does not write or touch DB
  5. File-write rollback on validator failure (mocked)
  6. State store row + transition created on success
"""
from __future__ import annotations

import datetime as _dt
import tempfile
from pathlib import Path

import pytest
import yaml

from engine.research.library_yaml_renderer import (
    StrategyIdentity,
    render_library_yaml_scaffold,
    write_library_yaml,
    yaml_to_string,
)
from engine.research.promote_candidate import (
    PromotionResult,
    promote_candidate,
)
from engine.research.strategy_lifecycle import (
    AuditBlocks,
    CapacityBlock,
    CostModelAudit,
    FactorExposureAudit,
    MultiAumSharpe,
    StrategyState,
)
from engine.research.strategy_state_store import (
    get_strategy,
    get_transition_history,
    reset_db_for_test,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_library_dir(tmp_path: Path) -> Path:
    d = tmp_path / "mechanism_library"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test_phase1.db"
    yield db
    reset_db_for_test(db)


@pytest.fixture
def full_audit_blocks() -> AuditBlocks:
    return AuditBlocks(
        cost_model=CostModelAudit(
            audit_status="audited",
            audit_date=_dt.date(2026, 5, 31),
            audit_script="scripts/audit_x.py",
            audit_commit="abc1234",
            type="almgren_chriss",
            half_spread_bps=5.0, impact_coef=0.5, daily_sigma_estimate=0.015,
            universe_median_adv_usd=50_000_000, n_positions_typical=110,
            monthly_turnover_estimate=0.50, stress_multiplier=2.5,
            rationale="X" * 60,
            multi_aum_sharpe_sleeve=MultiAumSharpe(
                at_10M=2.0, at_100M=1.9, at_1B=1.7,
            ),
            capacity=CapacityBlock(
                hard_capacity_usd=550_000_000,
                binding_constraint="x",
                safe_deploy_band_usd=(10_000_000, 165_000_000),
                max_participation_assumed=0.05,
            ),
        ),
        factor_exposure=FactorExposureAudit(
            audit_status="audited",
            audit_date=_dt.date(2026, 5, 31),
            audit_script="scripts/audit_x.py",
            audit_commit="abc1234",
            phase=3, proposed_role="alpha_seeker", n_months=114,
            alpha_annualized=0.1371, alpha_t_hac=9.65,
            betas={"MKT": -0.09, "SMB": 0.19, "MOM": 0.38},
            t_stats_hac={"alpha": 9.65, "MKT": -1.48, "SMB": 1.05, "MOM": 4.44},
            r_squared=0.41,
            verdict="X" * 50,
        ),
    )


@pytest.fixture
def identity() -> StrategyIdentity:
    return StrategyIdentity(
        strategy_id="post_earnings_drift_pit_sn_test",
        family="earnings_underreaction",
        parent_family="equity_factor",
        purpose="deploy_replacement",
        relation_to_parent="REPLACEMENT",
        parent_strategy_id="post_earnings_drift",
        canonical_paper_id="bernard_thomas_1989_jar",
    )


# ── Renderer tests ──────────────────────────────────────────────────────


class TestRenderer:
    def test_scaffold_has_required_sections(self, identity, full_audit_blocks):
        s = render_library_yaml_scaffold(
            identity=identity, audit_blocks=full_audit_blocks,
            last_audited=_dt.date(2026, 5, 31),
        )
        for key in ("id", "family", "cost_model", "factor_exposure",
                    "audit_checklist_passed", "audit_signature",
                    "mechanism_economics"):
            assert key in s

    def test_cost_model_round_trip(self, identity, full_audit_blocks):
        s = render_library_yaml_scaffold(
            identity=identity, audit_blocks=full_audit_blocks,
            last_audited=_dt.date(2026, 5, 31),
        )
        assert s["cost_model"]["audit_status"] == "audited"
        assert s["cost_model"]["multi_aum_sharpe_sleeve"]["at_10M"] == 2.0
        assert s["cost_model"]["capacity"]["hard_capacity_usd"] == 550_000_000

    def test_factor_exposure_round_trip(self, identity, full_audit_blocks):
        s = render_library_yaml_scaffold(
            identity=identity, audit_blocks=full_audit_blocks,
            last_audited=_dt.date(2026, 5, 31),
        )
        assert s["factor_exposure"]["alpha_t_hac"] == 9.65
        assert s["factor_exposure"]["proposed_role"] == "alpha_seeker"
        assert s["factor_exposure"]["betas"]["MOM"] == 0.38

    def test_human_fields_have_todo_placeholders(self, identity, full_audit_blocks):
        s = render_library_yaml_scaffold(
            identity=identity, audit_blocks=full_audit_blocks,
            last_audited=_dt.date(2026, 5, 31),
        )
        # Required-but-needs-human fields must signal incompleteness
        assert "TODO" in s["mechanism_economics"]
        assert any("TODO" in c for c in s["mechanism_break_conditions"])
        assert s["audit_signature"] == "pending"  # NEVER auto-flip

    def test_yaml_round_trip_via_pyyaml(self, identity, full_audit_blocks, tmp_path):
        s = render_library_yaml_scaffold(
            identity=identity, audit_blocks=full_audit_blocks,
            last_audited=_dt.date(2026, 5, 31),
        )
        yaml_str = yaml_to_string(s)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["id"] == identity.strategy_id
        assert parsed["cost_model"]["audit_status"] == "audited"
        assert parsed["factor_exposure"]["proposed_role"] == "alpha_seeker"

    def test_write_refuses_existing_no_overwrite(self, identity, full_audit_blocks,
                                                  tmp_path):
        p = tmp_path / "exists.yaml"
        p.write_text("existing content", encoding="utf-8")
        s = render_library_yaml_scaffold(
            identity=identity, audit_blocks=full_audit_blocks,
            last_audited=_dt.date(2026, 5, 31),
        )
        with pytest.raises(FileExistsError):
            write_library_yaml(path=p, scaffold=s, overwrite=False)
        # Original content preserved
        assert p.read_text(encoding="utf-8") == "existing content"


# ── Promotion orchestrator tests ───────────────────────────────────────


class TestPromoteCandidate:
    def test_dry_run_does_not_write(self, identity, full_audit_blocks,
                                    tmp_library_dir, tmp_db):
        result = promote_candidate(
            identity=identity,
            audit_blocks=full_audit_blocks,
            pipeline_run_id="test_run_123",
            actor="pytest",
            library_dir=tmp_library_dir,
            db_path=tmp_db,
            dry_run=True,
            skip_validators=True,
        )
        assert result.dry_run is True
        assert result.yaml_preview != ""
        # File NOT written
        assert not (tmp_library_dir / f"{identity.strategy_id}.yaml").exists()
        # State store row NOT created
        with pytest.raises(KeyError):
            get_strategy(identity.strategy_id, db_path=tmp_db)

    def test_full_promotion_writes_file_and_state_row(self, identity,
                                                       full_audit_blocks,
                                                       tmp_library_dir, tmp_db):
        result = promote_candidate(
            identity=identity,
            audit_blocks=full_audit_blocks,
            pipeline_run_id="test_run_456",
            actor="pytest",
            library_dir=tmp_library_dir,
            db_path=tmp_db,
            skip_validators=True,
            git_sha="abc1234",
        )
        assert result.error is None
        assert result.validators_passed is True
        # File written
        yaml_path = tmp_library_dir / f"{identity.strategy_id}.yaml"
        assert yaml_path.exists()
        # State store row created + transitioned
        rec = get_strategy(identity.strategy_id, db_path=tmp_db)
        assert rec.current_state == StrategyState.AUDITED
        assert rec.candidate_pipeline_run_id == "test_run_456"
        # History has create + transition
        hist = get_transition_history(identity.strategy_id, db_path=tmp_db)
        assert len(hist) == 2
        assert hist[0].from_state is None
        assert hist[1].to_state == StrategyState.AUDITED

    def test_existing_file_no_overwrite_returns_error(self, identity,
                                                       full_audit_blocks,
                                                       tmp_library_dir, tmp_db):
        # Pre-place file
        yaml_path = tmp_library_dir / f"{identity.strategy_id}.yaml"
        yaml_path.write_text("dont overwrite me", encoding="utf-8")
        result = promote_candidate(
            identity=identity,
            audit_blocks=full_audit_blocks,
            pipeline_run_id="x", actor="pytest",
            library_dir=tmp_library_dir, db_path=tmp_db,
            overwrite_existing=False,
            skip_validators=True,
        )
        assert result.error is not None
        assert "already exists" in result.error
        # File preserved unchanged
        assert yaml_path.read_text(encoding="utf-8") == "dont overwrite me"
        # No state-store row
        with pytest.raises(KeyError):
            get_strategy(identity.strategy_id, db_path=tmp_db)

    def test_validator_failure_rolls_back(self, identity, full_audit_blocks,
                                          tmp_library_dir, tmp_db, monkeypatch):
        # Force validators to fail
        from engine.research import promote_candidate as pc_mod
        monkeypatch.setattr(
            pc_mod, "_run_all_validators",
            lambda: (False, {"forced": "FAIL: forced failure"}),
        )
        result = promote_candidate(
            identity=identity,
            audit_blocks=full_audit_blocks,
            pipeline_run_id="x", actor="pytest",
            library_dir=tmp_library_dir, db_path=tmp_db,
        )
        assert result.validators_passed is False
        assert result.rollback_performed is True
        # File should be gone (rolled back)
        assert not (tmp_library_dir / f"{identity.strategy_id}.yaml").exists()
        # No state-store row
        with pytest.raises(KeyError):
            get_strategy(identity.strategy_id, db_path=tmp_db)
