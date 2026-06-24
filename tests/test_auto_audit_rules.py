"""
tests/test_auto_audit_rules.py — R-1.B contradiction rules.

Spot-check on the most consequential rules. Per-rule adversarial poison tests
for n_trials math + universe drift + ALIGNMENT_SURFACE — the rules whose
silent failure has the worst blast radius.
"""
import pytest


def test_critical_rules_count():
    """Sanity: registry has expected rule count after each cumulative sprint."""
    from engine.auto_audit_rules import CRITICAL_RULES, WEEKLY_RULES
    # 8 from R-1.B.1+B.2 + 2 from B.3 (#14 + #10b grep) + 1 from R-1.E (path_consistency)
    # + 1 from P-LAB (rule_factor_lab_state_consistent, 2026-05-08)
    # + 1 from Factor Library v1 (rule_factor_lab_no_factor_library_import, 2026-05-09)
    # + 2 from ETF Holdings Monitor (rule_etf_holdings_cap_clamp_bounds +
    #     rule_etf_holdings_no_llm_in_eval, 2026-05-08, spec id=49)
    # + 3 from Factor Ensemble v1 (rule_factor_ensemble_no_lookahead +
    #     rule_factor_ensemble_no_param_tuning +
    #     rule_factor_ensemble_baseline_reproducibility, 2026-05-09, spec id=50 §4.7)
    # + 1 from MS-7 (rule_sleeve_id_integrity, 2026-05-10)
    # + 2 from FOMC Override (rule_fomc_override_clamp_bounds +
    #     rule_fomc_override_no_llm_in_eval, 2026-05-12 unlock, spec id=48)
    assert len(CRITICAL_RULES) == 21
    # 1 from B.1 (heartbeat) + 5 from B.3 + 1 from Wave 5 (capability_vs_data_congruence)
    # + 1 from Phase 1 c (rule_llm_removal_test_doc_exists, 2026-05-12)
    # + 1 from Watchdog Phase 3 (rule_watchdog_auto_repair_no_raw_sql, 2026-05-12)
    # + 1 from Watchdog Phase 4 (rule_watchdog_halt_flag_not_stuck, 2026-05-12)
    # + 1 from Watchdog Phase 5 (rule_watchdog_runs_daily, 2026-05-13)
    assert len(WEEKLY_RULES) == 11


def test_factor_lab_no_factor_library_import_rule_clean():
    """Default state: factor_lab/ does not import factor_library (one-way dep
    per spec_factor_library_v1.md §4.1 + spec_factor_lab.md boundary)."""
    from engine.auto_audit_rules import rule_factor_lab_no_factor_library_import
    assert rule_factor_lab_no_factor_library_import() is None


def test_factor_lab_no_factor_library_import_rule_detects_violation(tmp_path, monkeypatch):
    """Inject a synthetic violation: write a temp .py inside factor_lab/ that
    imports factor_library, run rule, expect HIGH severity finding. Cleanup
    after."""
    import pathlib, importlib
    from engine import auto_audit_rules
    factor_lab_dir = pathlib.Path(__file__).resolve().parent.parent / "engine" / "factor_lab"
    poison = factor_lab_dir / "_test_poison_.py"
    poison.write_text(
        "from engine.factor_library import FACTOR_REGISTRY  # synthetic violation\n",
        encoding="utf-8",
    )
    try:
        result = auto_audit_rules.rule_factor_lab_no_factor_library_import()
        assert result is not None, "rule failed to detect engine.factor_library import in factor_lab/"
        assert result["severity"] == "HIGH"
        violations = result["snapshot"]["violations"]
        assert any("_test_poison_" in v["file"] for v in violations), (
            f"expected _test_poison_.py in violations; got {violations}"
        )
    finally:
        poison.unlink(missing_ok=True)


def test_path_consistency_rule_clean():
    """Default state: proposer↔gate FORBIDDEN/FLAGGED match."""
    from engine.auto_audit_rules import rule_path_consistency
    assert rule_path_consistency() is None


def test_path_consistency_detects_drift():
    """Inject mismatch → rule reports HIGH severity."""
    from engine import auto_audit_proposer
    from engine.auto_audit_rules import rule_path_consistency
    orig = auto_audit_proposer.LLM_FORBIDDEN_PATHS
    try:
        auto_audit_proposer.LLM_FORBIDDEN_PATHS = orig + ("engine/_synthetic_drift_.py",)
        result = rule_path_consistency()
        assert result is not None
        assert result["severity"] == "HIGH"
        issues = result["snapshot"]["issues"]
        assert any("FORBIDDEN_PATHS" in i["list"] for i in issues)
    finally:
        auto_audit_proposer.LLM_FORBIDDEN_PATHS = orig


def test_n_trials_math_consistency_clean():
    """Project's real spec_registry should be self-consistent at the
    n_trials math level."""
    from engine.auto_audit_rules import rule_effective_n_trials_math_consistency
    result = rule_effective_n_trials_math_consistency()
    assert result is None, f"n_trials math drift on real DB: {result}"


def test_n_trials_math_detects_poison(tmp_path):
    """Mutate stored n_trials_contributed → rule emits HIGH."""
    from engine.preregistration import register_spec
    from engine.memory import SessionFactory, SpecRegistry
    from engine.auto_audit_rules import rule_effective_n_trials_math_consistency
    f = tmp_path / "spec_poison.md"
    f.write_text("# poison test", encoding="utf-8")
    sid = register_spec(str(f), retro=True)  # base 0
    with SessionFactory() as s:
        r = s.get(SpecRegistry, sid)
        r.n_trials_contributed = 99   # poison (expected 0 for retro)
        s.commit()
    try:
        result = rule_effective_n_trials_math_consistency()
        assert result is not None
        assert result["severity"] == "HIGH"
        issues = result["snapshot"]["issues"]
        # Find the poisoned row
        poisoned = [i for i in issues if i.get("spec_path") == r.spec_path]
        assert poisoned, f"Poisoned spec not in issues: {issues}"
    finally:
        # Revert
        with SessionFactory() as s:
            s.get(SpecRegistry, sid).n_trials_contributed = 0
            s.commit()


def test_production_signal_vs_falsification_clean():
    """Default PRODUCTION_SIGNAL='ql01_bab' is NOT in REJECTED set."""
    from engine.auto_audit_rules import rule_production_signal_vs_falsification_chain
    assert rule_production_signal_vs_falsification_chain() is None


def test_production_signal_vs_falsification_catches_tsmom(monkeypatch):
    """Switching PRODUCTION_SIGNAL to a rejected value → HIGH finding."""
    import engine.config as cfg
    from engine.auto_audit_rules import rule_production_signal_vs_falsification_chain
    monkeypatch.setattr(cfg, "PRODUCTION_SIGNAL", "tsmom")
    result = rule_production_signal_vs_falsification_chain()
    assert result is not None
    assert result["severity"] == "HIGH"
    assert result["snapshot"]["PRODUCTION_SIGNAL"] == "tsmom"


def test_param_alignment_clean():
    """Default config matches ALIGNMENT_SURFACE."""
    from engine.auto_audit_rules import rule_backtest_vs_production_param_alignment
    assert rule_backtest_vs_production_param_alignment() is None


def test_param_alignment_detects_target_vol_drift(monkeypatch):
    import engine.config as cfg
    from engine.auto_audit_rules import rule_backtest_vs_production_param_alignment
    monkeypatch.setattr(cfg, "TARGET_VOL", 0.99)
    result = rule_backtest_vs_production_param_alignment()
    assert result is not None
    assert result["severity"] == "HIGH"
    target_diffs = [d for d in result["snapshot"]["diffs"]
                    if d["param"] == "TARGET_VOL"]
    assert target_diffs and target_diffs[0]["actual"] == 0.99


def test_universe_drift_self_initializes():
    """First call writes baseline_hash to SystemConfig; returns no finding."""
    from engine.memory import SessionFactory
    from engine.db_models import SystemConfig
    from engine.auto_audit_rules import rule_universe_drift_vs_registered

    # Wipe any existing baseline so we exercise self-init
    with SessionFactory() as s:
        s.query(SystemConfig).filter(
            SystemConfig.key.like("auto_audit.universe%")
        ).delete()
        s.commit()

    result = rule_universe_drift_vs_registered()
    assert result is None  # self-init returns clean
    with SessionFactory() as s:
        baseline = s.query(SystemConfig).filter_by(
            key="auto_audit.universe_baseline_hash"
        ).first()
        assert baseline is not None
        assert len(baseline.value) == 64  # sha256 hex


def test_universe_drift_detects_hash_change():
    """Second call after baseline mutation → HIGH finding."""
    from engine.memory import SessionFactory
    from engine.db_models import SystemConfig
    from engine.auto_audit_rules import rule_universe_drift_vs_registered

    # Ensure baseline exists
    rule_universe_drift_vs_registered()
    with SessionFactory() as s:
        baseline = s.query(SystemConfig).filter_by(
            key="auto_audit.universe_baseline_hash"
        ).first()
        baseline.value = "0" * 64  # poison
        s.commit()
    try:
        result = rule_universe_drift_vs_registered()
        assert result is not None
        assert result["severity"] == "HIGH"
    finally:
        # Revert: re-init by deleting + re-running
        with SessionFactory() as s:
            s.query(SystemConfig).filter(
                SystemConfig.key.like("auto_audit.universe%")
            ).delete()
            s.commit()


def test_db_schema_vs_orm_clean():
    """If migrations are current, schema vs ORM should match (or only have
    pre-known drift). This test asserts no NEW drift relative to a known
    baseline of 0 issues — which is the project's current state."""
    from engine.auto_audit_rules import rule_db_schema_vs_orm_consistency
    result = rule_db_schema_vs_orm_consistency()
    # The fixture DB is fresh (init_db created tables matching ORM exactly);
    # any drift here means a real schema-vs-ORM bug.
    assert result is None, f"schema drift detected: {result}"
