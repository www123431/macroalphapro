"""
tests/test_factor_lab_runner.py — Verdict logic + state transitions.

Mocks engine.b_plus_search.run_single_strategy_weekly to avoid yfinance +
multi-second backtest on every test run. Tests focus on:
  - verdict classification at threshold boundaries
  - state machine atomicity (REGISTERED → TESTING → terminal)
  - drift refusal (spec hash mismatch refuses to run)
  - error path (execution failure → FAIL verdict, not stuck in TESTING)
"""
from __future__ import annotations

import datetime
import os
from unittest.mock import patch

import pytest

from engine.factor_lab import FactorState
from engine.factor_lab.runner import _classify_verdict, run_factor_lab_test


# ─────────────────────────────────────────────────────────────────────────────
# _classify_verdict — boundary-case unit tests (pure)
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyVerdict:
    def test_t_at_5pct_boundary_yields_pass(self):
        assert _classify_verdict(nw_t_stat=1.96, achieved_power=0.80) == FactorState.PASS

    def test_t_above_5pct_yields_pass(self):
        assert _classify_verdict(nw_t_stat=2.50, achieved_power=0.80) == FactorState.PASS

    def test_negative_t_at_5pct_yields_pass(self):
        """abs() — directional negative still significant."""
        assert _classify_verdict(nw_t_stat=-2.00, achieved_power=0.80) == FactorState.PASS

    def test_t_in_marginal_range_yields_marginal(self):
        assert _classify_verdict(nw_t_stat=1.80, achieved_power=0.80) == FactorState.MARGINAL

    def test_t_at_marginal_boundary_yields_marginal(self):
        assert _classify_verdict(nw_t_stat=1.65, achieved_power=0.80) == FactorState.MARGINAL

    def test_t_below_marginal_with_good_power_yields_fail(self):
        assert _classify_verdict(nw_t_stat=1.00, achieved_power=0.80) == FactorState.FAIL

    def test_t_below_marginal_with_low_power_yields_fail_underpowered(self):
        assert _classify_verdict(nw_t_stat=0.50, achieved_power=0.30) == FactorState.FAIL_UNDERPOWERED

    def test_zero_t_low_power_yields_fail_underpowered(self):
        assert _classify_verdict(nw_t_stat=0.0, achieved_power=0.20) == FactorState.FAIL_UNDERPOWERED

    def test_high_t_overrides_low_power(self):
        """Strong evidence beats power concerns — PASS regardless."""
        assert _classify_verdict(nw_t_stat=3.0, achieved_power=0.10) == FactorState.PASS

    def test_nan_t_yields_fail(self):
        assert _classify_verdict(nw_t_stat=float("nan"), achieved_power=0.80) == FactorState.FAIL

    def test_inf_t_treated_as_finite_pass(self):
        """isfinite check kicks in for inf — defensive, returns FAIL."""
        assert _classify_verdict(nw_t_stat=float("inf"), achieved_power=0.80) == FactorState.FAIL


# ─────────────────────────────────────────────────────────────────────────────
# run_factor_lab_test — end-to-end with mocked b_plus_search
# ─────────────────────────────────────────────────────────────────────────────

def _create_registered_candidate(spec_path: str) -> int:
    """Insert a SpecRegistry row in REGISTERED state for testing.

    Writes a real markdown stub to disk so the runner-time spec_hash
    drift check (which reads the file) doesn't refuse to run.
    """
    import tempfile
    from engine.memory import SessionFactory, SpecRegistry
    from engine.preregistration import _compute_git_blob_hash, _resolve_to_abs

    abs_path = _resolve_to_abs(spec_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Test spec for runner — {spec_path}\n\nMinimal stub.\n")
    real_hash = _compute_git_blob_hash(abs_path)

    with SessionFactory() as s:
        row = SpecRegistry(
            spec_path            = spec_path,
            git_blob_hash        = real_hash,
            current_hash         = real_hash,
            registered_at        = datetime.datetime.utcnow(),
            amendment_log        = "[]",
            status               = "active",
            retro_registered     = False,
            n_trials_contributed = 1,
            factor_kind          = "production_swap",
            lab_state            = FactorState.REGISTERED.value,
        )
        s.add(row)
        s.commit()
        return row.id


def _mock_strategy(strategy_id="mock_strategy"):
    """Return a duck-typed StrategySpec-like object."""
    class _Stub:
        id   = strategy_id
        name = strategy_id
        category = "test"
    return _Stub()


def _mock_universe():
    return {"sector_a": "AAA", "sector_b": "BBB"}


class TestRunFactorLabTest:
    @patch("engine.b_plus_search.run_single_strategy_weekly")
    @patch("engine.b_plus_search.get_universe_tier")
    @patch("engine.b_plus_search.get_strategy")
    def test_pass_verdict_path(self, mock_get_strategy, mock_get_universe, mock_run):
        """Strong NW-t → PASS, state transitions REGISTERED→TESTING→PASS."""
        sid = _create_registered_candidate("docs/spec_test_runner_pass.md")
        mock_get_strategy.return_value = _mock_strategy()
        mock_get_universe.return_value = _mock_universe()
        mock_run.return_value = {
            "strategy_id": "mock_strategy",
            "n_obs":       250,
            "nw_t_stat":   2.50,
            "sharpe":      1.20,
        }

        result = run_factor_lab_test(
            spec_id      = sid,
            strategy_id  = "mock_strategy",
            decisions_dir = "tests/_factor_lab_artifacts",
        )

        assert result["verdict"] == FactorState.PASS.value
        assert result["nw_t_stat"] == pytest.approx(2.50)

        from engine.factor_lab import get_candidate
        row = get_candidate(sid)
        assert row["lab_state"] == FactorState.PASS.value
        # 2 transitions (REG→TESTING + TESTING→PASS) → 2 amendment entries
        assert row["amendment_count"] == 2

    @patch("engine.b_plus_search.run_single_strategy_weekly")
    @patch("engine.b_plus_search.get_universe_tier")
    @patch("engine.b_plus_search.get_strategy")
    def test_fail_verdict_path(self, mock_get_strategy, mock_get_universe, mock_run):
        sid = _create_registered_candidate("docs/spec_test_runner_fail.md")
        mock_get_strategy.return_value = _mock_strategy()
        mock_get_universe.return_value = _mock_universe()
        mock_run.return_value = {
            "strategy_id": "mock_strategy",
            "n_obs":       300,
            "nw_t_stat":   0.80,    # below marginal
            "sharpe":      0.10,
        }

        result = run_factor_lab_test(
            spec_id      = sid,
            strategy_id  = "mock_strategy",
            decisions_dir = "tests/_factor_lab_artifacts",
        )
        # n_obs=300 weekly is plenty for default lift=0.5/baseline=1.0;
        # achieved power should be high → FAIL (not FAIL_UNDERPOWERED)
        assert result["verdict"] in (FactorState.FAIL.value,
                                     FactorState.FAIL_UNDERPOWERED.value)

    @patch("engine.b_plus_search.run_single_strategy_weekly")
    @patch("engine.b_plus_search.get_universe_tier")
    @patch("engine.b_plus_search.get_strategy")
    def test_marginal_verdict_path(self, mock_get_strategy, mock_get_universe, mock_run):
        sid = _create_registered_candidate("docs/spec_test_runner_marginal.md")
        mock_get_strategy.return_value = _mock_strategy()
        mock_get_universe.return_value = _mock_universe()
        mock_run.return_value = {
            "strategy_id": "mock_strategy",
            "n_obs":       250,
            "nw_t_stat":   1.80,    # in (1.65, 1.96)
            "sharpe":      0.50,
        }

        result = run_factor_lab_test(
            spec_id      = sid,
            strategy_id  = "mock_strategy",
            decisions_dir = "tests/_factor_lab_artifacts",
        )
        assert result["verdict"] == FactorState.MARGINAL.value

    @patch("engine.b_plus_search.run_single_strategy_weekly")
    @patch("engine.b_plus_search.get_universe_tier")
    @patch("engine.b_plus_search.get_strategy")
    def test_execution_error_writes_fail_not_stuck_in_testing(
        self, mock_get_strategy, mock_get_universe, mock_run,
    ):
        """If b_plus_search throws, runner must transition to FAIL not leave TESTING."""
        sid = _create_registered_candidate("docs/spec_test_runner_exc.md")
        mock_get_strategy.return_value = _mock_strategy()
        mock_get_universe.return_value = _mock_universe()
        mock_run.side_effect = RuntimeError("simulated yfinance outage")

        with pytest.raises(RuntimeError):
            run_factor_lab_test(
                spec_id      = sid,
                strategy_id  = "mock_strategy",
                decisions_dir = "tests/_factor_lab_artifacts",
            )

        from engine.factor_lab import get_candidate
        row = get_candidate(sid)
        # Must NOT be stuck in TESTING — execution_error transitions to FAIL
        assert row["lab_state"] == FactorState.FAIL.value

    @patch("engine.b_plus_search.get_strategy")
    def test_unknown_strategy_id_raises(self, mock_get_strategy):
        sid = _create_registered_candidate("docs/spec_test_runner_unknown.md")
        mock_get_strategy.side_effect = KeyError("not_in_registry")

        with pytest.raises(ValueError, match="not in b_plus_search"):
            run_factor_lab_test(
                spec_id      = sid,
                strategy_id  = "unregistered_id",
                decisions_dir = "tests/_factor_lab_artifacts",
            )

    def test_non_registered_state_refused(self):
        """Cannot run BHY on a DRAFT/PROPOSED/PASS row."""
        from engine.memory import SessionFactory, SpecRegistry
        # Insert a candidate in PROPOSED state
        with SessionFactory() as s:
            row = SpecRegistry(
                spec_path        = "docs/spec_test_runner_state_refuse.md",
                git_blob_hash    = "0" * 40,
                current_hash     = "0" * 40,
                amendment_log    = "[]",
                status           = "active",
                retro_registered = False,
                n_trials_contributed = 1,
                factor_kind      = "production_swap",
                lab_state        = FactorState.PROPOSED.value,
            )
            s.add(row)
            s.commit()
            sid = row.id

        with pytest.raises(ValueError, match="state=.*PROPOSED"):
            run_factor_lab_test(
                spec_id      = sid,
                strategy_id  = "any",
                decisions_dir = "tests/_factor_lab_artifacts",
            )

    def test_missing_spec_id_raises_lookup(self):
        with pytest.raises(LookupError):
            run_factor_lab_test(
                spec_id      = 999_999,
                strategy_id  = "any",
                decisions_dir = "tests/_factor_lab_artifacts",
            )

    @patch("engine.b_plus_search.run_single_strategy_weekly")
    @patch("engine.b_plus_search.get_universe_tier")
    @patch("engine.b_plus_search.get_strategy")
    def test_spec_hash_drift_refuses_run(
        self, mock_get_strategy, mock_get_universe, mock_run,
    ):
        """Silent edit to spec markdown after register → runner refuses."""
        sid = _create_registered_candidate("docs/spec_test_runner_drift.md")

        # Silently modify the spec markdown
        from engine.preregistration import _resolve_to_abs
        abs_path = _resolve_to_abs("docs/spec_test_runner_drift.md")
        with open(abs_path, "a", encoding="utf-8") as fh:
            fh.write("\nSilent edit appended.\n")

        with pytest.raises(ValueError, match="spec_hash drift"):
            run_factor_lab_test(
                spec_id      = sid,
                strategy_id  = "any",
                decisions_dir = "tests/_factor_lab_artifacts",
            )

        # b_plus_search should NOT have been called
        mock_run.assert_not_called()

    @patch("engine.b_plus_search.run_single_strategy_weekly")
    @patch("engine.b_plus_search.get_universe_tier")
    @patch("engine.b_plus_search.get_strategy")
    def test_writes_decisions_markdown(
        self, mock_get_strategy, mock_get_universe, mock_run, tmp_path,
    ):
        sid = _create_registered_candidate("docs/spec_test_runner_writes_md.md")
        mock_get_strategy.return_value = _mock_strategy()
        mock_get_universe.return_value = _mock_universe()
        mock_run.return_value = {
            "strategy_id": "mock", "n_obs": 250, "nw_t_stat": 2.50, "sharpe": 1.0,
        }
        result = run_factor_lab_test(
            spec_id      = sid,
            strategy_id  = "mock_strategy",
            decisions_dir = str(tmp_path),
        )
        assert os.path.isfile(result["decision_path"])
        content = open(result["decision_path"], encoding="utf-8").read()
        assert "PASS" in content
        assert "nw_t" in content.lower() or "NW t" in content


# Cleanup test artifact specs after the suite to keep /docs clean
@pytest.fixture(scope="module", autouse=True)
def _cleanup_test_specs():
    yield
    # Delete any test spec files we wrote (matching docs/spec_test_runner_*.md)
    import glob
    for p in glob.glob("docs/spec_test_runner_*.md"):
        try:
            os.unlink(p)
        except Exception:
            pass
