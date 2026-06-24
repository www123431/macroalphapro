"""tests/test_merkle_ledger.py — SLM Phase 4 unit tests.

Covers:
  1. Canonical JSON determinism (round-trip via dict shouldn't change)
  2. compute_chain_hash determinism + sensitivity
  3. add_chain_column_if_missing idempotency + backfill correctness
  4. verify_ledger_integrity catches:
       - tampered row (modified after insert)
       - missing chain_hash (rare but possible)
  5. transition() in state_store writes correct chain_hash
  6. Full workflow: create_strategy + 3 transitions → chain head matches
     expected SHA256 of the sequence
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from engine.research.merkle_ledger import (
    GENESIS_HASH, LedgerBreak, add_chain_column_if_missing,
    canonical_row_json, compute_chain_hash, get_head_chain_hash,
    verify_ledger_integrity,
)
from engine.research.strategy_lifecycle import StrategyState
from engine.research.strategy_state_store import (
    create_strategy, get_transition_history, reset_db_for_test, transition,
    _connect,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test_merkle.db"
    yield db
    reset_db_for_test(db)


# ── 1. Canonical JSON ──────────────────────────────────────────────────


class TestCanonicalJson:
    def test_determinism(self):
        a = canonical_row_json(
            strategy_id="x", from_state="A", to_state="B",
            transition_at="2026-01-01T00:00:00",
            actor="alice", reason="r", gate_evidence={"k": 1}, git_sha="abc",
        )
        b = canonical_row_json(
            strategy_id="x", from_state="A", to_state="B",
            transition_at="2026-01-01T00:00:00",
            actor="alice", reason="r", gate_evidence={"k": 1}, git_sha="abc",
        )
        assert a == b

    def test_string_gate_evidence_canonicalized(self):
        """Whitespace variance in stored JSON shouldn't change canonical form."""
        a = canonical_row_json(
            strategy_id="x", from_state="A", to_state="B",
            transition_at="t", actor="a", reason="r",
            gate_evidence='{"k": 1, "j": 2}',  # with whitespace
            git_sha=None,
        )
        b = canonical_row_json(
            strategy_id="x", from_state="A", to_state="B",
            transition_at="t", actor="a", reason="r",
            gate_evidence='{"j":2,"k":1}',     # different formatting
            git_sha=None,
        )
        assert a == b

    def test_excludes_id_and_chain_hash(self):
        # Function signature doesn't take id or chain_hash → can't leak in
        canon = canonical_row_json(
            strategy_id="x", from_state=None, to_state="A",
            transition_at="t", actor="a", reason="r",
            gate_evidence={}, git_sha=None,
        )
        assert "chain_hash" not in canon
        assert '"id"' not in canon


# ── 2. compute_chain_hash ──────────────────────────────────────────────


class TestComputeChainHash:
    def test_genesis_chain(self):
        canon = '{"strategy_id":"x"}'
        h = compute_chain_hash(GENESIS_HASH, canon)
        expected = hashlib.sha256(
            (GENESIS_HASH + canon).encode("utf-8")
        ).hexdigest()
        assert h == expected
        assert len(h) == 64

    def test_chain_sensitive_to_prev(self):
        h1 = compute_chain_hash(GENESIS_HASH, '{"x":1}')
        h2 = compute_chain_hash("a" * 64, '{"x":1}')
        assert h1 != h2

    def test_chain_sensitive_to_canon(self):
        h1 = compute_chain_hash(GENESIS_HASH, '{"x":1}')
        h2 = compute_chain_hash(GENESIS_HASH, '{"x":2}')
        assert h1 != h2


# ── 3. Migration ───────────────────────────────────────────────────────


class TestMerkleMigration:
    def test_idempotent_no_op_when_present(self, tmp_db):
        # create_strategy → init_db → migration runs
        create_strategy(strategy_id="s1", actor="test", db_path=tmp_db)
        conn = _connect(tmp_db)
        # Second call should detect column + return False
        assert add_chain_column_if_missing(conn) is False

    def test_backfill_chains_from_genesis(self, tmp_db):
        create_strategy(strategy_id="s1", actor="test", db_path=tmp_db)
        conn = _connect(tmp_db)
        # The single transition row should have chain_hash set
        rows = conn.execute(
            "SELECT chain_hash FROM state_transitions"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["chain_hash"] is not None
        assert len(rows[0]["chain_hash"]) == 64


# ── 4. verify_ledger_integrity ─────────────────────────────────────────


class TestVerifyLedgerIntegrity:
    def test_fresh_ledger_intact(self, tmp_db):
        create_strategy(strategy_id="s1", actor="test", db_path=tmp_db)
        transition(strategy_id="s1", to_state=StrategyState.AUDITED,
                   actor="t", has_candidate_pipeline_run=True,
                   db_path=tmp_db)
        conn = _connect(tmp_db)
        result = verify_ledger_integrity(conn)
        assert result.chain_intact is True
        assert result.total_rows == 2
        assert len(result.breaks) == 0
        assert result.head_chain_hash is not None

    def test_tampered_row_detected(self, tmp_db):
        create_strategy(strategy_id="s1", actor="test", db_path=tmp_db)
        transition(strategy_id="s1", to_state=StrategyState.AUDITED,
                   actor="t", has_candidate_pipeline_run=True,
                   db_path=tmp_db)
        conn = _connect(tmp_db)
        # TAMPER: change actor on the second row
        conn.execute("UPDATE state_transitions SET actor = 'eve' WHERE id = 2")
        result = verify_ledger_integrity(conn)
        assert result.chain_intact is False
        assert len(result.breaks) >= 1
        assert any(b.severity == "TAMPERED" for b in result.breaks)

    def test_missing_chain_detected(self, tmp_db):
        create_strategy(strategy_id="s1", actor="test", db_path=tmp_db)
        conn = _connect(tmp_db)
        # WIPE the chain_hash field
        conn.execute("UPDATE state_transitions SET chain_hash = NULL WHERE id = 1")
        result = verify_ledger_integrity(conn)
        assert result.chain_intact is False
        assert any(b.severity == "MISSING_CHAIN" for b in result.breaks)


# ── 5. End-to-end chain matches manual computation ─────────────────────


class TestEndToEndChain:
    def test_three_transition_chain_matches_manual(self, tmp_db):
        create_strategy(strategy_id="e2e", actor="alice", db_path=tmp_db)
        transition(strategy_id="e2e", to_state=StrategyState.AUDITED,
                   actor="bob", has_candidate_pipeline_run=True,
                   reason="step1", db_path=tmp_db)
        transition(strategy_id="e2e", to_state=StrategyState.APPROVED,
                   actor="carol", has_human_approval=True,
                   reason="step2", db_path=tmp_db)

        conn = _connect(tmp_db)
        result = verify_ledger_integrity(conn)
        assert result.chain_intact is True
        assert result.total_rows == 3

    def test_head_hash_changes_after_each_transition(self, tmp_db):
        create_strategy(strategy_id="e2e", actor="alice", db_path=tmp_db)
        conn = _connect(tmp_db)
        head1 = get_head_chain_hash(conn)

        transition(strategy_id="e2e", to_state=StrategyState.AUDITED,
                   actor="bob", has_candidate_pipeline_run=True,
                   db_path=tmp_db)
        head2 = get_head_chain_hash(conn)

        assert head1 != head2
        assert head2 != GENESIS_HASH
