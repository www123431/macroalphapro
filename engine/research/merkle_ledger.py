"""engine/research/merkle_ledger.py — SLM Phase 4: cryptographic chain
on state_transitions for SOX-equivalent audit integrity.

Each transition row is bound to its predecessor via:

    chain_hash[n] = SHA256(chain_hash[n-1] || canonical_row_json[n])

where canonical_row_json excludes the chain_hash field itself plus the
auto-increment id (which is not deterministic across DB rebuilds).

Tampering detection: recompute the chain from the genesis hash and
compare with stored chain_hashes. Any mismatch indicates either:
  (a) a row was modified after insertion
  (b) a row was inserted out-of-order
  (c) chain_hash itself was rewritten
All three are equally suspicious for an audit trail.

Why this matters institutionally:
  - Regulator (SEC) audit: prove the audit trail wasn't doctored
  - Internal compliance: detect malicious or accidental edits
  - Production replay: detect data corruption from disk failures

Implementation:
  - Genesis hash = "0" * 64 (literal string, not zero bytes — easier to
    inspect in SQL)
  - Canonical JSON: sort_keys=True, separators=(",", ":"), default=str
  - Chain field: TEXT column added via migration v2
  - Verification: read all transitions ORDER BY transition_at, id;
    recompute + compare
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

GENESIS_HASH = "0" * 64


def canonical_row_json(
    *,
    strategy_id: str,
    from_state: Optional[str],
    to_state: str,
    transition_at: str,
    actor: str,
    reason: str,
    gate_evidence: dict[str, Any] | str,
    git_sha: Optional[str],
) -> str:
    """Canonical JSON representation of a transition row.

    Excludes:
      - id (auto-increment, non-deterministic across rebuilds)
      - chain_hash (the field this hash is for — would be self-referential)

    Includes everything else with sort_keys + no-whitespace separators
    for deterministic byte-identical output.
    """
    if isinstance(gate_evidence, str):
        # gate_evidence stored as TEXT in DB; canonicalize the JSON content
        # rather than the encoded string (which may have whitespace variance).
        try:
            ge = json.loads(gate_evidence)
        except json.JSONDecodeError:
            ge = gate_evidence
    else:
        ge = gate_evidence
    payload = {
        "strategy_id":   strategy_id,
        "from_state":    from_state,
        "to_state":      to_state,
        "transition_at": transition_at,
        "actor":         actor,
        "reason":        reason,
        "gate_evidence": ge,
        "git_sha":       git_sha,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str, ensure_ascii=False)


def compute_chain_hash(prev_chain_hash: str, canonical_json: str) -> str:
    """SHA256(prev_chain_hash || canonical_json), hex-encoded.

    Concatenation uses the literal string forms — `prev || canon` —
    which is the standard Merkle chain construction.
    """
    h = hashlib.sha256()
    h.update(prev_chain_hash.encode("utf-8"))
    h.update(canonical_json.encode("utf-8"))
    return h.hexdigest()


# ── Migration helper ───────────────────────────────────────────────────


def add_chain_column_if_missing(conn: sqlite3.Connection) -> bool:
    """Idempotent migration: add chain_hash column + populate from genesis.

    Returns True if column was added (migration ran), False if already present.

    SAFE TO RUN MULTIPLE TIMES: detects column presence via PRAGMA
    table_info and short-circuits.
    """
    rows = conn.execute("PRAGMA table_info(state_transitions)").fetchall()
    columns = {row[1] for row in rows}
    if "chain_hash" in columns:
        return False

    conn.execute("ALTER TABLE state_transitions ADD COLUMN chain_hash TEXT")

    # Backfill: walk the existing transitions in chronological order +
    # compute chain_hash for each.
    existing = conn.execute("""
        SELECT id, strategy_id, from_state, to_state, transition_at,
               actor, reason, gate_evidence, git_sha
        FROM state_transitions
        ORDER BY transition_at, id
    """).fetchall()

    prev = GENESIS_HASH
    for row in existing:
        rid = row[0]
        canon = canonical_row_json(
            strategy_id=row[1], from_state=row[2], to_state=row[3],
            transition_at=row[4], actor=row[5], reason=row[6],
            gate_evidence=row[7], git_sha=row[8],
        )
        ch = compute_chain_hash(prev, canon)
        conn.execute(
            "UPDATE state_transitions SET chain_hash = ? WHERE id = ?",
            (ch, rid),
        )
        prev = ch
    return True


# ── Verification ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LedgerBreak:
    """One detected ledger integrity break."""

    row_id: int
    strategy_id: str
    transition_at: str
    expected_chain_hash: str
    stored_chain_hash: Optional[str]
    severity: str    # "TAMPERED" | "MISSING_CHAIN" | "REORDERED"


@dataclass(frozen=True)
class LedgerVerifyResult:
    """Output of verify_ledger_integrity."""

    total_rows: int
    chain_intact: bool
    breaks: list[LedgerBreak]
    head_chain_hash: Optional[str]    # the latest row's chain_hash if intact


def verify_ledger_integrity(conn: sqlite3.Connection) -> LedgerVerifyResult:
    """Walk the entire state_transitions table in chronological order,
    recompute chain_hash for each row, compare with stored value.

    Returns LedgerVerifyResult with breaks list (empty if intact).
    """
    rows = conn.execute("""
        SELECT id, strategy_id, from_state, to_state, transition_at,
               actor, reason, gate_evidence, git_sha, chain_hash
        FROM state_transitions
        ORDER BY transition_at, id
    """).fetchall()

    breaks: list[LedgerBreak] = []
    prev = GENESIS_HASH
    head: Optional[str] = None

    for row in rows:
        rid = row[0]
        stored_ch = row[9] if len(row) > 9 else None
        canon = canonical_row_json(
            strategy_id=row[1], from_state=row[2], to_state=row[3],
            transition_at=row[4], actor=row[5], reason=row[6],
            gate_evidence=row[7], git_sha=row[8],
        )
        expected = compute_chain_hash(prev, canon)

        if stored_ch is None:
            breaks.append(LedgerBreak(
                row_id=rid, strategy_id=row[1], transition_at=row[4],
                expected_chain_hash=expected, stored_chain_hash=None,
                severity="MISSING_CHAIN",
            ))
        elif stored_ch != expected:
            breaks.append(LedgerBreak(
                row_id=rid, strategy_id=row[1], transition_at=row[4],
                expected_chain_hash=expected, stored_chain_hash=stored_ch,
                severity="TAMPERED",
            ))
        # Advance the chain using STORED hash (so a single break doesn't
        # cascade into 1000 breaks downstream — caller can investigate
        # the one bad row in isolation).
        prev = stored_ch if stored_ch is not None else expected
        head = stored_ch

    return LedgerVerifyResult(
        total_rows=len(rows),
        chain_intact=len(breaks) == 0,
        breaks=breaks,
        head_chain_hash=head,
    )


def get_head_chain_hash(conn: sqlite3.Connection) -> str:
    """Return the chain_hash of the most recent transition row.

    Useful for external attestation: publish head_chain_hash to an
    immutable external store (e.g. blockchain anchor / public gist) at
    audit checkpoints. Later, verify integrity by recomputing the
    chain locally + checking the head matches the published value.
    """
    row = conn.execute("""
        SELECT chain_hash FROM state_transitions
        ORDER BY transition_at DESC, id DESC LIMIT 1
    """).fetchone()
    if row is None:
        return GENESIS_HASH
    return row[0] or GENESIS_HASH
