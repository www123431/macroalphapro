"""engine/research/strategy_state_store.py — SQLite ACID store for
Strategy Lifecycle Manager Phase 0.

Why SQLite (not JSONL files like the older ledger)?
  - ACID transactions (no torn writes if process dies mid-update)
  - Concurrent reader access without manual locking
  - SQL-level constraints (FK + CHECK) catch bugs JSONL files cannot
  - Single-file portability (one .db, can be backed up + diffed)
  - Cryptographic chaining (Phase 4) trivially layered on top

Schema (v1):
  strategy_state         — current row per strategy_id
  state_transitions      — append-only history of every transition
  schema_version         — migration tracker

Migration policy:
  - All schema changes go through `_MIGRATIONS` list; each migration
    is an idempotent SQL block executed in version order.
  - `init_db()` reads schema_version and applies any missing migrations.
  - Never edit an already-applied migration; add a new one.
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from engine.research.strategy_lifecycle import (
    GateNotMetError,
    InvalidTransitionError,
    StrategyRecord,
    StrategyState,
    StateTransitionRecord,
    enforce_transition,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "strategy_lifecycle.db"

# ── Migrations ──────────────────────────────────────────────────────────

_MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE IF NOT EXISTS strategy_state (
        strategy_id                 TEXT PRIMARY KEY,
        current_state               TEXT NOT NULL,
        proposed_at                 TEXT,
        audited_at                  TEXT,
        approved_at                 TEXT,
        approved_by                 TEXT,
        paper_trade_started         TEXT,
        shadow_started              TEXT,
        live_started                TEXT,
        decommissioned_at           TEXT,
        current_allocation_pct      REAL NOT NULL DEFAULT 0.0,
        target_allocation_pct       REAL NOT NULL DEFAULT 0.0,
        library_yaml_path           TEXT,
        candidate_pipeline_run_id   TEXT,
        parent_strategy_id          TEXT,
        notes                       TEXT NOT NULL DEFAULT '',
        created_at                  TEXT NOT NULL,
        updated_at                  TEXT NOT NULL,
        CHECK (current_allocation_pct >= 0 AND current_allocation_pct <= 1.0),
        CHECK (target_allocation_pct  >= 0 AND target_allocation_pct  <= 1.0)
    );

    CREATE TABLE IF NOT EXISTS state_transitions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id       TEXT NOT NULL,
        from_state        TEXT,
        to_state          TEXT NOT NULL,
        transition_at     TEXT NOT NULL,
        actor             TEXT NOT NULL,
        reason            TEXT NOT NULL DEFAULT '',
        gate_evidence     TEXT NOT NULL DEFAULT '{}',
        git_sha           TEXT,
        FOREIGN KEY (strategy_id) REFERENCES strategy_state(strategy_id)
    );

    CREATE INDEX IF NOT EXISTS idx_transitions_strategy
        ON state_transitions(strategy_id, transition_at);

    CREATE INDEX IF NOT EXISTS idx_strategy_state_current
        ON strategy_state(current_state);

    CREATE TABLE IF NOT EXISTS schema_version (
        version     INTEGER PRIMARY KEY,
        applied_at  TEXT NOT NULL
    );
    """),
]


def _apply_merkle_migration(conn: sqlite3.Connection) -> None:
    """v2 migration: add chain_hash column + backfill existing rows.
    Implemented in merkle_ledger module to keep migration logic close
    to the chain construction code."""
    from engine.research.merkle_ledger import add_chain_column_if_missing
    add_chain_column_if_missing(conn)


# ── Connection management ───────────────────────────────────────────────


_local = threading.local()


def _connect(db_path: Path) -> sqlite3.Connection:
    """Per-thread connection cache. SQLite is thread-affine; each thread
    gets its own connection. WAL mode for concurrent reader support."""
    cache: dict[str, sqlite3.Connection] = getattr(_local, "conns", {})
    key = str(db_path)
    if key not in cache:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            key,
            isolation_level=None,        # autocommit; we manage txns explicitly
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=True,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        cache[key] = conn
        _local.conns = cache
    return cache[key]


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Apply any missing migrations. Idempotent.

    Includes Phase 4 Merkle migration (v2): adds chain_hash column to
    state_transitions + backfills from genesis. Runs once via the
    schema_version tracker."""
    conn = _connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL
        )
    """)
    cur = conn.execute("SELECT MAX(version) FROM schema_version")
    current = cur.fetchone()[0] or 0
    for version, ddl in _MIGRATIONS:
        if version <= current:
            continue
        conn.executescript("BEGIN; " + ddl + ";\nCOMMIT;")
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, _dt.datetime.now(_dt.timezone.utc).isoformat()),
        )

    # Phase 4: Merkle chain_hash column (treated as v2 migration but
    # tracked separately because it needs Python-level row computation,
    # not pure SQL).
    if current < 2:
        _apply_merkle_migration(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
            (2, _dt.datetime.now(_dt.timezone.utc).isoformat()),
        )


# ── Datetime helpers ────────────────────────────────────────────────────


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(dt: Optional[_dt.datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _parse(s: Optional[str]) -> Optional[_dt.datetime]:
    if s is None:
        return None
    return _dt.datetime.fromisoformat(s)


# ── CRUD ────────────────────────────────────────────────────────────────


def _row_to_record(row: sqlite3.Row) -> StrategyRecord:
    return StrategyRecord(
        strategy_id=row["strategy_id"],
        current_state=StrategyState(row["current_state"]),
        proposed_at=_parse(row["proposed_at"]),
        audited_at=_parse(row["audited_at"]),
        approved_at=_parse(row["approved_at"]),
        approved_by=row["approved_by"],
        paper_trade_started=_parse(row["paper_trade_started"]),
        shadow_started=_parse(row["shadow_started"]),
        live_started=_parse(row["live_started"]),
        decommissioned_at=_parse(row["decommissioned_at"]),
        current_allocation_pct=row["current_allocation_pct"],
        target_allocation_pct=row["target_allocation_pct"],
        library_yaml_path=row["library_yaml_path"],
        candidate_pipeline_run_id=row["candidate_pipeline_run_id"],
        parent_strategy_id=row["parent_strategy_id"],
        notes=row["notes"],
        created_at=_parse(row["created_at"]),
        updated_at=_parse(row["updated_at"]),
    )


def create_strategy(
    *,
    strategy_id: str,
    initial_state: StrategyState = StrategyState.PROPOSED,
    library_yaml_path: Optional[str] = None,
    parent_strategy_id: Optional[str] = None,
    candidate_pipeline_run_id: Optional[str] = None,
    notes: str = "",
    actor: str = "system",
    db_path: Path = DEFAULT_DB_PATH,
) -> StrategyRecord:
    """Create a new strategy in `initial_state`. Most callers should use
    PROPOSED; tests / migrations may seed in later states.

    Records the initial creation as a transition (from_state=NULL).
    Raises IntegrityError if strategy_id already exists.
    """
    init_db(db_path)
    conn = _connect(db_path)
    now = _now()
    now_iso = now.isoformat()
    proposed_at_iso = now_iso if initial_state == StrategyState.PROPOSED else None

    with conn:
        conn.execute("BEGIN")
        try:
            conn.execute(
                """INSERT INTO strategy_state (
                    strategy_id, current_state, proposed_at,
                    library_yaml_path, parent_strategy_id,
                    candidate_pipeline_run_id, notes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    strategy_id, initial_state.value, proposed_at_iso,
                    library_yaml_path, parent_strategy_id,
                    candidate_pipeline_run_id, notes,
                    now_iso, now_iso,
                ),
            )
            # Compute chain_hash for the genesis transition row
            from engine.research.merkle_ledger import (
                canonical_row_json, compute_chain_hash, get_head_chain_hash,
            )
            prev_hash = get_head_chain_hash(conn)
            reason = f"create_strategy initial_state={initial_state.value}"
            canon = canonical_row_json(
                strategy_id=strategy_id, from_state=None,
                to_state=initial_state.value, transition_at=now_iso,
                actor=actor, reason=reason, gate_evidence={}, git_sha=None,
            )
            chain_hash = compute_chain_hash(prev_hash, canon)
            conn.execute(
                """INSERT INTO state_transitions (
                    strategy_id, from_state, to_state, transition_at,
                    actor, reason, gate_evidence, chain_hash
                ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?)""",
                (strategy_id, initial_state.value, now_iso, actor,
                 reason, "{}", chain_hash),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    return get_strategy(strategy_id, db_path=db_path)


def get_strategy(
    strategy_id: str, db_path: Path = DEFAULT_DB_PATH
) -> StrategyRecord:
    init_db(db_path)
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT * FROM strategy_state WHERE strategy_id = ?", (strategy_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"strategy_id {strategy_id!r} not found")
    return _row_to_record(row)


def list_strategies(
    state: Optional[StrategyState] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[StrategyRecord]:
    init_db(db_path)
    conn = _connect(db_path)
    if state is None:
        rows = conn.execute(
            "SELECT * FROM strategy_state ORDER BY created_at"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM strategy_state WHERE current_state = ? ORDER BY created_at",
            (state.value,),
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def transition(
    *,
    strategy_id: str,
    to_state: StrategyState,
    actor: str,
    reason: str = "",
    has_candidate_pipeline_run: bool = False,
    has_human_approval: bool = False,
    paper_trade_months: int = 0,
    sequential_test_pass: bool = False,
    ramp_protocol_step: int = 0,
    decay_alert_level: Optional[str] = None,
    explicit_override: bool = False,
    git_sha: Optional[str] = None,
    extra_evidence: Optional[dict[str, Any]] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> StrategyRecord:
    """Apply a state transition with full ACID semantics.

    1. Reads current row in BEGIN IMMEDIATE txn (exclusive lock).
    2. Validates via enforce_transition() — raises on invalid / gate-fail.
    3. Updates strategy_state row + writes state_transitions entry.
    4. COMMIT atomically; ROLLBACK on any error.
    """
    init_db(db_path)
    conn = _connect(db_path)
    now = _now()
    now_iso = now.isoformat()

    with conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT current_state FROM strategy_state WHERE strategy_id = ?",
                (strategy_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"strategy_id {strategy_id!r} not found")
            from_state = StrategyState(row["current_state"])

            # Validate transition — may raise InvalidTransitionError or
            # GateNotMetError. Both propagate after ROLLBACK.
            # role passed here is sourced from extra_evidence if caller
            # supplied it; the state store is lifecycle-agnostic so the
            # role-specific layer is opt-in (Phase 2 wires this through
            # role inference from the sleeve registry).
            _role = None
            if extra_evidence and "role" in extra_evidence:
                from engine.research.strategy_lifecycle import SleeveRole as _SR
                _role = _SR.from_yaml_value(extra_evidence["role"])
            _role_evidence_passed = bool(
                extra_evidence and extra_evidence.get("role_specific_evidence_passed", False)
            )
            gate, role_gate = enforce_transition(
                from_state=from_state,
                to_state=to_state,
                role=_role,
                has_candidate_pipeline_run=has_candidate_pipeline_run,
                has_human_approval=has_human_approval,
                paper_trade_months=paper_trade_months,
                sequential_test_pass=sequential_test_pass,
                role_specific_evidence_passed=_role_evidence_passed,
                ramp_protocol_step=ramp_protocol_step,
                decay_alert_level=decay_alert_level,  # type: ignore[arg-type]
                explicit_override=explicit_override,
            )

            # Build gate-evidence dict for audit trail
            evidence: dict[str, Any] = {
                "gate_rationale": gate.rationale,
                "role_specific_metric": role_gate.metric_name if role_gate else None,
                "has_candidate_pipeline_run": has_candidate_pipeline_run,
                "has_human_approval": has_human_approval,
                "paper_trade_months": paper_trade_months,
                "sequential_test_pass": sequential_test_pass,
                "ramp_protocol_step": ramp_protocol_step,
                "decay_alert_level": decay_alert_level,
                "explicit_override": explicit_override,
            }
            if extra_evidence:
                evidence["extra"] = extra_evidence

            # Update timestamp columns based on state entered
            timestamp_field = {
                StrategyState.AUDITED: "audited_at",
                StrategyState.APPROVED: "approved_at",
                StrategyState.PAPER_TRADE: "paper_trade_started",
                StrategyState.SHADOW: "shadow_started",
                StrategyState.LIVE: "live_started",
                StrategyState.ARCHIVED: "decommissioned_at",
            }.get(to_state)

            if timestamp_field is not None:
                conn.execute(
                    f"UPDATE strategy_state SET current_state = ?, {timestamp_field} = ?, "
                    f"updated_at = ?{', approved_by = ?' if to_state == StrategyState.APPROVED else ''} "
                    "WHERE strategy_id = ?",
                    (to_state.value, now_iso, now_iso, actor, strategy_id)
                    if to_state == StrategyState.APPROVED
                    else (to_state.value, now_iso, now_iso, strategy_id),
                )
            else:
                conn.execute(
                    "UPDATE strategy_state SET current_state = ?, updated_at = ? "
                    "WHERE strategy_id = ?",
                    (to_state.value, now_iso, strategy_id),
                )

            # Compute Merkle chain_hash before inserting
            from engine.research.merkle_ledger import (
                canonical_row_json, compute_chain_hash, get_head_chain_hash,
            )
            prev_hash = get_head_chain_hash(conn)
            evidence_json = json.dumps(evidence, default=str)
            canon = canonical_row_json(
                strategy_id=strategy_id, from_state=from_state.value,
                to_state=to_state.value, transition_at=now_iso,
                actor=actor, reason=reason,
                gate_evidence=evidence_json, git_sha=git_sha,
            )
            chain_hash = compute_chain_hash(prev_hash, canon)
            conn.execute(
                """INSERT INTO state_transitions (
                    strategy_id, from_state, to_state, transition_at,
                    actor, reason, gate_evidence, git_sha, chain_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (strategy_id, from_state.value, to_state.value, now_iso,
                 actor, reason, evidence_json, git_sha, chain_hash),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    return get_strategy(strategy_id, db_path=db_path)


def update_allocation(
    *,
    strategy_id: str,
    current_allocation_pct: float,
    target_allocation_pct: Optional[float] = None,
    actor: str = "ramp_protocol",
    db_path: Path = DEFAULT_DB_PATH,
) -> StrategyRecord:
    """Update current and/or target allocation without changing state.
    Used by ramp protocol to record 1% → 5% → 15% → target steps.
    """
    init_db(db_path)
    conn = _connect(db_path)
    now_iso = _now().isoformat()
    with conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            if target_allocation_pct is None:
                conn.execute(
                    "UPDATE strategy_state SET current_allocation_pct = ?, "
                    "updated_at = ? WHERE strategy_id = ?",
                    (current_allocation_pct, now_iso, strategy_id),
                )
            else:
                conn.execute(
                    "UPDATE strategy_state SET current_allocation_pct = ?, "
                    "target_allocation_pct = ?, updated_at = ? "
                    "WHERE strategy_id = ?",
                    (current_allocation_pct, target_allocation_pct,
                     now_iso, strategy_id),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return get_strategy(strategy_id, db_path=db_path)


def get_transition_history(
    strategy_id: str, db_path: Path = DEFAULT_DB_PATH
) -> list[StateTransitionRecord]:
    init_db(db_path)
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM state_transitions WHERE strategy_id = ? "
        "ORDER BY transition_at, id",
        (strategy_id,),
    ).fetchall()
    out: list[StateTransitionRecord] = []
    for r in rows:
        out.append(StateTransitionRecord(
            strategy_id=r["strategy_id"],
            from_state=StrategyState(r["from_state"]) if r["from_state"] else None,
            to_state=StrategyState(r["to_state"]),
            transition_at=_parse(r["transition_at"]),
            actor=r["actor"],
            reason=r["reason"],
            gate_evidence=json.loads(r["gate_evidence"]),
            git_sha=r["git_sha"],
        ))
    return out


def reset_db_for_test(db_path: Path) -> None:
    """Test-only: drop all tables in `db_path` and reset schema_version.

    Refuses to operate on DEFAULT_DB_PATH to prevent accidents.
    """
    if db_path.resolve() == DEFAULT_DB_PATH.resolve():
        raise RuntimeError("reset_db_for_test cannot operate on DEFAULT_DB_PATH")
    if db_path.exists():
        # Drop the per-thread cached connection so the file lock releases
        cache = getattr(_local, "conns", {})
        if str(db_path) in cache:
            cache[str(db_path)].close()
            del cache[str(db_path)]
        db_path.unlink()
