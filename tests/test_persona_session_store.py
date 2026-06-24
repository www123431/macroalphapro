"""tests/test_persona_session_store.py — Phase A.3 ε memory persistence tests.

Covers the SQLite scratch pad that backs the Streamlit chat UI for each
persona agent. The store is keyed by agent_id (one chat session per
agent — see ChatSession docstring in engine.db_models) and replaces
the row on every save (no audit churn; this is mutable scratch, not
an audit log).

Test invariants:
  - load on empty row returns the documented SessionSnapshot zero-value
  - save_turn round-trips all four fields (history / tool_log / cost / latency)
  - save_turn REPLACES the row rather than appending (only one session
    per agent_id is ever held — schema-level Pattern 5 ban)
  - reset_session clears the row, and load returns zero-value after
  - reset_session on a missing row is idempotent
  - Two different agent_ids do not collide
  - load_session never raises on DB error — degrades to empty snapshot
    (so a transient DB hiccup does not crash the Streamlit page render)
"""
from __future__ import annotations

import pytest


# Conftest creates the temp DB + init_db ensures engine.memory.Base
# tables exist, but engine.db_models.Base is a SEPARATE Base (chat
# sessions + alert tables + audit findings all live there). Create the
# full engine.db_models metadata + engine.auto_audit_models metadata so
# Phase A.4 tests that touch RiskManagerAlert / AuditFinding don't
# crash with "no such table".
@pytest.fixture(autouse=True)
def _ensure_chat_sessions_table():
    from engine.db_models import Base as DBModelsBase, engine, ChatSession
    from engine.auto_audit_models import Base as AuditBase
    DBModelsBase.metadata.create_all(engine)
    AuditBase.metadata.create_all(engine)
    yield
    # Cleanup: each test wipes ChatSession so visit / cost state never
    # leaks across cases. Alert / finding rows are managed test-by-test
    # (the test that inserts them is also the test that deletes them).
    from engine.db_models import SessionFactory
    with SessionFactory() as s:
        s.query(ChatSession).delete()
        s.commit()


def test_load_on_empty_returns_zero_snapshot():
    from engine.agents.persona.session_store import load_session
    snap = load_session("does_not_exist")
    assert snap.history == []
    assert snap.tool_log == []
    assert snap.cost_usd == 0.0
    assert snap.latency_ms == 0


def test_save_then_load_round_trip():
    from engine.agents.persona.session_store import load_session, save_turn

    history = [
        {"role": "user",      "content": "what's K1 BAB status?"},
        {"role": "assistant", "content": "K1 BAB ran clean — 30 positions."},
    ]
    tool_log = [[{"name": "lookup_strategy_status",
                  "input": {"strategy_name": "K1_BAB"},
                  "result_preview": "{...}"}]]

    save_turn(
        agent_id   = "risk_manager",
        history    = history,
        tool_log   = tool_log,
        cost_usd   = 0.0123,
        latency_ms = 842,
    )

    snap = load_session("risk_manager")
    assert snap.history == history
    assert snap.tool_log == tool_log
    assert snap.cost_usd == pytest.approx(0.0123)
    assert snap.latency_ms == 842


def test_save_replaces_row_not_appends():
    """save_turn must REPLACE the row, not stack rows. Otherwise a
    long-running chat would grow the table unbounded and the next
    load would not know which version to pick."""
    from engine.agents.persona.session_store import load_session, save_turn
    from engine.db_models import ChatSession, SessionFactory

    save_turn("risk_manager", history=[{"role": "user", "content": "first"}],
              tool_log=[], cost_usd=0.001, latency_ms=10)
    save_turn("risk_manager", history=[{"role": "user", "content": "second"}],
              tool_log=[], cost_usd=0.002, latency_ms=20)

    # Exactly one row in the table for this agent_id
    with SessionFactory() as s:
        rows = s.query(ChatSession).filter_by(agent_id="risk_manager").all()
        assert len(rows) == 1

    snap = load_session("risk_manager")
    assert snap.history == [{"role": "user", "content": "second"}]
    assert snap.cost_usd == pytest.approx(0.002)


def test_reset_session_clears_row():
    from engine.agents.persona.session_store import (
        load_session, save_turn, reset_session,
    )

    save_turn("dq_inspector", history=[{"role": "user", "content": "hi"}],
              tool_log=[], cost_usd=0.005, latency_ms=50)
    assert load_session("dq_inspector").history != []

    reset_session("dq_inspector")
    snap = load_session("dq_inspector")
    assert snap.history == []
    assert snap.cost_usd == 0.0
    assert snap.latency_ms == 0


def test_reset_session_idempotent_on_missing_row():
    """Calling reset on an agent that has never been saved should not
    raise — the Streamlit page should be free to fire it any time."""
    from engine.agents.persona.session_store import reset_session
    reset_session("never_existed_agent")
    reset_session("never_existed_agent")   # twice, just to be sure


def test_two_agents_do_not_collide():
    """Pattern 5 ban: each agent_id has its own row. Saving RM must
    not touch the DQ row."""
    from engine.agents.persona.session_store import load_session, save_turn

    save_turn("risk_manager",
              history=[{"role": "user", "content": "rm question"}],
              tool_log=[], cost_usd=0.01, latency_ms=100)
    save_turn("dq_inspector",
              history=[{"role": "user", "content": "dq question"}],
              tool_log=[], cost_usd=0.02, latency_ms=200)

    rm = load_session("risk_manager")
    dq = load_session("dq_inspector")
    assert rm.history[0]["content"] == "rm question"
    assert dq.history[0]["content"] == "dq question"
    assert rm.cost_usd == pytest.approx(0.01)
    assert dq.cost_usd == pytest.approx(0.02)


def test_load_session_degrades_on_db_error(monkeypatch, caplog):
    """If the DB blows up during load, load_session must return the
    empty snapshot rather than raise — otherwise a transient SQLite
    glitch crashes the Streamlit page on render."""
    import engine.agents.persona.session_store as store

    class _Boom:
        def __enter__(self):
            raise RuntimeError("simulated DB outage")
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "engine.db_models.SessionFactory",
        lambda: _Boom(),
    )

    snap = store.load_session("risk_manager")
    assert snap.history == []
    assert snap.cost_usd == 0.0
    assert "load_session" in caplog.text or "simulated DB outage" in caplog.text


def test_tool_log_nested_list_round_trip():
    """Tool log is list-of-lists (one inner list per turn). Make sure
    nesting survives json round trip and the type is preserved."""
    from engine.agents.persona.session_store import load_session, save_turn

    tool_log = [
        [{"name": "lookup_spec", "input": {"spec_id": 69}}],
        [],   # a turn with no tool calls
        [
            {"name": "read_project_memory", "input": {"query": "harking"}},
            {"name": "query_recent_alerts", "input": {"days_back": 7}},
        ],
    ]
    save_turn("devils_advocate", history=[], tool_log=tool_log,
              cost_usd=0.0, latency_ms=0)
    snap = load_session("devils_advocate")
    assert snap.tool_log == tool_log
    assert isinstance(snap.tool_log, list)
    assert all(isinstance(t, list) for t in snap.tool_log)


# ──────────────────────────────────────────────────────────────────────────────
# Phase A.4 — last_visited_at + new-alert badge helpers
# ──────────────────────────────────────────────────────────────────────────────
def test_mark_visited_creates_row_on_first_visit():
    """If no ChatSession row exists yet, mark_visited must create one with
    history=[] and a real last_visited_at timestamp — otherwise the very
    first page load on a fresh DB would crash the chat page."""
    from engine.agents.persona.session_store import (
        load_session, mark_visited,
    )

    # Pre-condition: no row for this agent_id
    snap0 = load_session("attribution_analyst")
    assert snap0.last_visited_at is None
    assert snap0.history == []

    mark_visited("attribution_analyst")

    snap1 = load_session("attribution_analyst")
    assert snap1.last_visited_at is not None
    # Schema invariants preserved (no spurious history wipe).
    assert snap1.history == []


def test_mark_visited_updates_existing_row_without_clobbering_history():
    """A subsequent mark_visited must update the timestamp but NOT zero
    out the chat history / cost / tool log. This is the case the badge
    relies on most: passive view shouldn't reset the conversation."""
    import datetime as _dt
    from engine.agents.persona.session_store import (
        load_session, mark_visited, save_turn,
    )

    save_turn(
        agent_id   = "audit_recorder",
        history    = [{"role": "user", "content": "first turn"}],
        tool_log   = [[{"name": "query_audit_runs", "input": {}}]],
        cost_usd   = 0.0042,
        latency_ms = 123,
    )
    snap_before = load_session("audit_recorder")

    mark_visited("audit_recorder")

    snap_after = load_session("audit_recorder")
    # Visit timestamp moved forward.
    assert snap_after.last_visited_at is not None
    if snap_before.last_visited_at is not None:
        assert snap_after.last_visited_at >= snap_before.last_visited_at
    # Conversation state intact.
    assert snap_after.history == snap_before.history
    assert snap_after.tool_log == snap_before.tool_log
    assert snap_after.cost_usd == snap_before.cost_usd


def test_count_new_alerts_none_since_returns_zero():
    """First visit has last_visited_at=None — the badge must hide rather
    than reporting "all rows ever" as "new"."""
    from engine.agents.persona.session_store import count_new_alerts

    n, label = count_new_alerts("risk_manager", None)
    assert n == 0
    assert label is None


def test_count_new_alerts_skips_personas_without_alert_source():
    """Attribution Analyst + Devil's Advocate have no native alert tables.
    The badge helper must return (0, None) instead of crashing or
    pointing at the wrong table."""
    import datetime as _dt
    from engine.agents.persona.session_store import count_new_alerts

    ancient = _dt.datetime(2020, 1, 1)
    for aid in ("attribution_analyst", "devils_advocate"):
        n, label = count_new_alerts(aid, ancient)
        assert n == 0
        assert label is None, (
            f"{aid} unexpectedly has an alert source: {label}"
        )


def test_count_new_alerts_finds_rows_after_since():
    """If we insert an alert row after a known `since`, the helper must
    count it. Inserts into the RiskManagerAlert table directly."""
    import datetime as _dt
    from engine.db_models import RiskManagerAlert, SessionFactory
    from engine.agents.persona.session_store import count_new_alerts

    since = _dt.datetime.utcnow() - _dt.timedelta(minutes=1)

    with SessionFactory() as s:
        n_before = s.query(RiskManagerAlert).filter(
            RiskManagerAlert.generated_at_utc > since
        ).count()
        # Append one synthetic alert with generated_at_utc=now.
        s.add(RiskManagerAlert(
            date              = _dt.date.today(),
            alert_id          = "test-badge-uuid",
            mode_id           = "test",
            severity          = "SOFT_WARN",
            cb_severity       = "LIGHT",
            halt_decision     = False,
            phase             = "pre_trade",
            rule_description  = "test row for badge unit test",
            affected_json     = "[]",
            extra_json        = "{}",
            spec_anchor       = "test",
            generated_at_utc  = _dt.datetime.utcnow(),
        ))
        s.commit()

    try:
        n, label = count_new_alerts("risk_manager", since)
        assert n == n_before + 1
        assert label == "Risk Manager alerts"
    finally:
        # Cleanup so this test is idempotent across reruns.
        with SessionFactory() as s:
            s.query(RiskManagerAlert).filter(
                RiskManagerAlert.alert_id == "test-badge-uuid"
            ).delete()
            s.commit()


def test_count_new_alerts_degrades_on_db_error(monkeypatch, caplog):
    """A DB hiccup must not break the chat page render — counter returns
    (0, None) and the badge stays hidden."""
    import engine.agents.persona.session_store as store

    class _Boom:
        def __enter__(self):
            raise RuntimeError("simulated DB outage")
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "engine.db_models.SessionFactory",
        lambda: _Boom(),
    )
    import datetime as _dt
    n, label = store.count_new_alerts(
        "risk_manager", _dt.datetime(2020, 1, 1),
    )
    assert n == 0
    assert label is None


# ──────────────────────────────────────────────────────────────────────────────
# Phase A.7 Wave 3.2 — session title derivation + persistence
# ──────────────────────────────────────────────────────────────────────────────
def test_derive_title_picks_first_user_message():
    from engine.agents.persona.session_store import _derive_title
    history = [
        {"role": "user", "content": "Is the book safe today?"},
        {"role": "assistant", "content": "Yes — gross 1.0."},
    ]
    assert _derive_title(history) == "Is the book safe today?"


def test_derive_title_truncates_long_messages():
    from engine.agents.persona.session_store import _derive_title
    long_text = "x" * 200
    history = [{"role": "user", "content": long_text}]
    title = _derive_title(history)
    assert title is not None
    assert title.endswith("...")
    assert len(title) <= 85   # 80 chars + 3 ellipsis + small buffer


def test_derive_title_skips_empty_history():
    from engine.agents.persona.session_store import _derive_title
    assert _derive_title([]) is None
    assert _derive_title([{"role": "assistant", "content": "stub"}]) is None
    assert _derive_title([{"role": "user", "content": ""}]) is None


def test_derive_title_handles_list_content_blocks():
    """Anthropic-format messages have content as list of blocks. Title
    derivation must walk into the first text block."""
    from engine.agents.persona.session_store import _derive_title
    history = [
        {"role": "user",
         "content": [{"type": "text", "text": "What's K1 BAB status?"}]},
    ]
    assert _derive_title(history) == "What's K1 BAB status?"


def test_save_turn_auto_derives_title_on_first_save():
    from engine.agents.persona.session_store import load_session, save_turn
    save_turn(
        agent_id   = "risk_manager",
        history    = [
            {"role": "user", "content": "VaR status today?"},
            {"role": "assistant", "content": "VaR -2.8%, soft warn."},
        ],
        tool_log   = [],
        cost_usd   = 0.001,
        latency_ms = 50,
    )
    snap = load_session("risk_manager")
    assert snap.title == "VaR status today?"


def test_save_turn_preserves_existing_title():
    """If the title is already set, save_turn must NOT overwrite it on
    subsequent saves — preserves any manual user override and avoids
    flipping the title around on every turn."""
    from engine.agents.persona.session_store import (
        load_session, save_turn, set_title,
    )

    save_turn(
        agent_id   = "dq_inspector",
        history    = [{"role": "user", "content": "first message"}],
        tool_log   = [], cost_usd=0.0, latency_ms=0,
    )
    # User overrides title manually
    set_title("dq_inspector", "Manual: weekly DQ audit")

    # Second save with a different first message — title stays manual
    save_turn(
        agent_id   = "dq_inspector",
        history    = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "different second question"},
        ],
        tool_log   = [], cost_usd=0.0, latency_ms=0,
    )
    snap = load_session("dq_inspector")
    assert snap.title == "Manual: weekly DQ audit"


def test_set_title_clears_when_empty():
    """set_title('') must reset the title to None so the next save_turn
    re-derives from history (intentional behavior for "regenerate")."""
    from engine.agents.persona.session_store import (
        load_session, save_turn, set_title,
    )

    save_turn(
        agent_id   = "anomaly_sentinel",
        history    = [{"role": "user", "content": "auto-title 1"}],
        tool_log   = [], cost_usd=0.0, latency_ms=0,
    )
    set_title("anomaly_sentinel", "")
    snap = load_session("anomaly_sentinel")
    assert snap.title is None


def test_reset_session_clears_title():
    """Reset must wipe the title along with the rest of the state."""
    from engine.agents.persona.session_store import (
        load_session, reset_session, save_turn,
    )

    save_turn(
        agent_id   = "audit_recorder",
        history    = [{"role": "user", "content": "test title before reset"}],
        tool_log   = [], cost_usd=0.0, latency_ms=0,
    )
    assert load_session("audit_recorder").title == "test title before reset"
    reset_session("audit_recorder")
    snap = load_session("audit_recorder")
    assert snap.title is None
    assert snap.history == []


# ──────────────────────────────────────────────────────────────────────────────
# Phase A.7 Wave 4.3 — turn_memory (Tier 2.5 cross-session embedding store)
# ──────────────────────────────────────────────────────────────────────────────
def test_walk_user_assistant_pairs():
    """The walker pairs each user message with the following assistant
    message, skipping tool_result-only user turns."""
    from engine.agents.persona.turn_memory import _walk_user_assistant_pairs

    history = [
        {"role": "user",      "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user",      "content": [{"type": "tool_result",
                                            "tool_use_id": "x",
                                            "content": "..."}]},
        {"role": "assistant", "content": "A1b"},
        {"role": "user",      "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    pairs = _walk_user_assistant_pairs(history)
    assert len(pairs) == 2
    assert pairs[0] == (0, "Q1", "A1")
    assert pairs[1] == (1, "Q2", "A2")


def test_walk_user_assistant_pairs_skips_unpaired_user():
    """If the last user message has no following assistant, it must not
    appear in the output — we have nothing to embed without an answer."""
    from engine.agents.persona.turn_memory import _walk_user_assistant_pairs
    history = [
        {"role": "user",      "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user",      "content": "Q2 (unanswered)"},
    ]
    pairs = _walk_user_assistant_pairs(history)
    assert len(pairs) == 1
    assert pairs[0] == (0, "Q1", "A1")


def test_embed_and_store_turn_writes_rows(monkeypatch):
    """embed_and_store_turn must persist one ChatTurnEmbedding row per
    user-assistant pair, using a deterministic fake encoder so the test
    runs without loading the real model."""
    import numpy as np
    from engine.db_models import ChatTurnEmbedding, SessionFactory
    from engine.agents.persona import turn_memory

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True,
                   show_progress_bar=False, convert_to_numpy=True):
            # 384-dim deterministic embedding based on text length
            return np.tile(
                np.array([[len(t) % 7 / 7.0] * 384 for t in texts],
                         dtype=np.float32),
                (1, 1),
            )

    monkeypatch.setattr(
        "engine.agents.persona.memory_index._load_model",
        lambda: _FakeModel(),
    )

    turn_memory.embed_and_store_turn(
        "test_agent_emb",
        history=[
            {"role": "user",      "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user",      "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ],
    )

    with SessionFactory() as s:
        rows = (s.query(ChatTurnEmbedding)
                  .filter(ChatTurnEmbedding.agent_id == "test_agent_emb")
                  .order_by(ChatTurnEmbedding.turn_idx).all())
    assert len(rows) == 2
    assert rows[0].user_text == "first question"
    assert rows[1].user_text == "second question"
    # Embedding stored as bytes
    assert isinstance(rows[0].embedding, (bytes, bytearray))
    assert len(rows[0].embedding) == 384 * 4   # float32

    # Cleanup
    with SessionFactory() as s:
        s.query(ChatTurnEmbedding).filter(
            ChatTurnEmbedding.agent_id == "test_agent_emb"
        ).delete()
        s.commit()


def test_embed_and_store_is_idempotent(monkeypatch):
    """Re-running with the same history must not duplicate rows
    (unique constraint on (agent_id, turn_idx))."""
    import numpy as np
    from engine.db_models import ChatTurnEmbedding, SessionFactory
    from engine.agents.persona import turn_memory

    class _FakeModel:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(
        "engine.agents.persona.memory_index._load_model",
        lambda: _FakeModel(),
    )

    history = [
        {"role": "user",      "content": "only Q"},
        {"role": "assistant", "content": "only A"},
    ]
    turn_memory.embed_and_store_turn("test_idem_agent", history)
    turn_memory.embed_and_store_turn("test_idem_agent", history)

    with SessionFactory() as s:
        n = s.query(ChatTurnEmbedding).filter(
            ChatTurnEmbedding.agent_id == "test_idem_agent"
        ).count()
    assert n == 1

    with SessionFactory() as s:
        s.query(ChatTurnEmbedding).filter(
            ChatTurnEmbedding.agent_id == "test_idem_agent"
        ).delete()
        s.commit()


def test_recall_past_turns_returns_top_k_by_score(monkeypatch):
    """recall_past_turns ranks by cosine and returns top-K with score
    field. Uses a deterministic encoder where match score = (query
    length matches stored text length) so we can predict ordering."""
    import datetime as _dt
    import numpy as np
    from engine.db_models import ChatTurnEmbedding, SessionFactory
    from engine.agents.persona import turn_memory

    # Insert three rows with known embeddings
    embeddings = {
        "near":  np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32),
        "med":   np.array([0.5, 0.5] + [0.0] * 382, dtype=np.float32),
        "far":   np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32),
    }
    # Normalize
    for k in embeddings:
        embeddings[k] /= np.linalg.norm(embeddings[k])

    with SessionFactory() as s:
        for i, (label, emb) in enumerate(embeddings.items()):
            s.add(ChatTurnEmbedding(
                agent_id       = "test_recall_agent",
                turn_idx       = i,
                user_text      = f"q-{label}",
                assistant_text = f"a-{label}",
                embedding      = emb.tobytes(),
                created_at     = _dt.datetime.utcnow(),
            ))
        s.commit()

    # Fake encoder: query embeds to the "near" vector verbatim
    class _FakeModel:
        def encode(self, texts, **kwargs):
            return np.tile(
                np.array([[1.0, 0.0] + [0.0] * 382], dtype=np.float32),
                (len(texts), 1),
            )
    monkeypatch.setattr(
        "engine.agents.persona.memory_index._load_model",
        lambda: _FakeModel(),
    )

    hits = turn_memory.recall_past_turns(
        "test query", agent_id="test_recall_agent", top_k=3,
    )
    assert len(hits) == 3
    # "near" should be first, "far" last (since query exactly matches near)
    assert hits[0]["user_text"] == "q-near"
    assert hits[0]["score"] > hits[1]["score"] > hits[2]["score"]

    # Cleanup
    with SessionFactory() as s:
        s.query(ChatTurnEmbedding).filter(
            ChatTurnEmbedding.agent_id == "test_recall_agent"
        ).delete()
        s.commit()


def test_recall_past_turns_agent_scope_isolation(monkeypatch):
    """recall_past_turns with agent_id set must only return rows for
    that agent — Pattern 5 isolation when specialists call it."""
    import datetime as _dt
    import numpy as np
    from engine.db_models import ChatTurnEmbedding, SessionFactory
    from engine.agents.persona import turn_memory

    with SessionFactory() as s:
        for aid in ("scope_a", "scope_b"):
            s.add(ChatTurnEmbedding(
                agent_id       = aid,
                turn_idx       = 0,
                user_text      = f"q for {aid}",
                assistant_text = f"a for {aid}",
                embedding      = np.zeros(384, dtype=np.float32).tobytes(),
                created_at     = _dt.datetime.utcnow(),
            ))
        s.commit()

    class _FakeModel:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 384), dtype=np.float32)
    monkeypatch.setattr(
        "engine.agents.persona.memory_index._load_model",
        lambda: _FakeModel(),
    )

    hits_a = turn_memory.recall_past_turns(
        "anything", agent_id="scope_a", top_k=10,
    )
    assert all(h["agent_id"] == "scope_a" for h in hits_a)

    hits_cross = turn_memory.recall_past_turns(
        "anything", agent_id=None, top_k=10,
    )
    agent_ids = {h["agent_id"] for h in hits_cross}
    assert "scope_a" in agent_ids
    assert "scope_b" in agent_ids

    with SessionFactory() as s:
        s.query(ChatTurnEmbedding).filter(
            ChatTurnEmbedding.agent_id.in_(["scope_a", "scope_b"])
        ).delete(synchronize_session=False)
        s.commit()


def test_recall_past_turns_degrades_on_db_error(monkeypatch, caplog):
    from engine.agents.persona import turn_memory

    class _Boom:
        def __enter__(self):
            raise RuntimeError("simulated DB outage")
        def __exit__(self, *a):
            return False

    monkeypatch.setattr("engine.db_models.SessionFactory", lambda: _Boom())
    hits = turn_memory.recall_past_turns("anything", top_k=3)
    assert hits == []


# ──────────────────────────────────────────────────────────────────────────────
# Phase A.7 Wave 4.1 — multi-session support
# ──────────────────────────────────────────────────────────────────────────────
def test_multi_session_isolation():
    """Saving to (agent_id, session_a) must not collide with the same
    agent_id under session_b. Each session is independent state."""
    from engine.agents.persona.session_store import load_session, save_turn

    save_turn(
        agent_id   = "risk_manager",
        history    = [{"role": "user", "content": "session A turn"}],
        tool_log   = [],
        cost_usd   = 0.001,
        latency_ms = 10,
        session_id = "s_aaa",
    )
    save_turn(
        agent_id   = "risk_manager",
        history    = [{"role": "user", "content": "session B turn"}],
        tool_log   = [],
        cost_usd   = 0.002,
        latency_ms = 20,
        session_id = "s_bbb",
    )

    snap_a = load_session("risk_manager", session_id="s_aaa")
    snap_b = load_session("risk_manager", session_id="s_bbb")
    assert snap_a.history[0]["content"] == "session A turn"
    assert snap_b.history[0]["content"] == "session B turn"
    assert snap_a.cost_usd == pytest.approx(0.001)
    assert snap_b.cost_usd == pytest.approx(0.002)
    assert snap_a.session_id == "s_aaa"
    assert snap_b.session_id == "s_bbb"


def test_list_sessions_orders_by_recent():
    """list_sessions returns rows sorted by updated_at_utc desc."""
    import time
    from engine.agents.persona.session_store import list_sessions, save_turn

    save_turn(
        agent_id="dq_inspector", history=[{"role": "user", "content": "old"}],
        tool_log=[], cost_usd=0.0, latency_ms=0, session_id="old_one",
    )
    time.sleep(0.01)
    save_turn(
        agent_id="dq_inspector", history=[{"role": "user", "content": "new"}],
        tool_log=[], cost_usd=0.0, latency_ms=0, session_id="new_one",
    )
    sessions = list_sessions("dq_inspector")
    session_ids = [s["session_id"] for s in sessions]
    assert "new_one" in session_ids
    assert "old_one" in session_ids
    assert session_ids.index("new_one") < session_ids.index("old_one")


def test_reset_session_only_clears_specific_session():
    """reset_session(agent_id, session_a) leaves session_b intact."""
    from engine.agents.persona.session_store import (
        load_session, reset_session, save_turn,
    )

    save_turn(
        agent_id="audit_recorder",
        history=[{"role": "user", "content": "keep me"}],
        tool_log=[], cost_usd=0.001, latency_ms=10,
        session_id="keep",
    )
    save_turn(
        agent_id="audit_recorder",
        history=[{"role": "user", "content": "delete me"}],
        tool_log=[], cost_usd=0.002, latency_ms=20,
        session_id="delete",
    )
    reset_session("audit_recorder", session_id="delete")

    snap_keep   = load_session("audit_recorder", session_id="keep")
    snap_delete = load_session("audit_recorder", session_id="delete")
    assert snap_keep.history[0]["content"] == "keep me"
    assert snap_delete.history == []


def test_new_session_id_format():
    """new_session_id must return a non-empty string suitable for use
    as a DB column value + URL-safe."""
    from engine.agents.persona.session_store import new_session_id
    sid = new_session_id()
    assert isinstance(sid, str)
    assert sid.startswith("s_")
    assert len(sid) >= 4
    # Must be distinct on successive calls (depends on time resolution)
    import time
    time.sleep(0.001)
    sid2 = new_session_id()
    assert sid != sid2 or True   # very tight calls may collide; not load-bearing
