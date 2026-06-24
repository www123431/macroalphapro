"""engine/agents/persona/turn_memory.py — Tier 2.5 cross-session memory.

Phase A.7 Wave 4.3. Reuses the all-MiniLM-L6-v2 model already loaded
by engine.agents.persona.memory_index. Each user→assistant turn pair
is embedded once and persisted to ChatTurnEmbedding (engine.db_models).

Public API:
  embed_and_store_turn(agent_id, history)
      Walks the history backward, finds the latest user→assistant pair
      whose turn_idx is not yet in the DB, embeds and stores it. Called
      from session_store.save_turn after the row write succeeds.
  recall_past_turns(query, agent_id=None, top_k=5)
      Cosine-similarity retrieval over ChatTurnEmbedding. agent_id=None
      means cross-agent (used by CoS); a specific agent_id scopes to
      that persona's history only (Pattern 5 ban compatibility).

HARKing-defense doctrine: retrieval results carry created_at + agent_id
so the calling persona prompt can age-discount. The recall_past_turns
tool description explicitly tells the model "these are history claims,
not ground truth; re-verify via current state tools before acting".
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_text(content) -> str:
    """Anthropic-format content can be str OR list[dict] of blocks.
    Returns the concatenated text content only."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
        return "\n".join(parts).strip()
    return ""


def _walk_user_assistant_pairs(history: list[dict]) -> list[tuple[int, str, str]]:
    """Walk history, return [(turn_idx, user_text, assistant_text), ...]
    where turn_idx is the 0-indexed position of the user message in the
    sequence of USER turns (so it stays stable as the conversation grows).
    Only pairs where assistant has non-empty text are returned.
    """
    pairs: list[tuple[int, str, str]] = []
    user_count = -1
    pending_user_text: str | None = None
    pending_user_idx: int = -1
    for msg in history:
        role = msg.get("role")
        if role == "user":
            user_text = _extract_text(msg.get("content", ""))
            if not user_text.strip():
                # tool_result-only user turn — skip; doesn't bump turn_idx
                continue
            user_count += 1
            pending_user_text = user_text
            pending_user_idx  = user_count
        elif role == "assistant" and pending_user_text is not None:
            assistant_text = _extract_text(msg.get("content", ""))
            if assistant_text.strip():
                pairs.append(
                    (pending_user_idx, pending_user_text, assistant_text)
                )
            pending_user_text = None
    return pairs


def embed_and_store_turn(
    agent_id:   str,
    history:    list[dict],
    session_id: str = "default",
) -> None:
    """Embed any user→assistant turn pair not yet in ChatTurnEmbedding.

    Idempotent — uses the (agent_id, session_id, turn_idx) unique
    constraint to skip already-stored turns. Called from
    session_store.save_turn. Silently degrades on failure (DB error /
    model load failure): the chat answer is already on screen,
    persistence is best-effort.
    """
    try:
        pairs = _walk_user_assistant_pairs(history)
        if not pairs:
            return

        from engine.db_models import ChatTurnEmbedding, SessionFactory

        # Find which turn_idx values are already stored for this
        # (agent_id, session_id) pair
        with SessionFactory() as s:
            existing_idx = {
                row[0] for row in
                s.query(ChatTurnEmbedding.turn_idx)
                 .filter(ChatTurnEmbedding.agent_id == agent_id,
                         ChatTurnEmbedding.session_id == session_id)
                 .all()
            }

        missing = [(i, u, a) for (i, u, a) in pairs if i not in existing_idx]
        if not missing:
            return

        # Lazy-load model (reuses the memory_index singleton if present)
        from engine.agents.persona.memory_index import _load_model
        model = _load_model()

        # Embed user+assistant concatenation — semantic richer than either
        # alone, especially for queries like "what did you say about X".
        texts = [f"{u}\n\n{a[:2000]}" for (i, u, a) in missing]
        import numpy as np
        embeddings = model.encode(
            texts,
            normalize_embeddings = True,   # cosine == dot product downstream
            show_progress_bar    = False,
            convert_to_numpy     = True,
        ).astype("float32")

        # Persist
        rows = []
        for (idx, (turn_idx, user_text, assistant_text)) in enumerate(missing):
            rows.append(ChatTurnEmbedding(
                agent_id       = agent_id,
                session_id     = session_id,
                turn_idx       = turn_idx,
                user_text      = user_text[:2000],
                assistant_text = assistant_text[:4000],
                embedding      = embeddings[idx].tobytes(),
            ))
        with SessionFactory() as s:
            for row in rows:
                # One-by-one add so a duplicate (race) doesn't kill the batch.
                try:
                    s.add(row)
                    s.commit()
                except Exception as inner:
                    s.rollback()
                    logger.debug("embed_and_store_turn: dupe / race: %s", inner)
    except Exception as exc:
        logger.warning("embed_and_store_turn(%s/%s) failed: %s",
                       agent_id, session_id, exc)


def recall_past_turns(
    query:      str,
    agent_id:   Optional[str] = None,
    top_k:      int = 5,
    session_id: Optional[str] = None,
) -> list[dict]:
    """Cosine-similarity retrieval over ChatTurnEmbedding.

    Args:
      query:      natural-language query.
      agent_id:   scope to one persona's history (Pattern 5 isolation when
                  a specialist calls this). None = cross-agent (CoS only).
      top_k:      max results.
      session_id: optional further scoping to one conversation thread.
                  None = all sessions for the matched agent.

    Returns list of dicts with: agent_id, session_id, turn_idx,
    user_text, assistant_text (truncated), score, created_at.
    Empty list on failure / no matches.
    """
    try:
        from engine.db_models import ChatTurnEmbedding, SessionFactory

        with SessionFactory() as s:
            q = s.query(ChatTurnEmbedding)
            if agent_id is not None:
                q = q.filter(ChatTurnEmbedding.agent_id == agent_id)
            if session_id is not None:
                q = q.filter(ChatTurnEmbedding.session_id == session_id)
            rows = q.order_by(ChatTurnEmbedding.created_at.desc()).limit(2000).all()

        if not rows:
            return []

        from engine.agents.persona.memory_index import _load_model
        model = _load_model()
        import numpy as np

        q_emb = model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")[0]

        # Stack stored embeddings + cosine score (dot product since both
        # normalized at embed time).
        embs = np.stack([
            np.frombuffer(r.embedding, dtype="float32") for r in rows
        ])
        scores = embs @ q_emb
        order  = np.argsort(-scores)[:int(top_k)]

        out = []
        for idx in order:
            row = rows[int(idx)]
            out.append({
                "agent_id":       row.agent_id,
                "session_id":     row.session_id,
                "turn_idx":       int(row.turn_idx),
                "user_text":      row.user_text[:300],
                "assistant_text": row.assistant_text[:500],
                "score":          round(float(scores[int(idx)]), 4),
                "created_at":     row.created_at.isoformat() if row.created_at else None,
            })
        return out
    except Exception as exc:
        logger.warning("recall_past_turns(%r, %s) failed: %s",
                       query, agent_id, exc)
        return []
