"""tests/test_overlay_executor.py — L2 operator-overlay executor (2026-05-24).

Pins the deterministic core (validation + apply) and the full
propose(overlay) → approve → EXECUTE loop. This is the piece that turns the agent
from "proposes only" into "you command, it executes" — while keeping
0-LLM-in-DECISION (the executor is pure code behind a human approval) and never
touching the systematic book (isolated file-backed store).

The store paths are monkeypatched to a tmp dir so the real data/overlay/ is untouched;
DB rows are cleaned up so the real Approvals inbox is never polluted.
"""
from __future__ import annotations

import json

import pytest

import engine.overlay_executor as ox
from engine.memory import PendingApproval, SessionFactory, resolve_pending_approval


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    pos = tmp_path / "positions.json"
    trades = tmp_path / "trades.jsonl"
    monkeypatch.setattr(ox, "OVERLAY_DIR", tmp_path)
    monkeypatch.setattr(ox, "POSITIONS_PATH", pos)
    monkeypatch.setattr(ox, "TRADES_PATH", trades)
    return tmp_path


def _delete(pid: int) -> None:
    with SessionFactory() as s:
        row = s.get(PendingApproval, pid)
        if row is not None:
            s.delete(row)
            s.commit()


# ── deterministic validation ──────────────────────────────────────────────────

def test_validate_accepts_in_budget(tmp_store):
    ok, _ = ox.validate_overlay_intent("GLD", 0.05)
    assert ok


def test_validate_rejects_over_single_name_cap(tmp_store):
    ok, reason = ox.validate_overlay_intent("GLD", 0.50)
    assert not ok and "single-name" in reason


def test_validate_rejects_over_gross_cap(tmp_store):
    # fill the sleeve near the gross cap, then a new name tips it over
    ox.apply_overlay("AAA", 0.10)
    ox.apply_overlay("BBB", 0.10)
    ok, reason = ox.validate_overlay_intent("CCC", 0.10)  # 0.30 > 0.25 gross cap
    assert not ok and "gross" in reason


def test_validate_rejects_bad_ticker(tmp_store):
    ok, _ = ox.validate_overlay_intent("", 0.05)
    assert not ok
    ok, _ = ox.validate_overlay_intent("GL D!", 0.05)
    assert not ok


def test_validate_rejects_non_numeric(tmp_store):
    ok, _ = ox.validate_overlay_intent("GLD", "lots")
    assert not ok


# ── deterministic apply ─────────────────────────────────────────────────────────

def test_apply_sets_position_and_logs_trade(tmp_store):
    res = ox.apply_overlay("GLD", 0.05, approval_id=1, rationale="test")
    assert res["ok"]
    book = ox.read_overlay()
    assert book["n"] == 1
    p = book["positions"][0]
    assert p["ticker"] == "GLD" and abs(p["weight"] - 0.05) < 1e-9
    assert abs(book["gross"] - 0.05) < 1e-9
    trades = ox.read_overlay_trades()
    assert trades and trades[0]["ticker"] == "GLD" and trades[0]["action"] == "SET"


def test_apply_short_is_signed(tmp_store):
    ox.apply_overlay("VXX", -0.04)
    book = ox.read_overlay()
    assert abs(book["net"] + 0.04) < 1e-9       # net reflects the short
    assert abs(book["gross"] - 0.04) < 1e-9      # gross is absolute


def test_apply_exit_removes_position(tmp_store):
    ox.apply_overlay("GLD", 0.05)
    ox.apply_overlay("GLD", 0.0)   # exit
    book = ox.read_overlay()
    assert book["n"] == 0
    assert ox.read_overlay_trades()[0]["action"] == "EXIT"


def test_apply_over_cap_does_not_write(tmp_store):
    res = ox.apply_overlay("GLD", 0.99)
    assert not res["ok"]
    assert ox.read_overlay()["n"] == 0   # nothing written on refusal


# ── full propose → approve → EXECUTE loop ────────────────────────────────────────

def test_propose_overlay_then_approve_executes(tmp_store):
    from engine.agents.persona.tools import execute_tool

    out, is_err = execute_tool("propose_action", {
        "kind": "overlay", "detail": "TEST(pytest): tactical GLD overlay",
        "ticker": "GLD", "suggested_weight": 0.04,
        "rationale": "overlay executor test — safe to delete.",
    })
    assert not is_err, out
    res = json.loads(out)
    assert res["ok"] and res["approval_type"] == "overlay"
    pid = res["approval_id"]
    try:
        with SessionFactory() as s:
            row = s.get(PendingApproval, pid)
            assert row.approval_type == "overlay" and row.status == "pending"

        result = resolve_pending_approval(
            approval_id=pid, approved=True, resolved_by="pytest",
            review_rationale="approve overlay", review_category="other")
        assert result["ok"] is True
        # EXECUTED: overlay branch ran (unlike advisory, which is record-only)
        assert result["exec_detail"].get("ticker") == "GLD"
        book = ox.read_overlay()
        assert book["n"] == 1 and abs(book["positions"][0]["weight"] - 0.04) < 1e-9
    finally:
        _delete(pid)


def test_propose_overlay_then_reject_no_execute(tmp_store):
    from engine.agents.persona.tools import execute_tool

    out, _ = execute_tool("propose_action", {
        "kind": "add", "detail": "TEST(pytest): overlay reject path",
        "ticker": "TLT", "suggested_weight": 0.03,
        "rationale": "overlay reject test — safe to delete.",
    })
    pid = json.loads(out)["approval_id"]
    try:
        result = resolve_pending_approval(
            approval_id=pid, approved=False, resolved_by="pytest",
            review_rationale="reject", review_category="other")
        assert result["ok"] is True
        assert ox.read_overlay()["n"] == 0   # reject → nothing executed
    finally:
        _delete(pid)


def test_propose_over_budget_is_refused_before_filing(tmp_store):
    # An over-cap intent must not even reach the inbox (validated at propose time).
    from engine.agents.persona.tools import execute_tool
    out, _ = execute_tool("propose_action", {
        "kind": "overlay", "detail": "TEST(pytest): oversized", "ticker": "GLD",
        "suggested_weight": 0.80, "rationale": "should be refused.",
    })
    res = json.loads(out)
    assert "error" in res and "risk budget" in res["error"]
