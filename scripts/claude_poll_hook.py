"""scripts/claude_poll_hook.py — Claude Code polling hook.

R5.1 — closes the last collab loop. R3.1 added /api/intents/file +
/api/intents/pending so the UI's typed CTAs file structured intents
for Claude to pick up. R2.5 added /api/sessions/forward-approvals/
pending so PM-approved hypotheses surface too. But until now there
was no SCRIPT for Claude Code to actually run on a hook to consume
those queues.

What this script does:
  1. GET /api/intents/pending          — UI-filed typed intents
  2. GET /api/sessions/forward-approvals/pending
                                       — PM-approved hypotheses not
                                         yet picked up by a session
  3. GET /api/sessions/active           — current active session
                                         (for context)
  4. Print a JSON digest to stdout:
       {
         "ts": "<iso-utc>",
         "active_session": {...} | None,
         "n_intents": N,
         "intents":  [ {...} ],
         "n_approvals": M,
         "approvals": [ {...} ],
       }

Three execution modes:

  python scripts/claude_poll_hook.py
      One-shot. Prints current state and exits 0 if there's pending
      work, exits 0 silently if nothing's pending. Suitable for a
      Claude Code hook fired on a cron-like schedule.

  python scripts/claude_poll_hook.py --watch [--interval 60]
      Long-running daemon. Polls every N seconds, prints a digest
      each cycle ONLY when the queue changed since last cycle.
      Suitable for a sidecar process while you're working in Claude
      Code.

  python scripts/claude_poll_hook.py --ack <intent_id>
      POST /api/intents/{id}/ack — mark an intent acknowledged.
      Useful when Claude starts work; the user's PendingIntents
      pill updates immediately.

  python scripts/claude_poll_hook.py --fulfill <intent_id>
                                     [--event-id <event_id>]
                                     [--note "<text>"]
      POST /api/intents/{id}/fulfill — mark the work done. Optional
      event_id links to the research_store event that captures the
      outcome; --note adds a free-form fulfillment note.

Environment:
  CLAUDE_POLL_API_BASE   default: http://localhost:8000
  CLAUDE_POLL_TIMEOUT_S  default: 10

This script intentionally has no external Python dependencies
(only stdlib urllib + json) so Claude Code can run it on any
machine without a venv setup.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import urllib.error
import urllib.request


_API_BASE = os.environ.get("CLAUDE_POLL_API_BASE", "http://localhost:8000")
_TIMEOUT  = float(os.environ.get("CLAUDE_POLL_TIMEOUT_S", "10"))


def _http(method: str, path: str, body: dict | None = None) -> tuple[int, dict | list | str]:
    """Tiny stdlib HTTP. Returns (status, body) where body is decoded
    JSON when possible, else raw text."""
    url = f"{_API_BASE.rstrip('/')}{path}"
    data = None
    headers: dict[str, str] = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            try:    return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                    return resp.status, raw
    except urllib.error.HTTPError as e:
        try:    body_text = e.read().decode("utf-8")
        except Exception:
                body_text = ""
        try:    return e.code, json.loads(body_text)
        except Exception:
                return e.code, body_text
    except urllib.error.URLError as e:
        return 0, f"connection error: {e.reason}"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── digest builder ─────────────────────────────────────────────


def build_digest(since_minutes: int = 1440) -> dict:
    """Pull all three queues into one structured digest."""
    digest: dict = {"ts": _utc_iso()}

    # active session (for context — what session are intents being
    # auto-tagged to)
    s_code, s_body = _http("GET", "/api/sessions/active")
    if s_code == 200 and isinstance(s_body, dict):
        digest["active_session"] = s_body.get("active") or None
    else:
        digest["active_session"] = None

    # pending intents
    i_code, i_body = _http(
        "GET", f"/api/intents/pending?since_minutes={since_minutes}")
    if i_code == 200 and isinstance(i_body, list):
        digest["n_intents"] = len(i_body)
        digest["intents"]   = i_body
    else:
        digest["n_intents"] = 0
        digest["intents"]   = []
        digest["intents_error"] = i_body if i_code != 200 else None

    # forward-approval queue
    a_code, a_body = _http(
        "GET", f"/api/sessions/forward-approvals/pending?since_minutes={since_minutes}")
    if a_code == 200 and isinstance(a_body, dict):
        digest["n_approvals"] = a_body.get("n_pending", 0)
        digest["approvals"]   = a_body.get("approvals", [])
    else:
        digest["n_approvals"] = 0
        digest["approvals"]   = []
        digest["approvals_error"] = a_body if a_code != 200 else None

    return digest


def digest_signature(d: dict) -> str:
    """Identity hash for the queue state. Used by --watch to suppress
    no-op redraws."""
    ids = sorted(
        [f"i:{x.get('intent_id','')[:12]}" for x in d.get("intents", [])]
      + [f"a:{x.get('hypothesis_id','')[:12]}" for x in d.get("approvals", [])]
      + [f"s:{(d.get('active_session') or {}).get('session_id','')[:12]}"]
    )
    return "|".join(ids)


# ─── commands ──────────────────────────────────────────────────


def cmd_oneshot(args) -> int:
    d = build_digest(args.since_minutes)
    has_work = d["n_intents"] > 0 or d["n_approvals"] > 0
    if has_work or args.always_print:
        print(json.dumps(d, indent=None, ensure_ascii=False))
    return 0


def cmd_watch(args) -> int:
    last_sig = ""
    interval = max(5.0, float(args.interval))
    while True:
        d = build_digest(args.since_minutes)
        sig = digest_signature(d)
        if sig != last_sig:
            last_sig = sig
            print(json.dumps(d, indent=None, ensure_ascii=False), flush=True)
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            return 0


def cmd_ack(args) -> int:
    code, body = _http("POST", f"/api/intents/{args.ack}/ack",
                       body={"ack_by": args.actor})
    print(json.dumps({"status": code, "body": body}, ensure_ascii=False))
    return 0 if code == 200 else 1


def cmd_fulfill(args) -> int:
    payload: dict = {"fulfill_by": args.actor}
    if args.event_id: payload["fulfill_event_id"] = args.event_id
    if args.note:     payload["note"]             = args.note
    code, body = _http("POST", f"/api/intents/{args.fulfill}/fulfill",
                       body=payload)
    print(json.dumps({"status": code, "body": body}, ensure_ascii=False))
    return 0 if code == 200 else 1


# ─── CLI ───────────────────────────────────────────────────────


def main() -> int:
    global _API_BASE
    p = argparse.ArgumentParser(
        prog="claude_poll_hook",
        description="Poll the typed-intent + forward-approval queues for Claude Code.",
    )
    p.add_argument("--api-base", default=None,
                   help=f"override API base (default {_API_BASE})")
    p.add_argument("--since-minutes", type=int, default=1440,
                   help="window for pending queries (default 1440 = 24h)")
    p.add_argument("--always-print", action="store_true",
                   help="one-shot mode: print digest even when both queues empty")
    p.add_argument("--watch", action="store_true",
                   help="daemon mode: loop forever, emit digest on queue change")
    p.add_argument("--interval", type=float, default=60.0,
                   help="--watch poll interval seconds (default 60)")
    p.add_argument("--ack", metavar="INTENT_ID",
                   help="mark intent acknowledged (Claude saw it)")
    p.add_argument("--fulfill", metavar="INTENT_ID",
                   help="mark intent fulfilled (Claude finished)")
    p.add_argument("--event-id", default=None,
                   help="research_store event_id to link in --fulfill")
    p.add_argument("--note", default="",
                   help="free-form note to attach to --fulfill")
    p.add_argument("--actor", default="claude",
                   help="who's acking/fulfilling (default 'claude')")
    args = p.parse_args()

    if args.api_base:
        _API_BASE = args.api_base

    if args.ack:     return cmd_ack(args)
    if args.fulfill: return cmd_fulfill(args)
    if args.watch:   return cmd_watch(args)
    return cmd_oneshot(args)


if __name__ == "__main__":
    sys.exit(main())
