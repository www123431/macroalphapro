"""scripts/cron/check_llm_provider_health.py — single cheap provider probe.

Purpose: pre-flight check for cron jobs that consume expensive LLM
workloads. Costs ~$0.0001 per probe (Haiku, 10 tokens in / 10 out).
Saves ~$0.30 per skipped failed-burndown run when API is down.

Exit codes:
  0   provider healthy — caller proceeds
  10  provider returned BadRequestError (auth / billing / model)
  11  provider returned timeout / network error
  12  unknown failure
  20  no API key configured

Usage:
  python scripts/cron/check_llm_provider_health.py --provider anthropic
  python scripts/cron/check_llm_provider_health.py --provider deepseek
  python scripts/cron/check_llm_provider_health.py --provider anthropic --quiet

Built 2026-06-22 in service of W5-a-B (FORWARD cron resilience).
Per [[llm-credit-conservation-standing-2026-06-22]], wraps the
cheapest possible health-check call: 10 tokens in / 10 tokens out
on the cheapest model per provider.

Anthropic credit-balance failure mode (the actual incident this
script defends against):

  2026-06-18 cron run -> Sonnet -> "Your credit balance is too low"
  2026-06-21 cron run -> Sonnet -> same error, 5min wall-clock wasted
  2026-06-22 cron run -> would be same WITHOUT this precheck

With precheck: wrapper bat skips the burndown call entirely,
writes a one-line diagnostic to the cron log, exits clean.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Load API keys from .streamlit/secrets.toml so the probe works
# from cron / Task Scheduler context (env may be minimal).
_SECRETS = REPO_ROOT / ".streamlit" / "secrets.toml"
if _SECRETS.is_file():
    for _ln in _SECRETS.read_text(encoding="utf-8").splitlines():
        if "=" in _ln and not _ln.strip().startswith("#"):
            _k, _, _v = _ln.partition("=")
            _v = _v.strip().strip('"').strip("'")
            os.environ.setdefault(_k.strip(), _v)


# Cheapest available model per provider.
_PROBE_MODEL = {
    "anthropic": "claude-haiku-4-5",
    "deepseek":  "deepseek-v4-pro",
}


def _check_anthropic(quiet: bool = False) -> int:
    """One ~10-token Haiku call. Returns exit code per top-of-file enum."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        if not quiet:
            print("[health] anthropic: no ANTHROPIC_API_KEY set")
        return 20
    try:
        import anthropic
    except ImportError:
        if not quiet:
            print("[health] anthropic: anthropic SDK not installed")
        return 12
    try:
        client = anthropic.Anthropic()
        r = client.messages.create(
            model      = _PROBE_MODEL["anthropic"],
            max_tokens = 10,
            messages   = [{"role": "user", "content": "ping"}],
        )
        # Strip non-ascii before print — Windows GBK console crashes on
        # unicode emoji that Anthropic models sometimes return.
        text = (r.content[0].text if r.content else "")[:30]
        ascii_text = text.encode("ascii", "replace").decode("ascii")
        if not quiet:
            print(f"[health] anthropic: OK (reply: {ascii_text!r})")
        return 0
    except anthropic.BadRequestError as exc:
        if not quiet:
            print(f"[health] anthropic: BadRequestError "
                  f"(likely billing/credit): {str(exc)[:200]}")
        return 10
    except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
        if not quiet:
            print(f"[health] anthropic: network/timeout: {str(exc)[:200]}")
        return 11
    except Exception as exc:
        if not quiet:
            print(f"[health] anthropic: unknown failure "
                  f"({type(exc).__name__}): {str(exc)[:200]}")
        return 12


def _check_deepseek(quiet: bool = False) -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        if not quiet:
            print("[health] deepseek: no DEEPSEEK_API_KEY set")
        return 20
    try:
        from engine.llm.providers.deepseek_provider import call_deepseek
    except ImportError as exc:
        if not quiet:
            print(f"[health] deepseek: import failed: {exc}")
        return 12
    try:
        # Deepseek is a reasoning model — need ~50 tokens for "ping" reply
        # so it has reasoning + output budget. Still < $0.0001 per probe.
        r = call_deepseek(
            model      = _PROBE_MODEL["deepseek"],
            system     = "Reply with one word: pong.",
            messages   = [{"role": "user", "content": "ping"}],
            max_tokens = 50,
        )
        if not quiet:
            print(f"[health] deepseek: OK "
                  f"(output_tokens={r.output_tokens})")
        return 0
    except Exception as exc:
        msg = str(exc)[:200]
        # Heuristic: API-key / balance errors typically have 401/400 in msg
        if "401" in msg or "402" in msg or "balance" in msg.lower():
            if not quiet:
                print(f"[health] deepseek: auth/billing: {msg}")
            return 10
        if not quiet:
            print(f"[health] deepseek: failure ({type(exc).__name__}): {msg}")
        return 12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", choices=["anthropic", "deepseek"],
                       default="anthropic")
    ap.add_argument("--quiet", action="store_true",
                       help="Suppress stdout; just exit with status code.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.provider == "anthropic":
        return _check_anthropic(quiet=args.quiet)
    return _check_deepseek(quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
