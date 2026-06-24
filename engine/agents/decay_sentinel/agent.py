"""engine/agents/decay_sentinel/agent.py — Decay Sentinel daily cron entry point.

Mirrors engine.portfolio.correlation_sentinel.main + the ops_watchdog cron: build the
LIVE book config, run the DETERMINISTIC sentinel_report(), narrate it, persist a JSON
artifact, return an exit code (1 if book health == ACTION). Schedule daily via Windows
Task Scheduler alongside the paper-trade run, OR call run_daily() from an orchestrator.

0-LLM-in-DECISION: every number and verdict in the artifact comes from
engine.validation.decay_sentinel (pure math). This wrapper only persists + narrates.

Output: data/decay_sentinel/decay_sentinel_<date>.json
Exit codes: 0 = HEALTHY/WATCH, 1 = ACTION (structural decay -> re-allocate).
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "data" / "decay_sentinel"


def _f(x):
    """numpy/py float -> JSON-safe (NaN -> None)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(xf) else xf


def _jsonable_report(report: dict, narration_text: str) -> dict:
    """Pick the scalar / serializable parts of the deterministic report (drop the
    rolling Series and signal panels) into a stable artifact schema."""
    mechs = {}
    for name, h in report.get("mechanisms", {}).items():
        b = report.get("betas", {}).get(name, {})
        dcy = report.get("decay", {}).get(name, {})
        mechs[name] = {
            "role":            report.get("roles", {}).get(name),
            "weight":          _f(report.get("base_weights", {}).get(name)),
            "full_sharpe":     _f(h.get("full_sharpe")),
            "rolling_sharpe":  _f(h.get("rolling_sharpe")),
            "rolling_t":       _f(h.get("rolling_t")),
            "decay_ratio":     _f(h.get("decay_ratio")),
            "mkt_beta":        _f(b.get("beta")),
            "stress_beta":     _f(b.get("stress_beta")),
            "crisis_payoff":   _f(report.get("crisis", {}).get(name)),
            "structural_decay": bool(dcy.get("structural_decay", False)),
            "signal_ic":       _f(dcy.get("signal_ic")),
            "decay_reason":    dcy.get("reason"),
        }
    pairs = []
    for (a, b), pv in report.get("pairs", {}).items():
        xc, dd = pv["cross_corr"], pv["downside"]
        pairs.append({
            "pair":          f"{a}|{b}",
            "rolling_corr":  _f(xc.get("rolling_corr")),
            "full_corr":     _f(xc.get("full_corr")),
            "downside_corr": _f(dd.get("downside_corr")),
            "stress_corr":   _f(dd.get("stress_corr")),
            "co_drawdown_frac": _f(dd.get("co_drawdown_frac")),
        })
    return {
        "as_of":               None,  # filled by run_daily
        "window_months":       report.get("window"),
        "overall":             report.get("overall"),
        "realloc_action":      bool(report.get("realloc_action", False)),
        "n_mechanisms":        len(mechs),
        "mechanisms":          mechs,
        "pairs":               pairs,
        "base_weights":        {k: _f(v) for k, v in report.get("base_weights", {}).items()},
        "recommended_weights": {k: _f(v) for k, v in report.get("recommended_weights", {}).items()},
        "alarms":              [{"level": lvl, "message": msg} for lvl, msg in report.get("alarms", [])],
        "narrative":           narration_text,
    }


def run_daily(as_of: Optional[datetime.date] = None, *, save: bool = True,
              backend: Optional[str] = None, llm_reasoning: bool = False) -> dict:
    """Build the live book, run the deterministic report, narrate, ENRICH with
    evidence-cited reasoning, persist. Returns the JSON-able payload.

    Schema additions 2026-05-29 (Phase 1 Task II.B of research_agenda_2026-05-29):
      payload["reasoning"] = {
        mode: "deterministic" | "llm" | "deterministic_fallback_<reason>",
        overall: {book_health, narrative, recommended_action, counts},
        per_mechanism: {<name>: {status, narrative, evidence[], recommended_action, ...}}
      }
    The reasoning layer NEVER changes the deterministic verdicts; the LLM mode
    asserts equality and falls back if violated (see reasoning.narrate_with_llm).
    """
    from engine.validation.decay_sentinel import build_mechanisms, sentinel_report, _market_monthly
    from engine.agents.decay_sentinel.narrator import narrate_report
    from engine.agents.decay_sentinel.reasoning import narrate_report as narrate_reasoning

    as_of = as_of or datetime.datetime.utcnow().date()
    mechs = build_mechanisms()
    try:
        market = _market_monthly()
    except Exception as exc:
        logger.warning("market series unavailable (%s) — beta/stress/crisis metrics skipped", exc)
        market = None
    report = sentinel_report(mechs, market=market)
    narration = narrate_report(report, backend=backend)
    payload = _jsonable_report(report, narration.text)
    payload["as_of"] = as_of.isoformat()
    payload["narrative_backend"] = narration.backend
    payload["narrative_cost_usd"] = narration.cost_usd

    # ── II.B: evidence-cited reasoning enrichment ────────────────────────────
    try:
        payload["reasoning"] = narrate_reasoning(payload, llm=llm_reasoning)
    except Exception as exc:
        logger.warning("reasoning layer failed (%s) — payload still has deterministic verdicts", exc)
        payload["reasoning"] = {"mode": f"failed_{type(exc).__name__}", "error": str(exc)}

    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        out = ARTIFACT_DIR / f"decay_sentinel_{as_of.isoformat()}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Decay Sentinel artifact saved: %s", out)
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description="Decay Sentinel daily cron")
    p.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD (default today UTC)")
    p.add_argument("--save", action="store_true", help="persist JSON artifact")
    p.add_argument("--backend", type=str, default=None, help="deterministic | gemini_flash")
    p.add_argument("--llm-reasoning", action="store_true",
                   help="use Anthropic Claude for II.B reasoning (falls back to deterministic on failure)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    as_of = datetime.date.fromisoformat(args.as_of) if args.as_of else datetime.datetime.utcnow().date()

    payload = run_daily(as_of, save=args.save, backend=args.backend,
                         llm_reasoning=args.llm_reasoning)
    print("\n" + "=" * 78)
    print(payload["narrative"])
    print("=" * 78)
    if "reasoning" in payload and "overall" in payload["reasoning"]:
        ro = payload["reasoning"]["overall"]
        print(f"\n[reasoning mode: {payload['reasoning'].get('mode')}]")
        print(f"OVERALL: {ro.get('narrative')}")
        if ro.get("recommended_action"):
            print(f"ACTION:  {ro['recommended_action']}")
    return 1 if payload["overall"] == "ACTION" else 0


if __name__ == "__main__":
    sys.exit(main())
