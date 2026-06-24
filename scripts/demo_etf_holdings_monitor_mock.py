"""
scripts/demo_etf_holdings_monitor_mock.py — Demo run with mocked LLM outputs.

Purpose: populate cap_state.json + last_run_summary.json + DecisionLog rows
SO THE UI HAS SOMETHING TO SHOW (per user feedback "看不见的 capability 不算
evidence" / `feedback_engineering_change_requires_ui.md`).

Uses synthetic risk scores (deterministic) to trigger 2-3 cap activations
across 24 equity ETFs. Does NOT call the real LLM — saves $1-2 budget.

Real production run: scripts/run_etf_holdings_monitor_monthly.py (calls LLM).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    from engine.etf_holdings_ingestion import (
        fetch_all_equity_etf_holdings,
        deduplicate_holdings_to_unique_names,
    )
    from engine.etf_holdings_risk_monitor import (
        screen_name,
        aggregate_etf_risk,
        trigger_etf_cap,
        _persist_cap_trigger,
        _write_decision_log_cap_activation,
        get_cost_status,
        _DATA_DIR,
    )

    as_of = datetime.date.today()
    logger.info("Demo run as_of=%s (MOCK LLM, no real cost)", as_of)

    # Step 1: fetch holdings (uses cache from earlier smoke test)
    holdings_by_etf = fetch_all_equity_etf_holdings(as_of)
    logger.info("Fetched holdings for %d ETFs", len(holdings_by_etf))

    # Step 2: dedupe
    unique_names = deduplicate_holdings_to_unique_names(holdings_by_etf)
    logger.info("%d unique names", len(unique_names))

    # Step 3: synthetic risk scores — programmatically pick names with
    # highest aggregate weight across ETFs (most concentration risk).
    # Mark them score 4 to trigger demo cap activations.
    # Other names get score 1 (no_signal). Real LLM run will compute differently.
    name_total_weight: dict[str, float] = {}
    for etf, holdings in holdings_by_etf.items():
        for h in holdings:
            n = str(h.get("name", "")).upper()
            if n:
                name_total_weight[n] = name_total_weight.get(n, 0.0) + h.get("weight", 0.0)
    # Use top 30 + score 5 to ensure several ETFs cross trigger threshold (3.5)
    # in this demo. Real LLM run typically produces 1-3 cap activations/month
    # based on actual fundamental signals (this is mock data, NOT real LLM output).
    top_concentration_names = sorted(
        name_total_weight.items(), key=lambda x: -x[1],
    )[:30]
    SYNTHETIC_HIGH_RISK = {n: 5 for n, _ in top_concentration_names}
    logger.info(
        "Synthetic high-risk names (top 30 by cross-ETF weight, score=5): %s",
        ", ".join(n for n, _ in top_concentration_names[:10]) + ", ...",
    )

    # Synthetic event_class + rationale templates by score (mimic real LLM output)
    DEMO_TEMPLATES = {
        5: ("regulatory_action",
            "DEMO MOCK: {name} synthetic SEVERE risk — top concentration name flagged "
            "for UI demonstration (e.g., regulatory enforcement action). "
            "Real LLM run will produce specific event-driven rationale."),
        4: ("earnings_warning",
            "DEMO MOCK: {name} synthetic ELEVATED risk — guidance reduction or material "
            "impairment scenario. Real LLM run uses actual SEC 8-K + news evidence."),
        3: ("sec_filing_material",
            "DEMO MOCK: {name} synthetic MODERATE risk — material event filed. "
            "Real LLM cites specific 8-K item + news article."),
        2: ("other_fundamental",
            "DEMO MOCK: {name} synthetic MINOR signal."),
        1: ("no_signal",
            "DEMO MOCK: {name} no material event detected."),
    }

    name_scores = {}
    n_screened = 0
    n_fallbacks = 0
    # Cache directory for synthetic per-name responses (so drill-down UI works)
    from engine.etf_holdings_risk_monitor import _CACHE_DIR as _EHRM_CACHE_DIR
    _EHRM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for name in sorted(unique_names):
        score = SYNTHETIC_HIGH_RISK.get(name, 1)
        event_class, rationale_template = DEMO_TEMPLATES.get(score, DEMO_TEMPLATES[1])
        injection = {
            "name":          name,
            "risk_score":    score,
            "event_class":   event_class,
            "rationale":     rationale_template.format(name=name),
            "evidence_refs": [
                f"DEMO-MOCK-{name}-{as_of.isoformat()}",
                f"synthetic-8K-ref",
            ] if score >= 3 else [],
            "as_of_date":    as_of.isoformat(),
        }
        result = screen_name(
            name=name, as_of=as_of,
            skip_llm_call=True,
            inject_classification=injection,
        )
        name_scores[name] = result["risk_score"]
        n_screened += 1
        if result.get("fallback"):
            n_fallbacks += 1

        # ALSO write synthetic cache entry so drill-down UI has data to display
        # (production code skip_llm_call path doesn't write cache; demo populates manually)
        _cache_file = _EHRM_CACHE_DIR / f"{name}_{as_of.strftime('%Y%m')}.json"
        _synthetic_cache = {
            "name":          name,
            "as_of":         as_of.isoformat(),
            "prompt_hash":   "demo_mock_prompt_hash_" + name,
            "response_hash": "demo_mock_response_hash_" + name,
            "prompt":        (
                f"[DEMO MOCK PROMPT for {name}]\n"
                f"This is a synthetic prompt populated by demo script for UI demonstration.\n"
                f"Real production prompts include: ticker, sector, recent SEC 8-K filings (last 30d),\n"
                f"recent news (last 30d, top 8), prior 30d return, next earnings date,\n"
                f"plus locked system instructions for risk classification."
            ),
            "response":      json.dumps({
                "name":          name,
                "risk_score":    score,
                "event_class":   event_class,
                "rationale":     rationale_template.format(name=name),
                "evidence_refs": injection["evidence_refs"],
                "as_of_date":    as_of.isoformat(),
            }, ensure_ascii=False),
            "input_tokens":  450,  # typical
            "output_tokens": 120,
            "cost_usd":      0.0,  # demo has 0 cost (skip_llm_call)
            "timestamp":     datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model_version": "gemini-2.5-flash",
            "_demo_note":    "MOCK CACHE — not real LLM call",
        }
        _cache_file.write_text(
            json.dumps(_synthetic_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    logger.info(
        "Mock-screened %d names (%d high-risk synthetic, %d default-low)",
        n_screened,
        sum(1 for v in name_scores.values() if v >= 3),
        sum(1 for v in name_scores.values() if v < 3),
    )

    # Step 4: aggregate per-ETF + trigger
    etf_aggregates = {}
    cap_activations = []
    for etf, holdings in holdings_by_etf.items():
        score = aggregate_etf_risk(holdings, name_scores)
        etf_aggregates[etf] = round(score, 4)
        # v2 amendment: pass holdings + name_scores for max-of fallback
        if trigger_etf_cap(score, holdings=holdings, name_scores=name_scores):
            top_contributors = sorted(
                holdings,
                key=lambda h: -name_scores.get(str(h.get("name", "")).upper(), 1) * h.get("weight", 0.0),
            )[:3]
            top_names = [str(h.get("name", "")).upper() for h in top_contributors]
            n_high = sum(
                1 for h in holdings
                if name_scores.get(str(h.get("name", "")).upper(), 1) >= 3
            )
            rationale = (
                f"[DEMO MOCK] Aggregate risk {score:.2f} ≥ 3.5; "
                f"top contributors: {', '.join(top_names)}; "
                f"{n_high} holdings ≥ score 3"
            )
            _persist_cap_trigger(etf, as_of, score, rationale)
            decision_id = _write_decision_log_cap_activation(
                etf=etf, aggregate_score=score, rationale=rationale,
                n_holdings_above_3=n_high, triggered_at=as_of,
            )
            cap_activations.append({
                "etf":              etf,
                "aggregate_score":  round(score, 4),
                "rationale":        rationale,
                "decision_log_id":  decision_id,
            })

    logger.info("%d cap activations fired (DEMO data)", len(cap_activations))
    for ca in cap_activations:
        logger.info("  → %s (score=%.2f)", ca["etf"], ca["aggregate_score"])

    # Step 5: persist summary for UI
    cost_status = get_cost_status(as_of)
    summary = {
        "as_of":              as_of.isoformat(),
        "spec_id":            49,
        "spec_hash_prefix":   "0c3696fc4145",
        "n_etfs":             len(holdings_by_etf),
        "n_with_holdings":    sum(1 for h in holdings_by_etf.values() if h),
        "n_unique_names":     len(unique_names),
        "n_screened":         n_screened,
        "n_fallbacks":        n_fallbacks,
        "n_high_score_names": sum(1 for v in name_scores.values() if v >= 3),
        "etf_aggregates":     etf_aggregates,
        "cap_activations":    cap_activations,
        "n_cap_activations":  len(cap_activations),
        "cost_status":        cost_status,
        "completed_at":       datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "_demo_note":         "MOCK DATA — synthetic risk scores; NOT real LLM output",
    }

    summary_path = _DATA_DIR / "last_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Demo summary persisted to %s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as exc:
        logger.error("Demo run failed: %s", exc, exc_info=True)
        sys.exit(1)
