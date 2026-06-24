"""engine.research_store.shadow_emit — write-side discipline hook.

Single-source-of-truth bridge for the legacy factor-verdict ledgers
(factory_ledger.jsonl, gate_runs.jsonl). Every successful append to
those ledgers is shadowed into research_store as a factor_verdict_filed
event, closing the gap T4.6 exposed (43 legacy RED events invisible to
the agentic loop until a manual backfill ran).

Doctrine context
----------------
Per CLAUDE.md (2026-06-02 STANDING):
  "Every git commit that completes a research work block MUST emit
   at least one event into the research store."

The legacy ledgers historically bypassed the doctrine — they wrote
directly to disk. Each new RED there became an O(N×M) producer-
consumer mismatch the audit_verifier / direction_proposer /
graveyard_collision could not see. The shadow hook is the
SYSTEMIC fix: rather than continue to backfill on demand, it
ensures every NEW legacy-ledger row produces a research_store
event in the same transaction.

Design constraints
------------------
1. NEVER raise. A shadow failure must not break the primary verdict
   ledger write. Wrap everything in try/except and log warnings.
2. Idempotent. A retry of the SAME verdict (same name + same minute)
   must not double-emit. Use a time-windowed dedupe.
3. Subject registration is automatic + idempotent.
4. Tagged with ("shadow_emit", "from_<legacy_source>") so downstream
   consumers can distinguish shadow-emitted events from primary
   emits (useful during the 2-week transition period).
5. Family inferred from the source dict (prefer caller-provided
   value; fall back to UNKNOWN).

Usage pattern (caller-side)
---------------------------
After appending to a legacy ledger:

    with _LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(res, ...) + "\n")
    # S1.B (2026-06-05) shadow into research_store
    from engine.research_store.shadow_emit import shadow_emit_factor_verdict
    shadow_emit_factor_verdict(res, source="gate_runs")

The function returns the event_id on success, None on any failure
(logged). The caller does NOT need to handle exceptions — the
helper swallows everything.

Transition plan
---------------
- Phase 1 (now): shadow co-exists with legacy ledger writes. Both
  paths produce data. Use this to verify coverage for 2 weeks.
- Phase 2 (after coverage confirmed): mark legacy ledgers as
  deprecated; new code emits directly via emit.factor_verdict and
  bypasses legacy.
- Phase 3 (later): legacy ledgers become read-only mirrors built
  from research_store events. Eventual deletion when no consumer
  reads them anymore.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


_DEDUPE_WINDOW_SEC = 60  # same name + same minute => skip


def shadow_emit_factor_verdict(
    res: dict,
    *,
    source: str,
) -> Optional[str]:
    """Shadow-emit one legacy verdict row as a research_store event.

    Returns event_id on success, None on any failure (logged at warning).
    NEVER raises — shadow failure must not break the primary ledger write.

    Args:
      res:    the verdict dict the caller just appended. Required keys
              (one of each pair, whichever the caller produced):
                name              (subject_id)
                verdict / light   ("GREEN" / "MARGINAL" / "RED")
                ts                (ISO datetime string)
              Optional keys passed through to metrics / summary:
                deflated_sr, alpha_t, reasons, mechanism, family, ...
      source: which legacy ledger this came from. Used for tags +
              provenance ("factory_ledger" / "gate_runs").
    """
    try:
        name = (res.get("name") or "").strip()
        if not name:
            return None
        verdict = (res.get("verdict") or res.get("light") or "").strip().upper()
        if verdict not in ("GREEN", "MARGINAL", "RED", "NEUTRAL"):
            return None

        ts = res.get("ts") or ""
        # Family inference: prefer caller-provided, else derive from
        # name prefix (same heuristic as T4.6 backfill — kept local
        # to avoid an import cycle).
        family = (res.get("family") or _family_from_name(name) or "OTHER")

        # Idempotency: derive a stable token from the logical legacy
        # row (subject + verdict + source + original_ts). Same row =
        # same token = dedupe by tag match. A genuine re-test under
        # different conditions produces a different ts and emits cleanly.
        token = _idempotency_token(name, verdict, source, ts)
        if _recently_emitted_by_token(name, token):
            return None

        # Subject register (idempotent — re-register returns existing)
        from engine.research_store import registry
        from engine.research_store.schema import SubjectType
        try:
            registry.register_subject(
                name,
                subject_type=SubjectType.factor,
                family=family,
                description=f"Auto-registered via shadow_emit from {source}",
                created_by=f"shadow_emit:{source}",
            )
        except Exception as exc:
            # If registration itself fails, log + give up on this shadow.
            logger.warning(
                "shadow_emit: subject register failed for %s: %s", name, exc)
            return None

        # Build the event
        summary = _build_summary(res, source)
        metrics = _build_metrics(res, source)

        from engine.research_store import emit as rs_emit
        event_id = rs_emit.factor_verdict(
            subject_id=name,
            verdict=verdict,
            metrics=metrics,
            artifacts={"source_ledger": _source_path(source)},
            summary=summary,
            family=family,
            tags=("shadow_emit", f"from_{source}", token),
            actor=f"shadow_emit:{source}",
        )
        return event_id

    except Exception as exc:
        # Catch-all: shadow MUST NOT break the primary write
        logger.warning(
            "shadow_emit_factor_verdict failed for %s (source=%s): %s",
            res.get("name", "<no-name>"), source, exc,
        )
        return None


# ── Internals ───────────────────────────────────────────────────────


# Same prefix→family heuristic as scripts/backfill_literature_graveyard.py.
# Duplicated intentionally to avoid script→engine import.
_NAME_TO_FAMILY = [
    ("carry_equity_div",         "CARRY"),
    ("bond_carry",               "CARRY"),
    ("bond_xsmom",               "CROSS_ASSET_MOMENTUM"),
    ("cmdty_fx_rates_carry",     "CARRY"),
    ("carry_tsmom",              "CARRY"),
    ("g10_xc_carry",             "CARRY"),
    ("vix_conditional_carry",    "CARRY"),
    ("credit_spread_carry",      "CARRY"),
    ("vix_carry",                "CARRY"),
    ("iii4_credit_spread_carry", "CARRY"),
    ("vrp_",                     "VOL_RISK_PREMIUM"),
    ("iv_skew",                  "OPTIONS_IMPLIED"),
    ("K1_BAB",                   "LOW_VOL"),
    ("insider_",                 "HOLDINGS_BASED"),
    ("lazy_prices",              "ATTENTION"),
    ("lazy_lm",                  "SENTIMENT"),
    ("news_attention",           "ATTENTION"),
    ("news_ess",                 "SENTIMENT"),
    ("analyst_revision",         "EARNINGS_DRIFT"),
    ("sue_rev",                  "EARNINGS_DRIFT"),
    ("KOR_PEAD",                 "EARNINGS_DRIFT"),
    ("supplychain_mom",          "SUPPLY_CHAIN"),
    ("merger_arb",               "OTHER"),
    ("patents_ie",               "OTHER"),
    ("regime_overlay",           "OTHER"),
    ("sector_leadlag",           "CROSS_ASSET_MOMENTUM"),
    ("PATH_",                    "OTHER"),
    ("CTA_",                     "MOMENTUM"),
    ("AC_proxy",                 "OTHER"),
]


def _family_from_name(name: str) -> Optional[str]:
    n = (name or "").lower()
    for prefix, fam in _NAME_TO_FAMILY:
        p = prefix.lower()
        if n.startswith(p) or p in n:
            return fam
    return None


def _idempotency_token(name: str, verdict: str, source: str, original_ts: str) -> str:
    """Stable tag derived from (subject, verdict, source, original_ts).
    Same logical legacy-ledger row -> same token -> dedupe via tag match.

    The token format is fixed so a future re-run with the same source
    row produces the SAME token, regardless of emit_ts wall-clock drift.
    """
    import hashlib
    key = f"{name}|{verdict}|{source}|{(original_ts or '')[:19]}"
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=6).hexdigest()
    return f"shadow_id:{h}"


def _recently_emitted_by_token(name: str, token: str) -> bool:
    """Idempotency check: True if any prior factor_verdict event for
    this subject already carries the same shadow_id:<hex> tag. The
    token encodes the logical legacy row, so a re-run produces the
    same token and we suppress the duplicate emit.
    """
    try:
        from engine.research_store import store
        prior = store.filter_events(
            event_type="factor_verdict_filed",
            subject_id=name,
            limit=40,
        )
    except Exception:
        return False
    for ev in prior:
        if token in (ev.tags or ()):
            return True
    return False


def _build_summary(res: dict, source: str) -> str:
    """Compose a 1-2 sentence summary from the legacy verdict fields.
    Capped at 395 chars (research_store schema limit is 400)."""
    reasons = res.get("reasons") or []
    head = "; ".join(str(x) for x in reasons[:2]) if reasons else ""
    stats_bits = []
    for key, lbl in (("deflated_sr", "deflSR"), ("net_deflated_sr", "net"),
                      ("alpha_t_ff5umd_pead", "alpha_t"),
                      ("residual_alpha_t", "alpha_t"),
                      ("oos_sharpe", "oos_sharpe"),
                      ("corr_with_book", "book_corr")):
        v = res.get(key)
        if v is not None:
            stats_bits.append(f"{lbl}={v}")
    stats = " ".join(stats_bits)
    parts = [f"[shadow from {source}]"]
    if head:
        parts.append(head)
    if stats:
        parts.append(f"({stats})")
    return " ".join(parts)[:395]


def _build_metrics(res: dict, source: str) -> dict:
    """Pull the load-bearing numeric fields from the legacy row so
    audit_verifier C4 (n_trials check) etc. have something to work with."""
    out: dict = {"shadow_source": source}
    for key in ("n_trials", "n_obs", "n_months",
                 "deflated_sr", "net_deflated_sr",
                 "alpha_t_ff5umd", "alpha_t_ff5umd_pead", "residual_alpha_t",
                 "standalone_sharpe", "oos_sharpe", "corr_with_book",
                 "frequency", "benchmark", "cost_class", "annual_turnover"):
        v = res.get(key)
        if v is not None:
            out[key] = v
    out["original_ts"] = res.get("ts", "")
    return out


def _source_path(source: str) -> str:
    """Map source label to its legacy ledger artifact path. Existence
    is checked by emit.factor_verdict's artifact validator."""
    if source == "gate_runs":
        return "data/research/gate_runs.jsonl"
    if source == "factory_ledger":
        return "data/validation/factory_ledger.jsonl"
    return f"data/research_store/shadow_unknown_{source}.jsonl"
