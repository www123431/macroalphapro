"""
engine/agents/persona/tools.py — read-only tool implementations + schemas.

Three tools the Risk Manager agent can call autonomously to answer
user questions. All read-only — the agent never mutates state.

Tool surface (Anthropic JSON schema):
  query_recent_alerts(days_back, severity_min, source)
    → list of RM + DQ alerts from the database

  read_today_book_state()
    → per-strategy status + book gross/net/n_positions for today

  lookup_strategy_status(strategy_name)
    → details on one strategy's signal output today

Return format: each tool returns a string (JSON-encoded for structured
data) — Anthropic tool_result content accepts string or list of text
blocks. We use JSON-encoded strings so the model can parse fields.

Failure mode: each tool wraps internal calls in try/except and returns
an error string (NOT raise) so the agent can recover. Returning an
error result with `is_error=True` upstream is what the agent caller
should do; we just give it the error message text.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations (each returns a string — JSON-encoded if structured)
# ──────────────────────────────────────────────────────────────────────────────
def query_recent_alerts(
    days_back:    int = 7,
    severity_min: str = "LIGHT",
    source:       str = "all",
) -> str:
    """Query RM + DQ alert tables.

    Args:
      days_back:    look back N days from today
      severity_min: minimum severity ("LIGHT" | "MEDIUM" | "SEVERE")
      source:       "rm" | "dq" | "all"
    """
    try:
        results: list[dict] = []

        if source in ("rm", "all"):
            try:
                from engine.agents.risk_manager.persist import query_recent_alerts as q_rm
                rm_alerts = q_rm(days_back=days_back, severity_min=severity_min)
                for a in rm_alerts:
                    results.append({
                        "source":           "risk_manager",
                        "date":             str(a.get("date")),
                        "mode_id":          a.get("mode_id"),
                        "severity":         a.get("severity"),
                        "cb_severity":      a.get("cb_severity"),
                        "rule_description": a.get("rule_description", "")[:200],
                        "halt_decision":    a.get("halt_decision"),
                        "affected":         a.get("affected"),
                    })
            except Exception as exc:
                logger.warning("query_recent_alerts: RM source failed: %s", exc)

        if source in ("dq", "all"):
            try:
                from engine.agents.dq_inspector.persist import query_recent_alerts as q_dq
                dq_alerts = q_dq(days_back=days_back, severity_min=severity_min)
                for a in dq_alerts:
                    results.append({
                        "source":           "dq_inspector",
                        "date":             str(a.get("date")),
                        "mode_id":          a.get("mode_id"),
                        "severity":         a.get("severity"),
                        "cb_severity":      a.get("cb_severity"),
                        "rule_description": a.get("rule_description", "")[:200],
                        "halt_decision":    a.get("halt_decision"),
                        "source_id":        a.get("source_id"),
                    })
            except Exception as exc:
                logger.warning("query_recent_alerts: DQ source failed: %s", exc)

        if not results:
            return json.dumps({
                "n_alerts": 0,
                "message":  f"No alerts found (last {days_back} days, "
                            f"severity >= {severity_min}, source={source}).",
            })

        return json.dumps({
            "n_alerts": len(results),
            "alerts":   results,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"query_recent_alerts failed: {exc}"})


def read_today_book_state(as_of: str | None = None) -> str:
    """Read the paper-trade book state from the cached UI artifact.

    Avoids re-running the orchestrator (which calls yfinance) on each
    user question — uses the persisted artifact.

    Args:
      as_of: ISO date "YYYY-MM-DD" — read the latest artifact ON OR BEFORE this
             date (time-travel). None → the most recent artifact.
    """
    try:
        from pathlib import Path
        artifact_dir = Path("data/ui_artifact")
        if not artifact_dir.exists():
            return json.dumps({"error": "no ui_artifact directory"})
        artifacts = sorted(artifact_dir.glob("*.json"), reverse=True)
        if not artifacts:
            return json.dumps({"error": "no artifact files"})
        if as_of:
            picked = next((a for a in artifacts if a.stem <= as_of), None)
            latest = picked or artifacts[-1]   # nothing on/before as_of → earliest available
        else:
            latest = artifacts[0]
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract a compact summary the agent can reason about. Handle the current artifact
        # schema (v2: _meta / strategy_states / book_snapshot) AND the legacy schema
        # (meta / strategies / combined) so an older artifact still parses.
        meta       = data.get("_meta") or data.get("meta") or {}
        strategies = data.get("strategy_states") or data.get("strategies") or []
        book       = data.get("book_snapshot") or data.get("combined") or {}

        strat_summary = []
        for s in strategies:
            strat_summary.append({
                "name":        s.get("strategy_name"),
                "sleeve":      s.get("sleeve_id"),
                "status":      s.get("status"),
                "n_positions": s.get("n_positions"),
                "intra_w":     s.get("intra_sleeve_weight", s.get("intra_sleeve_w")),
                "notes":       (s.get("doctrine") or s.get("notes") or "")[:160],
            })

        # target gross = sum of strategy book_weights (the intended leverage, e.g. 1.50×).
        # Realized gross is usually lower (a NO_SIGNAL sleeve contributes 0 + cross-sleeve netting).
        target_gross = round(sum((s.get("book_weight") or 0.0) for s in strategies), 4) if strategies else None

        summary = {
            "as_of":             meta.get("as_of_date") or meta.get("as_of") or latest.stem,
            "artifact_path":     str(latest),
            "strategies":        strat_summary,
            "combined_gross":    book.get("gross"),
            "combined_target_gross": target_gross,
            "combined_net":      book.get("net"),
            "combined_n":        book.get("n_tickers", book.get("n_positions")),
            "sleeve_attribution": data.get("sleeve_attribution"),
        }
        return json.dumps(summary, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"read_today_book_state failed: {exc}"})


def lookup_spec(spec_id: int) -> str:
    """Look up a registered spec by spec_id (Tier 3a structured fact).

    Reads from engine.preregistration.list_specs (SpecRegistry SQLite table).
    Returns spec metadata + full amendment log so agent can answer questions
    like "why was K1 BAB hash changed" or "what amendments touched spec 69".

    Args:
      spec_id: integer SpecRegistry.id (e.g. 69 for Risk Manager, 70 for DQ).
    """
    try:
        from engine.preregistration import list_specs
        specs = list_specs()
        match = next((s for s in specs if int(s.get("id", -1)) == int(spec_id)), None)
        if match is None:
            return json.dumps({
                "error":          f"spec_id {spec_id} not found",
                "available_ids":  sorted(int(s.get("id", 0)) for s in specs)[:20],
                "n_total_specs":  len(specs),
            })
        return json.dumps({
            "spec_id":             match.get("id"),
            "spec_path":           match.get("spec_path"),
            "current_hash":        match.get("current_hash"),
            "git_blob_hash":       match.get("git_blob_hash"),
            "status":              match.get("status"),
            "registered_at":       match.get("registered_at"),
            "retro_registered":    match.get("retro_registered"),
            "n_amendments":        match.get("n_amendments"),
            "n_trials_contributed": match.get("n_trials_contributed"),
            "amendment_log":       match.get("amendment_log", [])[:10],   # cap at 10 newest
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"lookup_spec failed: {exc}"})


def read_project_memory(
    query:        str,
    max_results:  int = 5,
    mode:         str = "auto",
) -> str:
    """Search project memory files (Tier 3b curated knowledge).

    Three retrieval modes (set ``mode``):

      "exact" — query matches a memory file's filename stem. Returns
                the full file content (capped at 4000 chars).
      "keyword" — substring search across description + body. Returns
                  top-N matching files with description preview.
      "semantic" — sentence-transformers all-MiniLM-L6-v2 embedding
                   search (Phase A.7 Wave 3.3, 2026-05-19). Catches
                   paraphrases that keyword search misses.
      "auto" (default) — try exact first, then semantic if the query
                         looks like a question / phrase (>= 2 words),
                         otherwise keyword. Falls back gracefully if
                         the embedding index is unavailable.

    Memory files live at ~/.claude/projects/<project-slug>/memory/*.md
    and contain human-curated project state, feedback rules, and
    architectural decisions. They are the agent's long-term memory
    layer (Tier 3b).

    Args:
      query:       filename stem OR keyword OR natural-language question
      max_results: max files to return (default 5)
      mode:        "exact" | "keyword" | "semantic" | "auto"
    """
    try:
        from pathlib import Path

        # Derive Claude Code memory directory from current working dir
        cwd = Path.cwd().resolve()
        sanitized = (
            str(cwd).replace(":", "-").replace("\\", "-").replace("/", "-")
        )
        memory_dir = Path.home() / ".claude" / "projects" / sanitized / "memory"

        # Fallback: if path doesn't exist try without leading slash quirks
        if not memory_dir.exists():
            # Try lowercase drive variant
            sanitized_alt = sanitized.lower() if sanitized[0].isupper() else sanitized
            memory_dir_alt = Path.home() / ".claude" / "projects" / sanitized_alt / "memory"
            if memory_dir_alt.exists():
                memory_dir = memory_dir_alt
            else:
                return json.dumps({
                    "error": "memory directory not found",
                    "tried": [str(memory_dir), str(memory_dir_alt)],
                    "hint":  "Claude Code auto-memory layout assumed; "
                             "if project moved, the path won't resolve.",
                })

        all_files = sorted(memory_dir.glob("*.md"))
        if not all_files:
            return json.dumps({
                "error": "no memory files found",
                "dir":   str(memory_dir),
            })

        q_lower = query.lower().strip()
        valid_modes = ("exact", "keyword", "semantic", "auto")
        if mode not in valid_modes:
            return json.dumps({
                "error":         f"unknown mode {mode!r}",
                "valid_modes":   list(valid_modes),
            })

        # Mode 1: exact name match (filename stem). Runs in "exact" and
        # "auto" modes. Always wins over semantic / keyword if hit.
        if mode in ("exact", "auto"):
            for f in all_files:
                stem = f.stem.lower()
                if stem == q_lower or stem == q_lower.replace(" ", "_"):
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    return json.dumps({
                        "mode":     "exact_name_match",
                        "file":     f.name,
                        "content":  content[:4000],
                        "truncated": len(content) > 4000,
                        "full_chars": len(content),
                    }, ensure_ascii=False)
            if mode == "exact":
                return json.dumps({
                    "mode":   "exact_name_match",
                    "query":  query,
                    "n_hits": 0,
                    "hint":   "No exact filename match. Try mode='keyword' or "
                              "mode='semantic' for substring / question-style search.",
                    "available_recent": [f.name for f in all_files[-10:]],
                })

        # Mode 2: semantic search (sentence-transformers). Runs in
        # "semantic" mode and in "auto" mode IF the query looks like a
        # natural-language question (>=2 words, has a space, no
        # underscores indicating a slug).
        wants_semantic = mode == "semantic" or (
            mode == "auto"
            and " " in q_lower
            and len(q_lower) >= 8
            and "_" not in q_lower
        )
        if wants_semantic:
            try:
                from engine.agents.persona.memory_index import search_memory
                hits = search_memory(query, top_k=int(max_results))
            except Exception as exc:
                hits = []
                logger.warning("read_project_memory: semantic failed: %s", exc)
            if hits:
                return json.dumps({
                    "mode":   "semantic_search",
                    "query":  query,
                    "n_hits": len(hits),
                    "top":    hits,
                    "hint":   "Scores are cosine similarity (0-1, higher=closer). "
                              "Call read_project_memory(<file_stem>) for full content.",
                }, ensure_ascii=False, default=str)
            # Semantic failed / empty — fall through to keyword if in auto
            if mode == "semantic":
                return json.dumps({
                    "mode":   "semantic_search",
                    "query":  query,
                    "n_hits": 0,
                    "hint":   "Semantic index returned no results. Try mode='keyword'.",
                })

        # Mode 3: keyword substring search across name + description + body
        matches: list[dict] = []
        for f in all_files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            text_lower = text.lower()
            if q_lower in text_lower or q_lower in f.stem.lower():
                # Extract `description:` line for preview
                desc = ""
                for line in text.split("\n")[:15]:
                    if line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip()
                        break
                matches.append({
                    "file":         f.name,
                    "description":  desc[:300],
                    "match_count":  text_lower.count(q_lower),
                })

        if not matches:
            return json.dumps({
                "mode":   "keyword_search",
                "query":  query,
                "n_hits": 0,
                "hint":   f"No files matched. Try a different keyword, "
                          f"exact slug (e.g. 'feedback_no_emojis_2026-05-19'), "
                          f"or mode='semantic' for question-style search.",
                "available_recent": [f.name for f in all_files[-10:]],
            })

        # Sort by match_count desc, take top N
        matches.sort(key=lambda m: -m["match_count"])
        return json.dumps({
            "mode":   "keyword_search",
            "query":  query,
            "n_hits": len(matches),
            "top":    matches[:max_results],
            "hint":   "Call read_project_memory(<file_stem>) for full content.",
        }, ensure_ascii=False, default=str)

    except Exception as exc:
        return json.dumps({"error": f"read_project_memory failed: {exc}"})


def query_recent_anomalies(
    ticker:         str | None = None,
    days_back:      int = 14,
    min_confidence: int = 2,
    detector:       str = "all",
) -> str:
    """Query the AnomalyFlag table (forensic per-ticker anomaly history).

    Args:
      ticker:         filter to one ticker (None = cross-ticker view)
      days_back:      look back N calendar days (default 14)
      min_confidence: minimum confidence_likert 1-5 (default 2 = solid)
      detector:       "rule_baseline_a" | "rule_baseline_b" | "llm" | "all"
    """
    try:
        import datetime as _dt
        from engine.db_models import AnomalyFlag
        from engine.memory import SessionFactory

        cutoff = _dt.date.today() - _dt.timedelta(days=int(days_back))
        with SessionFactory() as s:
            q = s.query(AnomalyFlag).filter(AnomalyFlag.scan_date >= cutoff)
            if ticker:
                q = q.filter(AnomalyFlag.ticker == ticker.upper())
            if detector != "all":
                q = q.filter(AnomalyFlag.detector == detector)
            q = q.filter(AnomalyFlag.confidence_likert >= int(min_confidence))
            q = q.order_by(AnomalyFlag.scan_date.desc(),
                           AnomalyFlag.confidence_likert.desc())
            rows = q.limit(50).all()

        if not rows:
            return json.dumps({
                "n_flags":  0,
                "message":  f"No anomalies found (ticker={ticker or 'any'}, "
                            f"last {days_back}d, confidence>={min_confidence}, "
                            f"detector={detector}).",
            })

        out = []
        for r in rows:
            out.append({
                "scan_date":         str(r.scan_date),
                "ticker":            r.ticker,
                "sector":            r.sector,
                "detector":          r.detector,
                "event_class":       r.event_class,
                "confidence_likert": r.confidence_likert,
                "horizon_days":      r.horizon_days,
                "evidence":          (r.evidence_summary or "")[:240],
            })
        return json.dumps({
            "n_flags": len(out),
            "flags":   out,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"query_recent_anomalies failed: {exc}"})


# Forensic-check thresholds. Match engine.anomaly_screener rule_baseline_a
# semantics but inline here so this tool stays independent of any future
# anomaly_screener refactor.
_FORENSIC_PRICE_SIGMA       = 2.0    # |daily return| / 60d-sigma threshold
_FORENSIC_MIN_ABS_RETURN    = 0.01   # economic-significance floor: also require a >=1% absolute
                                     # move, so a high z-score on a low-vol ETF (a 0.2% "9σ" day)
                                     # doesn't false-flag. Mirrors anomaly_screener.
_FORENSIC_VOLUME_MULT       = 3.0    # today vol / 30d median multiple
_FORENSIC_ROLLING_VOL_WIN   = 60
_FORENSIC_VOLUME_MEDIAN_WIN = 30
_FORENSIC_DRAWDOWN_LOOKBACK = 90


def forensic_ticker_check(ticker: str, as_of: str | None = None) -> str:
    """Run live forensic rule detectors against ``ticker``.

    Three checks (mirroring engine.anomaly_screener rule_baseline_a):
      - price z-score: |daily return| / 60d sigma  (threshold 2.0)
      - volume multiple: today vol / 30d median    (threshold 3.0)
      - drawdown: peak-to-current over 90d         (threshold -10%)

    Args:
      ticker: ticker symbol (case-insensitive)
      as_of:  ISO date string ("YYYY-MM-DD"). None → today.
    """
    try:
        import datetime as _dt
        import yfinance as yf

        tkr = ticker.upper().strip()
        scan_date = (
            _dt.date.fromisoformat(as_of) if as_of else _dt.date.today()
        )
        start = scan_date - _dt.timedelta(days=_FORENSIC_DRAWDOWN_LOOKBACK + 30)
        end_excl = scan_date + _dt.timedelta(days=1)
        df = yf.download(
            tkr, start=str(start), end=str(end_excl),
            auto_adjust=True, progress=False, multi_level_index=False,
        )
        if df is None or df.empty:
            return json.dumps({
                "error":  f"no price history for {tkr} ending {scan_date}",
                "hint":   "yfinance returned empty; confirm symbol + market hours",
            })
        df.index = [d.date() if hasattr(d, "date") else d for d in df.index]

        closes = df["Close"].dropna()
        if len(closes) < _FORENSIC_ROLLING_VOL_WIN + 2:
            return json.dumps({
                "error":  f"insufficient history for {tkr}: "
                          f"{len(closes)} bars; need {_FORENSIC_ROLLING_VOL_WIN + 2}",
            })

        rule_hits: list[dict] = []

        # ── Rule 1: price z-score ────────────────────────────────────────
        rets = closes.pct_change().dropna()
        rets = rets[[d for d in rets.index if d <= scan_date]]
        if len(rets) >= _FORENSIC_ROLLING_VOL_WIN + 1:
            last_ret = float(rets.iloc[-1])
            sigma_60 = float(rets.iloc[-_FORENSIC_ROLLING_VOL_WIN:].std())
            if sigma_60 > 1e-9:
                z = abs(last_ret) / sigma_60
                if z >= _FORENSIC_PRICE_SIGMA and abs(last_ret) >= _FORENSIC_MIN_ABS_RETURN:
                    strength = ("strong" if z >= 3.0
                                else "solid" if z >= 2.5 else "weak")
                    rule_hits.append({
                        "rule":      "price_spike",
                        "strength":  strength,
                        "z_score":   round(z, 3),
                        "return":    round(last_ret, 5),
                        "sigma_60d": round(sigma_60, 5),
                    })

        # ── Rule 2: volume multiple ──────────────────────────────────────
        if "Volume" in df.columns:
            vol = df["Volume"].dropna()
            vol = vol[[d for d in vol.index if d <= scan_date]]
            if len(vol) >= _FORENSIC_VOLUME_MEDIAN_WIN + 1:
                today_vol = float(vol.iloc[-1])
                median_30 = float(
                    vol.iloc[-_FORENSIC_VOLUME_MEDIAN_WIN - 1:-1].median()
                )
                if median_30 > 0:
                    mult = today_vol / median_30
                    if mult >= _FORENSIC_VOLUME_MULT:
                        strength = "strong" if mult >= 5.0 else "solid"
                        rule_hits.append({
                            "rule":       "volume_spike",
                            "strength":   strength,
                            "multiple":   round(mult, 3),
                            "today":      int(today_vol),
                            "median_30d": int(median_30),
                        })

        # ── Rule 3: drawdown from 90d peak ───────────────────────────────
        closes_window = closes[[d for d in closes.index if d <= scan_date]]
        if len(closes_window) >= 20:
            peak = float(closes_window.iloc[-_FORENSIC_DRAWDOWN_LOOKBACK:].max())
            cur  = float(closes_window.iloc[-1])
            dd   = (cur / peak) - 1.0 if peak > 0 else 0.0
            if dd <= -0.10:
                strength = "strong" if dd <= -0.20 else "solid"
                rule_hits.append({
                    "rule":     "drawdown",
                    "strength": strength,
                    "dd_pct":   round(dd, 4),
                    "peak_90d": round(peak, 4),
                    "current":  round(cur, 4),
                })

        n_strong = sum(1 for h in rule_hits if h.get("strength") == "strong")
        n_solid  = sum(1 for h in rule_hits if h.get("strength") == "solid")
        last_close = float(closes_window.iloc[-1])
        last_date  = closes_window.index[-1]

        return json.dumps({
            "ticker":      tkr,
            "as_of":       str(scan_date),
            "last_close":  round(last_close, 4),
            "last_bar":    str(last_date),
            "n_rule_hits": len(rule_hits),
            "n_strong":    n_strong,
            "n_solid":     n_solid,
            "rule_hits":   rule_hits,
            "thresholds":  {
                "price_sigma":  _FORENSIC_PRICE_SIGMA,
                "volume_mult":  _FORENSIC_VOLUME_MULT,
                "drawdown":     -0.10,
            },
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"forensic_ticker_check failed: {exc}"})


def read_nav_history(days_back: int = 30) -> str:
    """Read recent rows from PortfolioNavSnapshot (daily NAV + Modified
    Dietz daily return). Use this for performance attribution work.

    Args:
      days_back: look back N calendar days (default 30, max 365).
    """
    try:
        import datetime as _dt
        from engine.db_models import PortfolioNavSnapshot
        from engine.memory import SessionFactory

        n = max(1, min(int(days_back), 365))
        cutoff = _dt.date.today() - _dt.timedelta(days=n)
        with SessionFactory() as s:
            rows = (
                s.query(PortfolioNavSnapshot)
                 .filter(PortfolioNavSnapshot.snapshot_date >= cutoff)
                 .order_by(PortfolioNavSnapshot.snapshot_date.asc())
                 .all()
            )
        if not rows:
            return json.dumps({
                "n_rows":  0,
                "message": f"No PortfolioNavSnapshot rows in last {n} days.",
            })

        nav_first = float(rows[0].nav_close or 0.0)
        nav_last  = float(rows[-1].nav_close or 0.0)
        total_return = (
            nav_last / nav_first - 1.0 if nav_first > 0 else 0.0
        )

        # Compact daily payload — only the essentials.
        days_out = []
        for r in rows[-min(60, len(rows)):]:   # cap response at 60 most recent
            days_out.append({
                "date":          str(r.snapshot_date),
                "nav_close":     float(r.nav_close) if r.nav_close else None,
                "daily_dietz":   float(r.daily_modified_dietz)
                                 if r.daily_modified_dietz is not None else None,
                "external_flow": float(r.external_flow or 0.0),
                # 2026-06-02: ship SPY benchmark close so the NAV chart
                # can render SPY overlay for like-for-like context. The
                # frontend normalizes to start=1 (apples comparison).
                "benchmark_close": float(r.benchmark_close)
                                    if r.benchmark_close is not None else None,
            })

        return json.dumps({
            "n_rows":       len(rows),
            "first_date":   str(rows[0].snapshot_date),
            "last_date":    str(rows[-1].snapshot_date),
            "nav_first":    nav_first,
            "nav_last":     nav_last,
            "total_return": round(total_return, 6),
            "days":         days_out,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"read_nav_history failed: {exc}"})


def recall_past_turns(
    query:    str,
    agent_id: str | None = None,
    top_k:    int = 5,
) -> str:
    """Tier 2.5 cross-session memory retrieval (Phase A.7 Wave 4.3).

    Returns the top-K most semantically similar past chat turns from
    the ChatTurnEmbedding table. Each result has user_text +
    assistant_text + agent_id + created_at + cosine score.

    Args:
      query:    natural-language query (e.g. "what did we say about
                K1 BAB allocation"). Embedded with all-MiniLM-L6-v2.
      agent_id: scope to one persona's history. None = cross-agent
                (intended for the Chief of Staff supervisor).
      top_k:    max results (default 5).

    Doctrine: retrieved turns are HISTORICAL CLAIMS to be re-verified
    via current-state tools (lookup_strategy_status, read_nav_history,
    etc.) before any new decision. The retrieval surface is intended
    for "what did we discuss" follow-ups, NOT for citing past
    statements as authoritative current state. See each persona's
    "memory and evidence boundary" prompt block.
    """
    try:
        from engine.agents.persona.turn_memory import (
            recall_past_turns as _recall,
        )
        hits = _recall(query, agent_id=agent_id, top_k=int(top_k))
        if not hits:
            return json.dumps({
                "n_hits":  0,
                "message": "No relevant past turns found. Re-phrase the "
                           "query or ask the user to provide context — "
                           "the cross-session memory may not have an "
                           "embedded match.",
            })
        return json.dumps({
            "n_hits":  len(hits),
            "top":     hits,
            "warning": (
                "Retrieved turns are HISTORICAL claims, not current "
                "state. Re-verify any factual reference via current-"
                "state tools before acting on it."
            ),
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"recall_past_turns failed: {exc}"})


def run_dq_pre_batch_check(as_of: str | None = None) -> str:
    """Run all four DQ Inspector pre-batch gates live (Modes 1-4).

    Mirrors the cron-time evaluate_pre_batch call. Returns the structured
    Breach list — useful for the DQ Inspector persona to answer "is
    today's data fit to run on" without waiting for the 06:30 SGT cron
    to land in the DataQualityAlert table.

    Args:
      as_of: ISO date "YYYY-MM-DD" (default today)

    Gates run (per spec id=70 §2.1):
      Mode 1 — FRED series staleness (HARD_HALT)
      Mode 2 — yfinance bab_compat cache staleness (HARD_HALT)
      Mode 3 — D-PEAD panel cache staleness (SOFT_WARN)
      Mode 4 — S&P 500 reconstitution feed staleness (SOFT_WARN)
    """
    try:
        import datetime as _dt
        scan_date = (
            _dt.date.fromisoformat(as_of) if as_of else _dt.date.today()
        )
        from engine.agents.dq_inspector.gates import (
            classify_severity, evaluate_pre_batch,
        )
        breaches = evaluate_pre_batch(scan_date)
        severity = classify_severity(breaches)

        if not breaches:
            return json.dumps({
                "as_of":         str(scan_date),
                "n_breaches":    0,
                "severity":      severity,
                "halt_decision": False,
                "message":       "All pre-batch gates clean. Modes 1-4 PASS.",
            })

        out = []
        for b in breaches:
            out.append({
                "mode_id":          b.mode_id,
                "severity":         b.severity,
                "cb_severity":      b.cb_severity,
                "halt_decision":    b.halt_decision,
                "rule_description": b.rule_description,
                "observed":         b.observed_value,
                "threshold":        b.threshold,
                "affected":         list(b.affected),
            })

        return json.dumps({
            "as_of":         str(scan_date),
            "n_breaches":    len(breaches),
            "severity":      severity,
            "halt_decision": any(b.halt_decision for b in breaches),
            "breaches":      out,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"run_dq_pre_batch_check failed: {exc}"})


def query_audit_findings(
    severity_min: str = "LOW",
    days_back:    int = 7,
    status:       str = "all",
) -> str:
    """Query the AuditFinding table (auto_audit contradiction findings).

    Args:
      severity_min: minimum severity "LOW" | "MID" | "HIGH"
      days_back:    look back N days (default 7)
      status:       OPEN | PROPOSED | PROMOTED | RESOLVED | IGNORED | all
    """
    try:
        import datetime as _dt
        from engine.auto_audit_models import AuditFinding
        from engine.memory import SessionFactory

        sev_order = {"LOW": 0, "MID": 1, "HIGH": 2}
        sev_min = sev_order.get(severity_min.upper(), 0)
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=int(days_back))

        with SessionFactory() as s:
            q = s.query(AuditFinding).filter(
                AuditFinding.detected_at >= cutoff
            )
            if status != "all":
                q = q.filter(AuditFinding.status == status)
            q = q.order_by(AuditFinding.detected_at.desc())
            rows = q.limit(50).all()

        # Severity filter after-the-fact (severity is a string column)
        filtered = [
            r for r in rows
            if sev_order.get((r.severity or "").upper(), -1) >= sev_min
        ]
        if not filtered:
            return json.dumps({
                "n_findings": 0,
                "message":    f"No audit findings (last {days_back}d, "
                              f"sev>={severity_min}, status={status}).",
            })

        out = []
        for r in filtered:
            out.append({
                "id":          r.id,
                "run_id":      r.run_id,
                "rule_name":   r.rule_name,
                "severity":    r.severity,
                "detected_at": str(r.detected_at),
                "status":      r.status,
                "snapshot":    (r.snapshot_json or "")[:200],
                "notes":       (r.notes or "")[:200],
            })
        return json.dumps({
            "n_findings": len(out),
            "findings":   out,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"query_audit_findings failed: {exc}"})


def query_audit_runs(scope: str = "all", days_back: int = 30) -> str:
    """Query the AuditRun table (one row per audit orchestrator tick).

    Args:
      scope:     "critical" | "weekly" | "all"
      days_back: look back N days (default 30)
    """
    try:
        import datetime as _dt
        from engine.auto_audit_models import AuditRun
        from engine.memory import SessionFactory

        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=int(days_back))
        with SessionFactory() as s:
            q = s.query(AuditRun).filter(AuditRun.run_at >= cutoff)
            if scope != "all":
                q = q.filter(AuditRun.scope == scope)
            q = q.order_by(AuditRun.run_at.desc())
            rows = q.limit(50).all()

        if not rows:
            return json.dumps({
                "n_runs":  0,
                "message": f"No audit runs (last {days_back}d, scope={scope}).",
            })

        out = []
        for r in rows:
            out.append({
                "id":           r.id,
                "run_at":       str(r.run_at),
                "scope":        r.scope,
                "n_rules_run":  r.n_rules_run,
                "n_findings":   r.n_findings,
                "n_errors":     r.n_errors,
                "n_suppressed": r.n_suppressed,
                "duration_sec": float(r.duration_sec or 0.0),
                "exit_status":  r.exit_status,
            })
        return json.dumps({
            "n_runs": len(out),
            "runs":   out,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"query_audit_runs failed: {exc}"})


def read_decay_sentinel_report(refresh: bool = False) -> str:
    """Read the Decay Sentinel's deterministic book-health report.

    By default reads the most recent persisted artifact
    (data/decay_sentinel/decay_sentinel_<date>.json) — instant, no recompute.
    With refresh=True, recomputes live (build_mechanisms -> sentinel_report; ~10s,
    touches the registry + parquet caches).

    2026-06-14 staleness auto-trigger: when the latest artifact is more
    than 3 days old, recompute LIVE even if refresh=False. The cron path
    keeps the artifact fresh in normal operation; this trigger covers
    the case where the cron drifts dead (which previously surfaced a
    permanent 20d-stale "as of" pill on /dashboard).

    Args:
      refresh: True = always recompute live; False (default) = read
               artifact, falling back to live recompute if missing or
               > 3d old.
    """
    import datetime as _dt
    try:
        if refresh:
            from engine.agents.decay_sentinel.agent import run_daily
            payload = run_daily(save=True)
            payload["_source"] = "live_recompute (caller requested refresh)"
            return json.dumps(payload, ensure_ascii=False, default=str)
        from pathlib import Path
        art_dir = Path("data/decay_sentinel")
        files = sorted(art_dir.glob("decay_sentinel_*.json"), reverse=True) if art_dir.exists() else []
        if not files:
            from engine.agents.decay_sentinel.agent import run_daily
            payload = run_daily(save=True)
            payload["_source"] = "live_recompute (no artifact existed)"
            return json.dumps(payload, ensure_ascii=False, default=str)
        # Staleness auto-trigger: artifact > 3d old → recompute live.
        try:
            stem = files[0].stem.removeprefix("decay_sentinel_")  # YYYY-MM-DD
            age_days = (_dt.date.today() - _dt.date.fromisoformat(stem)).days
        except Exception:
            age_days = 0
        if age_days > 3:
            from engine.agents.decay_sentinel.agent import run_daily
            payload = run_daily(save=True)
            payload["_source"] = f"live_recompute (artifact was {age_days}d stale)"
            return json.dumps(payload, ensure_ascii=False, default=str)
        with open(files[0], "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_source"] = f"artifact {files[0].name}"
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"read_decay_sentinel_report failed: {exc}"})


def lookup_strategy_status(strategy_name: str) -> str:
    """Look up a single strategy's signal output for today.

    Args:
      strategy_name: one of K1_BAB / D_PEAD / PATH_N / CTA_PQTIX / AC_TLT_GLD
    """
    try:
        from engine.strategies import get_registry
        reg = get_registry()
        try:
            strat = reg.get(strategy_name)
        except KeyError:
            return json.dumps({
                "error":     f"unknown strategy {strategy_name!r}",
                "available": sorted(s.NAME for s in reg),
            })

        today = datetime.date.today()
        signal = strat.generate_signal(today)

        # Top 5 holdings by absolute weight
        top_holdings: list[dict] = []
        try:
            weights = signal.weights.dropna()
            weights = weights[weights.abs() > 1e-9]
            for ticker, w in weights.abs().nlargest(5).items():
                top_holdings.append({
                    "ticker": str(ticker),
                    "weight": float(weights[ticker]),
                })
        except Exception:
            pass

        return json.dumps({
            "strategy":     signal.strategy_name,
            "sleeve":       signal.sleeve_id,
            "intra_w":      signal.intra_sleeve_weight,
            "n_positions":  signal.n_positions,
            "status":       signal.status,
            "notes":        signal.notes,
            "as_of":        today.isoformat(),
            "top_holdings": top_holdings,
            "spec_id":      strat.META.spec_id,
            "spec_hash":    strat.META.spec_hash_short,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": f"lookup_strategy_status failed: {exc}"})


# Position-directive kinds → an EXECUTABLE operator-overlay proposal (vs a record-only
# advisory). Requires ticker + suggested_weight (the target overlay weight; 0 = exit).
_OVERLAY_KINDS = frozenset({
    "overlay", "position", "trim", "add", "reduce", "increase", "exit", "buy", "sell", "short",
})


def propose_action(
    kind:             str,
    detail:           str,
    ticker:           str | None = None,
    suggested_weight: float | None = None,
    rationale:        str = "",
) -> str:
    """File a proposal into the PendingApproval inbox for HUMAN approval. PROPOSAL ONLY —
    this NEVER executes here; the LLM emits an intent, the human approves, deterministic code runs.

    Two flavours, chosen by `kind`:
    • POSITION directive (kind in trim/add/reduce/exit/position/… WITH ticker + suggested_weight)
      → an EXECUTABLE 'overlay' proposal. On the human's Approve, the deterministic overlay
      executor (engine.overlay_executor, RM-cap validated) sets that position in the discretionary
      OVERLAY sleeve — SEPARATE from the systematic book, which is never hand-edited. Validated at
      propose time too, so an over-budget intent is refused before it reaches the inbox.
    • Anything else → a record-only 'advisory' proposal (the resolver records the decision, no trade).

    The systematic 5-strategy book is NOT touched by either path (0-LLM-in-DECISION; the overlay is
    a human-originated discretionary sleeve, measured separately)."""
    try:
        import datetime as _dt
        from engine.memory import PendingApproval, SessionFactory
        if not (detail or "").strip():
            return json.dumps({"error": "detail is required"})

        kind_l = (kind or "").strip().lower()
        is_overlay = kind_l in _OVERLAY_KINDS and bool(ticker) and (suggested_weight is not None)
        if is_overlay:
            # Defense-in-depth: validate against the overlay risk budget now; the executor
            # re-validates authoritatively on approve.
            from engine.overlay_executor import validate_overlay_intent
            ok, reason = validate_overlay_intent(ticker, suggested_weight)
            if not ok:
                return json.dumps(
                    {"error": f"overlay intent refused by sleeve risk budget: {reason}. "
                              f"Re-propose within the overlay caps, or file as advisory."},
                    ensure_ascii=False)
            approval_type, approval_class = "overlay", "operator_overlay"
        else:
            approval_type, approval_class = "advisory", "agent_proposal"

        with SessionFactory() as s:
            pa = PendingApproval(
                approval_type       = approval_type,
                approval_class      = approval_class,
                priority            = "normal",
                status              = "pending",
                sector              = (kind or approval_type)[:50],
                ticker              = ((ticker or "BOOK").upper())[:20],
                triggered_date      = _dt.date.today(),
                triggered_condition = (f"{kind}: {detail}" if kind else detail)[:1000],
                suggested_weight    = (float(suggested_weight) if suggested_weight is not None else None),
                review_rationale    = (rationale or "")[:1000],
                created_at          = _dt.datetime.utcnow(),
            )
            s.add(pa)
            s.commit()
            pid = pa.id

        if is_overlay:
            msg = (f"Proposal #{pid} filed (overlay, status=pending). It has NOT executed. On the "
                   f"user's Approve, the engine sets {ticker.upper()} to {float(suggested_weight):+.1%} "
                   f"in the discretionary OVERLAY sleeve (separate from the systematic book), after "
                   f"RM-cap validation. Reject → nothing changes.")
        else:
            msg = (f"Proposal #{pid} filed (advisory, status=pending). It will NOT execute — the user "
                   f"reviews it in the Approvals inbox and Approves/Rejects. Advisory proposals are "
                   f"recorded on approval, never auto-traded.")
        return json.dumps({"ok": True, "approval_id": pid, "approval_type": approval_type,
                           "message": msg}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("propose_action failed")
        return json.dumps({"error": f"propose_action failed: {exc}"})


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic tool_use JSON schemas
# ──────────────────────────────────────────────────────────────────────────────
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "query_recent_alerts",
        "description": (
            "Query the Risk Manager and DQ Inspector alert tables. "
            "Use this when the user asks about recent alerts, halts, "
            "warnings, or what risks fired in the last N days."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type":        "integer",
                    "description": "Look back N days from today (default 7).",
                    "default":     7,
                },
                "severity_min": {
                    "type":        "string",
                    "description": "Minimum severity to include.",
                    "enum":        ["LIGHT", "MEDIUM", "SEVERE"],
                    "default":     "LIGHT",
                },
                "source": {
                    "type":        "string",
                    "description": "Which alert source to query.",
                    "enum":        ["rm", "dq", "all"],
                    "default":     "all",
                },
            },
        },
    },
    {
        "name": "read_today_book_state",
        "description": (
            "Read the most recent paper-trade book state: per-strategy "
            "status, position counts, sleeve attribution, combined "
            "gross/net. Use this when the user asks about today's "
            "book, current positions, or what the portfolio looks like now."
        ),
        "input_schema": {
            "type":       "object",
            "properties": {},
        },
    },
    {
        "name": "lookup_spec",
        "description": (
            "Look up a registered spec by spec_id from the SpecRegistry "
            "table (Tier 3a — structured ground-truth fact). Returns spec "
            "metadata + full amendment log. Use this when the user asks "
            "about a specific spec's hash, status, or amendment history "
            "(e.g. 'why was spec 69 re-hashed', 'what amendments touched K1 BAB')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "spec_id": {
                    "type":        "integer",
                    "description": "SpecRegistry.id integer (e.g. 69 RM, 70 DQ).",
                },
            },
            "required": ["spec_id"],
        },
    },
    {
        "name": "read_project_memory",
        "description": (
            "Search project memory files (Tier 3b — human-curated knowledge "
            "in ~/.claude/projects/<slug>/memory/*.md). Use this when the "
            "user asks about historical decisions, feedback rules, or "
            "architectural choices (e.g. 'what did we decide about emojis', "
            "'which LLM provider for narrator', 'when was Research Co-Pilot "
            "superseded'). Three search modes: 'exact' (filename stem), "
            "'keyword' (substring), 'semantic' (sentence-transformers "
            "embedding for question-style queries). Default 'auto' picks "
            "the right mode based on the query shape."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "Filename stem (e.g. 'feedback_no_emojis_2026-05-19'), "
                                   "keyword ('emoji'), or natural-language "
                                   "question ('what was the rule about emojis?').",
                },
                "max_results": {
                    "type":        "integer",
                    "description": "Max files to return (default 5).",
                    "default":     5,
                },
                "mode": {
                    "type":        "string",
                    "description": "Retrieval mode (default 'auto').",
                    "enum":        ["auto", "exact", "keyword", "semantic"],
                    "default":     "auto",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_recent_anomalies",
        "description": (
            "Query the AnomalyFlag table (forensic per-ticker anomaly "
            "history). Returns past detector hits with confidence, "
            "event_class, and evidence summary. Use this when the user "
            "asks 'has X flagged before', 'what anomalies fired this week', "
            "or 'show me LLM-detected events for AAPL'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type":        "string",
                    "description": "Filter to one ticker (omit / null for cross-ticker view).",
                },
                "days_back": {
                    "type":        "integer",
                    "description": "Look back N calendar days (default 14).",
                    "default":     14,
                },
                "min_confidence": {
                    "type":        "integer",
                    "description": "Min confidence_likert 1-5 (default 2 = solid).",
                    "default":     2,
                },
                "detector": {
                    "type":        "string",
                    "description": "Which detector source to include.",
                    "enum":        ["rule_baseline_a", "rule_baseline_b",
                                    "llm", "all"],
                    "default":     "all",
                },
            },
        },
    },
    {
        "name": "forensic_ticker_check",
        "description": (
            "Run live forensic rule detectors against one ticker (price "
            "z-score / volume multiple / drawdown). Computes current state "
            "using the same engine.anomaly_screener helpers as the daily "
            "cron. Use this when the user asks 'is GLD anomalous today', "
            "'what's TLT's z-score right now', or 'forensic check on PQTIX'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type":        "string",
                    "description": "Ticker symbol (case-insensitive).",
                },
                "as_of": {
                    "type":        "string",
                    "description": "ISO date 'YYYY-MM-DD' (omit / null for today).",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "read_nav_history",
        "description": (
            "Read recent PortfolioNavSnapshot rows (daily NAV + Modified-"
            "Dietz daily return). Use this for attribution work: P&L over "
            "the last N days, daily-return path, external flow effects. "
            "Returns up to the most recent 60 daily snapshots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type":        "integer",
                    "description": "Look back N calendar days (default 30, max 365).",
                    "default":     30,
                },
            },
        },
    },
    {
        "name": "recall_past_turns",
        "description": (
            "Tier 2.5 cross-session memory: retrieve top-K past chat "
            "turns most semantically similar to ``query`` via sentence-"
            "transformers embedding. Use this when the user references "
            "an earlier conversation ('what did we say about X', 'remind "
            "me about Y'). agent_id scopes to one persona's history; "
            "omit to search across all agents (Chief of Staff use). "
            "DOCTRINE: retrieved turns are HISTORICAL claims, not "
            "current state — re-verify via lookup_spec / current-state "
            "tools before any new decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "Natural-language query.",
                },
                "agent_id": {
                    "type":        "string",
                    "description": "Scope to one persona's history. Omit for cross-agent (CoS only).",
                },
                "top_k": {
                    "type":        "integer",
                    "description": "Max results (default 5).",
                    "default":     5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_dq_pre_batch_check",
        "description": (
            "Run all four DQ Inspector pre-batch gates live (Modes 1-4: "
            "FRED staleness, yfinance bab_compat cache, D-PEAD panel "
            "cache, S&P 500 feed). Returns breaches + overall severity. "
            "Use this when the user asks 'is today's data fresh', 'did "
            "FRED update', 'is bab cache stale' — gets the LIVE state "
            "instead of waiting for the cron to land in DataQualityAlert."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "as_of": {
                    "type":        "string",
                    "description": "ISO date 'YYYY-MM-DD' (default today).",
                },
            },
        },
    },
    {
        "name": "query_audit_findings",
        "description": (
            "Query the AuditFinding table (auto_audit contradiction "
            "findings from the daily / weekly orchestrator). Filter by "
            "severity / days / status. Use this when the user asks "
            "'what audit findings are open', 'show me high-severity "
            "findings this week', or 'what got promoted to PendingApproval'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "severity_min": {
                    "type":        "string",
                    "description": "Minimum severity to include.",
                    "enum":        ["LOW", "MID", "HIGH"],
                    "default":     "LOW",
                },
                "days_back": {
                    "type":        "integer",
                    "description": "Look back N days (default 7).",
                    "default":     7,
                },
                "status": {
                    "type":        "string",
                    "description": "Lifecycle status filter.",
                    "enum":        ["OPEN", "PROPOSED", "PROMOTED",
                                    "RESOLVED", "IGNORED", "all"],
                    "default":     "all",
                },
            },
        },
    },
    {
        "name": "query_audit_runs",
        "description": (
            "Query the AuditRun table (one row per auto_audit "
            "orchestrator tick). Use this when the user asks 'when did "
            "the audit last run', 'how many findings per day this week', "
            "or 'show me weekly audit summary'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scope": {
                    "type":        "string",
                    "description": "Audit scope.",
                    "enum":        ["critical", "weekly", "all"],
                    "default":     "all",
                },
                "days_back": {
                    "type":        "integer",
                    "description": "Look back N days (default 30).",
                    "default":     30,
                },
            },
        },
    },
    {
        "name": "delegate_to_specialist",
        "description": (
            "Route one question to one specialist persona in an isolated "
            "sub-context. Use this when the question requires a "
            "specialist's tool palette (book state, alerts, audit "
            "findings, z-scores, NAV history). Returns the specialist's "
            "synthesized answer + cost. Sequential only — at most 3 "
            "delegations per user turn. Available specialists: "
            "risk_manager / dq_inspector / anomaly_sentinel / "
            "attribution_analyst / audit_recorder / devils_advocate / "
            "decay_sentinel. "
            "Call list_personas() if you need to refresh on each "
            "specialist's scope."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type":        "string",
                    "description": "One of the six specialist agent_ids.",
                    "enum":        ["risk_manager", "dq_inspector",
                                    "anomaly_sentinel", "attribution_analyst",
                                    "audit_recorder", "devils_advocate",
                                    "decay_sentinel"],
                },
                "query": {
                    "type":        "string",
                    "description": "The question rephrased for the specialist's scope (terse, factual).",
                },
                "max_iterations": {
                    "type":        "integer",
                    "description": "Hard cap on specialist tool round-trips (default 4).",
                    "default":     4,
                },
            },
            "required": ["agent_id", "query"],
        },
    },
    {
        "name": "list_personas",
        "description": (
            "Return the six specialist agent_ids + one-line scope summary. "
            "Use this at decision time to confirm which specialist owns "
            "which question. Cheap — does not invoke any LLM."
        ),
        "input_schema": {
            "type":       "object",
            "properties": {},
        },
    },
    {
        "name": "read_decay_sentinel_report",
        "description": (
            "Read the Decay Sentinel's deterministic book-health report: per-mechanism "
            "rolling Sharpe / decay-ratio + role (alpha/insurance/trend/regime_premium), "
            "crisis-payoff for hedges, signal-IC for alpha/carry, structural-decay flags, "
            "all pairwise downside/stress correlations, the OVERALL verdict "
            "(HEALTHY/WATCH/ACTION) and the recommended re-allocation. Use this when the "
            "user asks 'is D_PEAD/carry decaying', 'is the book still diversified', "
            "'what's the recommended allocation', 'is any mechanism dying'. Default reads "
            "the latest daily artifact; set refresh=True to recompute live (~10s)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "refresh": {
                    "type":        "boolean",
                    "description": "True = recompute live; False (default) = read latest artifact.",
                    "default":     False,
                },
            },
        },
    },
    {
        "name": "lookup_strategy_status",
        "description": (
            "Look up a single strategy's signal output for today: "
            "status (OK/NO_SIGNAL/ERROR), position count, top holdings, "
            "and notes. Use this when the user asks about a specific "
            "strategy (e.g. 'why is K1 BAB not trading')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_name": {
                    "type":        "string",
                    "description": "Strategy NAME (case-sensitive).",
                    "enum":        ["K1_BAB", "D_PEAD", "PATH_N",
                                    "CTA_PQTIX", "AC_TLT_GLD"],
                },
            },
            "required": ["strategy_name"],
        },
    },
    {
        "name": "propose_action",
        "description": (
            "File an ADVISORY proposal into the human Approvals inbox. This is the ONLY way a "
            "chat command becomes an actionable item, and it is PROPOSAL-ONLY: it writes one "
            "pending row a human must Approve/Reject — it NEVER executes a trade, mutates the "
            "book, or amends a spec. Use this when the user issues a directive ('propose cutting "
            "GLD to 3%', 'flag K1 BAB for review', 'recommend trimming the carry sleeve') and you "
            "want to turn it into a tracked, auditable proposal rather than just talking about it. "
            "Always ground the proposal in current state first (delegate / read the relevant tool), "
            "then file it with a concrete rationale. After filing, tell the user it is pending in "
            "the Approvals inbox for their decision — do NOT claim anything was executed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type":        "string",
                    "description": "Short proposal category, e.g. 'rebalance' / 'risk_review' / "
                                   "'trim' / 'add' / 'investigate'. Free text, kept for the audit row.",
                },
                "detail": {
                    "type":        "string",
                    "description": "One-line description of what is being proposed (required).",
                },
                "ticker": {
                    "type":        "string",
                    "description": "Ticker / sleeve the proposal concerns (omit / null for a "
                                   "book-level proposal; defaults to BOOK).",
                },
                "suggested_weight": {
                    "type":        "number",
                    "description": "Optional target weight (decimal, e.g. 0.03 for 3%) if the "
                                   "proposal names one.",
                },
                "rationale": {
                    "type":        "string",
                    "description": "Why this proposal — cite the current-state evidence you "
                                   "gathered this turn. Becomes the audit rationale.",
                },
            },
            "required": ["kind", "detail"],
        },
    },
]


_REFUSAL_TOKENS = (
    "refused",
    "refused —",
    "cannot ",
    "out of my scope",
    "out-of-scope",
    "i cannot",
    "no factor-regression tool",
    "no tool",
    "insufficient evidence",
)


def _looks_like_refusal(text: str) -> bool:
    """Heuristic: does the specialist's answer start with / contain a
    standard refusal phrase from the locked persona prompts?

    Used by CoS to distinguish "specialist gave you a real answer" from
    "specialist routed back to user / refused". CoS can then either
    re-route or surface the refusal verbatim. Derived signal — does not
    trust the LLM to self-label.
    """
    if not text:
        return False
    head = text.strip()[:200].lower()
    return any(tok in head for tok in _REFUSAL_TOKENS)


def _confidence_signal(
    stop_reason:    str,
    n_iterations:   int,
    max_iterations: int,
    tools_called:   list[str],
    answer:         str,
) -> str:
    """Derive a high / med / low confidence label from the AgentTurnResult.

    NOT model-generated — purely from observable execution metadata. This
    is the seam that lets CoS make programmatic decisions ("RM confidence
    low, also ask DQ Inspector for cross-check") without trusting any LLM
    self-assessment.

    Rules (deliberate, simple, auditable):
      high:  end_turn   AND  n_iterations >= 2  AND  tools_called non-empty
             (specialist actually consulted state + terminated cleanly)
      low:   stop_reason == "max_tokens"  OR  hit the iteration cap
             (truncated mid-thought OR ran out of room)
      med:   everything else (clean termination but didn't consult tools,
             or made one tool call only)
    """
    if not answer or not answer.strip():
        return "low"
    if stop_reason == "max_tokens":
        return "low"
    if max_iterations is not None and n_iterations >= int(max_iterations):
        return "low"
    if stop_reason == "end_turn" and n_iterations >= 2 and tools_called:
        return "high"
    return "med"


def delegate_to_specialist(
    agent_id:       str,
    query:          str,
    max_iterations: int = 4,
) -> str:
    """Route one isolated question to one specialist persona.

    This is the seam of the Supervisor pattern (spec_chief_of_staff_agent_v1
    id=74 §3.3). Each call runs the named specialist's chat_turn in an
    EMPTY-history sub-context — no cross-agent contamination, no Pattern
    5 cascade. Returns a compact JSON the calling CoS can quote verbatim.

    Args:
      agent_id:       one of 'risk_manager' / 'dq_inspector' /
                      'anomaly_sentinel' / 'attribution_analyst' /
                      'audit_recorder' / 'devils_advocate'.
      query:          the question to ask that specialist — phrased as
                      if a colleague were asking, not the user's exact
                      words (CoS may rephrase for the specialist's scope).
      max_iterations: per-spec §3.3 hard cap (default 4 to keep the
                      whole CoS turn bounded; CoS's own max is 6).

    Returns: JSON string with keys
      from_agent, answer, n_iterations, total_cost_usd, total_latency_ms.
      Or {"error": "..."} on failure.
    """
    _SPECIALIST_PERSONAS = {
        # Lazy-imported so adding a new persona file doesn't require
        # touching this dispatch in two places.
        "risk_manager":        ("engine.agents.persona", "RISK_MANAGER"),
        "dq_inspector":        ("engine.agents.persona", "DQ_INSPECTOR"),
        "anomaly_sentinel":    ("engine.agents.persona", "ANOMALY_SENTINEL"),
        "attribution_analyst": ("engine.agents.persona", "ATTRIBUTION_ANALYST"),
        "audit_recorder":      ("engine.agents.persona", "AUDIT_RECORDER"),
        "devils_advocate":     ("engine.agents.persona", "DEVILS_ADVOCATE"),
        "decay_sentinel":      ("engine.agents.persona", "DECAY_SENTINEL"),
    }
    try:
        if agent_id not in _SPECIALIST_PERSONAS:
            return json.dumps({
                "error":     f"unknown specialist {agent_id!r}",
                "available": sorted(_SPECIALIST_PERSONAS),
            })
        import dataclasses
        import importlib
        mod_path, attr = _SPECIALIST_PERSONAS[agent_id]
        persona = getattr(importlib.import_module(mod_path), attr)

        # Honor the caller-supplied iteration cap (default 4). The
        # persona's own default (typically 6) is the ceiling for direct
        # chat; under CoS delegation we cap tighter so the whole user
        # turn stays bounded. dataclasses.replace creates a new frozen
        # instance — doesn't mutate the original singleton.
        capped_iter = min(int(max_iterations), persona.max_iterations)
        bounded_persona = dataclasses.replace(
            persona, max_iterations=capped_iter,
        )

        from engine.agents.persona.base import chat_turn
        result = chat_turn(
            persona        = bounded_persona,
            user_message   = query,
            history        = [],                        # isolation contract
            max_tokens     = bounded_persona.default_max_tokens,
            effort         = bounded_persona.default_effort,
        )
        # Honor the caller-supplied max_iterations cap. The persona's
        # configured max_iterations is the ceiling; we surface this in
        # the structured payload so CoS can detect a timed-out call.
        tools_called = [tc.get("name") for tc in result.tool_calls_log]
        confidence_signal = _confidence_signal(
            stop_reason   = result.stop_reason,
            n_iterations  = result.n_iterations,
            max_iterations= max_iterations,
            tools_called  = tools_called,
            answer        = result.final_text,
        )
        # The specialist's internal tool-call log stays in the specialist's
        # own audit trail — CoS only sees the synthesized final_text + meta
        # + structured signal (Phase A.7 Wave 3.1 enrichment, 2026-05-19).
        return json.dumps({
            "from_agent":         agent_id,
            "answer":             result.final_text,
            "n_iterations":       result.n_iterations,
            "total_cost_usd":     round(result.total_cost_usd, 6),
            "total_latency_ms":   int(result.total_latency_ms),
            "stop_reason":        result.stop_reason,
            # Structured signal — derived, NOT model-generated. Safe to
            # parse programmatically without trusting the LLM's output.
            "tools_called":       tools_called,
            "confidence_signal":  confidence_signal,
            "answer_is_refusal":  _looks_like_refusal(result.final_text),
            "answer_is_empty":    not result.final_text.strip(),
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.exception("delegate_to_specialist(%s) failed", agent_id)
        return json.dumps({
            "error":      f"specialist {agent_id} failed: {exc}",
            "from_agent": agent_id,
        })


def list_personas() -> str:
    """Return the six specialist agent_ids + a one-line scope summary.

    CoS calls this at decision time to confirm which specialist owns
    which question. Static dispatch — does not actually invoke the
    persona modules, so cheap (single-digit ms).
    """
    return json.dumps({
        "specialists": [
            {"agent_id": "risk_manager",
             "scope":    "book-level risk gates: VaR / HHI / leverage / "
                         "sleeve drift / HARD HALT verdicts"},
            {"agent_id": "dq_inspector",
             "scope":    "data layer: FRED / yfinance / cache freshness / "
                         "NaN burst / universe coverage / row-count regression"},
            {"agent_id": "anomaly_sentinel",
             "scope":    "per-ticker forensic: live z-score, volume "
                         "multiple, drawdown, AnomalyFlag history"},
            {"agent_id": "attribution_analyst",
             "scope":    "P&L decomposition: NAV path, sleeve attribution, "
                         "per-strategy intra_w. NO factor regression tool."},
            {"agent_id": "audit_recorder",
             "scope":    "governance trail: AuditFinding / AuditRun / "
                         "SpecRegistry amendment_log. Reports state, "
                         "does NOT rule on state."},
            {"agent_id": "devils_advocate",
             "scope":    "counterfactual / p-hacking critique. "
                         "Evidence-only — refuses to fabricate citations."},
            {"agent_id": "decay_sentinel",
             "scope":    "mechanism/strategy decay + book diversification "
                         "integrity + disciplined re-allocation (rolling "
                         "Sharpe/signal-IC, role-aware, downside/stress corr). "
                         "Reads the deterministic book-health report; math "
                         "decides, it explains."},
        ],
        "delegation_rule": (
            "Per spec_chief_of_staff_agent_v1 §3.2: route to the "
            "specialist matching the question's primary keyword set. "
            "Up to 3 delegations per user turn, sequential. If no "
            "specialist clearly matches, ask the user to clarify."
        ),
    }, ensure_ascii=False)


# Dispatch table — name → callable. Defined AFTER all tool function
# definitions so forward references resolve at import time.
_TOOL_IMPLS = {
    "query_recent_alerts":     query_recent_alerts,
    "read_today_book_state":   read_today_book_state,
    "lookup_strategy_status":  lookup_strategy_status,
    "lookup_spec":             lookup_spec,
    "read_project_memory":     read_project_memory,
    "query_recent_anomalies":  query_recent_anomalies,
    "forensic_ticker_check":   forensic_ticker_check,
    "read_decay_sentinel_report": read_decay_sentinel_report,
    "read_nav_history":        read_nav_history,
    "run_dq_pre_batch_check":  run_dq_pre_batch_check,
    "query_audit_findings":    query_audit_findings,
    "query_audit_runs":        query_audit_runs,
    "recall_past_turns":       recall_past_turns,
    "delegate_to_specialist":  delegate_to_specialist,
    "list_personas":           list_personas,
    "propose_action":          propose_action,
}


def select_tools(names: list[str]) -> list[dict]:
    """Return the TOOL_SCHEMAS entries whose name matches one in ``names``,
    preserving the requested order.

    Each persona declares which subset of the shared registry it exposes
    so adding a new tool here does not silently widen another agent's
    capability set. Unknown names raise ValueError — typo-fails fast.
    """
    by_name = {t["name"]: t for t in TOOL_SCHEMAS}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(
            f"select_tools: unknown tool name(s) {missing}; available: "
            f"{sorted(by_name)}"
        )
    return [by_name[n] for n in names]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Dispatch a tool call to its implementation.

    Returns:
      (content_str, is_error) tuple.
        content_str: JSON-encoded result (data or error description)
        is_error:    True if execution failed (caller should set
                     is_error=True on the Anthropic tool_result block
                     so the model knows to recover rather than treat
                     the error string as legitimate data)

    Never raises — errors are caught and returned with is_error=True.
    """
    fn = _TOOL_IMPLS.get(name)
    if fn is None:
        return (
            json.dumps({
                "error":     f"unknown tool {name!r}",
                "available": sorted(_TOOL_IMPLS),
            }),
            True,
        )
    try:
        result = fn(**(tool_input or {}))
        # Tool-output injection guard (governance, blueprint spec id=78 §6): detect/cap/
        # (enforce) wrap untrusted tool text that tries to override the model. Default
        # 'warn' = detect+log, output unchanged. Wrapped so it can never break a tool call.
        try:
            from engine.agents.governance.tool_output_guard import guard_tool_output
            result = guard_tool_output(name, result, scope="execute_tool").output
        except Exception:
            pass
        # Tool function returned JSON; check if it surfaced an internal error
        try:
            parsed = json.loads(result)
            is_err = isinstance(parsed, dict) and "error" in parsed
        except Exception:
            is_err = False
        return (result, is_err)
    except TypeError as exc:
        # Bad arguments — model can self-correct
        return (
            json.dumps({"error": f"bad arguments to {name}: {exc}"}),
            True,
        )
    except Exception as exc:
        logger.exception("execute_tool(%s) failed", name)
        return (
            json.dumps({"error": f"{name} raised: {exc}"}),
            True,
        )
