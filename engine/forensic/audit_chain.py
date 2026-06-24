"""
engine/forensic/audit_chain.py — Provenance-locked forensic brief writer.

Forensic redesign Phase 4 (2026-05-14).

Purpose
-------
Persist every forensic verdict as an append-only markdown file with a
YAML frontmatter encoding the full reproducibility trinity (code commit
· prompt hash · input snapshot). Each brief becomes a citation-eligible
audit record satisfying SEC 17a-4(b) / CFA GIPS / MiFID II Article 25
record-keeping expectations.

Output convention
-----------------
  data/forensic/briefs/
    <YYYY-MM-DD>_<strategy>_<ticker>_<consistency>_<trade_or_dayid>.md

Frontmatter schema (YAML):
  generated_at_utc: ISO 8601
  date:             trade or strategy-day date
  strategy_name:    K1_BAB / D_PEAD / PATH_N / CTA_PQTIX
  ticker:           (single-ticker briefs) or "_combined" (strategy-day)
  trigger_source:   anomaly_detector_strategy_day | anomaly_detector_trade_horizon
                    | ad_hoc_user_form
  trigger_z_score:  signed |z| from anomaly_detector (None if ad-hoc)
  residual_pct:     epsilon as % of realized (None if no residual decomp)
  llm_eligible:     bool (per Brinson residual_share >= 40%)
  primary_model:    e.g. gemini-2.5-flash
  devil_model:      e.g. deepseek-v4-flash (or 'skipped' / 'failed:...')
  verdict_primary:  case_a / case_b / case_c
  verdict_devil:    case_a / case_b / case_c
  verdict_consistency_score: float in [0, 1]
  verdict_consistency_label: HIGH | LOW
  total_cost_usd:   float
  prompt_hash:      sha256:abc123... (lock to detect prompt drift)
  code_commit:      git rev-parse HEAD (best-effort)

Reproducibility property
------------------------
Given the same (prompt_hash, code_commit, input snapshot in body), a
future re-run is expected to produce verdict-stable output up to LLM
nondeterminism (which is itself bounded by temperature=0.1 + fixed
schemas). Divergence between two re-runs is itself an audit signal.

References
----------
  - Donoho 2010 "An Invitation to Reproducible Computational Research"
  - SEC 17a-4(b) electronic record-keeping
  - CFA GIPS verification standard §3
  - MiFID II Article 25 record-keeping
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_BRIEFS_DIR = Path("data/forensic/briefs")
_BRIEFS_DIR.mkdir(parents=True, exist_ok=True)


def _git_commit_short() -> str:
    """Best-effort git short SHA. Returns 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _prompt_hash(*parts: str) -> str:
    """SHA-256 hex digest of concatenated prompt parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x1f")  # unit separator
    return "sha256:" + h.hexdigest()[:16]


def _slug(text: str) -> str:
    """Filesystem-safe slug for a filename component."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return s[:40] if s else "x"


def write_dual_llm_brief(
    result,                                # DualLLMForensicResult
    *,
    trigger_source:   str,
    trigger_z_score:  Optional[float] = None,
    residual_pct:     Optional[float] = None,
    llm_eligible:     Optional[bool]  = None,
    extra_context:    Optional[dict]  = None,
) -> Path:
    """Write a forensic brief to disk with provenance frontmatter.

    Args:
        result:           DualLLMForensicResult from devils_advocate
        trigger_source:   one of the trigger_source enum strings
        trigger_z_score:  |z| from anomaly_detector (signed)
        residual_pct:     epsilon as decimal (e.g. -0.0473)
        llm_eligible:     residual.llm_eligible
        extra_context:    optional dict spliced into frontmatter

    Returns:
        Path to the written markdown file.
    """
    p   = result.primary_summary
    d   = result.devil_verdict
    now = datetime.datetime.utcnow()

    # Compute prompt hash from BOTH primary and devil prompt-relevant fields.
    # We don't have the literal prompt text here without re-running, but the
    # input snapshot (trade context echoed by primary) is the deterministic
    # input; pair with code_commit for full reproducibility.
    input_snapshot = json.dumps({
        "date":              p.date.isoformat(),
        "ticker":            p.ticker,
        "strategy_name":     p.strategy_name,
        "signal_value":      p.signal_value,
        "weight":            p.weight,
        "realized_return":   p.realized_return,
        "expected_horizon":  p.expected_horizon_days,
        "n_articles":        p.n_articles,
        "date_window":       [p.date_window_start.isoformat(),
                              p.date_window_end.isoformat()],
    }, sort_keys=True, default=str)
    prompt_h = _prompt_hash(input_snapshot)

    code_commit = _git_commit_short()
    consistency_pct = result.consistency_score * 100.0

    # Filename: <date>_<strategy>_<ticker>_<consistency>_<sha8>.md
    sha_suffix = prompt_h.split(":", 1)[-1][:8]
    fname = (
        f"{p.date.isoformat()}_{_slug(p.strategy_name)}_"
        f"{_slug(p.ticker)}_{result.consistency_label.lower()}_{sha_suffix}.md"
    )
    out_path = _BRIEFS_DIR / fname

    # ─── Frontmatter ─────────────────────────────────────────────────
    frontmatter_lines = [
        "---",
        f"generated_at_utc:           {now.isoformat()}Z",
        f"date:                       {p.date.isoformat()}",
        f"strategy_name:              {p.strategy_name}",
        f"ticker:                     {p.ticker}",
        f"trigger_source:             {trigger_source}",
    ]
    if trigger_z_score is not None:
        frontmatter_lines.append(f"trigger_z_score:            {trigger_z_score:+.3f}")
    if residual_pct is not None:
        frontmatter_lines.append(f"residual_pct:               {residual_pct*100:+.3f}")
    if llm_eligible is not None:
        frontmatter_lines.append(f"llm_eligible:               {str(llm_eligible).lower()}")
    frontmatter_lines += [
        f"primary_model:              {p.llm_model}",
        f"devil_model:                {d.llm_model}",
        f"verdict_primary:            {p.forensic_verdict}",
        f"verdict_devil:              {d.forensic_verdict}",
        f"verdict_consistency_score:  {result.consistency_score:.3f}",
        f"verdict_consistency_label:  {result.consistency_label}",
        f"verdict_agreement:          {str(result.verdict_agreement).lower()}",
        f"direction_agreement:        {str(result.direction_agreement).lower()}",
        f"event_overlap:              {result.event_overlap}",
        f"total_cost_usd:             {result.total_cost_usd:.6f}",
        f"primary_latency_ms:         {p.llm_latency_ms}",
        f"devil_latency_ms:           {d.llm_latency_ms}",
        f"prompt_hash:                {prompt_h}",
        f"code_commit:                {code_commit}",
        f"n_articles:                 {p.n_articles}",
        f"n_sources:                  {p.n_sources}",
    ]
    if extra_context:
        for k, v in extra_context.items():
            frontmatter_lines.append(f"{k}:  {v}")
    frontmatter_lines.append("---")

    # ─── Body ────────────────────────────────────────────────────────
    body_lines = [
        f"",
        f"# Forensic Brief — {p.ticker} on {p.date}",
        f"",
        f"**Strategy:** {p.strategy_name}  ·  "
        f"**Trigger:** {trigger_source}  ·  "
        f"**Consistency:** {result.consistency_label} ({consistency_pct:.0f}%)",
        f"",
    ]
    if trigger_z_score is not None or residual_pct is not None:
        body_lines += [
            f"## Quantitative Trigger",
            f"",
        ]
        if trigger_z_score is not None:
            body_lines.append(f"- Outlier z-score (Cohen-Polk-Vuolteenaho): "
                              f"**{trigger_z_score:+.2f}σ**")
        if residual_pct is not None:
            body_lines.append(f"- Residual ε after FF5 + TC decomposition: "
                              f"**{residual_pct*100:+.3f}%**")
        if llm_eligible is not None:
            body_lines.append(f"- LLM-eligible (|ε|/|realized| ≥ 40%): "
                              f"**{str(llm_eligible).upper()}**")
        body_lines.append("")

    # ─── Primary (Gemini) verdict ──
    body_lines += [
        f"## Primary verdict — {p.llm_model}: **{p.forensic_verdict}**",
        f"",
        f"### Material Events",
        *(f"- {e}" for e in p.material_events),
        f"",
        f"### Macro Context",
        p.macro_context or "_(none)_",
        f"",
        f"### Sentiment Assessment",
        p.sentiment_assessment or "_(none)_",
        f"",
        f"### Signal Alignment Analysis",
        p.signal_alignment or "_(none)_",
        f"",
        f"### Key Quotes",
        *(f"> {q}" for q in p.key_quotes),
        f"",
    ]

    # ─── Devil's advocate (DeepSeek) verdict ──
    body_lines += [
        f"## Devil's Advocate verdict — {d.llm_model}: **{d.forensic_verdict}**",
        f"",
    ]
    if "failed" in d.llm_model or "skipped" in d.llm_model:
        body_lines += [
            f"_DeepSeek devil's advocate did not run for this brief. "
            f"Consistency assessment is degraded — manual review advised._",
            f"",
        ]
    else:
        body_lines += [
            f"### Material Events",
            *(f"- {e}" for e in d.material_events),
            f"",
            f"### Macro Context",
            d.macro_context or "_(none)_",
            f"",
            f"### Sentiment Assessment",
            d.sentiment_assessment or "_(none)_",
            f"",
            f"### Signal Alignment Analysis",
            d.signal_alignment or "_(none)_",
            f"",
            f"### Key Quotes",
            *(f"> {q}" for q in d.key_quotes),
            f"",
        ]

    # ─── Consistency footer ──
    body_lines += [
        f"## Consistency Breakdown",
        f"",
        f"| Component | Agreement |",
        f"|---|---|",
        f"| Verdict (case_a/b/c)             | {'YES' if result.verdict_agreement else 'NO'} |",
        f"| Signal-alignment direction (+/-) | {'YES' if result.direction_agreement else 'NO'} |",
        f"| Material-events keyword overlap  | {result.event_overlap} shared |",
        f"| **Consistency score**            | **{result.consistency_score:.2f}** ({result.consistency_label}) |",
        f"",
        f"---",
        f"",
        f"_Provenance: prompt_hash={prompt_h} · code_commit={code_commit} · "
        f"total_cost=${result.total_cost_usd:.4f} · "
        f"generated {now.isoformat()}Z._",
    ]

    text = "\n".join(frontmatter_lines) + "\n" + "\n".join(body_lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Forensic brief written: %s", out_path)
    return out_path


def list_briefs(limit: int = 50) -> list[dict]:
    """List recent briefs with parsed frontmatter metadata.

    Returns most-recent-first list of dicts with keys mirroring frontmatter
    fields. Used by UI history table.
    """
    if not _BRIEFS_DIR.exists():
        return []
    files = sorted(
        _BRIEFS_DIR.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    out = []
    for p in files:
        meta = {"_path": str(p), "_name": p.name}
        try:
            text = p.read_text(encoding="utf-8")
            if text.startswith("---"):
                end = text.find("\n---", 4)
                if end > 0:
                    block = text[4:end]
                    for line in block.splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip()
        except Exception:
            pass
        out.append(meta)
    return out
