"""engine.research.external_audit — external adversarial LLM audit skeleton.

Self-audit by the same LLM that produced the work has structural blind
spots (see memory/feedback_self_audit_blind_spots_2026-06-13.md).
Mitigation #1 in that doctrine: route verdict-affecting code AND emitted
verdicts through an EXTERNAL LLM with adversarial prompt:

  "What known failure mode is this implementation likely to have missed?
   Be specific. Cite statistical / methodological literature."

The external LLM uses a DIFFERENT reasoning process than Claude — its
review surfaces blind-spot bugs that self-review cannot.

Architecture
============
- Provider-agnostic: this module defines the audit CONTRACT.
  Concrete provider adapters (OpenAI / DeepSeek / Gemini) live in
  engine/llm/providers/ and implement the ExternalAuditProvider Protocol.
- Audit subjects:
    1. verdict_event (factor_verdict_filed) — most critical
    2. template_diff (when new template ships)
    3. doctrine_lock (when standing memory is added)
- Output: data/research/external_audits.jsonl (append-only)
- Output routing: Inbox v2 digest pulls warnings into _LANE_ENGINE
- Decision authority: principal reads audit, decides. NEVER auto-
  amend verdict / revoke; audit output is INFORMATIONAL.

Cost considerations
===================
- 1 audit call per verdict ≈ $0.01-0.05 depending on provider
- At 15-20 verdicts/wk → $0.20-1.00/wk audit budget
- Provider should support skipping audit (e.g. "audit_provider=none")
  to allow dev-time bypass

Why NOT use Anthropic for this
==============================
- Different reasoning process REQUIRED. Same provider = same blind
  spots = no leverage. The whole point is independent reasoning.
- Anthropic Sonnet is producing the SPEC + writing the implementation.
  Audit must come from a different vendor.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_LOG_PATH = _REPO_ROOT / "data" / "research" / "external_audits.jsonl"


@_dc.dataclass(frozen=True)
class ExternalAudit:
    """Record of one external adversarial audit call."""
    audit_id:           str
    ts:                 str
    audit_subject:      str          # "verdict_event" / "template_diff" / "doctrine_lock"
    subject_ref:        str          # event_id / commit_sha / memory_id
    provider:           str          # "openai-gpt5" / "deepseek-v3" / "gemini-2.5" / "stub"
    audit_prompt:       str          # the adversarial prompt sent
    audit_response:     str          # external LLM's response (truncated)
    severity:           str          # "critical" / "concern" / "noted" / "no_issue" / "skipped"
    flagged_categories: list[str]    # ["statistical", "PIT", "lookahead", "lens", ...]
    cost_estimate_usd:  float        # for budget tracking

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


# ── Provider Protocol ─────────────────────────────────────────────


class ExternalAuditProvider(Protocol):
    """Implement this to plug in a non-Anthropic LLM."""

    name: str

    def adversarial_audit(
        self,
        *,
        subject_payload: dict,
        prompt:          str,
    ) -> tuple[str, str, list[str], float]:
        """Run the audit call. Returns:
          (response_text, severity, flagged_categories, cost_estimate_usd)

        severity must be one of: critical / concern / noted / no_issue / skipped
        """
        ...


# ── Built-in stub provider (no real LLM call) ─────────────────────


class _StubProvider:
    """Placeholder provider — logs that an audit WOULD have happened,
    returns 'skipped'. Real providers (OpenAI, DeepSeek, Gemini) plug
    into the same interface but actually call out.

    Active when no real provider is configured. Allows the audit hook
    to be wired into the dispatcher without API key dependency.
    """
    name = "stub"

    def adversarial_audit(
        self,
        *,
        subject_payload: dict,
        prompt:          str,
    ) -> tuple[str, str, list[str], float]:
        response = (
            "[stub provider] No external LLM configured. Set "
            "EXTERNAL_AUDIT_PROVIDER=<openai|deepseek|gemini> + matching "
            "API key in .streamlit/secrets.toml to enable adversarial "
            "review. Skipping audit call to preserve budget."
        )
        return response, "skipped", [], 0.0


_PROVIDER_REGISTRY: dict[str, ExternalAuditProvider] = {
    "stub": _StubProvider(),
}


def register_provider(provider: ExternalAuditProvider) -> None:
    """Register a concrete provider (called at module load by adapters)."""
    _PROVIDER_REGISTRY[provider.name] = provider


def _get_active_provider() -> ExternalAuditProvider:
    """Resolve the active provider via env var, with stub fallback."""
    import os
    name = os.environ.get("EXTERNAL_AUDIT_PROVIDER", "stub").lower()
    return _PROVIDER_REGISTRY.get(name, _PROVIDER_REGISTRY["stub"])


# ── Audit prompts (centralized so they're auditable themselves) ──


_VERDICT_AUDIT_PROMPT_TEMPLATE = """\
You are an independent reviewer of a quantitative factor research verdict
produced by another LLM-driven system. Adversarial mode: surface any
known failure modes in the methodology that the original system likely
missed.

Current date: {today_iso} (this matters — do NOT flag dates near today
as "future / look-ahead suspect"; only flag dates strictly AFTER today
or impossibly recent prediction windows). Your training cutoff may be
earlier than today; trust the date in this message.

Verdict payload (JSON):
{verdict_payload}

The payload includes `thresholds_applied` and `internal_machinery`
sections describing gates the originating system has ALREADY APPLIED.
DO NOT flag concerns about machinery already declared there
(multi-testing if thresholds_applied lists HLZ/Bonferroni; HAC SE if
internal_machinery names Newey-West; PIT if pit_whitelist is enforced;
spec drift if Stage 0 router ran). Flagging redundant concerns wastes
the reviewer's attention. Focus your adversarial energy on what the
machinery LIST does NOT cover.

Question: what KNOWN methodological failure modes — NOT covered by
the declared thresholds_applied / internal_machinery — might THIS
verdict suffer? Be specific. Cite statistical literature where
applicable.

Examples of failure modes WORTH flagging:
- Survivorship bias in the universe construction
- Cost model unrealistic for the signal frequency (e.g. high-turnover
  signal with 13bp cost assumption)
- Sample-period regime artifacts (e.g. pre/post-GFC distribution shift)
- Capacity constraints ignored (e.g. small-cap signal at $1B AUM)
- Replication claim that doesn't match the paper's actual setup
- Cross-sectional contamination (e.g. industry concentration in decile)
- Tail risk under-stated (e.g. short-vol Sharpe ignores variance crash)
- Spanning model mismatch (e.g. equity factor not spanned vs FF5+MOM)
- Implementation hazards specific to the signal type

Examples NOT to flag (already handled by internal_machinery):
- "Multi-testing not applied" when thresholds_applied lists Bonferroni
- "SE without autocorrelation correction" when HAC is declared
- "Look-ahead bias possible" when PIT whitelist is declared
- "Spec mismatch with claim" when Stage 0 router is declared

Severity rubric:
- critical: verdict likely WRONG; principal should not trust
- concern:  verdict has a meaningful caveat that should be surfaced
- noted:    minor issue worth recording but doesn't change verdict
- no_issue: methodology appears sound for this verdict

Return:
1. Severity (critical / concern / noted / no_issue)
2. Flagged categories (list of short tags, e.g. ["spanning", "PIT"])
3. Explanation (2-4 sentences citing literature where useful)
"""


# ── Public API ────────────────────────────────────────────────────


def audit_verdict_event(
    event: dict,
    *,
    provider: Optional[ExternalAuditProvider] = None,
    log_path: Optional[Path] = None,
) -> ExternalAudit:
    """Run adversarial audit on a factor_verdict_filed event.

    Returns the ExternalAudit record (also appended to log_path / default).

    NEVER raises — audit failure is a monitoring concern, not a hard error.
    """
    p = provider or _get_active_provider()

    # Truncate payload to keep prompt size bounded
    metrics = event.get("metrics") or {}
    n_trials = metrics.get("n_trials_at_dispatch") or metrics.get("n_trials") or 1

    # 2026-06-13 v2: include thresholds_applied + internal_machinery
    # metadata so DeepSeek doesn't false-flag concerns about gates we
    # have already applied internally (multi-testing, HAC SE, PIT
    # whitelist, replication anchors). LLM-as-judge has no internal
    # visibility otherwise; this metadata replaces "coaching the
    # adversary" with "telling the adversary what you've measured".
    try:
        from engine.research.verdict_thresholds import (
            t_green_threshold, t_marginal_threshold, alpha_t_green_threshold,
        )
        thresholds_applied = {
            "n_trials_denominator":    int(n_trials),
            "t_green_threshold":       float(t_green_threshold(n_trials)),
            "t_marginal_threshold":    float(t_marginal_threshold(n_trials)),
            "alpha_t_green_threshold": float(alpha_t_green_threshold(n_trials)),
            "method": ("HLZ-2016 floor 3.0 + Bonferroni body + HLZ ceiling 3.5; "
                         "scales with strategy_family n_trials per Bailey-LdP DSR"),
        }
    except Exception:
        thresholds_applied = {
            "n_trials_denominator": int(n_trials),
            "method": "thresholds_module_unavailable",
        }

    internal_machinery = {
        "se_method":         "Newey-West HAC lag 6 (autocorrelation-robust)",
        "pit_whitelist":     "signal_inputs validated against PIT_CORRECT_SOURCES",
        "replication_anchor":"template ships with M2 paper-replication test",
        "spec_drift_gate":   "Stage 0 claim-shape router pre-classified shape",
        "post_green_rigor":  "post-pub OOS + FF5+MOM spanning + borrow cost stress fire automatically",
    }

    payload = {
        "event_id":     event.get("event_id"),
        "subject_id":   event.get("subject_id"),
        "verdict":      event.get("verdict"),
        "family":       event.get("family"),
        "summary":      event.get("summary", "")[:300],
        "key_metrics": {
            k: v for k, v in metrics.items()
            if k in {
                "sharpe_gross", "nw_t_gross", "capm_alpha_t",
                "ff_complement_alpha_t", "ff_complement_anchor",
                "sharpe_diff_t", "jk_vs_a_t", "jk_vs_b_t",
                "strategy_family", "claim_family",
                "n_obs_months", "max_drawdown", "mean_pnl_monthly",
            }
        },
        # NEW: tell the auditor what gates we've already applied so it
        # focuses on remaining failure modes, not redundant flags
        "thresholds_applied": thresholds_applied,
        "internal_machinery": internal_machinery,
    }
    prompt = _VERDICT_AUDIT_PROMPT_TEMPLATE.format(
        today_iso=_dt.datetime.utcnow().strftime("%Y-%m-%d"),
        verdict_payload=json.dumps(payload, indent=2, ensure_ascii=False)[:3000],
    )

    try:
        response, severity, flagged, cost = p.adversarial_audit(
            subject_payload=payload, prompt=prompt,
        )
    except Exception as exc:
        logger.exception("external_audit: provider %s raised", p.name)
        response = f"PROVIDER_ERROR: {type(exc).__name__}: {exc}"
        severity = "skipped"
        flagged = []
        cost = 0.0

    record = ExternalAudit(
        audit_id           = str(uuid.uuid4()),
        ts                 = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        audit_subject      = "verdict_event",
        subject_ref        = event.get("event_id") or "",
        provider           = p.name,
        audit_prompt       = prompt[:1200],
        audit_response     = (response or "")[:2000],
        severity           = severity,
        flagged_categories = list(flagged),
        cost_estimate_usd  = float(cost),
    )

    try:
        out_path = log_path or AUDIT_LOG_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("external_audit: log write failed: %s", exc)

    return record


def recent_audits(
    days_back: int = 7,
    *,
    log_path: Optional[Path] = None,
) -> list[dict]:
    """Read recent audit records — used by inbox digest."""
    path = log_path or AUDIT_LOG_PATH
    if not path.is_file():
        return []
    cutoff = (_dt.datetime.utcnow() - _dt.timedelta(days=days_back)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ts", "") >= cutoff:
                out.append(row)
    return out


def severity_breakdown(audits: list[dict]) -> dict[str, int]:
    """Aggregate severity counts. Skip 'skipped' (no real audit happened)."""
    out = {"critical": 0, "concern": 0, "noted": 0, "no_issue": 0}
    for a in audits:
        sev = a.get("severity", "")
        if sev in out:
            out[sev] += 1
    return out
