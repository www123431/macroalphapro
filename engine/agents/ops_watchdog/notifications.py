"""
engine/agents/ops_watchdog/notifications.py — Severity-driven dispatcher.

Per spec §2.6 (LOCKED 4 channels, modifying requires spec amend):
  - LIGHT  : dashboard widget yellow
  - MEDIUM : + Windows toast (win10toast, 10s)
  - SEVERE : + Windows toast persist (30s)
             + email (smtplib, if SMTP configured in .streamlit/secrets.toml)
             + halt flag in circuit_breaker.json (level=SEVERE, auto_reset=False)

Each channel is FAIL-SOFT: any individual channel that errors (e.g. win10toast
not installed, SMTP host unreachable, disk write failure) is logged but does
NOT propagate. The orchestrator captures per-channel result so the trace JSON
records which channels successfully fired.

INVARIANT (spec §六): only Watchdog SEVERE can SET halt flag. Only human can
CLEAR via dashboard button (existing engine.circuit_breaker.manual_reset).
This module NEVER calls manual_reset.

PATH NOTE: Spec mentions `.streamlit/circuit_breaker.json`, but the actual
production path is `engine/state/circuit_breaker.json` per
engine.circuit_breaker._STATE_FILE. Implementation follows the real code,
not the stale spec wording. Phase 6 amendment will reconcile.

dry_run mode: emit_notification short-circuits at entry (returns all-False
result) so CI smoke / debug runs don't emit user-visible toasts/emails or
halt production.
"""
from __future__ import annotations

import datetime
import json
import logging
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: paths + config readers
# ─────────────────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _widget_state_path() -> Path:
    """Dashboard widget state lives next to trace JSON for co-location."""
    return _repo_root() / "data" / "ops_watchdog" / "widget_state.json"


def _read_smtp_config() -> Optional[dict]:
    """
    Read SMTP config from .streamlit/secrets.toml. Returns None if missing or
    malformed (Phase 4 fail-soft — email is OPTIONAL per spec §2.6).

    Expected schema (under [ops_watchdog.smtp] section):
      host       = "smtp.gmail.com"
      port       = 587
      user       = "alerts@example.com"
      password   = "..."
      from_addr  = "alerts@example.com"
      to_addrs   = ["recipient@example.com"]
      use_tls    = true
    """
    secrets_path = _repo_root() / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None
    try:
        import tomllib   # py 3.11+ stdlib
    except ImportError:
        try:
            import tomli as tomllib   # py 3.10 fallback
        except ImportError:
            return None
    try:
        with open(secrets_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        logger.warning("watchdog notifications: secrets.toml parse failed: %s", exc)
        return None
    cfg = (data.get("ops_watchdog") or {}).get("smtp") or {}
    required = ("host", "port", "user", "password", "from_addr", "to_addrs")
    if not all(k in cfg for k in required):
        return None
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Channel 1: dashboard widget state (always for non-NONE severity)
# ─────────────────────────────────────────────────────────────────────────────

def _write_dashboard_widget(
    severity:    str,
    summary:     str,
    findings:    list[dict],
    today_iso:   str,
    repair_info: Optional[dict] = None,
) -> bool:
    """
    Persist widget state to data/ops_watchdog/widget_state.json. Streamlit
    dashboard page reads this for the operations health widget.
    """
    try:
        path = _widget_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "spec_id":        63,
            "updated_at_iso": datetime.datetime.utcnow().isoformat() + "Z",
            "today_iso":      today_iso,
            "severity":       severity,
            "summary":        summary[:500],
            "n_findings":     len(findings),
            "findings_brief": [
                {
                    "finding_id": f.get("finding_id"),
                    "rule_name":  f.get("rule_name"),
                    "severity":   f.get("severity"),
                }
                for f in findings[:10]
            ],
            "auto_repair":    repair_info or {},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as exc:
        logger.warning("watchdog notifications: dashboard widget write failed: %s",
                       exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Channel 2: Windows toast (medium + severe)
# ─────────────────────────────────────────────────────────────────────────────

def _send_windows_toast(
    severity:         str,
    summary:          str,
    duration_seconds: int,
) -> bool:
    """
    Try win10toast. Fail-soft if library not installed or Windows API errors.
    spec §五 Gate 10 validates real toast render on Windows 10 Home China in
    Phase 5; here we just attempt + log result.
    """
    try:
        from win10toast import ToastNotifier
    except ImportError:
        logger.info("watchdog notifications: win10toast not installed — skipping toast")
        return False
    try:
        title = f"Ops Watchdog [{severity.upper()}]"
        toaster = ToastNotifier()
        # `threaded=True` returns immediately; toast displays asynchronously.
        # `threaded=False` would block; not acceptable in cron context.
        toaster.show_toast(
            title       = title,
            msg         = summary[:200],
            duration    = max(5, int(duration_seconds)),
            threaded    = True,
            icon_path   = None,
        )
        return True
    except Exception as exc:
        logger.warning("watchdog notifications: toast send failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Channel 3: email (severe only, if SMTP configured)
# ─────────────────────────────────────────────────────────────────────────────

def _send_email_if_configured(
    summary:   str,
    findings:  list[dict],
    today_iso: str,
) -> bool:
    """
    Send a plain-text email alert per .streamlit/secrets.toml [ops_watchdog.smtp]
    config. Fail-soft on any error (missing config / DNS / auth / TLS).
    """
    cfg = _read_smtp_config()
    if cfg is None:
        logger.info("watchdog notifications: SMTP not configured — skipping email")
        return False

    try:
        subject = f"[Ops Watchdog SEVERE] {today_iso} — {summary[:80]}"
        lines = [
            f"Ops Watchdog SEVERE finding on {today_iso}",
            "",
            f"Summary: {summary}",
            "",
            f"Total findings: {len(findings)}",
            "",
            "Per-finding (first 10):",
        ]
        for f in findings[:10]:
            lines.append(
                f"  - finding_id={f.get('finding_id')} "
                f"rule={f.get('rule_name')} severity={f.get('severity')}"
            )
        lines += [
            "",
            "Trace JSON: data/ops_watchdog/{}_run.json".format(today_iso),
            "Halt flag: engine/state/circuit_breaker.json (SEVERE; clear via dashboard)",
            "",
            "-- Ops Watchdog Agent v1.0 (spec id=63)",
        ]
        body = "\n".join(lines)

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = cfg["from_addr"]
        msg["To"]      = ", ".join(cfg["to_addrs"])

        host = cfg["host"]
        port = int(cfg["port"])
        use_tls = bool(cfg.get("use_tls", True))

        with smtplib.SMTP(host, port, timeout=15) as server:
            server.ehlo()
            if use_tls:
                server.starttls()
                server.ehlo()
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["from_addr"], cfg["to_addrs"], msg.as_string())
        return True
    except Exception as exc:
        logger.warning("watchdog notifications: email send failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Channel 4: halt flag (severe only)
# ─────────────────────────────────────────────────────────────────────────────

def _set_halt_flag(reason: str) -> bool:
    """
    Persist SEVERE halt flag via engine.circuit_breaker.set_external_halt_flag
    (public API since 2026-05-13 spec amendment 3 follow-up). The public API
    handles `ops_watchdog:` reason prefixing + locked state construction;
    this module just calls it with `source='ops_watchdog'`.

    INVARIANT: only Watchdog SEVERE can SET; only human can CLEAR via
    engine.circuit_breaker.manual_reset (called from Streamlit dashboard
    "Acknowledge Watchdog Halt" button on pages/ops_watchdog.py). This
    module never invokes manual_reset.
    """
    try:
        from engine.circuit_breaker import set_external_halt_flag
        set_external_halt_flag(reason=reason, source="ops_watchdog")
        logger.error("Watchdog SEVERE — circuit_breaker halt flag SET: %s", reason)
        return True
    except Exception as exc:
        logger.warning("watchdog notifications: halt flag write failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry: emit_notification
# ─────────────────────────────────────────────────────────────────────────────

# Toast duration thresholds (spec §2.6)
_TOAST_DURATION_MEDIUM_S: int = 10
_TOAST_DURATION_SEVERE_S: int = 30


def emit_notification(
    *,
    severity:    str,
    summary:     str,
    findings:    list[dict],
    today_iso:   str,
    repair_info: Optional[dict] = None,
    dry_run:     bool = False,
) -> dict:
    """
    Dispatch per spec §2.6 severity-mapped channels.

    Args:
        severity:    "none" / "light" / "medium" / "severe"
        summary:     short narrative (≤ 500 chars used; trimmed)
        findings:    list of findings_summary entries (rule_name + severity etc.)
        today_iso:   ISO date string for the Watchdog run
        repair_info: optional dict from auto_repair_summary
        dry_run:     if True, return all-False without emitting anything

    Returns:
        dict {dashboard: bool, toast: bool, email: bool, halt_flag: bool}
        indicating which channels successfully fired.
    """
    result = {"dashboard": False, "toast": False, "email": False, "halt_flag": False}

    if dry_run:
        result["skipped_reason"] = "dry_run"
        return result
    if severity not in ("light", "medium", "severe"):
        result["skipped_reason"] = "severity_none_or_invalid"
        return result

    # Channel 1: dashboard widget (always for light/medium/severe)
    result["dashboard"] = _write_dashboard_widget(
        severity=severity, summary=summary, findings=findings,
        today_iso=today_iso, repair_info=repair_info,
    )

    # Channel 2: Windows toast (medium + severe)
    if severity in ("medium", "severe"):
        duration = _TOAST_DURATION_SEVERE_S if severity == "severe" \
            else _TOAST_DURATION_MEDIUM_S
        result["toast"] = _send_windows_toast(
            severity=severity, summary=summary, duration_seconds=duration,
        )

    # Channel 3: email + Channel 4: halt flag (severe only)
    if severity == "severe":
        result["email"] = _send_email_if_configured(
            summary=summary, findings=findings, today_iso=today_iso,
        )
        result["halt_flag"] = _set_halt_flag(reason=summary)

    return result
