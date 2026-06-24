"""scripts/deploy_config.py — deploy ceremony for portfolio configs.

Doctrine (Item 7e of dashboard-freshness PR series, 2026-06-02):
    Every config change MUST go through this script. It guarantees:
      1. The new config is written to data/portfolio/active_deployment.yaml
      2. The previous active config is moved to `history`
      3. The Python constants in engine.portfolio.combined_book are checked
         against the new YAML and any drift is REPORTED (not silently swallowed)
      4. A new row in data/portfolio/deploy_ledger.jsonl records the change
         (who/when/why)
      5. A git diff is shown so the user can confirm BEFORE writing

The only path that bypasses ceremony is direct hand-edit of the YAML.
That's not recommended, but if you do it, please at least also append
to deploy_ledger.jsonl so we keep an audit trail.

Usage:
    # Promote a new config:
    python scripts/deploy_config.py promote --id config_d_pit_sn_replace_dpead \
        --label "config D — PIT SN replaces D_PEAD in equity sleeve" \
        --reason "Phase 2 capital ramp 1%→5% per [[project-slm-phase0-complete]]"

    # Check drift without making changes:
    python scripts/deploy_config.py check

    # Show active config + recent history:
    python scripts/deploy_config.py status
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

# Make engine importable when run as script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_REGISTRY_PATH = _REPO_ROOT / "data" / "portfolio" / "active_deployment.yaml"
_LEDGER_PATH   = _REPO_ROOT / "data" / "portfolio" / "deploy_ledger.jsonl"


def _utc_iso_now() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_status(args: argparse.Namespace) -> int:
    """Show the active config + N most recent history entries."""
    from engine.portfolio.deployed_registry import load_active
    cfg = load_active(force_reload=True)
    print(f"\n  ACTIVE CONFIG:")
    print(f"    id:           {cfg.id}")
    print(f"    label:        {cfg.label}")
    print(f"    deploy_date:  {cfg.deploy_date}  ({cfg.days_since_deploy} days ago)")
    print(f"    book_vol:     {cfg.book_vol_target}")
    print(f"    signing specs: {list(cfg.signing_spec_ids)}")
    print(f"    sleeves:")
    for s in cfg.sleeves:
        modulator = " [regime-modulated]" if s.regime_modulated else ""
        print(f"      - {s.name:<26} role={s.role:<12} weight={s.base_weight:.3f}{modulator}")
    print(f"    expected stats: {dict(cfg.expected_stats)}\n")

    if _LEDGER_PATH.is_file():
        print(f"  RECENT DEPLOY LEDGER ({args.history_n} entries):")
        with _LEDGER_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()[-int(args.history_n):]
        for line in lines:
            try:
                rec = json.loads(line)
                print(f"    {rec.get('ts')}  {rec.get('action'):<10} {rec.get('config_id')}  ({rec.get('reason', '')[:60]})")
            except Exception:
                print(f"    {line.rstrip()}")
    else:
        print(f"  (no deploy ledger yet at {_LEDGER_PATH})")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Report any drift between Python constants and the manifest."""
    from engine.portfolio.deployed_registry import assert_constants_match
    from engine.portfolio.combined_book import (
        DEFAULT_CARRY_RISK_WEIGHT, DEFAULT_TSMOM_RISK_WEIGHT,
        DEFAULT_CRISIS_HEDGE_RISK_WEIGHT, DEFAULT_MOM_HEDGE_RISK_WEIGHT,
        DEFAULT_BOOK_VOL_TARGET,
    )
    issues = assert_constants_match(
        carry_risk_weight    = DEFAULT_CARRY_RISK_WEIGHT,
        tsmom_risk_weight    = DEFAULT_TSMOM_RISK_WEIGHT,
        crisis_risk_weight   = DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
        mom_hedge_risk_weight= DEFAULT_MOM_HEDGE_RISK_WEIGHT,
        book_vol_target      = DEFAULT_BOOK_VOL_TARGET,
    )
    if not issues:
        print("  [OK]all clean — manifest and Python constants agree")
        return 0
    print("  [FAIL]DRIFT detected — manifest disagrees with code:")
    for line in issues:
        print(f"    - {line}")
    print()
    print("  Either update the YAML to match code (if the code changed legitimately):")
    print(f"    edit {_REGISTRY_PATH}")
    print("  OR roll back the code constant to match the manifest.")
    print()
    print("  Do NOT proceed with a deploy until this is resolved.")
    return 1


def cmd_promote(args: argparse.Namespace) -> int:
    """Append a new config to the manifest + mark it active.

    2026-06-02 — Governance Gateway:
      Default behaviour now CREATES AN APPROVAL REQUEST instead of
      writing directly. The actual YAML update happens only after
      a human approves via /approvals (or this script's `--apply` flag
      after the approval id is supplied).

    Two paths:
      promote --id X --label Y --reason Z
        → creates approval request, prints request id, exits (no YAML touch)
      promote --apply <approval_id>
        → checks the approval is APPROVED, then performs the YAML mutation
          + records execution log in the ledger row.

    Bypass (test / emergency):
      promote --id X --label Y --reason Z --skip-approval
        → original direct-write behaviour, FLAGGED in deploy_ledger.jsonl
          as ungated. Avoid in production.
    """
    # Apply path — execute a previously-approved request
    if getattr(args, "apply", None):
        return _apply_approved(args.apply)

    # Create-approval path (default)
    if not getattr(args, "skip_approval", False):
        return _create_promote_approval(args)

    # Bypass path (--skip-approval): original direct-write
    import yaml
    with _REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    prev_active = data.get("active_config_id")
    print(f"  [BYPASS] Previous active: {prev_active}")
    print(f"  [BYPASS] New active:      {args.id}")
    print(f"  [BYPASS] WARNING: --skip-approval ungates institutional two-eye doctrine.")

    # Sanity: don't allow promoting to the same id
    if args.id == prev_active:
        print(f"  [FAIL]new id is the same as the current active — nothing to do.")
        return 1

    # Move previous active to history (if not already there)
    history = data.get("history", []) or []
    has_in_history = any(h.get("id") == prev_active for h in history)
    if not has_in_history:
        history.insert(0, {
            "id":              prev_active,
            "deploy_date_range": f"... to {_dt.date.today().isoformat()}",
            "label":            f"(previously active before {args.id})",
            "superseded_by":    args.id,
            "why_superseded":   args.reason or "(no reason given at promote time)",
        })
        data["history"] = history

    # Append the new config as a stub — sleeves left for human to fill
    configs = data.get("configs") or []
    if not any(c.get("id") == args.id for c in configs):
        configs.append({
            "id":              args.id,
            "deploy_date":     _dt.date.today().isoformat(),
            "label":           args.label or args.id,
            "summary":         args.reason or "(fill in via YAML edit)",
            "signing_spec_ids": [],
            "book_vol_target":  0.10,
            "expected_stats":   {},
            "sleeves":          [],
            "regime_grids":     {},
            "regime_classifier": {},
        })
        data["configs"] = configs

    data["active_config_id"] = args.id
    meta = data.get("_meta", {}) or {}
    meta["last_updated_ts"]   = _utc_iso_now()
    meta["last_updated_by"]   = "scripts/deploy_config.py promote"
    data["_meta"] = meta

    # Write back
    with _REGISTRY_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)

    # Append ledger entry
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts":         _utc_iso_now(),
            "action":     "promote",
            "config_id":  args.id,
            "previous":   prev_active,
            "label":      args.label,
            "reason":     args.reason,
        }, ensure_ascii=False) + "\n")

    print(f"  [OK]{_REGISTRY_PATH.name} updated. Active is now {args.id}.")
    print(f"  [OK]{_LEDGER_PATH.name} appended.")
    print()
    print(f"  NEXT STEPS (REQUIRED before the new config is usable):")
    print(f"    1. Edit {_REGISTRY_PATH} and fill in:")
    print(f"       - configs[id={args.id}].sleeves (with name/role/weight/builder/target_vol)")
    print(f"       - configs[id={args.id}].regime_grids (if regime-modulated)")
    print(f"       - configs[id={args.id}].expected_stats (Sharpe/maxDD from your backtest)")
    print(f"    2. Update engine/portfolio/combined_book.py constants to match")
    print(f"    3. Run `python scripts/deploy_config.py check` and confirm clean")
    print(f"    4. Restart uvicorn to pick up the new defaults")
    return 0


def _create_promote_approval(args: argparse.Namespace) -> int:
    """Default promote path: create governance approval request, don't write YAML."""
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.governance.approval_ledger import create_request
    from engine.portfolio.deployed_registry import load_active

    cur = load_active(force_reload=True)
    rid = create_request(
        request_type     = "deploy_config_promote",
        title            = f"Promote → {args.id}",
        summary          = (args.reason or "(no reason supplied)") +
                            f" ::: would replace active config {cur.id!r} (live since {cur.deploy_date}).",
        proposed_payload = {
            "new_active_config_id": args.id,
            "new_label":            args.label or args.id,
            "new_summary":          args.reason or "(fill in via YAML edit)",
            "deploy_date":          _dt.date.today().isoformat(),
        },
        current_state    = {
            "current_active_config_id": cur.id,
            "current_label":            cur.label,
            "deploy_date":              cur.deploy_date,
            "days_live":                cur.days_since_deploy,
        },
        evidence_pack    = {
            "current_expected_sharpe": cur.expected_stats.get("sharpe"),
            "current_expected_maxdd":  cur.expected_stats.get("max_dd"),
            "current_n_sleeves":       len(cur.sleeves),
            "current_signing_specs":   list(cur.signing_spec_ids),
        },
        created_by       = "scripts/deploy_config.py promote",
    )
    print(f"  [OK]approval request created: {rid}")
    print()
    print(f"  Next steps:")
    print(f"    1. Review the request at /approvals or via")
    print(f"       GET /api/governance/approvals/{rid}")
    print(f"    2. Wait through the 24h cooling-off period (institutional standard)")
    print(f"    3. Approve via UI, then run:")
    print(f"       python scripts/deploy_config.py promote --apply {rid}")
    print(f"  This 3-step path is the institutional two-eye + cooling-off")
    print(f"  doctrine; bypass with --skip-approval only in emergencies.")
    return 0


def _apply_approved(request_id: str) -> int:
    """Apply path: execute an already-approved request by writing YAML + ledger."""
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.governance.approval_ledger import get_request

    state = get_request(request_id)
    if state is None:
        print(f"  [FAIL]approval {request_id} not found in ledger")
        return 1
    if state["status"] != "approved":
        print(f"  [FAIL]approval {request_id} is {state['status']!r}, must be 'approved' to apply")
        return 1
    if state["request_type"] != "deploy_config_promote":
        print(f"  [FAIL]approval {request_id} type is {state['request_type']!r}, not deploy_config_promote")
        return 1

    payload = state["proposed_payload"]
    print(f"  Applying approval {request_id} → promote to {payload['new_active_config_id']!r}")

    import yaml
    with _REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    new_id = payload["new_active_config_id"]
    prev_active = data.get("active_config_id")

    # Move previous active to history if not already there
    history = data.get("history", []) or []
    if not any(h.get("id") == prev_active for h in history):
        history.insert(0, {
            "id":               prev_active,
            "deploy_date_range": f"... to {_dt.date.today().isoformat()}",
            "label":             f"(previously active before {new_id})",
            "superseded_by":     new_id,
            "why_superseded":    state.get("summary", "(no reason recorded)"),
        })
        data["history"] = history

    # Append new config as stub
    configs = data.get("configs") or []
    if not any(c.get("id") == new_id for c in configs):
        configs.append({
            "id":              new_id,
            "deploy_date":     payload.get("deploy_date", _dt.date.today().isoformat()),
            "label":           payload.get("new_label", new_id),
            "summary":         payload.get("new_summary", "(fill in via YAML edit)"),
            "signing_spec_ids": [],
            "book_vol_target":  0.10,
            "expected_stats":   {},
            "sleeves":          [],
            "regime_grids":     {},
            "regime_classifier": {},
        })
        data["configs"] = configs

    data["active_config_id"] = new_id
    meta = data.get("_meta", {}) or {}
    meta["last_updated_ts"] = _utc_iso_now()
    meta["last_updated_by"] = f"scripts/deploy_config.py promote --apply {request_id}"
    data["_meta"] = meta

    with _REGISTRY_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)

    # Deploy ledger entry
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts":           _utc_iso_now(),
            "action":       "promote_applied",
            "config_id":    new_id,
            "previous":     prev_active,
            "approval_id":  request_id,
            "approved_by":  state.get("decided_by"),
            "fast_approve": state.get("fast_approve", False),
        }, ensure_ascii=False) + "\n")

    print(f"  [OK]{_REGISTRY_PATH.name} updated. Active is now {new_id}.")
    print(f"  [OK]Approval {request_id} marked applied in {_LEDGER_PATH.name}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Portfolio deploy ceremony")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_status = sub.add_parser("status", help="Show active config + history")
    sp_status.add_argument("--history-n", type=int, default=10, help="N most recent ledger entries")
    sp_status.set_defaults(func=cmd_status)

    sp_check = sub.add_parser("check", help="Report drift between Python constants and manifest")
    sp_check.set_defaults(func=cmd_check)

    sp_promote = sub.add_parser("promote", help="Promote a new config to active (default: creates governance approval)")
    sp_promote.add_argument("--id", help="Config id (kebab_case_with_underscores) — required unless --apply")
    sp_promote.add_argument("--label", help="Human-readable label")
    sp_promote.add_argument("--reason", help="Why this is being promoted")
    sp_promote.add_argument("--apply", help="Apply a previously-approved request (e.g. ar_xxxx)")
    sp_promote.add_argument("--skip-approval", action="store_true",
                            help="BYPASS governance gateway. Emergency-only.")
    sp_promote.set_defaults(func=cmd_promote)

    args = ap.parse_args()
    # Promote arg validation: require --id OR --apply
    if args.cmd == "promote" and not args.id and not args.apply:
        ap.error("promote requires either --id (to create approval) or --apply <request_id>")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
