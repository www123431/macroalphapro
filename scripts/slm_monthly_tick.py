"""scripts/slm_monthly_tick.py — Phase 5: monthly tick of paper_trade
+ shadow_trade monitors.

Designed to run as a scheduled task (Windows Task Scheduler / cron):

  Trigger:  monthly, last calendar day of the month
  Action:   python scripts/slm_monthly_tick.py
  Result:   - tick every PAPER_TRADE + SHADOW sleeve through its
              role-specific 3-layer validator
            - append result to data/research/slm_monthly_tick_log.jsonl
            - exit 0 normal, 1 on errors
            - print summary to stdout

For each sleeve the script:
  1. Resolves role from sleeve.audit_blocks().factor_exposure.proposed_role
  2. Resolves family from library YAML (for family-aware n_trials)
  3. Auto-derives prior_mean_sharpe from cost_model.multi_aum_sharpe_sleeve.at_10M
     (the audited honest target at the safe deploy band)
  4. Calls paper_trade_monitor.tick_single_sleeve()
  5. Logs full result + decision

USAGE:
  python scripts/slm_monthly_tick.py                  # live tick
  python scripts/slm_monthly_tick.py --dry-run        # log only, no DB writes
  python scripts/slm_monthly_tick.py --verbose        # full output
  python scripts/slm_monthly_tick.py --register-task  # print Windows TaskSched setup

Windows Task Scheduler registration is NOT auto-executed (would modify
system state outside the repo). The --register-task flag prints the
schtasks.exe command for the user to copy-paste.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine.research.sleeves  # noqa: F401  triggers sleeve registration

from engine.research.paper_trade_monitor import (
    MonitorTickResult, tick_single_sleeve,
)
from engine.research.sleeve_registry import get_sleeve
from engine.research.strategy_lifecycle import SleeveRole, StrategyState
from engine.research.strategy_state_store import list_strategies

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "data" / "research" / "slm_monthly_tick_log.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("slm_monthly_tick")


def _resolve_role_and_family(strategy_id: str) -> tuple[Optional[SleeveRole],
                                                          Optional[str],
                                                          Optional[float]]:
    """Pull role, family, and prior_mean_sharpe from the registered
    sleeve's library YAML (via audit_blocks)."""
    try:
        sleeve = get_sleeve(strategy_id)
    except KeyError:
        return None, None, None
    try:
        audit = sleeve.audit_blocks()
    except Exception as exc:
        logger.warning("audit_blocks failed for %s: %s", strategy_id, exc)
        return None, None, None
    role_str = audit.factor_exposure.proposed_role
    if not role_str:
        return None, None, None
    role = SleeveRole.from_yaml_value(role_str)
    # Family read from raw YAML
    family = None
    try:
        import yaml as _pyyaml
        raw = _pyyaml.safe_load(sleeve.library_yaml_path.read_text(encoding="utf-8"))
        family = raw.get("family")
    except Exception as exc:
        logger.warning("family parse failed for %s: %s", strategy_id, exc)
    prior_mean = None
    if audit.cost_model.multi_aum_sharpe_sleeve:
        prior_mean = audit.cost_model.multi_aum_sharpe_sleeve.at_10M
    return role, family, prior_mean


def _append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _summarize_result(r: MonitorTickResult) -> dict:
    out: dict = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "strategy_id": r.strategy_id,
        "role": r.role.value if r.role else None,
        "months_observed": r.months_observed,
        "action_taken": r.action_taken,
        "error": r.error,
    }
    if r.metric_result:
        out["metric"] = {
            "name": r.metric_result.metric_name,
            "value": r.metric_result.metric_value,
            "t_stat": r.metric_result.t_stat,
            "evidence_passed_minimum": r.metric_result.evidence_passed,
        }
    if r.boundary_result:
        out["boundary"] = {
            "decision": r.boundary_result.decision.value,
            "upper_critical_t": r.boundary_result.upper_critical_t,
        }
    if r.three_layer_result:
        tl = r.three_layer_result
        out["three_layer"] = {
            "final": tl.final_decision.value,
            "layer1_bayesian_vote": tl.layer1_vote,
            "layer1_posterior_mean": tl.layer1_bayesian.posterior_mean,
            "layer1_p_above_threshold": tl.layer1_bayesian.posterior_prob_above_threshold,
            "layer2_dsr_vote": tl.layer2_vote,
            "layer2_deflated_sr": tl.layer2_deflated_sr.deflated_sr,
            "layer3_obf_vote": tl.layer3_vote,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                     help="evaluate but do not transition states")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--register-task", action="store_true",
                     help="print Windows Task Scheduler setup command + exit")
    args = p.parse_args()

    if args.register_task:
        python_exe = sys.executable
        script = str(Path(__file__).resolve())
        cmd = (
            f'schtasks /Create /TN "SLM_Monthly_Tick" '
            f'/TR "\\"{python_exe}\\" \\"{script}\\"" '
            f'/SC MONTHLY /MO LASTDAY /ST 18:00 /F'
        )
        print("Windows Task Scheduler registration command:")
        print(f"\n  {cmd}\n")
        print("Run this in elevated PowerShell to register a monthly tick at 18:00")
        print("on the last day of each month. Logs append to "
              f"{LOG_PATH.relative_to(REPO_ROOT)}")
        return 0

    print("=" * 90)
    print(f" SLM Monthly Tick — {_dt.datetime.now().isoformat()}")
    print(f" dry_run={args.dry_run}")
    print("=" * 90)

    sleeves_in_paper = list_strategies(state=StrategyState.PAPER_TRADE)
    sleeves_in_shadow = list_strategies(state=StrategyState.SHADOW)
    print(f"\n  PAPER_TRADE sleeves: {len(sleeves_in_paper)}")
    print(f"  SHADOW sleeves:      {len(sleeves_in_shadow)}")
    if not sleeves_in_paper and not sleeves_in_shadow:
        print("\n  Nothing to tick. Exiting.")
        return 0

    n_errors = 0
    for rec in sleeves_in_paper:
        sid = rec.strategy_id
        print(f"\n  ── PAPER_TRADE: {sid} ─────────────────────────────────────")
        role, family, prior = _resolve_role_and_family(sid)
        if role is None:
            print(f"    SKIP — could not resolve role/family/prior from library YAML")
            _append_log({
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "strategy_id": sid, "skipped": True,
                "reason": "role/family/prior unresolved",
            })
            n_errors += 1
            continue
        print(f"    role={role.value}  family={family}  prior_sharpe={prior:.3f}")

        try:
            result = tick_single_sleeve(
                strategy_id=sid,
                role=role,
                family=family,
                prior_mean_sharpe=prior,
                auto_reject_on_early_stop_loss=not args.dry_run,
            )
        except Exception as exc:
            logger.exception("tick failed for %s", sid)
            _append_log({
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "strategy_id": sid, "error": str(exc),
            })
            n_errors += 1
            continue

        summary = _summarize_result(result)
        _append_log(summary)
        print(f"    months_observed: {result.months_observed}")
        if result.metric_result:
            m = result.metric_result
            print(f"    {m.metric_name}: {m.metric_value:+.3f} (t={m.t_stat:+.3f})")
        if result.three_layer_result:
            tl = result.three_layer_result
            print(f"    3-layer composite: {tl.final_decision.value}")
            print(f"      layer1: {tl.layer1_vote}  layer2: {tl.layer2_vote}  layer3: {tl.layer3_vote}")
        print(f"    action: {result.action_taken}")
        if result.error:
            print(f"    ERROR: {result.error}")
            n_errors += 1

    if sleeves_in_shadow:
        print(f"\n  [NOTE] SHADOW sleeves present but Phase 5 only ticks PAPER_TRADE.")
        print(f"         shadow_trade_monitor will be added when first sleeve reaches SHADOW.")

    print(f"\n{'=' * 90}")
    print(f" Tick complete. Errors: {n_errors}. Full log: {LOG_PATH.relative_to(REPO_ROOT)}")
    print(f"{'=' * 90}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
