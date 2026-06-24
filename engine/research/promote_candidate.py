"""engine/research/promote_candidate.py — SLM Phase 1: promotion
orchestrator. Single-call API that takes a candidate + audit blocks →
typed library YAML + validators + state-store row.

Contract:

    result = promote_candidate(
        strategy_id="post_earnings_drift_pit_sn",
        identity=StrategyIdentity(...),
        audit_blocks=AuditBlocks(cost_model=..., factor_exposure=...),
        pipeline_run_id="manual_2026-05-31_audit",
        actor="zhangxizhe",
        dry_run=False,
    )
    # → PromotionResult with library_yaml_path + state_record +
    #   validators_passed + validator_summary

Workflow:
  1. Render typed YAML scaffold (library_yaml_renderer)
  2. Write to library dir (refuses overwrite without explicit flag)
  3. Run 3 validators: library_integrity + cost_model_audit +
     factor_exposure_audit. ANY failure → rollback file write + raise.
  4. Create state-store row in PROPOSED state
  5. Transition PROPOSED → AUDITED with pipeline_run_id evidence

Dry-run mode: returns the YAML content as string + validator preview
WITHOUT writing files or touching the state store. Use this in CI
preview / human review before commit.

Doctrine:
  - Promote here is mechanical only. The library entry goes to AUDITED,
    not APPROVED. Human signoff (AUDITED → APPROVED) remains separate.
  - Validator failure ALWAYS rolls back the file write. Never leave
    partial state on disk.
"""
from __future__ import annotations

import datetime as _dt
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine.research.library_yaml_renderer import (
    StrategyIdentity,
    render_library_yaml_scaffold,
    write_library_yaml,
    yaml_to_string,
)
from engine.research.strategy_lifecycle import (
    AuditBlocks,
    GateNotMetError,
    InvalidTransitionError,
    StrategyRecord,
    StrategyState,
)
from engine.research.strategy_state_store import (
    DEFAULT_DB_PATH,
    create_strategy,
    get_strategy,
    transition,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"


# ── Result type ────────────────────────────────────────────────────────


@dataclass
class PromotionResult:
    """Output of promote_candidate(). Dry-run results omit state_record
    + library_yaml_path is the prospective path (file not written)."""

    strategy_id: str
    library_yaml_path: Path
    state_record: Optional[StrategyRecord]
    validators_passed: bool
    validator_summary: dict[str, str]
    yaml_preview: str = ""
    dry_run: bool = False
    rollback_performed: bool = False
    error: Optional[str] = None


# ── Validator runner ───────────────────────────────────────────────────


_VALIDATOR_MODULES = [
    "engine.research.library_integrity",
    "engine.research.library_cost_model_audit",
    "engine.research.library_factor_exposure_audit",
]


def _run_validator(module: str) -> tuple[bool, str]:
    """Run one validator module via `python -m module --strict`.

    Returns (passed, last_line_of_output). The --strict flag makes
    validators exit non-zero on any issue.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", module, "--strict"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            timeout=120,
        )
        last_line = (result.stdout.strip().splitlines() or [""])[-1]
        return result.returncode == 0, last_line
    except subprocess.TimeoutExpired:
        return False, f"validator {module} TIMEOUT"
    except Exception as exc:
        return False, f"validator {module} ERROR: {exc}"


def _run_all_validators() -> tuple[bool, dict[str, str]]:
    """Run all 3 library validators; return (all_passed, per-module summary)."""
    summary: dict[str, str] = {}
    all_passed = True
    for mod in _VALIDATOR_MODULES:
        passed, line = _run_validator(mod)
        summary[mod] = ("PASS: " if passed else "FAIL: ") + line
        all_passed = all_passed and passed
    return all_passed, summary


# ── Main orchestrator ──────────────────────────────────────────────────


def promote_candidate(
    *,
    identity: StrategyIdentity,
    audit_blocks: AuditBlocks,
    pipeline_run_id: str,
    actor: str,
    snooping_publication_date: Optional[_dt.date] = None,
    library_dir: Path = LIBRARY_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    overwrite_existing: bool = False,
    dry_run: bool = False,
    git_sha: Optional[str] = None,
    skip_validators: bool = False,
) -> PromotionResult:
    """Promote a candidate from PROPOSED to AUDITED state.

    Steps (all atomic — failure at any step rolls back disk + DB):
      1. Render typed YAML scaffold
      2. Write to library_dir/{strategy_id}.yaml
      3. Run 3 validators (--strict)
      4. Create state-store row in PROPOSED
      5. Transition PROPOSED → AUDITED with pipeline_run_id

    dry_run=True: render + return YAML preview, run validators against
    a temp copy, but DO NOT write to library or DB.

    skip_validators=True: ONLY for tests that don't want the subprocess
    overhead. Never use in production code paths.
    """
    last_audited = _dt.date.today()
    scaffold = render_library_yaml_scaffold(
        identity=identity,
        audit_blocks=audit_blocks,
        last_audited=last_audited,
        snooping_publication_date=snooping_publication_date,
    )
    yaml_str = yaml_to_string(scaffold)
    target_path = library_dir / f"{identity.strategy_id}.yaml"

    # ── DRY RUN ──────────────────────────────────────────────────────
    if dry_run:
        # Write to a temp file in library_dir so validators see it in
        # the SAME directory context, then delete after validation.
        temp_path = library_dir / f"_dryrun_{identity.strategy_id}.yaml"
        target_existed_before = target_path.exists()
        try:
            write_library_yaml(path=temp_path, scaffold=scaffold, overwrite=True)
            if skip_validators:
                v_passed, v_summary = True, {"skipped": "skip_validators=True"}
            else:
                v_passed, v_summary = _run_all_validators()
        finally:
            if temp_path.exists():
                temp_path.unlink()
        return PromotionResult(
            strategy_id=identity.strategy_id,
            library_yaml_path=target_path,
            state_record=None,
            validators_passed=v_passed,
            validator_summary=v_summary,
            yaml_preview=yaml_str,
            dry_run=True,
        )

    # ── LIVE PROMOTION ───────────────────────────────────────────────
    # Step 1+2: write YAML
    target_existed_before = target_path.exists()
    target_backup: Optional[Path] = None
    if target_existed_before and overwrite_existing:
        target_backup = target_path.with_suffix(".yaml.bak")
        shutil.copy2(target_path, target_backup)
    try:
        write_library_yaml(path=target_path, scaffold=scaffold,
                           overwrite=overwrite_existing)
    except FileExistsError as exc:
        return PromotionResult(
            strategy_id=identity.strategy_id,
            library_yaml_path=target_path,
            state_record=None,
            validators_passed=False,
            validator_summary={"write": f"FAIL: {exc}"},
            yaml_preview=yaml_str,
            error=str(exc),
        )

    # Step 3: validators
    if skip_validators:
        v_passed, v_summary = True, {"skipped": "skip_validators=True"}
    else:
        v_passed, v_summary = _run_all_validators()
    if not v_passed:
        # Rollback file write
        if target_existed_before and target_backup is not None:
            shutil.move(str(target_backup), str(target_path))
        else:
            target_path.unlink(missing_ok=True)
        return PromotionResult(
            strategy_id=identity.strategy_id,
            library_yaml_path=target_path,
            state_record=None,
            validators_passed=False,
            validator_summary=v_summary,
            yaml_preview=yaml_str,
            rollback_performed=True,
            error="validators failed; YAML write rolled back",
        )

    # Cleanup backup on success
    if target_backup is not None:
        target_backup.unlink(missing_ok=True)

    # Step 4+5: state store create + transition. If either fails,
    # roll back the YAML write to avoid orphan files.
    try:
        try:
            existing = get_strategy(identity.strategy_id, db_path=db_path)
            # Already exists — log + skip create (transition still attempted)
            logger.info(
                "strategy %s already in state store as %s; skipping create",
                identity.strategy_id, existing.current_state.value,
            )
        except KeyError:
            create_strategy(
                strategy_id=identity.strategy_id,
                library_yaml_path=str(target_path),
                candidate_pipeline_run_id=pipeline_run_id,
                parent_strategy_id=identity.parent_strategy_id,
                notes=f"Promoted via Phase 1 automation; "
                      f"role={audit_blocks.factor_exposure.proposed_role}",
                actor=actor,
                db_path=db_path,
            )

        record = transition(
            strategy_id=identity.strategy_id,
            to_state=StrategyState.AUDITED,
            actor=actor,
            reason=f"Phase 1 automated promotion; pipeline_run={pipeline_run_id}",
            has_candidate_pipeline_run=True,
            git_sha=git_sha,
            db_path=db_path,
        )
    except (InvalidTransitionError, GateNotMetError) as exc:
        # Roll back YAML write
        if target_existed_before and target_backup is not None and target_backup.exists():
            shutil.move(str(target_backup), str(target_path))
        elif not target_existed_before:
            target_path.unlink(missing_ok=True)
        return PromotionResult(
            strategy_id=identity.strategy_id,
            library_yaml_path=target_path,
            state_record=None,
            validators_passed=v_passed,
            validator_summary=v_summary,
            yaml_preview=yaml_str,
            rollback_performed=True,
            error=f"state store transition failed: {exc}",
        )

    return PromotionResult(
        strategy_id=identity.strategy_id,
        library_yaml_path=target_path,
        state_record=record,
        validators_passed=True,
        validator_summary=v_summary,
        yaml_preview=yaml_str,
    )
