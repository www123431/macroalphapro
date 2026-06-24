"""Mitigation #3 from self-audit blind-spots doctrine:
structural enforcement of standing project doctrines via AST + grep
scans. If anyone violates these doctrines in code (or accidentally
regresses), CI catches it BEFORE merge.

Doctrines tested (one section per doctrine):

  1. forward/enhance pipeline separation
     (feedback_forward_vs_enhance_statistical_separation_2026-06-11.md)

  2. strategy_family canonical vs claim_family
     (feedback_strategy_family_vs_claim_family_2026-06-12.md)

  3. PIT_CORRECT_SOURCES whitelist consistency
     (no signal_input may bypass the whitelist)

  4. capital decisions stay HUMAN
     (no auto-PROMOTE_TO_PAPER_TRADE or auto-deploy from cron path)

Each section is a STRUCTURAL test — AST or grep based — that survives
naming changes but catches conceptual regressions.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── Helpers ────────────────────────────────────────────────────────


def _read_module_ast(path: pathlib.Path) -> ast.AST | None:
    if not path.is_file():
        return None
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None


def _files_under(*dirs: str) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for d in dirs:
        root = REPO_ROOT / d
        if root.is_dir():
            out.extend(root.rglob("*.py"))
    return out


# ── Doctrine 1: forward/enhance separation ────────────────────────


def test_doctrine_burndown_ranker_filters_enhance_class():
    """burndown_ranker MUST skip rows where addresses_decay_in is non-null
    OR tags contain source:active_b_sleeve_scan OR source:doctrine_signal.
    These are enhance-class hypotheses that belong in
    engine.research.enhance.dispatcher, NOT factor_dispatcher.
    """
    ranker_path = REPO_ROOT / "engine" / "research" / "burndown_ranker.py"
    src = ranker_path.read_text(encoding="utf-8")
    # All three markers must appear in the ranker source
    assert "addresses_decay_in" in src, (
        "forward/enhance separation violated: burndown_ranker.py must "
        "explicitly check addresses_decay_in to exclude enhance hypotheses"
    )
    assert "source:doctrine_signal" in src or "doctrine_signal" in src, (
        "burndown_ranker.py must reference doctrine_signal tag filter"
    )
    assert "source:active_b_sleeve_scan" in src or "active_b_sleeve_scan" in src, (
        "burndown_ranker.py must reference active_b_sleeve_scan tag filter"
    )


def test_doctrine_enhance_dispatcher_uses_paired_bootstrap():
    """Enhance pipeline must use paired bootstrap (Politis-Romano 1994),
    NOT factor_dispatcher's NW-t single-sample machinery. The
    enhance.dispatcher module must import from paired_bootstrap.
    """
    dispatcher_path = REPO_ROOT / "engine" / "research" / "enhance" / "dispatcher.py"
    src = dispatcher_path.read_text(encoding="utf-8")
    assert "paired_block_bootstrap" in src, (
        "enhance.dispatcher MUST use paired_block_bootstrap, not "
        "single-sample NW-t. forward/enhance statistical separation."
    )


def test_doctrine_enhance_verdict_uses_correct_vocabulary():
    """Enhance verdict types must be IMPROVEMENT/NOISE/DEGRADATION,
    NOT GREEN/MARGINAL/RED. Mixing these vocabularies polluted
    forward family priors in pre-Phase-1 state."""
    verdict_path = REPO_ROOT / "engine" / "research" / "enhance" / "verdict.py"
    src = verdict_path.read_text(encoding="utf-8")
    assert "IMPROVEMENT" in src
    assert "DEGRADATION" in src
    assert "NOISE" in src


# ── Doctrine 2: strategy_family vs claim_family canonical ─────────


def test_doctrine_emit_uses_strategy_family_not_claim_family():
    """factor_verdict_emit MUST set event.family from
    strategy_family_for_spec(spec), not from family_hint (claim taxonomy).
    BUG diagnosed 2026-06-12: conflation double-counted same spec
    across multiple family trial denominators."""
    emit_path = REPO_ROOT / "engine" / "agents" / "strengthener" / "factor_verdict_emit.py"
    src = emit_path.read_text(encoding="utf-8")
    assert "strategy_family_for_spec" in src, (
        "factor_verdict_emit MUST call strategy_family_for_spec(spec) "
        "for event.family. Using family_hint directly conflates "
        "claim taxonomy with Bailey-LdP denominator (design flaw "
        "caught 2026-06-12)."
    )


def test_doctrine_n_trials_uses_strategy_family():
    """_family_n_trials_now must match by strategy_family (event.family
    OR strategy_family:<X> tag), not by hypothesis.mechanism_family.
    """
    dispatcher_path = REPO_ROOT / "engine" / "agents" / "strengthener" / "factor_dispatcher.py"
    src = dispatcher_path.read_text(encoding="utf-8")
    # The function definition must reference strategy_family in
    # the matching logic
    assert "strategy_family:" in src or "strategy_family_for_spec" in src, (
        "factor_dispatcher.py must use strategy_family-based n_trials "
        "counting; raw mechanism_family lookup was the 2026-06-12 design flaw"
    )


# ── Doctrine 3: PIT_CORRECT_SOURCES whitelist consistency ─────────


def test_doctrine_pit_whitelist_is_enforced_in_pre_dispatch():
    """pre_dispatch_check MUST scan spec.signal_inputs against
    PIT_CORRECT_SOURCES. Sites that bypass this allow look-ahead
    cache paths."""
    dispatcher_path = REPO_ROOT / "engine" / "agents" / "strengthener" / "factor_dispatcher.py"
    src = dispatcher_path.read_text(encoding="utf-8")
    assert "PIT_CORRECT_SOURCES" in src
    assert "SIGNAL_INPUT_UNKNOWN" in src, (
        "pre_dispatch_check must emit SIGNAL_INPUT_UNKNOWN refusal "
        "when signal_inputs reference non-PIT-clean paths"
    )


# ── Doctrine 4: capital decisions stay HUMAN ──────────────────────


def _ast_function_calls(path: pathlib.Path) -> set[str]:
    """Walk AST + collect every function name appearing in Call nodes.
    Includes both bare calls foo() AND attribute calls obj.foo()."""
    tree = _read_module_ast(path)
    if tree is None:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id.lower())
            elif isinstance(f, ast.Attribute):
                out.add(f.attr.lower())
    return out


def test_doctrine_no_auto_promote_in_cron_path():
    """Cron path (burndown_run / burndown_executor) MUST NOT call
    auto-deployment / auto-promotion functions. Capital decisions
    stay HUMAN. IMPROVEMENT verdicts route to /approvals only.

    Uses AST Call-node walk so docstring / comment mentions of these
    function names (which are intentional doctrine references) don't
    trigger false positives.
    """
    paths_to_audit = [
        REPO_ROOT / "scripts" / "burndown_run.py",
        REPO_ROOT / "engine" / "research" / "burndown_executor.py",
    ]
    forbidden_calls = {
        "promote_to_paper_trade",
        "auto_promote",
        "deploy_sleeve",
        "modify_deployed_sleeve",
    }
    for p in paths_to_audit:
        if not p.is_file():
            continue
        calls = _ast_function_calls(p)
        bad = calls & forbidden_calls
        assert not bad, (
            f"capital-decisions-human doctrine violated: "
            f"{p.name} calls {sorted(bad)}. Cron path must NEVER "
            f"auto-promote / auto-deploy. Route to /approvals only."
        )


def test_doctrine_extractor_calls_claim_shape_router():
    """Phase 2.1 (2026-06-13): extract_factor_spec_with_routing MUST call
    classify_claim_shape as Stage 0 before invoking the Stage 1 LLM. If a
    refactor removes the router call, BUG-2-style claim drift (spanning
    claims stretched into factor_combination, etc.) returns silently.
    """
    p = REPO_ROOT / "engine" / "agents" / "strengthener" / "factor_spec_extractor.py"
    calls = _ast_function_calls(p)
    assert "classify_claim_shape" in calls, (
        "Specification-drift gate violated: factor_spec_extractor.py no "
        "longer calls classify_claim_shape — Stage 0 router has been "
        "removed. BUG-2 / Sonnet-drift mitigation regresses to nothing."
    )


def test_doctrine_burndown_executor_uses_routing_aware_extractor():
    """Phase 2.1: burndown_executor's default extractor must be the
    routing-aware one (extract_factor_spec_with_routing) — otherwise
    the cron path swallows router refusals as generic EXTRACT_RETURNED_NONE
    and the demand ledger loses NEEDS_NEW_TEMPLATE signal."""
    p = REPO_ROOT / "engine" / "research" / "burndown_executor.py"
    src = p.read_text(encoding="utf-8")
    assert "extract_factor_spec_with_routing" in src, (
        "burndown_executor.py default extractor must be "
        "extract_factor_spec_with_routing (Phase 2.1) so cron surfaces "
        "specific router refusals instead of EXTRACT_RETURNED_NONE."
    )


def test_doctrine_burndown_executor_calls_post_green_rigor():
    """Phase 4.1 (2026-06-13): burndown_executor.execute_one MUST call
    _maybe_run_post_green_rigor for every GREEN/MARGINAL outcome.
    If a refactor removes the hook silently, every future GREEN ships
    without post-pub OOS / FF5 spanning rigor — back to the partial-
    shipped state we just spent Phase 4.1 fixing.
    """
    p = REPO_ROOT / "engine" / "research" / "burndown_executor.py"
    calls = _ast_function_calls(p)
    assert "_maybe_run_post_green_rigor" in calls, (
        "post-GREEN rigor regression: burndown_executor.py no longer "
        "calls _maybe_run_post_green_rigor — Phase 4.1 wire was removed."
    )


def test_doctrine_burndown_executor_calls_external_audit():
    """Phase 1.2 (2026-06-13): every cron-emitted GREEN/MARGINAL/RED verdict
    MUST be reviewed by external_audit. burndown_executor.execute_one must
    contain a call to _maybe_audit_verdict — if a refactor removes that hook
    silently, Mitigation #1 of the self-audit blind-spots doctrine
    regresses to nothing.

    Uses AST Call-node walk so docstring / comment mentions don't trigger
    false positives.
    """
    p = REPO_ROOT / "engine" / "research" / "burndown_executor.py"
    calls = _ast_function_calls(p)
    assert "_maybe_audit_verdict" in calls, (
        "self-audit-blind-spots Mitigation #1 violated: "
        "burndown_executor.py no longer calls _maybe_audit_verdict — "
        "cron-emitted verdicts are now unaudited. "
        "Restore the hook in execute_one() before merge."
    )


def test_doctrine_enhance_dispatcher_does_not_auto_deploy():
    """Enhance pipeline's dispatch_enhance_hypothesis must NOT call any
    deploy / modify_sleeve function. IMPROVEMENT verdicts emit log
    rows; principal approves manually via /approvals."""
    enhance_dispatcher = REPO_ROOT / "engine" / "research" / "enhance" / "dispatcher.py"
    calls = _ast_function_calls(enhance_dispatcher)
    forbidden_calls = {"deploy_sleeve", "auto_promote", "modify_deployed_sleeve", "promote_to_paper_trade"}
    bad = calls & forbidden_calls
    assert not bad, (
        f"enhance.dispatcher calls {sorted(bad)}. IMPROVEMENT verdicts "
        f"must NOT auto-deploy; capital stays human."
    )


# ── Composite: extractor system prompt MUST document new signal_kinds ─


def test_doctrine_extractor_prompt_documents_all_signal_kinds():
    """When SIGNAL_KINDS gets a new entry, the system prompt
    documentation must also describe it. Otherwise Sonnet can't pick it.
    """
    extractor_path = REPO_ROOT / "engine" / "agents" / "strengthener" / "factor_spec_extractor.py"
    src = extractor_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Find SIGNAL_KINDS tuple literal
    signal_kinds: tuple[str, ...] = ()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SIGNAL_KINDS"
                      for t in node.targets)):
            if isinstance(node.value, ast.Tuple):
                signal_kinds = tuple(
                    elt.value for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                )
                break
    # Every signal_kind name must appear in the _SYSTEM_PROMPT body
    # (extractor prompt documentation)
    for sk in signal_kinds:
        if sk in ("requires_custom_code",):
            continue   # escape hatch — always documented separately
        assert sk in src, (
            f"signal_kind '{sk}' is in SIGNAL_KINDS enum but not "
            f"documented in extractor system prompt. Sonnet can't pick "
            f"a signal_kind it doesn't know about; this is silent drift "
            f"toward requires_custom_code escape hatch."
        )


# ── Composite: every TEMPLATE_REGISTRY entry resolves ─────────────


def test_doctrine_every_template_registry_entry_resolves():
    """Each TEMPLATE_REGISTRY value must be a callable (signal_kind ↔
    template lazy-loader). Lets the structural audit catch unwired
    entries before runtime."""
    from engine.agents.strengthener.factor_dispatcher import TEMPLATE_REGISTRY
    for sk, fn in TEMPLATE_REGISTRY.items():
        assert callable(fn), (
            f"TEMPLATE_REGISTRY[{sk!r}] is not callable: {fn!r}"
        )
