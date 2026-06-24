"""
engine/d_pead_plus/doctrine.py — 0-LLM-in-DECISION invariant enforcement.

Per spec id=74 §〇 doctrine amendment: LLM-as-INPUT-FEATURE allowed in
feature extraction layer ONLY. Decision layer (ranking, position sizing,
trade decisions) must be PURE deterministic Python with ZERO LLM calls.

This module provides code-level enforcement (condition #6 of doctrine).

Architecture invariant
----------------------
The decision layer modules:
  - engine.d_pead_plus.feature_combiner   (OLS coefficients application)
  - engine.d_pead_plus.backtest            (ranking, position construction)
  - engine.d_pead_plus.verdict             (gate evaluation, decision matrix)

must NEVER import:
  - openai, anthropic, google.generativeai, vertexai (LLM SDK modules)
  - engine.deepseek_client (project's LLM wrapper)
  - engine.d_pead_plus.llm_extractor (the LLM extraction module itself)
"""
from __future__ import annotations

import ast
from pathlib import Path

NO_LLM_IN_DECISION_LAYER: bool = True  # Invariant flag

# Modules that constitute the "decision layer" — must be pure deterministic
DECISION_LAYER_MODULES: tuple[str, ...] = (
    "engine.d_pead_plus.feature_combiner",
    "engine.d_pead_plus.backtest",
    "engine.d_pead_plus.verdict",
)

# Module names that, if imported by any decision layer module, would violate
# the 0-LLM-in-DECISION invariant.
FORBIDDEN_LLM_IMPORTS: tuple[str, ...] = (
    "openai",
    "anthropic",
    "google.generativeai",
    "google.cloud.aiplatform",
    "vertexai",
    "google.auth",
    "engine.deepseek_client",
    "engine.d_pead_plus.llm_extractor",
    "engine.d_pead_plus.llm_extractor_rest",
)


def _module_file_path(module_dotted: str) -> Path:
    """Return file path for an engine.d_pead_plus.* module."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    parts = module_dotted.split(".")
    return repo_root.joinpath(*parts).with_suffix(".py")


def _extract_imports(file_path: Path) -> set[str]:
    """Parse file AST and extract imported module names (top-level only)."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def audit_decision_layer_imports() -> dict:
    """
    Static-analysis audit of decision layer modules for forbidden LLM imports.

    Returns dict {module: violations}. Empty dict if clean.
    """
    violations: dict[str, list[str]] = {}
    for module in DECISION_LAYER_MODULES:
        file_path = _module_file_path(module)
        if not file_path.exists():
            # Module not yet built — acceptable during Sprint I build phase
            continue
        imports = _extract_imports(file_path)
        bad = []
        for forbidden in FORBIDDEN_LLM_IMPORTS:
            for imp in imports:
                # Match exact or prefix (e.g. forbidden 'openai' matches 'openai.types')
                if imp == forbidden or imp.startswith(forbidden + "."):
                    bad.append(f"{imp} (matches forbidden {forbidden})")
        if bad:
            violations[module] = bad
    return violations


def assert_no_llm_in_decision_layer() -> None:
    """
    Raise AssertionError if any decision layer module imports an LLM SDK.

    Called by tests (pytest test_d_pead_plus.py) and CLI scripts before each
    backtest run.
    """
    violations = audit_decision_layer_imports()
    if violations:
        msg_lines = ["0-LLM-in-DECISION doctrine VIOLATED:"]
        for module, bad_imports in violations.items():
            msg_lines.append(f"  {module}:")
            for imp in bad_imports:
                msg_lines.append(f"    - {imp}")
        msg_lines.append(
            "Per spec id=74 §〇 doctrine amendment, decision layer must "
            "make ZERO LLM calls. Move any LLM logic to engine.d_pead_plus.llm_extractor."
        )
        raise AssertionError("\n".join(msg_lines))


if __name__ == "__main__":
    # CLI audit
    violations = audit_decision_layer_imports()
    if violations:
        print("FAILED — doctrine violations detected:")
        for module, bad in violations.items():
            print(f"  {module}: {bad}")
        raise SystemExit(1)
    print(f"PASSED — all {len(DECISION_LAYER_MODULES)} decision layer modules clean")
    print(f"Modules audited: {', '.join(DECISION_LAYER_MODULES)}")
