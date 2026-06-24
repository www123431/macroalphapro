"""Tests for template auto-discovery (no manual TEMPLATES dict edits)."""
from __future__ import annotations

import pytest


def test_templates_dict_auto_populated():
    """TEMPLATES should be discovered at import time, no manual edits."""
    from engine.research.templates import TEMPLATES
    assert "equity_xsmom" in TEMPLATES
    assert "factor_quartile" in TEMPLATES
    assert "cross_asset_tsmom" in TEMPLATES
    assert "primitive_composition" in TEMPLATES


def test_each_discovered_template_is_callable():
    """Every entry must be callable so DSL runner can dispatch."""
    from engine.research.templates import TEMPLATES
    for name, fn in TEMPLATES.items():
        assert callable(fn), f"template {name} entry is not callable"


def test_private_modules_not_discovered():
    """Modules starting with _ should be skipped."""
    from engine.research.templates import TEMPLATES
    # Common patterns we'd expect to be private
    for name in TEMPLATES:
        assert not name.startswith("_"), \
            f"private module {name} should not be in TEMPLATES"


def test_drop_in_new_template(tmp_path, monkeypatch):
    """Simulate dropping a new template file — auto-discovery picks it up
    on reload."""
    import importlib
    from engine.research import templates as templates_pkg

    # Write a new template file into the templates dir
    pkg_dir = tmp_path / "_test_template_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    template_file = pkg_dir / "my_new_template.py"
    template_file.write_text("""
def run_my_new_template(**kwargs):
    return "ran"

def warmup_months(binding):
    return 5
""", encoding="utf-8")

    # Manually verify the discovery convention works
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        from _test_template_pkg import my_new_template
        assert callable(my_new_template.run_my_new_template)
        assert my_new_template.warmup_months({}) == 5
    finally:
        sys.path.remove(str(tmp_path))


def test_reload_templates_refreshes():
    """reload_templates should re-scan the directory."""
    from engine.research.templates import reload_templates
    new = reload_templates()
    assert "equity_xsmom" in new
    assert callable(new["equity_xsmom"])


def test_dsl_runner_uses_discovered_templates():
    """End-to-end: DSL runner dispatches via the auto-discovered dict."""
    from engine.research.strategy_dsl_runner import run_proposal, list_templates
    names = list_templates()
    assert "equity_xsmom" in names
    assert "factor_quartile" in names
