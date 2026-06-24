"""Guardrail: FastAPI route registration order.

Bug class
---------
FastAPI/Starlette match routes by REGISTRATION ORDER on path. If a
dynamic catch-all route like @router.get("/{x}") is registered BEFORE
a static literal route like @router.post("/extract"), the dynamic one
swallows the literal path (with x="extract"). The POST request then
hits a GET-only handler and returns HTTP 405 Method Not Allowed —
NOT 404 (route exists) and NOT 422 (handler runs but validates) —
so the failure mode is confusing in the field.

Concrete incident (2026-06-05, commit f4d59c5d):
  /api/hypothesis_spec/extract returned 405 because:
    @router.get("/{hypothesis_id}")   was registered first
    @router.post("/extract")          was registered second
  Frontend "Extract spec via LLM" button broke silently in production.

The fix
-------
Static routes MUST be registered before dynamic catch-all routes at
the SAME path level. This test enumerates every FastAPI route decorator
in the repo, groups by router object, and asserts no static route is
shadowed by a previously-registered dynamic one.

Adding a new route? Put @router.<method>("/literal") above
@router.<method>("/{param}") in the same file.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DECORATOR_RE = re.compile(
    r'@(\w+)\.(get|post|put|delete|patch|head|options)\(\s*["\']([^"\']+)["\']'
)
_SKIP_DIRS = {".venv", "node_modules", "__pycache__", ".git",
              ".pytest_cache", "tests"}


def _enumerate_routes() -> list[tuple[Path, int, str, str, str]]:
    """Return (file, line_no, router_obj, METHOD, path) for every
    decorator in the repo, in encounter order."""
    out: list[tuple[Path, int, str, str, str]] = []
    for f in sorted(_REPO_ROOT.rglob("*.py")):
        if any(p in _SKIP_DIRS for p in f.parts):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if "fastapi" not in text.lower():
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            m = _DECORATOR_RE.search(line)
            if not m:
                continue
            obj, method, path = m.group(1), m.group(2), m.group(3)
            out.append((f.relative_to(_REPO_ROOT), line_no, obj,
                         method.upper(), path))
    return out


def _shadow_check(routes: list) -> list[str]:
    """For routes grouped by (file, router_obj), detect dynamic-first /
    static-later pairs that would cause the 405-class bug.

    Returns a list of human-readable issue strings (empty if clean).
    """
    issues: list[str] = []
    by_router: dict[tuple, list] = {}
    for row in routes:
        f, ln, obj, m, p = row
        by_router.setdefault((f, obj), []).append((ln, m, p))

    for (f, obj), rs in by_router.items():
        for i, (ln_a, m_a, p_a) in enumerate(rs):
            seg_a = p_a.strip("/").split("/")
            # Dynamic at first segment?
            if not (seg_a and seg_a[0].startswith("{") and seg_a[0].endswith("}")):
                continue
            for ln_b, m_b, p_b in rs[i + 1:]:
                seg_b = p_b.strip("/").split("/")
                if not seg_b or not seg_b[0] or seg_b[0].startswith("{"):
                    continue
                if len(seg_a) == 1 and len(seg_b) == 1:
                    issues.append(
                        f"{f}: L{ln_a} {m_a} {p_a} (dynamic) shadows "
                        f"L{ln_b} {m_b} {p_b} (static) — move static above")
                elif (len(seg_a) > 1 and len(seg_b) >= len(seg_a)
                      and all(a.startswith("{") or a == b
                              for a, b in zip(seg_a, seg_b))):
                    issues.append(
                        f"{f}: L{ln_a} {m_a} {p_a} (dynamic) shadows "
                        f"L{ln_b} {m_b} {p_b} (static) — move static above")
    return issues


def test_no_dynamic_shadows_static_route():
    """Fails if any FastAPI router has a dynamic /{x} route registered
    before a sibling static /literal route. See module docstring for
    the 2026-06-05 incident this prevents recurring."""
    routes = _enumerate_routes()
    assert routes, "expected to find at least some FastAPI route decorators"
    issues = _shadow_check(routes)
    if issues:
        msg = "Dynamic FastAPI routes shadow static ones:\n  - " + \
              "\n  - ".join(issues) + \
              "\n\nFix: move static @router.<method>(\"/literal\") " + \
              "decorators ABOVE dynamic @router.<method>(\"/{param}\") " + \
              "in the same file."
        raise AssertionError(msg)
