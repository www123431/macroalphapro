"""Verify plotly theme unification: register OK + AppTest 4 plotly-heavy pages."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# A. Template registration + apply
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — register + apply mac_alpha_pro templates")
print("=" * 70)

import plotly.io as pio
from ui import _plotly_theme

# Before apply
default_before = pio.templates.default

_plotly_theme.register()
assert "mac_alpha_pro_dark" in pio.templates
assert "mac_alpha_pro_light" in pio.templates
print("  registered: mac_alpha_pro_dark + mac_alpha_pro_light")

# Inspect template structure (smoke that essential layout fields set)
tmpl_dark = pio.templates["mac_alpha_pro_dark"]
assert tmpl_dark.layout.paper_bgcolor is not None
assert tmpl_dark.layout.plot_bgcolor is not None
assert tmpl_dark.layout.colorway is not None and len(tmpl_dark.layout.colorway) >= 5
assert tmpl_dark.layout.font.family is not None
print(f"  dark template layout: paper={tmpl_dark.layout.paper_bgcolor}, "
      f"plot={tmpl_dark.layout.plot_bgcolor}, "
      f"colorway[:3]={list(tmpl_dark.layout.colorway[:3])}")

# Apply (requires Streamlit session_state)
import streamlit as st
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True
applied = _plotly_theme.apply()
print(f"  applied template: {applied}")
assert applied == "mac_alpha_pro_dark"
assert pio.templates.default == "mac_alpha_pro_dark"


# ─────────────────────────────────────────────────────────────────────────────
# B. theme.plotly_template_name + PLOTLY_TEMPLATE constant
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — theme.plotly_template_name + PLOTLY_TEMPLATE")
print("=" * 70)
from ui import theme
assert hasattr(theme, "PLOTLY_TEMPLATE")
assert hasattr(theme, "plotly_template_name")
assert theme.plotly_template_name() == "mac_alpha_pro_dark"
print(f"  theme.PLOTLY_TEMPLATE = {theme.PLOTLY_TEMPLATE!r}")
print(f"  theme.plotly_template_name() = {theme.plotly_template_name()!r}")


# ─────────────────────────────────────────────────────────────────────────────
# C. AppTest 4 plotly-heavy pages, exception-free
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — AppTest 4 plotly-heavy pages")
print("=" * 70)

from streamlit.testing.v1 import AppTest

PAGES_TO_TEST = [
    "live_dashboard.py",
    "performance_report.py",
    "agent_observability.py",
    "command_center.py",
]

PROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "pages")

results = {}
for page in PAGES_TO_TEST:
    p = os.path.join(PROOT, page)
    at = AppTest.from_file(p, default_timeout=180)
    at.run()
    n_exc = len(at.exception)
    excs = [str(e.value)[:200] for e in at.exception[:2]]
    results[page] = {"exceptions": n_exc, "excs": excs}
    status = "OK" if n_exc == 0 else "FAIL"
    print(f"  {page}: {status} (exceptions={n_exc})")
    for e in excs:
        print(f"    {e}")

assert all(r["exceptions"] == 0 for r in results.values()), \
    f"Page exceptions: {results}"
print("  OK: 4 pages render with new theme, 0 exceptions")


# ─────────────────────────────────────────────────────────────────────────────
# D. Light-mode toggle works
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — Light-mode toggle picks light template")
print("=" * 70)
st.session_state["dark_mode"] = False
applied = _plotly_theme.apply()
print(f"  applied: {applied}")
assert applied == "mac_alpha_pro_light"
assert pio.templates.default == "mac_alpha_pro_light"

# Restore dark
st.session_state["dark_mode"] = True
_plotly_theme.apply()


# ─────────────────────────────────────────────────────────────────────────────
# E. Palette accessors
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — palette accessors")
print("=" * 70)
qual = _plotly_theme.qualitative_palette()
div = _plotly_theme.diverging_palette()
print(f"  qualitative_palette: {len(qual)} colors")
print(f"  diverging_palette: {len(div)} stops")
assert len(qual) >= 5
assert len(div) >= 5

print()
print("=" * 70)
print("Plotly theme unification verification PASS (5 facets)")
print("=" * 70)
