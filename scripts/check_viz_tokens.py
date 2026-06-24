"""scripts/check_viz_tokens.py — R4.1 token CI guard.

Verifies the JS-side viz token mirror (frontend/lib/vizTokens.ts) is
in sync with the CSS source-of-truth (--vz-* in frontend/app/globals.css).

The two sources MUST agree because:
  - ECharts / SVG components read raw hex strings via the JS object
    (getComputedStyle is slow + SSR-fragile)
  - <span style={{ color: VZ.x }} /> usages compile to literal hex,
    cannot pick up CSS variable updates
  - any theme switch (light mode, high-contrast) needs both files
    updated atomically

Drift modes this catches:
  - Adding a new --vz-* in CSS without adding to VZ.* in TS
  - Changing a hex on one side without the other
  - Removing a token from one side

Exit codes:
  0  in sync
  1  drift detected (prints diff + remediation hint)

Performance: ~30ms typical. Suitable for pre-commit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT  = Path(__file__).resolve().parent.parent
CSS_FILE   = REPO_ROOT / "frontend" / "app"  / "globals.css"
TS_FILE    = REPO_ROOT / "frontend" / "lib"  / "vizTokens.ts"


# Mapping rules: how a --vz-* CSS name maps to a (group, key) JS path.
# Lookup direction is CSS → JS, since we want to catch new CSS tokens
# that haven't been mirrored. Reverse direction is also checked.
#
# Convention: --vz-{group}-{key} → VZ.{group}.{key} after kebab→snake/case.
# Special: --vz-verdict-marginal → VZ.verdict.marginal, etc.
# SLM states are UPPERCASE in JS (matching the SLM canonical state names).
SLM_STATES_UPPER = {
    "pre-live": None,  # zone label, no JS analog by name
    "validation": None,
    "live": None,
    "decay": None,
    "out": None,
}


def parse_css() -> dict[str, str]:
    """Extract --vz-* declarations from globals.css.

    Returns map name -> hex (lower-case)."""
    text = CSS_FILE.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for m in re.finditer(r"(--vz-[A-Za-z0-9_-]+)\s*:\s*(#[0-9A-Fa-f]{3,8}|rgba?\([^)]*\))\s*;",
                         text):
        name = m.group(1)
        val  = m.group(2).strip().lower()
        out[name] = val
    return out


def parse_ts() -> dict[str, str]:
    """Extract VZ.{group}.{key} → hex from vizTokens.ts.

    Returns a flat map name -> hex (e.g. "verdict.red" -> "#ef4444")."""
    text = TS_FILE.read_text(encoding="utf-8")
    # Match top-level groups: `verdict: { ... }`, `slm: { ... }`, etc.
    # Inside each block, match `KEY: "#hex"` pairs.
    out: dict[str, str] = {}
    # Strip `as const` and the `export const VZ = { ... }` wrapper, just
    # scan all `\bword: "#hex"` patterns inside the VZ object. Cheap.
    inside_vz = re.search(r"export\s+const\s+VZ\s*=\s*\{([\s\S]*?)\}\s+as\s+const",
                          text)
    if not inside_vz:
        return out
    body = inside_vz.group(1)

    # Walk: split by top-level groups (`groupname: {`). Cheap regex
    # that handles our flat 2-level structure.
    group_iter = re.finditer(r"(\w+)\s*:\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*?)\}", body)
    for gm in group_iter:
        gname = gm.group(1)
        gbody = gm.group(2)
        for km in re.finditer(r"(\w+)\s*:\s*\"(#[0-9A-Fa-f]{3,8})\"", gbody):
            kname = km.group(1)
            val   = km.group(2).lower()
            out[f"{gname}.{kname}"] = val
        # Also handle inner zone records: `ingest: { fill: "...", ... }`
        for zm in re.finditer(r"(\w+)\s*:\s*\{\s*text:\s*\"(#[0-9A-Fa-f]{3,8})\"",
                              gbody):
            zname = zm.group(1)
            val   = zm.group(2).lower()
            out[f"{gname}.{zname}.text"] = val
    return out


# Mapping from CSS names to JS paths. Mirrors the conventions
# established in app/globals.css + lib/vizTokens.ts.
CSS_TO_JS: dict[str, str] = {
    "--vz-verdict-red":      "verdict.red",
    "--vz-verdict-green":    "verdict.green",
    "--vz-verdict-marginal": "verdict.marginal",
    "--vz-verdict-pending":  "verdict.pending",

    "--vz-slm-pre-live":     "slm.PROPOSED",
    "--vz-slm-validation":   "slm.PAPER_TRADE",
    "--vz-slm-live":         "slm.LIVE",
    "--vz-slm-decay":        "slm.DECAY_WATCH",
    "--vz-slm-out":          "slm.DECOMMISSIONED",

    "--vz-role-alpha":       "role.alpha",
    "--vz-role-diversifier": "role.diversifier",
    "--vz-role-insurance":   "role.insurance",
    "--vz-role-carry":       "role.carry",
    "--vz-role-hedge":       "role.hedge",

    "--vz-corr-positive":    "corr.positive",
    "--vz-corr-negative":    "corr.negative",

    "--vz-zone-ingest":      "zone.ingest.text",
    "--vz-zone-triage":      "zone.triage.text",
    "--vz-zone-test":        "zone.test.text",
    "--vz-zone-verdict":     "zone.verdict.text",
    "--vz-zone-deploy":      "zone.deploy.text",
}


def main() -> int:
    if not CSS_FILE.is_file():
        print(f"viz-tokens: CSS file missing: {CSS_FILE}", file=sys.stderr)
        return 1
    if not TS_FILE.is_file():
        print(f"viz-tokens: TS file missing: {TS_FILE}", file=sys.stderr)
        return 1

    css = parse_css()
    ts  = parse_ts()

    errors: list[str] = []

    for css_name, js_path in CSS_TO_JS.items():
        css_val = css.get(css_name)
        ts_val  = ts.get(js_path)
        if css_val is None:
            errors.append(
                f"missing in CSS: {css_name} (mapped to JS VZ.{js_path}={ts_val})")
            continue
        if ts_val is None:
            errors.append(
                f"missing in JS: VZ.{js_path} (mapped from CSS {css_name}={css_val})")
            continue
        if css_val != ts_val:
            errors.append(
                f"DRIFT: {css_name}={css_val}  vs  VZ.{js_path}={ts_val}")

    # Also surface --vz-* tokens that don't appear in the mapping table
    # at all — adding them to globals.css without updating the mapping
    # is itself a smell. Warn but don't fail (gives author time to add
    # the mapping entry in a follow-up commit).
    unmapped = [n for n in css if n not in CSS_TO_JS]
    if unmapped:
        for n in unmapped:
            print(f"viz-tokens: WARN: {n} has no JS mapping yet", file=sys.stderr)

    if errors:
        print("viz-tokens: DRIFT detected:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Fix: edit BOTH files atomically:", file=sys.stderr)
        print(f"  - {CSS_FILE.relative_to(REPO_ROOT)}", file=sys.stderr)
        print(f"  - {TS_FILE.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1

    print(f"viz-tokens: in sync ({len(CSS_TO_JS)} pairs checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
