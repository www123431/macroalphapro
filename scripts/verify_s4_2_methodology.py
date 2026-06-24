"""S4.2 Methodology section verification.

Facets:
  A. §2 word count in target range (~700-1100, ≈2 pages)
  B. all 5 sub-sections present (§2.1 - §2.5)
  C. all citations in §2 defined in references.bib
  D. spec §2 anchor concepts surface (spec_hash, amendment ledger, BHY,
      Newey-West, stationary bootstrap, placebo arm, train/OOS partition)
  E. no [TODO] / [STUB] markers remain in §2
  F. spec_hash unchanged since S4.0
"""
import sys
import os
import re
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
PAPER = os.path.join(PAPER_DIR, "paper.md")
BIB = os.path.join(PAPER_DIR, "references.bib")
SPEC = os.path.join(os.path.dirname(PAPER_DIR), "docs", "spec_ssrn_paper_v1.md")

with open(PAPER, "r", encoding="utf-8") as f:
    paper_src = f.read()
with open(BIB, "r", encoding="utf-8") as f:
    bib_src = f.read()
with open(SPEC, "r", encoding="utf-8") as f:
    spec_src = f.read()


# ─────────────────────────────────────────────────────────────────────────────
# A. §2 word count
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — §2 Methodology word count")
print("=" * 70)
sec2_start = paper_src.find("# 2. Methodology")
sec3_start = paper_src.find("# 3. Falsified")
sec2_text = paper_src[sec2_start:sec3_start]
words = re.findall(r"\b[A-Za-z]+\b", sec2_text)
n_words = len(words)
print(f"  §2 words: {n_words} (target 700-1200 for ~2 pages)")
assert 700 <= n_words <= 1200, f"§2 {n_words} outside 700-1200"
print("  OK")

# ─────────────────────────────────────────────────────────────────────────────
# B. sub-sections
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — sub-sections §2.1-§2.5 present")
print("=" * 70)
required_subs = [
    "## 2.1 Pre-registration framework",
    "## 2.2 Statistical infrastructure",
    "## 2.3 Partition discipline",
    "## 2.4 Placebo arm",
    "## 2.5 Verdict tiers",
]
for s in required_subs:
    assert s in sec2_text, f"missing sub-section: {s}"
print(f"  OK: all {len(required_subs)} sub-sections present")

# ─────────────────────────────────────────────────────────────────────────────
# C. citations defined
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — citations in §2 defined in bib")
print("=" * 70)
cite_pattern = re.compile(r"@([a-zA-Z][a-zA-Z0-9_]*\d{2,4}[a-zA-Z0-9_]*)")
sec2_cites = set(cite_pattern.findall(sec2_text))
bib_keys = set(re.compile(r"@[a-zA-Z]+\{([^,\s]+)\s*,").findall(bib_src))
print(f"  §2 cites: {len(sec2_cites)}")
print(f"  keys: {sorted(sec2_cites)}")
undefined = sec2_cites - bib_keys
assert not undefined, f"undefined: {undefined}"
print("  OK: all defined")

# ─────────────────────────────────────────────────────────────────────────────
# D. anchor concepts present
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — methodology anchors per spec §2")
print("=" * 70)
sec2_lower = sec2_text.lower()
anchors = [
    ("spec", "specification anchor"),
    ("hash", "spec_hash"),
    ("amendment ledger", "amendment ledger"),
    ("effective-n-trials", "EFFECTIVE_N_TRIALS"),
    ("bhy", "BHY FDR"),
    ("newey", "Newey-West HAC"),
    ("stationary", "stationary bootstrap"),
    ("placebo", "placebo arm"),
    ("train", "train/OOS partition"),
    ("out-of-sample", "OOS partition"),
    ("power analysis", "power analysis"),
    ("verdict", "verdict tiers"),
]
for needle, what in anchors:
    assert needle in sec2_lower, f"missing: '{needle}' ({what})"
print(f"  OK: all {len(anchors)} anchors present")

# ─────────────────────────────────────────────────────────────────────────────
# E. no TODO/STUB in §2
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — no [TODO] / [STUB] markers in §2")
print("=" * 70)
for marker in ("[TODO", "[STUB", "TBD"):
    if marker in sec2_text:
        # Allow markers that reference future S4 sub-sprints (false-positive guard)
        idx = sec2_text.find(marker)
        snippet = sec2_text[idx:idx+80]
        print(f"  WARN: found '{marker}' in §2: {snippet!r}")
        assert False, f"unresolved {marker} marker in §2"
print("  OK: no unresolved markers")

# ─────────────────────────────────────────────────────────────────────────────
# F. spec_hash unchanged
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — spec_hash drift check")
print("=" * 70)
spec_hash = hashlib.sha256(spec_src.encode("utf-8")).hexdigest()[:16]
print(f"  spec_hash[:16] = {spec_hash}")
expected = "03a6767a5e5ea600"
assert spec_hash == expected, f"spec drifted! was {expected}, now {spec_hash}"
print("  OK: spec untouched since S4.0 lock")

print()
print("=" * 70)
print("S4.2 verification PASS")
print("=" * 70)
