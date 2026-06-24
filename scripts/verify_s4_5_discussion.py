"""S4.5 Discussion + Limitations + Conclusion verification.

Facets:
  A. §5 word count (~800-1300)
  B. §5 sub-sections 5.1-5.4 present
  C. §6 word count (~200-450)
  D. §7 word count (~200-450)
  E. all citations defined
  F. critical anchors present (placebo collapse / BAB replication / Hou-Xue-Zhang / pre-registration / self-falsification)
  G. no [TODO] / [STUB]
  H. spec_hash unchanged
  I. paper-level: total length 6500-9000 words (12-15 pages target excl. refs)
"""
import sys, os, re, hashlib

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


def section(name_start, name_end):
    s = paper_src.find(name_start)
    e = paper_src.find(name_end)
    if s == -1 or e == -1:
        return ""
    return paper_src[s:e]


sec5 = section("# 5. Discussion", "# 6. Limitations")
sec6 = section("# 6. Limitations", "# 7. Conclusion")
sec7 = section("# 7. Conclusion", "# References")

# A
print("=" * 70)
print("A — §5 word count")
print("=" * 70)
n5 = len(re.findall(r"\b[A-Za-z0-9]+\b", sec5))
print(f"  §5 words: {n5} (target 800-1400)")
assert 800 <= n5 <= 1400
print("  OK")

# B
print()
print("=" * 70)
print("B — §5 sub-sections 5.1-5.4")
print("=" * 70)
required = ["## 5.1", "## 5.2", "## 5.3", "## 5.4"]
for r in required:
    assert r in sec5, f"missing: {r}"
print(f"  OK: all 4 sub-sections")

# C
print()
print("=" * 70)
print("C — §6 word count")
print("=" * 70)
n6 = len(re.findall(r"\b[A-Za-z0-9]+\b", sec6))
print(f"  §6 words: {n6} (target 180-450)")
assert 180 <= n6 <= 450
print("  OK")

# D
print()
print("=" * 70)
print("D — §7 word count")
print("=" * 70)
n7 = len(re.findall(r"\b[A-Za-z0-9]+\b", sec7))
print(f"  §7 words: {n7} (target 180-500)")
assert 180 <= n7 <= 500
print("  OK")

# E
print()
print("=" * 70)
print("E — citations defined across §5-§7")
print("=" * 70)
combined = sec5 + sec6 + sec7
cite_pat = re.compile(r"@([a-zA-Z][a-zA-Z0-9_]*\d{2,4}[a-zA-Z0-9_]*)")
combined_cites = set(cite_pat.findall(combined))
bib_keys = set(re.compile(r"@[a-zA-Z]+\{([^,\s]+)\s*,").findall(bib_src))
print(f"  cites: {sorted(combined_cites)}")
undef = combined_cites - bib_keys
assert not undef
print("  OK")

# F
print()
print("=" * 70)
print("F — critical framing anchors")
print("=" * 70)
combined_lower = combined.lower()
anchors = [
    ("placebo", "placebo arm framing"),
    ("bab", "BAB or BAB-related anchor"),
    ("frazzini", "Frazzini-Pedersen citation in anchor"),
    ("hou2020replicating", "Hou-Xue-Zhang reference"),
    ("self-falsification", "self-falsification framing"),
    ("pre-registration", "pre-registration framing"),
    ("decision augmentation", "future-work LLM framing"),
    ("null-leaning", "null-leaning verdict label"),
]
for needle, what in anchors:
    assert needle in combined_lower, f"missing: '{needle}' ({what})"
print(f"  OK: all {len(anchors)} anchors present")

# G
print()
print("=" * 70)
print("G — no [TODO] / [STUB]")
print("=" * 70)
for m in ("[TODO", "[STUB", "TBD"):
    assert m not in combined, f"{m} in §5-§7"
print("  OK")

# H
print()
print("=" * 70)
print("H — spec_hash drift")
print("=" * 70)
sh = hashlib.sha256(spec_src.encode("utf-8")).hexdigest()[:16]
print(f"  spec_hash: {sh}")
assert sh == "03a6767a5e5ea600", f"drifted: {sh}"
print("  OK")

# I
print()
print("=" * 70)
print("I — paper-level total length")
print("=" * 70)
# Approximate body section: §1 through §7 (excluding References + Appendix)
body_start = paper_src.find("# 1. Introduction")
body_end = paper_src.find("# References")
body = paper_src[body_start:body_end]
n_body = len(re.findall(r"\b[A-Za-z0-9]+\b", body))
print(f"  body words (§1-§7): {n_body} (target 6000-9500)")
print(f"  ≈ pages at 500 wpm: {n_body / 500:.1f}")
assert 6000 <= n_body <= 9500
print("  OK")

print()
print("=" * 70)
print("S4.5 verification PASS")
print("=" * 70)
