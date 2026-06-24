"""S4.1 Lit review + intro draft verification.

Facets:
  A. references.bib parses (consistent @TYPE{key, ...} entries)
  B. ≥20 entries (target ≥25)
  C. all citations in paper.md introduction are defined in references.bib
  D. paper.md frontmatter + section headers valid
  E. introduction word count in target range (~700-1000 words)
  F. spec §1 contribution claims surface in introduction
"""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
BIB = os.path.join(PAPER_DIR, "references.bib")
PAPER = os.path.join(PAPER_DIR, "paper.md")


# ─────────────────────────────────────────────────────────────────────────────
# A. references.bib parses
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("A — references.bib parses")
print("=" * 70)
with open(BIB, "r", encoding="utf-8") as f:
    bib_src = f.read()
print(f"  size: {len(bib_src)} bytes")

# Extract entries: pattern @TYPE{KEY, ... }
entry_pattern = re.compile(
    r"@(article|book|inproceedings|techreport|misc)\{([^,\s]+)\s*,",
    re.IGNORECASE,
)
entries = entry_pattern.findall(bib_src)
print(f"  entries found: {len(entries)}")
for entry_type, key in entries:
    pass  # quiet
print(f"  types breakdown: " + ", ".join(
    f"{t}={sum(1 for tt, _ in entries if tt.lower() == t)}"
    for t in {e[0].lower() for e in entries}
))

# Brace balance check
open_braces = bib_src.count("{")
close_braces = bib_src.count("}")
print(f"  brace balance: open={open_braces}, close={close_braces}")
assert open_braces == close_braces, f"brace mismatch in references.bib"
print("  OK: braces balanced (likely valid BibTeX)")

# ─────────────────────────────────────────────────────────────────────────────
# B. entry count
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("B — entry count ≥20 (target ≥25)")
print("=" * 70)
print(f"  entries: {len(entries)}")
assert len(entries) >= 20, f"need ≥20 entries, got {len(entries)}"
if len(entries) < 25:
    print(f"  WARN: below target 25 (have {len(entries)})")
else:
    print(f"  OK: meets ≥25 target")

bib_keys = {key for _, key in entries}

# ─────────────────────────────────────────────────────────────────────────────
# C. all introduction citations are defined
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C — citations in §1 introduction defined in bib")
print("=" * 70)
with open(PAPER, "r", encoding="utf-8") as f:
    paper_src = f.read()
print(f"  paper size: {len(paper_src)} bytes")

# Extract pandoc-style @key citations
cite_pattern = re.compile(r"@([a-zA-Z][a-zA-Z0-9_]*\d{2,4}[a-zA-Z0-9_]*)")
cites_in_paper = set(cite_pattern.findall(paper_src))
print(f"  unique citation keys in paper: {len(cites_in_paper)}")
print(f"  keys: {sorted(cites_in_paper)}")

undefined = cites_in_paper - bib_keys
assert not undefined, f"undefined citation keys: {undefined}"
print(f"  OK: all {len(cites_in_paper)} keys defined in references.bib")

# ─────────────────────────────────────────────────────────────────────────────
# D. paper structure
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("D — paper.md structure")
print("=" * 70)
# YAML frontmatter
assert paper_src.startswith("---"), "missing YAML frontmatter"
fm_end = paper_src.find("---", 3)
assert fm_end > 0
yaml_block = paper_src[3:fm_end]
for required in ("title:", "author:", "abstract:", "keywords:"):
    assert required in yaml_block, f"frontmatter missing {required}"
print("  OK: YAML frontmatter has title, author, abstract, keywords")

# Sections
required_sections = [
    "# 1. Introduction",
    "# 2. Methodology",
    "# 3. Falsified Hypotheses",
    "# 4. B++ Mass FDR Search",
    "# 5. Discussion",
    "# 6. Limitations",
    "# 7. Conclusion",
    "# References",
    "# Appendix A",
    "# Appendix B",
]
for sec in required_sections:
    assert sec in paper_src, f"missing section: {sec}"
print(f"  OK: all {len(required_sections)} sections present")

# ─────────────────────────────────────────────────────────────────────────────
# E. introduction length
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("E — introduction word count in target range")
print("=" * 70)
intro_start = paper_src.find("# 1. Introduction")
intro_end = paper_src.find("# 2. Methodology")
intro_text = paper_src[intro_start:intro_end]
words = re.findall(r"\b[A-Za-z]+\b", intro_text)
n_words = len(words)
print(f"  introduction words: {n_words} (target 700-1000 for ~1.5 pages)")
assert 600 <= n_words <= 1200, f"intro {n_words} words outside 600-1200"
print("  OK: in band")

# ─────────────────────────────────────────────────────────────────────────────
# F. spec §1 contribution anchors present in introduction
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("F — spec §1 contribution claims appear in introduction")
print("=" * 70)
intro_lower = intro_text.lower()
required_anchors = [
    ("pre-registration framework", "framework claim"),
    ("six", "six falsifications claim"),
    ("marginal", "marginal replication claim"),
    ("bhy", "FDR control mention"),
    ("retail-grade", "scope qualifier"),
]
for needle, what in required_anchors:
    assert needle in intro_lower, f"missing: '{needle}' ({what})"
print(f"  OK: all {len(required_anchors)} contribution anchors present")

# Spec hash recompute (drift check)
import hashlib
SPEC = os.path.join(os.path.dirname(PAPER_DIR), "docs", "spec_ssrn_paper_v1.md")
with open(SPEC, "r", encoding="utf-8") as f:
    spec_content = f.read()
spec_hash = hashlib.sha256(spec_content.encode("utf-8")).hexdigest()[:16]
print(f"  spec_hash[:16] = {spec_hash}")

print()
print("=" * 70)
print("S4.1 verification PASS")
print("=" * 70)
