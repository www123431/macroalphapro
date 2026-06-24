"""S4.3 Falsified Hypotheses section verification.

Facets:
  A. §3 word count in target range (~1100-1700, ≈3 pages)
  B. all 6 sub-sections present + §3.7 cross-cutting
  C. summary Table 1 with 6 rows (one per hypothesis)
  D. all citations defined
  E. each sub-section names its verdict tier (REJECT/FAIL/SOFT REJECT/HARD REJECT)
  F. all 6 hypotheses headline statistics present (B-C numbers, NW t, etc)
  G. no [TODO] / [STUB] in §3
  H. spec_hash unchanged
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

sec3_start = paper_src.find("# 3. Falsified Hypotheses")
sec4_start = paper_src.find("# 4. B++ Mass FDR")
sec3_text = paper_src[sec3_start:sec4_start]


# A. word count
print("=" * 70)
print("A — §3 word count")
print("=" * 70)
words = re.findall(r"\b[A-Za-z0-9]+\b", sec3_text)
n_words = len(words)
print(f"  §3 words: {n_words} (target 1100-1800)")
assert 1100 <= n_words <= 1800, f"§3 {n_words} outside band"
print("  OK")

# B. sub-sections
print()
print("=" * 70)
print("B — sub-sections present (3.1-3.7)")
print("=" * 70)
required = [
    "## 3.1 D1",
    "## 3.2 D1.1",
    "## 3.3 Phase 0",
    "## 3.4 FactorMAD",
    "## 3.5 EFA",
    "## 3.6 S1 multi-window",
    "## 3.7 Cross-cutting",
]
for r in required:
    assert r in sec3_text, f"missing: {r}"
print(f"  OK: all {len(required)} sub-sections")

# C. summary table
print()
print("=" * 70)
print("C — summary Table 1 (6 hypothesis rows)")
print("=" * 70)
table_lines = [l for l in sec3_text.split("\n")
               if l.strip().startswith("|") and not l.strip().startswith("|---")]
data_rows = [l for l in table_lines
             if "Hypothesis" not in l and l.count("|") >= 6]
print(f"  table data rows: {len(data_rows)}")
assert len(data_rows) >= 6, f"need ≥6 rows, got {len(data_rows)}"
print("  OK: ≥6 rows")

# D. citations
print()
print("=" * 70)
print("D — §3 citations defined in bib")
print("=" * 70)
cite_pat = re.compile(r"@([a-zA-Z][a-zA-Z0-9_]*\d{2,4}[a-zA-Z0-9_]*)")
cites = set(cite_pat.findall(sec3_text))
bib_keys = set(re.compile(r"@[a-zA-Z]+\{([^,\s]+)\s*,").findall(bib_src))
print(f"  §3 cites: {sorted(cites)}")
undef = cites - bib_keys
assert not undef, f"undefined: {undef}"
print("  OK")

# E. verdict tiers
print()
print("=" * 70)
print("E — verdict tiers per sub-section")
print("=" * 70)
required_verdicts = {
    "## 3.1": ["soft reject"],
    "## 3.2": ["hard reject"],
    "## 3.3": ["reject"],
    "## 3.4": ["reject"],
    "## 3.5": ["fail"],
    "## 3.6": ["fail"],
}
for sec, possibles in required_verdicts.items():
    sec_idx = sec3_text.find(sec)
    next_sec_idx = sec3_text.find("## 3.", sec_idx + 5)
    if next_sec_idx == -1:
        next_sec_idx = len(sec3_text)
    sub = sec3_text[sec_idx:next_sec_idx].lower()
    assert any(p in sub for p in possibles), \
        f"{sec} missing verdict word from {possibles}"
print(f"  OK: all {len(required_verdicts)} sub-sections name their verdict")

# F. headline statistics
print()
print("=" * 70)
print("F — headline OOS statistics present")
print("=" * 70)
required_stats = [
    "+0.896",  # D1 NW t
    "+0.005",  # D1.1 B-C
    "0 / 24",  # FactorMAD promotion rate
    "-0.174",  # EFA Sharpe
    "-0.486",  # S1 bootstrap CI low
    "+0.454",  # S1 bootstrap CI high
    "192",     # D1.1 OOS months
    "GPR",     # Phase 0 macro proxy
]
missing = [s for s in required_stats if s not in sec3_text]
assert not missing, f"missing stats: {missing}"
print(f"  OK: all {len(required_stats)} headline statistics present")

# G. no TODO/STUB
print()
print("=" * 70)
print("G — no unresolved markers")
print("=" * 70)
for m in ("[TODO", "[STUB", "TBD"):
    assert m not in sec3_text, f"{m} in §3"
print("  OK")

# H. spec hash
print()
print("=" * 70)
print("H — spec_hash drift check")
print("=" * 70)
sh = hashlib.sha256(spec_src.encode("utf-8")).hexdigest()[:16]
print(f"  spec_hash: {sh}")
assert sh == "03a6767a5e5ea600", f"spec drifted to {sh}"
print("  OK: spec untouched")

print()
print("=" * 70)
print("S4.3 verification PASS")
print("=" * 70)
