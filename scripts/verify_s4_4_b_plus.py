"""S4.4 B++ Mass FDR Search section verification.

Facets:
  A. §4 word count in target range (~1200-1700)
  B. all 7 sub-sections present (4.1 - 4.7)
  C. Tables 2 + 3 present
  D. citations defined
  E. headline statistics present (Sharpe / NW t / β / α / R²)
  F. verdict tier MARGINAL named
  G. no [TODO] / [STUB]
  H. spec_hash unchanged
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

s4 = paper_src.find("# 4. B++ Mass FDR")
s5 = paper_src.find("# 5. Discussion")
sec4 = paper_src[s4:s5]


# A
print("=" * 70)
print("A — §4 word count")
print("=" * 70)
n_words = len(re.findall(r"\b[A-Za-z0-9]+\b", sec4))
print(f"  §4 words: {n_words} (target 1100-1800)")
assert 1100 <= n_words <= 1800
print("  OK")

# B
print()
print("=" * 70)
print("B — sub-sections 4.1-4.7")
print("=" * 70)
required = [
    "## 4.1 Pre-registered design",
    "## 4.2 Per-specification results",
    "## 4.3 QL01",
    "## 4.4 Beta-neutralisation",
    "## 4.5 Fama-French",
    "## 4.6 Combination",
    "## 4.7 Verdict",
]
for r in required:
    assert r in sec4, f"missing: {r}"
print(f"  OK: all {len(required)}")

# C
print()
print("=" * 70)
print("C — Tables 2 + 3")
print("=" * 70)
assert "*Table 2:" in sec4 and "*Table 3:" in sec4
print("  OK: Table 2 + Table 3 captioned")

# D
print()
print("=" * 70)
print("D — citations defined")
print("=" * 70)
cite_pat = re.compile(r"@([a-zA-Z][a-zA-Z0-9_]*\d{2,4}[a-zA-Z0-9_]*)")
sec4_cites = set(cite_pat.findall(sec4))
bib_keys = set(re.compile(r"@[a-zA-Z]+\{([^,\s]+)\s*,").findall(bib_src))
print(f"  cites: {sorted(sec4_cites)}")
undef = sec4_cites - bib_keys
assert not undef, f"undefined: {undef}"
print("  OK")

# E
print()
print("=" * 70)
print("E — headline statistics")
print("=" * 70)
required_stats = [
    "+0.985",  # QL01_T1 Sharpe
    "+2.312",  # NW t
    "0.011",   # raw p
    "+0.620",  # T2 Sharpe
    "0.057",   # T2 p
    "0.0029",  # BHY threshold
    "5.02",    # α annualised
    "0.020",   # FF median R²
    "+0.677",  # IC-meta Sharpe
    "-0.528",  # ERC meta Sharpe
    "364",     # OOS weeks
    "BHY FDR",
]
missing = [s for s in required_stats if s not in sec4]
assert not missing, f"missing: {missing}"
print(f"  OK: all {len(required_stats)} statistics present")

# F
print()
print("=" * 70)
print("F — MARGINAL verdict named")
print("=" * 70)
assert "MARGINAL" in sec4, "MARGINAL verdict label missing"
print("  OK: MARGINAL labeled")

# G
print()
print("=" * 70)
print("G — no [TODO] / [STUB]")
print("=" * 70)
for m in ("[TODO", "[STUB", "TBD"):
    assert m not in sec4
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

print()
print("=" * 70)
print("S4.4 verification PASS")
print("=" * 70)
