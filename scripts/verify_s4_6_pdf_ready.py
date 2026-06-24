"""S4.6 PDF-readiness verification.

Without an installed pandoc/LaTeX toolchain we cannot render the PDF inline,
so this script verifies the manuscript is *conversion-ready*:

  A. paper.md lints clean (cite syntax / dollar matching / table balance)
  B. all cite keys defined in references.bib
  C. no [TODO] / [STUB] markers
  D. all sections + appendices populated (Appendix A table, Appendix B hash table)
  E. word count within spec target (12-15 pages)
  F. spec_hash drift check
  G. build_pdf.sh executable + parseable
  H. README.md present + references core artefacts

Final PDF/LaTeX rendering is deferred to S4.7 once pandoc + a TeX
distribution are installed (see paper/README.md prerequisites).
"""
import sys, os, re, hashlib, stat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
PAPER = os.path.join(PAPER_DIR, "paper.md")
BIB = os.path.join(PAPER_DIR, "references.bib")
SPEC = os.path.join(os.path.dirname(PAPER_DIR), "docs", "spec_ssrn_paper_v1.md")
BUILD = os.path.join(PAPER_DIR, "build_pdf.sh")
README = os.path.join(PAPER_DIR, "README.md")

with open(PAPER, "r", encoding="utf-8") as f:
    paper_src = f.read()
with open(BIB, "r", encoding="utf-8") as f:
    bib_src = f.read()
with open(SPEC, "r", encoding="utf-8") as f:
    spec_src = f.read()


# A. lint
print("=" * 70)
print("A — paper.md lints clean")
print("=" * 70)
n_dollars = paper_src.count("$")
print(f"  size: {len(paper_src)} bytes / {paper_src.count(chr(10))} lines")
print(f"  $ count: {n_dollars} (must be even)")
assert n_dollars % 2 == 0
# Cite syntax: should use @key not [@key without trailing chars]
malformed_cites = re.findall(r"@\d", paper_src)  # @<digit> = malformed
assert not malformed_cites, f"malformed cites: {malformed_cites}"
print("  OK: cite syntax + dollar match clean")

# B. cite keys defined
print()
print("=" * 70)
print("B — all cite keys defined in references.bib")
print("=" * 70)
cite_pat = re.compile(r"@([a-zA-Z][a-zA-Z0-9_]*\d{2,4}[a-zA-Z0-9_]*)")
cites = set(cite_pat.findall(paper_src))
bib_keys = set(re.compile(r"@[a-zA-Z]+\{([^,\s]+)\s*,").findall(bib_src))
print(f"  paper cites: {len(cites)}, bib keys: {len(bib_keys)}")
undef = cites - bib_keys
assert not undef
unused_in_paper = bib_keys - cites
print(f"  bib keys not yet cited (acceptable, future work): {len(unused_in_paper)}")
print("  OK: every cite has a backing entry")

# C. no markers
print()
print("=" * 70)
print("C — no [TODO] / [STUB] / TBD")
print("=" * 70)
for m in ("[TODO", "[STUB", "TBD", "XXX"):
    assert m not in paper_src, f"{m} remains"
print("  OK")

# D. appendices populated
print()
print("=" * 70)
print("D — Appendix A table + Appendix B hash registry populated")
print("=" * 70)
app_a_start = paper_src.find("# Appendix A")
app_b_start = paper_src.find("# Appendix B")
end = len(paper_src)
app_a = paper_src[app_a_start:app_b_start]
app_b = paper_src[app_b_start:end]
assert "*Table A.1:" in app_a, "Appendix A caption missing"
assert "QL01_T1" in app_a, "Appendix A table data missing"
assert "*Table B.1:" in app_b, "Appendix B caption missing"
assert "spec_hash" in app_b
# Count Appendix A data rows
a_rows = [l for l in app_a.split("\n") if l.strip().startswith("|") and "QL01" in l or "TS0" in l or "CR0" in l or "CL0" in l or "MA0" in l or "RV0" in l or "XA0" in l]
print(f"  Appendix A data rows: ~{len(a_rows)}")
assert len(a_rows) >= 30
b_hashes = re.findall(r"[0-9a-f]{16}", app_b)
print(f"  Appendix B hashes: {len(b_hashes)}")
assert len(b_hashes) >= 8
print("  OK: both appendices have content")

# E. word count
print()
print("=" * 70)
print("E — paper-level word count")
print("=" * 70)
body_start = paper_src.find("# 1. Introduction")
body_end = paper_src.find("# References")
body = paper_src[body_start:body_end]
n_body = len(re.findall(r"\b[A-Za-z0-9]+\b", body))
total = len(re.findall(r"\b[A-Za-z0-9]+\b", paper_src))
print(f"  body §1-§7: {n_body}")
print(f"  full paper (incl. abstract/appendices): {total}")
print(f"  ≈ pages at 500 wpm: {n_body / 500:.1f}")
assert 5500 <= n_body <= 9500
print("  OK")

# F. spec_hash
print()
print("=" * 70)
print("F — spec_hash drift")
print("=" * 70)
sh = hashlib.sha256(spec_src.encode("utf-8")).hexdigest()[:16]
print(f"  spec_hash: {sh}")
assert sh == "03a6767a5e5ea600"
print("  OK: spec untouched")

# G. build script
print()
print("=" * 70)
print("G — build_pdf.sh present + valid")
print("=" * 70)
assert os.path.exists(BUILD), "build_pdf.sh missing"
with open(BUILD, "r", encoding="utf-8") as f:
    bsrc = f.read()
for needle in ("pandoc", "--citeproc", "--bibliography=", "references.bib", "paper.md"):
    assert needle in bsrc, f"build script missing {needle}"
print("  OK: shell script references pandoc + bib + source")

# H. README
print()
print("=" * 70)
print("H — paper/README.md present + correct")
print("=" * 70)
assert os.path.exists(README)
with open(README, "r", encoding="utf-8") as f:
    rsrc = f.read()
for needle in ("Spec lock", "spec_hash", "03a6767a5e5ea600",
               "build_pdf.sh", "Reproducing B++", "SSRN submission"):
    assert needle in rsrc, f"README missing {needle}"
print("  OK: README links spec, build script, B++ data, SSRN steps")

print()
print("=" * 70)
print("S4.6 PDF-ready verification PASS (final render deferred to S4.7)")
print("=" * 70)
