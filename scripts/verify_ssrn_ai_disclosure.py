"""Verify SSRN AI-disclosure amendment.

Facets:
  A. Abstract preamble contains AI-use disclosure (3+ sentences keyword test)
  B. Standalone §8 AI-Use Disclosure section present + nontrivial length
  C. §8 covers all three buckets: subject / assistant / judge
  D. spec_ssrn_paper_v1.md amended to add §8 contract
  E. SpecRegistry has 1 amendment recorded with kind=clarification, +0 n_trials
  F. paper.md still passes pdf-readiness lint (no broken cites, etc)
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import init_db, SessionFactory, SpecRegistry
init_db()

PAPER = "paper/paper.md"
SPEC = "docs/spec_ssrn_paper_v1.md"

with open(PAPER, "r", encoding="utf-8") as f:
    paper_src = f.read()
with open(SPEC, "r", encoding="utf-8") as f:
    spec_src = f.read()


# A. abstract AI-use preamble
print("=" * 70)
print("A — abstract contains AI-use disclosure preamble")
print("=" * 70)
fm_end = paper_src.find("---\n\n# 1.")
abstract_block = paper_src[:fm_end] if fm_end > 0 else paper_src[:3500]
ai_anchors = ["AI-use disclosure", "Anthropic Claude", "Google Gemini",
              "no LLM was used to evaluate", "Section 8"]
present = [a for a in ai_anchors if a.lower() in abstract_block.lower()]
print(f"  ai anchors found in abstract: {len(present)}/{len(ai_anchors)}")
for a in ai_anchors:
    print(f"    {'OK' if a.lower() in abstract_block.lower() else 'MISS'}  {a!r}")
assert len(present) >= 4, f"abstract weak on AI disclosure: {present}"
print("  OK")


# B. §8 standalone section
print()
print("=" * 70)
print("B — §8 AI-Use Disclosure section")
print("=" * 70)
s8 = paper_src.find("# 8. AI-Use Disclosure")
s_refs = paper_src.find("# References")
assert s8 > 0 and s_refs > s8, "§8 not in expected position"
sec8 = paper_src[s8:s_refs]
n8 = len(re.findall(r"\b[A-Za-z0-9]+\b", sec8))
print(f"  §8 word count: {n8}")
assert 250 <= n8 <= 1200, f"§8 {n8} outside 250-1200"
print("  OK")


# C. three buckets
print()
print("=" * 70)
print("C — §8 covers research subject / assistant / judge")
print("=" * 70)
buckets = ["research subject", "research assistant", "evaluation judge"]
for b in buckets:
    assert b in sec8.lower(), f"§8 missing bucket: {b}"
    print(f"    OK: {b}")
print("  OK")


# D. spec amended
print()
print("=" * 70)
print("D — spec_ssrn_paper_v1.md adds §8 contract")
print("=" * 70)
assert "AI-Use Disclosure" in spec_src
assert "no LLM-as-judge" in spec_src or "evaluation judge" in spec_src.lower()
print("  OK: spec lists §8 deliverable")


# E. SpecRegistry amendment ledger entry
print()
print("=" * 70)
print("E — amendment recorded in SpecRegistry")
print("=" * 70)
with SessionFactory() as s:
    row = s.query(SpecRegistry).filter(
        SpecRegistry.spec_path == "docs/spec_ssrn_paper_v1.md"
    ).one()
    ledger = json.loads(row.amendment_log or "[]")
    print(f"  amendments on SSRN spec: {len(ledger)}")
    assert len(ledger) >= 1
    last = ledger[-1]
    print(f"    kind: {last['kind']}")
    print(f"    n_trials_added: {last['n_trials_added']}")
    print(f"    reason: {last['reason'][:60]}...")
    assert last["kind"] == "clarification"
    assert last["n_trials_added"] == 0  # clarification does not contribute
    assert "AI-use disclosure" in last["reason"]
    print(f"  retro_registered={row.retro_registered}, "
          f"n_trials_contributed={row.n_trials_contributed}")
print("  OK: amendment audited via SpecRegistry")


# F. paper still lints
print()
print("=" * 70)
print("F — paper.md still pdf-ready")
print("=" * 70)
n_dollars = paper_src.count("$")
assert n_dollars % 2 == 0, "unmatched $"
# All cites still defined
cite_pat = re.compile(r"@([a-zA-Z][a-zA-Z0-9_]*\d{2,4}[a-zA-Z0-9_]*)")
cites = set(cite_pat.findall(paper_src))
with open("paper/references.bib", "r", encoding="utf-8") as f:
    bib = f.read()
bib_keys = set(re.compile(r"@[a-zA-Z]+\{([^,\s]+)\s*,").findall(bib))
undef = cites - bib_keys
assert not undef, f"undefined cites: {undef}"
# No TODO/STUB
for m in ("[TODO", "[STUB", "TBD"):
    assert m not in paper_src
print(f"  paper.md: {len(paper_src)} bytes, {paper_src.count(chr(10))} lines")
print(f"  cites: {len(cites)}, undefined: 0, $ matched, no TODO")
print("  OK")


print()
print("=" * 70)
print("SSRN AI-disclosure amendment verification PASS")
print("=" * 70)
