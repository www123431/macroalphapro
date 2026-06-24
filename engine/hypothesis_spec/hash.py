"""hypothesis_spec.hash — deterministic spec_hash.

The reproducibility contract (LdP §2):
  spec_hash(spec_A) == spec_hash(spec_B) ⇒ same series ⇒ same verdict

Determinism rules
-----------------
1. JSON-canonical with sorted keys + no whitespace
2. EXCLUDE per-instance fields: spec_id, created_ts, extraction.extracted_ts
   (these are about the SPEC OBJECT not the SPEC CONTENT — two different
   extraction runs of the same hypothesis produce different spec_ids but
   the same spec_hash)
3. EXCLUDE git_sha (different commits can yield identical specs)
4. INCLUDE: all content fields (family, legs, universe, construction,
   risk, outcome, extraction.method+extractor_v but NOT extracted_ts/conf)

Hash function: SHA-256 truncated to 16 hex chars (~64 bits → ~10^9
distinct specs before 50% collision probability). At our scale (~10^3
specs/year) collisions are practically zero.
"""
from __future__ import annotations

import hashlib
import json

from engine.hypothesis_spec.schema import HypothesisSpec


# Fields that DON'T contribute to content hash
_EXCLUDE_TOP_LEVEL = {
    "spec_id",        # per-instance UUID
    "version",        # revision number, not content
    "git_sha",        # extraction-time git, not content
    "created_ts",     # instance timestamp
}

_EXCLUDE_EXTRACTION = {
    "extracted_ts",   # extraction timestamp
    "confidence",     # provenance metric, not content
}


def spec_hash(spec: HypothesisSpec) -> str:
    """Deterministic 16-hex-char content hash."""
    d = spec.to_dict()
    # Strip per-instance noise
    for k in _EXCLUDE_TOP_LEVEL:
        d.pop(k, None)
    if "extraction" in d and isinstance(d["extraction"], dict):
        for k in _EXCLUDE_EXTRACTION:
            d["extraction"].pop(k, None)
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
