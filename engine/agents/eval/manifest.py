"""engine/agents/eval/manifest.py — eval manifest (model/prompt-change governance).

A frozen fingerprint per persona: system-prompt hash + routed model + tool-set hash. A
prompt / model / tool change flips the fingerprint -> the manifest test fails -> the change
must be ACKNOWLEDGED (re-freeze) and the eval gate re-run. This is SR-11-7-style change
detection for LLM agents: you cannot silently alter an agent's behavior surface.

Frozen baseline lives at engine/agents/eval/manifest_frozen.json (committed).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FROZEN_PATH = Path(__file__).resolve().parent / "manifest_frozen.json"


def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def build_manifest() -> dict:
    """Current fingerprint of every built persona's behavior surface."""
    from engine.agents.eval.runner import _personas
    try:
        from engine.llm.call import _WORKLOAD_ROUTING
    except Exception:
        _WORKLOAD_ROUTING = {}
    out = {}
    for aid, p in sorted(_personas().items()):
        tools = sorted(t["name"] for t in p.tools)
        out[aid] = {
            "prompt_sha": _sha(p.system_prompt),
            "model": list(_WORKLOAD_ROUTING.get(p.workload, ("?", "?"))),
            "n_tools": len(tools),
            "tools_sha": _sha("|".join(tools)),
        }
    return out


def freeze_manifest(path: Path = FROZEN_PATH) -> dict:
    m = build_manifest()
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return m


def load_frozen(path: Path = FROZEN_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def check_manifest(path: Path = FROZEN_PATH) -> dict:
    """Compare current vs frozen. Returns {changed, added, removed} agent_ids."""
    cur, frozen = build_manifest(), load_frozen(path)
    changed = sorted(a for a in cur if a in frozen and cur[a] != frozen[a])
    added = sorted(set(cur) - set(frozen))
    removed = sorted(set(frozen) - set(cur))
    return {"changed": changed, "added": added, "removed": removed,
            "clean": not (changed or added or removed)}


if __name__ == "__main__":
    import sys
    if "--freeze" in sys.argv:
        m = freeze_manifest()
        print(f"froze {len(m)} personas -> {FROZEN_PATH}")
    else:
        r = check_manifest()
        print(json.dumps(r, indent=2))
        if not r["clean"]:
            print("\nMANIFEST DRIFT — a persona behavior surface changed. Re-run the eval "
                  "gate, then `python -m engine.agents.eval.manifest --freeze` to acknowledge.")
        sys.exit(0 if r["clean"] else 1)
