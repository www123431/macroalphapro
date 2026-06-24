"""tests/test_spec_registry_hash_drift.py — workflow round-trip guard.

Background:
  ``engine.preregistration`` stores a git-blob-style SHA-1 of every locked
  spec at register / amend time. Two failure modes have surfaced in this
  project and the project relies on them being caught early:

    1. Spec file edited without a corresponding ``amend_spec`` call.
       The on-disk content drifts from the DB row and downstream code
       that quotes the DB hash (HALT marker JSON, agent telemetry,
       persona spec_ref) keeps citing a stale fingerprint.

    2. Literal hash strings pinned in a spec file's own frontmatter.
       Writing the hash into the file changes the file content and
       therefore invalidates the pin on the next byte — a self-
       referential fixed-point bug. The 2026-05-19 spec_risk_manager_
       agent_v1.md header cleanup was triggered by exactly this.

What this test covers:
  The WORKFLOW property — register_spec / amend_spec must store a hash
  that equals git_hash_object(file_bytes). If that invariant ever breaks
  (e.g. someone refactors _compute_git_blob_hash and accidentally swaps
  it for hashlib.sha256), this fails. It also covers drift detection: a
  silent edit to the file after register_spec but before amend_spec must
  be detectable by recomputing the hash.

What this test does NOT cover:
  Production-DB drift. That is the job of
  ``scripts/check_spec_registry_drift.py`` — run it from a pre-commit
  hook or CI workflow against the live macro_alpha_memory.db. A pytest
  test cannot validate production state because conftest.py isolates
  tests to a temp DB.
"""
from __future__ import annotations

import os
import tempfile

import pytest


def _write(path: str, body: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


@pytest.fixture
def temp_spec_file():
    """A throwaway spec-shaped markdown file inside the repo tree (so
    relative-path resolution inside engine.preregistration works) that
    we can register, mutate, and re-amend."""
    tmp = tempfile.NamedTemporaryFile(
        prefix     = "spec_drift_test_",
        suffix     = ".md",
        delete     = False,
        mode       = "w",
        encoding   = "utf-8",
    )
    tmp.write("# Test spec\n\nbody v1\n")
    tmp.close()
    yield tmp.name
    try:
        os.unlink(tmp.name)
    except Exception:
        pass


def test_register_spec_stores_git_blob_hash(temp_spec_file):
    """register_spec must store the git-blob SHA-1 of the file content."""
    from engine.preregistration import (
        _compute_git_blob_hash, list_specs, register_spec,
    )

    spec_id = register_spec(temp_spec_file, retro=False)
    expected = _compute_git_blob_hash(temp_spec_file)

    rows = [r for r in list_specs() if int(r["id"]) == spec_id]
    assert len(rows) == 1
    assert rows[0]["current_hash"] == expected, (
        "register_spec stored a hash that does not match the on-disk "
        "git-blob SHA-1. This breaks the spec-immutability contract."
    )


def test_silent_edit_is_detectable(temp_spec_file):
    """If the spec file is edited without calling amend_spec, the DB
    hash and recomputed hash must diverge. This is the property the
    drift-check script relies on."""
    from engine.preregistration import (
        _compute_git_blob_hash, list_specs, register_spec,
    )

    spec_id = register_spec(temp_spec_file, retro=False)
    rows = [r for r in list_specs() if int(r["id"]) == spec_id]
    stored = rows[0]["current_hash"]

    # Silent edit — no amend_spec call.
    _write(temp_spec_file, "# Test spec\n\nbody v2 (silently edited)\n")

    rehashed = _compute_git_blob_hash(temp_spec_file)
    assert stored != rehashed, (
        "Silent edits must change the git-blob hash. If they do not, "
        "_compute_git_blob_hash is broken (e.g. caching file size only) "
        "and drift cannot be detected."
    )


def test_amend_spec_updates_db_to_new_hash(temp_spec_file):
    """After amend_spec, the stored current_hash must equal the new
    file's git-blob hash — closing the drift gap."""
    from engine.preregistration import (
        _compute_git_blob_hash, amend_spec, list_specs, register_spec,
    )

    register_spec(temp_spec_file, retro=False)
    _write(temp_spec_file, "# Test spec\n\nbody v2\n")

    new_id = amend_spec(
        path   = temp_spec_file,
        kind   = "clarification",
        reason = "test: body bump v1 -> v2",
    )

    rows = [r for r in list_specs() if int(r["id"]) == new_id]
    assert len(rows) == 1
    expected = _compute_git_blob_hash(temp_spec_file)
    assert rows[0]["current_hash"] == expected, (
        "amend_spec failed to re-hash the file and store the result. "
        "This means an amendment workflow call leaves the DB stale."
    )
    log = rows[0].get("amendment_log") or []
    assert len(log) >= 1, "amend_spec must append to amendment_log"
