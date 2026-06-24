"""Tests for engine.research.repro_manifest (Phase 1 P0b)."""
from __future__ import annotations

import dataclasses
import json

import pytest

from engine.research import repro_manifest as rm


# ── build_manifest ──────────────────────────────────────────────────────

def test_build_manifest_returns_dataclass():
    m = rm.build_manifest(pipeline_config={"phase": 3})
    assert isinstance(m, rm.ReproManifest)
    assert m.git_commit_sha
    assert m.python_version
    assert m.run_timestamp
    assert m.pipeline_config == {"phase": 3}


def test_build_manifest_includes_data_files():
    m = rm.build_manifest()
    assert isinstance(m.data_file_manifest, dict)
    assert len(m.data_file_manifest) > 0


def test_build_manifest_library_versions_present():
    m = rm.build_manifest()
    assert "pandas" in m.library_versions
    assert "numpy" in m.library_versions


def test_build_manifest_custom_data_paths():
    m = rm.build_manifest(data_paths=["data/cache/_dpead_recon_base.parquet"])
    assert len(m.data_file_manifest) == 1


def test_build_manifest_pipeline_config_stored():
    config = {"phase": 2, "n_splits": 4, "random_seed": 42}
    m = rm.build_manifest(pipeline_config=config)
    assert m.pipeline_config == config


# ── hash_output ─────────────────────────────────────────────────────────

def test_hash_output_deterministic():
    out = {"a": 1, "b": "two", "c": [1, 2, 3]}
    h1 = rm.hash_output(out)
    h2 = rm.hash_output(out)
    assert h1 == h2
    assert len(h1) == 16


def test_hash_output_different_for_different_input():
    h1 = rm.hash_output({"a": 1})
    h2 = rm.hash_output({"a": 2})
    assert h1 != h2


def test_hash_output_key_order_invariant():
    """Same dict with different insertion order should hash the same."""
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"c": 3, "a": 1, "b": 2}
    assert rm.hash_output(d1) == rm.hash_output(d2)


def test_hash_output_handles_dataclass():
    @dataclasses.dataclass
    class X:
        a: int = 1
        b: str = "hi"
        def to_dict(self):
            return dataclasses.asdict(self)
    h = rm.hash_output(X())
    assert len(h) == 16


# ── verify_reproducible ─────────────────────────────────────────────────

def test_verify_same_manifest_reproducible():
    m1 = rm.build_manifest(pipeline_config={"phase": 3})
    m2 = rm.build_manifest(pipeline_config={"phase": 3})
    m1.output_hash = m2.output_hash = "abc"
    result = rm.verify_reproducible(m1, m2)
    assert result["fully_reproducible"] is True


def test_verify_different_output_hash_not_reproducible():
    m1 = rm.build_manifest()
    m2 = rm.build_manifest()
    m1.output_hash = "abc"
    m2.output_hash = "def"
    result = rm.verify_reproducible(m1, m2)
    assert result["fully_reproducible"] is False
    assert result["output_hash_match"] is False


def test_verify_returns_diffs_on_library_version_change():
    m1 = rm.build_manifest()
    m2 = rm.build_manifest()
    m1.output_hash = m2.output_hash = "x"
    m1.library_versions = {"pandas": "1.0.0"}
    m2.library_versions = {"pandas": "2.0.0"}
    result = rm.verify_reproducible(m1, m2)
    assert result["fully_reproducible"] is False
    assert "library_versions" in result["differences"]


def test_verify_returns_diffs_on_data_checksum_change():
    m1 = rm.build_manifest()
    m2 = rm.build_manifest()
    m1.output_hash = m2.output_hash = "x"
    m1.data_file_manifest["dummy.parquet"] = {"sha256_first_2mb": "abc"}
    m2.data_file_manifest["dummy.parquet"] = {"sha256_first_2mb": "def"}
    result = rm.verify_reproducible(m1, m2)
    assert "data_files_changed" in result["differences"]


def test_manifest_to_dict_round_trips():
    m = rm.build_manifest(pipeline_config={"phase": 3})
    m.output_hash = "abc"
    d = m.to_dict()
    assert d["git_commit_sha"] == m.git_commit_sha
    assert d["pipeline_config"]["phase"] == 3
    assert d["output_hash"] == "abc"
    # Should be JSON-serializable
    json.dumps(d, default=str)
