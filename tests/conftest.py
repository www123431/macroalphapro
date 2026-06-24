"""
tests/conftest.py — Shared pytest fixtures (S-1, 2026-05-06).

Critical decision: tests run against an isolated tempfile SQLite DB, not
against the production macro_alpha_memory.db. The DATABASE_URL env var
must be set BEFORE engine.memory imports, so this conftest sets it at
module import time (pytest loads conftest.py before any test module).

Live-LLM tests are marked `@pytest.mark.live_llm`. They are skipped by
default; run with `pytest --run-live` to include them. Cost ~$0.04
per pytest run; intended for monthly / pre-release smoke only.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# ── Path setup so `import engine.*` works from anywhere ─────────────────────
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Test DB isolation: must run BEFORE any engine import ────────────────────
_TEST_DB_FILE = tempfile.NamedTemporaryFile(suffix=".test.db", delete=False)
_TEST_DB_FILE.close()
_TEST_DB_PATH = _TEST_DB_FILE.name
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"


def pytest_configure(config):
    """Register the live_llm + network markers so '@pytest.mark.*' isn't a warning."""
    config.addinivalue_line(
        "markers",
        "live_llm: real Gemini API call (skipped by default; --run-live to include)",
    )
    config.addinivalue_line(
        "markers",
        "network: requires external network (e.g., yfinance / AV API)",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live_llm tests (real Gemini API calls; ~\\$0.04/run)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return  # don't skip
    skip_live = pytest.mark.skip(reason="needs --run-live (real Gemini call)")
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip_live)


# ── Initialise test DB once per session ─────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    """Create all tables in the test DB on session start; drop file on teardown."""
    # Ensure all ORM models register with Base before create_all
    import engine.auto_audit_models   # noqa: F401
    from engine.memory import init_db
    init_db()
    # UniverseETF uses its OWN declarative_base (engine/universe_manager.py)
    # so init_db's Base.metadata.create_all doesn't cover it. Seed it here
    # with the production INITIAL_18 — this matches what the live DB has.
    from engine.universe_manager import init_universe_db
    init_universe_db()
    yield
    # Teardown: best-effort delete temp DB
    try:
        os.unlink(_TEST_DB_PATH)
    except Exception:
        pass


# ── Per-test DB cleanup: each test starts with empty audit + approvals ──────
@pytest.fixture(autouse=True)
def _clean_audit_state():
    """
    Keep persistent fixtures (SpecRegistry, UniverseETF, SystemConfig) intact;
    wipe only what tests churn (audit log + approval queue + cost tracker).
    """
    from engine.memory import SessionFactory, PendingApproval
    from engine.auto_audit_models import AuditFinding, AuditProposal, AuditRun
    yield
    with SessionFactory() as s:
        s.query(AuditProposal).delete()
        s.query(AuditFinding).delete()
        s.query(AuditRun).delete()
        s.query(PendingApproval).filter(
            PendingApproval.approval_type.in_([
                "auto_audit_proposal", "anomaly_screener",
            ])
        ).delete(synchronize_session=False)
        s.commit()


# ── Reusable sample builders ────────────────────────────────────────────────
@pytest.fixture
def make_finding():
    """Factory: build & persist an AuditFinding with sane defaults."""
    import datetime
    import json as _json
    from engine.memory import SessionFactory
    from engine.auto_audit_models import AuditFinding, AuditRun

    def _make(rule_name="rule_skill_library_dormancy",
              severity="LOW",
              snapshot=None,
              status="OPEN"):
        with SessionFactory() as s:
            run = AuditRun(scope="weekly", n_rules_run=1, n_findings=1, exit_status="ok")
            s.add(run)
            s.flush()
            f = AuditFinding(
                run_id        = run.id,
                rule_name     = rule_name,
                severity      = severity,
                detected_at   = datetime.datetime.utcnow(),
                snapshot_json = _json.dumps(snapshot or {"n_rows": 0, "kind": "no_rows"}),
                status        = status,
            )
            s.add(f)
            s.flush()
            fid = f.id
            s.commit()
            return fid
    return _make


@pytest.fixture
def make_proposal_payload():
    """Factory: build a valid proposal payload dict (gate would PASS)."""
    def _make(**overrides):
        base = {
            "summary":              "Test summary describing the contradiction concisely",
            "diagnosis":            "Test diagnosis explaining root cause via snapshot evidence",
            "options": [
                {
                    "action":               "test action describing remediation step",
                    "pros":                 ["test pro citing snapshot evidence"],
                    "cons":                 ["test con acknowledging trade-off"],
                    "estimated_effort_min": 5,
                    "risk_level":           "LOW",
                    "files_to_touch":       ["docs/spec_temp.md"],
                    "diff_size_estimate":   3,
                },
            ],
            "recommendation_index": 0,
            "amendment_kind":       "clarification",
            "rationale_short":      "Test rationale explaining the proposed clarification action.",
            "evidence_refs":        ["snapshot.kind", "test reference 2"],
        }
        base.update(overrides)
        return base
    return _make


@pytest.fixture
def make_proposal(make_finding, make_proposal_payload):
    """Factory: build & persist an AuditProposal linked to a finding."""
    import json as _json
    from engine.memory import SessionFactory
    from engine.auto_audit_models import AuditProposal

    def _make(finding_id=None,
              generation_status="success",
              payload=None,
              **proposal_kwargs):
        if finding_id is None:
            finding_id = make_finding()
        if payload is None:
            payload = make_proposal_payload()
        with SessionFactory() as s:
            p = AuditProposal(
                finding_id          = finding_id,
                model_version       = "test-model",
                prompt_hash         = "test-hash",
                parsed_payload_json = _json.dumps(payload),
                generation_status   = generation_status,
                **proposal_kwargs,
            )
            s.add(p)
            s.flush()
            pid = p.id
            s.commit()
            return pid, finding_id
    return _make


@pytest.fixture
def mock_gemini(monkeypatch):
    """
    Replace `_call_gemini` in auto_audit_proposer with a mock returning a
    deterministic fixture payload. Use `payload=...` to override.
    """
    import json as _json

    def _install(payload=None, raise_exc=None):
        """Install the mock. Returns the payload that will be returned."""
        from engine import auto_audit_proposer as p

        if payload is None:
            payload = {
                "summary":              "Mock proposal summary text",
                "diagnosis":            "Mock proposal diagnosis citing snapshot.",
                "options": [{
                    "action":               "mock action: amend_spec clarification",
                    "pros":                 ["mock pro"],
                    "cons":                 ["mock con"],
                    "estimated_effort_min": 5,
                    "risk_level":           "LOW",
                    "files_to_touch":       ["docs/spec_temp.md"],
                    "diff_size_estimate":   3,
                }],
                "recommendation_index": 0,
                "amendment_kind":       "clarification",
                "rationale_short":      "Mock rationale acknowledging dead-branch state.",
                "evidence_refs":        ["mock_snapshot_ref"],
            }

        def _fake_call(prompt: str) -> dict:
            if raise_exc is not None:
                raise raise_exc
            return {
                "parsed":        payload,
                "raw_text":      _json.dumps(payload),
                "prompt_hash":   "mock-prompt-hash",
                "input_tokens":  100,
                "output_tokens": 200,
                "cost_usd":      0.0,
            }

        monkeypatch.setattr(p, "_call_gemini", _fake_call)
        return payload
    return _install
