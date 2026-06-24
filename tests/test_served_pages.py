"""tests/test_served_pages.py — frontend served-page smoke (e2e-lite).

Verifies the SPA-aware FastAPI serving: every route returns 200 and its prerendered HTML carries a
page-unique marker (so a 404→landing fallback can't false-pass), and unknown routes fall back to
the landing. Skips when frontend/out is absent (e.g. the api-contract CI job that doesn't build the
frontend) — the frontend CI job covers build+typecheck. Markers are the EN i18n defaults present at
static prerender time.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

_OUT = Path(__file__).resolve().parents[1] / "frontend" / "out"
pytestmark = pytest.mark.skipif(not _OUT.is_dir(), reason="frontend/out not built")

client = TestClient(app)

PAGES = {
    "/": "Enter Terminal",
    "/dashboard": "Book Health",
    "/agents": "Agent Constellation",
    "/chat": "Chief of Staff",
    "/book": "Book &amp; Positions",
    "/risk": "Risk Console",
    "/research": "What survived",
    "/ops": "Agent Ops",
    "/alerts": "Alerts &amp; Anomalies",
    "/approvals": "Decision queue",
    "/approvals/review": "Decision review",
    "/settings": "Terminal preferences",
}


@pytest.mark.parametrize("path,marker", list(PAGES.items()))
def test_page_served(path: str, marker: str):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    assert marker in r.text, f"{path} missing marker {marker!r}"


def test_refresh_safe_trailing_slash():
    assert client.get("/dashboard/").status_code == 200


def test_unknown_route_falls_back_to_landing():
    r = client.get("/totally-not-a-route")
    assert r.status_code == 200
    assert "Enter Terminal" in r.text   # SPA fallback to the landing
