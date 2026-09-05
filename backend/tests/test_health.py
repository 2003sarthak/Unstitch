"""API contract test for the one endpoint that exists so far."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import APP_VERSION, create_app


def test_health_reports_effective_provider_not_requested_one() -> None:
    settings = Settings(_env_file=None, vision_provider="gemini", gemini_api_key="")
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/health").json()

    assert body == {
        "status": "ok",
        "version": APP_VERSION,
        "vision_provider": "stub",
        "vision_provider_requested": "gemini",
        "removal_mode": "delogo",
    }


def test_cors_headers_reflect_configured_origin() -> None:
    settings = Settings(_env_file=None, cors_origins="https://unstitch.vercel.app")
    with TestClient(create_app(settings)) as client:
        r = client.get("/api/health", headers={"Origin": "https://unstitch.vercel.app"})

    assert r.headers["access-control-allow-origin"] == "https://unstitch.vercel.app"
