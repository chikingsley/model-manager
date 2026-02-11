from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from model_manager.api import server


def test_capabilities_endpoint_includes_openapi_links() -> None:
    client = TestClient(server.app)
    response = client.get("/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["openapi"]["swagger_ui"] == "/docs"
    assert payload["openapi"]["openapi_json"] == "/openapi.json"
    assert payload["mode_behavior"]["single_backend_at_a_time"] is True


def test_llms_txt_endpoint_has_capability_links() -> None:
    client = TestClient(server.app)
    response = client.get("/llms.txt")

    assert response.status_code == 200
    body = response.text
    assert "/openapi.json" in body
    assert "/capabilities/ocr" in body


def test_ocr_capabilities_reads_benchmark_file(tmp_path, monkeypatch) -> None:
    benchmark_file = tmp_path / "ocr.json"
    benchmark_file.write_text(
        json.dumps(
            {
                "meta": {"model": "zai-org/GLM-OCR", "timeout_s": 90},
                "results": [
                    {
                        "concurrency": 40,
                        "ok": 40,
                        "total": 40,
                        "errors": 0,
                        "req_s": 0.58,
                        "p95_s": 68.2,
                        "timeout_errors": 0,
                    },
                    {
                        "concurrency": 64,
                        "ok": 45,
                        "total": 64,
                        "errors": 19,
                        "req_s": 0.71,
                        "p95_s": 90.0,
                        "timeout_errors": 19,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(server, "OCR_BREAKPOINT_RESULT_FILE", benchmark_file)

    client = TestClient(server.app)
    response = client.get("/capabilities/ocr")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "zai-org/GLM-OCR"
    assert payload["no_error_max_concurrency"] == 40
    assert payload["first_failure_concurrency"] == 64
    assert payload["source_file"] == "ocr.json"
    assert payload["sla_profiles"][0]["name"] == "safe"
    assert payload["sla_profiles"][1]["name"] == "balanced"
    assert payload["sla_profiles"][2]["name"] == "aggressive"


def test_status_endpoint_survives_dependency_failures(monkeypatch) -> None:
    class Boom(Exception):
        pass

    def bad_services() -> list[Any]:
        raise Boom("service lookup failed")

    async def bad_ollama_running() -> bool:
        raise Boom("ollama ping failed")

    monkeypatch.setattr(server, "get_running_services", bad_services)
    monkeypatch.setattr(server, "ollama_is_running", bad_ollama_running)

    client = TestClient(server.app)
    response = client.get("/status")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["services"], list)
    assert payload["services"] == []
    assert payload["ollama_model"] is None


def test_activate_endpoint_returns_structured_failure_on_exception(monkeypatch) -> None:
    async def bad_activate(mode: str, model: str | None = None) -> Any:
        raise RuntimeError("activation blew up")

    monkeypatch.setattr(server, "activate", bad_activate)

    client = TestClient(server.app)
    response = client.post("/activate/ocr", json={"model": "zai-org/GLM-OCR"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["mode"] == "ocr"
    assert "Activation failed: RuntimeError" in payload["message"]
