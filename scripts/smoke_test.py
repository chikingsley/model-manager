# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""
Smoke test for the deployed model-manager API.

Validates the live container is healthy and returning real data.
Run after any deploy: uv run scripts/smoke_test.py

Checks:
  - All endpoints respond (not 500)
  - GPU metrics are real (not fallback zeros)
  - State file is resolving correctly (models registry populated)
  - Docker socket is working (services list populated when containers run)
  - TUI config matches actual API port
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

API_URL = "http://localhost:8890"
EXPECTED_PORT = 8890
TUI_CLIENT_PATH = Path(__file__).parent.parent / "tui" / "src" / "lib" / "mm.ts"


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main() -> int:
    print(f"Smoke testing {API_URL}\n")
    client = httpx.Client(timeout=10)
    passed = 0
    failed = 0

    def run(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if check(name, ok, detail):
            passed += 1
        else:
            failed += 1

    # ── Health ────────────────────────────────────────────────────────────
    try:
        r = client.get(f"{API_URL}/health")
        run("Health endpoint", r.status_code == 200, f"HTTP {r.status_code}")
    except httpx.ConnectError:
        run("Health endpoint", False, "Connection refused — is the container running?")
        print(f"\n  ABORT: API not reachable at {API_URL}")
        return 1

    # ── Status ───────────────────────────────────────────────────────────
    r = client.get(f"{API_URL}/status")
    run("Status endpoint", r.status_code == 200, f"HTTP {r.status_code}")

    if r.status_code == 200:
        status = r.json()

        # GPU metrics should be real, not fallback zeros
        vram = status["resources"]["vram"]
        gpu_real = vram["total_gb"] > 0 and vram["percent"] > 0
        run(
            "GPU metrics (real, not zeros)",
            gpu_real,
            f"{vram['used_gb']}/{vram['total_gb']} GB ({vram['percent']}%)",
        )

        gpu_temp = status["resources"]["gpu_temperature"]
        run("GPU temperature", gpu_temp > 0, f"{gpu_temp}°C")

        # Active state should not be "none" if services are running
        active = status["active"]
        services = status["services"]
        if services:
            run(
                "Active state matches running services",
                active != "none",
                f"active={active}, {len(services)} service(s) running",
            )
        else:
            run("Active state (no services)", True, f"active={active}")

        # Services should have real data
        for svc in services:
            run(
                f"Service '{svc['name']}' has health",
                svc["healthy"] in ("healthy", "starting", "unhealthy", "none"),
                f"healthy={svc['healthy']}",
            )

        # Tunnels
        tunnels = status["tunnels"]
        run("Tunnels detected", isinstance(tunnels, list), f"{len(tunnels)} tunnel(s)")

    # ── Models registry ──────────────────────────────────────────────────
    r = client.get(f"{API_URL}/models")
    run("Models endpoint", r.status_code == 200, f"HTTP {r.status_code}")

    if r.status_code == 200:
        models = r.json()
        run(
            "Models registry populated",
            len(models) > 0,
            f"{len(models)} model(s) registered",
        )

    # ── Capabilities ─────────────────────────────────────────────────────
    r = client.get(f"{API_URL}/capabilities")
    run("Capabilities endpoint", r.status_code == 200, f"HTTP {r.status_code}")

    # ── Resources ────────────────────────────────────────────────────────
    r = client.get(f"{API_URL}/resources")
    run("Resources endpoint", r.status_code == 200, f"HTTP {r.status_code}")

    if r.status_code == 200:
        res = r.json()
        ram = res["ram"]
        run("RAM metrics", ram["total_gb"] > 80, f"{ram['used_gb']}/{ram['total_gb']} GB")

    # ── llms.txt ─────────────────────────────────────────────────────────
    r = client.get(f"{API_URL}/llms.txt")
    run("llms.txt endpoint", r.status_code == 200 and "/openapi.json" in r.text)

    # ── OpenAPI schema ───────────────────────────────────────────────────
    r = client.get(f"{API_URL}/openapi.json")
    run("OpenAPI schema", r.status_code == 200, f"HTTP {r.status_code}")

    if r.status_code == 200:
        schema = r.json()
        paths = list(schema.get("paths", {}).keys())
        run("OpenAPI has paths", len(paths) > 5, f"{len(paths)} paths")

    # ── TUI port check ───────────────────────────────────────────────────
    if TUI_CLIENT_PATH.exists():
        tui_content = TUI_CLIENT_PATH.read_text()
        port_correct = f"localhost:{EXPECTED_PORT}" in tui_content
        run("TUI API port matches", port_correct, f"expected {EXPECTED_PORT}")
    else:
        run("TUI client file exists", False, str(TUI_CLIENT_PATH))

    # ── Summary ──────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n  {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED")
    else:
        print(" — all good")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
