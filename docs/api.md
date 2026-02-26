# API Documentation Guide

## Interactive API Docs

When the API server container is running, use:

```bash
cd /home/simon/docker/model-manager
docker compose -f docker-compose.api.yml up -d --build
```

- Swagger UI: `http://localhost:4001/docs`
- ReDoc: `http://localhost:4001/redoc`
- OpenAPI JSON: `http://localhost:4001/openapi.json`

These are auto-generated from FastAPI and should be treated as the canonical contract for integrations.

## Operational Limits Endpoints

- `GET /capabilities`
  - Returns links to docs/OpenAPI and mode-switch behavior guarantees.
- `GET /capabilities/ocr`
  - Returns measured OCR throughput envelope from benchmark artifacts.
  - Current source: `benchmarks/results/ocr-load/glm_ocr_budget_8192_breakpoint_timeout90.json`
- `GET /llms.txt`
  - Concise model-facing manifest for agent/tool discovery.
- `GET /llms-full.txt`
  - Expanded model-facing context with OCR throughput summary and SLA guidance.

## SAM3 Backend Endpoints

- `POST /activate/sam3`
  - Starts the dedicated SAM3 segmentation backend (`services/sam3`).
  - Stops conflicting GPU services before activation.
- `GET /status`
  - Shows `active: sam3` and running SAM3 service details when enabled.

SAM3 service API (local container):
- `GET http://localhost:8095/health`
- `POST http://localhost:8095/segment`

Runbook:
- `docs/sam3.md`

## Current OCR Throughput Envelope (Heavy Docs)

Environment:
- Single GPU: RTX 5070 12GB
- Timeout: 90s per request
- Runtime profile: GLM-OCR on vLLM with `--max-num-batched-tokens 8192`

Observed:
- Stable (0 errors): up to concurrency `56`
- First failure point: concurrency `64` (timeouts begin)
- Practical recommended range: `40-56` concurrent heavy OCR requests

Use these as operating guidance, not hard universal limits. Re-run sweeps when model/runtime flags or hardware change.

## Suggested Public Tunnel Routing

Use a dedicated subdomain for this API and map it to the API service port.

- Recommended host: `mm.<your-domain>`
- Recommended target: `http://localhost:4001`
- Recommended public entrypoints:
  - `/docs` for humans
  - `/openapi.json` for tooling
  - `/llms.txt` and `/llms-full.txt` for agents
