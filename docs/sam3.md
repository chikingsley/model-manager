# SAM3 Backend (Official Meta)

This project includes a dedicated SAM3 segmentation service at `services/sam3/`.

## Why Separate from vLLM/llama.cpp

- `facebook/sam3` is a segmentation model, not a chat/completions model.
- vLLM and llama.cpp are not the right runtime for SAM3 inference.
- `mm sam3` runs SAM3 as its own managed backend and handles GPU switching.

## Activate

```bash
cd /home/simon/docker/model-manager
uv run mm sam3
```

This command:
- Stops conflicting GPU services (`vllm`, `ollama`, `llama-server`, `nemotron`).
- Starts the SAM3 container from `services/sam3/docker-compose.yml`.
- Waits for health before setting active mode to `sam3`.

## API

- Base URL: `http://localhost:8095`
- Health: `GET /health`
- Segmentation: `POST /segment`

Request body:

```json
{
  "image_url": "https://.../image.jpg",
  "prompt": "receipt",
  "top_k": 5
}
```

You can use `image_path` instead of `image_url` for local files.

Response includes:
- `scores` (descending)
- `boxes`
- `mask_pixels` (non-zero pixel counts per returned mask)

## Environment

Container env vars:
- `HF_TOKEN` (required for gated access to `facebook/sam3`)
- `SAM3_MODEL_ID` (default `facebook/sam3`)
- `SAM3_DEVICE` (default `cuda`)

## Logs and Troubleshooting

```bash
docker logs sam3 --tail 200
docker exec sam3 python -c "import torch; print(torch.cuda.is_available())"
```
