# vLLM Environment Files

- Active runtime config: `../.env`
- Model/profile presets: `profiles/*.env`
- Historical/benchmark snapshots: `archive/*`

To switch profiles:

1. `cp env/profiles/<name>.env .env`
2. `docker compose up -d --force-recreate vllm`
