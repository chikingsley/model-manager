"""
Benchmarking for any OpenAI-compatible LLM endpoint.

Sends streaming chat completions and measures:
- tok/s (tokens per second, generation phase)
- TTFT (time to first token)
- ITL (inter-token latency, median + P95)

Works identically across vLLM, Ollama, and llama.cpp.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

# Diverse prompts to exercise different generation patterns
BENCH_PROMPTS = [
    "Explain how a CPU processes instructions step by step.",
    "Write a short story about a robot learning to paint.",
    "What are the key differences between TCP and UDP?",
    "Describe the process of photosynthesis in detail.",
    "List and explain 5 creative uses for a paperclip.",
    "How does a hash table work internally?",
    "Write a haiku about each season of the year.",
    "Explain the difference between concurrency and parallelism.",
    "Describe how DNS resolution works from browser to server.",
    "What makes a good API design? Give concrete examples.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SingleResult:
    """Result from a single streaming request."""

    ttft_ms: float
    itl_ms: list[float] = field(default_factory=list)
    token_count: int = 0
    total_s: float = 0.0

    @property
    def tok_s(self) -> float:
        gen_time = self.total_s - (self.ttft_ms / 1000)
        return self.token_count / gen_time if gen_time > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    tok_s: float  # overall tokens/sec
    ttft_ms: float  # median time to first token
    itl_ms: float  # median inter-token latency
    p95_itl_ms: float  # P95 inter-token latency
    total_tokens: int
    num_requests: int
    duration_s: float
    model: str
    backend: str


ProgressCallback = Callable[[str], None]


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def _median(values: list[float]) -> float:
    return _percentile(values, 50)


# ─────────────────────────────────────────────────────────────────────────────
# Model detection
# ─────────────────────────────────────────────────────────────────────────────


async def detect_model(base_url: str) -> str | None:
    """Get the model name from /v1/models."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{base_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    return models[0]["id"]
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Single request benchmark
# ─────────────────────────────────────────────────────────────────────────────


async def _bench_single(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 128,
) -> SingleResult:
    """Send one streaming chat completion and measure timing."""
    start = time.monotonic()
    first_token_time: float | None = None
    prev_token_time: float | None = None
    itl_times: list[float] = []
    token_count = 0

    async with client.stream(
        "POST",
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.7,
        },
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line.strip() == "data: [DONE]":
                break
            try:
                data = json.loads(line[6:])
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    now = time.monotonic()
                    if first_token_time is None:
                        first_token_time = now
                    elif prev_token_time is not None:
                        itl_times.append((now - prev_token_time) * 1000)
                    prev_token_time = now
                    token_count += 1  # each content chunk ≈ 1 decode step
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    end = time.monotonic()
    ttft = (first_token_time - start) * 1000 if first_token_time else 0

    return SingleResult(
        ttft_ms=ttft,
        itl_ms=itl_times,
        token_count=token_count,
        total_s=end - start,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark
# ─────────────────────────────────────────────────────────────────────────────


async def run_benchmark(
    base_url: str,
    model: str | None = None,
    num_requests: int = 10,
    max_tokens: int = 128,
    on_progress: ProgressCallback | None = None,
    backend: str = "unknown",
) -> BenchmarkResult:
    """
    Run a benchmark against any OpenAI-compatible endpoint.

    Sends num_requests streaming chat completions, measures timing,
    and returns aggregated results.
    """

    def report(msg: str):
        if on_progress:
            on_progress(msg)

    # Auto-detect model
    if not model:
        model = await detect_model(base_url)
        if not model:
            raise RuntimeError(f"Could not detect model at {base_url}")

    report(f"Benchmarking {model}...")

    results: list[SingleResult] = []
    total_start = time.monotonic()

    async with httpx.AsyncClient() as client:
        # Warmup — prime the model, ignore timing
        report("  Warming up...")
        with contextlib.suppress(Exception):
            await _bench_single(client, base_url, model, "Say hello.", max_tokens=16)

        # Benchmark requests
        for i in range(num_requests):
            prompt = BENCH_PROMPTS[i % len(BENCH_PROMPTS)]
            report(f"  Request {i + 1}/{num_requests}...")
            try:
                result = await _bench_single(client, base_url, model, prompt, max_tokens)
                results.append(result)
            except Exception as e:
                report(f"  Request {i + 1} failed: {e}")

    total_duration = time.monotonic() - total_start

    if not results:
        raise RuntimeError("All benchmark requests failed")

    # Aggregate results
    all_ttfts = [r.ttft_ms for r in results]
    all_itls = [itl for r in results for itl in r.itl_ms]
    total_tokens = sum(r.token_count for r in results)
    total_gen_time = sum(max(r.total_s - r.ttft_ms / 1000, 0.001) for r in results)

    overall_tok_s = total_tokens / total_gen_time if total_gen_time > 0 else 0

    return BenchmarkResult(
        tok_s=round(overall_tok_s, 1),
        ttft_ms=round(_median(all_ttfts)),
        itl_ms=round(_median(all_itls), 1),
        p95_itl_ms=round(_percentile(all_itls, 95), 1),
        total_tokens=total_tokens,
        num_requests=len(results),
        duration_s=round(total_duration, 1),
        model=model,
        backend=backend,
    )
