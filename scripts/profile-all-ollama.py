#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Profile all Ollama models — context limits + speed at each level."""

import asyncio
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from model_manager.cli import BOLD, CYAN, GREEN, NC, RED, YELLOW
from model_manager.containers import docker_kill, is_running, ollama_list_models
from model_manager.ollama import test_model_context
from model_manager.modes import _get_model_key
from model_manager.state import ContextSpeedPoint, ModelEntry, StateManager


async def profile_model(model: str, state: StateManager) -> bool:
    """Run full context-speed profile for one model."""
    print(f"\n{BOLD}{'═' * 60}{NC}")
    print(f"{BOLD}  Profiling: {model}{NC}")
    print(f"{BOLD}{'═' * 60}{NC}\n")

    def on_progress(msg: str):
        print(f"  {msg}")

    try:
        result = await test_model_context(model, on_progress)
    except Exception as e:
        print(f"  {RED}Failed: {e}{NC}")
        return False

    if result.tested_max_ctx == 0:
        print(f"  {RED}No context sizes worked!{NC}")
        return False

    # Build context-speed profile
    profile = [
        ContextSpeedPoint(
            num_ctx=r.num_ctx,
            tok_s=r.tok_s,
            ttft_ms=r.ttft_ms,
            vram_mb=r.vram_mb,
        )
        for r in result.results
        if r.success and r.tok_s > 0
    ]

    # Save to registry
    model_key = _get_model_key("ollama", model)
    existing = state.get_model(model_key)

    if existing:
        existing.context_tested = True
        existing.claimed_num_ctx = result.claimed_max_ctx
        existing.tested_num_ctx = result.tested_max_ctx
        existing.num_ctx = result.recommended_ctx
        existing.vram_estimate = round(result.vram_at_max_mb / 1024, 1)
        existing.context_profile = profile or None
        existing.notes = f"Auto-tested: max {result.tested_max_ctx:,}, using {result.recommended_ctx:,}"
        state.register_model(model_key, existing)
    else:
        entry = ModelEntry(
            source="ollama",
            model=model,
            backend="ollama",
            context_tested=True,
            claimed_num_ctx=result.claimed_max_ctx,
            tested_num_ctx=result.tested_max_ctx,
            num_ctx=result.recommended_ctx,
            vram_estimate=round(result.vram_at_max_mb / 1024, 1),
            notes=f"Auto-tested: max {result.tested_max_ctx:,}, using {result.recommended_ctx:,}",
            context_profile=profile or None,
        )
        state.register_model(model_key, entry)

    # Print speed profile table
    if profile:
        print(f"\n  {CYAN}Context-Speed Profile:{NC}")
        print(f"  {'Context':>10}  {'Tok/s':>7}  {'TTFT':>7}  {'VRAM':>8}")
        print(f"  {'─' * 10}  {'─' * 7}  {'─' * 7}  {'─' * 8}")
        for p in profile:
            ctx_str = f"{p.num_ctx:,}"
            print(f"  {ctx_str:>10}  {p.tok_s:>6.1f}  {p.ttft_ms:>5.0f}ms  {p.vram_mb:>6}MB")

    print(f"\n  {GREEN}Saved as '{model_key}'{NC}")
    return True


async def main():
    state = StateManager()

    # Get all available Ollama models
    models = await ollama_list_models()
    if not models:
        print(f"{RED}No Ollama models found. Is Ollama running?{NC}")
        return

    print(f"{BOLD}═══ Ollama Full Profiler ═══{NC}")

    # Free GPU — stop other services that compete for VRAM
    for svc in ["vllm", "llama-server", "nemotron"]:
        if is_running(svc):
            print(f"  Stopping {svc} to free GPU...")
            docker_kill(svc)
            await asyncio.sleep(2)

    print(f"\n  Models to profile ({len(models)}):")
    for m in models:
        print(f"    • {m}")
    print()

    succeeded = 0
    failed = 0

    for i, model in enumerate(models, 1):
        print(f"\n{CYAN}[{i}/{len(models)}]{NC}")
        ok = await profile_model(model, state)
        if ok:
            succeeded += 1
        else:
            failed += 1

    # Final summary
    print(f"\n\n{BOLD}{'═' * 60}{NC}")
    print(f"{BOLD}  Profiling Complete{NC}")
    print(f"{BOLD}{'═' * 60}{NC}")
    print(f"  {GREEN}Succeeded:{NC} {succeeded}")
    if failed:
        print(f"  {RED}Failed:{NC} {failed}")
    print(f"\n  View results: mm benchmark compare")
    print(f"  Full data:    models.yaml")


if __name__ == "__main__":
    asyncio.run(main())
