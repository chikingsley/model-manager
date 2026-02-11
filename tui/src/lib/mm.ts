/**
 * Model Manager API Client
 *
 * Communicates with the Python API server via HTTP.
 * Falls back to shell commands if API is not running.
 */

const API_BASE = "http://localhost:8888";

export interface GpuInfo {
  used: number;
  total: number;
  utilization: number;
  percent: number;
  temperature: number;
}

export interface ServiceStatus {
  name: string;
  running: boolean;
  healthy: string;
  model?: string;
  port?: number;
  endpoint?: string;
}

export interface MmStatus {
  active: string;
  gpu: GpuInfo;
  services: ServiceStatus[];
  tunnels: string[];
  ollamaModel?: string;
}

export type Mode = "voice" | "llama" | "ocr" | "chat" | "perf" | "embed" | "ollama" | "stop";

export interface ActivationProgress {
  step: string;
  elapsed: number;
  health?: string;
}

export interface ActivationResult {
  success: boolean;
  mode: string;
  message: string;
  details?: Record<string, unknown>;
}

// ─────────────────────────────────────────────────────────────────────────────
// API Health Check
// ─────────────────────────────────────────────────────────────────────────────

let apiAvailable: boolean | null = null;

async function checkApiAvailable(): Promise<boolean> {
  if (apiAvailable !== null) return apiAvailable;

  try {
    const response = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(1000),
    });
    apiAvailable = response.ok;
  } catch {
    apiAvailable = false;
  }
  return apiAvailable;
}

// ─────────────────────────────────────────────────────────────────────────────
// API-based Functions
// ─────────────────────────────────────────────────────────────────────────────

async function fetchStatus(): Promise<MmStatus | null> {
  try {
    const response = await fetch(`${API_BASE}/status`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) return null;

    const data = await response.json();

    return {
      active: data.active,
      gpu: {
        used: Math.round(data.resources.vram.used_gb * 1024),
        total: Math.round(data.resources.vram.total_gb * 1024),
        utilization: data.resources.gpu_util_percent,
        percent: data.resources.vram.percent,
        temperature: data.resources.gpu_temperature,
      },
      services: data.services.map((s: any) => ({
        name: s.name,
        running: s.running,
        healthy: s.healthy,
        model: s.model,
        port: s.port,
        endpoint: s.endpoint,
      })),
      tunnels: data.tunnels,
      ollamaModel: data.ollama_model,
    };
  } catch {
    return null;
  }
}

async function apiActivate(mode: Mode, model?: string): Promise<ActivationResult> {
  try {
    const response = await fetch(`${API_BASE}/activate/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
      signal: AbortSignal.timeout(300000), // 5 min timeout for model loading
    });

    const data = await response.json();
    return {
      success: data.success,
      mode: data.mode,
      message: data.message,
      details: data.details,
    };
  } catch (e: any) {
    return {
      success: false,
      mode,
      message: e.message || "API request failed",
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shell Fallbacks (when API not running)
// ─────────────────────────────────────────────────────────────────────────────

import { $ } from "bun";

const MM_CLI = "uv run mm";

async function shellGetGpuInfo(): Promise<GpuInfo> {
  try {
    const result =
      await $`nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits`.text();
    const [used, total, util, temp] = result
      .trim()
      .split(",")
      .map((s) => parseInt(s.trim()));
    return {
      used,
      total,
      utilization: util,
      percent: Math.round((used / total) * 100),
      temperature: temp,
    };
  } catch {
    return { used: 0, total: 12227, utilization: 0, percent: 0, temperature: 0 };
  }
}

async function shellIsRunning(name: string): Promise<boolean> {
  try {
    const result = await $`docker ps --format '{{.Names}}' | grep -q "^${name}$"`.quiet();
    return result.exitCode === 0;
  } catch {
    return false;
  }
}

async function shellGetActiveState(): Promise<string> {
  try {
    const result =
      await $`grep "active:" /home/simon/github/model-manager/models.yaml | awk '{print $2}'`.text();
    return result.trim() || "none";
  } catch {
    return "none";
  }
}

async function shellGetStatus(): Promise<MmStatus> {
  const [
    gpu,
    active,
    nemotronRunning,
    vllmRunning,
    llamaRunning,
    ollamaRunning,
    voiceTunnel,
    vllmTunnel,
    llamaTunnel,
    ollamaTunnel,
  ] = await Promise.all([
    shellGetGpuInfo(),
    shellGetActiveState(),
    shellIsRunning("nemotron"),
    shellIsRunning("vllm"),
    shellIsRunning("llama-server"),
    shellIsRunning("ollama"),
    shellIsRunning("voice-tunnel"),
    shellIsRunning("vllm-tunnel"),
    shellIsRunning("llama-tunnel"),
    shellIsRunning("ollama-tunnel"),
  ]);

  const services: ServiceStatus[] = [];

  if (nemotronRunning) {
    services.push({
      name: "nemotron",
      running: true,
      healthy: "healthy",
      port: 18000,
      endpoint: "https://llm-voice.peacockery.studio",
    });
  }

  if (vllmRunning) {
    services.push({
      name: "vllm",
      running: true,
      healthy: "healthy",
      port: 8000,
      endpoint: "https://vllm.peacockery.studio",
    });
  }

  if (llamaRunning) {
    services.push({
      name: "llama-server",
      running: true,
      healthy: "healthy",
      port: 8090,
      endpoint: "https://llama.peacockery.studio",
    });
  }

  if (ollamaRunning) {
    services.push({
      name: "ollama",
      running: true,
      healthy: "healthy",
      port: 11434,
      endpoint: "https://ollama.peacockery.studio",
    });
  }

  const tunnels: string[] = [];
  if (voiceTunnel) tunnels.push("voice");
  if (vllmTunnel) tunnels.push("vllm");
  if (llamaTunnel) tunnels.push("llama");
  if (ollamaTunnel) tunnels.push("ollama");

  return { active, gpu, services, tunnels };
}

async function shellActivate(mode: Mode, model?: string): Promise<ActivationResult> {
  try {
    const args = model ? `${mode} ${model}` : mode;
    const result = await $`uv run mm ${args}`.cwd("/home/simon/github/model-manager").text();
    return { success: true, mode, message: result };
  } catch (e: any) {
    return { success: false, mode, message: e.message || "Failed to activate mode" };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exported Functions (use API when available, fallback to shell)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get current status
 */
export async function getStatus(): Promise<MmStatus> {
  if (await checkApiAvailable()) {
    const status = await fetchStatus();
    if (status) return status;
  }
  return shellGetStatus();
}

/**
 * Get GPU info
 */
export async function getGpuInfo(): Promise<GpuInfo> {
  if (await checkApiAvailable()) {
    const status = await fetchStatus();
    if (status) return status.gpu;
  }
  return shellGetGpuInfo();
}

/**
 * Activate a mode with progress updates
 */
export async function activateModeWithProgress(
  mode: Mode,
  model: string | undefined,
  onProgress: (progress: ActivationProgress) => void
): Promise<ActivationResult> {
  const startTime = Date.now();

  // Start progress polling
  let done = false;
  const pollProgress = async () => {
    while (!done) {
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      onProgress({ step: "Working...", elapsed });
      await new Promise((r) => setTimeout(r, 500));
    }
  };

  // Start polling in background
  pollProgress();

  try {
    let result: ActivationResult;

    if (await checkApiAvailable()) {
      result = await apiActivate(mode, model);
    } else {
      result = await shellActivate(mode, model);
    }

    done = true;
    return result;
  } catch (e: any) {
    done = true;
    return { success: false, mode, message: e.message || "Failed" };
  }
}

/**
 * Activate a mode (simple, no progress)
 */
export async function activateMode(
  mode: Mode,
  model?: string
): Promise<ActivationResult> {
  if (await checkApiAvailable()) {
    return apiActivate(mode, model);
  }
  return shellActivate(mode, model);
}

/**
 * List available GGUF models
 */
export async function listModels(): Promise<string[]> {
  if (await checkApiAvailable()) {
    try {
      const response = await fetch(`${API_BASE}/models/gguf`);
      if (response.ok) {
        return await response.json();
      }
    } catch {}
  }

  // Fallback to shell
  try {
    const result =
      await $`ls -1 /home/simon/models/*.gguf 2>/dev/null | xargs -I{} basename {}`.text();
    return result
      .trim()
      .split("\n")
      .filter(Boolean);
  } catch {
    return [];
  }
}

/**
 * Default context size for models that haven't been tested yet.
 * The API will auto-test and find the optimal context for this hardware.
 */
const DEFAULT_CONTEXT = 32768;

/**
 * List available Ollama models
 */
export async function listOllamaModels(): Promise<string[]> {
  if (await checkApiAvailable()) {
    try {
      const response = await fetch(`${API_BASE}/ollama/models`);
      if (response.ok) {
        const data = await response.json();
        return data.available || [];
      }
    } catch {}
  }

  // Fallback to direct Ollama API
  try {
    const response = await fetch("http://localhost:11434/api/tags");
    if (!response.ok) return [];
    const data = await response.json();
    return data.models?.map((m: any) => m.name) ?? [];
  } catch {
    return [];
  }
}

/**
 * Check if Ollama is running
 */
export async function isOllamaRunning(): Promise<boolean> {
  try {
    const response = await fetch("http://localhost:11434/api/tags", {
      signal: AbortSignal.timeout(2000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get current Ollama model (if any loaded)
 */
export async function getOllamaModel(): Promise<string | undefined> {
  try {
    const response = await fetch("http://localhost:11434/api/ps");
    if (!response.ok) return undefined;
    const data = await response.json();
    return data.models?.[0]?.name;
  } catch {
    return undefined;
  }
}

/**
 * Switch Ollama model
 */
export async function switchOllamaModel(
  model: string
): Promise<{ success: boolean; output: string }> {
  if (await checkApiAvailable()) {
    try {
      const response = await fetch(`${API_BASE}/ollama/load/${encodeURIComponent(model)}`, {
        method: "POST",
      });
      if (response.ok) {
        const data = await response.json();
        return { success: true, output: `Loaded ${model} with ${data.num_ctx} context` };
      } else {
        const error = await response.text();
        return { success: false, output: error };
      }
    } catch (e: any) {
      return { success: false, output: e.message };
    }
  }

  // Fallback to direct Ollama API (use default context - API would auto-test)
  try {
    const response = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt: "test",
        stream: false,
        keep_alive: -1,
        options: { num_ctx: DEFAULT_CONTEXT, num_predict: 1 },
      }),
    });

    if (response.ok) {
      return { success: true, output: `Loaded ${model} with ${DEFAULT_CONTEXT} context (use API for auto-tested context)` };
    } else {
      const error = await response.text();
      return { success: false, output: error };
    }
  } catch (e: any) {
    return { success: false, output: e.message };
  }
}
