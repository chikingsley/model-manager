/**
 * Model Manager TUI - App Component
 */
import { useKeyboard } from "@opentui/solid";
import { createSignal, createEffect, onMount, onCleanup, For, Show } from "solid-js";
import {
  getStatus,
  activateModeWithProgress,
  listModels,
  listOllamaModels,
  isOllamaRunning,
  getOllamaModel,
  switchOllamaModel,
  type MmStatus,
  type Mode,
  type ActivationProgress
} from "./lib/mm";
import { recordTiming, getExpectedTime, formatProgress } from "./lib/timing";

// Mode options for the selector
export const MODES: { id: Mode; name: string; description: string }[] = [
  { id: "voice", name: "Voice Stack", description: "nemotron (ASR + LLM + TTS)" },
  { id: "llama", name: "llama.cpp", description: "GGUF models" },
  { id: "ollama", name: "Ollama", description: "GLM-OCR, Ministral" },
  { id: "ocr", name: "OCR", description: "LightOn-1B" },
  { id: "chat", name: "Chat", description: "vLLM Qwen 7B" },
  { id: "perf", name: "Performance", description: "vLLM optimized" },
  { id: "stop", name: "Stop All", description: "Free GPU" },
];

type View = "modes" | "models" | "ollama-models";

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export function GpuMeter(props: { used: number; total: number; percent: number; utilization: number; temperature: number }) {
  const barWidth = 20;
  const filled = Math.round((props.percent / 100) * barWidth);
  const bar = "█".repeat(filled) + "░".repeat(barWidth - filled);

  const memColor = props.percent > 90 ? "red" : props.percent > 70 ? "yellow" : "green";
  const tempColor = props.temperature > 80 ? "red" : props.temperature > 65 ? "yellow" : "green";

  return (
    <box flexDirection="row">
      <text fg={memColor}>{bar}</text>
      <text fg="gray"> {props.used}/{props.total}MB </text>
      <text fg={tempColor}>{props.temperature}°C</text>
    </box>
  );
}

export function ServiceItem(props: { name: string; model?: string; endpoint?: string }) {
  return (
    <box flexDirection="row" paddingLeft={1}>
      <text fg="green">● </text>
      <text><b>{props.name}</b></text>
      <Show when={props.model}>
        <text fg="gray"> ({props.model})</text>
      </Show>
      <Show when={props.endpoint}>
        <text fg="cyan"> → {props.endpoint}</text>
      </Show>
    </box>
  );
}

export default function App() {
  const [status, setStatus] = createSignal<MmStatus | null>(null);
  const [selectedIndex, setSelectedIndex] = createSignal(0);
  const [loading, setLoading] = createSignal(false);
  const [progress, setProgress] = createSignal<ActivationProgress & { mode: string; expected: number | null } | null>(null);
  const [message, setMessage] = createSignal("");
  const [models, setModels] = createSignal<string[]>([]);
  const [ollamaModels, setOllamaModels] = createSignal<string[]>([]);
  const [ollamaRunning, setOllamaRunning] = createSignal(false);
  const [currentOllamaModel, setCurrentOllamaModel] = createSignal<string | undefined>();
  const [view, setView] = createSignal<View>("modes");
  const [modelIndex, setModelIndex] = createSignal(0);
  const [ollamaIndex, setOllamaIndex] = createSignal(0);
  const [spinnerFrame, setSpinnerFrame] = createSignal(0);

  // Spinner animation
  createEffect(() => {
    if (loading()) {
      const interval = setInterval(() => {
        setSpinnerFrame(f => (f + 1) % SPINNER_FRAMES.length);
      }, 80);
      onCleanup(() => clearInterval(interval));
    }
  });

  // Fetch status on mount
  onMount(async () => {
    const [s, m, ollamaUp] = await Promise.all([getStatus(), listModels(), isOllamaRunning()]);
    setStatus(s);
    setModels(m);
    setOllamaRunning(ollamaUp);

    if (ollamaUp) {
      const [oModels, currentModel] = await Promise.all([listOllamaModels(), getOllamaModel()]);
      setOllamaModels(oModels);
      setCurrentOllamaModel(currentModel);
    }
  });

  // Refresh status every second
  createEffect(() => {
    const interval = setInterval(async () => {
      try {
        const [s, ollamaUp] = await Promise.all([getStatus(), isOllamaRunning()]);
        setStatus(s);
        setOllamaRunning(ollamaUp);

        if (ollamaUp) {
          const currentModel = await getOllamaModel();
          setCurrentOllamaModel(currentModel);
        }
      } catch {}
    }, 1000);
    onCleanup(() => clearInterval(interval));
  });

  // Keyboard handling
  useKeyboard((key) => {
    if (key.name === "q" && !loading()) {
      process.exit(0);
    }
    if (loading()) return;

    const currentView = view();

    if (currentView === "modes") {
      if (key.name === "up" || key.name === "k") {
        setSelectedIndex((i) => Math.max(0, i - 1));
      } else if (key.name === "down" || key.name === "j") {
        setSelectedIndex((i) => Math.min(MODES.length - 1, i + 1));
      } else if (key.name === "return") {
        const mode = MODES[selectedIndex()];
        if (mode.id === "llama" && models().length > 0) {
          setView("models");
          setModelIndex(0);
        } else if (mode.id === "ollama" && ollamaRunning()) {
          setView("ollama-models");
          setOllamaIndex(0);
        } else {
          handleActivate(mode.id);
        }
      }
    } else if (currentView === "models") {
      if (key.name === "up" || key.name === "k") {
        setModelIndex((i) => Math.max(0, i - 1));
      } else if (key.name === "down" || key.name === "j") {
        setModelIndex((i) => Math.min(models().length - 1, i + 1));
      } else if (key.name === "return") {
        handleActivate("llama", models()[modelIndex()]);
        setView("modes");
      } else if (key.name === "escape") {
        setView("modes");
      }
    } else if (currentView === "ollama-models") {
      if (key.name === "up" || key.name === "k") {
        setOllamaIndex((i) => Math.max(0, i - 1));
      } else if (key.name === "down" || key.name === "j") {
        setOllamaIndex((i) => Math.min(ollamaModels().length - 1, i + 1));
      } else if (key.name === "return") {
        handleOllamaSwitch(ollamaModels()[ollamaIndex()]);
        setView("modes");
      } else if (key.name === "escape") {
        setView("modes");
      }
    }
  });

  const handleActivate = async (mode: Mode, model?: string) => {
    const startTime = Date.now();
    const expected = getExpectedTime(mode);

    setLoading(true);
    setProgress({ mode, step: "Starting...", elapsed: 0, expected });
    setMessage("");

    try {
      const result = await activateModeWithProgress(mode, model, (p) => {
        setProgress({ mode, ...p, expected });
      });

      const elapsed = Math.round((Date.now() - startTime) / 1000);
      recordTiming(mode, elapsed);

      if (result.success) {
        setMessage(`✓ ${mode} ready`);
      } else {
        setMessage(`✗ ${result.message}`);
      }
    } catch (e: any) {
      setMessage(`✗ ${e.message}`);
    }

    setLoading(false);
    setProgress(null);

    const s = await getStatus();
    setStatus(s);
  };

  const handleOllamaSwitch = async (model: string) => {
    setLoading(true);
    setProgress({ mode: "ollama", step: `Loading ${model}...`, elapsed: 0, expected: null });
    setMessage("");

    const startTime = Date.now();

    try {
      const result = await switchOllamaModel(model);
      const elapsed = Math.round((Date.now() - startTime) / 1000);

      if (result.success) {
        setMessage(`✓ ${model} loaded (${elapsed}s)`);
        setCurrentOllamaModel(model);
      } else {
        setMessage(`✗ ${result.output}`);
      }
    } catch (e: any) {
      setMessage(`✗ ${e.message}`);
    }

    setLoading(false);
    setProgress(null);
  };

  return (
    <box flexDirection="column" padding={1} borderStyle="rounded" borderColor="cyan" border>
      {/* Header with GPU */}
      <box flexDirection="row" justifyContent="space-between">
        <text fg="cyan"><b>mm</b></text>
        <Show when={status()}>
          <GpuMeter
            used={status()!.gpu.used}
            total={status()!.gpu.total}
            percent={status()!.gpu.percent}
            utilization={status()!.gpu.utilization}
            temperature={status()!.gpu.temperature}
          />
        </Show>
      </box>

      <text fg="gray">────────────────────────────────────────────────</text>

      {/* Status message at top when present */}
      <Show when={message() && !loading()}>
        <text fg={message().startsWith("✓") ? "green" : "red"}>{message()}</text>
        <text />
      </Show>

      {/* Loading progress */}
      <Show when={loading() && progress()}>
        <box flexDirection="column">
          <box flexDirection="row">
            <text fg="yellow">{SPINNER_FRAMES[spinnerFrame()]} </text>
            <text fg="white"><b>{progress()!.mode}</b></text>
            <text fg="gray"> • </text>
            <text fg="cyan">{progress()!.step}</text>
          </box>
          <box flexDirection="row" paddingLeft={2}>
            <text fg="yellow">{formatProgress(progress()!.elapsed, progress()!.expected)}</text>
          </box>
          <Show when={progress()!.health && progress()!.health !== "not_running"}>
            <box flexDirection="row" paddingLeft={2}>
              <text fg="gray">health: </text>
              <text fg={progress()!.health === "healthy" ? "green" : progress()!.health === "starting" ? "yellow" : "red"}>
                {progress()!.health}
              </text>
            </box>
          </Show>
        </box>
        <text />
      </Show>

      {/* Mode Selector */}
      <Show when={view() === "modes" && !loading()}>
        <For each={MODES}>
          {(mode, i) => (
            <box flexDirection="row">
              <text fg={i() === selectedIndex() ? "cyan" : "gray"}>
                {i() === selectedIndex() ? "▸ " : "  "}
              </text>
              <text fg={status()?.active === mode.id ? "green" : i() === selectedIndex() ? "white" : "gray"}>
                {mode.name}
              </text>
              <text fg="gray"> - {mode.description}</text>
            </box>
          )}
        </For>
      </Show>

      {/* Model Picker (llama.cpp) */}
      <Show when={view() === "models" && !loading()}>
        <text fg="cyan">Select model:</text>
        <For each={models()}>
          {(model, i) => (
            <box flexDirection="row">
              <text fg={i() === modelIndex() ? "cyan" : "gray"}>
                {i() === modelIndex() ? "▸ " : "  "}
              </text>
              <text fg={i() === modelIndex() ? "white" : "gray"}>{model}</text>
            </box>
          )}
        </For>
      </Show>

      {/* Ollama Model Picker */}
      <Show when={view() === "ollama-models" && !loading()}>
        <text fg="cyan">Select Ollama model:</text>
        <Show when={currentOllamaModel()}>
          <text fg="gray">  current: {currentOllamaModel()}</text>
        </Show>
        <For each={ollamaModels()}>
          {(model, i) => (
            <box flexDirection="row">
              <text fg={i() === ollamaIndex() ? "cyan" : "gray"}>
                {i() === ollamaIndex() ? "▸ " : "  "}
              </text>
              <text fg={model === currentOllamaModel() ? "green" : i() === ollamaIndex() ? "white" : "gray"}>
                {model}
              </text>
            </box>
          )}
        </For>
      </Show>

      <text />

      {/* Running Services - compact */}
      <Show when={status()?.services && status()!.services.length > 0}>
        <For each={status()!.services}>
          {(svc) => <ServiceItem name={svc.name} model={svc.model} endpoint={svc.endpoint} />}
        </For>
      </Show>

      {/* Ollama status */}
      <Show when={ollamaRunning()}>
        <box flexDirection="row" paddingLeft={1}>
          <text fg="green">● </text>
          <text><b>ollama</b></text>
          <Show when={currentOllamaModel()}>
            <text fg="gray"> ({currentOllamaModel()})</text>
          </Show>
          <text fg="cyan"> → https://ollama.peacockery.studio</text>
        </box>
      </Show>

      {/* Tunnels inline */}
      <Show when={status()?.tunnels && status()!.tunnels.length > 0}>
        <box flexDirection="row" paddingLeft={1}>
          <text fg="blue">⚡ </text>
          <text fg="gray">tunnels: {status()!.tunnels.join(", ")}</text>
        </box>
      </Show>

      {/* Footer - help */}
      <text />
      <text fg="gray">
        {view() === "modes" ? "↑↓ select • enter activate • q quit" : "↑↓ select • enter confirm • esc back"}
      </text>
    </box>
  );
}
