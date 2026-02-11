/**
 * Integration tests for Model Manager TUI
 *
 * Uses OpenTUI's testRender for snapshot and interaction testing
 */
import { describe, test, expect, mock, beforeAll } from "bun:test";
import { testRender } from "@opentui/solid";

// Mock data
const mockStatus = {
  active: "llama",
  gpu: { used: 8000, total: 12227, utilization: 45, percent: 65, temperature: 52 },
  services: [
    { name: "llama-server", running: true, model: "test-model.gguf", endpoint: "https://test.example.com" }
  ],
  tunnels: ["voice", "llama"]
};

const mockModels = ["model-a.gguf", "model-b.gguf", "model-c.gguf"];

// Mock the mm module before importing App
beforeAll(() => {
  mock.module("./lib/mm", () => ({
    getStatus: async () => mockStatus,
    listModels: async () => mockModels,
    activateMode: async (mode: string, model?: string) => ({ success: true, output: `Activated ${mode}` }),
    activateModeWithProgress: async (mode: string, model: string | undefined, onProgress: any) => {
      onProgress({ step: "Testing...", elapsed: 1, health: "starting" });
      return { success: true, output: `Activated ${mode}` };
    },
    getContainerHealth: async () => "healthy",
  }));
});

// Dynamic import to ensure mocks are in place
const getApp = async () => {
  const module = await import("./app");
  return module.default;
};

describe("Model Manager TUI", () => {
  test("renders header and mode selector", async () => {
    const App = await getApp();
    const { renderOnce, captureCharFrame } = await testRender(() => <App />, {
      width: 80,
      height: 24,
    });

    await renderOnce();
    const frame = captureCharFrame();

    // Header
    expect(frame).toContain("mm");

    // Mode selector
    expect(frame).toContain("Voice Stack");
    expect(frame).toContain("llama.cpp");
    expect(frame).toContain("OCR");
    expect(frame).toContain("Chat");
    expect(frame).toContain("Performance");
    expect(frame).toContain("Stop All");

    // Help text
    expect(frame).toContain("select");
    expect(frame).toContain("quit");
  });

  test("shows GPU meter after data loads", async () => {
    const App = await getApp();
    const { renderOnce, captureCharFrame } = await testRender(() => <App />, {
      width: 80,
      height: 24,
    });

    // Initial render
    await renderOnce();

    // Wait for async onMount
    await new Promise(r => setTimeout(r, 50));
    await renderOnce();

    const frame = captureCharFrame();

    expect(frame).toContain("█"); // Progress bar filled
    expect(frame).toContain("░"); // Progress bar empty
    expect(frame).toContain("MB");
    expect(frame).toContain("°C");
  });

  test("shows services section", async () => {
    const App = await getApp();
    const { renderOnce, captureCharFrame } = await testRender(() => <App />, {
      width: 80,
      height: 24,
    });

    await renderOnce();
    await new Promise(r => setTimeout(r, 50));
    await renderOnce();

    const frame = captureCharFrame();

    // Services shown inline with green bullet
    expect(frame).toContain("●");
    expect(frame).toContain("llama-server");
  });

  test("shows tunnel status", async () => {
    const App = await getApp();
    const { renderOnce, captureCharFrame } = await testRender(() => <App />, {
      width: 80,
      height: 24,
    });

    await renderOnce();
    await new Promise(r => setTimeout(r, 50));
    await renderOnce();

    const frame = captureCharFrame();

    expect(frame).toContain("tunnels:");
  });

  test("navigates with arrow keys", async () => {
    const App = await getApp();
    const { renderOnce, captureCharFrame, mockInput } = await testRender(() => <App />, {
      width: 80,
      height: 24,
    });

    await renderOnce();

    // First item selected (Voice Stack has ▸)
    let frame = captureCharFrame();
    const lines = frame.split("\n");
    const voiceLine = lines.find(l => l.includes("Voice Stack"));
    expect(voiceLine).toContain("▸");

    // Press down
    mockInput.pressArrow("down");
    await renderOnce();

    frame = captureCharFrame();
    const llamaLine = frame.split("\n").find(l => l.includes("llama.cpp"));
    expect(llamaLine).toContain("▸");
  });

  test("selection indicator moves correctly", async () => {
    const App = await getApp();
    const { renderOnce, captureCharFrame, mockInput } = await testRender(() => <App />, {
      width: 80,
      height: 24,
    });

    await renderOnce();

    // Navigate down twice
    mockInput.pressArrow("down");
    await renderOnce();
    mockInput.pressArrow("down");
    await renderOnce();

    let frame = captureCharFrame();
    const ocrLine = frame.split("\n").find(l => l.includes("OCR"));
    expect(ocrLine).toContain("▸");

    // Navigate back up
    mockInput.pressArrow("up");
    await renderOnce();

    frame = captureCharFrame();
    const llamaLine = frame.split("\n").find(l => l.includes("llama.cpp"));
    expect(llamaLine).toContain("▸");
  });
});

describe("GpuMeter component", () => {
  test("renders progress bar with correct fill", async () => {
    const { GpuMeter } = await import("./app");
    const { renderOnce, captureCharFrame } = await testRender(
      () => <GpuMeter used={6000} total={12000} percent={50} utilization={30} temperature={45} />,
      { width: 60, height: 5 }
    );

    await renderOnce();
    const frame = captureCharFrame();

    // Compact GPU meter shows bar, memory, and temp
    expect(frame).toContain("█"); // Progress bar filled
    expect(frame).toContain("░"); // Progress bar empty
    expect(frame).toContain("6000/12000MB");
    expect(frame).toContain("45°C");
  });
});

describe("ServiceItem component", () => {
  test("renders service with model and endpoint", async () => {
    const { ServiceItem } = await import("./app");
    const { renderOnce, captureCharFrame } = await testRender(
      () => <ServiceItem name="test-service" model="test.gguf" endpoint="https://test.com" />,
      { width: 60, height: 5 }
    );

    await renderOnce();
    const frame = captureCharFrame();

    expect(frame).toContain("●");
    expect(frame).toContain("test-service");
    expect(frame).toContain("test.gguf");
    expect(frame).toContain("https://test.com");
  });

  test("renders service without optional fields", async () => {
    const { ServiceItem } = await import("./app");
    const { renderOnce, captureCharFrame } = await testRender(
      () => <ServiceItem name="minimal-service" />,
      { width: 60, height: 5 }
    );

    await renderOnce();
    const frame = captureCharFrame();

    expect(frame).toContain("minimal-service");
    expect(frame).not.toContain("undefined");
  });
});
