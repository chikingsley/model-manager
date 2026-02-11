/**
 * Track historical timing for mode activations
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";

const CONFIG_DIR = join(homedir(), ".config", "mm-tui");
const TIMING_FILE = join(CONFIG_DIR, "timing.json");

interface TimingData {
  [mode: string]: number[]; // Array of completion times in seconds
}

function ensureConfigDir() {
  if (!existsSync(CONFIG_DIR)) {
    mkdirSync(CONFIG_DIR, { recursive: true });
  }
}

function loadTiming(): TimingData {
  try {
    if (existsSync(TIMING_FILE)) {
      return JSON.parse(readFileSync(TIMING_FILE, "utf-8"));
    }
  } catch {}
  return {};
}

function saveTiming(data: TimingData) {
  ensureConfigDir();
  writeFileSync(TIMING_FILE, JSON.stringify(data, null, 2));
}

/**
 * Record a completed activation time
 */
export function recordTiming(mode: string, seconds: number) {
  const data = loadTiming();
  if (!data[mode]) {
    data[mode] = [];
  }
  // Keep last 10 times
  data[mode].push(seconds);
  if (data[mode].length > 10) {
    data[mode] = data[mode].slice(-10);
  }
  saveTiming(data);
}

/**
 * Get expected time for a mode (median of past runs)
 */
export function getExpectedTime(mode: string): number | null {
  const data = loadTiming();
  const times = data[mode];
  if (!times || times.length === 0) {
    return null;
  }
  // Return median
  const sorted = [...times].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
}

/**
 * Format time with optional expected time
 */
export function formatProgress(elapsed: number, expected: number | null): string {
  const formatTime = (secs: number) => {
    if (secs < 60) return `${secs}s`;
    return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  };

  if (expected === null) {
    return formatTime(elapsed);
  }

  const pct = Math.round((elapsed / expected) * 100);
  const over = elapsed > expected;

  if (over) {
    return `${formatTime(elapsed)} (${pct}% - taking longer than usual)`;
  }
  return `${formatTime(elapsed)} / ~${formatTime(expected)}`;
}
