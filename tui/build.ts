import solidPlugin from "@opentui/solid/bun-plugin";
import { $ } from "bun";
import { copyFileSync } from "fs";

// Build with Solid plugin
await Bun.build({
  entrypoints: ["./src/index.tsx"],
  target: "bun",
  outdir: "./dist",
  plugins: [solidPlugin],
  minify: true,
});

console.log("Built to ./dist/index.js");

// Compile from /tmp to avoid bunfig.toml being baked in
const distPath = import.meta.dir + "/dist/index.js";
const tmpBinary = "/tmp/mm-tui-compiled";
const outPath = import.meta.dir + "/mm-tui";

await $`cd /tmp && bun build ${distPath} --compile --outfile ${tmpBinary}`.quiet();

// Copy back to project dir
copyFileSync(tmpBinary, outPath);

console.log("Compiled to ./mm-tui (107MB standalone binary)");
console.log("Install with: cp mm-tui ~/.local/bin/");
