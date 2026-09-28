/**
 * Warm-path latency benchmark.
 *
 * Measures what the editor actually experiences: models already loaded, one
 * completion per keystroke. Run after `npm run build && node dist/cli.js train`:
 *
 *   node bench/latency.mjs [--neural] [--iterations 300]
 */
import { Bundle } from "../dist/core/bundle.js";
import { CompletionEngine } from "../dist/core/complete.js";
import { workspace } from "../dist/util/paths.js";
import { walkSources, readSource } from "../dist/index/walker.js";

const args = process.argv.slice(2);
const neural = args.includes("--neural");
const iterations = Number(args[args.indexOf("--iterations") + 1]) || 300;

const ws = workspace();
const bundle = new Bundle(ws);
if (!bundle.isTrained) {
  console.error("Not trained. Run `node dist/cli.js train` first.");
  process.exit(1);
}

const engine = new CompletionEngine(bundle);
const loadStart = performance.now();
engine.warm({ neural, retrieval: true });
const loadMs = performance.now() - loadStart;

// Build realistic cursor positions from actual project files.
const cursors = [];
for (const file of walkSources(ws.root)) {
  let text;
  try {
    text = readSource(file);
  } catch {
    continue;
  }
  const lines = text.split("\n");
  let offset = 0;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.length > 20) {
      const at = offset + line.length - Math.floor(trimmed.length * 0.4);
      cursors.push({ prefix: text.slice(0, at), suffix: text.slice(at), filePath: file.rel });
    }
    offset += line.length + 1;
  }
  if (cursors.length > iterations * 2) break;
}
if (cursors.length === 0) {
  console.error("No usable cursor positions found.");
  process.exit(1);
}

// Warm the JIT before measuring.
for (let i = 0; i < 30; i++) {
  engine.complete({ ...cursors[i % cursors.length], neural, retrieval: false, count: 3 });
}

const samples = [];
for (let i = 0; i < iterations; i++) {
  const cursor = cursors[(i * 7) % cursors.length];
  const t0 = performance.now();
  engine.complete({ ...cursor, neural, retrieval: false, count: 3, maxTokens: 10 });
  samples.push(performance.now() - t0);
}
samples.sort((a, b) => a - b);

const pick = (q) => samples[Math.min(samples.length - 1, Math.floor(samples.length * q))];
const mean = samples.reduce((a, b) => a + b, 0) / samples.length;

console.log("");
console.log(`  mode            ${neural ? "n-gram + transformer" : "n-gram only"}`);
console.log(`  model load      ${loadMs.toFixed(0)}ms (once per process)`);
console.log(`  completions     ${iterations}`);
console.log("");
console.log(`  mean            ${mean.toFixed(3)}ms`);
console.log(`  p50             ${pick(0.5).toFixed(3)}ms`);
console.log(`  p95             ${pick(0.95).toFixed(3)}ms`);
console.log(`  p99             ${pick(0.99).toFixed(3)}ms`);
console.log(`  max             ${samples[samples.length - 1].toFixed(3)}ms`);
console.log("");
