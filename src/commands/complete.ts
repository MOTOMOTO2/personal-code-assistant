import fs from "node:fs";
import path from "node:path";
import { Bundle } from "../core/bundle.js";
import { CompletionEngine } from "../core/complete.js";
import { ParsedArgs, flagBool, flagNumber, flagString } from "../util/args.js";
import { color, info } from "../util/log.js";
import { relative, workspace } from "../util/paths.js";

/**
 * `pca complete` - one-shot completion from the CLI.
 *
 * Three ways to say where the cursor is:
 *   --file F --line N [--col C]   read the file, cursor at that position
 *   --stdin                       read the prefix from stdin (cursor at the end)
 *   "<prefix text>"               positional argument
 */
export async function cmdComplete(args: ParsedArgs): Promise<number> {
  const ws = workspace();
  const bundle = new Bundle(ws);
  if (!bundle.isTrained) {
    throw new Error("Nothing trained yet - run `pca train` first.");
  }

  const { prefix, suffix, filePath } = await resolveCursor(args, ws.root);
  if (prefix.length === 0) {
    throw new Error("Empty prefix - nothing to complete from.");
  }

  const engine = new CompletionEngine(bundle);
  // Load the models before timing, so the reported latency is inference only
  // (the daemon pays this cost once at startup).
  const loadStart = performance.now();
  engine.warm({ neural: flagBool(args, "neural"), retrieval: flagBool(args, "retrieval", true) });
  const loadMs = performance.now() - loadStart;

  const result = engine.complete({
    prefix,
    suffix,
    filePath,
    neural: flagBool(args, "neural"),
    retrieval: flagBool(args, "retrieval", true),
    count: flagNumber(args, "count", 3),
    maxTokens: flagNumber(args, "max-tokens", 12),
    beamWidth: flagNumber(args, "beam", 5),
    singleLine: flagBool(args, "single-line", true),
  });

  if (flagBool(args, "json")) {
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    return 0;
  }

  if (result.suggestions.length === 0) {
    info(color.dim("no suggestions"));
    return 0;
  }

  // Show the tail of the prefix so the cursor position is obvious.
  const lastLine = prefix.slice(prefix.lastIndexOf("\n") + 1);
  info("");
  info(`  ${color.dim(truncateStart(lastLine, 70))}${color.dim("▏")}`);
  info("");
  result.suggestions.forEach((s, i) => {
    const marker = i === 0 ? color.green("▸") : color.dim(" ");
    const shown = s.text.replace(/\n/g, color.dim("⏎"));
    const neural = s.neuralLogProb === null ? "" : ` ${color.dim(`nn ${s.neuralLogProb.toFixed(2)}`)}`;
    info(
      `  ${marker} ${color.bold(shown)}  ${color.dim(`score ${s.score.toFixed(3)}`)} ${color.dim(`lm ${s.ngramLogProb.toFixed(2)}`)}${neural}`,
    );
  });
  info("");
  info(
    color.dim(
      `  ${result.timing.total}ms inference (propose ${result.timing.propose}ms · ` +
        `neural ${result.timing.neural}ms · rank ${result.timing.rank}ms) · ` +
        `${loadMs.toFixed(0)}ms model load, paid once per process` +
        (result.partial ? ` · completing "${result.partial}"` : ""),
    ),
  );
  if (!result.usedNeural && flagBool(args, "neural")) {
    info(color.dim("  (no neural checkpoint - run `pca train --neural`)"));
  }
  info("");
  return 0;
}

async function resolveCursor(
  args: ParsedArgs,
  root: string,
): Promise<{ prefix: string; suffix: string; filePath: string | undefined }> {
  const file = flagString(args, "file");
  if (file) {
    const abs = path.resolve(root, file);
    const text = fs.readFileSync(abs, "utf8");
    const lines = text.split("\n");
    const line = flagNumber(args, "line", lines.length);
    const lineIdx = Math.max(0, Math.min(lines.length - 1, line - 1));
    const col = args.flags.has("col") ? flagNumber(args, "col", 0) : lines[lineIdx].length;
    let offset = 0;
    for (let i = 0; i < lineIdx; i++) offset += lines[i].length + 1;
    offset += Math.max(0, Math.min(lines[lineIdx].length, col));
    return {
      prefix: text.slice(0, offset),
      suffix: text.slice(offset),
      filePath: relative({ root, stateDir: "" }, abs),
    };
  }

  if (flagBool(args, "stdin")) {
    const prefix = await readStdin();
    return { prefix, suffix: "", filePath: undefined };
  }

  const positional = args.positional.join(" ");
  if (positional.length > 0) {
    return { prefix: positional, suffix: "", filePath: undefined };
  }
  // No explicit source: fall back to stdin if it is piped.
  if (!process.stdin.isTTY) {
    return { prefix: await readStdin(), suffix: "", filePath: undefined };
  }
  throw new Error(
    'Give me a cursor: `pca complete --file src/app.ts --line 42`, `pca complete "const x = "`, or pipe text with --stdin.',
  );
}

function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    process.stdin.on("data", (c: Buffer) => chunks.push(c));
    process.stdin.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    process.stdin.on("error", reject);
  });
}

function truncateStart(s: string, max: number): string {
  return s.length <= max ? s : `…${s.slice(s.length - max + 1)}`;
}
