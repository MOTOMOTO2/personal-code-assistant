import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { Bundle } from "../core/bundle.js";
import { SearchHit } from "../index/store.js";
import { ClaudeClient, Effort, StreamResult } from "../llm/claude.js";
import {
  askPrompt,
  explainPrompt,
  refactorPrompt,
  retrievalQuery,
  reviewPrompt,
  systemPrompt,
} from "../llm/prompts.js";
import { normalize } from "../ml/embeddings.js";
import { ParsedArgs, flagBool, flagNumber, flagString } from "../util/args.js";
import { color, info, step, warn } from "../util/log.js";
import { relative, workspace } from "../util/paths.js";

const VALID_EFFORT = new Set(["low", "medium", "high", "xhigh", "max"]);

function effortOf(args: ParsedArgs, fallback: Effort): Effort {
  const raw = flagString(args, "effort");
  if (!raw) return fallback;
  if (!VALID_EFFORT.has(raw)) {
    throw new Error(`--effort must be one of: ${[...VALID_EFFORT].join(", ")}`);
  }
  return raw as Effort;
}

/** Retrieve grounding context from the local index. */
function retrieve(bundle: Bundle, query: string, limit: number): SearchHit[] {
  const index = bundle.index;
  if (!index) return [];
  const q = retrievalQuery(query);
  const embeddings = bundle.embeddings;
  const queryVector = embeddings
    ? normalize(embeddings.sifEmbed(bundle.tokenizer.encode(q)))
    : null;
  return index.search(q, { limit, queryVector });
}

/** Stream a Claude turn to stdout and print the usage footer. */
async function run(
  bundle: Bundle,
  userContent: string,
  opts: { effort: Effort; showThinking: boolean; maxTokens?: number },
): Promise<StreamResult> {
  const client = new ClaudeClient();
  let thinkingOpen = false;
  const result = await client.stream({
    system: systemPrompt(bundle.style),
    messages: [{ role: "user", content: userContent }],
    effort: opts.effort,
    showThinking: opts.showThinking,
    maxTokens: opts.maxTokens,
    onThinking: (delta) => {
      if (!thinkingOpen) {
        process.stderr.write(color.dim("\n[thinking] "));
        thinkingOpen = true;
      }
      process.stderr.write(color.dim(delta));
    },
    onText: (delta) => {
      if (thinkingOpen) {
        process.stderr.write("\n\n");
        thinkingOpen = false;
      }
      process.stdout.write(delta);
    },
  });
  process.stdout.write("\n");

  if (result.stopReason === "refusal") {
    warn(
      `Claude declined this request${result.refusal?.category ? ` (${result.refusal.category})` : ""}` +
        `${result.refusal?.explanation ? `: ${result.refusal.explanation}` : ""}`,
    );
  }
  const u = result.usage;
  info(
    color.dim(
      `\n${result.model} · in ${u.inputTokens.toLocaleString()} (cached ${u.cacheReadTokens.toLocaleString()}) · ` +
        `out ${u.outputTokens.toLocaleString()} · ~$${result.estimatedCostUsd.toFixed(4)}`,
    ),
  );
  return result;
}

export async function cmdAsk(args: ParsedArgs): Promise<number> {
  const ws = workspace();
  const bundle = new Bundle(ws);
  const question = [...args.positional, ...args.rest].join(" ");
  if (question.length === 0) {
    throw new Error('Usage: pca ask "why does the upload retry twice?"');
  }
  const limit = flagNumber(args, "context", 8);
  const hits = bundle.index ? retrieve(bundle, question, limit) : [];
  if (hits.length > 0) {
    step(
      `Grounding on ${hits.length} chunks: ${hits.slice(0, 4).map((h) => `${h.file}:${h.chunk.startLine}`).join(", ")}${hits.length > 4 ? ", …" : ""}`,
    );
  } else if (!bundle.index) {
    warn("No local index - answering without codebase context. Run `pca train` to fix.");
  }
  await run(bundle, askPrompt(question, hits), {
    effort: effortOf(args, "high"),
    showThinking: flagBool(args, "think"),
  });
  return 0;
}

export async function cmdExplain(args: ParsedArgs): Promise<number> {
  const ws = workspace();
  const bundle = new Bundle(ws);
  const target = args.positional[0];
  if (!target) throw new Error("Usage: pca explain <file> [question]");
  const abs = path.resolve(ws.root, target);
  if (!fs.existsSync(abs)) throw new Error(`No such file: ${target}`);
  const code = fs.readFileSync(abs, "utf8");
  const rel = relative(ws, abs);
  const question = args.positional.slice(1).join(" ") || undefined;

  const hits = bundle.index ? retrieve(bundle, question ?? `${rel} ${firstLines(code, 40)}`, 6) : [];
  await run(bundle, explainPrompt(rel, code, hits, question), {
    effort: effortOf(args, "high"),
    showThinking: flagBool(args, "think"),
  });
  return 0;
}

export async function cmdReview(args: ParsedArgs): Promise<number> {
  const ws = workspace();
  const bundle = new Bundle(ws);
  const diff = collectDiff(ws.root, args);
  if (diff.trim().length === 0) {
    info("No changes to review.");
    return 0;
  }
  const maxChars = flagNumber(args, "max-diff", 200_000);
  if (diff.length > maxChars) {
    throw new Error(
      `Diff is ${diff.length} characters, over the ${maxChars} limit. Review a subset with \`--base <ref>\` or raise --max-diff.`,
    );
  }
  step(`Reviewing ${diff.split("\n").length} diff lines`);
  // Ground the review in the files it touches, so Claude can see the callers.
  const touched = [...diff.matchAll(/^\+\+\+ b\/(.+)$/gm)].map((m) => m[1]).join(" ");
  const hits = bundle.index ? retrieve(bundle, touched || diff.slice(0, 2000), 8) : [];
  await run(bundle, reviewPrompt(diff, hits), {
    // Review is correctness-sensitive, so it gets more thinking by default.
    effort: effortOf(args, "xhigh"),
    showThinking: flagBool(args, "think"),
  });
  return 0;
}

export async function cmdRefactor(args: ParsedArgs): Promise<number> {
  const ws = workspace();
  const bundle = new Bundle(ws);
  const target = args.positional[0];
  const instruction = args.positional.slice(1).join(" ") || args.rest.join(" ");
  if (!target || instruction.length === 0) {
    throw new Error('Usage: pca refactor <file> "extract the retry logic into a helper"');
  }
  const abs = path.resolve(ws.root, target);
  if (!fs.existsSync(abs)) throw new Error(`No such file: ${target}`);
  const code = fs.readFileSync(abs, "utf8");
  const rel = relative(ws, abs);
  const hits = bundle.index ? retrieve(bundle, `${rel} ${instruction}`, 6) : [];

  const result = await run(bundle, refactorPrompt(rel, code, instruction, hits), {
    effort: effortOf(args, "xhigh"),
    showThinking: flagBool(args, "think"),
    maxTokens: 64_000,
  });

  if (flagBool(args, "write")) {
    const block = extractCodeBlock(result.text);
    if (!block) {
      warn("Could not find a single fenced code block in the response; nothing written.");
      return 1;
    }
    const backup = `${abs}.pca-bak`;
    fs.copyFileSync(abs, backup);
    fs.writeFileSync(abs, block, "utf8");
    info("");
    info(`${color.green("wrote")} ${rel}  ${color.dim(`(backup at ${path.basename(backup)})`)}`);
    info(color.dim("Review with `git diff` before committing."));
  } else {
    info(color.dim("\nPass --write to apply this to the file (a .pca-bak backup is kept)."));
  }
  return 0;
}

function collectDiff(root: string, args: ParsedArgs): string {
  const base = flagString(args, "base");
  const gitArgs = flagBool(args, "staged")
    ? ["diff", "--cached", "--no-color"]
    : base
      ? ["diff", "--no-color", `${base}...HEAD`]
      : ["diff", "--no-color", "HEAD"];
  try {
    return execFileSync("git", gitArgs, {
      cwd: root,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch (err) {
    const message = (err as { stderr?: string }).stderr ?? (err as Error).message;
    throw new Error(`git ${gitArgs.join(" ")} failed: ${String(message).trim()}`);
  }
}

/** Pull the first fenced block out of a response, tolerating a language tag. */
function extractCodeBlock(text: string): string | null {
  const match = /```[A-Za-z0-9_+-]*\n([\s\S]*?)```/.exec(text);
  if (!match) return null;
  const body = match[1];
  return body.endsWith("\n") ? body : `${body}\n`;
}

function firstLines(text: string, n: number): string {
  return text.split("\n").slice(0, n).join(" ");
}
