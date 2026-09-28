import readline from "node:readline";
import { Bundle } from "../core/bundle.js";
import { ChatSession, Route } from "../chat/session.js";
import { loadLessons } from "../chat/memory.js";
import { trainProject } from "../core/train.js";
import { Effort, hasCredentials } from "../llm/claude.js";
import { ParsedArgs, flagBool, flagString } from "../util/args.js";
import { color, info, warn } from "../util/log.js";
import { workspace } from "../util/paths.js";

/**
 * `pca chat` - the conversational front end.
 *
 * One prompt, two engines. The router picks per turn; `/claude` and `/local`
 * override it and, in doing so, teach it. Every Claude answer lands in memory,
 * so the same question later is answered locally for free.
 */

const HELP = `
  ${color.bold("Commands")}
    /claude [question]   Re-ask the last question (or a new one) with Claude
    /local [question]    Force the local models
    /why                 How the last turn was routed, and on what evidence
    /memory [query]      What the assistant remembers about this codebase
    /train               Fold new answers into the models and retrain
    /stats               Model sizes, router weights, session cost
    /new                 Start a fresh conversation
    /help                This list
    /exit                Leave
`;

const VALID_EFFORT = new Set(["low", "medium", "high", "xhigh", "max"]);

export async function cmdChat(args: ParsedArgs): Promise<number> {
  const ws = workspace(flagString(args, "root"));
  const bundle = new Bundle(ws);
  const session = new ChatSession(bundle, { persist: !flagBool(args, "no-save") });
  const effortFlag = flagString(args, "effort");
  if (effortFlag && !VALID_EFFORT.has(effortFlag)) {
    throw new Error(`--effort must be one of: ${[...VALID_EFFORT].join(", ")}`);
  }
  const opts = {
    effort: effortFlag as Effort | undefined,
    showThinking: flagBool(args, "think"),
    forceDefault: flagBool(args, "local") ? ("local" as Route) : null,
  };

  // `--local "question"` parses the question as the flag's value; treat it as
  // the question rather than making the user remember flag order.
  const localValue = args.flags.get("local");
  const oneShot = [
    ...args.positional,
    ...args.rest,
    ...(typeof localValue === "string" ? [localValue] : []),
  ]
    .join(" ")
    .trim();
  if (oneShot.length > 0) {
    await turn(session, oneShot, opts);
    return 0;
  }

  if (!process.stdin.isTTY) {
    // Piped input: one question per line, same routing, no prompt chrome.
    const text = await readAll();
    for (const line of text.split("\n")) {
      const q = line.trim();
      if (q.length === 0) continue;
      await turn(session, q, opts);
    }
    return 0;
  }

  banner(bundle, session);
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const prompt = () => {
    rl.setPrompt(`${color.cyan("you")} ${color.dim("›")} `);
    rl.prompt();
  };

  let done = false;
  rl.on("close", () => {
    done = true;
  });
  prompt();

  for await (const raw of rl) {
    const line = raw.trim();
    if (line.length === 0) {
      prompt();
      continue;
    }
    if (line === "/exit" || line === "/quit") break;
    try {
      const handled = await slashCommand(line, session, bundle, opts);
      if (!handled) await turn(session, line, opts);
    } catch (err) {
      warn((err as Error).message);
    }
    if (done) break;
    prompt();
  }
  rl.close();
  if (session.totalCostUsd > 0) {
    info(color.dim(`\nsession cost ~$${session.totalCostUsd.toFixed(4)}`));
  }
  return 0;
}

interface TurnOpts {
  effort?: Effort;
  showThinking?: boolean;
  forceDefault: Route | null;
}

async function turn(
  session: ChatSession,
  question: string,
  opts: TurnOpts,
  force?: Route,
): Promise<void> {
  let first = true;
  const result = await session.ask(question, {
    force: force ?? opts.forceDefault,
    effort: opts.effort,
    showThinking: opts.showThinking,
    onRoute: (route, _p, why) => {
      const tag = route === "claude" ? color.magenta("claude") : color.green("local");
      info(`\n${tag} ${color.dim(why)}`);
    },
    onThinking: (delta) => {
      if (first) {
        process.stderr.write(color.dim("[thinking] "));
        first = false;
      }
      process.stderr.write(color.dim(delta));
    },
    onText: (delta) => {
      if (!first) {
        process.stdout.write("\n\n");
        first = true;
      }
      process.stdout.write(delta);
    },
  });

  if (result.route === "local") {
    process.stdout.write(`${result.text}\n`);
  } else {
    process.stdout.write("\n");
    info(
      color.dim(
        `~$${result.costUsd.toFixed(4)}${result.lessonId ? " · remembered, `/train` to learn from it" : ""}`,
      ),
    );
  }
}

async function slashCommand(
  line: string,
  session: ChatSession,
  bundle: Bundle,
  opts: TurnOpts,
): Promise<boolean> {
  if (!line.startsWith("/")) return false;
  const [cmd, ...restParts] = line.split(/\s+/);
  const rest = restParts.join(" ").trim();

  switch (cmd) {
    case "/help":
      info(HELP);
      return true;

    case "/claude":
      if (!hasCredentials()) {
        warn("No ANTHROPIC_API_KEY set - the Claude half is unavailable.");
        return true;
      }
      if (rest.length > 0) await turn(session, rest, opts, "claude");
      else await escalate(session, opts);
      return true;

    case "/local":
      if (rest.length > 0) await turn(session, rest, opts, "local");
      else warn("Usage: /local <question>");
      return true;

    case "/why": {
      const last = session.turns[session.turns.length - 1];
      if (!last || last.role !== "assistant") {
        info("Nothing answered yet.");
        return true;
      }
      info(
        [
          `route        ${last.route}`,
          `kind         ${last.kind}`,
          `p(claude)    ${last.pClaude !== undefined ? last.pClaude.toFixed(3) : "n/a"}`,
          `sources      ${last.sources && last.sources.length > 0 ? last.sources.join(", ") : "none"}`,
          `cost         $${(last.costUsd ?? 0).toFixed(4)}`,
        ].join("\n"),
      );
      return true;
    }

    case "/memory": {
      const store = loadLessons(bundle);
      if (store.size === 0) {
        info("Memory is empty. Ask Claude something and the answer is kept here.");
        return true;
      }
      const shown =
        rest.length > 0
          ? store.recall(rest, { limit: 5, tokenizer: bundle.tokenizer, embeddings: bundle.embeddings })
          : store.lessons.slice(-5).map((lesson) => ({ lesson, score: 1, lexical: 1, semantic: 1 }));
      info(`${store.size} answers remembered${rest ? `, closest to "${rest}"` : ", most recent"}:`);
      for (const r of shown) {
        info(
          `  ${color.dim(r.lesson.at.slice(0, 10))} ${truncate(r.lesson.question, 60)} ${color.dim(
            `(${(r.score * 100).toFixed(0)}%)`,
          )}`,
        );
      }
      return true;
    }

    case "/train": {
      info("Retraining on your code plus everything in memory...");
      const report = trainProject(bundle, {});
      session.refreshMemory();
      info(
        color.dim(
          `${report.files} files, ${report.corpusTokens.toLocaleString()} tokens, ` +
            `${report.chunks.toLocaleString()} chunks, ${report.seconds.toFixed(1)}s`,
        ),
      );
      return true;
    }

    case "/stats": {
      const m = bundle.manifest;
      const router = bundle.router;
      info(
        [
          `corpus       ${m ? `${m.corpusTokens.toLocaleString()} tokens, ${m.corpusFiles} files` : "not trained"}`,
          `memory       ${session.memorySize} answers`,
          `router       ${router.updates} updates (${router.toLocal} local, ${router.toClaude} claude)`,
          `session      $${session.totalCostUsd.toFixed(4)}`,
        ].join("\n"),
      );
      for (const w of router.describe().slice(0, 5)) {
        info(color.dim(`  ${w.feature.padEnd(20)} ${w.weight.toFixed(3)}`));
      }
      return true;
    }

    case "/new":
      session.clear();
      info("New conversation.");
      return true;

    default:
      warn(`Unknown command ${cmd}. /help for the list.`);
      return true;
  }
}

async function escalate(session: ChatSession, opts: TurnOpts): Promise<void> {
  let first = true;
  const result = await session.escalate({
    effort: opts.effort,
    showThinking: opts.showThinking,
    onThinking: (delta) => {
      if (first) {
        process.stderr.write(color.dim("[thinking] "));
        first = false;
      }
      process.stderr.write(color.dim(delta));
    },
    onText: (delta) => {
      if (!first) {
        process.stdout.write("\n\n");
        first = true;
      }
      process.stdout.write(delta);
    },
  });
  process.stdout.write("\n");
  info(color.dim(`~$${result.costUsd.toFixed(4)} · the router learned from this`));
}

function banner(bundle: Bundle, session: ChatSession): void {
  const m = bundle.manifest;
  info(`${color.bold("pca chat")} ${color.dim("- local models first, Claude when it matters")}`);
  info(
    color.dim(
      `  ${m ? `${m.corpusTokens.toLocaleString()} tokens trained` : "not trained yet - run `pca train`"} · ` +
        `${session.memorySize} answers in memory · ` +
        `${hasCredentials() ? "Claude available" : "no API key, local only"}`,
    ),
  );
  info(color.dim("  /help for commands, /exit to leave\n"));
}

function truncate(text: string, n: number): string {
  const one = text.replace(/\s+/g, " ").trim();
  return one.length <= n ? one : `${one.slice(0, n - 1)}…`;
}

function readAll(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}
