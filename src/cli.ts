#!/usr/bin/env node
import { cmdChat } from "./commands/chat.js";
import { cmdComplete } from "./commands/complete.js";
import { cmdLearn } from "./commands/learn.js";
import { cmdDoctor, cmdNeighbors, cmdSearch, cmdStats } from "./commands/inspect.js";
import { cmdAsk, cmdExplain, cmdRefactor, cmdReview } from "./commands/llm.js";
import { cmdEval, cmdTrain } from "./commands/train.js";
import { DEFAULT_PORT, createDaemon } from "./server/daemon.js";
import { NotTrainedError } from "./core/bundle.js";
import { MissingApiKeyError } from "./llm/claude.js";
import { ParsedArgs, flagBool, flagNumber, flagString, parseArgs } from "./util/args.js";
import { color, fail, setQuiet } from "./util/log.js";
import { workspace } from "./util/paths.js";

const VERSION = "0.1.0";

const HELP = `${color.bold("pca")} - personal code assistant ${color.dim(`v${VERSION}`)}

  Local models that learn your codebase, plus Claude for the hard questions.

${color.bold("Chat")} ${color.dim("(routes each turn: local models first, Claude when it matters)")}
  pca chat                       Conversation in the terminal
  pca chat "where is the retry logic?"   One question, then exit
  pca serve --ui                 Same chat in the browser

${color.bold("Train")}
  pca train                      Train the fast models (tokenizer, n-gram, index, embeddings)
  pca learn                      Retrain, folding in everything chat has remembered
  pca learn --watch              Keep retraining as you edit
  pca train --neural --minutes 10  Also train the transformer, for 10 minutes
  pca train --neural-only --resume Continue training the transformer only
  pca eval                       Measure quality on held-out files

${color.bold("Use locally")} ${color.dim("(no network, no API key)")}
  pca complete --file src/a.ts --line 42 [--col 18]
  pca complete "const user = await db."
  pca search "where do we retry uploads"
  pca neighbors userId           Nearest tokens in the learned embedding space
  pca stats [--weights] [--router] [--memory]
  pca doctor                     Check the setup

${color.bold("Ask Claude")} ${color.dim("(needs ANTHROPIC_API_KEY)")}
  pca ask "why does the token refresh run twice?"
  pca explain src/auth.ts [question]
  pca review [--staged] [--base main]
  pca refactor src/auth.ts "extract the retry logic" [--write]

${color.bold("Editor integration")}
  pca serve [--port ${DEFAULT_PORT}] [--ui]  Warm daemon for completion and chat

${color.bold("Common flags")}
  --neural        Include the transformer in scoring (slower, more context)
  --json          Machine-readable output
  --effort LEVEL  low|medium|high|xhigh|max for Claude commands
  --think         Stream a summary of Claude's reasoning
  --local         Keep a chat turn on the local models
  --quiet         Suppress status output
  --help          Show this

${color.dim("Everything trained is stored in .pca/ in your project. Add it to .gitignore.")}
`;

type Handler = (args: ParsedArgs) => Promise<number>;

const COMMANDS: Record<string, Handler> = {
  chat: cmdChat,
  train: cmdTrain,
  learn: cmdLearn,
  eval: cmdEval,
  complete: cmdComplete,
  search: cmdSearch,
  neighbors: cmdNeighbors,
  neighbours: cmdNeighbors,
  stats: cmdStats,
  doctor: cmdDoctor,
  ask: cmdAsk,
  explain: cmdExplain,
  review: cmdReview,
  refactor: cmdRefactor,
  serve: cmdServe,
};

async function cmdServe(args: ParsedArgs): Promise<number> {
  const ws = workspace(flagString(args, "root"));
  const daemon = createDaemon(ws, {
    port: flagNumber(args, "port", DEFAULT_PORT),
    idleTimeoutMs: flagNumber(args, "idle-timeout", 0),
    ui: flagBool(args, "ui"),
  });
  await daemon.listen();
  // Resolve only when the server closes, so the process stays up.
  await new Promise<void>((resolve) => {
    daemon.server.once("close", resolve);
    const stop = () => {
      void daemon.close();
    };
    process.once("SIGINT", stop);
    process.once("SIGTERM", stop);
  });
  return 0;
}

async function main(): Promise<number> {
  const argv = process.argv.slice(2);
  const name = argv[0];

  if (!name || name === "help" || name === "--help" || name === "-h") {
    process.stdout.write(HELP);
    return name ? 0 : 1;
  }
  if (name === "--version" || name === "-v" || name === "version") {
    process.stdout.write(`${VERSION}\n`);
    return 0;
  }

  const handler = COMMANDS[name];
  if (!handler) {
    fail(`Unknown command "${name}". Run \`pca help\` for the list.`);
    const near = Object.keys(COMMANDS).filter((c) => c.startsWith(name[0]));
    if (near.length > 0) process.stderr.write(`Did you mean: ${near.join(", ")}?\n`);
    return 1;
  }

  const args = parseArgs(argv.slice(1));
  if (flagBool(args, "help")) {
    process.stdout.write(HELP);
    return 0;
  }
  if (flagBool(args, "quiet")) setQuiet(true);

  return handler(args);
}

main()
  .then((code) => {
    process.exitCode = code;
  })
  .catch((err: unknown) => {
    if (err instanceof NotTrainedError || err instanceof MissingApiKeyError) {
      // These are expected setup states, not crashes - no stack trace.
      fail(err.message);
    } else if (err instanceof Error) {
      fail(err.message);
      if (process.env.PCA_DEBUG) process.stderr.write(`${err.stack}\n`);
      else process.stderr.write(color.dim("Set PCA_DEBUG=1 for a stack trace.\n"));
    } else {
      fail(String(err));
    }
    process.exitCode = 1;
  });
