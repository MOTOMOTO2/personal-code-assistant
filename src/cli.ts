#!/usr/bin/env node
import { cmdChat } from "./commands/chat.js";
import { cmdComplete } from "./commands/complete.js";
import { cmdCorpus } from "./commands/corpus.js";
import { cmdLearn } from "./commands/learn.js";
import { cmdDoctor, cmdNeighbors, cmdSearch, cmdStats } from "./commands/inspect.js";
import { cmdAsk, cmdExplain, cmdRefactor, cmdReview } from "./commands/llm.js";
import { cmdEval, cmdTrain } from "./commands/train.js";
import { cmdPhone, cmdServe } from "./commands/serve.js";
import { cmdThreads } from "./commands/threads.js";
import { DEFAULT_PORT } from "./server/daemon.js";
import { NotTrainedError } from "./core/bundle.js";
import { MissingApiKeyError } from "./llm/claude.js";
import { ParsedArgs, flagBool, parseArgs } from "./util/args.js";
import { color, fail, setQuiet } from "./util/log.js";

const VERSION = "0.1.0";

const HELP = `${color.bold("pca")} - personal code assistant ${color.dim(`v${VERSION}`)}

  Local models that learn your codebase, plus Claude for the hard questions.

${color.bold("Chat")} ${color.dim("(routes each turn: local models first, Claude when it matters)")}
  pca chat                       Conversation in the terminal
  pca chat "where is the retry logic?"   One question, then exit
  pca chat --resume              Continue the most recent conversation
  pca chat --thread <id>         Continue a particular one
  pca threads                    Every conversation, resumable from either side

${color.bold("Message it from your phone")}
  pca phone                      Serve the chat to this network, and print a QR code
  pca serve --lan                The same thing; --host/--token to be explicit
  pca serve --ui                 Browser chat, this machine only

${color.bold("Train")}
  pca train                      Train the fast models (tokenizer, n-gram, index, embeddings)
  pca learn                      Retrain, folding in everything chat has remembered
  pca learn --watch              Keep retraining as you edit
  pca train --neural --minutes 10  Also train the transformer, for 10 minutes
  pca train --neural-only --resume Continue training the transformer only
  pca eval                       Measure quality on held-out files
  pca eval --curve               Measure whether more code would still help

${color.bold("Feed it")} ${color.dim("(what it learns from, beyond this project)")}
  pca corpus                     Where the corpus comes from, broken down
  pca corpus scan ~/src          Nearby projects, and what each would add
  pca corpus add ../other-repo   Train on another directory too
  pca corpus add-all ~/src       Train on every repo there that matches your languages
  pca corpus remove other-repo   Stop training on it

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
  pca serve [--port ${DEFAULT_PORT}] [--ui] [--lan] [--token T]  Warm daemon for completion and chat

${color.bold("Common flags")}
  --neural        Include the transformer in scoring (slower, more context)
  --json          Machine-readable output
  --effort LEVEL  low|medium|high|xhigh|max for Claude commands
  --think         Stream a summary of Claude's reasoning
  --local         Keep a chat turn on the local models
  --lan           Serve to this network (implies --ui, requires a token)
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
  corpus: cmdCorpus,
  complete: cmdComplete,
  search: cmdSearch,
  neighbors: cmdNeighbors,
  neighbours: cmdNeighbors,
  stats: cmdStats,
  threads: cmdThreads,
  doctor: cmdDoctor,
  ask: cmdAsk,
  explain: cmdExplain,
  review: cmdReview,
  refactor: cmdRefactor,
  serve: cmdServe,
  phone: cmdPhone,
};

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
