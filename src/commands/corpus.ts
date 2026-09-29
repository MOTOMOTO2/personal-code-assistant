import path from "node:path";
import { Bundle } from "../core/bundle.js";
import {
  DEFAULT_SOURCE_MAX_TOKENS,
  addSource,
  estimateTokens,
  loadSources,
  removeSource,
  scanSource,
} from "../core/sources.js";
import { LESSON_PATH_PREFIX } from "../chat/memory.js";
import { ParsedArgs, flagBool, flagList, flagNumber, flagString } from "../util/args.js";
import { color, info, ok, warn } from "../util/log.js";
import { humanBytes } from "../util/io.js";
import { workspace } from "../util/paths.js";

const USAGE = `${color.bold("pca corpus")} - what the local models train on

  pca corpus                       Show the corpus: this project, memory, sources
  pca corpus add <dir> [options]   Train on another directory as well
  pca corpus remove <name>         Stop training on it
  pca corpus list                  Registered sources only

${color.bold("Options for add")}
  --name NAME        Handle for the source (default: the directory name)
  --max-tokens N     Budget for this source (default ${DEFAULT_SOURCE_MAX_TOKENS.toLocaleString()})
  --ext .astro,.erb  Extra file extensions to include
  --no-index         Train on it, but keep it out of search results
  --note "..."       Why you added it

  Changes take effect on the next \`pca train\` or \`pca learn\`.
`;

export async function cmdCorpus(args: ParsedArgs): Promise<number> {
  const sub = args.positional[0];
  switch (sub) {
    case undefined:
      return showCorpus();
    case "list":
      return listSources();
    case "add":
      return addCorpusSource(args);
    case "remove":
    case "rm":
      return removeCorpusSource(args);
    default:
      process.stdout.write(USAGE);
      warn(`Unknown subcommand "${sub}".`);
      return 1;
  }
}

/**
 * The whole corpus at a glance, broken down by where each part came from.
 *
 * This exists because "how do I train it" is really the question "what is it
 * currently learning from", and until now the only answer was a token count
 * with no provenance attached.
 */
function showCorpus(): number {
  const ws = workspace();
  const bundle = new Bundle(ws);
  const manifest = bundle.manifest;
  const index = bundle.index;

  info("");
  info(color.bold("Corpus"));

  if (!manifest) {
    info(color.dim("  Nothing trained yet. Run `pca train`."));
  } else {
    const total = manifest.corpusTokens;
    let external = 0;
    let lessons = 0;
    if (index) {
      for (const file of index.files) {
        if (file.external) external += file.tokens;
        else if (file.path.startsWith(LESSON_PATH_PREFIX)) lessons += file.tokens;
      }
    }
    const own = Math.max(0, total - external - lessons);
    row("your code", own, total);
    if (lessons > 0) row("remembered answers", lessons, total);
    if (external > 0) row("external sources", external, total);
    info(`  ${color.dim("total".padEnd(20))} ${total.toLocaleString().padStart(9)} tokens`);
    info(
      color.dim(
        `  trained ${new Date(manifest.trainedAt).toLocaleString()} · vocab ${manifest.vocabSize}`,
      ),
    );
  }

  const sources = loadSources(ws);
  info("");
  info(color.bold("Sources"));
  if (sources.length === 0) {
    info(color.dim("  none registered"));
    info("");
    info(color.dim("  Point it at more code you resemble:"));
    info(color.dim("    pca corpus add ../my-other-project"));
    info(color.dim("    pca corpus add ~/src/some-library --max-tokens 200000"));
  } else {
    printSources(sources);
  }
  info("");
  info(
    color.dim(
      "  Video, screen recordings and audio are not usable here: these models\n" +
        "  predict the next token of code. An hour of tutorial transcribes to\n" +
        "  ~10k tokens of English about code; the same hour spent pointing this\n" +
        "  at a repository you work in is hundreds of thousands of tokens of code.",
    ),
  );
  info("");
  return 0;
}

function row(label: string, tokens: number, total: number): void {
  const share = total > 0 ? `${((tokens / total) * 100).toFixed(0)}%` : "-";
  info(`  ${color.dim(label.padEnd(20))} ${tokens.toLocaleString().padStart(9)} tokens  ${color.dim(share)}`);
}

function listSources(): number {
  const ws = workspace();
  const sources = loadSources(ws);
  if (sources.length === 0) {
    info(color.dim("No sources registered. `pca corpus add <dir>` adds one."));
    return 0;
  }
  printSources(sources);
  return 0;
}

function printSources(sources: ReturnType<typeof loadSources>): void {
  for (const source of sources) {
    const files = scanSource(source);
    const bytes = files.reduce((a, f) => a + f.size, 0);
    const gone = files.length === 0;
    info(
      `  ${color.cyan(`@${source.name}`)}  ${color.dim(source.path)}` +
        (gone ? ` ${color.yellow("(unreadable or empty)")}` : ""),
    );
    const est = estimateTokens(bytes);
    const capped = est > source.maxTokens;
    info(
      `    ${color.dim("available")} ${files.length} files, ${humanBytes(bytes)} ` +
        `(~${est.toLocaleString()} tokens)`,
    );
    info(
      `    ${color.dim("budget")}    ${source.maxTokens.toLocaleString()} tokens` +
        (capped ? color.yellow("  ← capped, raise with --max-tokens on re-add") : ""),
    );
    if (!source.index) info(`    ${color.dim("search")}    excluded (--no-index)`);
    if (source.note) info(`    ${color.dim("note")}      ${source.note}`);
  }
}

function addCorpusSource(args: ParsedArgs): number {
  const dir = args.positional[1];
  if (!dir) {
    process.stdout.write(USAGE);
    throw new Error("Usage: pca corpus add <dir>");
  }
  const ws = workspace();
  const source = addSource(ws, dir, {
    ...(flagString(args, "name") ? { name: flagString(args, "name") } : {}),
    maxTokens: flagNumber(args, "max-tokens", DEFAULT_SOURCE_MAX_TOKENS),
    ext: flagList(args, "ext"),
    index: flagBool(args, "index", true),
    ...(flagString(args, "note") ? { note: flagString(args, "note") } : {}),
  });

  const files = scanSource(source);
  const bytes = files.reduce((a, f) => a + f.size, 0);
  ok(`Added @${source.name} → ${source.path}`);
  info(
    `  ${files.length} files, ${humanBytes(bytes)} (~${estimateTokens(bytes).toLocaleString()} tokens available, ` +
      `budget ${source.maxTokens.toLocaleString()})`,
  );
  if (files.length === 0) {
    warn("No indexable source files found there - check the path, or pass --ext for unusual extensions.");
  }
  info(color.dim("  Run `pca train` to fold it in, then `pca eval --curve` to see whether it helped."));
  return 0;
}

function removeCorpusSource(args: ParsedArgs): number {
  const name = args.positional[1];
  if (!name) throw new Error("Usage: pca corpus remove <name>");
  const ws = workspace();
  const bare = name.startsWith("@") ? name.slice(1) : name;
  if (!removeSource(ws, bare)) {
    const known = loadSources(ws).map((s) => `@${s.name}`).join(", ");
    throw new Error(
      `No source named "${bare}".` + (known ? ` Registered: ${known}` : " None are registered."),
    );
  }
  ok(`Removed @${bare}. Run \`pca train\` to drop it from the models.`);
  return 0;
}

/** Exported for the daemon and tests: a one-line corpus summary. */
export function corpusSummary(bundle: Bundle): string {
  const sources = loadSources(bundle.ws);
  const tokens = bundle.manifest?.corpusTokens ?? 0;
  const where = sources.length > 0 ? ` + ${sources.length} source${sources.length === 1 ? "" : "s"}` : "";
  return `${tokens.toLocaleString()} tokens${where}`;
}

/** Resolve a `@source/rel/path` display path back to a real file on disk. */
export function resolveDisplayPath(bundle: Bundle, display: string): string | null {
  if (!display.startsWith("@")) return path.join(bundle.ws.root, display);
  const slash = display.indexOf("/");
  if (slash < 0) return null;
  const name = display.slice(1, slash);
  const source = loadSources(bundle.ws).find((s) => s.name === name);
  return source ? path.join(source.path, display.slice(slash + 1)) : null;
}
