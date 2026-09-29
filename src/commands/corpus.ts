import path from "node:path";
import { Bundle } from "../core/bundle.js";
import {
  AdoptionPlan,
  DEFAULT_ADOPTION_BUDGET,
  DEFAULT_MAX_SOURCES,
  DEFAULT_MIN_OVERLAP,
  DEFAULT_MIN_TOKENS,
  DEFAULT_SOURCE_MAX_TOKENS,
  addSource,
  commitAdoption,
  discoverProjects,
  estimateTokens,
  loadSources,
  planAdoption,
  projectTrainSeconds,
  removeSource,
  scanSource,
  topLanguages,
} from "../core/sources.js";
import { LESSON_PATH_PREFIX } from "../chat/memory.js";
import { ParsedArgs, flagBool, flagList, flagNumber, flagString } from "../util/args.js";
import { color, fmtDuration, info, ok, progress, progressDone, warn } from "../util/log.js";
import { humanBytes } from "../util/io.js";
import { workspace } from "../util/paths.js";

const USAGE = `${color.bold("pca corpus")} - what the local models train on

  pca corpus                       Show the corpus: this project, memory, sources
  pca corpus scan <dir>            Find nearby projects worth training on
  pca corpus add <dir> [options]   Train on another directory as well
  pca corpus add-all <dir>         Train on every repo under <dir> worth having
  pca corpus remove <name>         Stop training on it
  pca corpus list                  Registered sources only

${color.bold("Options for add")}
  --name NAME        Handle for the source (default: the directory name)
  --max-tokens N     Budget for this source (default ${DEFAULT_SOURCE_MAX_TOKENS.toLocaleString()})
  --ext .astro,.erb  Extra file extensions to include
  --no-index         Train on it, but keep it out of search results
  --note "..."       Why you added it

${color.bold("Options for add-all")}
  --dry-run          Show the plan and register nothing
  --depth N          How deep to look for projects (default 2)
  --min-overlap F    Reject repos less than this much in your languages (default ${DEFAULT_MIN_OVERLAP})
  --min-tokens N     Reject repos smaller than this (default ${DEFAULT_MIN_TOKENS.toLocaleString()})
  --budget N         Tokens to share across everything added (default ${DEFAULT_ADOPTION_BUDGET.toLocaleString()})
  --max-tokens N     Ceiling for any one repo (default ${DEFAULT_SOURCE_MAX_TOKENS.toLocaleString()})
  --max-sources N    Stop after this many (default ${DEFAULT_MAX_SOURCES})

  Changes take effect on the next \`pca train\` or \`pca learn\`.
`;

export async function cmdCorpus(args: ParsedArgs): Promise<number> {
  const sub = args.positional[0];
  switch (sub) {
    case undefined:
      return showCorpus();
    case "help":
      process.stdout.write(USAGE);
      return 0;
    case "list":
      return listSources();
    case "scan":
      return scanForProjects(args);
    case "add":
      return addCorpusSource(args);
    case "add-all":
    case "addall":
      return addAllCorpusSources(args);
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

/**
 * Show what is nearby and what each would add. Registers nothing: whether a
 * codebase resembles yours closely enough to help is a judgement call, and a
 * heuristic that made it for you would quietly make the model worse.
 */
function scanForProjects(args: ParsedArgs): number {
  const dir = args.positional[1] ?? "..";
  const ws = workspace();
  const found = discoverProjects(ws, dir, flagNumber(args, "depth", 2), (msg) => progress(msg));
  progressDone();

  if (found.length === 0) {
    info(color.dim(`No projects found under ${path.resolve(dir)}. Try --depth 3.`));
    return 0;
  }

  const current = bundleTokens();
  info("");
  info(color.bold(`Projects under ${path.resolve(dir)}`));
  for (const p of found) {
    const tokens = estimateTokens(p.bytes);
    const ratio = current > 0 ? tokens / current : 0;
    const share =
      current === 0 ? "" : ratio < 0.1 ? " (<0.1x this corpus)" : ` (${ratio.toFixed(1)}x this corpus)`;
    const tag = p.known ? color.dim("  already in the corpus") : "";
    info(
      `  ${color.cyan(path.basename(p.path).padEnd(24))} ${p.files.toString().padStart(5)} files  ` +
        `${humanBytes(p.bytes).padStart(7)}  ~${tokens.toLocaleString()} tokens${share}${tag}`,
    );
    if (!p.known) info(`    ${color.dim(`pca corpus add "${p.path}"`)}`);
  }
  info("");
  info(
    color.dim(
      "  Add the ones written in languages you use and in a style you would write.\n" +
        "  A codebase unlike yours spends vocabulary on patterns you will never type,\n" +
        "  which makes completion worse. `pca eval --curve` before and after tells you.\n" +
        `  To take every one that matches your languages: pca corpus add-all "${path.resolve(dir)}"`,
    ),
  );
  info("");
  return 0;
}

function bundleTokens(): number {
  try {
    return new Bundle(workspace()).manifest?.corpusTokens ?? 0;
  } catch {
    return 0;
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

/**
 * `pca corpus add-all <dir>` - every repository under a directory that is worth
 * training on, in one command.
 *
 * This is the honest version of "read all my code". Adding literally everything
 * is the wrong thing to do: a repository in a language you never write spends
 * vocabulary on patterns you will never type, and the completion models get
 * worse. So the plan filters on language overlap and prints why each rejection
 * happened, which is information you cannot get from a silent bulk add.
 */
function addAllCorpusSources(args: ParsedArgs): number {
  const dir = args.positional[1];
  if (!dir) {
    process.stdout.write(USAGE);
    throw new Error("Usage: pca corpus add-all <dir> [--dry-run]");
  }
  const dryRun = flagBool(args, "dry-run");
  const ws = workspace();

  const plan = planAdoption(ws, dir, {
    depth: flagNumber(args, "depth", 2),
    minOverlap: flagNumber(args, "min-overlap", DEFAULT_MIN_OVERLAP),
    minTokens: flagNumber(args, "min-tokens", DEFAULT_MIN_TOKENS),
    budget: flagNumber(args, "budget", DEFAULT_ADOPTION_BUDGET),
    maxPerSource: flagNumber(args, "max-tokens", DEFAULT_SOURCE_MAX_TOKENS),
    maxSources: flagNumber(args, "max-sources", DEFAULT_MAX_SOURCES),
    onProgress: (msg) => progress(msg),
  });
  progressDone();

  info("");
  info(`${color.bold("Looking under")} ${plan.root}`);
  info(color.dim(`  matching against your code: ${describeLanguages(topLanguages(plan.own))}`));

  if (plan.candidates.length === 0) {
    info("");
    info(color.dim("  No projects found there. Try --depth 3, or a directory further up."));
    info("");
    return 0;
  }

  printPlan(plan, dryRun);

  if (plan.add.length === 0) {
    info("");
    info(
      color.dim(
        "  Nothing to add. Lower --min-overlap to accept repos in other languages,\n" +
          "  or add one by hand with `pca corpus add <dir>`.",
      ),
    );
    info("");
    return 0;
  }

  if (dryRun) {
    info("");
    info(color.dim("  --dry-run: nothing was registered. Re-run without it to commit."));
    info("");
    return 0;
  }

  const result = commitAdoption(ws, plan, `added by corpus add-all from ${plan.root}`);
  for (const f of result.failed) info(`  ${color.yellow("!")} ${f.candidate.name}: ${f.error}`);
  info("");
  ok(
    `Registered ${result.added.length} source${result.added.length === 1 ? "" : "s"}. ` +
      "Run `pca train` to fold them in.",
  );
  info(
    color.dim(
      "  Then `pca eval --curve` tells you whether the extra code actually helped.\n" +
        `  Undo any one of them with \`pca corpus remove <name>\`.`,
    ),
  );
  info("");
  return 0;
}

function printPlan(plan: AdoptionPlan, dryRun: boolean): void {
  const verb = dryRun ? "Would add" : "Adding";
  info("");
  if (plan.add.length > 0) {
    info(
      `${color.bold(`${verb} ${plan.add.length}`)} ${color.dim(
        `of ${plan.candidates.length} projects · budget ${plan.budget.toLocaleString()} tokens`,
      )}`,
    );
    for (const c of plan.add) {
      const capped = c.budget < c.tokens ? color.dim(` of ~${c.tokens.toLocaleString()}`) : "";
      info(
        `  ${color.cyan(`@${c.name}`.padEnd(24))} ${pct(c.overlap).padStart(4)} match  ` +
          `${c.files.toString().padStart(5)} files  ` +
          `${c.budget.toLocaleString().padStart(9)} tokens${capped}`,
      );
      info(`    ${color.dim(c.path)}`);
      info(`    ${color.dim(describeLanguages(c.languages))}`);
    }
  }

  if (plan.skipped.length > 0) {
    info("");
    info(`${color.bold(`Skipped ${plan.skipped.length}`)}`);
    for (const c of plan.skipped) {
      info(
        `  ${color.dim(shortName(c.path).padEnd(24))} ${c.skip}` +
          (c.languages.length > 0 ? color.dim(`  [${describeLanguages(c.languages)}]`) : ""),
      );
    }
  }

  // What this costs, in the only two units that matter before you commit: how
  // much bigger the corpus gets, and how long the next train takes.
  const bundle = new Bundle(workspace());
  const current = bundle.manifest?.corpusTokens ?? 0;
  const total = current + plan.tokens;
  const rate =
    bundle.manifest?.trainSeconds && current > 0
      ? current / bundle.manifest.trainSeconds
      : undefined;
  info("");
  info(
    `  ${color.dim("corpus")} ${current.toLocaleString()} → ${total.toLocaleString()} tokens` +
      (current > 0 ? color.dim(` (${(total / current).toFixed(1)}x)`) : "") +
      color.dim(` · next \`pca train\` ~${fmtDuration(projectTrainSeconds(total, rate) * 1000)}`),
  );
}

function describeLanguages(langs: Array<{ ext: string; share: number }>): string {
  if (langs.length === 0) return "no source files";
  return langs.map((l) => `${l.ext || "(none)"} ${pct(l.share)}`).join(", ");
}

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

function shortName(dir: string): string {
  const base = path.basename(dir);
  return base.length <= 24 ? base : `${base.slice(0, 23)}…`;
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
