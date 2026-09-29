import fs from "node:fs";
import path from "node:path";
import { ALWAYS_SKIP_DIRS, SourceFile, readSource, walkSources } from "../index/walker.js";
import { Workspace, ensureStateDir, statePath } from "../util/paths.js";
import { warn } from "../util/log.js";

/**
 * Extra corpora to train on, beyond the project you are standing in.
 *
 * The models here learn from *tokens of code*. That is the only thing they can
 * learn from, and it is worth being blunt about the consequences:
 *
 *   - A video tutorial is not training data. Transcribed, an hour of talking is
 *     ~10k tokens of English prose *about* code, which teaches a next-token
 *     model over code approximately nothing. This repository alone is ~170k
 *     tokens of the real thing.
 *   - Prose documentation is weak data for completion and decent data for
 *     retrieval, which is why `.md` is indexed but the style profile ignores it.
 *   - Other people's code, in the languages and frameworks you actually use, is
 *     strong data: it is the same shape as what you are about to type.
 *
 * So the way to make the local half better is to point it at more code you
 * resemble. That is what a source is: a directory elsewhere on disk, walked by
 * the same rules as the project, folded into the corpus and the index, and
 * labelled `@name/...` so a hit from someone else's repository can never be
 * mistaken for a hit from yours.
 */

/** Observed ratio for source code at this vocabulary size. */
const BYTES_PER_TOKEN = 3.5;

/** Default per-source budget. Big enough to matter, small enough to stay fast. */
export const DEFAULT_SOURCE_MAX_TOKENS = 400_000;

export interface CorpusSource {
  /** Short handle. Becomes the `@name/` prefix on every path from this source. */
  name: string;
  /** Absolute path to the directory on disk. */
  path: string;
  addedAt: string;
  /** Token budget for this source. Files are taken in path order until spent. */
  maxTokens: number;
  /** Extra file extensions, as for `pca train --ext`. */
  ext?: string[];
  /** Include in the retrieval index as well as the language model. */
  index: boolean;
  /** Free-text reminder of why this source is here. */
  note?: string;
}

export interface SourceDoc {
  /** The source this came from. */
  source: string;
  /** Display path, always `@name/relative/path.ts`. */
  rel: string;
  abs: string;
  text: string;
  size: number;
  mtimeMs: number;
  /** Whether this document should be indexed for retrieval. */
  index: boolean;
}

export interface SourceStats {
  name: string;
  files: number;
  bytes: number;
  /** True when the byte budget stopped the read early. */
  truncated: boolean;
  missing: boolean;
}

/**
 * Registry file. Deliberately *not* in `ARTIFACTS`: everything there is a
 * derived model that a version bump is allowed to throw away, and this is
 * hand-editable configuration that must survive one.
 */
function registryPath(ws: Workspace): string {
  return statePath(ws, "sources.json");
}

interface Registry {
  version: 1;
  sources: CorpusSource[];
}

export function loadSources(ws: Workspace): CorpusSource[] {
  const file = registryPath(ws);
  if (!fs.existsSync(file)) return [];
  try {
    const data = JSON.parse(fs.readFileSync(file, "utf8")) as Registry;
    if (!Array.isArray(data.sources)) return [];
    return data.sources.map((s) => ({
      ...s,
      maxTokens: s.maxTokens ?? DEFAULT_SOURCE_MAX_TOKENS,
      index: s.index !== false,
    }));
  } catch (err) {
    warn(`Ignoring unreadable ${file}: ${(err as Error).message}`);
    return [];
  }
}

export function saveSources(ws: Workspace, sources: CorpusSource[]): void {
  ensureStateDir(ws);
  const data: Registry = { version: 1, sources };
  fs.writeFileSync(registryPath(ws), `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

/** Slug a directory name into a handle usable as a path prefix. */
export function sourceName(dir: string): string {
  const base = path.basename(path.resolve(dir));
  const slug = base
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug.length > 0 ? slug : "source";
}

export interface AddSourceOptions {
  name?: string;
  maxTokens?: number;
  ext?: string[];
  index?: boolean;
  note?: string;
}

/**
 * Register a directory as a training source. Rejects the cases that would
 * quietly corrupt the corpus rather than fail loudly: a path that is not a
 * directory, one that is already registered, and one that lives inside the
 * project, where it is already being walked and would be counted twice.
 */
export function addSource(ws: Workspace, dir: string, opts: AddSourceOptions = {}): CorpusSource {
  const abs = path.resolve(dir);
  if (!fs.existsSync(abs)) throw new Error(`No such directory: ${abs}`);
  if (!fs.statSync(abs).isDirectory()) throw new Error(`Not a directory: ${abs}`);

  const root = path.resolve(ws.root);
  const rel = path.relative(root, abs);
  if (rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel))) {
    throw new Error(
      `${abs} is inside this project, so it is already part of the corpus. ` +
        "Sources are for code that lives somewhere else.",
    );
  }

  const sources = loadSources(ws);
  if (opts.name && sources.some((s) => s.name === opts.name)) {
    throw new Error(
      `A source named "${opts.name}" already exists. Pick another --name, or remove it first.`,
    );
  }
  if (sources.some((s) => path.resolve(s.path) === abs)) {
    throw new Error(`${abs} is already registered.`);
  }

  const source: CorpusSource = {
    name: opts.name ?? uniqueSourceName(sourceName(abs), new Set(sources.map((s) => s.name))),
    path: abs,
    addedAt: new Date().toISOString(),
    maxTokens: opts.maxTokens ?? DEFAULT_SOURCE_MAX_TOKENS,
    index: opts.index !== false,
    ...(opts.ext && opts.ext.length > 0 ? { ext: opts.ext } : {}),
    ...(opts.note ? { note: opts.note } : {}),
  };
  sources.push(source);
  saveSources(ws, sources);
  return source;
}

/** `base`, or `base-2`, `base-3`... until it collides with nothing in `taken`. */
export function uniqueSourceName(base: string, taken: Set<string>): string {
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) {
    const candidate = `${base}-${i}`;
    if (!taken.has(candidate)) return candidate;
  }
}

export function removeSource(ws: Workspace, name: string): boolean {
  const sources = loadSources(ws);
  const next = sources.filter((s) => s.name !== name);
  if (next.length === sources.length) return false;
  saveSources(ws, next);
  return true;
}

/** Files a source would contribute, before its token budget is applied. */
export function scanSource(source: CorpusSource): SourceFile[] {
  if (!fs.existsSync(source.path)) return [];
  return walkSources(source.path, { extraExtensions: source.ext });
}

/**
 * Read every registered source, honouring each one's token budget.
 *
 * The budget is enforced in bytes at read time rather than in tokens after
 * encoding, because the tokenizer is trained on this text: capping afterwards
 * would mean a forty-repository corpus still paid full price for vocabulary
 * induction. Files are taken in sorted path order, so the truncation point is
 * stable between runs and the corpus does not shuffle underneath you.
 */
export function readSources(
  sources: CorpusSource[],
  onProgress?: (msg: string) => void,
): { docs: SourceDoc[]; stats: SourceStats[] } {
  const docs: SourceDoc[] = [];
  const stats: SourceStats[] = [];

  for (const source of sources) {
    if (!fs.existsSync(source.path)) {
      warn(`Source "${source.name}" is gone from disk (${source.path}); skipping.`);
      stats.push({ name: source.name, files: 0, bytes: 0, truncated: false, missing: true });
      continue;
    }
    const budget = Math.round(source.maxTokens * BYTES_PER_TOKEN);
    const files = scanSource(source);
    let spent = 0;
    let count = 0;
    let truncated = false;
    for (const file of files) {
      if (spent + file.size > budget) {
        truncated = true;
        break;
      }
      let text: string;
      try {
        text = readSource(file);
      } catch {
        continue;
      }
      docs.push({
        source: source.name,
        rel: `@${source.name}/${file.rel}`,
        abs: file.abs,
        text,
        size: file.size,
        mtimeMs: file.mtimeMs,
        index: source.index,
      });
      spent += file.size;
      count++;
      if ((count & 63) === 0) onProgress?.(`${source.name}: ${count} files`);
    }
    stats.push({ name: source.name, files: count, bytes: spent, truncated, missing: false });
  }

  return { docs, stats };
}

/** Estimated token count for a byte count, for pre-train reporting. */
export function estimateTokens(bytes: number): number {
  return Math.round(bytes / BYTES_PER_TOKEN);
}

/** Markers that make a directory a project in its own right. */
const PROJECT_MARKERS = [".git", "package.json", "pyproject.toml", "go.mod", "Cargo.toml"];

export interface DiscoveredProject {
  path: string;
  /** The marker that identified it, for the "why is this here" column. */
  marker: string;
  files: number;
  bytes: number;
  /** Already registered as a source, or the project we are standing in. */
  known: boolean;
}

/**
 * Look for other projects near `dir` and report what each would contribute.
 *
 * "Add more code" is useless advice if you do not know what you have. This
 * walks a directory tree looking for project markers, measures each candidate
 * with the same walker that training uses - so the token estimate is the real
 * one, not a guess from `du` - and registers nothing. Deciding stays manual:
 * a source that does not look like code you write makes the model worse, and
 * no heuristic here can judge that.
 */
export function discoverProjects(
  ws: Workspace,
  dir: string,
  depth = 2,
  onProgress?: (msg: string) => void,
): DiscoveredProject[] {
  const root = path.resolve(dir);
  if (!fs.existsSync(root)) throw new Error(`No such directory: ${root}`);
  const registered = new Set(loadSources(ws).map((s) => path.resolve(s.path)));
  const self = path.resolve(ws.root);

  const found: DiscoveredProject[] = [];
  const visit = (current: string, level: number): void => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    const marker = PROJECT_MARKERS.find((m) => entries.some((e) => e.name === m));
    if (marker) {
      onProgress?.(`measuring ${path.basename(current)}`);
      const files = walkSources(current);
      found.push({
        path: current,
        marker,
        files: files.length,
        bytes: files.reduce((a, f) => a + f.size, 0),
        known: registered.has(current) || current === self,
      });
      // A project's subdirectories are part of it, not separate sources.
      return;
    }
    if (level >= depth) return;
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.isSymbolicLink()) continue;
      if (entry.name.startsWith(".") || ALWAYS_SKIP_DIRS.has(entry.name)) continue;
      visit(path.join(current, entry.name), level + 1);
    }
  };
  visit(root, 0);

  return found.sort((a, b) => b.bytes - a.bytes);
}

// ---------------------------------------------------------------------------
// Bulk adoption: "point it at everything I have"
// ---------------------------------------------------------------------------

/**
 * A language that is under this share of your own code is not really a language
 * you write, so a candidate repository written in it earns only partial credit.
 * Five percent is low enough to count a genuine second language and high enough
 * to stop one stray `.md` file from making every prose repository look like a
 * match.
 */
const OWN_LANGUAGE_FLOOR = 0.05;

/** Default overlap a candidate needs before it is worth adding. */
export const DEFAULT_MIN_OVERLAP = 0.5;

/** Below this, a repository is too small to move any of the models. */
export const DEFAULT_MIN_TOKENS = 5_000;

/**
 * Default ceiling on everything adopted in one go. `trainProject` caps the
 * language-model corpus at 2M tokens, and your own code has to fit inside that
 * cap too, so borrowing more than this only slows training down for tokens that
 * get dropped.
 */
export const DEFAULT_ADOPTION_BUDGET = 1_500_000;

/** Registry size cap, so one `add-all ~` cannot leave forty sources behind. */
export const DEFAULT_MAX_SOURCES = 25;

/**
 * Measured fast-train throughput, used to project training time when no
 * previous train has recorded its own rate. Deliberately pessimistic: a plan
 * that under-promises is easier to live with than one that over-promises.
 */
export const FALLBACK_TOKENS_PER_SECOND = 45_000;

/** How a codebase's bytes are distributed across file types. */
export interface LanguageMix {
  bytes: number;
  /** Extension (lowercase, with dot) to bytes. */
  byExt: Map<string, number>;
}

export function languageMix(files: SourceFile[]): LanguageMix {
  const byExt = new Map<string, number>();
  let bytes = 0;
  for (const file of files) {
    const ext = path.extname(file.rel).toLowerCase();
    byExt.set(ext, (byExt.get(ext) ?? 0) + file.size);
    bytes += file.size;
  }
  return { bytes, byExt };
}

/**
 * The biggest slices of a mix, for a one-line "what is this written in".
 * Slices under one percent are dropped: a lone `package.json` in a TypeScript
 * repository is not a language it is written in.
 */
export function topLanguages(mix: LanguageMix, limit = 3): Array<{ ext: string; share: number }> {
  if (mix.bytes === 0) return [];
  return [...mix.byExt.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([ext, bytes]) => ({ ext, share: bytes / mix.bytes }))
    .filter((l) => l.share >= 0.01)
    .slice(0, limit);
}

/**
 * Share of `candidate`'s bytes written in languages that are meaningfully part
 * of `own`, from 0 (nothing in common) to 1 (entirely languages you write).
 *
 * The measure is deliberately asymmetric. The question being asked is not "are
 * these two repositories similar" but "is this code the same shape as what I am
 * about to type", and a 50k-line Rust project is a bad source for a TypeScript
 * developer even though the reverse would also be true. A symmetric measure
 * like cosine similarity would also punish a single-language library for being
 * narrower than your own polyglot repository, which is the wrong call: a pure
 * TypeScript library is excellent data for a TypeScript project.
 */
export function languageOverlap(candidate: LanguageMix, own: LanguageMix): number {
  if (candidate.bytes === 0 || own.bytes === 0) return 0;
  let matched = 0;
  for (const [ext, bytes] of candidate.byExt) {
    const ownShare = (own.byExt.get(ext) ?? 0) / own.bytes;
    if (ownShare <= 0) continue;
    const credit = Math.min(1, ownShare / OWN_LANGUAGE_FLOOR);
    matched += bytes * credit;
  }
  return matched / candidate.bytes;
}

export interface AdoptionOptions {
  /** How deep to look for projects. */
  depth?: number;
  minOverlap?: number;
  minTokens?: number;
  /** Total token budget shared across everything adopted. */
  budget?: number;
  /** Ceiling for any single source. */
  maxPerSource?: number;
  maxSources?: number;
  onProgress?: (msg: string) => void;
}

export interface AdoptionCandidate {
  path: string;
  /** Handle it would be registered under. Unique within the plan. */
  name: string;
  marker: string;
  files: number;
  bytes: number;
  /** Tokens available, before any budget is applied. */
  tokens: number;
  /** Tokens it would actually contribute. 0 when skipped. */
  budget: number;
  overlap: number;
  languages: Array<{ ext: string; share: number }>;
  /** null when it would be added; otherwise the reason it was not. */
  skip: string | null;
}

export interface AdoptionPlan {
  root: string;
  /** Everything found, accepted and skipped, ranked most-like-your-code first. */
  candidates: AdoptionCandidate[];
  add: AdoptionCandidate[];
  skipped: AdoptionCandidate[];
  /** Tokens the accepted sources would add to the corpus. */
  tokens: number;
  /** The language mix the overlap scores were measured against. */
  own: LanguageMix;
  budget: number;
}

/**
 * Decide which nearby projects are worth training on, and how much of each.
 *
 * `pca corpus add` answers "train on this one as well"; this answers the
 * question people actually arrive with, which is "train on everything I have".
 * The difference matters because "everything" is usually wrong: a repository in
 * a language you never write spends vocabulary on patterns you will never type
 * and makes completion measurably worse. So the plan filters on language
 * overlap rather than adding whatever it finds, and it explains every rejection,
 * so one you disagree with can still be added by hand.
 *
 * Nothing is registered here. `commitAdoption` does that, so the same plan can
 * be printed as a preview and then applied unchanged.
 */
export function planAdoption(
  ws: Workspace,
  dir: string,
  opts: AdoptionOptions = {},
): AdoptionPlan {
  const minOverlap = opts.minOverlap ?? DEFAULT_MIN_OVERLAP;
  const minTokens = opts.minTokens ?? DEFAULT_MIN_TOKENS;
  const budget = opts.budget ?? DEFAULT_ADOPTION_BUDGET;
  const maxPerSource = opts.maxPerSource ?? DEFAULT_SOURCE_MAX_TOKENS;
  const maxSources = opts.maxSources ?? DEFAULT_MAX_SOURCES;

  const own = languageMix(walkSources(ws.root));
  const found = discoverProjects(ws, dir, opts.depth ?? 2, opts.onProgress);

  const taken = new Set(loadSources(ws).map((s) => s.name));
  const candidates: AdoptionCandidate[] = found.map((project) => {
    opts.onProgress?.(`weighing ${path.basename(project.path)}`);
    const mix = languageMix(walkSources(project.path));
    const overlap = languageOverlap(mix, own);
    const tokens = estimateTokens(project.bytes);
    return {
      path: project.path,
      name: sourceName(project.path),
      marker: project.marker,
      files: project.files,
      bytes: project.bytes,
      tokens,
      budget: 0,
      overlap,
      languages: topLanguages(mix),
      skip: skipReason(project.known, project.files, tokens, minTokens, overlap, minOverlap),
    };
  });

  // Most like your code first, biggest as the tie-break: when the budget or the
  // source cap runs out, what gets dropped should be the least useful thing.
  candidates.sort((a, b) => b.overlap - a.overlap || b.bytes - a.bytes);

  const eligible = candidates.filter((c) => c.skip === null);
  for (const c of eligible.slice(maxSources)) {
    c.skip = `over the ${maxSources}-source limit (--max-sources)`;
  }
  const accepted = eligible.slice(0, maxSources);

  allocate(accepted, budget, maxPerSource);
  for (const c of accepted) {
    if (c.budget < minTokens) {
      // The share left for it is too small to matter; spend nothing rather than
      // registering a source that contributes a rounding error.
      c.skip = "no budget left (raise --budget)";
      c.budget = 0;
    }
  }

  // Names must be unique in the registry, and the preview has to show the name
  // that will actually be used, so they are resolved here rather than at add
  // time.
  const add = accepted.filter((c) => c.skip === null);
  for (const c of add) {
    c.name = uniqueSourceName(c.name, taken);
    taken.add(c.name);
  }

  return {
    root: path.resolve(dir),
    candidates,
    add,
    skipped: candidates.filter((c) => c.skip !== null),
    tokens: add.reduce((a, c) => a + c.budget, 0),
    own,
    budget,
  };
}

function skipReason(
  known: boolean,
  files: number,
  tokens: number,
  minTokens: number,
  overlap: number,
  minOverlap: number,
): string | null {
  if (known) return "already in the corpus";
  if (files === 0) return "no indexable source files";
  if (tokens < minTokens) return `only ~${tokens.toLocaleString()} tokens (--min-tokens)`;
  if (overlap < minOverlap) {
    return `${(overlap * 100).toFixed(0)}% of it is in languages you write (--min-overlap)`;
  }
  return null;
}

/**
 * Share `budget` across `accepted` by water-filling: hand out the budget in
 * equal shares, smallest repository first, and let whatever a small one cannot
 * use flow to the larger ones.
 *
 * The effect is breadth before depth. Twenty repositories against a 1.5M-token
 * budget each contribute their first ~75k tokens rather than three of them
 * contributing everything, which is the right default when the point is to see
 * a wide range of code you resemble. `--budget` buys depth.
 */
function allocate(accepted: AdoptionCandidate[], budget: number, maxPerSource: number): void {
  const order = [...accepted].sort((a, b) => a.tokens - b.tokens);
  let remaining = budget;
  for (let i = 0; i < order.length; i++) {
    const fairShare = remaining / (order.length - i);
    const give = Math.floor(Math.min(order[i].tokens, maxPerSource, fairShare));
    order[i].budget = Math.max(0, give);
    remaining -= order[i].budget;
  }
}

export interface AdoptionResult {
  added: CorpusSource[];
  /** Candidates that failed at registration time, with the reason. */
  failed: Array<{ candidate: AdoptionCandidate; error: string }>;
}

/**
 * Headroom for a repository the budget did not have to cap. Plan figures are
 * estimated from bytes at `BYTES_PER_TOKEN`; dense code tokenizes above that
 * ratio, and without a margin a repository that was meant to fit entirely would
 * lose its last file to "budget reached", which reads like a cap you asked for.
 */
const ESTIMATE_MARGIN = 1.2;

/**
 * Register everything a plan accepted. Individual failures are collected rather
 * than thrown: a directory that disappeared between planning and committing
 * should not cost you the other nineteen.
 */
export function commitAdoption(ws: Workspace, plan: AdoptionPlan, note?: string): AdoptionResult {
  const added: CorpusSource[] = [];
  const failed: AdoptionResult["failed"] = [];
  for (const candidate of plan.add) {
    const capped = candidate.budget < candidate.tokens;
    try {
      added.push(
        addSource(ws, candidate.path, {
          name: candidate.name,
          maxTokens: capped ? candidate.budget : Math.ceil(candidate.budget * ESTIMATE_MARGIN),
          ...(note ? { note } : {}),
        }),
      );
    } catch (err) {
      failed.push({ candidate, error: (err as Error).message });
    }
  }
  return { added, failed };
}

/** Seconds a train is expected to take for a corpus of this size. */
export function projectTrainSeconds(tokens: number, tokensPerSecond?: number): number {
  const rate =
    tokensPerSecond && tokensPerSecond > 0 ? tokensPerSecond : FALLBACK_TOKENS_PER_SECOND;
  return tokens / rate;
}
