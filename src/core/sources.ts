import fs from "node:fs";
import path from "node:path";
import { SourceFile, readSource, walkSources } from "../index/walker.js";
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
    name: opts.name ?? uniqueName(sourceName(abs), sources),
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

function uniqueName(base: string, existing: CorpusSource[]): string {
  if (!existing.some((s) => s.name === base)) return base;
  for (let i = 2; ; i++) {
    const candidate = `${base}-${i}`;
    if (!existing.some((s) => s.name === candidate)) return candidate;
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
