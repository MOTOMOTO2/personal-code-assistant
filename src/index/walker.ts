import fs from "node:fs";
import path from "node:path";

/** Directories never worth indexing, regardless of .gitignore. */
const ALWAYS_SKIP_DIRS = new Set([
  ".git",
  ".hg",
  ".svn",
  ".pca",
  "node_modules",
  "bower_components",
  "vendor",
  "dist",
  "build",
  "out",
  "target",
  "coverage",
  ".next",
  ".nuxt",
  ".svelte-kit",
  ".turbo",
  ".cache",
  "__pycache__",
  ".pytest_cache",
  ".mypy_cache",
  ".ruff_cache",
  ".venv",
  "venv",
  "env",
  ".tox",
  ".gradle",
  ".idea",
  ".vscode-test",
  "Pods",
  "DerivedData",
]);

/**
 * Extensions treated as source code. Keeping this an allowlist (rather than
 * "anything not binary") keeps lockfiles, minified bundles and data dumps out of
 * the training corpus, where they would badly skew the language model.
 */
export const CODE_EXTENSIONS = new Set([
  ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs",
  ".py", ".pyi", ".rb", ".go", ".rs", ".java", ".kt", ".kts", ".scala",
  ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".m", ".mm",
  ".cs", ".fs", ".swift", ".php", ".pl", ".pm", ".lua", ".dart",
  ".ex", ".exs", ".erl", ".hs", ".ml", ".clj", ".cljs", ".zig", ".nim",
  ".sh", ".bash", ".zsh", ".fish", ".ps1",
  ".sql", ".graphql", ".gql", ".proto", ".thrift",
  ".html", ".vue", ".svelte", ".css", ".scss", ".sass", ".less",
  ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".tf", ".hcl",
  ".md", ".mdx", ".rst",
  ".cmake", ".mk", ".gradle", ".bzl",
]);

/** Files that are technically code-ish but are noise for a personal corpus. */
const SKIP_FILENAMES = new Set([
  "package-lock.json",
  "yarn.lock",
  "pnpm-lock.yaml",
  "bun.lockb",
  "composer.lock",
  "Gemfile.lock",
  "poetry.lock",
  "Cargo.lock",
  "go.sum",
]);

export interface WalkOptions {
  /** Skip files larger than this. Defaults to 1 MiB. */
  maxFileBytes?: number;
  /** Extra extensions to include, e.g. [".astro"]. */
  extraExtensions?: string[];
  /** Respect .gitignore files. Defaults to true. */
  useGitignore?: boolean;
  /** Skip files whose longest line exceeds this (minified/generated). */
  maxLineLength?: number;
}

export interface SourceFile {
  /** Absolute path. */
  abs: string;
  /** Path relative to the walk root, with forward slashes. */
  rel: string;
  size: number;
  mtimeMs: number;
}

interface IgnoreRule {
  re: RegExp;
  negated: boolean;
  dirOnly: boolean;
  /** Directory (relative to root) the rule was declared in. */
  base: string;
}

/** Translate one .gitignore pattern into a regex over root-relative paths. */
function compileRule(pattern: string, base: string): IgnoreRule | null {
  let pat = pattern.trim();
  if (!pat || pat.startsWith("#")) return null;
  let negated = false;
  if (pat.startsWith("!")) {
    negated = true;
    pat = pat.slice(1);
  }
  let dirOnly = false;
  if (pat.endsWith("/")) {
    dirOnly = true;
    pat = pat.slice(0, -1);
  }
  // A pattern containing a slash (other than a trailing one) is anchored to the
  // directory holding the .gitignore; otherwise it matches at any depth.
  const anchored = pat.includes("/");
  if (pat.startsWith("/")) pat = pat.slice(1);

  let re = "";
  for (let i = 0; i < pat.length; i++) {
    const c = pat[i];
    if (c === "*") {
      if (pat[i + 1] === "*") {
        // "**/" matches zero or more directories.
        if (pat[i + 2] === "/") {
          re += "(?:.*/)?";
          i += 2;
        } else {
          re += ".*";
          i += 1;
        }
      } else {
        re += "[^/]*";
      }
    } else if (c === "?") {
      re += "[^/]";
    } else if (".+^${}()|[]\\".includes(c)) {
      re += `\\${c}`;
    } else {
      re += c;
    }
  }
  const prefix = anchored ? "" : "(?:.*/)?";
  return {
    re: new RegExp(`^${prefix}${re}(?:/.*)?$`),
    negated,
    dirOnly,
    base,
  };
}

function loadIgnoreRules(dir: string, base: string): IgnoreRule[] {
  const file = path.join(dir, ".gitignore");
  if (!fs.existsSync(file)) return [];
  let text: string;
  try {
    text = fs.readFileSync(file, "utf8");
  } catch {
    return [];
  }
  const rules: IgnoreRule[] = [];
  for (const line of text.split(/\r?\n/)) {
    const rule = compileRule(line, base);
    if (rule) rules.push(rule);
  }
  return rules;
}

function isIgnored(relPath: string, isDir: boolean, rules: IgnoreRule[]): boolean {
  let ignored = false;
  // Later rules win, which is how git resolves negations.
  for (const rule of rules) {
    if (rule.dirOnly && !isDir) continue;
    const scoped = rule.base ? (relPath.startsWith(`${rule.base}/`) ? relPath.slice(rule.base.length + 1) : null) : relPath;
    if (scoped === null) continue;
    if (rule.re.test(scoped)) ignored = !rule.negated;
  }
  return ignored;
}

/** True if the buffer looks like binary content (NUL byte in the first 8 KiB). */
function looksBinary(buf: Buffer): boolean {
  const n = Math.min(buf.length, 8192);
  for (let i = 0; i < n; i++) if (buf[i] === 0) return true;
  return false;
}

/**
 * Walk a project for indexable source files. Honours .gitignore, skips vendored
 * and generated trees, and rejects minified files by line length.
 */
export function walkSources(root: string, opts: WalkOptions = {}): SourceFile[] {
  const maxFileBytes = opts.maxFileBytes ?? 1024 * 1024;
  const maxLineLength = opts.maxLineLength ?? 1000;
  const useGitignore = opts.useGitignore !== false;
  const extensions = new Set(CODE_EXTENSIONS);
  for (const e of opts.extraExtensions ?? []) extensions.add(e.startsWith(".") ? e : `.${e}`);

  const out: SourceFile[] = [];
  const stack: Array<{ dir: string; rel: string; rules: IgnoreRule[] }> = [
    { dir: root, rel: "", rules: useGitignore ? loadIgnoreRules(root, "") : [] },
  ];

  while (stack.length > 0) {
    const { dir, rel, rules } = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const abs = path.join(dir, entry.name);
      const childRel = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isSymbolicLink()) continue; // avoid cycles and escaping the root
      if (entry.isDirectory()) {
        if (ALWAYS_SKIP_DIRS.has(entry.name)) continue;
        if (isIgnored(childRel, true, rules)) continue;
        const childRules = useGitignore
          ? [...rules, ...loadIgnoreRules(abs, childRel)]
          : rules;
        stack.push({ dir: abs, rel: childRel, rules: childRules });
        continue;
      }
      if (!entry.isFile()) continue;
      if (SKIP_FILENAMES.has(entry.name)) continue;
      if (entry.name.endsWith(".min.js") || entry.name.endsWith(".min.css")) continue;
      if (entry.name.endsWith(".d.ts")) continue; // generated declarations
      const ext = path.extname(entry.name).toLowerCase();
      if (!extensions.has(ext)) continue;
      if (isIgnored(childRel, false, rules)) continue;
      let stat: fs.Stats;
      try {
        stat = fs.statSync(abs);
      } catch {
        continue;
      }
      if (stat.size === 0 || stat.size > maxFileBytes) continue;
      out.push({ abs, rel: childRel, size: stat.size, mtimeMs: stat.mtimeMs });
    }
  }

  // Deterministic order so index ids are stable across runs.
  out.sort((a, b) => (a.rel < b.rel ? -1 : a.rel > b.rel ? 1 : 0));

  return out.filter((f) => {
    let buf: Buffer;
    try {
      buf = fs.readFileSync(f.abs);
    } catch {
      return false;
    }
    if (looksBinary(buf)) return false;
    const text = buf.toString("utf8");
    let longest = 0;
    let current = 0;
    for (let i = 0; i < text.length; i++) {
      if (text.charCodeAt(i) === 10) {
        if (current > longest) longest = current;
        current = 0;
      } else current++;
    }
    if (current > longest) longest = current;
    return longest <= maxLineLength;
  });
}

export function readSource(file: SourceFile): string {
  return fs.readFileSync(file.abs, "utf8");
}
