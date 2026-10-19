import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { Bundle } from "../core/bundle.js";
import { ALWAYS_SKIP_DIRS } from "../index/walker.js";
import { ToolSpec } from "../llm/ollama.js";
import { normalize } from "../ml/embeddings.js";

/**
 * What the local model can reach beyond its own weights.
 *
 * Four kinds, each kept deliberately narrow:
 *
 *   - your projects   list, search and read files under one root, read-only,
 *                     with anything that looks like a secret refused
 *   - your code index the retrieval index pca trained, searched by meaning
 *   - the web         DuckDuckGo's HTML results and plain-text pages, no key
 *   - anything else   a command that lists tools as JSON and runs them, which
 *                     is how the dashboard hands over its jobs and usage
 */

export interface Tool {
  spec: ToolSpec;
  /**
   * The answer depends on the moment it was asked (a web page, a running
   * job). An answer built on one is not remembered as a lesson, because
   * recalling it next week would recall something no longer true.
   */
  live: boolean;
  /** Reads something nobody vouched for. After it, `acts` tools are refused for the rest of the answer. */
  untrusted?: boolean;
  /** Changes something in the world rather than reading it. */
  acts?: boolean;
  /** What to show while it runs, e.g. `searching the web for "ollama"`. */
  note(args: Record<string, unknown>): string;
  run(args: Record<string, unknown>): Promise<string>;
}

const str = (v: unknown): string => (typeof v === "string" ? v : v == null ? "" : String(v));
const int = (v: unknown): number | undefined => {
  const n = typeof v === "number" ? v : Number.parseInt(str(v), 10);
  return Number.isFinite(n) ? n : undefined;
};

// ── your projects ─────────────────────────────────────────────────────

/**
 * Files never read, whatever the model asks. Its answers end up on a phone,
 * by way of a database row - that is not where a password belongs.
 */
export function isSecretFile(name: string): boolean {
  const base = path.basename(name).toLowerCase();
  if (/\.(example|sample|template|dist)$/.test(base)) return false;
  return (
    /^\.env(\..*)?$/.test(base) ||
    /\.(pem|key|p12|pfx|keystore|jks|ppk)$/.test(base) ||
    /^(id_rsa|id_dsa|id_ecdsa|id_ed25519|\.npmrc|\.netrc|\.pypirc|\.git-credentials)$/.test(base) ||
    /credentials/.test(base) ||
    /^secrets?\./.test(base)
  );
}

/** Resolve a path the model gave against `root`, refusing anything that leaves it. */
export function insideRoot(root: string, given: string): string {
  const realRoot = fs.realpathSync(root);
  const abs = path.resolve(realRoot, given || ".");
  let real = abs;
  try {
    real = fs.realpathSync(abs);
  } catch {
    // does not exist - the caller says so
  }
  const rel = path.relative(realRoot, real);
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    throw new Error(`${given} is outside ${root}, which is as far as this assistant can look.`);
  }
  return real;
}

const display = (root: string, abs: string): string =>
  path.relative(fs.realpathSync(root), abs).split(path.sep).join("/") || ".";

function looksBinary(buf: Buffer): boolean {
  const n = Math.min(buf.length, 4096);
  for (let i = 0; i < n; i++) if (buf[i] === 0) return true;
  return false;
}

const READ_MAX_LINES = 300;
const READ_MAX_BYTES = 48_000;
const SEARCH_MAX_HITS = 40;
const SEARCH_MAX_FILE = 512 * 1024;
const SEARCH_BUDGET_MS = 8_000;

export function projectTools(root: string): Tool[] {
  const listFiles: Tool = {
    spec: {
      name: "list_files",
      description: `List what is in a folder under ${root}. Use it to find a project or see how one is laid out. Paths are relative to ${root}.`,
      parameters: {
        type: "object",
        properties: { path: { type: "string", description: "Folder to list. Empty for the top level." } },
      },
    },
    live: false,
    note: (a) => `looking in ${str(a.path) || "your projects"}`,
    async run(a) {
      const dir = insideRoot(root, str(a.path));
      const entries = fs.readdirSync(dir, { withFileTypes: true })
        .filter((e) => !ALWAYS_SKIP_DIRS.has(e.name))
        .sort((x, y) => Number(y.isDirectory()) - Number(x.isDirectory()) || x.name.localeCompare(y.name));
      const lines = entries.slice(0, 200).map((e) => {
        if (e.isDirectory()) return `${e.name}/`;
        let size = "";
        try {
          size = ` (${fs.statSync(path.join(dir, e.name)).size} bytes)`;
        } catch {
          // unreadable - list the name anyway
        }
        return `${e.name}${size}`;
      });
      if (entries.length > 200) lines.push(`… and ${entries.length - 200} more`);
      return `${display(root, dir)}/\n${lines.join("\n") || "(empty)"}`;
    },
  };

  const readFile: Tool = {
    spec: {
      name: "read_file",
      description: `Read a text file under ${root}, with line numbers. Read a range for long files.`,
      parameters: {
        type: "object",
        required: ["path"],
        properties: {
          path: { type: "string", description: `File path relative to ${root}.` },
          start_line: { type: "integer", description: "First line to read, from 1." },
          end_line: { type: "integer", description: "Last line to read." },
        },
      },
    },
    live: false,
    note: (a) => `reading ${str(a.path)}`,
    async run(a) {
      const file = insideRoot(root, str(a.path));
      if (isSecretFile(file)) throw new Error(`${str(a.path)} looks like it holds secrets, so it is not read.`);
      if (!fs.existsSync(file)) throw new Error(`There is no ${str(a.path)}.`);
      if (fs.statSync(file).isDirectory()) throw new Error(`${str(a.path)} is a folder - use list_files.`);
      const buf = fs.readFileSync(file);
      if (looksBinary(buf)) throw new Error(`${str(a.path)} is a binary file.`);
      const all = buf.toString("utf8").split(/\r?\n/);
      const start = Math.max(1, int(a.start_line) ?? 1);
      const end = Math.min(all.length, int(a.end_line) ?? start + READ_MAX_LINES - 1, start + READ_MAX_LINES - 1);
      let out = "";
      for (let i = start; i <= end; i++) {
        const line = `${String(i).padStart(5)} | ${all[i - 1]}\n`;
        if (out.length + line.length > READ_MAX_BYTES) break;
        out += line;
      }
      const shown = out.split("\n").length - 1;
      const more = start + shown - 1 < all.length ? `\n(lines ${start}-${start + shown - 1} of ${all.length}; ask for more with start_line)` : "";
      return `${display(root, file)}\n${out}${more}`;
    },
  };

  const searchFiles: Tool = {
    spec: {
      name: "search_files",
      description: `Search the text of every file under ${root} (or one folder of it) for a word or phrase. Returns path:line matches.`,
      parameters: {
        type: "object",
        required: ["query"],
        properties: {
          query: { type: "string", description: "Text to find, case-insensitive." },
          path: { type: "string", description: "Folder to search in. Empty for everything." },
        },
      },
    },
    live: false,
    note: (a) => `searching your projects for "${str(a.query)}"`,
    async run(a) {
      const needle = str(a.query).toLowerCase();
      if (!needle.trim()) throw new Error("Give search_files something to look for.");
      const start = insideRoot(root, str(a.path));
      const hits: string[] = [];
      const began = Date.now();
      let scanned = 0;
      let stopped = "";
      const stack = [start];
      while (stack.length > 0 && hits.length < SEARCH_MAX_HITS) {
        if (Date.now() - began > SEARCH_BUDGET_MS) {
          stopped = "(stopped early - narrow it with path)";
          break;
        }
        const dir = stack.pop() as string;
        let entries: fs.Dirent[];
        try {
          entries = fs.readdirSync(dir, { withFileTypes: true });
        } catch {
          continue;
        }
        for (const e of entries) {
          const full = path.join(dir, e.name);
          if (e.isDirectory()) {
            if (!ALWAYS_SKIP_DIRS.has(e.name) && !e.name.startsWith(".")) stack.push(full);
            continue;
          }
          if (!e.isFile() || isSecretFile(e.name)) continue;
          let buf: Buffer;
          try {
            if (fs.statSync(full).size > SEARCH_MAX_FILE) continue;
            buf = fs.readFileSync(full);
          } catch {
            continue;
          }
          scanned++;
          if (looksBinary(buf)) continue;
          const text = buf.toString("utf8");
          if (!text.toLowerCase().includes(needle)) continue;
          const lines = text.split(/\r?\n/);
          for (let i = 0; i < lines.length && hits.length < SEARCH_MAX_HITS; i++) {
            if (lines[i].toLowerCase().includes(needle)) {
              hits.push(`${display(root, full)}:${i + 1}: ${lines[i].trim().slice(0, 160)}`);
            }
          }
        }
      }
      if (hits.length === 0) return `No file under ${display(root, start)} mentions "${str(a.query)}" (${scanned} files read).`;
      return [...hits, stopped].filter(Boolean).join("\n");
    },
  };

  const overview: Tool = {
    spec: {
      name: "project_overview",
      description:
        "Everything needed to say what one project is, in one call: where it is, its README, its package.json, and what is in it. " +
        "Use this first for any question about a project by name.",
      parameters: {
        type: "object",
        required: ["name"],
        properties: { name: { type: "string", description: "The project's name, or part of it." } },
      },
    },
    live: false,
    note: (a) => `reading about ${str(a.name)}`,
    async run(a) {
      const found = findProjects(root, str(a.name));
      if (found.length === 0) {
        return `No folder under ${root} is called anything like "${str(a.name)}". list_files shows what is there.`;
      }
      const dir = found[0];
      const parts = [`${display(root, dir)}/`];
      if (found.length > 1) parts.push(`(also matching: ${found.slice(1, 5).map((d) => display(root, d)).join(", ")})`);
      const readme = fs.readdirSync(dir).find((f) => /^readme(\.(md|txt|rst))?$/i.test(f));
      if (readme) {
        const lines = fs.readFileSync(path.join(dir, readme), "utf8").split(/\r?\n/);
        parts.push(`\n${readme}:\n${lines.slice(0, 60).join("\n")}${lines.length > 60 ? `\n… (${lines.length} lines; read_file for the rest)` : ""}`);
      } else {
        parts.push("\nNo README.");
      }
      const pkgFile = path.join(dir, "package.json");
      if (fs.existsSync(pkgFile)) {
        try {
          const pkg = JSON.parse(fs.readFileSync(pkgFile, "utf8")) as Record<string, unknown>;
          const deps = Object.keys({ ...(pkg.dependencies as object), ...(pkg.devDependencies as object) });
          parts.push(
            `\npackage.json: ${str(pkg.name)} - ${str(pkg.description) || "no description"}` +
              `\n  scripts: ${Object.keys((pkg.scripts as object) ?? {}).join(", ") || "none"}` +
              `\n  depends on: ${deps.slice(0, 20).join(", ") || "nothing"}`,
          );
        } catch {
          // not valid JSON - skip it
        }
      }
      const entries = fs.readdirSync(dir, { withFileTypes: true })
        .filter((e) => !ALWAYS_SKIP_DIRS.has(e.name) && !e.name.startsWith("."))
        .map((e) => (e.isDirectory() ? `${e.name}/` : e.name));
      parts.push(`\nIn it: ${entries.slice(0, 60).join("  ")}`);
      return parts.join("\n");
    },
  };

  return [overview, listFiles, readFile, searchFiles];
}

const squash = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

/** Folders under `root`, three levels down at most, whose name matches; exact matches first, then the shallowest. */
export function findProjects(root: string, name: string): string[] {
  const want = squash(name);
  if (!want) return [];
  const exact: string[] = [];
  const partial: string[] = [];
  const walk = (dir: string, depth: number) => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      if (!e.isDirectory() || ALWAYS_SKIP_DIRS.has(e.name) || e.name.startsWith(".")) continue;
      const full = path.join(dir, e.name);
      const have = squash(e.name);
      if (have === want) exact.push(full);
      else if (have.includes(want) || (have.length > 3 && want.includes(have))) partial.push(full);
      if (depth < 3) walk(full, depth + 1);
    }
  };
  walk(fs.realpathSync(root), 1);
  const byDepth = (x: string, y: string) => x.split(path.sep).length - y.split(path.sep).length;
  return [...exact.sort(byDepth), ...partial.sort(byDepth)];
}

// ── your code index ───────────────────────────────────────────────────

/** Search by meaning in the index pca trained, or null when it has not been trained. */
export function codeIndexTool(bundle: Bundle): Tool | null {
  if (!bundle.isTrained || !bundle.index) return null;
  return {
    spec: {
      name: "search_code",
      description:
        "Find code by what it does, in the repositories pca has been trained on (its own retrieval index). " +
        "Better than search_files when you know the idea but not the exact words.",
      parameters: {
        type: "object",
        required: ["query"],
        properties: { query: { type: "string", description: "What the code does, in a few words." } },
      },
    },
    live: false,
    note: (a) => `searching the code index for "${str(a.query)}"`,
    async run(a) {
      const index = bundle.index;
      if (!index) return "The code index is not trained.";
      const q = str(a.query);
      const embeddings = bundle.embeddings;
      const queryVector = embeddings ? normalize(embeddings.sifEmbed(bundle.tokenizer.encode(q))) : null;
      const hits = index.search(q, { limit: 6, queryVector });
      if (hits.length === 0) return `Nothing in the index matches "${q}".`;
      return hits
        .map((h) => {
          const head = h.chunk.symbol ? `${h.file}:${h.chunk.startLine} (${h.chunk.symbol})` : `${h.file}:${h.chunk.startLine}`;
          const body = h.chunk.text.split("\n").slice(0, 24).join("\n");
          return `--- ${head}\n${body}`;
        })
        .join("\n\n");
    },
  };
}

// ── the web ───────────────────────────────────────────────────────────

const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36";

export function decodeEntities(text: string): string {
  return text
    .replace(/&#x([0-9a-f]+);/gi, (_, h: string) => String.fromCodePoint(Number.parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d: string) => String.fromCodePoint(Number.parseInt(d, 10)))
    .replace(/&nbsp;/g, " ")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

const stripTags = (html: string): string => decodeEntities(html.replace(/<[^>]+>/g, "")).replace(/\s+/g, " ").trim();

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}

/** Results out of DuckDuckGo's HTML page, with its redirect links unwrapped. */
export function parseDuckDuckGo(html: string): SearchResult[] {
  const results: SearchResult[] = [];
  const blocks = html.split(/<div[^>]+class="[^"]*\bresult\b[^"]*"/).slice(1);
  for (const block of blocks) {
    const link = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/.exec(block);
    if (!link) continue;
    let url = decodeEntities(link[1]);
    const wrapped = /[?&]uddg=([^&]+)/.exec(url);
    if (wrapped) url = decodeURIComponent(wrapped[1]);
    if (url.startsWith("//")) url = `https:${url}`;
    // Ads link back through duckduckgo.com/y.js; they are not results.
    if (/duckduckgo\.com\/y\.js/.test(url)) continue;
    const snippet = /class="result__snippet"[^>]*>([\s\S]*?)<\/a>/.exec(block);
    results.push({ title: stripTags(link[2]), url, snippet: snippet ? stripTags(snippet[1]) : "" });
  }
  return results;
}

/** Visible text of a page: scripts, styles and markup gone, block breaks kept as newlines. */
export function pageText(html: string): string {
  const title = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html);
  const body = html
    .replace(/<(script|style|noscript|svg|head)[\s\S]*?<\/\1>/gi, " ")
    .replace(/<\/(p|div|li|h[1-6]|tr|br|section|article|header|footer)>|<br\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, " ");
  const text = decodeEntities(body)
    .split("\n")
    .map((l) => l.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .join("\n");
  return title ? `${stripTags(title[1])}\n\n${text}` : text;
}

/** Hosts a page fetched for the model may not point it at: this machine and the home network. */
export function isPrivateHost(host: string): boolean {
  const h = host.toLowerCase().replace(/^\[|\]$/g, "");
  return (
    h === "localhost" ||
    h.endsWith(".local") ||
    h.endsWith(".localhost") ||
    /^127\./.test(h) ||
    /^10\./.test(h) ||
    /^192\.168\./.test(h) ||
    /^169\.254\./.test(h) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(h) ||
    /^0\./.test(h) ||
    h === "::1" ||
    /^f[cd][0-9a-f]{2}:/.test(h) ||
    /^fe80:/.test(h)
  );
}

const PAGE_MAX_CHARS = 10_000;

export function webTools(fetchImpl: typeof fetch = fetch): Tool[] {
  const search: Tool = {
    spec: {
      name: "web_search",
      description: "Search the web (DuckDuckGo). Use it for anything current, or anything you are not sure of.",
      parameters: {
        type: "object",
        required: ["query"],
        properties: { query: { type: "string", description: "What to search for." } },
      },
    },
    live: true,
    untrusted: true,
    note: (a) => `searching the web for "${str(a.query)}"`,
    async run(a) {
      const q = str(a.query).trim();
      if (!q) throw new Error("Give web_search something to search for.");
      const res = await fetchImpl(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}`, {
        headers: { "user-agent": UA, accept: "text/html" },
        signal: AbortSignal.timeout(15_000),
      });
      if (!res.ok) throw new Error(`The search answered ${res.status}.`);
      const results = parseDuckDuckGo(await res.text()).slice(0, 6);
      if (results.length === 0) return `No results for "${q}".`;
      return (
        results.map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.snippet}`).join("\n") +
        "\n\nThese are only snippets. Before stating a fact from them, read the best result with fetch_page."
      );
    },
  };

  const fetchPage: Tool = {
    spec: {
      name: "fetch_page",
      description: "Read the text of a web page, e.g. one web_search found.",
      parameters: {
        type: "object",
        required: ["url"],
        properties: { url: { type: "string", description: "An http(s) address." } },
      },
    },
    live: true,
    untrusted: true,
    note: (a) => {
      try {
        return `reading ${new URL(str(a.url)).host}`;
      } catch {
        return "reading a page";
      }
    },
    async run(a) {
      let url: URL;
      try {
        url = new URL(str(a.url));
      } catch {
        throw new Error(`${str(a.url)} is not a web address.`);
      }
      if (!/^https?:$/.test(url.protocol)) throw new Error("Only http and https pages can be read.");
      if (isPrivateHost(url.hostname)) throw new Error("Pages on this machine or this network are not fetched.");
      const res = await fetchImpl(url, {
        headers: { "user-agent": UA, accept: "text/html,text/plain,application/json" },
        signal: AbortSignal.timeout(15_000),
      });
      if (!res.ok) throw new Error(`${url.host} answered ${res.status}.`);
      const type = res.headers.get("content-type") ?? "";
      if (!/text|json|xml/.test(type)) throw new Error(`That is not a page of text (${type || "unknown type"}).`);
      const raw = (await res.text()).slice(0, 2_000_000);
      const text = /html/.test(type) ? pageText(raw) : raw;
      return text.length > PAGE_MAX_CHARS ? `${text.slice(0, PAGE_MAX_CHARS)}\n… (cut at ${PAGE_MAX_CHARS} characters)` : text;
    },
  };

  return [search, fetchPage];
}

// ── anything else ─────────────────────────────────────────────────────
// A provider is a command. `<cmd> list` prints a JSON array of
//   { name, description, parameters, live?, acts?, note? }
// and `<cmd> call <name>` reads the arguments as JSON on stdin and prints the
// result. A non-zero exit is a failed call, with stderr as the reason.

interface ProviderSpec extends ToolSpec {
  live?: boolean;
  acts?: boolean;
  note?: string;
}

export type CommandRunner = (cmd: string, input: string, timeoutMs: number) => Promise<string>;

export const runCommand: CommandRunner = (cmd, input, timeoutMs) =>
  new Promise((resolve, reject) => {
    const child = spawn(cmd, { shell: true, stdio: ["pipe", "pipe", "pipe"], windowsHide: true });
    let out = "";
    let err = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`took longer than ${Math.round(timeoutMs / 1000)}s`));
    }, timeoutMs);
    child.stdout.setEncoding("utf8").on("data", (d: string) => (out += d));
    child.stderr.setEncoding("utf8").on("data", (d: string) => (err += d));
    child.stdin.on("error", () => {});
    child.stdin.end(input);
    child.on("error", (e) => {
      clearTimeout(timer);
      reject(e);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(out);
      else reject(new Error(err.trim() || `exited ${code}`));
    });
  });

/** The tools a provider command offers. A provider that will not list is skipped, not fatal. */
export async function externalTools(cmd: string, run: CommandRunner = runCommand): Promise<{ tools: Tool[]; problem: string | null }> {
  let specs: ProviderSpec[];
  try {
    specs = JSON.parse(await run(`${cmd} list`, "", 15_000)) as ProviderSpec[];
    if (!Array.isArray(specs)) throw new Error("did not print a JSON array");
  } catch (e) {
    return { tools: [], problem: `${cmd} list failed: ${(e as Error).message}` };
  }
  const tools = specs
    .filter((s) => s && typeof s.name === "string" && /^[A-Za-z0-9_-]+$/.test(s.name))
    .map(
      (s): Tool => ({
        spec: {
          name: s.name,
          description: str(s.description),
          parameters: s.parameters ?? { type: "object", properties: {} },
        },
        live: s.live !== false,
        acts: Boolean(s.acts),
        note: () => s.note ?? `using ${s.name}`,
        run: (a) => run(`${cmd} call ${s.name}`, JSON.stringify(a ?? {}), 30_000),
      }),
    );
  return { tools, problem: null };
}
