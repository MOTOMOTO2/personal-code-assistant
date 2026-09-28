import { readJsonGz, writeJsonGz } from "../util/io.js";

/**
 * A measured description of how this codebase is written.
 *
 * This is the cheapest high-leverage part of the whole system: telling the
 * language model "2-space indent, double quotes, no semicolons, snake_case
 * functions" removes most of the friction of generated code, and all of it is
 * counted from the corpus rather than guessed or configured.
 */
export interface StyleProfile {
  indentKind: "spaces" | "tabs";
  indentWidth: number;
  quote: "single" | "double" | "backtick";
  semicolons: boolean;
  /** 90th-percentile line length, a good proxy for the print width in use. */
  lineWidth: number;
  /** Dominant convention for multi-word function and variable names. */
  naming: "camelCase" | "snake_case" | "PascalCase" | "mixed";
  /** Comment lines per 100 lines of code. */
  commentDensity: number;
  /** Extensions by file count, most common first. */
  languages: Array<{ ext: string; files: number }>;
  /** Frequently imported modules, most common first. */
  topImports: Array<{ name: string; count: number }>;
  filesAnalysed: number;
}

export const DEFAULT_STYLE: StyleProfile = {
  indentKind: "spaces",
  indentWidth: 2,
  quote: "double",
  semicolons: true,
  lineWidth: 100,
  naming: "camelCase",
  commentDensity: 8,
  languages: [],
  topImports: [],
  filesAnalysed: 0,
};

interface Counters {
  tabIndent: number;
  spaceIndent: number;
  indentWidths: Map<number, number>;
  single: number;
  double: number;
  backtick: number;
  semi: number;
  noSemi: number;
  lineLengths: number[];
  camel: number;
  snake: number;
  pascal: number;
  comments: number;
  codeLines: number;
  exts: Map<string, number>;
  imports: Map<string, number>;
  files: number;
}

function newCounters(): Counters {
  return {
    tabIndent: 0,
    spaceIndent: 0,
    indentWidths: new Map(),
    single: 0,
    double: 0,
    backtick: 0,
    semi: 0,
    noSemi: 0,
    lineLengths: [],
    camel: 0,
    snake: 0,
    pascal: 0,
    comments: 0,
    codeLines: 0,
    exts: new Map(),
    imports: new Map(),
    files: 0,
  };
}

const IMPORT_PATTERNS = [
  /^\s*import\s+(?:[\w*{}\s,$]+\s+from\s+)?["']([^"']+)["']/,
  /^\s*(?:from|import)\s+([\w.]+)/, // Python
  /\brequire\(\s*["']([^"']+)["']\s*\)/,
  /^\s*use\s+([\w:]+)/, // Rust
  /^\s*#include\s*[<"]([^>"]+)[>"]/,
];

export class StyleAnalyzer {
  private c = newCounters();

  /** Accumulate one file. `ext` should include the leading dot. */
  add(text: string, ext: string): void {
    this.c.files++;
    this.c.exts.set(ext, (this.c.exts.get(ext) ?? 0) + 1);
    const lines = text.split(/\r?\n/);
    let prevIndent = 0;
    for (const line of lines) {
      if (line.trim().length === 0) continue;
      this.c.lineLengths.push(line.length);

      // --- indentation ---
      const lead = /^[ \t]*/.exec(line)![0];
      if (lead.includes("\t")) this.c.tabIndent++;
      else if (lead.length > 0) {
        this.c.spaceIndent++;
        const delta = lead.length - prevIndent;
        if (delta > 0 && delta <= 8) {
          this.c.indentWidths.set(delta, (this.c.indentWidths.get(delta) ?? 0) + 1);
        }
      }
      if (!lead.includes("\t")) prevIndent = lead.length;

      const trimmed = line.trim();

      // --- comments ---
      if (
        trimmed.startsWith("//") ||
        trimmed.startsWith("#") ||
        trimmed.startsWith("/*") ||
        trimmed.startsWith("*") ||
        trimmed.startsWith("--")
      ) {
        this.c.comments++;
        continue;
      }
      this.c.codeLines++;

      // --- quotes (count per line, not per char, to avoid long-string skew) ---
      if (/'[^']*'/.test(trimmed)) this.c.single++;
      if (/"[^"]*"/.test(trimmed)) this.c.double++;
      if (/`[^`]*`/.test(trimmed)) this.c.backtick++;

      // --- statement terminators ---
      if (/[;]\s*$/.test(trimmed)) this.c.semi++;
      else if (/[\w)\]"'`]\s*$/.test(trimmed)) this.c.noSemi++;

      // --- imports ---
      for (const re of IMPORT_PATTERNS) {
        const m = re.exec(line);
        if (m && m[1]) {
          const name = m[1].split("/")[0].replace(/^\./, "");
          if (name.length > 1) this.c.imports.set(name, (this.c.imports.get(name) ?? 0) + 1);
          break;
        }
      }

      // --- naming conventions (only multi-word names are informative) ---
      const names = trimmed.match(/\b[A-Za-z_$][\w$]{2,}\b/g) ?? [];
      for (const name of names) {
        if (/_/.test(name) && /^[a-z]/.test(name)) this.c.snake++;
        else if (/^[a-z][a-z0-9]*[A-Z]/.test(name)) this.c.camel++;
        else if (/^[A-Z][a-z0-9]*[A-Z]/.test(name)) this.c.pascal++;
      }
    }
  }

  finish(): StyleProfile {
    const c = this.c;
    if (c.files === 0) return { ...DEFAULT_STYLE };
    const indentKind = c.tabIndent > c.spaceIndent ? "tabs" : "spaces";
    let indentWidth = 2;
    let best = -1;
    for (const [width, n] of c.indentWidths) {
      // Favour the common widths; noisy alignment deltas like 3 or 5 are rarely
      // the real indent unit.
      const weight = width === 2 || width === 4 || width === 8 ? n * 1.5 : n;
      if (weight > best) {
        best = weight;
        indentWidth = width;
      }
    }
    const quote =
      c.single > c.double && c.single > c.backtick
        ? "single"
        : c.backtick > c.double
          ? "backtick"
          : "double";
    c.lineLengths.sort((a, b) => a - b);
    const p90 = c.lineLengths[Math.floor(c.lineLengths.length * 0.9)] ?? 100;
    const naming =
      c.snake > c.camel * 1.5
        ? "snake_case"
        : c.camel > c.snake * 1.5
          ? "camelCase"
          : c.pascal > c.camel && c.pascal > c.snake
            ? "PascalCase"
            : "mixed";
    return {
      indentKind,
      indentWidth: indentKind === "tabs" ? 4 : indentWidth,
      quote,
      semicolons: c.semi > c.noSemi * 0.4,
      lineWidth: Math.min(200, Math.max(60, Math.round(p90 / 10) * 10)),
      naming,
      commentDensity: c.codeLines > 0 ? Math.round((c.comments / c.codeLines) * 1000) / 10 : 0,
      languages: [...c.exts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([ext, files]) => ({ ext, files })),
      topImports: [...c.imports.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .map(([name, count]) => ({ name, count })),
      filesAnalysed: c.files,
    };
  }
}

/** Render the profile as instructions for the language model. */
export function styleToPrompt(style: StyleProfile): string {
  const lines = [
    `Indentation: ${style.indentKind === "tabs" ? "tabs" : `${style.indentWidth} spaces`}`,
    `String quotes: ${style.quote === "single" ? "'single'" : style.quote === "backtick" ? "`backtick`" : '"double"'}`,
    `Semicolons: ${style.semicolons ? "yes" : "no"}`,
    `Target line width: ${style.lineWidth}`,
    `Naming for functions/variables: ${style.naming}`,
    `Comment density: about ${style.commentDensity} comment lines per 100 code lines - match this, do not over-comment`,
  ];
  if (style.languages.length > 0) {
    lines.push(`Primary languages: ${style.languages.map((l) => l.ext).join(", ")}`);
  }
  if (style.topImports.length > 0) {
    lines.push(`Commonly used modules: ${style.topImports.slice(0, 10).map((i) => i.name).join(", ")}`);
  }
  return lines.join("\n");
}

export function saveStyle(file: string, style: StyleProfile): void {
  writeJsonGz(file, style);
}

export function loadStyle(file: string): StyleProfile {
  return { ...DEFAULT_STYLE, ...readJsonGz<StyleProfile>(file) };
}
