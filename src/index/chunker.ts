/**
 * Language-agnostic, symbol-aware chunking.
 *
 * Retrieval quality depends far more on chunk boundaries than on the embedding
 * model, so chunks are cut at plausible declaration starts rather than every N
 * lines. A regex-per-language parser would be more precise but would also mean
 * maintaining a grammar per language; the heuristic below covers the C-family,
 * Python, Ruby, Go, Rust, and the JS/TS family in one pass by looking for
 * declaration keywords at shallow indentation.
 */

export interface Chunk {
  /** Line number of the first line, 1-based. */
  startLine: number;
  /** Line number of the last line, 1-based inclusive. */
  endLine: number;
  text: string;
  /** Primary declared name - the first declaration inside the chunk. */
  symbol: string | null;
  /** Every declaration detected inside the chunk, in source order. */
  symbols: string[];
}

export interface ChunkOptions {
  /** Soft target size in characters. Chunks grow past it to finish a symbol. */
  targetChars?: number;
  /** Hard cap; a symbol larger than this is split. */
  maxChars?: number;
  /** Lines of trailing context repeated into the next chunk. */
  overlapLines?: number;
}

const DECL_PATTERNS: Array<{ re: RegExp; group: number }> = [
  // export function foo / async function foo / function* foo
  { re: /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)/, group: 1 },
  // class / interface / enum / struct / trait / impl / type
  { re: /^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:final\s+)?(?:public\s+|private\s+|protected\s+|internal\s+)?(class|interface|enum|struct|trait|impl|type|record|protocol|module|namespace)\s+([A-Za-z_$][\w$<>,.\s]*)/, group: 2 },
  // Python / Ruby / Elixir style defs
  { re: /^\s*(?:async\s+)?def\s+([A-Za-z_][\w!?]*)/, group: 1 },
  { re: /^\s*(?:defp?|defmodule)\s+([A-Za-z_][\w.!?]*)/, group: 1 },
  // Go / Rust / Zig functions
  { re: /^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)/, group: 1 },
  { re: /^\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][\w]*)/, group: 1 },
  // const foo = (…) => / const foo = function
  { re: /^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\(|function|<)/, group: 1 },
  // Java/C#/C++ methods and top-level C functions: `Type name(...) {`
  { re: /^(?:public|private|protected|internal|static|virtual|override|inline|extern|\s)*[A-Za-z_][\w:<>,*&\s[\]]*\s+([A-Za-z_][\w]*)\s*\([^;]*\)\s*(?:const\s*)?\{\s*$/, group: 1 },
  // TS/JS class members: `private async fetchAll(x: T): Promise<U> {`. Written
  // last because it is the loosest pattern; RESERVED_NAMES rejects `if (...) {`.
  { re: /^\s*(?:public\s+|private\s+|protected\s+|static\s+|readonly\s+|override\s+|abstract\s+)*(?:async\s+)?(?:get\s+|set\s+)?\*?\s*([A-Za-z_$][\w$]*)\s*(?:<[^<>()]*>)?\s*\([^;]*\)\s*(?::\s*[^{;=]+)?\s*\{\s*$/, group: 1 },
];

/** Indentation depth in visual columns (tabs count as 4). */
function indentOf(line: string): number {
  let n = 0;
  for (const ch of line) {
    if (ch === " ") n++;
    else if (ch === "\t") n += 4;
    else break;
  }
  return n;
}

function declarationName(line: string): string | null {
  if (line.trim().length === 0) return null;
  // Only treat shallow lines as boundaries: a nested helper should stay with its
  // parent, which keeps retrieved chunks self-contained.
  if (indentOf(line) > 4) return null;
  for (const { re, group } of DECL_PATTERNS) {
    const m = re.exec(line);
    if (m && m[group]) {
      const name = m[group].trim().split(/[\s<(:{]/)[0];
      if (name && !RESERVED_NAMES.has(name)) return name;
    }
  }
  return null;
}

/** Keywords that the loose C-family method pattern can pick up by mistake. */
const RESERVED_NAMES = new Set([
  "if", "for", "while", "switch", "catch", "return", "else", "do", "try",
  "with", "match", "when", "case", "using", "new", "delete", "sizeof",
]);

export function chunkFile(text: string, opts: ChunkOptions = {}): Chunk[] {
  const targetChars = opts.targetChars ?? 1200;
  const maxChars = opts.maxChars ?? 4000;
  const overlapLines = opts.overlapLines ?? 2;
  const lines = text.split(/\r?\n/);
  if (lines.length === 0) return [];

  // Find candidate cut points: line indices that start a declaration.
  const boundaries = new Set<number>([0]);
  const symbolAt = new Map<number, string>();
  for (let i = 0; i < lines.length; i++) {
    const name = declarationName(lines[i]);
    if (name !== null) {
      // Pull preceding comment lines and decorators into the symbol's chunk.
      let start = i;
      while (start > 0 && isAttachedPreamble(lines[start - 1])) start--;
      boundaries.add(start);
      symbolAt.set(start, name);
    }
  }
  const cuts = [...boundaries].sort((a, b) => a - b);

  // Merge consecutive regions until each chunk reaches the target size.
  const chunks: Chunk[] = [];
  let i = 0;
  while (i < cuts.length) {
    const start = cuts[i];
    let end = i + 1 < cuts.length ? cuts[i + 1] : lines.length; // exclusive
    let size = charLen(lines, start, end);
    let j = i + 1;
    while (size < targetChars && j < cuts.length) {
      const nextEnd = j + 1 < cuts.length ? cuts[j + 1] : lines.length;
      const nextSize = charLen(lines, start, nextEnd);
      if (nextSize > maxChars) break;
      end = nextEnd;
      size = nextSize;
      j++;
    }
    // Merging absorbs later cut points, so the chunk's name is the first symbol
    // declared anywhere inside it - not just one declared on its first line.
    const symbolsIn = (from: number, to: number): string[] => {
      const found: string[] = [];
      for (let c = 0; c < cuts.length; c++) {
        if (cuts[c] >= from && cuts[c] < to) {
          const name = symbolAt.get(cuts[c]);
          if (name && !found.includes(name)) found.push(name);
        }
      }
      return found;
    };

    // A single oversized symbol still has to be split somewhere.
    if (size > maxChars) {
      for (const piece of splitOversized(lines, start, end, maxChars)) {
        const symbols = symbolsIn(piece.start, piece.end);
        const inherited = symbolAt.get(start);
        // A continuation piece carries the enclosing symbol's name so a search
        // hit in the middle of a long function still says what it is in.
        if (symbols.length === 0 && inherited) symbols.push(inherited);
        chunks.push({
          startLine: piece.start + 1,
          endLine: piece.end,
          text: lines.slice(piece.start, piece.end).join("\n"),
          symbol: symbols[0] ?? null,
          symbols,
        });
      }
    } else {
      const from = Math.max(0, start - (chunks.length > 0 ? overlapLines : 0));
      const symbols = symbolsIn(start, end);
      chunks.push({
        startLine: from + 1,
        endLine: end,
        text: lines.slice(from, end).join("\n"),
        symbol: symbols[0] ?? null,
        symbols,
      });
    }
    i = Math.max(j, i + 1);
  }

  return chunks.filter((c) => c.text.trim().length > 0);
}

/** Comment lines, decorators and annotations belong to the symbol below them. */
function isAttachedPreamble(line: string): boolean {
  const t = line.trim();
  if (t.length === 0) return false;
  return (
    t.startsWith("//") ||
    t.startsWith("#") ||
    t.startsWith("/*") ||
    t.startsWith("*") ||
    t.startsWith("*/") ||
    t.startsWith("@") ||
    t.startsWith('"""') ||
    t.startsWith("'''") ||
    t.startsWith("---") ||
    t.startsWith("export ")
  );
}

function charLen(lines: string[], start: number, end: number): number {
  let n = 0;
  for (let i = start; i < end; i++) n += lines[i].length + 1;
  return n;
}

function splitOversized(
  lines: string[],
  start: number,
  end: number,
  maxChars: number,
): Array<{ start: number; end: number }> {
  const out: Array<{ start: number; end: number }> = [];
  let cur = start;
  while (cur < end) {
    let size = 0;
    let stop = cur;
    while (stop < end && size < maxChars) {
      size += lines[stop].length + 1;
      stop++;
    }
    out.push({ start: cur, end: stop });
    cur = stop;
  }
  return out;
}
