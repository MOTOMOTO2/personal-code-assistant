import { readJsonGz, writeJsonGz } from "../util/io.js";

/**
 * Reserved token ids. 0..255 are raw UTF-8 bytes, which makes the tokenizer
 * *lossless*: any piece missing from the learned vocabulary is spelled out
 * byte-by-byte, so `decode(encode(s)) === s` for every string. There is no
 * `<unk>` token and no information is ever dropped.
 */
export const BYTE_BASE = 0;
export const BYTE_COUNT = 256;
export const TOK_BOS = 256;
export const TOK_EOS = 257;
export const TOK_PAD = 258;
export const RESERVED = 259;

/**
 * Multi-character operators that should stay glued together. Longest-first, so
 * the alternation prefers `===` over `==` over `=`.
 */
const OPERATORS = [
  ">>>=", "...", "<=>", "===", "!==", "**=", "<<=", ">>=", ">>>", "&&=", "||=", "??=",
  "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "::", "->", "<-", "++", "--",
  "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "**", "<<", ">>", "//", "/*", "*/",
  "#!", "|>", ":=",
];

const RE_SPECIAL = /[.*+?^${}()|[\]\\]/g;

function escapeRe(s: string): string {
  return s.replace(RE_SPECIAL, (m) => `\\${m}`);
}

/**
 * Code-aware pre-tokenizer. Alternatives are ordered; the final catch-all
 * matches any single code point, which guarantees the matches cover the input
 * with no gaps - i.e. the pieces always concatenate back to the original text.
 */
const PRE_TOKEN_RE = new RegExp(
  [
    String.raw`\r\n|\n|\r`, // newline, one per token so line structure is learnable
    String.raw`[ \t]{1,32}`, // indentation / spacing run
    String.raw`[A-Za-z_$][A-Za-z0-9_$]*`, // identifier or keyword
    String.raw`0[xXbBoO][0-9a-fA-F_]+`, // radix literal
    String.raw`\d[\d_]*(?:\.[\d_]+)?(?:[eE][-+]?\d+)?`, // number
    OPERATORS.map(escapeRe).join("|"), // multi-character operator
    String.raw`[\s\S]`, // single code point - the lossless catch-all
  ].join("|"),
  "gu",
);

/** Split text into lossless pieces: `preTokenize(t).join("") === t`. */
export function preTokenize(text: string): string[] {
  const out: string[] = [];
  PRE_TOKEN_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = PRE_TOKEN_RE.exec(text)) !== null) out.push(m[0]);
  return out;
}

export interface TokenizerData {
  version: 1;
  pieces: string[];
}

export class Tokenizer {
  /** Learned pieces, indexed by `id - RESERVED`. */
  readonly pieces: string[];
  private readonly lookup: Map<string, number>;

  constructor(pieces: string[] = []) {
    this.pieces = pieces;
    this.lookup = new Map();
    for (let i = 0; i < pieces.length; i++) this.lookup.set(pieces[i], i + RESERVED);
  }

  get vocabSize(): number {
    return RESERVED + this.pieces.length;
  }

  /**
   * Learn a vocabulary by frequency over a corpus. Pieces that do not make the
   * cut still round-trip through the byte tokens.
   */
  static train(texts: Iterable<string>, vocabSize = 16384, minFreq = 2): Tokenizer {
    const freq = new Map<string, number>();
    for (const text of texts) {
      for (const piece of preTokenize(text)) {
        freq.set(piece, (freq.get(piece) ?? 0) + 1);
      }
    }
    const budget = Math.max(0, vocabSize - RESERVED);
    const ranked = [...freq.entries()]
      .filter(([piece, n]) => n >= minFreq || piece.length === 1)
      // Frequency first, ties broken lexicographically so training is deterministic.
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
      .slice(0, budget)
      .map(([piece]) => piece);
    return new Tokenizer(ranked);
  }

  idOf(piece: string): number | undefined {
    return this.lookup.get(piece);
  }

  encode(text: string): number[] {
    const out: number[] = [];
    for (const piece of preTokenize(text)) this.encodePiece(piece, out);
    return out;
  }

  /** Encode, also returning the character offset in `text` where each token starts. */
  encodeWithOffsets(text: string): { ids: number[]; offsets: number[] } {
    const ids: number[] = [];
    const offsets: number[] = [];
    let pos = 0;
    for (const piece of preTokenize(text)) {
      const before = ids.length;
      this.encodePiece(piece, ids);
      for (let i = before; i < ids.length; i++) offsets.push(pos);
      pos += piece.length;
    }
    return { ids, offsets };
  }

  private encodePiece(piece: string, out: number[]): void {
    const id = this.lookup.get(piece);
    if (id !== undefined) {
      out.push(id);
      return;
    }
    const bytes = Buffer.from(piece, "utf8");
    for (const b of bytes) out.push(BYTE_BASE + b);
  }

  /**
   * Text for a single token. Byte tokens decode as one latin1 character, which is
   * only meaningful when the whole run is decoded together - use `decode` for
   * anything user-visible.
   */
  pieceOf(id: number): string {
    if (id >= RESERVED) return this.pieces[id - RESERVED] ?? "";
    if (id < BYTE_COUNT) return Buffer.from([id]).toString("latin1");
    return "";
  }

  /** True when `id` is a raw-byte token. */
  isByte(id: number): boolean {
    return id >= BYTE_BASE && id < BYTE_BASE + BYTE_COUNT;
  }

  decode(ids: Iterable<number>): string {
    let out = "";
    let pending: number[] = [];
    const flush = () => {
      if (pending.length > 0) {
        out += Buffer.from(pending).toString("utf8");
        pending = [];
      }
    };
    for (const id of ids) {
      if (id >= RESERVED) {
        flush();
        out += this.pieces[id - RESERVED] ?? "";
      } else if (id < BYTE_COUNT) {
        pending.push(id);
      } else {
        flush(); // BOS/EOS/PAD render as nothing
      }
    }
    flush();
    return out;
  }

  toJSON(): TokenizerData {
    return { version: 1, pieces: this.pieces };
  }

  save(file: string): void {
    writeJsonGz(file, this.toJSON());
  }

  static load(file: string): Tokenizer {
    const data = readJsonGz<TokenizerData>(file);
    return new Tokenizer(data.pieces);
  }
}

const WORD_CHAR = /[A-Za-z0-9_$]/;

/**
 * Split a completion prefix into the settled part and the identifier fragment
 * still being typed. Completion conditions on `stable` and constrains candidates
 * to start with `partial` - that is what makes mid-word completion work instead
 * of predicting from a garbled final token.
 */
export function splitPartial(prefix: string): { stable: string; partial: string } {
  let i = prefix.length;
  while (i > 0 && WORD_CHAR.test(prefix[i - 1])) i--;
  return { stable: prefix.slice(0, i), partial: prefix.slice(i) };
}
