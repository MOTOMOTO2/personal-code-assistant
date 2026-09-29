/**
 * A QR encoder, so getting the chat onto your phone is "point the camera at the
 * terminal" rather than "type a URL with a 32-character token into a phone
 * keyboard".
 *
 * Deliberately narrow: byte mode, error correction level L, versions 1 to 5.
 * That covers 106 bytes - a LAN URL with a token is about 60 - and stops before
 * the two parts of the spec that are pure bookkeeping at this scale: multi-block
 * interleaving (level L needs one block up to version 5) and the BCH-coded
 * version information blocks (version 7 and up). A payload that does not fit
 * throws, and the caller falls back to printing the URL.
 *
 * Reed-Solomon over GF(256) with the standard QR primitive polynomial, the eight
 * standard masks scored by the four penalty rules, and format information by
 * BCH(15,5). No dependencies, because the rest of this project has none either.
 */

/** Total and data codewords per version at error-correction level L, one block each. */
const LEVEL_L = [
  { version: 1, total: 26, data: 19 },
  { version: 2, total: 44, data: 34 },
  { version: 3, total: 70, data: 55 },
  { version: 4, total: 100, data: 80 },
  { version: 5, total: 134, data: 108 },
] as const;

/** Error-correction level L, as the two bits that go into the format information. */
const EC_BITS_L = 0b01;
const BCH_FORMAT_GENERATOR = 0b10100110111;
const FORMAT_MASK = 0b101010000010010;
/** x^8 + x^4 + x^3 + x^2 + 1, the field polynomial QR uses. */
const GF_PRIMITIVE = 0x11d;

export const QR_MAX_BYTES = LEVEL_L[LEVEL_L.length - 1].data - 2;

const EXP = new Uint8Array(512);
const LOG = new Uint8Array(256);
(() => {
  let x = 1;
  for (let i = 0; i < 255; i++) {
    EXP[i] = x;
    LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= GF_PRIMITIVE;
  }
  // Doubled so a product of two logs can be indexed without a modulo.
  for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
})();

export function gfMul(a: number, b: number): number {
  if (a === 0 || b === 0) return 0;
  return EXP[LOG[a] + LOG[b]];
}

/** Generator polynomial for `count` error-correction codewords: prod (x - a^i). */
export function rsGenerator(count: number): Uint8Array {
  let poly = new Uint8Array([1]);
  for (let i = 0; i < count; i++) {
    const next = new Uint8Array(poly.length + 1);
    for (let j = 0; j < poly.length; j++) {
      next[j] ^= poly[j];
      next[j + 1] ^= gfMul(poly[j], EXP[i]);
    }
    poly = next;
  }
  return poly;
}

/** Remainder of `data` * x^count divided by the generator: the ECC codewords. */
export function rsRemainder(data: Uint8Array, count: number): Uint8Array {
  const generator = rsGenerator(count);
  const remainder = new Uint8Array(count);
  for (const byte of data) {
    const factor = byte ^ remainder[0];
    remainder.copyWithin(0, 1);
    remainder[count - 1] = 0;
    if (factor !== 0) {
      for (let i = 0; i < count; i++) {
        remainder[i] ^= gfMul(generator[i + 1], factor);
      }
    }
  }
  return remainder;
}

/** 15-bit format information for level L and the given mask. */
export function formatInfo(mask: number): number {
  const data = (EC_BITS_L << 3) | mask;
  let rest = data << 10;
  for (let i = 4; i >= 0; i--) {
    if (rest & (1 << (i + 10))) rest ^= BCH_FORMAT_GENERATOR << i;
  }
  return ((data << 10) | rest) ^ FORMAT_MASK;
}

export interface QrCode {
  version: number;
  size: number;
  mask: number;
  /** Row-major, true is a dark module. */
  modules: boolean[][];
}

export class QrTooLongError extends Error {
  constructor(bytes: number) {
    super(`${bytes} bytes is too long for a version 5-L QR code (max ${QR_MAX_BYTES}).`);
    this.name = "QrTooLongError";
  }
}

export function encodeQr(text: string): QrCode {
  const bytes = new TextEncoder().encode(text);
  const spec = LEVEL_L.find((s) => bytes.length + 2 <= s.data);
  if (!spec) throw new QrTooLongError(bytes.length);

  const codewords = buildCodewords(bytes, spec.data);
  const ecc = rsRemainder(codewords, spec.total - spec.data);
  const all = new Uint8Array(spec.total);
  all.set(codewords, 0);
  all.set(ecc, codewords.length);

  const size = 21 + 4 * (spec.version - 1);
  const reserved = functionModules(size, spec.version);
  const base = drawFunctionPatterns(size, spec.version);
  placeData(base, reserved, all);

  // The spec picks the mask that scores lowest on the four penalty rules; the
  // point is to avoid patterns a scanner could mistake for a finder.
  let best = { mask: 0, penalty: Number.POSITIVE_INFINITY, modules: base };
  for (let mask = 0; mask < 8; mask++) {
    const candidate = base.map((row) => [...row]);
    applyMask(candidate, reserved, mask);
    drawFormatInfo(candidate, mask);
    const penalty = maskPenalty(candidate);
    if (penalty < best.penalty) best = { mask, penalty, modules: candidate };
  }

  return { version: spec.version, size, mask: best.mask, modules: best.modules };
}

/** Mode indicator, length, payload, terminator and the alternating pad bytes. */
function buildCodewords(bytes: Uint8Array, dataCodewords: number): Uint8Array {
  const bits: number[] = [];
  const push = (value: number, width: number) => {
    for (let i = width - 1; i >= 0; i--) bits.push((value >> i) & 1);
  };
  push(0b0100, 4); // byte mode
  push(bytes.length, 8); // 8-bit count, versions 1 to 9
  for (const byte of bytes) push(byte, 8);

  const capacity = dataCodewords * 8;
  for (let i = 0; i < 4 && bits.length < capacity; i++) bits.push(0);
  while (bits.length % 8 !== 0) bits.push(0);

  const out = new Uint8Array(dataCodewords);
  for (let i = 0; i < bits.length; i += 8) {
    let byte = 0;
    for (let b = 0; b < 8; b++) byte = (byte << 1) | bits[i + b];
    out[i / 8] = byte;
  }
  const PAD = [0xec, 0x11];
  for (let i = bits.length / 8, p = 0; i < dataCodewords; i++, p++) out[i] = PAD[p % 2];
  return out;
}

/** True where a module is part of a function pattern and so holds no data. */
export function functionModules(size: number, version: number): boolean[][] {
  const reserved = grid(size);
  const mark = (row: number, col: number, height: number, width: number) => {
    for (let r = row; r < row + height; r++) {
      for (let c = col; c < col + width; c++) {
        if (r >= 0 && r < size && c >= 0 && c < size) reserved[r][c] = true;
      }
    }
  };
  // Finders plus their separators.
  mark(0, 0, 9, 9);
  mark(0, size - 8, 9, 8);
  mark(size - 8, 0, 8, 9);
  // Timing patterns.
  mark(6, 0, 1, size);
  mark(0, 6, size, 1);
  // The single alignment pattern present in versions 2 to 5.
  if (version >= 2) {
    const center = 4 * version + 10;
    mark(center - 2, center - 2, 5, 5);
  }
  return reserved;
}

function drawFunctionPatterns(size: number, version: number): boolean[][] {
  const m = grid(size);
  const finder = (row: number, col: number) => {
    for (let r = -1; r <= 7; r++) {
      for (let c = -1; c <= 7; c++) {
        const rr = row + r;
        const cc = col + c;
        if (rr < 0 || rr >= size || cc < 0 || cc >= size) continue;
        const ring = Math.max(Math.abs(r - 3), Math.abs(c - 3));
        m[rr][cc] = ring !== 2 && ring <= 3;
      }
    }
  };
  finder(0, 0);
  finder(0, size - 7);
  finder(size - 7, 0);

  for (let i = 8; i < size - 8; i++) {
    const dark = i % 2 === 0;
    m[6][i] = dark;
    m[i][6] = dark;
  }

  if (version >= 2) {
    const center = 4 * version + 10;
    for (let r = -2; r <= 2; r++) {
      for (let c = -2; c <= 2; c++) {
        const ring = Math.max(Math.abs(r), Math.abs(c));
        m[center + r][center + c] = ring !== 1;
      }
    }
  }

  // The dark module, always set, just above the bottom-left format strip.
  m[size - 8][8] = true;
  return m;
}

/** The zigzag: two-module columns right to left, alternating up and down. */
function placeData(m: boolean[][], reserved: boolean[][], codewords: Uint8Array): void {
  const size = m.length;
  let bit = 0;
  const next = (): boolean => {
    const index = bit >> 3;
    const value = index < codewords.length ? (codewords[index] >> (7 - (bit & 7))) & 1 : 0;
    bit++;
    return value === 1;
  };
  let upward = true;
  for (let right = size - 1; right >= 1; right -= 2) {
    // Column 6 is the vertical timing pattern; the pairs skip over it.
    const rightCol = right <= 6 ? right - 1 : right;
    for (let step = 0; step < size; step++) {
      const row = upward ? size - 1 - step : step;
      for (const col of [rightCol, rightCol - 1]) {
        if (col < 0 || reserved[row][col]) continue;
        m[row][col] = next();
      }
    }
    upward = !upward;
  }
}

const MASKS: Array<(row: number, col: number) => boolean> = [
  (r, c) => (r + c) % 2 === 0,
  (r) => r % 2 === 0,
  (_r, c) => c % 3 === 0,
  (r, c) => (r + c) % 3 === 0,
  (r, c) => (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0,
  (r, c) => ((r * c) % 2) + ((r * c) % 3) === 0,
  (r, c) => (((r * c) % 2) + ((r * c) % 3)) % 2 === 0,
  (r, c) => (((r + c) % 2) + ((r * c) % 3)) % 2 === 0,
];

function applyMask(m: boolean[][], reserved: boolean[][], mask: number): void {
  const test = MASKS[mask];
  for (let r = 0; r < m.length; r++) {
    for (let c = 0; c < m.length; c++) {
      if (!reserved[r][c] && test(r, c)) m[r][c] = !m[r][c];
    }
  }
}

function drawFormatInfo(m: boolean[][], mask: number): void {
  const size = m.length;
  const format = formatInfo(mask);
  const bitAt = (i: number) => ((format >> i) & 1) === 1;
  for (let i = 0; i < 15; i++) {
    const bit = bitAt(i);
    // Around the top-left finder.
    if (i < 6) m[8][i] = bit;
    else if (i === 6) m[8][7] = bit;
    else if (i === 7) m[8][8] = bit;
    else if (i === 8) m[7][8] = bit;
    else m[14 - i][8] = bit;
    // The second copy: down the left edge, then right of the top-right finder.
    if (i < 7) m[size - 1 - i][8] = bit;
    else m[8][size - 15 + i] = bit;
  }
  m[size - 8][8] = true;
}

/** The four penalty rules from the spec, used only to choose between masks. */
export function maskPenalty(m: boolean[][]): number {
  const size = m.length;
  let penalty = 0;

  // Rule 1: runs of five or more same-coloured modules in a row or column.
  for (let i = 0; i < size; i++) {
    for (const line of [m[i], m.map((row) => row[i])]) {
      let run = 1;
      for (let j = 1; j < size; j++) {
        if (line[j] === line[j - 1]) {
          run++;
          if (run === 5) penalty += 3;
          else if (run > 5) penalty += 1;
        } else {
          run = 1;
        }
      }
    }
  }

  // Rule 2: every 2x2 block of one colour.
  for (let r = 0; r < size - 1; r++) {
    for (let c = 0; c < size - 1; c++) {
      const v = m[r][c];
      if (v === m[r][c + 1] && v === m[r + 1][c] && v === m[r + 1][c + 1]) penalty += 3;
    }
  }

  // Rule 3: the finder-like sequence 1011101 with four light modules on either
  // side, in each direction. Scanners look for that ratio, so it is penalised
  // anywhere outside a real finder pattern.
  const FORWARD = [true, false, true, true, true, false, true, false, false, false, false];
  const BACKWARD = [...FORWARD].reverse();
  const matches = (line: boolean[], at: number, pattern: boolean[]): boolean => {
    for (let i = 0; i < pattern.length; i++) if (line[at + i] !== pattern[i]) return false;
    return true;
  };
  for (let i = 0; i < size; i++) {
    for (const line of [m[i], m.map((row) => row[i])]) {
      for (let at = 0; at + FORWARD.length <= size; at++) {
        if (matches(line, at, FORWARD) || matches(line, at, BACKWARD)) penalty += 40;
      }
    }
  }

  // Rule 4: deviation from an even split of dark and light.
  let dark = 0;
  for (const row of m) for (const v of row) if (v) dark++;
  const percent = (dark * 100) / (size * size);
  penalty += Math.floor(Math.abs(percent - 50) / 5) * 10;
  return penalty;
}

function grid(size: number): boolean[][] {
  return Array.from({ length: size }, () => new Array<boolean>(size).fill(false));
}

export interface RenderOptions {
  /** Modules of white space around the symbol. The spec asks for 4. */
  quietZone?: number;
  /**
   * ANSI colour. Without it the symbol cannot be drawn with the light-on-dark
   * polarity scanners need, so callers should print the URL instead.
   */
  color?: boolean;
}

/**
 * The symbol as terminal text, two module rows per line of output.
 *
 * Half-block characters keep the aspect ratio roughly square (a cell is about
 * twice as tall as it is wide) and halve the height, which is what makes a
 * version 4 symbol fit in a normal terminal window. Colours are explicit black
 * and white rather than the terminal's own: an inverted QR code does not scan.
 */
export function renderQr(code: QrCode, opts: RenderOptions = {}): string {
  const quiet = opts.quietZone ?? 2;
  const size = code.size;
  const span = size + quiet * 2;
  const dark = (row: number, col: number): boolean => {
    const r = row - quiet;
    const c = col - quiet;
    if (r < 0 || r >= size || c < 0 || c >= size) return false;
    return code.modules[r][c];
  };

  const lines: string[] = [];
  for (let row = 0; row < span; row += 2) {
    let line = "";
    for (let col = 0; col < span; col++) {
      const top = dark(row, col);
      const bottom = row + 1 < span ? dark(row + 1, col) : false;
      if (opts.color === false) {
        line += top ? (bottom ? "█" : "▀") : bottom ? "▄" : " ";
      } else {
        // Foreground paints the upper half, background the lower half.
        line += `\u001b[${top ? 30 : 97};${bottom ? 40 : 107}m▀`;
      }
    }
    lines.push(opts.color === false ? line : `${line}\u001b[0m`);
  }
  return lines.join("\n");
}

/** Convenience: encode and render, or null when the text is too long for v5-L. */
export function qrForTerminal(text: string, opts: RenderOptions = {}): string | null {
  try {
    return renderQr(encodeQr(text), opts);
  } catch (err) {
    if (err instanceof QrTooLongError) return null;
    throw err;
  }
}
