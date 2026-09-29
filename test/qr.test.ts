import assert from "node:assert/strict";
import test from "node:test";
import {
  QR_MAX_BYTES,
  QrTooLongError,
  encodeQr,
  formatInfo,
  functionModules,
  gfMul,
  renderQr,
  rsRemainder,
  qrForTerminal,
} from "../src/web/qr.js";

/**
 * The encoder is only useful if a scanner can read what it draws, and there is
 * no camera in a test run. So these tests decode the symbol back independently:
 * the reader below re-walks the zigzag, unmasks, checks the Reed-Solomon
 * syndrome and parses the byte-mode segment, without reusing the encoder's
 * placement code. A round trip through that, plus the spec's own published
 * format-information constant, is as close to "my phone reads it" as an offline
 * test can get.
 */

interface Decoded {
  text: string;
  mask: number;
  version: number;
  /** Mode indicator read from the first codeword; 0b0100 is byte mode. */
  mode: number;
  /** Zero when the codewords are consistent with their error-correction data. */
  syndrome: number;
}

/** The error-correction level as recorded in the format information. */
function ecLevelOf(modules: boolean[][]): number {
  let format = 0;
  const positions: Array<[number, number]> = [];
  for (let i = 0; i < 6; i++) positions.push([8, i]);
  positions.push([8, 7], [8, 8], [7, 8]);
  for (let i = 9; i < 15; i++) positions.push([14 - i, 8]);
  for (let i = 0; i < 15; i++) {
    const [r, c] = positions[i];
    if (modules[r][c]) format |= 1 << i;
  }
  return ((format ^ 0b101010000010010) >> 13) & 0b11;
}

function decode(modules: boolean[][]): Decoded {
  const size = modules.length;
  const version = (size - 21) / 4 + 1;
  const spec = [
    { total: 26, data: 19 },
    { total: 44, data: 34 },
    { total: 70, data: 55 },
    { total: 100, data: 80 },
    { total: 134, data: 108 },
  ][version - 1];

  // Format information, top-left copy, unmasked with the spec's constant.
  let format = 0;
  const bitPositions: Array<[number, number]> = [];
  for (let i = 0; i < 6; i++) bitPositions.push([8, i]);
  bitPositions.push([8, 7], [8, 8], [7, 8]);
  for (let i = 9; i < 15; i++) bitPositions.push([14 - i, 8]);
  for (let i = 0; i < 15; i++) {
    const [r, c] = bitPositions[i];
    if (modules[r][c]) format |= 1 << i;
  }
  format ^= 0b101010000010010;
  const mask = (format >> 10) & 0b111;

  const maskFns = [
    (r: number, c: number) => (r + c) % 2 === 0,
    (r: number) => r % 2 === 0,
    (_r: number, c: number) => c % 3 === 0,
    (r: number, c: number) => (r + c) % 3 === 0,
    (r: number, c: number) => (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0,
    (r: number, c: number) => ((r * c) % 2) + ((r * c) % 3) === 0,
    (r: number, c: number) => (((r * c) % 2) + ((r * c) % 3)) % 2 === 0,
    (r: number, c: number) => (((r + c) % 2) + ((r * c) % 3)) % 2 === 0,
  ];
  const isFunction = functionModules(size, version);
  const value = (r: number, c: number): boolean =>
    maskFns[mask](r, c) ? !modules[r][c] : modules[r][c];

  // Walk the zigzag: upward and downward two-module columns, right to left,
  // stepping over the vertical timing pattern in column 6.
  const bits: boolean[] = [];
  let up = true;
  for (let right = size - 1; right >= 1; right -= 2) {
    const rightCol = right <= 6 ? right - 1 : right;
    for (let step = 0; step < size; step++) {
      const row = up ? size - 1 - step : step;
      for (const col of [rightCol, rightCol - 1]) {
        if (col < 0 || isFunction[row][col]) continue;
        bits.push(value(row, col));
      }
    }
    up = !up;
  }

  const codewords = new Uint8Array(spec.total);
  for (let i = 0; i < spec.total; i++) {
    let byte = 0;
    for (let b = 0; b < 8; b++) byte = (byte << 1) | (bits[i * 8 + b] ? 1 : 0);
    codewords[i] = byte;
  }

  // A correct codeword stream divides exactly by the generator polynomial.
  const remainder = rsRemainder(codewords, spec.total - spec.data);
  const syndrome = remainder.reduce((acc, b) => acc | b, 0);

  // Deliberately tolerant: a corrupted symbol is decoded far enough to show a
  // non-zero syndrome rather than throwing on its scrambled header.
  const mode = codewords[0] >> 4;
  const declared = ((codewords[0] & 0x0f) << 4) | (codewords[1] >> 4);
  const length = Math.min(declared, spec.data - 2);
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    bytes[i] = ((codewords[i + 1] & 0x0f) << 4) | (codewords[i + 2] >> 4);
  }
  return { text: new TextDecoder().decode(bytes), mask, version, mode, syndrome };
}

test("a QR symbol decodes back to the URL it was made from", () => {
  const inputs = [
    "http://192.168.1.42:7717/?t=0a1b2c3d4e5f60718293a4b5c6d7e8f9",
    "pca",
    "http://10.0.0.7:7717/",
    "http://192.168.100.200:65535/?t=" + "ab".repeat(16),
  ];
  for (const text of inputs) {
    const code = encodeQr(text);
    const decoded = decode(code.modules);
    assert.equal(decoded.mode, 0b0100, "byte mode");
    assert.equal(decoded.text, text, `round trip for ${text}`);
    assert.equal(decoded.mask, code.mask, "the decoder must recover the chosen mask");
    assert.equal(decoded.syndrome, 0, "error correction codewords must be consistent");
    assert.equal(decoded.version, code.version);
    // The two format bits above the mask say level L in every symbol.
    assert.equal(ecLevelOf(code.modules), 0b01, "error correction level L");
    assert.equal(code.size, 21 + 4 * (code.version - 1));
  }
});

test("the symbol has the structure a scanner looks for", () => {
  const code = encodeQr("http://192.168.1.42:7717/?t=deadbeef");
  const m = code.modules;
  const size = code.size;

  // Finder patterns: dark 7x7 ring, light ring inside, dark 3x3 core.
  for (const [r0, c0] of [
    [0, 0],
    [0, size - 7],
    [size - 7, 0],
  ]) {
    for (let r = 0; r < 7; r++) {
      for (let c = 0; c < 7; c++) {
        const ring = Math.max(Math.abs(r - 3), Math.abs(c - 3));
        assert.equal(m[r0 + r][c0 + c], ring !== 2, `finder at ${r0},${c0} module ${r},${c}`);
      }
    }
  }
  // Separators: the row and column just inside each finder are light.
  for (let i = 0; i < 8; i++) {
    assert.equal(m[7][i], false, "top-left separator row");
    assert.equal(m[i][7], false, "top-left separator column");
  }
  // Timing patterns alternate, starting and ending dark.
  for (let i = 8; i < size - 8; i++) {
    assert.equal(m[6][i], i % 2 === 0, `horizontal timing at ${i}`);
    assert.equal(m[i][6], i % 2 === 0, `vertical timing at ${i}`);
  }
  // The dark module is always set.
  assert.equal(m[size - 8][8], true, "dark module");
  // Alignment pattern for version 2 and up.
  if (code.version >= 2) {
    const center = 4 * code.version + 10;
    assert.equal(m[center][center], true, "alignment centre");
    assert.equal(m[center][center + 1], false, "alignment light ring");
    assert.equal(m[center - 2][center - 2], true, "alignment outer ring");
  }
});

test("error correction is real, and the published format constant matches", () => {
  // The spec tabulates level L with mask 0 as 111011111000100; a BCH bug would
  // not reproduce that by accident.
  assert.equal(formatInfo(0), 0b111011111000100);
  for (let mask = 0; mask < 8; mask++) {
    const info = formatInfo(mask);
    assert.equal((info ^ 0b101010000010010) >> 10, (0b01 << 3) | mask, `mask ${mask} round trip`);
  }

  // GF(256) basics: 2^7 is still 128, 2^8 wraps by the field polynomial 0x11d.
  assert.equal(gfMul(64, 2), 128);
  assert.equal(gfMul(128, 2), 0x1d);
  assert.equal(gfMul(3, 7), 9);

  // Corrupting a codeword must show up as a non-zero syndrome, or the error
  // correction data is not being computed over the payload at all.
  const code = encodeQr("http://192.168.1.42:7717/?t=cafe");
  const clean = decode(code.modules);
  assert.equal(clean.syndrome, 0);

  const damaged = code.modules.map((row) => [...row]);
  const isFunction = functionModules(code.size, code.version);
  let flipped = 0;
  for (let r = code.size - 1; r >= 0 && flipped < 3; r--) {
    for (let c = code.size - 1; c >= 0 && flipped < 3; c--) {
      if (!isFunction[r][c]) {
        damaged[r][c] = !damaged[r][c];
        flipped++;
      }
    }
  }
  assert.notEqual(decode(damaged).syndrome, 0, "three flipped modules must be detectable");
});

test("versions grow with the payload, and an oversized payload is refused", () => {
  assert.equal(encodeQr("x").version, 1);
  assert.equal(encodeQr("x".repeat(17)).version, 1);
  assert.equal(encodeQr("x".repeat(18)).version, 2);
  assert.equal(encodeQr("x".repeat(32)).version, 2);
  assert.equal(encodeQr("x".repeat(53)).version, 3);
  assert.equal(encodeQr("x".repeat(78)).version, 4);
  assert.equal(encodeQr("x".repeat(QR_MAX_BYTES)).version, 5);
  assert.throws(() => encodeQr("x".repeat(QR_MAX_BYTES + 1)), QrTooLongError);
  // Multi-byte characters count as bytes, not characters.
  assert.equal(encodeQr("é".repeat(9)).version, 2);

  // The caller that prints a URL wants a fallback, not an exception.
  assert.equal(qrForTerminal("x".repeat(QR_MAX_BYTES + 1)), null);
  const plain = qrForTerminal("http://192.168.1.42:7717/", { color: false });
  assert.ok(plain && plain.includes("█"), "monochrome rendering uses block characters");
  const colored = renderQr(encodeQr("http://192.168.1.42:7717/"));
  assert.match(colored, /\u001b\[/, "coloured rendering carries ANSI codes");
  // Two module rows per line of output, plus the quiet zone.
  assert.equal(colored.split("\n").length, Math.ceil((encodeQr("http://192.168.1.42:7717/").size + 4) / 2));
});
