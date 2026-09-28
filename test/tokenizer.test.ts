import assert from "node:assert/strict";
import test from "node:test";
import { RESERVED, Tokenizer, preTokenize, splitPartial } from "../src/tokenizer/tokenizer.js";

const SAMPLES = [
  "const x = 42;\n",
  "function getUserId(user: User): string { return user.id; }",
  "if (a === b && c !== d) { return a ?? b; }",
  "  \tindented\twith\tmixed\twhitespace\n",
  "def snake_case_name(arg=None):\n    return arg or 'x'\n",
  "// comment with émojis 🚀 and ünïcödé\n",
  "windows\r\nline\r\nendings\r\n",
  "0xFF 0b1010 1_000 3.14159 1e-9 .5",
  "a\u0000b", // NUL byte inside a string
  "x".repeat(500),
  "",
  "🏴󠁧󠁢󠁳󠁣󠁴󠁿", // multi-code-point grapheme cluster
  "const s = `template ${expr} literal`;",
  "}\n})\n];\n",
];

test("pre-tokenization is lossless", () => {
  for (const sample of SAMPLES) {
    assert.equal(preTokenize(sample).join(""), sample, `lost data on: ${JSON.stringify(sample)}`);
  }
});

test("encode/decode round-trips exactly, trained or not", () => {
  const trained = Tokenizer.train(SAMPLES, 512);
  const untrained = new Tokenizer();
  for (const tokenizer of [trained, untrained]) {
    for (const sample of SAMPLES) {
      const ids = tokenizer.encode(sample);
      assert.equal(
        tokenizer.decode(ids),
        sample,
        `round-trip failed (vocab ${tokenizer.vocabSize}) on: ${JSON.stringify(sample)}`,
      );
    }
  }
});

test("out-of-vocabulary text still round-trips through byte fallback", () => {
  // Trained only on English-ish code, then asked to encode text it never saw.
  const tokenizer = Tokenizer.train(["const a = 1;"], 300);
  const unseen = "функция(参数) { return 'ñ'; } 🎯";
  assert.equal(tokenizer.decode(tokenizer.encode(unseen)), unseen);
});

test("frequent pieces become single tokens", () => {
  const corpus = Array.from({ length: 50 }, () => "export function handleRequest(req, res) {\n");
  const tokenizer = Tokenizer.train(corpus, 1024);
  assert.ok(tokenizer.idOf("handleRequest") !== undefined, "frequent identifier should be learned");
  assert.ok(tokenizer.idOf("function") !== undefined, "keyword should be learned");
  // One token per piece, so the line is far shorter than its byte length.
  const ids = tokenizer.encode("export function handleRequest(req, res) {\n");
  assert.ok(ids.length < 20, `expected compact encoding, got ${ids.length} tokens`);
});

test("vocabulary size is respected and ids stay in range", () => {
  const tokenizer = Tokenizer.train(SAMPLES, 400);
  assert.ok(tokenizer.vocabSize <= 400, `vocabSize ${tokenizer.vocabSize} exceeded the budget`);
  for (const sample of SAMPLES) {
    for (const id of tokenizer.encode(sample)) {
      assert.ok(id >= 0 && id < tokenizer.vocabSize, `id ${id} out of range`);
    }
  }
});

test("training is deterministic", () => {
  const a = Tokenizer.train(SAMPLES, 512);
  const b = Tokenizer.train(SAMPLES, 512);
  assert.deepEqual(a.pieces, b.pieces);
});

test("offsets line up with the source text", () => {
  const tokenizer = Tokenizer.train(SAMPLES, 512);
  const text = "const value = compute(a, b);";
  const { ids, offsets } = tokenizer.encodeWithOffsets(text);
  assert.equal(ids.length, offsets.length);
  assert.equal(offsets[0], 0);
  for (let i = 1; i < offsets.length; i++) {
    assert.ok(offsets[i] >= offsets[i - 1], "offsets must be non-decreasing");
    assert.ok(offsets[i] < text.length, "offsets must stay inside the text");
  }
});

test("byte tokens are identified as such", () => {
  const tokenizer = new Tokenizer();
  const ids = tokenizer.encode("z");
  assert.ok(ids.every((id) => tokenizer.isByte(id)));
  const trained = Tokenizer.train(["hello hello hello"], 300);
  const helloId = trained.idOf("hello");
  assert.ok(helloId !== undefined && helloId >= RESERVED);
  assert.equal(trained.isByte(helloId), false);
});

test("splitPartial finds the identifier fragment under the cursor", () => {
  assert.deepEqual(splitPartial("const user = getUse"), {
    stable: "const user = ",
    partial: "getUse",
  });
  assert.deepEqual(splitPartial("obj.prop"), { stable: "obj.", partial: "prop" });
  assert.deepEqual(splitPartial("foo("), { stable: "foo(", partial: "" });
  assert.deepEqual(splitPartial(""), { stable: "", partial: "" });
  assert.deepEqual(splitPartial("  "), { stable: "  ", partial: "" });
  // A number is word-like too, which is what we want for `arr[12`.
  assert.deepEqual(splitPartial("arr[12"), { stable: "arr[", partial: "12" });
});

test("save/load preserves the vocabulary", async (t) => {
  const os = await import("node:os");
  const fs = await import("node:fs");
  const path = await import("node:path");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pca-tok-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const original = Tokenizer.train(SAMPLES, 512);
  const file = path.join(dir, "tok.json.gz");
  original.save(file);
  const loaded = Tokenizer.load(file);
  assert.deepEqual(loaded.pieces, original.pieces);
  for (const sample of SAMPLES) {
    assert.deepEqual(loaded.encode(sample), original.encode(sample));
  }
});
