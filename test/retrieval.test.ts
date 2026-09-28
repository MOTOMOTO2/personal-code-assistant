import assert from "node:assert/strict";
import test from "node:test";
import { chunkFile } from "../src/index/chunker.js";
import { IndexStore, searchTerms } from "../src/index/store.js";
import { EmbeddingModel, normalize } from "../src/ml/embeddings.js";
import { Reranker, emptyFeatures, featureIndex } from "../src/ml/reranker.js";
import { StyleAnalyzer } from "../src/core/style.js";
import { Rng } from "../src/util/rng.js";

const TS_SOURCE = `import { thing } from "./thing";

/** Doc comment for the first function. */
export function parseManifest(raw: string): Manifest {
  return JSON.parse(raw);
}

export class UserStore {
  private cache = new Map<string, User>();

  async fetchUser(id: string): Promise<User> {
    const hit = this.cache.get(id);
    if (hit) return hit;
    return this.load(id);
  }

  private async load(id: string): Promise<User> {
    const res = await fetch(\`/users/\${id}\`);
    return res.json();
  }
}

const retryUpload = async (file: File) => {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      return await upload(file);
    } catch {
      continue;
    }
  }
};
`;

test("chunking covers every line of the file", () => {
  const chunks = chunkFile(TS_SOURCE, { targetChars: 200, maxChars: 600 });
  assert.ok(chunks.length > 1, "expected the file to split into several chunks");
  const covered = new Set<number>();
  for (const chunk of chunks) {
    for (let line = chunk.startLine; line <= chunk.endLine; line++) covered.add(line);
  }
  const totalLines = TS_SOURCE.split("\n").length;
  for (let line = 1; line < totalLines; line++) {
    assert.ok(covered.has(line), `line ${line} was not covered by any chunk`);
  }
});

test("chunk text matches the reported line range", () => {
  const lines = TS_SOURCE.split("\n");
  for (const chunk of chunkFile(TS_SOURCE, { targetChars: 200, maxChars: 600 })) {
    const expected = lines.slice(chunk.startLine - 1, chunk.endLine).join("\n");
    assert.equal(chunk.text, expected, `chunk at line ${chunk.startLine} text/range mismatch`);
  }
});

test("declarations are detected as symbols, including class methods", () => {
  const symbols = new Set(
    chunkFile(TS_SOURCE, { targetChars: 100, maxChars: 400 }).flatMap((c) => c.symbols),
  );
  for (const expected of ["parseManifest", "UserStore", "fetchUser", "retryUpload"]) {
    assert.ok(symbols.has(expected), `expected symbol ${expected}, got ${[...symbols].join(", ")}`);
  }
});

test("control flow is not mistaken for a declaration", () => {
  const symbols = chunkFile(
    `function ok() {
  if (a) {
    for (const x of y) {
      while (z) {
      }
    }
  }
}
`,
    { targetChars: 20, maxChars: 80 },
  ).flatMap((c) => c.symbols);
  for (const bad of ["if", "for", "while", "switch", "catch"]) {
    assert.ok(!symbols.includes(bad), `"${bad}" should not be treated as a declaration`);
  }
});

test("search terms include both the whole identifier and its sub-words", () => {
  const terms = new Set(searchTerms("parseManifest snake_case_name XMLHttpRequest"));
  assert.ok(terms.has("parsemanifest"), "whole identifier missing");
  assert.ok(terms.has("parse") && terms.has("manifest"), "camelCase split missing");
  assert.ok(terms.has("snake") && terms.has("case"), "snake_case split missing");
  assert.ok(terms.has("xmlhttprequest"), "acronym identifier missing");
  assert.ok(terms.has("http"), "acronym boundary split missing");
});

function buildStore(): IndexStore {
  const chunks = [
    { id: 0, file: 0, startLine: 1, endLine: 5, symbol: "parseManifest", symbols: ["parseManifest"], text: "function parseManifest(raw) { return JSON.parse(raw); }" },
    { id: 1, file: 0, startLine: 7, endLine: 20, symbol: "UserStore", symbols: ["UserStore"], text: "class UserStore { async fetchUser(id) { return this.load(id); } }" },
    { id: 2, file: 1, startLine: 1, endLine: 12, symbol: "retryUpload", symbols: ["retryUpload"], text: "async function retryUpload(file) { for (let attempt = 0; attempt < 3; attempt++) { try { return await upload(file); } catch {} } }" },
    { id: 3, file: 1, startLine: 14, endLine: 18, symbol: "logger", symbols: ["logger"], text: "const logger = createLogger({ level: 'debug' });" },
  ];
  const files = [
    { path: "src/manifest.ts", size: 100, mtimeMs: 0, tokens: 40 },
    { path: "src/upload.ts", size: 200, mtimeMs: 0, tokens: 80 },
  ];
  return new IndexStore("/repo", files, chunks, IndexStore.buildBm25(chunks));
}

test("BM25 finds the chunk that matches the query", () => {
  const store = buildStore();
  const hits = store.search("retry upload attempt", { limit: 2 });
  assert.ok(hits.length > 0, "expected at least one hit");
  assert.equal(hits[0].chunk.symbol, "retryUpload");
  assert.equal(hits[0].file, "src/upload.ts");
});

test("sub-word matching lets a query find a camelCase symbol", () => {
  const store = buildStore();
  const hits = store.search("fetch user", { limit: 2 });
  assert.equal(hits[0].chunk.symbol, "UserStore");
});

test("search degrades to lexical-only without vectors", () => {
  const store = buildStore();
  assert.equal(store.hasVectors, false);
  const hits = store.search("manifest", { limit: 3, queryVector: null });
  assert.ok(hits.some((h) => h.chunk.symbol === "parseManifest"));
  assert.ok(hits.every((h) => h.vectorRank === null));
});

test("vector search participates in fusion once vectors are attached", () => {
  const store = buildStore();
  const dim = 4;
  const vectors = new Float32Array(store.chunks.length * dim);
  // Chunk 3 is the only one aligned with the query direction.
  vectors.set([0, 0, 0, 1], 3 * dim);
  vectors.set([1, 0, 0, 0], 0 * dim);
  vectors.set([0, 1, 0, 0], 1 * dim);
  vectors.set([0, 0, 1, 0], 2 * dim);
  store.attachVectors(dim, vectors);
  assert.equal(store.hasVectors, true);
  const hits = store.search("nothing lexical matches here", {
    limit: 2,
    queryVector: new Float32Array([0, 0, 0, 1]),
  });
  assert.equal(hits[0].chunk.id, 3);
  assert.equal(hits[0].vectorRank, 1);
});

test("index survives a save/load round trip", async (t) => {
  const os = await import("node:os");
  const fs = await import("node:fs");
  const path = await import("node:path");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pca-index-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const store = buildStore();
  const dim = 4;
  const vectors = new Float32Array(store.chunks.length * dim).fill(0.5);
  store.attachVectors(dim, vectors);
  const indexFile = path.join(dir, "index.json.gz");
  const vectorFile = path.join(dir, "vectors.bin");
  store.save(indexFile, vectorFile);

  const loaded = IndexStore.load(indexFile, vectorFile);
  assert.equal(loaded.chunks.length, store.chunks.length);
  assert.equal(loaded.hasVectors, true);
  assert.deepEqual(loaded.stats(), store.stats());
  assert.deepEqual(
    loaded.search("retry upload", { limit: 2 }).map((h) => h.chunk.id),
    store.search("retry upload", { limit: 2 }).map((h) => h.chunk.id),
  );
});

test("embeddings place co-occurring tokens near each other", () => {
  // Two disjoint "topics": tokens 1-3 always appear together, as do 5-7.
  const rng = new Rng(2);
  const docs: number[][] = [];
  for (let d = 0; d < 60; d++) {
    const doc: number[] = [];
    const topic = d % 2 === 0 ? [1, 2, 3] : [5, 6, 7];
    for (let i = 0; i < 60; i++) doc.push(topic[rng.int(topic.length)]);
    docs.push(doc);
  }
  const model = new EmbeddingModel(10, 16, 5);
  model.countTokens(docs);
  model.train(docs, { dim: 16, epochs: 6, window: 3, negatives: 4 });

  const cos = (a: number, b: number): number => {
    const va = normalize(model.vector(a));
    const vb = normalize(model.vector(b));
    let s = 0;
    for (let i = 0; i < va.length; i++) s += va[i] * vb[i];
    return s;
  };
  const withinTopic = (cos(1, 2) + cos(2, 3) + cos(5, 6) + cos(6, 7)) / 4;
  const acrossTopics = (cos(1, 5) + cos(2, 6) + cos(3, 7)) / 3;
  assert.ok(
    withinTopic > acrossTopics + 0.2,
    `within-topic ${withinTopic.toFixed(3)} should clearly exceed across-topic ${acrossTopics.toFixed(3)}`,
  );
});

test("SIF pooling produces a unit vector and ignores unknown ids", () => {
  const docs = [[1, 2, 3, 1, 2, 3, 4]];
  const model = new EmbeddingModel(8, 8, 1);
  model.countTokens(docs);
  model.train(docs, { dim: 8, epochs: 2 });
  const vec = model.sifEmbed([1, 2, 3, 999, -1]);
  let norm = 0;
  for (const v of vec) norm += v * v;
  assert.ok(Math.abs(Math.sqrt(norm) - 1) < 1e-5, `expected a unit vector, got norm ${norm}`);
});

test("reranker learns to flip a wrong ranking", () => {
  const reranker = new Reranker();
  // The candidate the user wants scores lower on the language model but is a
  // local symbol - exactly the case the learned ranker exists to fix.
  const good = emptyFeatures();
  good[featureIndex("ngramLogProb")] = 0.2;
  good[featureIndex("localFileHits")] = 1;
  const bad = emptyFeatures();
  bad[featureIndex("ngramLogProb")] = 0.9;
  bad[featureIndex("localFileHits")] = 0;

  assert.ok(reranker.score(bad) > reranker.score(good), "prior should initially prefer the LM pick");
  for (let i = 0; i < 200; i++) reranker.learnPairwise(good, [bad], 0.1);
  assert.ok(
    reranker.score(good) > reranker.score(bad),
    "after consistent feedback the accepted candidate should win",
  );
  assert.equal(reranker.accepted, 200);
});

test("reranker save/load preserves learned weights", async (t) => {
  const os = await import("node:os");
  const fs = await import("node:fs");
  const path = await import("node:path");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pca-rank-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const reranker = new Reranker();
  const a = emptyFeatures();
  a[featureIndex("isIdentifier")] = 1;
  reranker.learnPointwise(a, 1, 0.2);
  const file = path.join(dir, "reranker.json.gz");
  reranker.save(file);
  const loaded = Reranker.load(file);
  assert.deepEqual([...loaded.weights], [...reranker.weights]);
  assert.equal(loaded.updates, reranker.updates);
});

test("style analysis recovers the conventions it was shown", () => {
  const analyzer = new StyleAnalyzer();
  const source = [
    "import os from 'os'",
    "",
    "const compute_total = (items) => {",
    "    let total = 0",
    "    for (const item of items) {",
    "        total += item.price",
    "    }",
    "    return total",
    "}",
  ].join("\n");
  for (let i = 0; i < 5; i++) analyzer.add(source, ".js");
  const style = analyzer.finish();
  assert.equal(style.indentKind, "spaces");
  assert.equal(style.indentWidth, 4);
  assert.equal(style.quote, "single");
  assert.equal(style.semicolons, false);
  assert.equal(style.naming, "snake_case");
  assert.deepEqual(style.languages[0], { ext: ".js", files: 5 });
});

test("style analysis detects tabs and semicolons", () => {
  const analyzer = new StyleAnalyzer();
  const source = ['import fs from "fs";', "function getValue() {", "\treturn 1;", "}"].join("\n");
  for (let i = 0; i < 5; i++) analyzer.add(source, ".ts");
  const style = analyzer.finish();
  assert.equal(style.indentKind, "tabs");
  assert.equal(style.quote, "double");
  assert.equal(style.semicolons, true);
});
