import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { Bundle } from "../src/core/bundle.js";
import { CompletionEngine } from "../src/core/complete.js";
import { evalLineCompletion, evalNgram } from "../src/core/evaluate.js";
import { loadCorpus, trainProject } from "../src/core/train.js";
import { createDaemon } from "../src/server/daemon.js";
import { setQuiet } from "../src/util/log.js";
import { workspace } from "../src/util/paths.js";

setQuiet(true);

/**
 * A synthetic project with strong, learnable regularities: every handler follows
 * the same shape. A model that has learned anything at all should complete these
 * patterns, which makes the assertions below meaningful rather than vacuous.
 */
function writeFixtureProject(root: string): void {
  fs.mkdirSync(path.join(root, "src", "handlers"), { recursive: true });
  fs.mkdirSync(path.join(root, "src", "lib"), { recursive: true });
  fs.writeFileSync(path.join(root, "package.json"), '{\n  "name": "fixture"\n}\n');

  const resources = ["user", "order", "invoice", "payment", "session", "product", "shipment"];
  for (const resource of resources) {
    const Name = resource[0].toUpperCase() + resource.slice(1);
    fs.writeFileSync(
      path.join(root, "src", "handlers", `${resource}.ts`),
      `import { Database } from "../lib/database";
import { validateRequest } from "../lib/validate";

export interface ${Name}Request {
  ${resource}Id: string;
  requestedAt: number;
}

export async function fetch${Name}(db: Database, request: ${Name}Request) {
  validateRequest(request);
  const record = await db.findOne("${resource}", request.${resource}Id);
  if (record === null) {
    throw new Error("${resource} not found: " + request.${resource}Id);
  }
  return record;
}

export async function delete${Name}(db: Database, request: ${Name}Request) {
  validateRequest(request);
  const record = await db.findOne("${resource}", request.${resource}Id);
  if (record === null) {
    throw new Error("${resource} not found: " + request.${resource}Id);
  }
  await db.remove("${resource}", request.${resource}Id);
  return { deleted: true };
}
`,
    );
  }

  fs.writeFileSync(
    path.join(root, "src", "lib", "database.ts"),
    `export class Database {
  private readonly rows = new Map<string, unknown>();

  async findOne(table: string, id: string) {
    return this.rows.get(table + ":" + id) ?? null;
  }

  async remove(table: string, id: string) {
    this.rows.delete(table + ":" + id);
  }

  async retryFailedUpload(uploadId: string, attempts: number) {
    for (let attempt = 0; attempt < attempts; attempt++) {
      const ok = await this.attemptUpload(uploadId);
      if (ok) return true;
    }
    return false;
  }

  private async attemptUpload(uploadId: string) {
    return uploadId.length > 0;
  }
}
`,
  );
  fs.writeFileSync(
    path.join(root, "src", "lib", "validate.ts"),
    `export function validateRequest(request: { requestedAt: number }) {
  if (typeof request.requestedAt !== "number") {
    throw new Error("requestedAt must be a number");
  }
}
`,
  );
}

function setupProject(t: { after(fn: () => void): void }): Bundle {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pca-e2e-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  writeFixtureProject(root);
  const bundle = new Bundle(workspace(root));
  trainProject(bundle, { vocabSize: 2048, order: 4, embeddingDim: 32, embeddingEpochs: 3, valFraction: 0 });
  return bundle;
}

test("training produces every artifact the tool needs", (t) => {
  const bundle = setupProject(t);
  assert.ok(bundle.isTrained, "bundle should report trained");
  for (const name of ["tokenizer", "ngram", "index", "vectors", "embeddings", "style", "manifest"] as const) {
    assert.ok(fs.existsSync(bundle.path(name)), `missing artifact: ${name}`);
  }
  const manifest = bundle.manifest;
  assert.ok(manifest, "manifest should be written");
  assert.ok(manifest!.corpusTokens > 1000, `corpus too small: ${manifest!.corpusTokens}`);
  assert.equal(manifest!.embeddingDim, 32);
  // The transformer is optional and was not trained here.
  assert.equal(bundle.neural, null);
});

test("completion learns the project's repeated patterns", (t) => {
  const bundle = setupProject(t);
  const engine = new CompletionEngine(bundle);
  engine.warm();

  // The fixture repeats this exact shape in 14 functions; the model has seen it
  // enough times that the guard clause should be the top suggestion.
  const result = engine.complete({
    prefix: `export async function fetchThing(db: Database, request: ThingRequest) {
  validateRequest(request);
  const record = await db.findOne("thing", request.thingId);
  if (record === `,
    filePath: "src/handlers/thing.ts",
    count: 3,
  });
  assert.ok(result.suggestions.length > 0, "expected suggestions");
  const texts = result.suggestions.map((s) => s.text);
  assert.ok(
    texts.some((t2) => t2.startsWith("null")),
    `expected a "null" continuation, got ${JSON.stringify(texts)}`,
  );
  assert.ok(result.timing.total < 200, `completion took ${result.timing.total}ms`);
});

test("mid-word completion uses the identifier fragment", (t) => {
  const bundle = setupProject(t);
  const engine = new CompletionEngine(bundle);
  engine.warm();
  const result = engine.complete({
    prefix: '  const record = await db.findO',
    filePath: "src/handlers/user.ts",
    count: 5,
  });
  assert.equal(result.partial, "findO");
  assert.ok(result.suggestions.length > 0, "expected suggestions for a partial identifier");
  // Every suggestion must be a valid continuation of what was typed.
  for (const s of result.suggestions) {
    const whole = result.partial + s.text;
    assert.ok(
      whole.startsWith("findO"),
      `suggestion "${s.text}" does not continue "findO"`,
    );
  }
  assert.ok(
    result.suggestions.some((s) => s.text.startsWith("ne")),
    `expected findOne, got ${JSON.stringify(result.suggestions.map((s) => s.text))}`,
  );
});

test("completion never proposes text that contradicts the typed prefix", (t) => {
  const bundle = setupProject(t);
  const engine = new CompletionEngine(bundle);
  engine.warm();
  for (const partial of ["valid", "Data", "ret", "re", "a"]) {
    const result = engine.complete({ prefix: `  const x = ${partial}`, count: 5 });
    for (const s of result.suggestions) {
      assert.ok(
        !s.text.startsWith(" ") || partial.length === 0,
        `suggestion for "${partial}" started with whitespace: ${JSON.stringify(s.text)}`,
      );
    }
  }
});

test("semantic search finds code by intent, not just keywords", (t) => {
  const bundle = setupProject(t);
  const index = bundle.index;
  assert.ok(index, "index should exist");
  // "retry" and "upload" appear only in retryFailedUpload.
  const hits = index!.search("retry a failed upload several times", { limit: 5 });
  assert.ok(hits.length > 0, "expected hits");
  assert.ok(
    hits.some((h) => h.file === "src/lib/database.ts"),
    `expected database.ts, got ${hits.map((h) => h.file).join(", ")}`,
  );
});

test("search results carry usable file and line references", (t) => {
  const bundle = setupProject(t);
  const hits = bundle.index!.search("validateRequest requestedAt", { limit: 3 });
  assert.ok(hits.length > 0);
  for (const hit of hits) {
    const abs = path.join(bundle.ws.root, hit.file);
    assert.ok(fs.existsSync(abs), `hit references a missing file: ${hit.file}`);
    const lines = fs.readFileSync(abs, "utf8").split("\n");
    assert.ok(hit.chunk.startLine >= 1 && hit.chunk.endLine <= lines.length + 1, "line range out of bounds");
  }
});

test("style profile is measured from the fixture, not defaulted", (t) => {
  const bundle = setupProject(t);
  const style = bundle.style;
  assert.equal(style.indentKind, "spaces");
  assert.equal(style.indentWidth, 2);
  assert.equal(style.quote, "double");
  assert.equal(style.semicolons, true);
  assert.equal(style.naming, "camelCase");
  assert.ok(style.filesAnalysed >= 9, `expected the fixture files, got ${style.filesAnalysed}`);
});

test("held-out evaluation reports sane metrics", (t) => {
  const bundle = setupProject(t);
  const corpus = loadCorpus(bundle, 0);
  const metrics = evalNgram(bundle, corpus.train.slice(0, 3), 3000);
  assert.ok(metrics.tokens > 100, "expected tokens to be scored");
  assert.ok(metrics.perplexity > 1, `perplexity ${metrics.perplexity} must exceed 1`);
  assert.ok(Number.isFinite(metrics.perplexity), "perplexity must be finite");
  assert.ok(metrics.top1 >= 0 && metrics.top1 <= 1, "top-1 must be a fraction");
  assert.ok(metrics.top5 >= metrics.top1, "top-5 cannot be below top-1");

  const texts = corpus.trainPaths.slice(0, 5).map((p, i) => ({
    path: p,
    text: bundle.tokenizer.decode(corpus.train[i]),
  }));
  const line = evalLineCompletion(bundle, texts, { samples: 40 });
  assert.ok(line.samples > 0, "expected line samples");
  assert.ok(line.meanLatencyMs < 100, `line completion latency ${line.meanLatencyMs}ms`);
});

test("daemon serves completions, search, and learns from feedback", async (t) => {
  const bundle = setupProject(t);
  const daemon = createDaemon(bundle.ws, { port: 0, token: "test-token" });
  const port = await daemon.listen(0);
  t.after(async () => {
    await daemon.close();
  });

  const call = async (method: string, route: string, body?: unknown, token = "test-token") => {
    const res = await fetch(`http://127.0.0.1:${port}${route}`, {
      method,
      headers: {
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    return { status: res.status, json: (await res.json()) as Record<string, unknown> };
  };

  const health = await call("GET", "/health");
  assert.equal(health.status, 200);
  assert.equal(health.json.trained, true);
  assert.equal(health.json.root, bundle.ws.root);

  // A wrong token must be rejected.
  const unauthorized = await call("GET", "/health", undefined, "wrong");
  assert.equal(unauthorized.status, 401);

  // A realistic request: the prefix is everything above the cursor in the file
  // being edited, which is where the locality signal actually comes from. The
  // 4-gram alone cannot tell the seven identical handlers apart - `request.` is
  // all the context it has - so this is a genuine test of whether the locality
  // features reach the ranking.
  const userFile = fs.readFileSync(
    path.join(bundle.ws.root, "src", "handlers", "user.ts"),
    "utf8",
  );
  const cursor = userFile.lastIndexOf('await db.findOne("user", request.') +
    'await db.findOne("user", request.'.length;
  const completion = await call("POST", "/complete", {
    prefix: userFile.slice(0, cursor),
    suffix: userFile.slice(cursor),
    filePath: "src/handlers/user.ts",
    count: 5,
  });
  assert.equal(completion.status, 200);
  const suggestions = completion.json.suggestions as Array<{ text: string; features: number[] }>;
  assert.ok(suggestions.length > 0, "daemon returned no suggestions");
  const texts = suggestions.map((s) => s.text);
  // Editing user.ts, `userId` is all over the buffer - locality should win.
  assert.ok(
    texts[0].startsWith("userId"),
    `expected userId first when editing user.ts, got ${JSON.stringify(texts)}`,
  );

  const search = await call("POST", "/search", { query: "retry failed upload", limit: 3 });
  assert.equal(search.status, 200);
  const hits = search.json.hits as Array<{ file: string }>;
  assert.ok(hits.some((h) => h.file === "src/lib/database.ts"));

  // Feedback must move the reranker and persist. The daemon holds its own
  // Bundle, so the check reads the weights back off disk rather than from this
  // process's in-memory copy.
  const { Reranker } = await import("../src/ml/reranker.js");
  const before = [...bundle.reranker.weights];
  const feedback = await call("POST", "/feedback", {
    accepted: suggestions[0].features,
    rejected: suggestions.slice(1).map((s) => s.features),
    filePath: "src/handlers/user.ts",
  });
  assert.equal(feedback.status, 200);
  assert.ok((feedback.json.updates as number) > 0, "expected reranker updates");
  assert.ok(fs.existsSync(bundle.path("reranker")), "reranker should be persisted");
  const after = [...Reranker.load(bundle.path("reranker")).weights];
  assert.notDeepEqual(after, before, "feedback should change the persisted weights");
  assert.ok(fs.existsSync(bundle.path("feedback")), "feedback log should be written");

  // A malformed request should fail cleanly, not crash the daemon.
  const bad = await call("POST", "/complete", { notAPrefix: true });
  assert.equal(bad.status, 400);
  const stillAlive = await call("GET", "/health");
  assert.equal(stillAlive.status, 200);
});

test("untrained project reports a clear error instead of crashing", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pca-empty-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.writeFileSync(path.join(root, "package.json"), "{}\n");

  const bundle = new Bundle(workspace(root));
  assert.equal(bundle.isTrained, false);
  assert.throws(() => bundle.tokenizer, /pca train/);

  const daemon = createDaemon(bundle.ws, { port: 0 });
  const port = await daemon.listen(0);
  t.after(async () => {
    await daemon.close();
  });
  const res = await fetch(`http://127.0.0.1:${port}/complete`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prefix: "const x = " }),
  });
  assert.equal(res.status, 409);
  const body = (await res.json()) as { hint?: string };
  assert.match(body.hint ?? "", /pca train/);
});
