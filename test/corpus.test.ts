import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { Bundle } from "../src/core/bundle.js";
import { evalLearningCurve, fitScalingLaw } from "../src/core/evaluate.js";
import {
  addSource,
  discoverProjects,
  loadSources,
  readSources,
  removeSource,
  sourceName,
} from "../src/core/sources.js";
import { loadCorpus, trainProject } from "../src/core/train.js";
import { setQuiet } from "../src/util/log.js";
import { workspace } from "../src/util/paths.js";

setQuiet(true);

/**
 * Two synthetic repositories with *different* vocabularies, so "did the source
 * get in" is decidable by looking for words that only exist in one of them.
 */
function writeProject(root: string): void {
  fs.mkdirSync(path.join(root, "src"), { recursive: true });
  for (let i = 0; i < 6; i++) {
    fs.writeFileSync(
      path.join(root, "src", `orders${i}.ts`),
      `export async function loadOrder${i}(db: Database, orderId: string) {
  const order = await db.findOne("orders", orderId);
  if (order === null) throw new NotFoundError("orders", orderId);
  return order;
}
`,
    );
  }
}

function writeLibrary(root: string): void {
  fs.mkdirSync(path.join(root, "lib"), { recursive: true });
  for (let i = 0; i < 6; i++) {
    fs.writeFileSync(
      path.join(root, "lib", `telemetry${i}.ts`),
      `export function recordHistogram${i}(registry: Registry, bucketBoundaries: number[]) {
  registry.observeHistogram("latency", bucketBoundaries);
  return registry.snapshotHistogram("latency");
}
`,
    );
  }
}

function setup(t: { after(fn: () => void): void }): { bundle: Bundle; lib: string } {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pca-corpus-"));
  const lib = fs.mkdtempSync(path.join(os.tmpdir(), "pca-lib-"));
  t.after(() => {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(lib, { recursive: true, force: true });
  });
  writeProject(root);
  writeLibrary(lib);
  return { bundle: new Bundle(workspace(root)), lib };
}

test("a registered source is folded into the corpus and the index", (t) => {
  const { bundle, lib } = setup(t);

  const before = trainProject(bundle, { vocabSize: 1024, embeddingDim: 0, valFraction: 0 });
  assert.equal(before.sources.length, 0);
  assert.equal(before.sourceTokens, 0);

  const source = addSource(bundle.ws, lib, { name: "telemetry" });
  assert.equal(source.name, "telemetry");
  assert.equal(source.index, true);

  bundle.reload();
  const after = trainProject(bundle, { vocabSize: 1024, embeddingDim: 0, valFraction: 0 });
  assert.equal(after.sources.length, 1);
  assert.equal(after.sources[0].files, 6);
  assert.ok(after.sourceTokens > 0, "source should contribute tokens");
  assert.ok(
    after.corpusTokens > before.corpusTokens,
    `corpus should grow: ${before.corpusTokens} -> ${after.corpusTokens}`,
  );

  bundle.reload();
  const index = bundle.index;
  assert.ok(index, "index should exist");
  const external = index!.files.filter((f) => f.external === "telemetry");
  assert.equal(external.length, 6);
  // The display path carries the source, so a borrowed hit is never mistaken
  // for one of yours.
  assert.ok(external.every((f) => f.path.startsWith("@telemetry/")));
});

test("sources train but never validate", (t) => {
  const { bundle, lib } = setup(t);
  addSource(bundle.ws, lib, { name: "telemetry" });
  trainProject(bundle, { vocabSize: 1024, embeddingDim: 0, valFraction: 0.5 });

  bundle.reload();
  const corpus = loadCorpus(bundle, 0.5);
  assert.ok(
    corpus.trainPaths.some((p) => p.startsWith("@telemetry/")),
    "source files should be in the training split",
  );
  assert.ok(
    !corpus.valPaths.some((p) => p.startsWith("@telemetry/")),
    "source files must never be held out - the metric is about your code",
  );
});

test("--no-index keeps a source out of search but in the language model", (t) => {
  const { bundle, lib } = setup(t);
  addSource(bundle.ws, lib, { name: "telemetry", index: false });
  const report = trainProject(bundle, { vocabSize: 1024, embeddingDim: 0, valFraction: 0 });

  assert.ok(report.sourceTokens > 0, "should still train on it");
  bundle.reload();
  assert.equal(
    bundle.index!.files.filter((f) => f.external).length,
    0,
    "an unindexed source should contribute no chunks",
  );
});

test("the token budget bounds what a source contributes", (t) => {
  const { bundle, lib } = setup(t);
  addSource(bundle.ws, lib, { name: "telemetry", maxTokens: 30 });
  const report = trainProject(bundle, { vocabSize: 1024, embeddingDim: 0, valFraction: 0 });
  assert.equal(report.sources[0].truncated, true, "budget should have been hit");
  assert.ok(report.sources[0].files < 6, "not every file should fit in 30 tokens");
});

test("the registry rejects the ways it could silently corrupt the corpus", (t) => {
  const { bundle, lib } = setup(t);

  assert.throws(() => addSource(bundle.ws, path.join(lib, "nope")), /No such directory/);
  assert.throws(
    () => addSource(bundle.ws, path.join(bundle.ws.root, "src")),
    /inside this project/,
    "a path inside the project would be counted twice",
  );

  addSource(bundle.ws, lib, { name: "telemetry" });
  assert.throws(() => addSource(bundle.ws, lib, { name: "other" }), /already registered/);
  assert.throws(() => addSource(bundle.ws, os.tmpdir(), { name: "telemetry" }), /already exists/);
});

test("sources survive a round trip through the registry file", (t) => {
  const { bundle, lib } = setup(t);
  addSource(bundle.ws, lib, { name: "telemetry", maxTokens: 12345, note: "why" });

  const loaded = loadSources(bundle.ws);
  assert.equal(loaded.length, 1);
  assert.equal(loaded[0].maxTokens, 12345);
  assert.equal(loaded[0].note, "why");

  const { docs, stats } = readSources(loaded);
  assert.equal(docs.length, 6);
  assert.equal(stats[0].missing, false);

  assert.equal(removeSource(bundle.ws, "telemetry"), true);
  assert.equal(removeSource(bundle.ws, "telemetry"), false);
  assert.equal(loadSources(bundle.ws).length, 0);
});

test("a source that has vanished from disk is skipped, not fatal", (t) => {
  const { bundle, lib } = setup(t);
  addSource(bundle.ws, lib, { name: "telemetry" });
  fs.rmSync(lib, { recursive: true, force: true });

  const report = trainProject(bundle, { vocabSize: 1024, embeddingDim: 0, valFraction: 0 });
  assert.equal(report.sources[0].missing, true);
  assert.equal(report.sourceTokens, 0);
  assert.ok(report.corpusTokens > 0, "the project itself should still train");
});

test("source names are slugged into safe path prefixes", () => {
  assert.equal(sourceName("/tmp/My Project"), "my-project");
  assert.equal(sourceName("/tmp/react-dom"), "react-dom");
  assert.equal(sourceName("/tmp/@scope"), "scope");
});

test("scanning finds nearby projects and marks the ones already in the corpus", (t) => {
  const { bundle, lib } = setup(t);
  const tree = fs.mkdtempSync(path.join(os.tmpdir(), "pca-scan-"));
  t.after(() => fs.rmSync(tree, { recursive: true, force: true }));

  // Two projects at depth 1, one at depth 2, one directory that is not a
  // project at all, and a subdirectory *inside* a project that must not be
  // reported separately.
  for (const name of ["alpha", "beta"]) {
    fs.mkdirSync(path.join(tree, name, "src"), { recursive: true });
    fs.writeFileSync(path.join(tree, name, "package.json"), "{}");
    fs.writeFileSync(path.join(tree, name, "src", "index.ts"), `export const ${name} = 1;\n`);
  }
  fs.mkdirSync(path.join(tree, "alpha", "packages", "inner"), { recursive: true });
  fs.writeFileSync(path.join(tree, "alpha", "packages", "inner", "package.json"), "{}");
  fs.mkdirSync(path.join(tree, "nested", "gamma", ".git"), { recursive: true });
  fs.writeFileSync(path.join(tree, "nested", "gamma", "main.go"), "package main\n");
  fs.mkdirSync(path.join(tree, "plain", "notes"), { recursive: true });
  fs.writeFileSync(path.join(tree, "plain", "notes", "todo.txt"), "not code\n");

  const found = discoverProjects(bundle.ws, tree, 2);
  const names = found.map((p) => path.basename(p.path)).sort();
  assert.deepEqual(names, ["alpha", "beta", "gamma"]);
  assert.ok(
    !found.some((p) => p.path.includes(`alpha${path.sep}packages`)),
    "a package inside a project is part of it, not a separate source",
  );
  assert.ok(found.every((p) => p.files > 0 || path.basename(p.path) === "gamma"));

  // Sorted largest first, so the most valuable addition is at the top.
  for (let i = 1; i < found.length; i++) {
    assert.ok(found[i - 1].bytes >= found[i].bytes, "results should be sorted by size");
  }

  // Registered sources and the project itself are flagged, not hidden.
  addSource(bundle.ws, path.join(tree, "alpha"), { name: "alpha" });
  const again = discoverProjects(bundle.ws, tree, 2);
  assert.equal(again.find((p) => path.basename(p.path) === "alpha")!.known, true);
  assert.equal(again.find((p) => path.basename(p.path) === "beta")!.known, false);

  assert.throws(() => discoverProjects(bundle.ws, path.join(tree, "gone"), 2), /No such directory/);
  assert.ok(lib.length > 0);
});

test("depth bounds how far the scan looks", (t) => {
  const { bundle } = setup(t);
  const tree = fs.mkdtempSync(path.join(os.tmpdir(), "pca-depth-"));
  t.after(() => fs.rmSync(tree, { recursive: true, force: true }));
  fs.mkdirSync(path.join(tree, "a", "b", "c", "deep"), { recursive: true });
  fs.writeFileSync(path.join(tree, "a", "b", "c", "deep", "package.json"), "{}");

  assert.equal(discoverProjects(bundle.ws, tree, 2).length, 0, "too deep to find at depth 2");
  assert.equal(discoverProjects(bundle.ws, tree, 4).length, 1, "found at depth 4");
});

test("the learning curve is monotone in data and fits a power law", (t) => {
  const { bundle } = setup(t);
  // More files than the default fixture, so the slices are distinguishable.
  for (let i = 0; i < 40; i++) {
    fs.writeFileSync(
      path.join(bundle.ws.root, "src", `handler${i}.ts`),
      `export async function handle${i}(db: Database, request: Request) {
  const record = await db.findOne("table${i % 7}", request.id);
  if (record === null) throw new NotFoundError("table${i % 7}", request.id);
  return { ok: true, record, handler: "handle${i}" };
}
`,
    );
  }
  trainProject(bundle, { vocabSize: 1024, embeddingDim: 0, valFraction: 0.2 });
  bundle.reload();

  const corpus = loadCorpus(bundle, 0.2);
  const curve = evalLearningCurve(bundle, corpus.train, corpus.val, {
    fractions: [0.25, 0.5, 1],
    maxTokens: 1500,
  });

  assert.equal(curve.length, 3);
  for (let i = 1; i < curve.length; i++) {
    assert.ok(curve[i].tokens > curve[i - 1].tokens, "each slice should hold more tokens");
  }
  assert.ok(
    curve[curve.length - 1].top1 >= curve[0].top1,
    `accuracy should not fall with more data: ${curve[0].top1} -> ${curve[curve.length - 1].top1}`,
  );

  const fit = fitScalingLaw(curve);
  assert.ok(fit, "a three-point curve should be fittable");
  assert.ok(Number.isFinite(fit!.alpha));
  assert.ok(fit!.r2 >= 0 && fit!.r2 <= 1, `r2 out of range: ${fit!.r2}`);

  // Reaching a higher accuracy must require at least as much data as the
  // corpus that produced the current one - a projection that says otherwise is
  // arithmetic run backwards.
  const last = curve[curve.length - 1];
  const need = fit!.tokensFor(Math.min(0.94, last.top1 + 0.05));
  if (need !== null && fit!.alpha > 0) {
    assert.ok(need > 0, "projection should be positive");
  }
});

test("a flat curve does not produce a projection", () => {
  // Same accuracy at every size: more data bought nothing, so there is nothing
  // to extrapolate and the fit must say so rather than invent a target.
  const flat = [0.25, 0.5, 1].map((fraction, i) => ({
    fraction,
    documents: 10 * (i + 1),
    tokens: 1000 * (i + 1),
    perplexity: 10,
    top1: 0.4,
    top5: 0.7,
  }));
  const fit = fitScalingLaw(flat);
  assert.ok(fit, "a flat curve is still fittable");
  assert.ok(Math.abs(fit!.alpha) < 1e-9, `expected a zero exponent, got ${fit!.alpha}`);
  assert.equal(fit!.tokensFor(0.5), null, "a non-positive exponent must not project");
});

test("fitting refuses to guess from a single point", () => {
  assert.equal(fitScalingLaw([]), null);
  assert.equal(
    fitScalingLaw([{ fraction: 1, documents: 1, tokens: 100, perplexity: 5, top1: 0.5, top5: 0.8 }]),
    null,
  );
});
