import assert from "node:assert/strict";
import test from "node:test";
import { NgramModel } from "../src/ml/ngram.js";
import { Rng } from "../src/util/rng.js";

function trained(docs: number[][], order = 3, vocabSize = 12): NgramModel {
  const model = new NgramModel({ order });
  model.vocabSize = vocabSize;
  for (const doc of docs) model.observe(doc);
  model.finalize();
  return model;
}

test("conditional distribution sums to one", () => {
  const rng = new Rng(3);
  const docs = Array.from({ length: 12 }, () =>
    Array.from({ length: 60 }, () => rng.int(12)),
  );
  const model = trained(docs, 3, 12);
  for (const history of [[1], [2, 3], [4, 5, 6], [], [11, 11, 11]]) {
    let total = 0;
    for (let w = 0; w < model.vocabSize; w++) total += model.prob(history, w);
    assert.ok(
      Math.abs(total - 1) < 1e-6,
      `history ${JSON.stringify(history)} summed to ${total}, not 1`,
    );
  }
});

test("probabilities are strictly positive for every token", () => {
  // Kneser-Ney must never assign zero mass, or perplexity becomes infinite on
  // any held-out token the corpus happens not to contain.
  const model = trained([[1, 2, 3, 1, 2, 3, 1, 2, 3]], 3, 12);
  for (let w = 0; w < model.vocabSize; w++) {
    assert.ok(model.prob([1, 2], w) > 0, `token ${w} had zero probability`);
    assert.ok(Number.isFinite(model.logProb([1, 2], w)), `token ${w} had non-finite log-prob`);
  }
});

test("a deterministic cycle is predicted exactly", () => {
  const cycle: number[] = [];
  for (let i = 0; i < 300; i++) cycle.push((i % 5) + 1);
  const model = trained([cycle], 4, 12);
  for (let start = 0; start < 5; start++) {
    const history = [((start + 2) % 5) + 1, ((start + 3) % 5) + 1, ((start + 4) % 5) + 1];
    const expected = ((start + 5) % 5) + 1;
    const top = model.predict(history, 1);
    assert.equal(top[0]?.id, expected, `after ${history} expected ${expected}`);
    assert.ok(model.prob(history, expected) > 0.8, "a memorised cycle should be near-certain");
  }
});

test("higher order lowers perplexity on structured data", () => {
  const rng = new Rng(17);
  // Data with real 3-gram structure: token depends on the two before it.
  const docs: number[][] = [];
  for (let d = 0; d < 20; d++) {
    const doc = [rng.int(6), rng.int(6)];
    for (let i = 2; i < 200; i++) doc.push((doc[i - 1] + doc[i - 2]) % 6);
    docs.push(doc);
  }
  const held = docs.pop()!;
  const perplexity = (order: number): number => {
    const model = trained(docs, order, 6);
    let sum = 0;
    let n = 0;
    for (let i = 2; i < held.length; i++) {
      sum += model.logProb(held.slice(Math.max(0, i - 4), i), held[i]);
      n++;
    }
    return Math.exp(-sum / n);
  };
  const unigram = perplexity(1);
  const trigram = perplexity(3);
  assert.ok(
    trigram < unigram * 0.5,
    `trigram (${trigram.toFixed(2)}) should beat unigram (${unigram.toFixed(2)}) decisively`,
  );
});

test("contextFor + probIn agree with the direct path", () => {
  const rng = new Rng(8);
  const docs = Array.from({ length: 6 }, () => Array.from({ length: 80 }, () => rng.int(10)));
  const model = trained(docs, 4, 10);
  const history = [3, 7, 1];
  const ctx = model.contextFor(history);
  for (let w = 0; w < model.vocabSize; w++) {
    assert.ok(
      Math.abs(model.probIn(ctx, w) - model.prob(history, w)) < 1e-12,
      `cached context disagreed for token ${w}`,
    );
  }
});

test("matchedOrder reports how much real evidence exists", () => {
  const model = trained([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]], 4, 12);
  // A context the model has seen many times resolves at full order.
  assert.equal(model.contextFor([1, 2, 3]).matchedOrder, 4);
  // A context it has never seen falls back to the unigram only.
  assert.equal(model.contextFor([9, 9, 9]).matchedOrder, 1);
});

test("discounts stay inside (0, 1)", () => {
  const rng = new Rng(21);
  for (const seedDocs of [1, 5, 20]) {
    const docs = Array.from({ length: seedDocs }, () =>
      Array.from({ length: 50 }, () => rng.int(8)),
    );
    const model = trained(docs, 3, 8);
    // Exercised indirectly: an out-of-range discount produces negative or
    // non-normalised probabilities, both of which the assertions below catch.
    let total = 0;
    for (let w = 0; w < model.vocabSize; w++) {
      const p = model.prob([1, 2], w);
      assert.ok(p > 0 && p <= 1, `probability ${p} out of range`);
      total += p;
    }
    assert.ok(Math.abs(total - 1) < 1e-6, `distribution summed to ${total}`);
  }
});

test("save/load preserves predictions", async (t) => {
  const os = await import("node:os");
  const fs = await import("node:fs");
  const path = await import("node:path");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pca-ngram-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const rng = new Rng(4);
  const docs = Array.from({ length: 8 }, () => Array.from({ length: 90 }, () => rng.int(11)));
  const original = trained(docs, 4, 11);
  const file = path.join(dir, "ngram.json.gz");
  original.save(file);
  const loaded = NgramModel.load(file);

  assert.equal(loaded.order, original.order);
  assert.equal(loaded.vocabSize, original.vocabSize);
  assert.deepEqual(loaded.stats(), original.stats());
  for (const history of [[1], [2, 4], [5, 6, 7]]) {
    for (let w = 0; w < original.vocabSize; w++) {
      assert.ok(
        Math.abs(loaded.prob(history, w) - original.prob(history, w)) < 1e-9,
        `probability drifted for ${JSON.stringify(history)} -> ${w}`,
      );
    }
    assert.deepEqual(loaded.predict(history, 5), original.predict(history, 5));
  }
});

test("empty and single-token documents do not break training", () => {
  const model = new NgramModel({ order: 4 });
  model.vocabSize = 10;
  model.observe([]);
  model.observe([5]);
  model.observe([1, 2]);
  model.finalize();
  assert.ok(model.prob([], 5) > 0);
  assert.ok(Number.isFinite(model.logProb([1], 2)));
});
