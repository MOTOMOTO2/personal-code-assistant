import assert from "node:assert/strict";
import test from "node:test";
import { AdamW } from "../src/ml/params.js";
import { Transformer } from "../src/ml/transformer.js";
import { Rng } from "../src/util/rng.js";
import { gelu, geluGrad } from "../src/ml/linalg.js";

/**
 * Gradient correctness is checked with *directional* derivatives rather than
 * per-element finite differences. Parameters are float32, so perturbing one
 * element by a step large enough to register also moves the loss by less than
 * the accumulated rounding noise. A directional derivative along a unit vector
 * aggregates the whole tensor, giving a signal far above the noise floor while
 * still failing if any single element's gradient is wrong.
 */
function directionalCheck(
  model: Transformer,
  tokens: Int32Array,
  targets: Int32Array,
  direction: Float32Array,
  h = 1e-3,
): { analytic: number; numeric: number } {
  const params = model.params.data;
  const original = Float32Array.from(params);

  model.params.zeroGrad();
  model.forward(tokens, targets);
  model.backward(tokens, targets);
  let analytic = 0;
  for (let i = 0; i < params.length; i++) analytic += model.params.grad[i] * direction[i];

  for (let i = 0; i < params.length; i++) params[i] = original[i] + h * direction[i];
  const lossPlus = model.forward(tokens, targets);
  for (let i = 0; i < params.length; i++) params[i] = original[i] - h * direction[i];
  const lossMinus = model.forward(tokens, targets);
  params.set(original);

  return { analytic, numeric: (lossPlus - lossMinus) / (2 * h) };
}

function randomDirection(size: number, rng: Rng, restrict?: { offset: number; length: number }): Float32Array {
  const d = new Float32Array(size);
  const from = restrict ? restrict.offset : 0;
  const to = restrict ? restrict.offset + restrict.length : size;
  let norm = 0;
  for (let i = from; i < to; i++) {
    d[i] = rng.normal();
    norm += d[i] * d[i];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = from; i < to; i++) d[i] /= norm;
  return d;
}

function tinyModel(): Transformer {
  return new Transformer({
    vocabSize: 23,
    dModel: 8,
    nHeads: 2,
    nLayers: 2,
    dFF: 16,
    blockSize: 6,
  });
}

test("GELU derivative matches finite differences", () => {
  for (const x of [-3, -1.2, -0.3, 0, 0.4, 1.7, 3.5]) {
    const h = 1e-4;
    const numeric = (gelu(x + h) - gelu(x - h)) / (2 * h);
    assert.ok(
      Math.abs(numeric - geluGrad(x)) < 1e-4,
      `geluGrad(${x}) = ${geluGrad(x)} but finite difference = ${numeric}`,
    );
  }
});

test("backward pass matches numeric gradients along random directions", () => {
  const model = tinyModel();
  const rng = new Rng(11);
  const T = model.config.blockSize;
  const tokens = new Int32Array(T);
  const targets = new Int32Array(T);
  for (let i = 0; i < T; i++) tokens[i] = rng.int(model.config.vocabSize);
  for (let i = 0; i < T; i++) targets[i] = rng.int(model.config.vocabSize);

  for (let trial = 0; trial < 4; trial++) {
    const d = randomDirection(model.params.size, rng);
    const { analytic, numeric } = directionalCheck(model, tokens, targets, d);
    const relative = Math.abs(analytic - numeric) / Math.max(1e-4, Math.abs(numeric));
    assert.ok(
      relative < 0.02,
      `trial ${trial}: analytic ${analytic.toFixed(6)} vs numeric ${numeric.toFixed(6)} (rel ${relative.toFixed(4)})`,
    );
  }
});

test("every parameter tensor receives a correct gradient", () => {
  const model = tinyModel();
  const rng = new Rng(5);
  const T = model.config.blockSize;
  const tokens = new Int32Array(T);
  const targets = new Int32Array(T);
  for (let i = 0; i < T; i++) tokens[i] = 3 + (i % 7);
  for (let i = 0; i < T; i++) targets[i] = 4 + (i % 9);

  for (const spec of model.params.allSpecs()) {
    const d = randomDirection(model.params.size, rng, {
      offset: spec.offset,
      length: spec.size,
    });
    let norm = 0;
    for (const v of d) norm += v * v;
    assert.ok(norm > 0.9, `${spec.name}: direction vector was not built`);
    const { analytic, numeric } = directionalCheck(model, tokens, targets, d);
    // A tensor whose gradient is genuinely ~0 (e.g. unused embedding rows) is
    // fine; what matters is that analytic and numeric agree.
    const scale = Math.max(1e-3, Math.abs(numeric), Math.abs(analytic));
    const relative = Math.abs(analytic - numeric) / scale;
    assert.ok(
      relative < 0.05,
      `${spec.name}: analytic ${analytic.toExponential(3)} vs numeric ${numeric.toExponential(3)} (rel ${relative.toFixed(4)})`,
    );
  }
});

test("training drives loss down on a learnable pattern", () => {
  const model = new Transformer({
    vocabSize: 16,
    dModel: 32,
    nHeads: 4,
    nLayers: 2,
    dFF: 64,
    blockSize: 16,
  });
  // A deterministic cycle: the model can reach near-zero loss only if the
  // forward pass, the backward pass and the optimizer all agree.
  const period = 5;
  const T = model.config.blockSize;
  const tokens = new Int32Array(T);
  const targets = new Int32Array(T);
  for (let i = 0; i < T; i++) tokens[i] = (i % period) + 1;
  for (let i = 0; i < T; i++) targets[i] = ((i + 1) % period) + 1;

  const optimizer = new AdamW(model.params, { lr: 0.02, weightDecay: 0 });
  const first = model.forward(tokens, targets);
  let last = first;
  for (let step = 0; step < 120; step++) {
    model.params.zeroGrad();
    last = model.forward(tokens, targets);
    model.backward(tokens, targets);
    optimizer.clipGradNorm(1.0);
    optimizer.update();
  }
  assert.ok(first > 2.0, `expected a high initial loss on 16 classes, got ${first}`);
  assert.ok(last < first * 0.25, `loss should collapse: ${first.toFixed(3)} -> ${last.toFixed(3)}`);
});

test("checkpoint round-trips exactly", async (t) => {
  const os = await import("node:os");
  const fs = await import("node:fs");
  const path = await import("node:path");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pca-test-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const model = tinyModel();
  const tokens = new Int32Array([1, 2, 3, 4, 5, 6]);
  const before = model.nextLogProbs(tokens);
  const file = path.join(dir, "model.pcam");
  model.save(file);

  const reloaded = Transformer.load(file);
  assert.deepEqual(reloaded.config, model.config);
  const after = reloaded.nextLogProbs(tokens);
  for (let i = 0; i < before.length; i++) {
    assert.ok(Math.abs(before[i] - after[i]) < 1e-6, `logprob ${i} drifted`);
  }
});

test("nextLogProbs returns a normalised distribution", () => {
  const model = tinyModel();
  const logProbs = model.nextLogProbs(new Int32Array([7, 8, 9]));
  let sum = 0;
  for (const lp of logProbs) sum += Math.exp(lp);
  assert.ok(Math.abs(sum - 1) < 1e-4, `probabilities summed to ${sum}`);
});

test("sequences longer than the context window are truncated, not rejected", () => {
  const model = tinyModel();
  const long = new Int32Array(model.config.blockSize * 3);
  for (let i = 0; i < long.length; i++) long[i] = i % model.config.vocabSize;
  const logProbs = model.nextLogProbs(long);
  assert.equal(logProbs.length, model.config.vocabSize);
});
