import { Bundle, Manifest, MANIFEST_VERSION } from "./bundle.js";
import { AdamW, lrSchedule } from "../ml/params.js";
import { DEFAULT_CONFIG, Transformer, TransformerConfig } from "../ml/transformer.js";
import { TOK_EOS } from "../tokenizer/tokenizer.js";
import { Rng } from "../util/rng.js";
import { fmtDuration, info, ok, progress, progressDone, step, warn } from "../util/log.js";

export interface NeuralTrainOptions {
  /** Wall-clock budget in minutes. Training always stops on time and saves. */
  minutes?: number;
  /** Sequences per optimizer step. */
  batchSize?: number;
  /** Peak learning rate. */
  lr?: number;
  /** Architecture overrides. */
  config?: Partial<TransformerConfig>;
  /** Continue from the existing checkpoint instead of starting fresh. */
  resume?: boolean;
  seed?: number;
  /** Validation windows sampled per evaluation. */
  valWindows?: number;
}

export interface NeuralTrainReport {
  steps: number;
  tokensSeen: number;
  trainLoss: number;
  valLoss: number;
  valPerplexity: number;
  params: number;
  seconds: number;
  improved: boolean;
}

/** Concatenate documents into one stream, separated by EOS. */
function flatten(docs: number[][]): Int32Array {
  let total = 0;
  for (const d of docs) total += d.length + 1;
  const out = new Int32Array(total);
  let p = 0;
  for (const d of docs) {
    for (let i = 0; i < d.length; i++) out[p++] = d[i];
    out[p++] = TOK_EOS;
  }
  return out.subarray(0, p);
}

/**
 * Train the transformer with AdamW under a wall-clock budget.
 *
 * Pure-JS training is around three orders of magnitude slower than a GPU, so the
 * budget is the primary control: `pca train --neural --minutes 10` trains for ten
 * minutes and saves the best validation checkpoint it reached. Quality scales
 * with the budget - the honest framing is that this is a small model that learns
 * *your* code, not a general-purpose coding model.
 */
export function trainNeural(
  bundle: Bundle,
  corpus: { train: number[][]; val: number[][] },
  opts: NeuralTrainOptions = {},
): NeuralTrainReport {
  const startTime = performance.now();
  const budgetMs = (opts.minutes ?? 5) * 60_000;
  const batchSize = opts.batchSize ?? 4;
  const peakLr = opts.lr ?? 3e-3;
  const valWindows = opts.valWindows ?? 24;
  const rng = new Rng(opts.seed ?? 1337);

  const tokenizer = bundle.tokenizer;
  const existing = opts.resume ? bundle.neural : null;
  const model =
    existing ??
    new Transformer({
      ...DEFAULT_CONFIG,
      vocabSize: tokenizer.vocabSize,
      ...opts.config,
    });
  if (model.config.vocabSize !== tokenizer.vocabSize) {
    throw new Error(
      `checkpoint vocabSize ${model.config.vocabSize} does not match tokenizer ${tokenizer.vocabSize}; retrain without --resume`,
    );
  }

  const T = model.config.blockSize;
  const trainStream = flatten(corpus.train);
  const valStream = corpus.val.length > 0 ? flatten(corpus.val) : trainStream;
  if (trainStream.length < T + 2) {
    throw new Error(`corpus too small for neural training (need > ${T + 2} tokens)`);
  }

  step(
    `Neural model: ${model.config.nLayers}L x ${model.config.dModel}d x ${model.config.nHeads}h, ` +
      `ctx ${T}, ${(model.numParams / 1e6).toFixed(2)}M params`,
  );
  if (existing) info("resuming from existing checkpoint");

  const optimizer = new AdamW(model.params, { lr: peakLr, weightDecay: 0.01 });
  // The step count is an estimate for the LR schedule; the real stop is the
  // clock. Measure throughput on the first few steps, then project.
  let plannedSteps = 200;
  const input = new Int32Array(T);
  const target = new Int32Array(T);

  let bestValLoss = Number.POSITIVE_INFINITY;
  let bestSaved = false;
  const baselineVal = existing ? bundle.manifest?.neural?.valLoss ?? Infinity : Infinity;
  let steps = 0;
  let tokensSeen = 0;
  let recentLoss = 0;
  let recentCount = 0;
  let lastReport = startTime;

  for (;;) {
    const elapsed = performance.now() - startTime;
    if (elapsed >= budgetMs) break;

    model.params.zeroGrad();
    let batchLoss = 0;
    for (let b = 0; b < batchSize; b++) {
      const start = rng.int(trainStream.length - T - 1);
      for (let i = 0; i < T; i++) {
        input[i] = trainStream[start + i];
        target[i] = trainStream[start + i + 1];
      }
      batchLoss += model.forward(input, target);
      model.backward(input, target);
      tokensSeen += T;
    }
    batchLoss /= batchSize;
    // Gradients were summed over the batch; average them before stepping.
    const g = model.params.grad;
    for (let i = 0; i < g.length; i++) g[i] /= batchSize;
    optimizer.clipGradNorm(1.0);

    if (steps === 8) {
      // Project the total step count from measured throughput so the cosine
      // schedule actually decays to its floor within the budget.
      const perStep = (performance.now() - startTime) / (steps + 1);
      plannedSteps = Math.max(32, Math.round(budgetMs / perStep));
      info(`throughput ${(T * batchSize / (perStep / 1000)).toFixed(0)} tok/s, planning ~${plannedSteps} steps`);
    }
    const lr = lrSchedule(steps, plannedSteps, peakLr, Math.max(4, Math.round(plannedSteps * 0.03)));
    optimizer.update(lr);

    recentLoss += batchLoss;
    recentCount++;
    steps++;

    const now = performance.now();
    if (now - lastReport > 2000) {
      const avg = recentLoss / Math.max(1, recentCount);
      const pct = Math.min(99, Math.round((now - startTime) / budgetMs * 100));
      progress(
        `step ${steps} · ${pct}% of budget · loss ${avg.toFixed(3)} · ppl ${Math.exp(avg).toFixed(1)} · lr ${lr.toExponential(1)}`,
      );
      lastReport = now;
      recentLoss = 0;
      recentCount = 0;
    }

    // Validate periodically and keep the best checkpoint.
    if (steps % 50 === 0) {
      const valLoss = evaluateStream(model, valStream, valWindows, new Rng(99));
      if (valLoss < bestValLoss) {
        bestValLoss = valLoss;
        model.save(bundle.path("neural"));
        bestSaved = true;
      }
    }
  }
  progressDone();

  const finalVal = evaluateStream(model, valStream, valWindows * 2, new Rng(99));
  if (finalVal <= bestValLoss) {
    bestValLoss = finalVal;
    model.save(bundle.path("neural"));
    bestSaved = true;
  } else if (!bestSaved) {
    model.save(bundle.path("neural"));
    bestSaved = true;
  }
  if (steps === 0) {
    warn("budget elapsed before a single step completed; try a larger --minutes or smaller model");
  }

  const trainLoss = evaluateStream(model, trainStream, valWindows, new Rng(7));
  const seconds = (performance.now() - startTime) / 1000;
  const improved = bestValLoss < baselineVal;

  const previous: Manifest | null = bundle.manifest;
  bundle.writeManifest({
    version: MANIFEST_VERSION,
    trainedAt: previous?.trainedAt ?? new Date().toISOString(),
    vocabSize: previous?.vocabSize ?? tokenizer.vocabSize,
    ngramOrder: previous?.ngramOrder ?? 4,
    corpusTokens: previous?.corpusTokens ?? 0,
    corpusFiles: previous?.corpusFiles ?? 0,
    embeddingDim: previous?.embeddingDim ?? 0,
    neural: {
      trainedAt: new Date().toISOString(),
      steps: (previous?.neural?.steps ?? 0) + steps,
      trainLoss,
      valLoss: bestValLoss,
      valPerplexity: Math.exp(bestValLoss),
      params: model.numParams,
      trainedSeconds: (previous?.neural?.trainedSeconds ?? 0) + seconds,
    },
  });

  ok(
    `Neural: ${steps} steps · ${tokensSeen.toLocaleString()} tokens · ` +
      `val loss ${bestValLoss.toFixed(3)} (ppl ${Math.exp(bestValLoss).toFixed(1)}) · ${fmtDuration(seconds * 1000)}`,
  );

  return {
    steps,
    tokensSeen,
    trainLoss,
    valLoss: bestValLoss,
    valPerplexity: Math.exp(bestValLoss),
    params: model.numParams,
    seconds,
    improved,
  };
}

/** Mean cross-entropy over `windows` random windows of the stream. */
function evaluateStream(
  model: Transformer,
  stream: Int32Array,
  windows: number,
  rng: Rng,
): number {
  const T = model.config.blockSize;
  if (stream.length < T + 2) return Number.NaN;
  const input = new Int32Array(T);
  const target = new Int32Array(T);
  let total = 0;
  let n = 0;
  for (let w = 0; w < windows; w++) {
    const start = rng.int(stream.length - T - 1);
    for (let i = 0; i < T; i++) {
      input[i] = stream[start + i];
      target[i] = stream[start + i + 1];
    }
    total += model.forward(input, target);
    n++;
  }
  return n > 0 ? total / n : Number.NaN;
}
