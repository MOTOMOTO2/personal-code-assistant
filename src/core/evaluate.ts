import { Bundle } from "./bundle.js";
import { CompletionEngine } from "./complete.js";
import { Rng } from "../util/rng.js";
import { progress, progressDone } from "../util/log.js";

/**
 * Held-out evaluation. Without this, "training" is just a progress bar.
 *
 * Three metrics, each answering a different question:
 *   - perplexity: how surprised is the language model by unseen code?
 *   - top-1 / top-5 next-token accuracy: would the first suggestion be right?
 *   - line-completion exact match: from a real mid-line cursor, does the
 *     suggestion match what the developer actually typed? This is the metric
 *     that correlates with the feature being useful.
 */

export interface TokenMetrics {
  tokens: number;
  perplexity: number;
  top1: number;
  top5: number;
}

export interface LineMetrics {
  samples: number;
  /** Suggestion is a prefix of the real continuation (any length). */
  prefixMatch: number;
  /** Suggestion exactly equals the rest of the line. */
  exactLine: number;
  /** Mean characters correctly predicted before the first mistake. */
  meanCorrectChars: number;
  /** Mean characters offered per suggestion. */
  meanOfferedChars: number;
  /** Mean end-to-end latency in milliseconds. */
  meanLatencyMs: number;
  p95LatencyMs: number;
}

export interface EvalReport {
  ngram: TokenMetrics;
  neural: TokenMetrics | null;
  /** n-gram and neural blended by log-linear interpolation. */
  hybrid: TokenMetrics | null;
  line: LineMetrics;
  valFiles: number;
}

/** Next-token metrics for the n-gram model over held-out documents. */
export function evalNgram(bundle: Bundle, docs: number[][], maxTokens = 20_000): TokenMetrics {
  const ngram = bundle.ngram;
  let logProbSum = 0;
  let n = 0;
  let top1 = 0;
  let top5 = 0;
  const history: number[] = [];
  outer: for (const doc of docs) {
    history.length = 0;
    for (let i = 0; i < doc.length; i++) {
      const target = doc[i];
      if (history.length > 0) {
        logProbSum += ngram.logProb(history, target);
        const preds = ngram.predict(history, 5, 128);
        if (preds.length > 0 && preds[0].id === target) top1++;
        if (preds.some((p) => p.id === target)) top5++;
        n++;
        if (n % 2000 === 0) progress(`n-gram eval ${n}/${maxTokens}`);
        if (n >= maxTokens) break outer;
      }
      history.push(target);
      if (history.length > 8) history.shift();
    }
  }
  progressDone();
  return {
    tokens: n,
    perplexity: n > 0 ? Math.exp(-logProbSum / n) : Number.NaN,
    top1: n > 0 ? top1 / n : 0,
    top5: n > 0 ? top5 / n : 0,
  };
}

/**
 * Next-token metrics for the transformer, and for the log-linear blend of both
 * models. Evaluated on the same positions so the comparison is fair.
 */
export function evalNeuralAndHybrid(
  bundle: Bundle,
  docs: number[][],
  maxTokens = 2000,
  blendWeight = 0.5,
): { neural: TokenMetrics; hybrid: TokenMetrics } | null {
  const neural = bundle.neural;
  if (!neural) return null;
  const ngram = bundle.ngram;
  const T = neural.config.blockSize;
  const V = neural.config.vocabSize;

  let nLogProb = 0;
  let hLogProb = 0;
  let nTop1 = 0;
  let hTop1 = 0;
  let nTop5 = 0;
  let hTop5 = 0;
  let n = 0;
  const blend = new Float32Array(V);

  outer: for (const doc of docs) {
    // Stride through the document so evaluation covers varied context lengths
    // without paying for every position (each one is a full forward pass).
    for (let pos = 8; pos < doc.length - 1; pos += Math.max(1, Math.floor(T / 4))) {
      const start = Math.max(0, pos - T);
      const window = doc.slice(start, pos);
      if (window.length === 0) continue;
      const target = doc[pos];
      const logProbs = neural.nextLogProbs(window);

      nLogProb += logProbs[target] ?? -20;
      const nTop = topK(logProbs, 5);
      if (nTop[0] === target) nTop1++;
      if (nTop.includes(target)) nTop5++;

      // Log-linear blend: w * neural + (1 - w) * n-gram, renormalised.
      const ngramHistory = window.slice(Math.max(0, window.length - 8));
      let maxV = -Infinity;
      for (let j = 0; j < V; j++) {
        const lp = blendWeight * logProbs[j] + (1 - blendWeight) * ngram.logProb(ngramHistory, j);
        blend[j] = lp;
        if (lp > maxV) maxV = lp;
      }
      let sum = 0;
      for (let j = 0; j < V; j++) sum += Math.exp(blend[j] - maxV);
      const logNorm = maxV + Math.log(sum);
      hLogProb += blend[target] - logNorm;
      const hTop = topK(blend, 5);
      if (hTop[0] === target) hTop1++;
      if (hTop.includes(target)) hTop5++;

      n++;
      if (n % 25 === 0) progress(`neural eval ${n}/${maxTokens}`);
      if (n >= maxTokens) break outer;
    }
  }
  progressDone();
  const mk = (lp: number, t1: number, t5: number): TokenMetrics => ({
    tokens: n,
    perplexity: n > 0 ? Math.exp(-lp / n) : Number.NaN,
    top1: n > 0 ? t1 / n : 0,
    top5: n > 0 ? t5 / n : 0,
  });
  return { neural: mk(nLogProb, nTop1, nTop5), hybrid: mk(hLogProb, hTop1, hTop5) };
}

/**
 * End-to-end line completion on held-out source text: place the cursor at a real
 * mid-line position, ask for a completion, compare to the truth.
 */
export function evalLineCompletion(
  bundle: Bundle,
  texts: Array<{ path: string; text: string }>,
  opts: {
    samples?: number;
    neural?: boolean;
    seed?: number;
    maxTokens?: number;
    minEvidenceOrder?: number;
    beamWidth?: number;
  } = {},
): LineMetrics {
  const samples = opts.samples ?? 200;
  const engine = new CompletionEngine(bundle);
  const rng = new Rng(opts.seed ?? 4242);
  const latencies: number[] = [];
  let prefixMatch = 0;
  let exactLine = 0;
  let correctChars = 0;
  let offeredChars = 0;
  let taken = 0;

  // Collect candidate cursor positions: inside a line with enough before/after.
  const positions: Array<{ file: number; offset: number; lineEnd: number }> = [];
  for (let f = 0; f < texts.length; f++) {
    const text = texts[f].text;
    const lines = text.split("\n");
    let offset = 0;
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.length >= 12 && !trimmed.startsWith("//") && !trimmed.startsWith("*")) {
        const indent = line.length - line.trimStart().length;
        // Cursor somewhere in the middle of the line's content.
        const lo = indent + Math.max(3, Math.floor(trimmed.length * 0.25));
        const hi = indent + Math.floor(trimmed.length * 0.8);
        if (hi > lo) positions.push({ file: f, offset: offset + lo, lineEnd: offset + line.length });
      }
      offset += line.length + 1;
    }
  }
  if (positions.length === 0) {
    return {
      samples: 0,
      prefixMatch: 0,
      exactLine: 0,
      meanCorrectChars: 0,
      meanOfferedChars: 0,
      meanLatencyMs: 0,
      p95LatencyMs: 0,
    };
  }
  rng.shuffle(positions);

  for (let i = 0; i < Math.min(samples, positions.length); i++) {
    const { file, offset, lineEnd } = positions[i];
    const text = texts[file].text;
    const prefix = text.slice(0, offset);
    const suffix = text.slice(offset);
    const truth = text.slice(offset, lineEnd);
    if (truth.length === 0) continue;

    const t0 = performance.now();
    const result = engine.complete({
      prefix,
      suffix,
      filePath: texts[file].path,
      neural: opts.neural === true,
      retrieval: false,
      count: 1,
      maxTokens: opts.maxTokens ?? 8,
      beamWidth: opts.beamWidth ?? 5,
      ...(opts.minEvidenceOrder !== undefined
        ? { minEvidenceOrder: opts.minEvidenceOrder }
        : {}),
    });
    latencies.push(performance.now() - t0);
    taken++;
    if ((i & 15) === 0) progress(`line eval ${i + 1}/${Math.min(samples, positions.length)}`);

    const best = result.suggestions[0];
    if (!best) continue;
    offeredChars += best.text.length;
    let correct = 0;
    while (correct < best.text.length && correct < truth.length && best.text[correct] === truth[correct]) {
      correct++;
    }
    correctChars += correct;
    if (best.text.length > 0 && correct === best.text.length) prefixMatch++;
    if (best.text === truth) exactLine++;
  }
  progressDone();

  latencies.sort((a, b) => a - b);
  return {
    samples: taken,
    prefixMatch: taken > 0 ? prefixMatch / taken : 0,
    exactLine: taken > 0 ? exactLine / taken : 0,
    meanCorrectChars: taken > 0 ? correctChars / taken : 0,
    meanOfferedChars: taken > 0 ? offeredChars / taken : 0,
    meanLatencyMs: latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0,
    p95LatencyMs: latencies.length > 0 ? latencies[Math.floor(latencies.length * 0.95)] ?? 0 : 0,
  };
}

function topK(values: Float32Array | number[], k: number): number[] {
  // Partial selection - a full sort over the vocabulary would dominate the loop.
  const best: number[] = [];
  const bestVals: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (best.length < k) {
      best.push(i);
      bestVals.push(v);
      for (let j = best.length - 1; j > 0 && bestVals[j] > bestVals[j - 1]; j--) {
        [bestVals[j], bestVals[j - 1]] = [bestVals[j - 1], bestVals[j]];
        [best[j], best[j - 1]] = [best[j - 1], best[j]];
      }
    } else if (v > bestVals[k - 1]) {
      bestVals[k - 1] = v;
      best[k - 1] = i;
      for (let j = k - 1; j > 0 && bestVals[j] > bestVals[j - 1]; j--) {
        [bestVals[j], bestVals[j - 1]] = [bestVals[j - 1], bestVals[j]];
        [best[j], best[j - 1]] = [best[j - 1], best[j]];
      }
    }
  }
  return best;
}
