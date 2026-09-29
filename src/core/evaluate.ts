import { Bundle } from "./bundle.js";
import { CompletionEngine } from "./complete.js";
import { Lesson, LessonStore } from "../chat/memory.js";
import { LocalAssistant, buildTurnContext } from "../chat/local.js";
import { NgramModel } from "../ml/ngram.js";
import { Rng } from "../util/rng.js";
import { progress, progressDone, step } from "../util/log.js";

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

export interface RoutingMetrics {
  /** Questions evaluated: one per answer in memory. */
  questions: number;
  /** Fraction the router would keep on the local models. */
  local: number;
  /** Mean router probability that a question needs Claude. */
  meanPClaude: number;
  /** What the local half would actually have said, by answer kind. */
  byKind: Record<string, number>;
  /**
   * Fraction answerable from *other* stored answers, with this one held out.
   * The honest measure of whether memory generalises rather than memorises.
   */
  coveredByOthers: number;
}

/**
 * Replay every remembered question through the router and the local answerer,
 * leaving that question's own answer out of memory.
 *
 * This is the metric for the part of the system that is supposed to compound.
 * Scoring a question against a store that contains its own answer would report
 * 100% and mean nothing; holding it out asks the question that matters - does
 * what this assistant already knows cover the next thing you will ask?
 *
 * Note what it is not: a quality measure. It says the local half would answer,
 * not that the answer would be as good as Claude's. Those are different claims
 * and only the first one is measurable without a human reading the output.
 */
export function evalRouting(bundle: Bundle, lessons: Lesson[]): RoutingMetrics {
  const byKind: Record<string, number> = {};
  if (lessons.length === 0) {
    return { questions: 0, local: 0, meanPClaude: 0, byKind, coveredByOthers: 0 };
  }
  const local = new LocalAssistant(bundle);
  const router = bundle.router;
  let localCount = 0;
  let pSum = 0;
  let covered = 0;

  for (let i = 0; i < lessons.length; i++) {
    progress(`routing ${i + 1}/${lessons.length}`);
    const heldOut = new LessonStore(lessons.filter((_, j) => j !== i));
    const ctx = buildTurnContext(bundle, heldOut, lessons[i].question);
    const p = router.probability(ctx.features);
    pSum += p;
    if (p < 0.5) localCount++;
    const answer = local.answer(ctx);
    byKind[answer.kind] = (byKind[answer.kind] ?? 0) + 1;
    if (answer.kind === "memory") covered++;
  }
  progressDone();

  return {
    questions: lessons.length,
    local: localCount / lessons.length,
    meanPClaude: pSum / lessons.length,
    byKind,
    coveredByOthers: covered / lessons.length,
  };
}

/** Next-token metrics for the n-gram model over held-out documents. */
export function evalNgram(bundle: Bundle, docs: number[][], maxTokens = 20_000): TokenMetrics {
  return scoreNgram(bundle.ngram, docs, maxTokens);
}

/** The same, for a model that is not the one on disk - used by the learning curve. */
export function scoreNgram(ngram: NgramModel, docs: number[][], maxTokens = 20_000): TokenMetrics {
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

export interface CurvePoint {
  /** Share of the training corpus used. */
  fraction: number;
  documents: number;
  tokens: number;
  perplexity: number;
  top1: number;
  top5: number;
}

/**
 * Retrain the n-gram on growing slices of the corpus and score each one on the
 * same held-out files.
 *
 * This is the only honest way to answer "how much more data do I need". The
 * alternative is to guess, and guesses about this are usually wrong by an order
 * of magnitude in whichever direction is more flattering. The curve says
 * plainly whether the model is still data-starved (accuracy climbing steeply
 * between slices) or saturated (the last doubling bought almost nothing), and
 * therefore whether adding another repository is worth the disk.
 *
 * Only the n-gram is swept. It trains by counting, so five slices cost about
 * what one `pca train` costs, and it is the model that actually serves inline
 * completion. Sweeping the transformer would take hours and answer a question
 * nobody asked.
 */
export function evalLearningCurve(
  bundle: Bundle,
  train: number[][],
  val: number[][],
  opts: { fractions?: number[]; order?: number; maxTokens?: number; seed?: number } = {},
): CurvePoint[] {
  const fractions = opts.fractions ?? [0.125, 0.25, 0.5, 1];
  const order = opts.order ?? bundle.manifest?.ngramOrder ?? 4;
  const vocabSize = bundle.tokenizer.vocabSize;

  // One fixed shuffle, sliced as prefixes, so every larger slice is a superset
  // of every smaller one. Independently sampled slices would let a lucky draw
  // at 25% beat an unlucky one at 50% and make the curve read as noise.
  const shuffled = train.slice();
  new Rng(opts.seed ?? 20250928).shuffle(shuffled);

  const points: CurvePoint[] = [];
  for (const fraction of fractions) {
    const take = Math.max(1, Math.round(shuffled.length * fraction));
    const slice = shuffled.slice(0, take);
    step(`Learning curve: ${(fraction * 100).toFixed(0)}% of corpus (${take} documents)`);
    const ngram = new NgramModel({ order });
    ngram.vocabSize = vocabSize;
    let tokens = 0;
    for (let i = 0; i < slice.length; i++) {
      if ((i & 31) === 0) progress(`counting ${i + 1}/${slice.length}`);
      ngram.observe(slice[i]);
      tokens += slice[i].length;
    }
    ngram.finalize();
    progressDone();
    const m = scoreNgram(ngram, val, opts.maxTokens ?? 8000);
    points.push({
      fraction,
      documents: take,
      tokens,
      perplexity: m.perplexity,
      top1: m.top1,
      top5: m.top5,
    });
  }
  return points;
}

export interface ScalingFit {
  /** Fitted exponent in error = A · tokens^(-alpha). Bigger means data helps more. */
  alpha: number;
  A: number;
  /** Goodness of fit in log-log space, 0..1. Below ~0.8 the projection is noise. */
  r2: number;
  /** Corpus size needed for a given top-1 accuracy, or null if unreachable. */
  tokensFor: (top1: number) => number | null;
}

/**
 * Fit a power law to the curve: top-1 error falls as A · N^(-alpha) in corpus
 * size N. This is the standard empirical form for language-model scaling, and
 * fitting it lets the curve be read forwards - "another 5x of corpus buys about
 * this much" - instead of only backwards.
 *
 * It is an extrapolation from four points on one machine, so it is a planning
 * number, not a promise, and `r2` is returned so callers can refuse to show it
 * when the points do not actually lie on a line.
 */
export function fitScalingLaw(points: CurvePoint[]): ScalingFit | null {
  const usable = points.filter((p) => p.tokens > 0 && p.top1 > 0 && p.top1 < 1);
  if (usable.length < 2) return null;

  const xs = usable.map((p) => Math.log(p.tokens));
  const ys = usable.map((p) => Math.log(1 - p.top1));
  const n = xs.length;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - meanX) * (ys[i] - meanY);
    sxx += (xs[i] - meanX) ** 2;
  }
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  const intercept = meanY - slope * meanX;

  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (ys[i] - (intercept + slope * xs[i])) ** 2;
    ssTot += (ys[i] - meanY) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;

  const alpha = -slope;
  const A = Math.exp(intercept);
  return {
    alpha,
    A,
    r2,
    tokensFor(top1: number): number | null {
      const err = 1 - top1;
      // A non-positive exponent means more data was not helping over the
      // measured range; projecting from it would invent a number.
      if (err <= 0 || alpha <= 0) return null;
      const tokens = Math.exp((Math.log(A) - Math.log(err)) / alpha);
      return Number.isFinite(tokens) ? tokens : null;
    },
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
