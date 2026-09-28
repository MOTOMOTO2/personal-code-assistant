import { readJsonGz, writeJsonGz } from "../util/io.js";

/**
 * Named features describing one candidate completion. The order is the weight
 * vector layout, so it is append-only: adding a feature in the middle would
 * silently reinterpret an existing trained model (guarded by `FEATURE_VERSION`).
 */
export const FEATURES = [
  "bias",
  "ngramLogProb", // statistical LM score, scaled
  "neuralLogProb", // transformer score, scaled (0 when the model is absent)
  "modelsAgree", // both models rank it top-3
  "ngramRank", // 1/(1+rank) from the n-gram proposal order
  "charLength", // longer suggestions are more valuable but riskier
  "isIdentifier",
  "isKeyword",
  "localFileHits", // appears nearby in the file being edited
  "retrievalHits", // appears in semantically retrieved context
  "embedSimilarity", // cosine(candidate vector, local context vector)
  "continuesPartial", // completes the identifier fragment being typed
  "openBracketBalance", // suggestion closes something that is currently open
] as const;

export type FeatureName = (typeof FEATURES)[number];
export const NUM_FEATURES = FEATURES.length;
export const FEATURE_VERSION = 1;

export function emptyFeatures(): Float64Array {
  const f = new Float64Array(NUM_FEATURES);
  f[0] = 1; // bias
  return f;
}

export function featureIndex(name: FeatureName): number {
  return FEATURES.indexOf(name);
}

export interface RerankerData {
  version: number;
  featureVersion: number;
  weights: number[];
  updates: number;
  accepted: number;
  rejected: number;
}

/**
 * Online learning-to-rank over completion candidates.
 *
 * Every accept or dismiss in the editor is a training signal. Accepts are
 * turned into *pairwise* comparisons against the candidates that were shown but
 * not taken, which is the right objective for ranking and learns from far fewer
 * events than pointwise classification would.
 *
 * Deliberately a linear model: it stays interpretable (`pca stats --weights`
 * prints what it learned about you), updates in microseconds, and cannot
 * overfit hard enough to wreck the base models' ordering.
 */
export class Reranker {
  weights: Float64Array;
  updates = 0;
  accepted = 0;
  rejected = 0;
  /** L2 penalty applied on each update. */
  l2 = 1e-5;

  constructor(weights?: Float64Array) {
    if (weights && weights.length === NUM_FEATURES) {
      this.weights = weights;
    } else {
      this.weights = new Float64Array(NUM_FEATURES);
      // Start from a sensible prior rather than zeros, so an untrained
      // assistant already ranks well: trust the language models, mildly prefer
      // longer and identifier-shaped completions.
      this.weights[featureIndex("ngramLogProb")] = 1.0;
      this.weights[featureIndex("neuralLogProb")] = 0.6;
      this.weights[featureIndex("modelsAgree")] = 0.3;
      this.weights[featureIndex("ngramRank")] = 0.4;
      this.weights[featureIndex("charLength")] = 0.05;
      this.weights[featureIndex("isIdentifier")] = 0.1;
      this.weights[featureIndex("localFileHits")] = 0.3;
      this.weights[featureIndex("retrievalHits")] = 0.25;
      this.weights[featureIndex("embedSimilarity")] = 0.35;
      this.weights[featureIndex("continuesPartial")] = 0.8;
      this.weights[featureIndex("openBracketBalance")] = 0.2;
    }
  }

  score(features: ArrayLike<number>): number {
    let s = 0;
    for (let i = 0; i < NUM_FEATURES; i++) s += this.weights[i] * features[i];
    return s;
  }

  /**
   * RankNet-style pairwise update: push `positive` above each `negative`.
   * Gradient of log(1 + exp(-(s+ - s-))) is -sigmoid(-(s+ - s-)) * (f+ - f-).
   */
  learnPairwise(
    positive: ArrayLike<number>,
    negatives: ArrayLike<number>[],
    lr = 0.05,
  ): void {
    if (negatives.length === 0) return;
    const sPos = this.score(positive);
    for (const negative of negatives) {
      const sNeg = this.score(negative);
      const margin = sPos - sNeg;
      const g = sigmoid(-margin); // 1 when the ranking is wrong, ~0 when right
      for (let i = 0; i < NUM_FEATURES; i++) {
        const diff = positive[i] - negative[i];
        this.weights[i] += lr * g * diff - lr * this.l2 * this.weights[i];
      }
      this.updates++;
    }
    this.accepted++;
  }

  /** Pointwise logistic update, used when there is no comparison set. */
  learnPointwise(features: ArrayLike<number>, label: 0 | 1, lr = 0.05): void {
    const p = sigmoid(this.score(features));
    const err = label - p;
    for (let i = 0; i < NUM_FEATURES; i++) {
      this.weights[i] += lr * err * features[i] - lr * this.l2 * this.weights[i];
    }
    this.updates++;
    if (label === 1) this.accepted++;
    else this.rejected++;
  }

  /** Human-readable weights, largest magnitude first. */
  describe(): Array<{ feature: string; weight: number }> {
    return FEATURES.map((feature, i) => ({ feature, weight: this.weights[i] })).sort(
      (a, b) => Math.abs(b.weight) - Math.abs(a.weight),
    );
  }

  toJSON(): RerankerData {
    return {
      version: 1,
      featureVersion: FEATURE_VERSION,
      weights: [...this.weights],
      updates: this.updates,
      accepted: this.accepted,
      rejected: this.rejected,
    };
  }

  save(file: string): void {
    writeJsonGz(file, this.toJSON());
  }

  static load(file: string): Reranker {
    const data = readJsonGz<RerankerData>(file);
    if (data.featureVersion !== FEATURE_VERSION || data.weights.length !== NUM_FEATURES) {
      // The feature layout changed under an old model: start fresh rather than
      // reinterpret weights against the wrong features.
      return new Reranker();
    }
    const r = new Reranker(Float64Array.from(data.weights));
    r.updates = data.updates;
    r.accepted = data.accepted;
    r.rejected = data.rejected;
    return r;
  }
}

function sigmoid(x: number): number {
  if (x > 30) return 1;
  if (x < -30) return 0;
  return 1 / (1 + Math.exp(-x));
}
