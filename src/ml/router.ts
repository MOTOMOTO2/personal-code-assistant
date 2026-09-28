import { readJsonGz, writeJsonGz } from "../util/io.js";

/**
 * Feature layout for the route decision. Append-only, like the reranker's:
 * inserting in the middle would silently reinterpret a trained model, which
 * `ROUTER_FEATURE_VERSION` guards against.
 */
export const ROUTER_FEATURES = [
  "bias",
  "memoryTop", // best similarity to an answer already in memory
  "memorySecond", // is the recall a clear winner, or a tie between many?
  "retrievalTop", // best fused retrieval score, scaled
  "retrievalCoverage", // fraction of query terms the index has seen at all
  "locateShape", // "where is", "find", "which file" - a lookup, not reasoning
  "reasonShape", // "why", "how does", "should we" - needs a real model
  "editShape", // "write", "add", "fix", "refactor" - needs a real model
  "hasCodeBlock", // pasted code, almost always a reasoning request
  "lengthTokens", // long questions are rarely lookups
] as const;

export type RouterFeatureName = (typeof ROUTER_FEATURES)[number];
export const NUM_ROUTER_FEATURES = ROUTER_FEATURES.length;
export const ROUTER_FEATURE_VERSION = 1;

export function routerFeatureIndex(name: RouterFeatureName): number {
  return ROUTER_FEATURES.indexOf(name);
}

export interface RouterData {
  version: number;
  featureVersion: number;
  weights: number[];
  updates: number;
  toClaude: number;
  toLocal: number;
}

/**
 * Decides, per turn, whether the local models can answer or whether the
 * question needs Claude.
 *
 * A positive score means "escalate". The model is linear and online for the
 * same reasons the reranker is: it has to explain itself (`pca stats --router`
 * prints the weights), it updates in microseconds, and the training signal is
 * sparse - a handful of corrections per session, not a labelled corpus.
 *
 * The signal is free and honest: when the local half answers and you retry with
 * `/claude`, that is a labelled example saying "should have escalated". When
 * you accept a local answer and move on, that is the opposite label. Nothing
 * has to be annotated by hand.
 */
export class Router {
  weights: Float64Array;
  updates = 0;
  toClaude = 0;
  toLocal = 0;
  l2 = 1e-5;

  constructor(weights?: Float64Array) {
    if (weights && weights.length === NUM_ROUTER_FEATURES) {
      this.weights = weights;
      return;
    }
    this.weights = new Float64Array(NUM_ROUTER_FEATURES);
    const w = (name: RouterFeatureName, value: number) => {
      this.weights[routerFeatureIndex(name)] = value;
    };
    // Prior: escalate unless there is concrete evidence the local half can
    // handle it. Being wrong towards Claude costs a few cents; being wrong
    // towards local costs a useless answer, which is worse. Set so that a
    // question of no particular shape - neither a lookup nor obviously
    // reasoning - lands just above the threshold and goes to Claude. Ties
    // break towards the model that can actually think.
    w("bias", 1.1);
    // A hit in memory is the single strongest local signal, and it is weighted
    // to win outright: a frontier model already answered this question about
    // this codebase and the answer is on disk, so paying to generate a near
    // duplicate is waste. Large enough that even the weakest replayable recall
    // outvotes a "why does ..." shape, which is the case it has to get right.
    w("memoryTop", -7.0);
    // Several memories matching equally well is ambiguity, not evidence.
    w("memorySecond", 1.2);
    // The retrieval features get small weights on purpose. Both sit near 1.0
    // for almost any question about a repository you have indexed, so a large
    // weight would act as a second bias term and drown out question shape -
    // which is what `pca eval` caught them doing. Their information is at the
    // *low* end: a question full of terms this index has never seen is one the
    // local half cannot possibly answer.
    w("retrievalTop", -0.4);
    w("retrievalCoverage", -0.6);
    // Question shape carries the decision. Finding code is a lookup; explaining
    // it, or changing it, is not - and a good retrieval hit does not make the
    // local models able to reason about what they retrieved.
    w("locateShape", -2.5);
    w("reasonShape", 2.0);
    w("editShape", 2.5);
    w("hasCodeBlock", 1.5);
    w("lengthTokens", 0.6);
  }

  score(features: ArrayLike<number>): number {
    let s = 0;
    for (let i = 0; i < NUM_ROUTER_FEATURES; i++) s += this.weights[i] * features[i];
    return s;
  }

  /** Probability that this turn should go to Claude. */
  probability(features: ArrayLike<number>): number {
    return sigmoid(this.score(features));
  }

  decide(features: ArrayLike<number>, threshold = 0.5): "claude" | "local" {
    return this.probability(features) >= threshold ? "claude" : "local";
  }

  /** Online logistic update. `label` is 1 when the turn needed Claude. */
  learn(features: ArrayLike<number>, label: 0 | 1, lr = 0.12): void {
    const p = this.probability(features);
    const err = label - p;
    for (let i = 0; i < NUM_ROUTER_FEATURES; i++) {
      this.weights[i] += lr * err * features[i] - lr * this.l2 * this.weights[i];
    }
    this.updates++;
    if (label === 1) this.toClaude++;
    else this.toLocal++;
  }

  describe(): Array<{ feature: string; weight: number }> {
    return ROUTER_FEATURES.map((feature, i) => ({ feature, weight: this.weights[i] })).sort(
      (a, b) => Math.abs(b.weight) - Math.abs(a.weight),
    );
  }

  toJSON(): RouterData {
    return {
      version: 1,
      featureVersion: ROUTER_FEATURE_VERSION,
      weights: [...this.weights],
      updates: this.updates,
      toClaude: this.toClaude,
      toLocal: this.toLocal,
    };
  }

  save(file: string): void {
    writeJsonGz(file, this.toJSON());
  }

  static load(file: string): Router {
    const data = readJsonGz<RouterData>(file);
    if (
      data.featureVersion !== ROUTER_FEATURE_VERSION ||
      data.weights.length !== NUM_ROUTER_FEATURES
    ) {
      return new Router();
    }
    const r = new Router(Float64Array.from(data.weights));
    r.updates = data.updates;
    r.toClaude = data.toClaude ?? 0;
    r.toLocal = data.toLocal ?? 0;
    return r;
  }
}

function sigmoid(x: number): number {
  if (x > 30) return 1;
  if (x < -30) return 0;
  return 1 / (1 + Math.exp(-x));
}
