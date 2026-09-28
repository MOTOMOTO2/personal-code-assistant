import { readJsonGz, writeJsonGz } from "../util/io.js";

/**
 * Interpolated Kneser-Ney n-gram language model over code tokens.
 *
 * This is the low-latency half of the assistant: it trains in seconds, scores a
 * candidate in microseconds, and is genuinely strong on code because code is far
 * more repetitive than prose. The neural model (see `transformer.ts`) adds
 * longer-range structure on top of it through a learned mixture.
 *
 * How the counts are laid out, because the code only makes sense with it:
 *   - The highest order holds raw counts c(ctx, w).
 *   - Every lower order holds continuation counts: how many distinct tokens
 *     precede that (ctx, w) pair. This is the Kneser-Ney insight - it stops a
 *     token from looking common just because one frequent phrase contains it.
 *   - Raw and continuation counts share the same support, so each order is
 *     derived from the order above it by dropping the oldest context token.
 */

/** A context key is the preceding token ids joined with ","; "" for unigrams. */
type CtxKey = string;

interface OrderStore {
  /** ctx -> (token id -> count). Raw counts at the top order, continuation below. */
  map: Map<CtxKey, Map<number, number>>;
  /** ctx -> sum of counts. */
  total: Map<CtxKey, number>;
  /** Absolute discount D for this order, estimated from the count-of-counts. */
  discount: number;
}

export interface NgramOptions {
  /** Maximum n-gram order. 4 suits most codebases; 5 helps on large corpora. */
  order?: number;
}

export interface NgramCandidate {
  id: number;
  logProb: number;
}

export interface NgramModelData {
  version: 1;
  order: number;
  vocabSize: number;
  tokensSeen: number;
  orders: Array<{
    discount: number;
    ctxKeys: string[];
    /** offsets[i]..offsets[i+1] slices `flat` for context i. */
    offsets: number[];
    /** Flat (id, count) pairs, sorted by descending count within a context. */
    flat: number[];
  }>;
}

export class NgramModel {
  readonly order: number;
  vocabSize = 0;
  tokensSeen = 0;
  /** orders[k] is the store for order k+1 (index 0 is the unigram). */
  private orders: OrderStore[];
  /** Descending-by-count successor lists, built by finalize() for prediction. */
  private sorted: Array<Map<CtxKey, { ids: Int32Array; counts: Float64Array }>>;

  constructor(opts: NgramOptions = {}) {
    this.order = Math.max(1, Math.min(8, opts.order ?? 4));
    this.orders = [];
    this.sorted = [];
    for (let k = 0; k < this.order; k++) {
      this.orders.push({ map: new Map(), total: new Map(), discount: 0.75 });
      this.sorted.push(new Map());
    }
  }

  /**
   * Count top-order n-grams in a token stream. Call once per document, then
   * `finalize()` once when the corpus is exhausted.
   */
  observe(tokens: ArrayLike<number>): void {
    const N = this.order;
    const top = this.orders[N - 1].map;
    const topTotal = this.orders[N - 1].total;
    this.tokensSeen += tokens.length;
    for (let i = 0; i < tokens.length; i++) {
      const w = tokens[i];
      if (w + 1 > this.vocabSize) this.vocabSize = w + 1;
      // Context is the min(N-1, i) preceding tokens. Near the start of a file
      // the context is genuinely short - that is an observation, not padding.
      const start = Math.max(0, i - (N - 1));
      let key = "";
      for (let j = start; j < i; j++) key += tokens[j] + ",";
      let succ = top.get(key);
      if (succ === undefined) {
        succ = new Map();
        top.set(key, succ);
      }
      succ.set(w, (succ.get(w) ?? 0) + 1);
      topTotal.set(key, (topTotal.get(key) ?? 0) + 1);
    }
  }

  /**
   * Derive continuation counts for every lower order, estimate the per-order
   * discounts, and build the sorted successor lists used at prediction time.
   */
  finalize(): void {
    for (let k = this.order - 1; k >= 1; k--) {
      const parent = this.orders[k].map;
      const child = this.orders[k - 1];
      for (const [ctx, succ] of parent) {
        // Drop the oldest context token: "12,45,9," becomes "45,9,".
        const cut = ctx.indexOf(",");
        const suffix = cut < 0 ? "" : ctx.slice(cut + 1);
        let bucket = child.map.get(suffix);
        if (bucket === undefined) {
          bucket = new Map();
          child.map.set(suffix, bucket);
        }
        let added = 0;
        for (const w of succ.keys()) {
          // Each distinct parent context is exactly one distinct predecessor
          // for (suffix, w) - that is precisely the continuation count.
          bucket.set(w, (bucket.get(w) ?? 0) + 1);
          added++;
        }
        child.total.set(suffix, (child.total.get(suffix) ?? 0) + added);
      }
    }
    for (let k = 0; k < this.order; k++) {
      this.orders[k].discount = estimateDiscount(this.orders[k].map);
      this.sorted[k] = buildSorted(this.orders[k].map);
    }
  }

  private ctxKeyFor(history: ArrayLike<number>, k: number): CtxKey {
    // The order-k distribution is conditioned on the k-1 most recent tokens.
    const want = k - 1;
    const start = Math.max(0, history.length - want);
    let key = "";
    for (let j = start; j < history.length; j++) key += history[j] + ",";
    return key;
  }

  /** log P(next | history) under the interpolated Kneser-Ney mixture. */
  logProb(history: ArrayLike<number>, next: number): number {
    return Math.log(Math.max(this.prob(history, next), 1e-12));
  }

  prob(history: ArrayLike<number>, next: number): number {
    const uniform = 1 / Math.max(this.vocabSize, 2);
    let p = uniform;
    // Walk upward from the unigram so each order interpolates the one below it.
    for (let k = 1; k <= this.order; k++) {
      const store = this.orders[k - 1];
      const ctx = this.ctxKeyFor(history, k);
      const denom = store.total.get(ctx);
      if (denom === undefined || denom <= 0) continue; // unseen context: pass through
      const succ = store.map.get(ctx)!;
      const D = store.discount;
      const c = succ.get(next) ?? 0;
      const lambda = (D * succ.size) / denom;
      p = Math.max(c - D, 0) / denom + lambda * p;
    }
    return p;
  }

  /**
   * Top-k next tokens. Candidates come from the longest matching context
   * downwards (high-order successors are the good ones, and there are few of
   * them), then every candidate is rescored with the full interpolated mixture.
   */
  predict(history: ArrayLike<number>, topK = 12, maxCandidates = 256): NgramCandidate[] {
    const seen = new Set<number>();
    const candidates: number[] = [];
    for (let k = this.order; k >= 1 && candidates.length < maxCandidates; k--) {
      const ctx = this.ctxKeyFor(history, k);
      const entry = this.sorted[k - 1].get(ctx);
      if (!entry) continue;
      for (let i = 0; i < entry.ids.length && candidates.length < maxCandidates; i++) {
        const id = entry.ids[i];
        if (seen.has(id)) continue;
        seen.add(id);
        candidates.push(id);
      }
    }
    const scored: NgramCandidate[] = [];
    for (const id of candidates) {
      scored.push({ id, logProb: this.logProb(history, id) });
    }
    scored.sort((a, b) => b.logProb - a.logProb);
    return scored.slice(0, topK);
  }

  stats(): { order: number; contexts: number; ngrams: number; tokensSeen: number } {
    let contexts = 0;
    let ngrams = 0;
    for (const store of this.orders) {
      contexts += store.map.size;
      for (const succ of store.map.values()) ngrams += succ.size;
    }
    return { order: this.order, contexts, ngrams, tokensSeen: this.tokensSeen };
  }

  toJSON(): NgramModelData {
    const orders = this.orders.map((store) => {
      const ctxKeys: string[] = [];
      const offsets: number[] = [0];
      const flat: number[] = [];
      for (const [ctx, succ] of store.map) {
        ctxKeys.push(ctx);
        const entries = [...succ.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
        for (const [id, count] of entries) flat.push(id, count);
        offsets.push(flat.length);
      }
      return { discount: store.discount, ctxKeys, offsets, flat };
    });
    return {
      version: 1,
      order: this.order,
      vocabSize: this.vocabSize,
      tokensSeen: this.tokensSeen,
      orders,
    };
  }

  save(file: string): void {
    writeJsonGz(file, this.toJSON());
  }

  static fromJSON(data: NgramModelData): NgramModel {
    const model = new NgramModel({ order: data.order });
    model.vocabSize = data.vocabSize;
    model.tokensSeen = data.tokensSeen;
    for (let k = 0; k < data.orders.length; k++) {
      const src = data.orders[k];
      const store = model.orders[k];
      store.discount = src.discount;
      for (let i = 0; i < src.ctxKeys.length; i++) {
        const lo = src.offsets[i];
        const hi = src.offsets[i + 1];
        const succ = new Map<number, number>();
        let total = 0;
        for (let p = lo; p < hi; p += 2) {
          succ.set(src.flat[p], src.flat[p + 1]);
          total += src.flat[p + 1];
        }
        store.map.set(src.ctxKeys[i], succ);
        store.total.set(src.ctxKeys[i], total);
      }
      model.sorted[k] = buildSorted(store.map);
    }
    return model;
  }

  static load(file: string): NgramModel {
    return NgramModel.fromJSON(readJsonGz<NgramModelData>(file));
  }
}

/**
 * Absolute discount estimated from the count-of-counts, D = n1 / (n1 + 2*n2)
 * (Ney et al.). Beats a hardcoded 0.75, especially on small personal corpora
 * where singletons dominate.
 */
function estimateDiscount(map: Map<CtxKey, Map<number, number>>): number {
  let n1 = 0;
  let n2 = 0;
  for (const succ of map.values()) {
    for (const c of succ.values()) {
      if (c === 1) n1++;
      else if (c === 2) n2++;
    }
  }
  if (n1 === 0) return 0.5;
  const d = n1 / (n1 + 2 * n2);
  // Keep D strictly inside (0, 1): D >= 1 can drive probabilities negative.
  return Math.min(0.95, Math.max(0.05, d));
}

function buildSorted(
  map: Map<CtxKey, Map<number, number>>,
): Map<CtxKey, { ids: Int32Array; counts: Float64Array }> {
  const out = new Map<CtxKey, { ids: Int32Array; counts: Float64Array }>();
  for (const [ctx, succ] of map) {
    const entries = [...succ.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    const ids = new Int32Array(entries.length);
    const counts = new Float64Array(entries.length);
    for (let i = 0; i < entries.length; i++) {
      ids[i] = entries[i][0];
      counts[i] = entries[i][1];
    }
    out.set(ctx, { ids, counts });
  }
  return out;
}
