import fs from "node:fs";
import path from "node:path";
import { Rng } from "../util/rng.js";

/**
 * Skip-gram with negative sampling (word2vec SGNS) over code tokens.
 *
 * Trained on the user's own corpus, the vectors learn project-specific
 * relationships that no generic embedding model knows: that `userId` sits near
 * `accountId`, that a helper name co-occurs with the module it lives in. Those
 * vectors are then pooled (see `sifEmbed`) into chunk embeddings for semantic
 * search.
 *
 * Trains in seconds because each update touches only (1 + negatives) rows.
 */
export interface EmbeddingOptions {
  dim?: number;
  window?: number;
  negatives?: number;
  epochs?: number;
  minLr?: number;
  lr?: number;
  /** Subsample tokens more frequent than this rate (word2vec's `t`). */
  sampleRate?: number;
  seed?: number;
}

const DEFAULTS: Required<EmbeddingOptions> = {
  dim: 64,
  window: 5,
  negatives: 5,
  epochs: 4,
  lr: 0.025,
  minLr: 0.0001,
  sampleRate: 1e-3,
  seed: 7,
};

export class EmbeddingModel {
  readonly dim: number;
  readonly vocabSize: number;
  /** Input ("center") vectors - these are the ones used downstream. */
  readonly input: Float32Array;
  /** Output ("context") vectors, kept only for training. */
  private output: Float32Array;
  /** Token frequency, used for subsampling and SIF weighting. */
  readonly freq: Float64Array;
  private totalCount = 0;

  constructor(vocabSize: number, dim = DEFAULTS.dim, seed = DEFAULTS.seed) {
    this.vocabSize = vocabSize;
    this.dim = dim;
    this.input = new Float32Array(vocabSize * dim);
    this.output = new Float32Array(vocabSize * dim);
    this.freq = new Float64Array(vocabSize);
    const rng = new Rng(seed);
    // Standard word2vec init: uniform in (-0.5/dim, 0.5/dim) for input, zeros
    // for output.
    for (let i = 0; i < this.input.length; i++) this.input[i] = (rng.next() - 0.5) / dim;
  }

  countTokens(docs: Iterable<ArrayLike<number>>): void {
    for (const doc of docs) {
      for (let i = 0; i < doc.length; i++) {
        const id = doc[i];
        if (id < this.vocabSize) {
          this.freq[id]++;
          this.totalCount++;
        }
      }
    }
  }

  /** Unigram distribution raised to 3/4, the classic negative-sampling table. */
  private buildNegativeTable(size = 1 << 21): Int32Array {
    const table = new Int32Array(size);
    let norm = 0;
    const pow = new Float64Array(this.vocabSize);
    for (let i = 0; i < this.vocabSize; i++) {
      pow[i] = Math.pow(this.freq[i], 0.75);
      norm += pow[i];
    }
    if (norm === 0) return table;
    let i = 0;
    let acc = pow[0] / norm;
    for (let a = 0; a < size; a++) {
      table[a] = i;
      if (a / size > acc && i < this.vocabSize - 1) {
        i++;
        acc += pow[i] / norm;
      }
    }
    return table;
  }

  /**
   * Train on token streams. `countTokens` must have been called first so the
   * subsampling and negative-sampling distributions are available.
   */
  train(
    docs: ArrayLike<number>[],
    opts: EmbeddingOptions = {},
    onProgress?: (done: number, total: number) => void,
  ): void {
    const o = { ...DEFAULTS, ...opts };
    const rng = new Rng(o.seed);
    const table = this.buildNegativeTable();
    if (table.length === 0 || this.totalCount === 0) return;
    const dim = this.dim;
    const grad = new Float32Array(dim);

    // Precompute subsampling keep-probabilities (word2vec formula).
    const keepProb = new Float64Array(this.vocabSize);
    for (let i = 0; i < this.vocabSize; i++) {
      const f = this.freq[i] / this.totalCount;
      keepProb[i] = f <= 0 ? 0 : Math.min(1, (Math.sqrt(f / o.sampleRate) + 1) * (o.sampleRate / f));
    }

    const totalWork = o.epochs * this.totalCount;
    let done = 0;
    for (let epoch = 0; epoch < o.epochs; epoch++) {
      for (const doc of docs) {
        // Subsample once per doc per epoch into a compact array.
        const kept: number[] = [];
        for (let i = 0; i < doc.length; i++) {
          const id = doc[i];
          if (id >= this.vocabSize) continue;
          if (keepProb[id] >= 1 || rng.next() < keepProb[id]) kept.push(id);
        }
        for (let i = 0; i < kept.length; i++) {
          const lr = Math.max(o.minLr, o.lr * (1 - done / totalWork));
          const center = kept[i];
          // Shrinking window, as in the reference implementation.
          const b = 1 + rng.int(o.window);
          for (let j = Math.max(0, i - b); j <= Math.min(kept.length - 1, i + b); j++) {
            if (j === i) continue;
            const ctx = kept[j];
            grad.fill(0);
            const inOff = center * dim;
            // One positive sample plus `negatives` sampled distractors.
            for (let n = 0; n <= o.negatives; n++) {
              let target: number;
              let label: number;
              if (n === 0) {
                target = ctx;
                label = 1;
              } else {
                target = table[rng.int(table.length)];
                if (target === ctx) continue;
                label = 0;
              }
              const outOff = target * dim;
              let dotv = 0;
              for (let d = 0; d < dim; d++) dotv += this.input[inOff + d] * this.output[outOff + d];
              const pred = sigmoid(dotv);
              const g = (label - pred) * lr;
              for (let d = 0; d < dim; d++) {
                grad[d] += g * this.output[outOff + d];
                this.output[outOff + d] += g * this.input[inOff + d];
              }
            }
            for (let d = 0; d < dim; d++) this.input[inOff + d] += grad[d];
          }
          done++;
          if (onProgress && (done & 0xffff) === 0) onProgress(done, totalWork);
        }
      }
    }
    if (onProgress) onProgress(totalWork, totalWork);
  }

  vector(id: number, out?: Float32Array): Float32Array {
    const dst = out ?? new Float32Array(this.dim);
    if (id < 0 || id >= this.vocabSize) {
      dst.fill(0);
      return dst;
    }
    dst.set(this.input.subarray(id * this.dim, (id + 1) * this.dim));
    return dst;
  }

  /**
   * Smooth Inverse Frequency pooling: weight each token by a / (a + p(token)).
   * Rare, information-dense tokens (your function names) dominate; `{`, `,` and
   * `return` contribute almost nothing. Cheap, and a strong baseline.
   */
  sifEmbed(tokens: ArrayLike<number>, a = 1e-3): Float32Array {
    const out = new Float32Array(this.dim);
    if (this.totalCount === 0) return out;
    let weightSum = 0;
    for (let i = 0; i < tokens.length; i++) {
      const id = tokens[i];
      if (id < 0 || id >= this.vocabSize) continue;
      const p = this.freq[id] / this.totalCount;
      const w = a / (a + p);
      weightSum += w;
      const off = id * this.dim;
      for (let d = 0; d < this.dim; d++) out[d] += w * this.input[off + d];
    }
    if (weightSum > 0) for (let d = 0; d < this.dim; d++) out[d] /= weightSum;
    normalize(out);
    return out;
  }

  /** Nearest neighbours by cosine similarity - powers `pca neighbors`. */
  nearest(id: number, k = 10): Array<{ id: number; score: number }> {
    const q = this.vector(id);
    normalize(q);
    const scored: Array<{ id: number; score: number }> = [];
    const probe = new Float32Array(this.dim);
    for (let other = 0; other < this.vocabSize; other++) {
      if (other === id || this.freq[other] === 0) continue;
      probe.set(this.input.subarray(other * this.dim, (other + 1) * this.dim));
      normalize(probe);
      let s = 0;
      for (let d = 0; d < this.dim; d++) s += q[d] * probe[d];
      scored.push({ id: other, score: s });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  save(file: string): void {
    const header = Buffer.from(
      JSON.stringify({
        version: 1,
        kind: "pca-embeddings",
        dim: this.dim,
        vocabSize: this.vocabSize,
        totalCount: this.totalCount,
      }),
      "utf8",
    );
    const lenBuf = Buffer.alloc(4);
    lenBuf.writeUInt32LE(header.length, 0);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const tmp = `${file}.tmp${process.pid}`;
    fs.writeFileSync(
      tmp,
      Buffer.concat([
        lenBuf,
        header,
        Buffer.from(this.input.buffer, this.input.byteOffset, this.input.byteLength),
        Buffer.from(this.freq.buffer, this.freq.byteOffset, this.freq.byteLength),
      ]),
    );
    fs.renameSync(tmp, file);
  }

  static load(file: string): EmbeddingModel {
    const buf = fs.readFileSync(file);
    const headerLen = buf.readUInt32LE(0);
    const meta = JSON.parse(buf.subarray(4, 4 + headerLen).toString("utf8")) as {
      dim: number;
      vocabSize: number;
      totalCount: number;
    };
    const model = new EmbeddingModel(meta.vocabSize, meta.dim);
    model.totalCount = meta.totalCount;
    let off = 4 + headerLen;
    const inputBytes = meta.vocabSize * meta.dim * 4;
    Buffer.from(model.input.buffer, model.input.byteOffset, model.input.byteLength).set(
      buf.subarray(off, off + inputBytes),
    );
    off += inputBytes;
    Buffer.from(model.freq.buffer, model.freq.byteOffset, model.freq.byteLength).set(
      buf.subarray(off, off + meta.vocabSize * 8),
    );
    // Output vectors are training-only; a loaded model is inference-only.
    model.output = new Float32Array(0);
    return model;
  }
}

function sigmoid(x: number): number {
  if (x > 30) return 1;
  if (x < -30) return 0;
  return 1 / (1 + Math.exp(-x));
}

export function normalize(v: Float32Array): Float32Array {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n);
  if (n > 0) for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}
