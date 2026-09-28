import fs from "node:fs";
import path from "node:path";
import {
  LayerNormCache,
  addBiasRows,
  biasGrad,
  gelu,
  geluGrad,
  layerNormBackward,
  layerNormForward,
  matmul,
  matmulABt,
  matmulAdd,
  matmulAddABt,
  matmulAddAtB,
  softmaxBackward,
  softmaxInPlace,
} from "./linalg.js";
import { ParamStore } from "./params.js";

/**
 * A decoder-only transformer with a hand-written backward pass.
 *
 * There is no autograd here on purpose: an explicit backward pass over a fixed
 * architecture is several times faster in plain JS than a tape, and it keeps the
 * dependency count at zero. The gradients are verified against central finite
 * differences in `test/transformer.test.ts` - that test is the contract.
 *
 * Architecture is the standard pre-LayerNorm GPT block:
 *   x = wte[tok] + wpe[pos]
 *   per layer:  x += attn(LN(x));  x += mlp(LN(x))
 *   logits = LN_f(x) @ wte^T        (tied input/output embedding)
 */
export interface TransformerConfig {
  vocabSize: number;
  dModel: number;
  nHeads: number;
  nLayers: number;
  dFF: number;
  /** Maximum context length in tokens. */
  blockSize: number;
}

export const DEFAULT_CONFIG: TransformerConfig = {
  vocabSize: 8192,
  dModel: 128,
  nHeads: 4,
  nLayers: 3,
  dFF: 512,
  blockSize: 96,
};

interface LayerBuffers {
  xIn: Float32Array; // [T, D] residual stream entering the block
  ln1: Float32Array; // [T, D]
  ln1Cache: LayerNormCache;
  q: Float32Array; // [T, D]
  k: Float32Array;
  v: Float32Array;
  att: Float32Array; // [H, T, T] post-softmax probabilities
  attOut: Float32Array; // [T, D] concatenated heads
  xMid: Float32Array; // [T, D] after the attention residual
  ln2: Float32Array; // [T, D]
  ln2Cache: LayerNormCache;
  h1: Float32Array; // [T, F] pre-activation
  h1a: Float32Array; // [T, F] post-GELU
}

export class Transformer {
  readonly config: TransformerConfig;
  readonly params: ParamStore;
  private readonly headDim: number;

  // Forward activation caches, allocated once for blockSize.
  private layers: LayerBuffers[] = [];
  private xFinal!: Float32Array; // [T, D] input to the final LayerNorm
  private lnfOut!: Float32Array; // [T, D]
  private lnfCache!: LayerNormCache;
  private logits!: Float32Array; // [T, V]
  private probs!: Float32Array; // [T, V]

  // Backward scratch.
  private dx!: Float32Array;
  private dxMid!: Float32Array;
  private dxIn!: Float32Array;
  private dTmp!: Float32Array; // [T, D] general purpose
  private dq!: Float32Array;
  private dk!: Float32Array;
  private dv!: Float32Array;
  private dAttOut!: Float32Array;
  private dAtt!: Float32Array; // [H, T, T]
  private dScores!: Float32Array; // [H, T, T]
  private dH1!: Float32Array; // [T, F]
  private dH1a!: Float32Array; // [T, F]
  private dLogits!: Float32Array; // [T, V]

  /** Number of tokens in the most recent forward pass. */
  private lastLen = 0;

  constructor(config: Partial<TransformerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    const { dModel, nHeads } = this.config;
    if (dModel % nHeads !== 0) {
      throw new Error(`dModel (${dModel}) must be divisible by nHeads (${nHeads})`);
    }
    this.headDim = dModel / nHeads;
    this.params = new ParamStore();
    this.declareParams();
    this.params.build();
    this.allocBuffers();
  }

  private declareParams(): void {
    const { vocabSize, dModel, dFF, nLayers, blockSize } = this.config;
    const p = this.params;
    p.declare("wte", [vocabSize, dModel], "embedding");
    p.declare("wpe", [blockSize, dModel], "embedding");
    for (let l = 0; l < nLayers; l++) {
      p.declare(`h${l}.ln1.g`, [dModel], "gain");
      p.declare(`h${l}.ln1.b`, [dModel], "bias");
      p.declare(`h${l}.wq`, [dModel, dModel], "weight");
      p.declare(`h${l}.bq`, [dModel], "bias");
      p.declare(`h${l}.wk`, [dModel, dModel], "weight");
      p.declare(`h${l}.bk`, [dModel], "bias");
      p.declare(`h${l}.wv`, [dModel, dModel], "weight");
      p.declare(`h${l}.bv`, [dModel], "bias");
      p.declare(`h${l}.wo`, [dModel, dModel], "weight");
      p.declare(`h${l}.bo`, [dModel], "bias");
      p.declare(`h${l}.ln2.g`, [dModel], "gain");
      p.declare(`h${l}.ln2.b`, [dModel], "bias");
      p.declare(`h${l}.w1`, [dModel, dFF], "weight");
      p.declare(`h${l}.b1`, [dFF], "bias");
      p.declare(`h${l}.w2`, [dFF, dModel], "weight");
      p.declare(`h${l}.b2`, [dModel], "bias");
    }
    p.declare("lnf.g", [dModel], "gain");
    p.declare("lnf.b", [dModel], "bias");
  }

  private allocBuffers(): void {
    const { dModel: D, dFF: F, nHeads: H, nLayers: L, blockSize: T, vocabSize: V } = this.config;
    const td = () => new Float32Array(T * D);
    const lnCache = (): LayerNormCache => ({
      mean: new Float32Array(T),
      rstd: new Float32Array(T),
    });
    for (let l = 0; l < L; l++) {
      this.layers.push({
        xIn: td(),
        ln1: td(),
        ln1Cache: lnCache(),
        q: td(),
        k: td(),
        v: td(),
        att: new Float32Array(H * T * T),
        attOut: td(),
        xMid: td(),
        ln2: td(),
        ln2Cache: lnCache(),
        h1: new Float32Array(T * F),
        h1a: new Float32Array(T * F),
      });
    }
    this.xFinal = td();
    this.lnfOut = td();
    this.lnfCache = lnCache();
    this.logits = new Float32Array(T * V);
    this.probs = new Float32Array(T * V);

    this.dx = td();
    this.dxMid = td();
    this.dxIn = td();
    this.dTmp = td();
    this.dq = td();
    this.dk = td();
    this.dv = td();
    this.dAttOut = td();
    this.dAtt = new Float32Array(H * T * T);
    this.dScores = new Float32Array(H * T * T);
    this.dH1 = new Float32Array(T * F);
    this.dH1a = new Float32Array(T * F);
    this.dLogits = new Float32Array(T * V);
  }

  get numParams(): number {
    return this.params.size;
  }

  /**
   * Run the forward pass over `tokens` (length <= blockSize). When `targets` is
   * given, also computes softmax probabilities and returns mean cross-entropy
   * (natural log) over positions whose target is >= 0.
   */
  forward(tokens: ArrayLike<number>, targets?: ArrayLike<number>): number {
    const { dModel: D, dFF: F, nHeads: H, nLayers: L, vocabSize: V } = this.config;
    const T = tokens.length;
    if (T === 0) throw new Error("forward requires at least one token");
    if (T > this.config.blockSize) {
      throw new Error(`sequence length ${T} exceeds blockSize ${this.config.blockSize}`);
    }
    this.lastLen = T;
    const p = this.params;
    const wte = p.view("wte");
    const wpe = p.view("wpe");

    // --- embeddings -------------------------------------------------------
    let x = this.layers[0].xIn;
    for (let t = 0; t < T; t++) {
      const tok = tokens[t];
      const src = tok * D;
      const pos = t * D;
      for (let i = 0; i < D; i++) x[pos + i] = wte[src + i] + wpe[pos + i];
    }

    // --- transformer blocks ----------------------------------------------
    for (let l = 0; l < L; l++) {
      // buf.xIn for l > 0 was written at the end of the previous iteration.
      const buf = this.layers[l];
      layerNormForward(buf.xIn, p.view(`h${l}.ln1.g`), p.view(`h${l}.ln1.b`), buf.ln1, T, D, buf.ln1Cache);

      matmul(buf.ln1, p.view(`h${l}.wq`), buf.q, T, D, D);
      addBiasRows(buf.q, p.view(`h${l}.bq`), T, D);
      matmul(buf.ln1, p.view(`h${l}.wk`), buf.k, T, D, D);
      addBiasRows(buf.k, p.view(`h${l}.bk`), T, D);
      matmul(buf.ln1, p.view(`h${l}.wv`), buf.v, T, D, D);
      addBiasRows(buf.v, p.view(`h${l}.bv`), T, D);

      this.attentionForward(buf, T);

      // projOut reuses dTmp as scratch; the result goes straight into xMid.
      const proj = this.dTmp;
      matmul(buf.attOut, p.view(`h${l}.wo`), proj, T, D, D);
      addBiasRows(proj, p.view(`h${l}.bo`), T, D);
      for (let i = 0; i < T * D; i++) buf.xMid[i] = buf.xIn[i] + proj[i];

      layerNormForward(buf.xMid, p.view(`h${l}.ln2.g`), p.view(`h${l}.ln2.b`), buf.ln2, T, D, buf.ln2Cache);
      matmul(buf.ln2, p.view(`h${l}.w1`), buf.h1, T, D, F);
      addBiasRows(buf.h1, p.view(`h${l}.b1`), T, F);
      for (let i = 0; i < T * F; i++) buf.h1a[i] = gelu(buf.h1[i]);

      const ffOut = this.dTmp;
      matmul(buf.h1a, p.view(`h${l}.w2`), ffOut, T, F, D);
      addBiasRows(ffOut, p.view(`h${l}.b2`), T, D);

      const next = l + 1 < L ? this.layers[l + 1].xIn : this.xFinal;
      for (let i = 0; i < T * D; i++) next[i] = buf.xMid[i] + ffOut[i];
    }

    // --- final norm + tied output head -----------------------------------
    layerNormForward(this.xFinal, p.view("lnf.g"), p.view("lnf.b"), this.lnfOut, T, D, this.lnfCache);
    matmulABt(this.lnfOut, wte, this.logits, T, D, V);

    if (!targets) return 0;

    let loss = 0;
    let count = 0;
    for (let t = 0; t < T; t++) {
      const off = t * V;
      this.probs.set(this.logits.subarray(off, off + V), off);
      softmaxInPlace(this.probs, off, V);
      const target = targets[t];
      if (target >= 0) {
        loss += -Math.log(Math.max(this.probs[off + target], 1e-12));
        count++;
      }
    }
    return count > 0 ? loss / count : 0;
  }

  private attentionForward(buf: LayerBuffers, T: number): void {
    const { dModel: D, nHeads: H } = this.config;
    const dh = this.headDim;
    const scale = 1 / Math.sqrt(dh);
    buf.att.fill(0, 0, H * T * T);
    for (let h = 0; h < H; h++) {
      const hOff = h * dh;
      for (let t = 0; t < T; t++) {
        const base = h * T * T + t * T;
        const qOff = t * D + hOff;
        // Causal mask: only attend to s <= t.
        for (let s = 0; s <= t; s++) {
          const kOff = s * D + hOff;
          let sum = 0;
          for (let i = 0; i < dh; i++) sum += buf.q[qOff + i] * buf.k[kOff + i];
          buf.att[base + s] = sum * scale;
        }
        softmaxInPlace(buf.att, base, t + 1);
        const outOff = t * D + hOff;
        for (let i = 0; i < dh; i++) buf.attOut[outOff + i] = 0;
        for (let s = 0; s <= t; s++) {
          const a = buf.att[base + s];
          const vOff = s * D + hOff;
          for (let i = 0; i < dh; i++) buf.attOut[outOff + i] += a * buf.v[vOff + i];
        }
      }
    }
  }

  /**
   * Backward pass for the loss computed by the last `forward(tokens, targets)`.
   * Accumulates into `params.grad` - call `params.zeroGrad()` first.
   */
  backward(tokens: ArrayLike<number>, targets: ArrayLike<number>): void {
    const { dModel: D, dFF: F, nHeads: H, nLayers: L, vocabSize: V } = this.config;
    const T = this.lastLen;
    if (T !== tokens.length) throw new Error("backward called with a different sequence");
    const p = this.params;
    const wte = p.view("wte");
    const dWte = p.gradView("wte");
    const dWpe = p.gradView("wpe");

    // --- cross-entropy -> dLogits ----------------------------------------
    let count = 0;
    for (let t = 0; t < T; t++) if (targets[t] >= 0) count++;
    const invN = count > 0 ? 1 / count : 0;
    this.dLogits.fill(0, 0, T * V);
    for (let t = 0; t < T; t++) {
      const target = targets[t];
      if (target < 0) continue;
      const off = t * V;
      for (let j = 0; j < V; j++) this.dLogits[off + j] = this.probs[off + j] * invN;
      this.dLogits[off + target] -= invN;
    }

    // logits = lnfOut @ wte^T
    const dLnfOut = this.dTmp;
    dLnfOut.fill(0, 0, T * D);
    matmulAdd(this.dLogits, wte, dLnfOut, T, V, D);
    matmulAddAtB(this.dLogits, this.lnfOut, dWte, T, V, D);

    this.dx.fill(0, 0, T * D);
    layerNormBackward(
      dLnfOut,
      this.xFinal,
      p.view("lnf.g"),
      this.lnfCache,
      this.dx,
      p.gradView("lnf.g"),
      p.gradView("lnf.b"),
      T,
      D,
    );

    // --- blocks, top down -------------------------------------------------
    for (let l = L - 1; l >= 0; l--) {
      const buf = this.layers[l];

      // ---- MLP branch ----
      const dFfOut = this.dx; // gradient arriving at the block output
      this.dH1a.fill(0, 0, T * F);
      matmulAddABt(dFfOut, p.view(`h${l}.w2`), this.dH1a, T, D, F);
      matmulAddAtB(buf.h1a, dFfOut, p.gradView(`h${l}.w2`), T, F, D);
      biasGrad(dFfOut, p.gradView(`h${l}.b2`), T, D);

      for (let i = 0; i < T * F; i++) this.dH1[i] = this.dH1a[i] * geluGrad(buf.h1[i]);

      const dLn2 = this.dxIn; // reuse as scratch before it holds dxIn
      dLn2.fill(0, 0, T * D);
      matmulAddABt(this.dH1, p.view(`h${l}.w1`), dLn2, T, F, D);
      matmulAddAtB(buf.ln2, this.dH1, p.gradView(`h${l}.w1`), T, D, F);
      biasGrad(this.dH1, p.gradView(`h${l}.b1`), T, F);

      this.dxMid.fill(0, 0, T * D);
      layerNormBackward(
        dLn2,
        buf.xMid,
        p.view(`h${l}.ln2.g`),
        buf.ln2Cache,
        this.dxMid,
        p.gradView(`h${l}.ln2.g`),
        p.gradView(`h${l}.ln2.b`),
        T,
        D,
      );
      // Residual: x_out = x_mid + ffOut, so the block-output gradient also
      // flows straight into x_mid.
      for (let i = 0; i < T * D; i++) this.dxMid[i] += dFfOut[i];

      // ---- attention branch ----
      this.dAttOut.fill(0, 0, T * D);
      matmulAddABt(this.dxMid, p.view(`h${l}.wo`), this.dAttOut, T, D, D);
      matmulAddAtB(buf.attOut, this.dxMid, p.gradView(`h${l}.wo`), T, D, D);
      biasGrad(this.dxMid, p.gradView(`h${l}.bo`), T, D);

      this.dq.fill(0, 0, T * D);
      this.dk.fill(0, 0, T * D);
      this.dv.fill(0, 0, T * D);
      this.attentionBackward(buf, T);

      const dLn1 = this.dTmp;
      dLn1.fill(0, 0, T * D);
      matmulAddABt(this.dq, p.view(`h${l}.wq`), dLn1, T, D, D);
      matmulAddABt(this.dk, p.view(`h${l}.wk`), dLn1, T, D, D);
      matmulAddABt(this.dv, p.view(`h${l}.wv`), dLn1, T, D, D);
      matmulAddAtB(buf.ln1, this.dq, p.gradView(`h${l}.wq`), T, D, D);
      matmulAddAtB(buf.ln1, this.dk, p.gradView(`h${l}.wk`), T, D, D);
      matmulAddAtB(buf.ln1, this.dv, p.gradView(`h${l}.wv`), T, D, D);
      biasGrad(this.dq, p.gradView(`h${l}.bq`), T, D);
      biasGrad(this.dk, p.gradView(`h${l}.bk`), T, D);
      biasGrad(this.dv, p.gradView(`h${l}.bv`), T, D);

      this.dxIn.fill(0, 0, T * D);
      layerNormBackward(
        dLn1,
        buf.xIn,
        p.view(`h${l}.ln1.g`),
        buf.ln1Cache,
        this.dxIn,
        p.gradView(`h${l}.ln1.g`),
        p.gradView(`h${l}.ln1.b`),
        T,
        D,
      );
      for (let i = 0; i < T * D; i++) this.dxIn[i] += this.dxMid[i];

      // dxIn becomes the incoming gradient for the layer below.
      this.dx.set(this.dxIn.subarray(0, T * D));
    }

    // --- embeddings -------------------------------------------------------
    for (let t = 0; t < T; t++) {
      const tok = tokens[t];
      const off = t * D;
      const dst = tok * D;
      for (let i = 0; i < D; i++) {
        dWte[dst + i] += this.dx[off + i];
        dWpe[off + i] += this.dx[off + i];
      }
    }
  }

  private attentionBackward(buf: LayerBuffers, T: number): void {
    const { dModel: D, nHeads: H } = this.config;
    const dh = this.headDim;
    const scale = 1 / Math.sqrt(dh);
    for (let h = 0; h < H; h++) {
      const hOff = h * dh;
      for (let t = 0; t < T; t++) {
        const base = h * T * T + t * T;
        const len = t + 1;
        const outOff = t * D + hOff;
        for (let s = 0; s < len; s++) {
          const vOff = s * D + hOff;
          const a = buf.att[base + s];
          let ds = 0;
          for (let i = 0; i < dh; i++) {
            const g = this.dAttOut[outOff + i];
            ds += g * buf.v[vOff + i];
            this.dv[vOff + i] += a * g;
          }
          this.dAtt[base + s] = ds;
        }
        softmaxBackward(buf.att, this.dAtt, this.dScores, base, len);
        const qOff = t * D + hOff;
        for (let s = 0; s < len; s++) {
          const g = this.dScores[base + s] * scale;
          if (g === 0) continue;
          const kOff = s * D + hOff;
          for (let i = 0; i < dh; i++) {
            this.dq[qOff + i] += g * buf.k[kOff + i];
            this.dk[kOff + i] += g * buf.q[qOff + i];
          }
        }
      }
    }
  }

  /**
   * log-softmax over the vocabulary at the final position of `tokens`. This is
   * the only inference entry point the completion engine needs: it scores
   * candidates proposed by the n-gram model rather than decoding on its own.
   */
  nextLogProbs(tokens: ArrayLike<number>, out?: Float32Array): Float32Array {
    const V = this.config.vocabSize;
    let window: ArrayLike<number> = tokens;
    if (tokens.length > this.config.blockSize) {
      const keep = new Int32Array(this.config.blockSize);
      const start = tokens.length - this.config.blockSize;
      for (let i = 0; i < keep.length; i++) keep[i] = tokens[start + i];
      window = keep;
    }
    this.forward(window);
    const off = (window.length - 1) * V;
    const dst = out ?? new Float32Array(V);
    let max = -Infinity;
    for (let j = 0; j < V; j++) {
      const v = this.logits[off + j];
      if (v > max) max = v;
    }
    let sum = 0;
    for (let j = 0; j < V; j++) sum += Math.exp(this.logits[off + j] - max);
    const logSum = max + Math.log(sum);
    for (let j = 0; j < V; j++) dst[j] = this.logits[off + j] - logSum;
    return dst;
  }

  // --- checkpointing ------------------------------------------------------

  save(file: string): void {
    const header = Buffer.from(
      JSON.stringify({ version: 1, kind: "pca-transformer", config: this.config }),
      "utf8",
    );
    const lenBuf = Buffer.alloc(4);
    lenBuf.writeUInt32LE(header.length, 0);
    const weights = Buffer.from(
      this.params.data.buffer,
      this.params.data.byteOffset,
      this.params.data.byteLength,
    );
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const tmp = `${file}.tmp${process.pid}`;
    fs.writeFileSync(tmp, Buffer.concat([lenBuf, header, weights]));
    fs.renameSync(tmp, file);
  }

  static load(file: string): Transformer {
    const buf = fs.readFileSync(file);
    const headerLen = buf.readUInt32LE(0);
    const header = JSON.parse(buf.subarray(4, 4 + headerLen).toString("utf8")) as {
      config: TransformerConfig;
    };
    const model = new Transformer(header.config);
    const weights = buf.subarray(4 + headerLen);
    const expected = model.params.data.byteLength;
    if (weights.byteLength !== expected) {
      throw new Error(
        `checkpoint weight size mismatch: got ${weights.byteLength}, expected ${expected}`,
      );
    }
    Buffer.from(
      model.params.data.buffer,
      model.params.data.byteOffset,
      model.params.data.byteLength,
    ).set(weights);
    return model;
  }
}
