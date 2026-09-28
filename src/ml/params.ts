import { Rng } from "../util/rng.js";

export interface ParamSpec {
  name: string;
  offset: number;
  size: number;
  shape: number[];
  /** Weight decay and init scale differ for matrices vs. biases/gains. */
  kind: "weight" | "bias" | "gain" | "embedding";
}

/**
 * All model parameters live in one flat Float32Array, with gradients in a
 * parallel array of the same shape. Named views are `subarray` slices, so
 * writing through a view writes the backing buffer.
 *
 * Keeping everything contiguous means the optimizer is a single loop over two
 * arrays, checkpoints are one binary blob, and there is no per-tensor object
 * overhead in the hot path.
 */
export class ParamStore {
  data!: Float32Array;
  grad!: Float32Array;
  private specs: ParamSpec[] = [];
  private byName = new Map<string, ParamSpec>();
  private cursor = 0;
  private built = false;

  declare(name: string, shape: number[], kind: ParamSpec["kind"]): void {
    if (this.built) throw new Error("ParamStore already built");
    if (this.byName.has(name)) throw new Error(`duplicate parameter ${name}`);
    const size = shape.reduce((a, b) => a * b, 1);
    const spec: ParamSpec = { name, offset: this.cursor, size, shape, kind };
    this.specs.push(spec);
    this.byName.set(name, spec);
    this.cursor += size;
  }

  build(seed = 1234): void {
    this.data = new Float32Array(this.cursor);
    this.grad = new Float32Array(this.cursor);
    const rng = new Rng(seed);
    for (const spec of this.specs) {
      const view = this.data.subarray(spec.offset, spec.offset + spec.size);
      switch (spec.kind) {
        case "gain":
          view.fill(1);
          break;
        case "bias":
          view.fill(0);
          break;
        case "embedding":
          for (let i = 0; i < view.length; i++) view[i] = rng.normal() * 0.02;
          break;
        case "weight": {
          // He/Xavier-ish: scale by 1/sqrt(fan_in), which keeps activation
          // variance stable through a pre-LN residual stack.
          const fanIn = spec.shape[0] ?? 1;
          const scale = 1 / Math.sqrt(fanIn);
          for (let i = 0; i < view.length; i++) view[i] = rng.normal() * scale;
          break;
        }
      }
    }
    this.built = true;
  }

  get size(): number {
    return this.cursor;
  }

  has(name: string): boolean {
    return this.byName.has(name);
  }

  spec(name: string): ParamSpec {
    const s = this.byName.get(name);
    if (!s) throw new Error(`unknown parameter ${name}`);
    return s;
  }

  view(name: string): Float32Array {
    const s = this.spec(name);
    return this.data.subarray(s.offset, s.offset + s.size);
  }

  gradView(name: string): Float32Array {
    const s = this.spec(name);
    return this.grad.subarray(s.offset, s.offset + s.size);
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  /** Names in declaration order - the checkpoint layout depends on this. */
  names(): string[] {
    return this.specs.map((s) => s.name);
  }

  allSpecs(): ParamSpec[] {
    return this.specs;
  }

  /** Mask of which slots should receive weight decay (matrices only). */
  decayMask(): Uint8Array {
    const mask = new Uint8Array(this.cursor);
    for (const spec of this.specs) {
      if (spec.kind === "weight" || spec.kind === "embedding") {
        mask.fill(1, spec.offset, spec.offset + spec.size);
      }
    }
    return mask;
  }
}

export interface AdamOptions {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
}

/**
 * AdamW over a flat parameter vector. Decoupled weight decay is applied only to
 * matrices (see `ParamStore.decayMask`) - decaying LayerNorm gains or biases
 * measurably hurts small models.
 */
export class AdamW {
  private m: Float32Array;
  private v: Float32Array;
  private mask: Uint8Array;
  private t = 0;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  lr: number;

  constructor(private store: ParamStore, opts: AdamOptions = {}) {
    this.m = new Float32Array(store.size);
    this.v = new Float32Array(store.size);
    this.mask = store.decayMask();
    this.lr = opts.lr ?? 3e-3;
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.95;
    this.eps = opts.eps ?? 1e-8;
    this.weightDecay = opts.weightDecay ?? 0.01;
  }

  get step(): number {
    return this.t;
  }

  /** Clip the global gradient norm to `maxNorm`; returns the pre-clip norm. */
  clipGradNorm(maxNorm: number): number {
    const g = this.store.grad;
    let sum = 0;
    for (let i = 0; i < g.length; i++) sum += g[i] * g[i];
    const norm = Math.sqrt(sum);
    if (norm > maxNorm && norm > 0) {
      const s = maxNorm / norm;
      for (let i = 0; i < g.length; i++) g[i] *= s;
    }
    return norm;
  }

  update(lr = this.lr): void {
    this.t++;
    const { data, grad } = this.store;
    const b1 = this.beta1;
    const b2 = this.beta2;
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    const wd = this.weightDecay * lr;
    for (let i = 0; i < data.length; i++) {
      const g = grad[i];
      const mi = b1 * this.m[i] + (1 - b1) * g;
      const vi = b2 * this.v[i] + (1 - b2) * g * g;
      this.m[i] = mi;
      this.v[i] = vi;
      const mHat = mi / bc1;
      const vHat = vi / bc2;
      let p = data[i];
      if (this.mask[i]) p -= wd * p;
      data[i] = p - (lr * mHat) / (Math.sqrt(vHat) + this.eps);
    }
  }
}

/** Linear warmup then cosine decay to 10% of peak. */
export function lrSchedule(
  step: number,
  totalSteps: number,
  peakLr: number,
  warmupSteps: number,
): number {
  if (step < warmupSteps) return (peakLr * (step + 1)) / Math.max(1, warmupSteps);
  const progress = Math.min(1, (step - warmupSteps) / Math.max(1, totalSteps - warmupSteps));
  return peakLr * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * progress)));
}
