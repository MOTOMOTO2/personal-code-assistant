/**
 * Dense float32 kernels. These are the hot loops of neural training, so the
 * shapes are passed explicitly and the loop order is chosen for cache locality
 * rather than readability: row-major operands, sequential writes to the output.
 *
 * All matrices are row-major flat Float32Arrays.
 */

/** C[M,N] = A[M,K] * B[K,N] */
export function matmul(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  M: number,
  K: number,
  N: number,
  aOff = 0,
  bOff = 0,
  cOff = 0,
): void {
  C.fill(0, cOff, cOff + M * N);
  matmulAdd(A, B, C, M, K, N, aOff, bOff, cOff);
}

/** C[M,N] += A[M,K] * B[K,N] - "ikj" order keeps B and C rows sequential. */
export function matmulAdd(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  M: number,
  K: number,
  N: number,
  aOff = 0,
  bOff = 0,
  cOff = 0,
): void {
  for (let i = 0; i < M; i++) {
    const aRow = aOff + i * K;
    const cRow = cOff + i * N;
    for (let k = 0; k < K; k++) {
      const a = A[aRow + k];
      if (a === 0) continue;
      const bRow = bOff + k * N;
      for (let j = 0; j < N; j++) C[cRow + j] += a * B[bRow + j];
    }
  }
}

/** C[M,N] += A[M,K] * B[N,K]^T - dot-product form, both operands sequential in k. */
export function matmulAddABt(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  M: number,
  K: number,
  N: number,
  aOff = 0,
  bOff = 0,
  cOff = 0,
): void {
  for (let i = 0; i < M; i++) {
    const aRow = aOff + i * K;
    const cRow = cOff + i * N;
    for (let j = 0; j < N; j++) {
      const bRow = bOff + j * K;
      let sum = 0;
      for (let k = 0; k < K; k++) sum += A[aRow + k] * B[bRow + k];
      C[cRow + j] += sum;
    }
  }
}

/** C[M,N] = A[M,K] * B[N,K]^T */
export function matmulABt(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  M: number,
  K: number,
  N: number,
  aOff = 0,
  bOff = 0,
  cOff = 0,
): void {
  C.fill(0, cOff, cOff + M * N);
  matmulAddABt(A, B, C, M, K, N, aOff, bOff, cOff);
}

/** C[M,N] += A[K,M]^T * B[K,N] - the classic weight-gradient shape. */
export function matmulAddAtB(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  K: number,
  M: number,
  N: number,
  aOff = 0,
  bOff = 0,
  cOff = 0,
): void {
  for (let k = 0; k < K; k++) {
    const aRow = aOff + k * M;
    const bRow = bOff + k * N;
    for (let i = 0; i < M; i++) {
      const a = A[aRow + i];
      if (a === 0) continue;
      const cRow = cOff + i * N;
      for (let j = 0; j < N; j++) C[cRow + j] += a * B[bRow + j];
    }
  }
}

/** Add a bias vector of length N to each of M rows, in place. */
export function addBiasRows(
  X: Float32Array,
  bias: Float32Array,
  M: number,
  N: number,
  xOff = 0,
): void {
  for (let i = 0; i < M; i++) {
    const row = xOff + i * N;
    for (let j = 0; j < N; j++) X[row + j] += bias[j];
  }
}

/** Accumulate a bias gradient: grad[j] += sum over rows of dY[i,j]. */
export function biasGrad(
  dY: Float32Array,
  grad: Float32Array,
  M: number,
  N: number,
  dOff = 0,
): void {
  for (let i = 0; i < M; i++) {
    const row = dOff + i * N;
    for (let j = 0; j < N; j++) grad[j] += dY[row + j];
  }
}

/** Numerically stable in-place softmax over `len` values starting at `off`. */
export function softmaxInPlace(X: Float32Array, off: number, len: number): void {
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = X[off + i];
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const e = Math.exp(X[off + i] - max);
    X[off + i] = e;
    sum += e;
  }
  const inv = 1 / (sum || 1);
  for (let i = 0; i < len; i++) X[off + i] *= inv;
}

/**
 * Backprop through a softmax whose output is `p`: dx = p * (dp - sum(dp * p)).
 * Operates on one row and writes into `dx`.
 */
export function softmaxBackward(
  p: Float32Array,
  dp: Float32Array,
  dx: Float32Array,
  off: number,
  len: number,
): void {
  let dot = 0;
  for (let i = 0; i < len; i++) dot += dp[off + i] * p[off + i];
  for (let i = 0; i < len; i++) dx[off + i] = p[off + i] * (dp[off + i] - dot);
}

/** Tanh approximation of GELU, matching the usual transformer implementation. */
const GELU_C = Math.sqrt(2 / Math.PI);

export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
}

/** d/dx of the tanh-approximated GELU. */
export function geluGrad(x: number): number {
  const x3 = x * x * x;
  const inner = GELU_C * (x + 0.044715 * x3);
  const t = Math.tanh(inner);
  const sech2 = 1 - t * t;
  return 0.5 * (1 + t) + 0.5 * x * sech2 * GELU_C * (1 + 3 * 0.044715 * x * x);
}

export interface LayerNormCache {
  /** Per-row mean. */
  mean: Float32Array;
  /** Per-row 1/sqrt(var + eps). */
  rstd: Float32Array;
}

export const LN_EPS = 1e-5;

/**
 * Row-wise layer norm: Y = (X - mean) * rstd * gamma + beta.
 * Caches mean/rstd because the backward pass needs both.
 */
export function layerNormForward(
  X: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  Y: Float32Array,
  M: number,
  N: number,
  cache: LayerNormCache,
  xOff = 0,
  yOff = 0,
): void {
  for (let i = 0; i < M; i++) {
    const xRow = xOff + i * N;
    const yRow = yOff + i * N;
    let mean = 0;
    for (let j = 0; j < N; j++) mean += X[xRow + j];
    mean /= N;
    let variance = 0;
    for (let j = 0; j < N; j++) {
      const d = X[xRow + j] - mean;
      variance += d * d;
    }
    variance /= N;
    const rstd = 1 / Math.sqrt(variance + LN_EPS);
    cache.mean[i] = mean;
    cache.rstd[i] = rstd;
    for (let j = 0; j < N; j++) {
      Y[yRow + j] = (X[xRow + j] - mean) * rstd * gamma[j] + beta[j];
    }
  }
}

/**
 * Layer norm backward. Accumulates into `dGamma`/`dBeta` and writes dX
 * (accumulating, so residual branches can share the buffer).
 */
export function layerNormBackward(
  dY: Float32Array,
  X: Float32Array,
  gamma: Float32Array,
  cache: LayerNormCache,
  dX: Float32Array,
  dGamma: Float32Array,
  dBeta: Float32Array,
  M: number,
  N: number,
  dyOff = 0,
  xOff = 0,
  dxOff = 0,
): void {
  for (let i = 0; i < M; i++) {
    const dyRow = dyOff + i * N;
    const xRow = xOff + i * N;
    const dxRow = dxOff + i * N;
    const mean = cache.mean[i];
    const rstd = cache.rstd[i];
    // Two reductions first: sum(dy*gamma) and sum(dy*gamma*xhat).
    let sumDg = 0;
    let sumDgXhat = 0;
    for (let j = 0; j < N; j++) {
      const xhat = (X[xRow + j] - mean) * rstd;
      const dg = dY[dyRow + j] * gamma[j];
      sumDg += dg;
      sumDgXhat += dg * xhat;
      dGamma[j] += dY[dyRow + j] * xhat;
      dBeta[j] += dY[dyRow + j];
    }
    const invN = 1 / N;
    for (let j = 0; j < N; j++) {
      const xhat = (X[xRow + j] - mean) * rstd;
      const dg = dY[dyRow + j] * gamma[j];
      dX[dxRow + j] += rstd * (dg - invN * sumDg - invN * xhat * sumDgXhat);
    }
  }
}

/** L2 norm of a slice, used for gradient clipping. */
export function l2Norm(a: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * a[i];
  return Math.sqrt(sum);
}

export function scaleInPlace(a: Float32Array, s: number): void {
  for (let i = 0; i < a.length; i++) a[i] *= s;
}

export function dot(a: Float32Array, b: Float32Array, aOff = 0, bOff = 0, len = a.length): number {
  let sum = 0;
  for (let i = 0; i < len; i++) sum += a[aOff + i] * b[bOff + i];
  return sum;
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let dotv = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dotv += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dotv / denom;
}
