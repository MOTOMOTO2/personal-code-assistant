import fs from "node:fs";
import path from "node:path";
import { readJsonGz, writeJsonGz } from "../util/io.js";

/**
 * On-disk code index plus hybrid retrieval.
 *
 * Retrieval is the difference between an assistant that guesses about your
 * codebase and one that quotes it. Two rankers run over the same chunks and
 * their rankings are fused:
 *
 *   1. BM25 over sub-word identifier terms - exact, great for names you
 *      remember ("parseManifest").
 *   2. Cosine similarity over locally-trained embeddings - fuzzy, great for
 *      intent you can only describe ("where do we retry failed uploads").
 *
 * Reciprocal Rank Fusion combines them without needing the two score scales to
 * be comparable, which they are not.
 */

export interface IndexedChunk {
  id: number;
  /** Index into `files`. */
  file: number;
  startLine: number;
  endLine: number;
  symbol: string | null;
  /** All declarations inside the chunk; `symbol` is the first of these. */
  symbols: string[];
  text: string;
}

export interface IndexedFile {
  path: string;
  size: number;
  mtimeMs: number;
  /** Token count contributed to the language-model corpus. */
  tokens: number;
  /**
   * Name of the `pca corpus` source this file came from, absent for your own
   * code. The path is already prefixed `@name/`, so this is for filtering and
   * counting rather than display.
   */
  external?: string;
}

interface Bm25Data {
  avgdl: number;
  /** term -> [chunkId, tf, chunkId, tf, ...] */
  postings: Record<string, number[]>;
  /** Per-chunk document length in search terms. */
  lengths: number[];
}

export interface IndexData {
  version: 2;
  createdAt: string;
  root: string;
  files: IndexedFile[];
  chunks: Array<Omit<IndexedChunk, "id">>;
  bm25: Bm25Data;
  embeddingDim: number;
}

export interface SearchHit {
  chunk: IndexedChunk;
  file: string;
  score: number;
  /** Which rankers contributed, for explainability in `pca search --explain`. */
  lexicalRank: number | null;
  vectorRank: number | null;
}

/**
 * Split code text into search terms: lowercase words, plus the sub-words of
 * camelCase / snake_case / kebab-case identifiers. Indexing both `parseManifest`
 * and `parse`+`manifest` is what lets a query match either form.
 */
export function searchTerms(text: string): string[] {
  const out: string[] = [];
  const words = text.match(/[A-Za-z_$][\w$]*|\d+/g);
  if (!words) return out;
  for (const word of words) {
    const lower = word.toLowerCase();
    if (lower.length > 1) out.push(lower);
    const parts = word
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
      .split(/[^A-Za-z0-9]+/);
    if (parts.length > 1) {
      for (const p of parts) {
        const lp = p.toLowerCase();
        if (lp.length > 1 && lp !== lower) out.push(lp);
      }
    }
  }
  return out;
}

const BM25_K1 = 1.2;
const BM25_B = 0.75;
const RRF_K = 60;

export class IndexStore {
  readonly root: string;
  readonly files: IndexedFile[];
  readonly chunks: IndexedChunk[];
  readonly createdAt: string;
  private bm25: Bm25Data;
  private embeddingDim: number;
  /** [numChunks x embeddingDim], row-normalised. Empty when not yet trained. */
  private vectors: Float32Array;

  constructor(
    root: string,
    files: IndexedFile[],
    chunks: IndexedChunk[],
    bm25: Bm25Data,
    embeddingDim = 0,
    vectors = new Float32Array(0),
    createdAt = new Date().toISOString(),
  ) {
    this.root = root;
    this.files = files;
    this.chunks = chunks;
    this.bm25 = bm25;
    this.embeddingDim = embeddingDim;
    this.vectors = vectors;
    this.createdAt = createdAt;
  }

  get hasVectors(): boolean {
    return this.embeddingDim > 0 && this.vectors.length === this.chunks.length * this.embeddingDim;
  }

  get dim(): number {
    return this.embeddingDim;
  }

  filePath(chunk: IndexedChunk): string {
    return this.files[chunk.file]?.path ?? "<unknown>";
  }

  /** Build BM25 statistics from the chunk texts. */
  static buildBm25(chunks: Array<{ text: string }>): Bm25Data {
    const postings: Record<string, number[]> = Object.create(null);
    const lengths: number[] = [];
    let totalLen = 0;
    for (let id = 0; id < chunks.length; id++) {
      const terms = searchTerms(chunks[id].text);
      lengths.push(terms.length);
      totalLen += terms.length;
      const tf = new Map<string, number>();
      for (const t of terms) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const [term, count] of tf) {
        const list = postings[term] ?? (postings[term] = []);
        list.push(id, count);
      }
    }
    return { avgdl: chunks.length > 0 ? totalLen / chunks.length : 0, postings, lengths };
  }

  attachVectors(dim: number, vectors: Float32Array): void {
    this.embeddingDim = dim;
    this.vectors = vectors;
  }

  vectorOf(chunkId: number, out?: Float32Array): Float32Array | null {
    if (!this.hasVectors) return null;
    const dst = out ?? new Float32Array(this.embeddingDim);
    dst.set(this.vectors.subarray(chunkId * this.embeddingDim, (chunkId + 1) * this.embeddingDim));
    return dst;
  }

  /** BM25 scores for a query, as a ranked list of chunk ids. */
  lexicalSearch(query: string, limit: number): Array<{ id: number; score: number }> {
    const terms = searchTerms(query);
    if (terms.length === 0) return [];
    const N = this.chunks.length;
    const scores = new Map<number, number>();
    const queryTf = new Map<string, number>();
    for (const t of terms) queryTf.set(t, (queryTf.get(t) ?? 0) + 1);

    for (const [term, qtf] of queryTf) {
      const list = this.bm25.postings[term];
      if (!list) continue;
      const df = list.length / 2;
      // Robertson/Sparck-Jones IDF with the +0.5 smoothing, floored at a small
      // positive value so terms in most documents do not score negatively.
      const idf = Math.max(0.05, Math.log(1 + (N - df + 0.5) / (df + 0.5)));
      for (let p = 0; p < list.length; p += 2) {
        const id = list[p];
        const tf = list[p + 1];
        const dl = this.bm25.lengths[id] || 1;
        const denom = tf + BM25_K1 * (1 - BM25_B + (BM25_B * dl) / (this.bm25.avgdl || 1));
        const contribution = idf * ((tf * (BM25_K1 + 1)) / denom) * Math.min(qtf, 3);
        scores.set(id, (scores.get(id) ?? 0) + contribution);
      }
    }
    return [...scores.entries()]
      .map(([id, score]) => ({ id, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  /** Cosine similarity ranking against a query vector (assumed normalised). */
  vectorSearch(query: Float32Array, limit: number): Array<{ id: number; score: number }> {
    if (!this.hasVectors) return [];
    const dim = this.embeddingDim;
    const out: Array<{ id: number; score: number }> = [];
    for (let id = 0; id < this.chunks.length; id++) {
      const off = id * dim;
      let s = 0;
      for (let d = 0; d < dim; d++) s += query[d] * this.vectors[off + d];
      if (s > 0) out.push({ id, score: s });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, limit);
  }

  /**
   * Hybrid search. `queryVector` is optional - without it this degrades to pure
   * BM25, which is exactly what happens before embeddings are trained.
   */
  search(
    query: string,
    opts: { limit?: number; queryVector?: Float32Array | null; pool?: number } = {},
  ): SearchHit[] {
    const limit = opts.limit ?? 8;
    const pool = opts.pool ?? Math.max(limit * 8, 50);
    const lexical = this.lexicalSearch(query, pool);
    const vector = opts.queryVector ? this.vectorSearch(opts.queryVector, pool) : [];

    const lexRank = new Map<number, number>();
    lexical.forEach((r, i) => lexRank.set(r.id, i + 1));
    const vecRank = new Map<number, number>();
    vector.forEach((r, i) => vecRank.set(r.id, i + 1));

    // Reciprocal Rank Fusion: sum of 1/(k + rank) across rankers.
    const fused = new Map<number, number>();
    for (const [id, rank] of lexRank) {
      fused.set(id, (fused.get(id) ?? 0) + 1 / (RRF_K + rank));
    }
    for (const [id, rank] of vecRank) {
      fused.set(id, (fused.get(id) ?? 0) + 1 / (RRF_K + rank));
    }

    return [...fused.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([id, score]) => ({
        chunk: this.chunks[id],
        file: this.filePath(this.chunks[id]),
        score,
        lexicalRank: lexRank.get(id) ?? null,
        vectorRank: vecRank.get(id) ?? null,
      }));
  }

  /** Whether a search term appears anywhere in the index at all. */
  knowsTerm(term: string): boolean {
    return this.bm25.postings[term] !== undefined;
  }

  /** Fraction of the given terms the index has ever seen. */
  coverage(terms: Iterable<string>): number {
    let total = 0;
    let known = 0;
    for (const term of terms) {
      total++;
      if (this.knowsTerm(term)) known++;
    }
    return total > 0 ? known / total : 0;
  }

  /** Chunks belonging to one file path, in line order. */
  chunksForFile(relPath: string): IndexedChunk[] {
    const fileIdx = this.files.findIndex((f) => f.path === relPath);
    if (fileIdx < 0) return [];
    return this.chunks.filter((c) => c.file === fileIdx).sort((a, b) => a.startLine - b.startLine);
  }

  stats(): { files: number; chunks: number; terms: number; tokens: number; hasVectors: boolean } {
    return {
      files: this.files.length,
      chunks: this.chunks.length,
      terms: Object.keys(this.bm25.postings).length,
      tokens: this.files.reduce((a, f) => a + f.tokens, 0),
      hasVectors: this.hasVectors,
    };
  }

  save(indexFile: string, vectorFile: string): void {
    const data: IndexData = {
      version: 2,
      createdAt: this.createdAt,
      root: this.root,
      files: this.files,
      chunks: this.chunks.map(({ id, ...rest }) => rest),
      bm25: this.bm25,
      embeddingDim: this.embeddingDim,
    };
    writeJsonGz(indexFile, data);
    if (this.hasVectors) {
      fs.mkdirSync(path.dirname(vectorFile), { recursive: true });
      const tmp = `${vectorFile}.tmp${process.pid}`;
      fs.writeFileSync(
        tmp,
        Buffer.from(this.vectors.buffer, this.vectors.byteOffset, this.vectors.byteLength),
      );
      fs.renameSync(tmp, vectorFile);
    }
  }

  static load(indexFile: string, vectorFile: string): IndexStore {
    const data = readJsonGz<IndexData>(indexFile);
    const chunks: IndexedChunk[] = data.chunks.map((c, id) => ({
      id,
      ...c,
      symbols: c.symbols ?? (c.symbol ? [c.symbol] : []),
    }));
    let vectors = new Float32Array(0);
    if (data.embeddingDim > 0 && fs.existsSync(vectorFile)) {
      const buf = fs.readFileSync(vectorFile);
      vectors = new Float32Array(data.embeddingDim * chunks.length);
      const expected = vectors.byteLength;
      if (buf.byteLength === expected) {
        Buffer.from(vectors.buffer, vectors.byteOffset, vectors.byteLength).set(buf);
      } else {
        vectors = new Float32Array(0);
      }
    }
    return new IndexStore(
      data.root,
      data.files,
      chunks,
      data.bm25,
      vectors.length > 0 ? data.embeddingDim : 0,
      vectors,
      data.createdAt,
    );
  }
}
