import crypto from "node:crypto";
import { Bundle, Manifest, MANIFEST_VERSION } from "./bundle.js";
import { StyleAnalyzer } from "./style.js";
import { chunkFile } from "../index/chunker.js";
import { IndexStore, IndexedChunk, IndexedFile } from "../index/store.js";
import { SourceFile, readSource, walkSources } from "../index/walker.js";
import { EmbeddingModel, normalize } from "../ml/embeddings.js";
import { NgramModel } from "../ml/ngram.js";
import { Tokenizer } from "../tokenizer/tokenizer.js";
import { ensureStateDir } from "../util/paths.js";
import { fmtDuration, ok, progress, progressDone, step, warn } from "../util/log.js";
import { humanBytes } from "../util/io.js";
import path from "node:path";

export interface TrainOptions {
  /** Tokenizer vocabulary size. Larger favours the n-gram, costs neural speed. */
  vocabSize?: number;
  /** Maximum n-gram order. */
  order?: number;
  /** Cap on corpus tokens, to bound n-gram memory on very large repos. */
  maxTokens?: number;
  /** Embedding dimensionality. 0 disables embedding training. */
  embeddingDim?: number;
  /** Embedding training epochs. */
  embeddingEpochs?: number;
  /** Fraction of files held out for evaluation. */
  valFraction?: number;
  /** Extra file extensions to index. */
  extraExtensions?: string[];
  /** Skip building the retrieval index (faster, but disables search/ask). */
  skipIndex?: boolean;
}

export interface TrainReport {
  files: number;
  bytes: number;
  corpusTokens: number;
  vocabSize: number;
  ngram: { order: number; contexts: number; ngrams: number };
  chunks: number;
  embeddingDim: number;
  valFiles: number;
  seconds: number;
}

/**
 * Deterministic train/validation split by path hash. Using a hash rather than a
 * random shuffle means the split survives re-indexing, so perplexity numbers
 * stay comparable between runs.
 */
export function isValidationFile(relPath: string, valFraction: number): boolean {
  if (valFraction <= 0) return false;
  const digest = crypto.createHash("sha1").update(relPath).digest();
  const bucket = digest.readUInt16BE(0) / 0x10000;
  return bucket < valFraction;
}

/**
 * Train the fast models: tokenizer, n-gram language model, retrieval index,
 * embeddings and the style profile. Everything here is counting-based or
 * single-pass SGD, so a mid-size repo finishes in seconds.
 */
export function trainProject(bundle: Bundle, opts: TrainOptions = {}): TrainReport {
  const start = performance.now();
  const vocabSize = opts.vocabSize ?? 8192;
  const order = opts.order ?? 4;
  const maxTokens = opts.maxTokens ?? 2_000_000;
  const embeddingDim = opts.embeddingDim ?? 64;
  const valFraction = opts.valFraction ?? 0.08;
  ensureStateDir(bundle.ws);

  step(`Scanning ${bundle.ws.root}`);
  const files = walkSources(bundle.ws.root, { extraExtensions: opts.extraExtensions });
  if (files.length === 0) {
    throw new Error(
      "No source files found. Check that you are in a code project, or pass --ext to add file types.",
    );
  }
  const totalBytes = files.reduce((a, f) => a + f.size, 0);
  ok(`Found ${files.length} source files (${humanBytes(totalBytes)})`);

  // --- read once, reuse everywhere -------------------------------------
  const texts: string[] = [];
  const kept: SourceFile[] = [];
  const style = new StyleAnalyzer();
  for (let i = 0; i < files.length; i++) {
    if ((i & 31) === 0) progress(`reading ${i + 1}/${files.length}`);
    let text: string;
    try {
      text = readSource(files[i]);
    } catch {
      continue;
    }
    texts.push(text);
    kept.push(files[i]);
    style.add(text, path.extname(files[i].rel).toLowerCase());
  }
  progressDone();
  bundle.setStyle(style.finish());

  const trainIdx: number[] = [];
  const valIdx: number[] = [];
  for (let i = 0; i < kept.length; i++) {
    (isValidationFile(kept[i].rel, valFraction) ? valIdx : trainIdx).push(i);
  }
  if (trainIdx.length === 0) {
    // Tiny repo: a held-out split would leave nothing to learn from.
    trainIdx.push(...valIdx.splice(0, valIdx.length));
  }

  // --- tokenizer --------------------------------------------------------
  step(`Learning vocabulary (target ${vocabSize} pieces)`);
  const tokenizer = Tokenizer.train(
    trainIdx.map((i) => texts[i]),
    vocabSize,
  );
  tokenizer.save(bundle.path("tokenizer"));
  ok(`Vocabulary: ${tokenizer.vocabSize} tokens (${tokenizer.pieces.length} learned + 259 reserved)`);

  // --- encode corpus ----------------------------------------------------
  step("Encoding corpus");
  const encoded: number[][] = [];
  let corpusTokens = 0;
  const fileTokens = new Map<string, number>();
  for (let n = 0; n < trainIdx.length; n++) {
    const i = trainIdx[n];
    if ((n & 31) === 0) progress(`encoding ${n + 1}/${trainIdx.length}`);
    const ids = tokenizer.encode(texts[i]);
    fileTokens.set(kept[i].rel, ids.length);
    if (corpusTokens + ids.length > maxTokens) {
      const room = maxTokens - corpusTokens;
      if (room > 0) {
        encoded.push(ids.slice(0, room));
        corpusTokens += room;
      }
      warn(`Corpus token cap reached (${maxTokens}); remaining files skipped for LM training`);
      break;
    }
    encoded.push(ids);
    corpusTokens += ids.length;
  }
  progressDone();
  ok(`Corpus: ${corpusTokens.toLocaleString()} tokens`);

  // --- n-gram -----------------------------------------------------------
  step(`Training ${order}-gram model (Kneser-Ney)`);
  const ngram = new NgramModel({ order });
  ngram.vocabSize = tokenizer.vocabSize;
  for (let n = 0; n < encoded.length; n++) {
    if ((n & 31) === 0) progress(`counting ${n + 1}/${encoded.length}`);
    ngram.observe(encoded[n]);
  }
  progressDone();
  progress("deriving continuation counts");
  ngram.finalize();
  progressDone();
  const ngramStats = ngram.stats();
  ngram.save(bundle.path("ngram"));
  ok(
    `n-gram: ${ngramStats.ngrams.toLocaleString()} n-grams across ${ngramStats.contexts.toLocaleString()} contexts`,
  );

  // --- retrieval index --------------------------------------------------
  let chunkCount = 0;
  let embeddingDimUsed = 0;
  if (!opts.skipIndex) {
    step("Building retrieval index");
    const indexFiles: IndexedFile[] = [];
    const chunks: IndexedChunk[] = [];
    for (let i = 0; i < kept.length; i++) {
      if ((i & 31) === 0) progress(`chunking ${i + 1}/${kept.length}`);
      const fileIdx = indexFiles.length;
      indexFiles.push({
        path: kept[i].rel,
        size: kept[i].size,
        mtimeMs: kept[i].mtimeMs,
        tokens: fileTokens.get(kept[i].rel) ?? 0,
      });
      for (const c of chunkFile(texts[i])) {
        chunks.push({
          id: chunks.length,
          file: fileIdx,
          startLine: c.startLine,
          endLine: c.endLine,
          symbol: c.symbol,
          symbols: c.symbols,
          text: c.text,
        });
      }
    }
    progressDone();
    const bm25 = IndexStore.buildBm25(chunks);
    const store = new IndexStore(bundle.ws.root, indexFiles, chunks, bm25);
    chunkCount = chunks.length;

    // --- embeddings ----------------------------------------------------
    if (embeddingDim > 0 && corpusTokens > 500) {
      step(`Training ${embeddingDim}-dim embeddings (skip-gram, negative sampling)`);
      const embeddings = new EmbeddingModel(tokenizer.vocabSize, embeddingDim);
      embeddings.countTokens(encoded);
      embeddings.train(
        encoded,
        { dim: embeddingDim, epochs: opts.embeddingEpochs ?? 4 },
        (done, total) => progress(`sgns ${Math.round((done / total) * 100)}%`),
      );
      progressDone();
      embeddings.save(bundle.path("embeddings"));

      progress("embedding chunks");
      const vectors = new Float32Array(chunks.length * embeddingDim);
      for (let i = 0; i < chunks.length; i++) {
        const vec = normalize(embeddings.sifEmbed(tokenizer.encode(chunks[i].text)));
        vectors.set(vec, i * embeddingDim);
      }
      progressDone();
      store.attachVectors(embeddingDim, vectors);
      embeddingDimUsed = embeddingDim;
      ok(`Embeddings: ${embeddingDim}-dim vectors for ${chunks.length} chunks`);
    }

    store.save(bundle.path("index"), bundle.path("vectors"));
    ok(`Index: ${chunkCount.toLocaleString()} chunks from ${indexFiles.length} files`);
  }

  const seconds = (performance.now() - start) / 1000;
  const manifest: Manifest = {
    version: MANIFEST_VERSION,
    trainedAt: new Date().toISOString(),
    vocabSize: tokenizer.vocabSize,
    ngramOrder: order,
    corpusTokens,
    corpusFiles: trainIdx.length,
    embeddingDim: embeddingDimUsed,
    ...(bundle.manifest?.neural ? { neural: bundle.manifest.neural } : {}),
  };
  bundle.writeManifest(manifest);

  ok(`Trained in ${fmtDuration(seconds * 1000)}`);
  return {
    files: kept.length,
    bytes: totalBytes,
    corpusTokens,
    vocabSize: tokenizer.vocabSize,
    ngram: { order, contexts: ngramStats.contexts, ngrams: ngramStats.ngrams },
    chunks: chunkCount,
    embeddingDim: embeddingDimUsed,
    valFiles: valIdx.length,
    seconds,
  };
}

/**
 * Re-read the corpus with an existing tokenizer. Used by neural training and
 * evaluation, which both need the same train/val split as `trainProject`.
 */
export function loadCorpus(
  bundle: Bundle,
  valFraction = 0.08,
  extraExtensions?: string[],
): { train: number[][]; val: number[][]; trainPaths: string[]; valPaths: string[] } {
  const tokenizer = bundle.tokenizer;
  const files = walkSources(bundle.ws.root, { extraExtensions });
  const train: number[][] = [];
  const val: number[][] = [];
  const trainPaths: string[] = [];
  const valPaths: string[] = [];
  for (const file of files) {
    let text: string;
    try {
      text = readSource(file);
    } catch {
      continue;
    }
    const ids = tokenizer.encode(text);
    if (ids.length < 8) continue;
    if (isValidationFile(file.rel, valFraction)) {
      val.push(ids);
      valPaths.push(file.rel);
    } else {
      train.push(ids);
      trainPaths.push(file.rel);
    }
  }
  if (train.length === 0 && val.length > 0) {
    train.push(...val.splice(0, val.length));
    trainPaths.push(...valPaths.splice(0, valPaths.length));
  }
  return { train, val, trainPaths, valPaths };
}
