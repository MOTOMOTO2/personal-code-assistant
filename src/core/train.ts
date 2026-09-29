import crypto from "node:crypto";
import { Bundle, Manifest, MANIFEST_VERSION } from "./bundle.js";
import { StyleAnalyzer } from "./style.js";
import { chunkFile } from "../index/chunker.js";
import { IndexStore, IndexedChunk, IndexedFile } from "../index/store.js";
import { SourceFile, readSource, walkSources } from "../index/walker.js";
import { EmbeddingModel, normalize } from "../ml/embeddings.js";
import { NgramModel } from "../ml/ngram.js";
import { Tokenizer } from "../tokenizer/tokenizer.js";
import { loadLessons } from "../chat/memory.js";
import { CorpusSource, SourceStats, loadSources, readSources } from "./sources.js";
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
  /** Include remembered chat answers in the corpus and index. Default true. */
  includeLessons?: boolean;
  /**
   * Train on the directories registered with `pca corpus add` as well as this
   * project. Default true; `--no-sources` turns it off for one run without
   * unregistering anything.
   */
  includeSources?: boolean;
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
  /** Remembered answers folded into the corpus. */
  lessons: number;
  /** Per-source contribution from `pca corpus`. */
  sources: SourceStats[];
  /** Tokens of the corpus that came from external sources. */
  sourceTokens: number;
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
  /** Parallel to `kept`: whether the document belongs in the retrieval index. */
  const indexable: boolean[] = [];
  /** Parallel to `kept`: source name for external documents, null for your own. */
  const origin: Array<string | null> = [];
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
    indexable.push(true);
    origin.push(null);
    style.add(text, path.extname(files[i].rel).toLowerCase());
  }
  progressDone();
  // The style profile is measured from your code alone. It is what the
  // completer imitates - quote style, indent width, semicolons - so letting a
  // borrowed repository vote on it would make the assistant write like someone
  // else in your own files.
  bundle.setStyle(style.finish());

  const projectTrainIdx: number[] = [];
  const valIdx: number[] = [];
  for (let i = 0; i < kept.length; i++) {
    (isValidationFile(kept[i].rel, valFraction) ? valIdx : projectTrainIdx).push(i);
  }
  if (projectTrainIdx.length === 0) {
    // Tiny repo: a held-out split would leave nothing to learn from.
    projectTrainIdx.push(...valIdx.splice(0, valIdx.length));
  }

  // --- memory -----------------------------------------------------------
  // Answers already given about this codebase are training data. They go in
  // after the style profile is measured (prose must not skew the code style)
  // and always land in the training split - holding out an answer to evaluate
  // perplexity on it would measure nothing useful.
  const lessonIdx: number[] = [];
  if (opts.includeLessons !== false) {
    const docs = loadLessons(bundle).documents();
    for (const doc of docs) {
      lessonIdx.push(texts.length);
      texts.push(doc.text);
      kept.push({
        abs: bundle.path("lessons"),
        rel: doc.path,
        size: Buffer.byteLength(doc.text, "utf8"),
        mtimeMs: Date.now(),
      });
      indexable.push(true);
      origin.push(null);
    }
    if (lessonIdx.length > 0) ok(`Memory: ${lessonIdx.length} remembered answers folded in`);
  }

  // --- external sources -------------------------------------------------
  // Directories registered with `pca corpus add`: other repositories of yours,
  // or libraries you write against. They never join the validation split,
  // because the question this project measures is how well it predicts *your*
  // code, and scoring that on someone else's would answer a different one.
  const sourceIdx: number[] = [];
  let sourceStats: SourceStats[] = [];
  let registered: CorpusSource[] = [];
  if (opts.includeSources !== false) {
    registered = loadSources(bundle.ws);
    if (registered.length > 0) {
      step(`Reading ${registered.length} extra corpus source${registered.length === 1 ? "" : "s"}`);
      const { docs, stats } = readSources(registered, (msg) => progress(msg));
      progressDone();
      sourceStats = stats;
      for (const doc of docs) {
        sourceIdx.push(texts.length);
        texts.push(doc.text);
        kept.push({ abs: doc.abs, rel: doc.rel, size: doc.size, mtimeMs: doc.mtimeMs });
        indexable.push(doc.index);
        origin.push(doc.source);
      }
      for (const s of stats) {
        if (s.missing) continue;
        ok(
          `Source @${s.name}: ${s.files} files (${humanBytes(s.bytes)})` +
            (s.truncated ? " - token budget reached, rest skipped" : ""),
        );
      }
    }
  }

  // Corpus priority, which decides what survives the token cap: remembered
  // answers first (a handful of documents, and the densest per token), then
  // your code, then borrowed code. Anything that has to be dropped should be
  // the least specific to you, and this ordering guarantees that.
  const trainIdx = [...lessonIdx, ...projectTrainIdx, ...sourceIdx];
  const lessonCount = lessonIdx.length;

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
  let sourceTokens = 0;
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
        if (origin[i] !== null) sourceTokens += room;
      }
      warn(
        `Corpus token cap reached (${maxTokens.toLocaleString()}); ` +
          `${trainIdx.length - n} documents skipped for LM training. ` +
          "Raise it with --max-tokens, or lower a source budget in .pca/sources.json.",
      );
      break;
    }
    encoded.push(ids);
    corpusTokens += ids.length;
    if (origin[i] !== null) sourceTokens += ids.length;
  }
  progressDone();
  ok(
    `Corpus: ${corpusTokens.toLocaleString()} tokens` +
      (sourceTokens > 0
        ? ` (${(corpusTokens - sourceTokens).toLocaleString()} yours, ${sourceTokens.toLocaleString()} from sources)`
        : ""),
  );

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
      if (!indexable[i]) continue; // source registered with --no-index
      const fileIdx = indexFiles.length;
      indexFiles.push({
        path: kept[i].rel,
        size: kept[i].size,
        mtimeMs: kept[i].mtimeMs,
        tokens: fileTokens.get(kept[i].rel) ?? 0,
        ...(origin[i] !== null ? { external: origin[i] as string } : {}),
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
    trainSeconds: seconds,
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
    lessons: lessonCount,
    sources: sourceStats,
    sourceTokens,
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
  includeLessons = true,
  includeSources = true,
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
  if (includeLessons) {
    for (const doc of loadLessons(bundle).documents()) {
      const ids = tokenizer.encode(doc.text);
      if (ids.length < 8) continue;
      train.push(ids);
      trainPaths.push(doc.path);
    }
  }
  if (includeSources) {
    // Always training, never validation - see the note in `trainProject`.
    const { docs } = readSources(loadSources(bundle.ws));
    for (const doc of docs) {
      const ids = tokenizer.encode(doc.text);
      if (ids.length < 8) continue;
      train.push(ids);
      trainPaths.push(doc.rel);
    }
  }
  if (train.length === 0 && val.length > 0) {
    train.push(...val.splice(0, val.length));
    trainPaths.push(...valPaths.splice(0, valPaths.length));
  }
  return { train, val, trainPaths, valPaths };
}
