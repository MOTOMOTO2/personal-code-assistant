import fs from "node:fs";
import { EmbeddingModel } from "../ml/embeddings.js";
import { NgramModel } from "../ml/ngram.js";
import { Reranker } from "../ml/reranker.js";
import { Transformer } from "../ml/transformer.js";
import { IndexStore } from "../index/store.js";
import { Tokenizer } from "../tokenizer/tokenizer.js";
import { DEFAULT_STYLE, StyleProfile, loadStyle, saveStyle } from "./style.js";
import { Workspace, ensureStateDir, statePath } from "../util/paths.js";
import { exists, readJsonGz, writeJsonGz } from "../util/io.js";

/** Filenames inside `.pca/`. Bumping the manifest version invalidates all of them. */
export const ARTIFACTS = {
  manifest: "manifest.json.gz",
  tokenizer: "tokenizer.json.gz",
  ngram: "ngram.json.gz",
  neural: "neural.pcam",
  embeddings: "embeddings.bin",
  index: "index.json.gz",
  vectors: "vectors.bin",
  reranker: "reranker.json.gz",
  style: "style.json.gz",
  feedback: "feedback.jsonl",
} as const;

export const MANIFEST_VERSION = 2;

export interface Manifest {
  version: number;
  trainedAt: string;
  vocabSize: number;
  ngramOrder: number;
  corpusTokens: number;
  corpusFiles: number;
  embeddingDim: number;
  /** Populated once the neural model has been trained at least once. */
  neural?: {
    trainedAt: string;
    steps: number;
    trainLoss: number;
    valLoss: number;
    valPerplexity: number;
    params: number;
    trainedSeconds: number;
  };
}

/**
 * Every trained artifact for one project, loaded lazily.
 *
 * Lazy matters: the inline-completion path must start in milliseconds, so it
 * touches the tokenizer, the n-gram model and the reranker and nothing else. The
 * transformer and the embedding matrix are only paged in when something actually
 * asks for them.
 */
export class Bundle {
  readonly ws: Workspace;
  private _tokenizer?: Tokenizer;
  private _ngram?: NgramModel;
  private _neural?: Transformer | null;
  private _embeddings?: EmbeddingModel | null;
  private _index?: IndexStore | null;
  private _reranker?: Reranker;
  private _style?: StyleProfile;
  private _manifest?: Manifest | null;

  constructor(ws: Workspace) {
    this.ws = ws;
  }

  path(name: keyof typeof ARTIFACTS): string {
    return statePath(this.ws, ARTIFACTS[name]);
  }

  get isTrained(): boolean {
    return exists(this.path("tokenizer")) && exists(this.path("ngram"));
  }

  get manifest(): Manifest | null {
    if (this._manifest === undefined) {
      this._manifest = exists(this.path("manifest"))
        ? readJsonGz<Manifest>(this.path("manifest"))
        : null;
    }
    return this._manifest;
  }

  writeManifest(manifest: Manifest): void {
    ensureStateDir(this.ws);
    writeJsonGz(this.path("manifest"), manifest);
    this._manifest = manifest;
  }

  get tokenizer(): Tokenizer {
    if (!this._tokenizer) {
      if (!exists(this.path("tokenizer"))) {
        throw new NotTrainedError("tokenizer");
      }
      this._tokenizer = Tokenizer.load(this.path("tokenizer"));
    }
    return this._tokenizer;
  }

  get ngram(): NgramModel {
    if (!this._ngram) {
      if (!exists(this.path("ngram"))) throw new NotTrainedError("n-gram model");
      this._ngram = NgramModel.load(this.path("ngram"));
    }
    return this._ngram;
  }

  /** null when the neural model has not been trained yet. */
  get neural(): Transformer | null {
    if (this._neural === undefined) {
      this._neural = exists(this.path("neural")) ? Transformer.load(this.path("neural")) : null;
    }
    return this._neural;
  }

  get embeddings(): EmbeddingModel | null {
    if (this._embeddings === undefined) {
      this._embeddings = exists(this.path("embeddings"))
        ? EmbeddingModel.load(this.path("embeddings"))
        : null;
    }
    return this._embeddings;
  }

  get index(): IndexStore | null {
    if (this._index === undefined) {
      this._index = exists(this.path("index"))
        ? IndexStore.load(this.path("index"), this.path("vectors"))
        : null;
    }
    return this._index;
  }

  get reranker(): Reranker {
    if (!this._reranker) {
      this._reranker = exists(this.path("reranker"))
        ? Reranker.load(this.path("reranker"))
        : new Reranker();
    }
    return this._reranker;
  }

  saveReranker(): void {
    ensureStateDir(this.ws);
    this.reranker.save(this.path("reranker"));
  }

  get style(): StyleProfile {
    if (!this._style) {
      this._style = exists(this.path("style")) ? loadStyle(this.path("style")) : { ...DEFAULT_STYLE };
    }
    return this._style;
  }

  setStyle(style: StyleProfile): void {
    ensureStateDir(this.ws);
    saveStyle(this.path("style"), style);
    this._style = style;
  }

  /** Append one JSON line of feedback. Used by the online reranker. */
  appendFeedback(record: unknown): void {
    ensureStateDir(this.ws);
    fs.appendFileSync(this.path("feedback"), `${JSON.stringify(record)}\n`, "utf8");
  }

  readFeedback<T>(): T[] {
    const file = this.path("feedback");
    if (!fs.existsSync(file)) return [];
    return fs
      .readFileSync(file, "utf8")
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .flatMap((l) => {
        try {
          return [JSON.parse(l) as T];
        } catch {
          return [];
        }
      });
  }
}

export class NotTrainedError extends Error {
  constructor(what: string) {
    super(
      `No trained ${what} found in .pca/. Run \`pca train\` in your project first (or \`pca train --help\`).`,
    );
    this.name = "NotTrainedError";
  }
}
