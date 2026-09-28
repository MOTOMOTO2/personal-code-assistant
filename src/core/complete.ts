import { Bundle } from "./bundle.js";
import { NUM_FEATURES, emptyFeatures, featureIndex } from "../ml/reranker.js";
import { splitPartial } from "../tokenizer/tokenizer.js";
import { normalize } from "../ml/embeddings.js";
import { searchTerms } from "../index/store.js";

/**
 * The completion engine, and the reason the assistant feels fast.
 *
 * Three stages, cheapest first:
 *   1. PROPOSE - beam search over the Kneser-Ney n-gram model. Microseconds.
 *      Local identifiers from the open file are injected as extra candidates so
 *      a symbol defined thirty seconds ago is completable immediately.
 *   2. RESCORE (optional) - one transformer forward pass supplies a
 *      longer-range score for each candidate's first token. Tens of
 *      milliseconds, so it is opt-in.
 *   3. RANK - the linear reranker combines both model scores with locality,
 *      retrieval and embedding features, using weights learned from which
 *      suggestions you actually accepted.
 *
 * Only stage 1 and 3 run on the keystroke path.
 */

export interface CompletionRequest {
  /** Text before the cursor. */
  prefix: string;
  /** Text after the cursor, used for context and bracket balancing. */
  suffix?: string;
  /** Repo-relative path of the file being edited, for retrieval context. */
  filePath?: string;
  /** Maximum tokens to generate per suggestion. */
  maxTokens?: number;
  /** How many suggestions to return. */
  count?: number;
  /** Beam width for the n-gram search. */
  beamWidth?: number;
  /**
   * Stop extending a suggestion once the longest matching n-gram context falls
   * below this order. Higher means shorter, more confident suggestions.
   */
  minEvidenceOrder?: number;
  /** Add transformer rescoring. Costs ~10-100ms depending on model size. */
  neural?: boolean;
  /** Use the code index to add retrieval features. */
  retrieval?: boolean;
  /** Stop each suggestion at the end of the current line. */
  singleLine?: boolean;
}

export interface Suggestion {
  /** Text to insert at the cursor. */
  text: string;
  /** Reranker score; higher is better. Not a probability. */
  score: number;
  /** Mean n-gram log-probability per token. */
  ngramLogProb: number;
  /** Mean neural log-probability, or null when the model did not run. */
  neuralLogProb: number | null;
  /** Feature vector, echoed back on feedback so the reranker can learn. */
  features: number[];
  tokens: number[];
}

export interface CompletionResult {
  suggestions: Suggestion[];
  /** Milliseconds spent, split by stage. */
  timing: { propose: number; neural: number; rank: number; total: number };
  /** Identifier fragment the cursor was inside, if any. */
  partial: string;
  usedNeural: boolean;
}

/** Keywords across the languages this tool is likely to see. */
const KEYWORDS = new Set([
  "if", "else", "for", "while", "do", "switch", "case", "default", "break",
  "continue", "return", "function", "class", "interface", "type", "enum",
  "const", "let", "var", "new", "delete", "typeof", "instanceof", "in", "of",
  "try", "catch", "finally", "throw", "async", "await", "yield", "import",
  "export", "from", "as", "extends", "implements", "public", "private",
  "protected", "static", "readonly", "abstract", "def", "lambda", "pass",
  "raise", "except", "elif", "None", "True", "False", "self", "nil", "end",
  "func", "go", "defer", "struct", "impl", "fn", "let", "mut", "pub", "use",
  "match", "trait", "where", "unsafe", "null", "true", "false", "undefined",
  "void", "this", "super", "with", "and", "or", "not",
]);

const IDENT_RE = /^[A-Za-z_$][\w$]*$/;
/** Local context window used to derive locality features, in characters. */
const LOCAL_WINDOW = 4000;
/** Tokens of history fed to the n-gram. More than this changes nothing. */
const HISTORY_TOKENS = 32;

interface Beam {
  tokens: number[];
  logProb: number;
  text: string;
  /** Rank of the first token in the initial proposal list. */
  firstRank: number;
  done: boolean;
}

export class CompletionEngine {
  private readonly bundle: Bundle;
  /** Cached retrieval terms, keyed by file path, so repeat keystrokes are free. */
  private retrievalCache = new Map<string, { terms: Set<string>; at: number }>();
  /**
   * Cached locality term counts. Tokenising the 4 KB window around the cursor is
   * the single most expensive per-keystroke step, and consecutive keystrokes see
   * a near-identical window, so it is re-derived only when the buffer moves
   * meaningfully.
   */
  private localCache: { key: string; length: number; counts: Map<string, number> } | null = null;

  constructor(bundle: Bundle) {
    this.bundle = bundle;
  }

  /**
   * Page in the models the keystroke path needs. Loading the n-gram tables costs
   * tens of milliseconds, which would otherwise be charged to the first
   * completion and make the timings meaningless. The daemon calls this at
   * startup; the CLI calls it before starting its timer.
   */
  warm(opts: { neural?: boolean; retrieval?: boolean } = {}): void {
    void this.bundle.tokenizer;
    void this.bundle.ngram;
    void this.bundle.reranker;
    void this.bundle.embeddings;
    if (opts.neural) void this.bundle.neural;
    if (opts.retrieval) void this.bundle.index;
  }

  complete(req: CompletionRequest): CompletionResult {
    const t0 = performance.now();
    // Defaults chosen by sweeping held-out line completion (see bench/sweep.mjs):
    // 8 tokens sits at the knee of the utility curve, and requiring a
    // full-order context match buys ~8 points of "suggestion fully correct" for
    // almost no loss in characters saved.
    const maxTokens = req.maxTokens ?? 8;
    const count = req.count ?? 3;
    const beamWidth = req.beamWidth ?? 5;
    const singleLine = req.singleLine !== false;

    const tokenizer = this.bundle.tokenizer;
    const ngram = this.bundle.ngram;
    const { stable, partial } = splitPartial(req.prefix);

    // Only the tail of the file matters to an n-gram; encoding megabytes would
    // dominate the latency budget for no gain.
    const tail = stable.length > LOCAL_WINDOW ? stable.slice(stable.length - LOCAL_WINDOW) : stable;
    const encoded = tokenizer.encode(tail);
    const history = encoded.slice(Math.max(0, encoded.length - HISTORY_TOKENS));

    const localText = `${tail}\n${(req.suffix ?? "").slice(0, LOCAL_WINDOW)}`;
    const localCounts = this.localTermCounts(req.filePath ?? "<buffer>", localText);
    const retrievalTerms = req.retrieval ? this.retrievalTerms(req, localText) : null;

    // --- stage 1: propose -------------------------------------------------
    const firstCandidates = this.firstTokenCandidates(history, partial, localText, beamWidth * 4);
    const beams = this.beamSearch(history, firstCandidates, {
      beamWidth,
      maxTokens,
      singleLine,
      minEvidenceOrder: req.minEvidenceOrder ?? ngram.order,
    });
    const tPropose = performance.now();

    // --- stage 2: neural rescoring ---------------------------------------
    let neuralLogProbs: Float32Array | null = null;
    const neural = req.neural ? this.bundle.neural : null;
    if (neural && history.length > 0) {
      neuralLogProbs = neural.nextLogProbs(history);
    }
    const tNeural = performance.now();

    // --- stage 3: rank ----------------------------------------------------
    const reranker = this.bundle.reranker;
    const embeddings = this.bundle.embeddings;
    const contextVector = embeddings ? embeddings.sifEmbed(history) : null;
    const ngramTop3 = new Set(firstCandidates.slice(0, 3).map((c) => c.id));
    const neuralTop3 = neuralLogProbs ? topKIndices(neuralLogProbs, 3) : new Set<number>();

    const scored: Suggestion[] = [];
    for (const beam of beams) {
      if (beam.tokens.length === 0) continue;
      const insertion = beam.text.slice(partial.length);
      if (insertion.trim().length === 0) continue;
      const firstToken = beam.tokens[0];
      const firstPiece = tokenizer.pieceOf(firstToken);
      const meanLogProb = beam.logProb / beam.tokens.length;
      const neuralLp = neuralLogProbs ? neuralLogProbs[firstToken] : null;

      const f = emptyFeatures();
      f[featureIndex("ngramLogProb")] = clamp(meanLogProb / 8 + 1, -1, 1);
      f[featureIndex("ngramLogProbSum")] = clamp(beam.logProb / 24 + 1, -1, 1);
      f[featureIndex("neuralLogProb")] = neuralLp === null ? 0 : clamp(neuralLp / 8 + 1, -1, 1);
      f[featureIndex("modelsAgree")] =
        ngramTop3.has(firstToken) && neuralTop3.has(firstToken) ? 1 : 0;
      f[featureIndex("ngramRank")] = 1 / (1 + beam.firstRank);
      f[featureIndex("charLength")] = clamp(insertion.length / 24, 0, 1);
      f[featureIndex("isIdentifier")] = IDENT_RE.test(firstPiece) && !KEYWORDS.has(firstPiece) ? 1 : 0;
      f[featureIndex("isKeyword")] = KEYWORDS.has(firstPiece) ? 1 : 0;
      f[featureIndex("localFileHits")] = clamp(localHits(beam.text, localCounts) / 3, 0, 1);
      f[featureIndex("retrievalHits")] = retrievalTerms
        ? clamp(retrievalHits(beam.text, retrievalTerms) / 3, 0, 1)
        : 0;
      f[featureIndex("embedSimilarity")] =
        contextVector && embeddings ? embedSim(embeddings, contextVector, beam.tokens) : 0;
      f[featureIndex("continuesPartial")] = partial.length > 0 ? 1 : 0;
      f[featureIndex("openBracketBalance")] = closesOpenBracket(req.prefix, insertion) ? 1 : 0;

      scored.push({
        text: insertion,
        score: reranker.score(f),
        ngramLogProb: meanLogProb,
        neuralLogProb: neuralLp,
        features: [...f],
        tokens: beam.tokens,
      });
    }

    scored.sort((a, b) => b.score - a.score);
    // Drop duplicates that differ only by trailing whitespace.
    const seen = new Set<string>();
    const unique = scored.filter((s) => {
      const key = s.text.trimEnd();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    const t1 = performance.now();
    return {
      suggestions: unique.slice(0, count),
      timing: {
        propose: round2(tPropose - t0),
        neural: round2(tNeural - tPropose),
        rank: round2(t1 - tNeural),
        total: round2(t1 - t0),
      },
      partial,
      usedNeural: neuralLogProbs !== null,
    };
  }

  /**
   * First-token candidates: the n-gram's proposals, plus identifiers harvested
   * from the open file. The latter is what makes a symbol you defined moments
   * ago completable before any retraining.
   */
  private firstTokenCandidates(
    history: number[],
    partial: string,
    localText: string,
    want: number,
  ): Array<{ id: number; logProb: number }> {
    const tokenizer = this.bundle.tokenizer;
    const ngram = this.bundle.ngram;
    const ctx = ngram.contextFor(history);
    // Ask for a wide pool: under a partial-word constraint most of the top
    // proposals get filtered out.
    const pool = ngram.predictIn(ctx, partial.length > 0 ? 400 : Math.max(want, 24), 600);
    const out: Array<{ id: number; logProb: number }> = [];
    const seen = new Set<number>();
    for (const cand of pool) {
      const piece = tokenizer.pieceOf(cand.id);
      if (piece.length === 0) continue;
      if (partial.length > 0 && !piece.startsWith(partial)) continue;
      if (tokenizer.isByte(cand.id)) continue; // byte tokens are not useful alone
      seen.add(cand.id);
      out.push(cand);
      if (out.length >= want) break;
    }
    if (partial.length > 0) {
      // Local symbols matching the fragment, even if the n-gram never saw them
      // in this context.
      for (const word of localIdentifiers(localText, partial)) {
        const id = tokenizer.idOf(word);
        if (id === undefined || seen.has(id)) continue;
        seen.add(id);
        out.push({ id, logProb: ngram.logProbIn(ctx, id) });
      }
      out.sort((a, b) => b.logProb - a.logProb);
    }
    return out.slice(0, Math.max(want, 8));
  }

  private beamSearch(
    history: number[],
    first: Array<{ id: number; logProb: number }>,
    opts: { beamWidth: number; maxTokens: number; singleLine: boolean; minEvidenceOrder: number },
  ): Beam[] {
    const tokenizer = this.bundle.tokenizer;
    const ngram = this.bundle.ngram;
    let beams: Beam[] = first.slice(0, opts.beamWidth).map((cand, rank) => {
      const piece = tokenizer.pieceOf(cand.id);
      return {
        tokens: [cand.id],
        logProb: cand.logProb,
        text: piece,
        firstRank: rank,
        done: opts.singleLine && piece.includes("\n"),
      };
    });
    const finished: Beam[] = [];

    for (let step = 1; step < opts.maxTokens; step++) {
      const next: Beam[] = [];
      for (const beam of beams) {
        if (beam.done) {
          finished.push(beam);
          continue;
        }
        const ctx = ngram.contextFor([...history, ...beam.tokens]);
        // Once the longest matching context is this short the model is just
        // backing off to generic code, and continuing produces the rambling,
        // plausible-looking nonsense that makes a completion feel untrustworthy.
        if (ctx.matchedOrder < opts.minEvidenceOrder) {
          finished.push({ ...beam, done: true });
          continue;
        }
        const preds = ngram.predictIn(ctx, 6, 64);
        if (preds.length === 0) {
          finished.push({ ...beam, done: true });
          continue;
        }
        for (const pred of preds) {
          const piece = tokenizer.pieceOf(pred.id);
          if (piece.length === 0) continue;
          const isBreak = piece.includes("\n");
          if (opts.singleLine && isBreak) {
            // Ending the line is a legitimate stopping point, not a candidate.
            finished.push({ ...beam, done: true });
            continue;
          }
          next.push({
            tokens: [...beam.tokens, pred.id],
            logProb: beam.logProb + pred.logProb,
            text: beam.text + piece,
            firstRank: beam.firstRank,
            done: false,
          });
        }
      }
      if (next.length === 0) break;
      // Length-normalised so short beams are not automatically preferred.
      next.sort((a, b) => b.logProb / b.tokens.length - a.logProb / a.tokens.length);
      beams = next.slice(0, opts.beamWidth);
      // Every prefix is a valid suggestion, so keep the partial beams too.
      for (const beam of beams) finished.push({ ...beam });
    }
    for (const beam of beams) if (!finished.includes(beam)) finished.push(beam);
    return dedupeBeams([...finished, ...beams]);
  }

  /**
   * Term frequencies for the window around the cursor. Reused while the cursor
   * stays in the same file and the window has not shifted by more than a few
   * tokens' worth of text - the counts are a locality prior, so a slightly stale
   * window changes nothing about the ranking.
   */
  private localTermCounts(key: string, localText: string): Map<string, number> {
    const cached = this.localCache;
    if (cached && cached.key === key && Math.abs(cached.length - localText.length) < 120) {
      return cached.counts;
    }
    const counts = countTerms(localText);
    this.localCache = { key, length: localText.length, counts };
    return counts;
  }

  /** Terms from semantically related chunks, cached per file. */
  private retrievalTerms(req: CompletionRequest, localText: string): Set<string> | null {
    const index = this.bundle.index;
    if (!index) return null;
    const key = req.filePath ?? "<buffer>";
    const cached = this.retrievalCache.get(key);
    if (cached && Date.now() - cached.at < 30_000) return cached.terms;

    const embeddings = this.bundle.embeddings;
    const query = localText.slice(Math.max(0, localText.length - 1200));
    const queryVector = embeddings
      ? normalize(embeddings.sifEmbed(this.bundle.tokenizer.encode(query)))
      : null;
    const hits = index.search(query, { limit: 5, queryVector });
    const terms = new Set<string>();
    for (const hit of hits) {
      if (hit.file === req.filePath) continue; // the open file is already local
      for (const term of searchTerms(hit.chunk.text)) terms.add(term);
    }
    this.retrievalCache.set(key, { terms, at: Date.now() });
    return terms;
  }
}

// --- feature helpers -------------------------------------------------------

function countTerms(text: string): Map<string, number> {
  const counts = new Map<string, number>();
  for (const term of searchTerms(text)) counts.set(term, (counts.get(term) ?? 0) + 1);
  return counts;
}

function localHits(text: string, counts: Map<string, number>): number {
  let hits = 0;
  for (const term of searchTerms(text)) hits += Math.min(counts.get(term) ?? 0, 3);
  return hits;
}

function retrievalHits(text: string, terms: Set<string>): number {
  let hits = 0;
  for (const term of searchTerms(text)) if (terms.has(term)) hits++;
  return hits;
}

function localIdentifiers(text: string, partial: string): string[] {
  const out = new Set<string>();
  const re = /[A-Za-z_$][\w$]*/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m[0].length > partial.length && m[0].startsWith(partial)) out.add(m[0]);
  }
  return [...out].slice(0, 32);
}

function embedSim(
  embeddings: { sifEmbed(t: ArrayLike<number>): Float32Array },
  context: Float32Array,
  tokens: number[],
): number {
  const candidate = embeddings.sifEmbed(tokens);
  let s = 0;
  for (let i = 0; i < context.length; i++) s += context[i] * candidate[i];
  return clamp(s, -1, 1);
}

/** True when the insertion closes a bracket that is currently left open. */
function closesOpenBracket(prefix: string, insertion: string): boolean {
  const pairs: Record<string, string> = { "(": ")", "[": "]", "{": "}" };
  const stack: string[] = [];
  for (const ch of prefix) {
    if (pairs[ch]) stack.push(pairs[ch]);
    else if (ch === ")" || ch === "]" || ch === "}") {
      if (stack[stack.length - 1] === ch) stack.pop();
    }
  }
  if (stack.length === 0) return false;
  return insertion.includes(stack[stack.length - 1]);
}

function dedupeBeams(beams: Beam[]): Beam[] {
  const seen = new Map<string, Beam>();
  for (const beam of beams) {
    const key = beam.text;
    const existing = seen.get(key);
    if (!existing || beam.logProb > existing.logProb) seen.set(key, beam);
  }
  return [...seen.values()];
}

function topKIndices(values: Float32Array, k: number): Set<number> {
  const idx: Array<{ i: number; v: number }> = [];
  for (let i = 0; i < values.length; i++) idx.push({ i, v: values[i] });
  idx.sort((a, b) => b.v - a.v);
  return new Set(idx.slice(0, k).map((e) => e.i));
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

export { NUM_FEATURES };
