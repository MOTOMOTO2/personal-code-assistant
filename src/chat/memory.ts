import fs from "node:fs";
import crypto from "node:crypto";
import { Bundle } from "../core/bundle.js";
import { EmbeddingModel, normalize } from "../ml/embeddings.js";
import { searchTerms } from "../index/store.js";
import { Tokenizer } from "../tokenizer/tokenizer.js";

/**
 * Conversation memory: every answer the assistant gives is written here, and
 * every answer written here becomes training data on the next `pca learn`.
 *
 * This is the mechanism behind "it gets better the more I use it". A frontier
 * model answered a question about *this* codebase; that answer is worth keeping
 * for three separate reasons:
 *
 *   1. Recall - the same question asked again is answered locally, instantly
 *      and offline, from the stored answer rather than a new API call.
 *   2. Retrieval - stored answers are indexed alongside your source, so future
 *      questions are grounded in prose that already explains your system.
 *   3. Training - the n-gram, embeddings and transformer all train on this text,
 *      so the local models learn how your codebase is *described*, not just how
 *      it is written.
 *
 * Append-only JSONL, because the write happens at the end of a streamed answer
 * and must not be able to corrupt earlier turns.
 */

export type LessonSource = "claude" | "local";

export interface Lesson {
  id: string;
  at: string;
  question: string;
  answer: string;
  source: LessonSource;
  model?: string;
  /** `path:line` citations extracted from the answer. */
  cites: string[];
  /** Files retrieved as grounding when the answer was produced. */
  grounding: string[];
  tokensIn?: number;
  tokensOut?: number;
  costUsd?: number;
}

export interface Recall {
  lesson: Lesson;
  /** 0..1, blended lexical and embedding similarity. */
  score: number;
  lexical: number;
  semantic: number;
}

const CITE_RE = /\b([\w./\\-]+\.[A-Za-z][\w]{0,7}):(\d+)/g;

/**
 * Search terms for memory, with plurals folded onto their singular.
 *
 * Questions and answers are prose, where "where are the handlers" and "the
 * handler" are the same query. Code search terms are identifiers, where they
 * often are not - `users` and `user` can be different tables. So this stemming
 * lives here rather than in the code index.
 */
export function memoryTerms(text: string): string[] {
  const out: string[] = [];
  for (const term of searchTerms(text)) {
    out.push(term);
    if (term.length > 3 && term.endsWith("s") && !term.endsWith("ss")) {
      out.push(term.slice(0, -1));
    }
  }
  return out;
}

/** Pull `path:line` references out of an answer so they can be listed and re-checked. */
export function extractCites(text: string): string[] {
  const out = new Set<string>();
  for (const m of text.matchAll(CITE_RE)) out.add(`${m[1]}:${m[2]}`);
  return [...out];
}

export function newLessonId(): string {
  return crypto.randomBytes(6).toString("hex");
}

/**
 * Lexical + semantic search over past answers.
 *
 * The lexical half is a plain IDF-weighted overlap rather than full BM25: the
 * lesson count is small enough (hundreds, not millions) that the extra
 * machinery buys nothing, and a bounded 0..1 score is what the router needs as
 * a feature.
 */
export class LessonStore {
  readonly lessons: Lesson[];
  private df = new Map<string, number>();
  private terms: Array<Map<string, number>> = [];
  private vectors: Float32Array[] | null = null;
  private dim = 0;

  constructor(lessons: Lesson[]) {
    this.lessons = lessons;
    for (const lesson of lessons) {
      const tf = new Map<string, number>();
      for (const t of memoryTerms(`${lesson.question}\n${lesson.answer}`)) {
        tf.set(t, (tf.get(t) ?? 0) + 1);
      }
      this.terms.push(tf);
      for (const t of tf.keys()) this.df.set(t, (this.df.get(t) ?? 0) + 1);
    }
  }

  get size(): number {
    return this.lessons.length;
  }

  /** Embed every lesson once, so repeated recalls in a session are cheap. */
  private ensureVectors(tokenizer: Tokenizer, embeddings: EmbeddingModel): void {
    if (this.vectors) return;
    this.dim = embeddings.dim;
    this.vectors = this.lessons.map((lesson) =>
      normalize(embeddings.sifEmbed(tokenizer.encode(`${lesson.question}\n${lesson.answer}`))),
    );
  }

  recall(
    query: string,
    opts: { limit?: number; tokenizer?: Tokenizer; embeddings?: EmbeddingModel | null } = {},
  ): Recall[] {
    if (this.lessons.length === 0) return [];
    const limit = opts.limit ?? 3;

    const queryTerms = new Set(memoryTerms(query));
    const n = this.lessons.length;
    let queryVector: Float32Array | null = null;
    if (opts.tokenizer && opts.embeddings) {
      this.ensureVectors(opts.tokenizer, opts.embeddings);
      queryVector = normalize(opts.embeddings.sifEmbed(opts.tokenizer.encode(query)));
    }

    const scored: Recall[] = this.lessons.map((lesson, i) => {
      // IDF-weighted overlap, normalised by the query's own weight so the
      // result is a comparable 0..1 fraction rather than an unbounded sum.
      let matched = 0;
      let total = 0;
      for (const term of queryTerms) {
        const idf = Math.log(1 + n / (1 + (this.df.get(term) ?? 0)));
        total += idf;
        if (this.terms[i].has(term)) matched += idf;
      }
      const lexical = total > 0 ? matched / total : 0;

      let semantic = 0;
      if (queryVector && this.vectors) {
        const v = this.vectors[i];
        let dot = 0;
        for (let d = 0; d < this.dim; d++) dot += queryVector[d] * v[d];
        // Cosine over SIF embeddings is rarely negative and rarely near 1;
        // clamp so the blend stays inside 0..1.
        semantic = Math.max(0, Math.min(1, dot));
      }
      const score = queryVector ? 0.6 * lexical + 0.4 * semantic : lexical;
      return { lesson, score, lexical, semantic };
    });

    return scored
      .filter((r) => r.score > 0.02)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  /**
   * Lessons rendered as training documents. Markdown, because that is what the
   * answers already are, and the tokenizer is lossless over any input.
   */
  documents(): Array<{ path: string; text: string }> {
    return this.lessons.map((lesson) => ({
      path: `${LESSON_PATH_PREFIX}${lesson.id}.md`,
      text: renderLesson(lesson),
    }));
  }
}

/** Synthetic path prefix for indexed lessons, so hits are visibly not source files. */
export const LESSON_PATH_PREFIX = "~memory/";

export function renderLesson(lesson: Lesson): string {
  const header = lesson.source === "claude" ? `Answered by ${lesson.model ?? "claude"}` : "Answered locally";
  return `# ${lesson.question}\n\n<!-- ${header} on ${lesson.at} -->\n\n${lesson.answer}\n`;
}

export function loadLessons(bundle: Bundle): LessonStore {
  const file = bundle.path("lessons");
  if (!fs.existsSync(file)) return new LessonStore([]);
  const lessons: Lesson[] = [];
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    if (line.trim().length === 0) continue;
    try {
      const parsed = JSON.parse(line) as Lesson;
      if (typeof parsed.question === "string" && typeof parsed.answer === "string") {
        lessons.push(parsed);
      }
    } catch {
      // A partially written line from a killed process: skip it rather than
      // losing every lesson recorded before it.
    }
  }
  return new LessonStore(lessons);
}

export function appendLesson(bundle: Bundle, lesson: Lesson): void {
  fs.mkdirSync(bundle.ws.stateDir, { recursive: true });
  fs.appendFileSync(bundle.path("lessons"), `${JSON.stringify(lesson)}\n`, "utf8");
}
