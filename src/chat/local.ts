import { Bundle } from "../core/bundle.js";
import { CompletionEngine } from "../core/complete.js";
import { SearchHit, searchTerms } from "../index/store.js";
import { normalize } from "../ml/embeddings.js";
import { NUM_ROUTER_FEATURES, RouterFeatureName, routerFeatureIndex } from "../ml/router.js";
import { retrievalQuery } from "../llm/prompts.js";
import { LessonStore, Recall } from "./memory.js";

/**
 * The local half of a chat turn.
 *
 * What a 1.6M-parameter model trained on one repository can honestly do in a
 * conversation is *not* free-form prose - an n-gram trained on TypeScript
 * generates TypeScript, not English. So the local answerer never pretends to
 * write an essay. It does four things it is genuinely good at:
 *
 *   - recall an answer already in memory, verbatim and attributed;
 *   - locate code, with `path:line` citations, using the trained retrieval index;
 *   - continue code at a cursor, using the completion engine;
 *   - name the tokens its embedding space considers related.
 *
 * Everything else routes to Claude. The result is that the local half is always
 * either useful or silent, never fluent and wrong.
 */

export type LocalAnswerKind = "memory" | "locate" | "complete" | "neighbors" | "unsure";

export interface LocalAnswer {
  text: string;
  kind: LocalAnswerKind;
  /** 0..1. Below ~0.3 the caller should offer Claude instead. */
  confidence: number;
  sources: string[];
  recalled: Recall | null;
}

export interface TurnContext {
  question: string;
  /** Retrieval hits from the code index, best first. */
  hits: SearchHit[];
  /** Matching answers already in memory, best first. */
  recalls: Recall[];
  /** Router feature vector for this turn. */
  features: Float64Array;
  /** True when the message is code to continue rather than a question. */
  looksLikeCode: boolean;
}

/** RRF scores top out near 2/(k+1); scale so the router sees a 0..1 feature. */
const RRF_SCALE = 30;
const LOCATE_RE =
  /\b(where|which file|find|locate|show me|list|what calls|who calls|defined|definition|look ?up)\b/i;
const REASON_RE =
  /\b(why|how (come|does|do|did|was|were|is|are)|explain|should|trade-?off|difference|reason|cause|understand|design|architecture|better)\b/i;
const EDIT_RE =
  /\b(write|add|create|implement|fix|refactor|rename|change|update|remove|delete|migrate|convert|generate|rewrite|optimise|optimize)\b/i;
const NEIGHBORS_RE =
  /^\s*(?:what\s+is\s+)?(?:related|similar|near|nearest|neighbors|neighbours)\s+(?:to\s+)?([A-Za-z_$][\w$]*)\s*\??\s*$/i;

export function buildTurnContext(
  bundle: Bundle,
  lessons: LessonStore,
  question: string,
  opts: { retrievalLimit?: number } = {},
): TurnContext {
  const index = bundle.isTrained ? bundle.index : null;
  const embeddings = bundle.isTrained ? bundle.embeddings : null;
  const q = retrievalQuery(question);

  let hits: SearchHit[] = [];
  let coverage = 0;
  if (index) {
    const queryVector = embeddings
      ? normalize(embeddings.sifEmbed(bundle.tokenizer.encode(q)))
      : null;
    hits = index.search(q, { limit: opts.retrievalLimit ?? 8, queryVector });
    coverage = index.coverage(new Set(searchTerms(q)));
  }

  const recalls = lessons.recall(question, {
    limit: 3,
    tokenizer: bundle.isTrained ? bundle.tokenizer : undefined,
    embeddings,
  });

  const looksLikeCode = isCodeContinuation(question);
  const features = routeFeatures(question, hits, recalls, coverage, looksLikeCode);
  return { question, hits, recalls, features, looksLikeCode };
}

export function routeFeatures(
  question: string,
  hits: SearchHit[],
  recalls: Recall[],
  coverage: number,
  looksLikeCode: boolean,
): Float64Array {
  const f = new Float64Array(NUM_ROUTER_FEATURES);
  const set = (name: RouterFeatureName, v: number) => {
    f[routerFeatureIndex(name)] = v;
  };
  set("bias", 1);
  // A stored answer answers a question; it does not do a piece of work. Asking
  // for a change is a request for new output, so memory must not pull an edit
  // request local however closely it matches something already discussed.
  const edit = EDIT_RE.test(question);
  set("memoryTop", edit ? 0 : recallStrength(recalls[0]));
  set("memorySecond", edit ? 0 : recallStrength(recalls[1]));
  set("retrievalTop", Math.min(1, (hits[0]?.score ?? 0) * RRF_SCALE));
  set("retrievalCoverage", coverage);
  set("locateShape", LOCATE_RE.test(question) ? 1 : 0);
  set("reasonShape", REASON_RE.test(question) ? 1 : 0);
  set("editShape", edit ? 1 : 0);
  set("hasCodeBlock", question.includes("```") || looksLikeCode ? 1 : 0);
  // Scaled to roughly 0..1 over the range of questions people actually type.
  set("lengthTokens", Math.min(1, question.split(/\s+/).length / 40));
  return f;
}

/** Similarity at which a stored answer is worth replaying verbatim. */
export const REPLAY_THRESHOLD = 0.45;

/**
 * Whether a recalled answer is close enough to replay as *the* answer.
 *
 * Two terms, not one: replaying a stored answer is the strongest claim the
 * local half makes, and one shared content word is a coincidence. A single term
 * only qualifies when it dominates the query outright - asking about one rare
 * identifier by name.
 */
export function isStrongRecall(recall: Recall): boolean {
  if (recall.score < REPLAY_THRESHOLD) return false;
  return recall.matchedTerms >= 2 || recall.score >= 0.9;
}

/**
 * Recall similarity rescaled to "how much this should pull the turn local",
 * 0 below the replay threshold and 1 at a perfect match.
 *
 * The raw score is the wrong thing to hand the router. Cosine over SIF
 * embeddings of text from one codebase has a floor well above zero, so every
 * question looked like a partial memory hit and a weight large enough to
 * matter on a real hit was quietly pulling unrelated questions local too -
 * visible in `pca eval` as a routing coverage that did not match the answer
 * kinds beside it. Rescaling against the same threshold the answerer uses keeps
 * the router from promising a recall the answerer will not deliver.
 */
export function recallStrength(recall: Recall | undefined): number {
  if (!recall || !isStrongRecall(recall)) return 0;
  // A step, then a ramp. A pure ramp would hand the router ~0 for a recall that
  // only just clears the threshold - so the answerer would replay a stored
  // answer while the router was sending the same turn to Claude. Anything the
  // answerer considers replayable starts at half weight.
  return 0.5 + 0.5 * Math.min(1, (recall.score - REPLAY_THRESHOLD) / (1 - REPLAY_THRESHOLD));
}

/**
 * A message is code to continue if it ends mid-expression the way a half-typed
 * line does. Deliberately conservative: answering a question with a code
 * completion is worse than the reverse, so this only fires on a single line
 * that ends in an operator or an opening bracket.
 */
export function isCodeContinuation(text: string): boolean {
  const line = text.trimEnd();
  if (line.length === 0 || line.includes("\n")) return false;
  if (!/[.([{=:]$/.test(line)) return false;
  return /[\w$)\]]/.test(line);
}

export class LocalAssistant {
  private readonly bundle: Bundle;
  private engine: CompletionEngine | null = null;

  constructor(bundle: Bundle) {
    this.bundle = bundle;
  }

  answer(ctx: TurnContext): LocalAnswer {
    const neighbors = NEIGHBORS_RE.exec(ctx.question);
    if (neighbors) {
      const found = this.neighbors(neighbors[1]);
      if (found) return found;
    }
    if (ctx.looksLikeCode && this.bundle.isTrained) return this.complete(ctx.question);

    const best = ctx.recalls[0];
    if (best && isStrongRecall(best)) return this.fromMemory(best);
    if (ctx.hits.length > 0) return this.locate(ctx, best);
    if (best) return this.fromMemory(best);

    return {
      text:
        "I have nothing local on that - no index hit and nothing in memory. " +
        "Ask Claude with `/claude`, and the answer will be here next time.",
      kind: "unsure",
      confidence: 0,
      sources: [],
      recalled: null,
    };
  }

  private fromMemory(recall: Recall): LocalAnswer {
    const { lesson } = recall;
    const when = lesson.at.slice(0, 10);
    const who = lesson.source === "claude" ? (lesson.model ?? "claude") : "the local models";
    return {
      text:
        `${lesson.answer.trim()}\n\n` +
        `_Recalled from memory: answered by ${who} on ${when}, in reply to ` +
        `"${lesson.question.trim()}". Use \`/claude\` to ask again from scratch._`,
      kind: "memory",
      confidence: Math.min(0.95, 0.4 + recall.score),
      sources: lesson.cites.length > 0 ? lesson.cites : lesson.grounding,
      recalled: recall,
    };
  }

  private locate(ctx: TurnContext, recall: Recall | undefined): LocalAnswer {
    const shown = ctx.hits.slice(0, 4);
    const lines: string[] = ["From the local index, most relevant first:"];
    for (const hit of shown) {
      const label = hit.chunk.symbol ? ` (${hit.chunk.symbol})` : "";
      lines.push("");
      lines.push(`**${hit.file}:${hit.chunk.startLine}**${label}`);
      lines.push("```");
      lines.push(excerpt(hit.chunk.text, 12));
      lines.push("```");
    }
    if (recall && recall.score > 0.15) {
      lines.push("");
      lines.push(
        `Related answer in memory: "${recall.lesson.question.trim()}" (${recall.lesson.at.slice(0, 10)}).`,
      );
    }
    lines.push("");
    lines.push("_That is retrieval, not reasoning - `/claude` if you want it explained._");
    // Confidence tracks how far the top hit stands above the pack; a flat
    // ranking means the index matched everything, and therefore nothing.
    const top = shown[0]?.score ?? 0;
    const second = shown[1]?.score ?? 0;
    const margin = top > 0 ? (top - second) / top : 0;
    return {
      text: lines.join("\n"),
      kind: "locate",
      confidence: Math.min(0.8, 0.25 + margin),
      sources: shown.map((h) => `${h.file}:${h.chunk.startLine}`),
      recalled: recall ?? null,
    };
  }

  private complete(prefix: string): LocalAnswer {
    if (!this.engine) this.engine = new CompletionEngine(this.bundle);
    const result = this.engine.complete({ prefix, count: 3, retrieval: true, singleLine: true });
    if (result.suggestions.length === 0) {
      return {
        text: "No completion - the n-gram has no evidence for that context.",
        kind: "complete",
        confidence: 0,
        sources: [],
        recalled: null,
      };
    }
    const lines = result.suggestions.map(
      (s, i) =>
        `${i === 0 ? "›" : " "} \`${prefix}${s.text}\`  ` +
        `score ${s.score.toFixed(2)} · lm ${s.ngramLogProb.toFixed(2)}`,
    );
    lines.push("");
    lines.push(`_${result.timing.total.toFixed(1)}ms, local, no network._`);
    return {
      text: lines.join("\n"),
      kind: "complete",
      confidence: 0.7,
      sources: [],
      recalled: null,
    };
  }

  private neighbors(token: string): LocalAnswer | null {
    if (!this.bundle.isTrained) return null;
    const embeddings = this.bundle.embeddings;
    if (!embeddings) return null;
    const id = this.bundle.tokenizer.idOf(token);
    if (id === undefined) return null;
    const near = embeddings.nearest(id, 8);
    if (near.length === 0) return null;
    const rows = near.map((n) => {
      const bar = "█".repeat(Math.max(1, Math.round(n.score * 16)));
      return `  ${this.bundle.tokenizer.pieceOf(n.id).padEnd(16)} ${bar} ${n.score.toFixed(3)}`;
    });
    return {
      text: [`Nearest to \`${token}\` in the learned embedding space:`, "", ...rows].join("\n"),
      kind: "neighbors",
      confidence: 0.6,
      sources: [],
      recalled: null,
    };
  }
}

function excerpt(text: string, maxLines: number): string {
  const lines = text.split("\n");
  if (lines.length <= maxLines) return text.trimEnd();
  return `${lines.slice(0, maxLines).join("\n").trimEnd()}\n…`;
}
