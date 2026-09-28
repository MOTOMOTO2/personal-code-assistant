import fs from "node:fs";
import path from "node:path";
import { Bundle } from "../core/bundle.js";
import { ClaudeClient, Effort, StreamOptions, StreamResult, hasCredentials } from "../llm/claude.js";
import { renderContext, systemPrompt } from "../llm/prompts.js";
import { statePath } from "../util/paths.js";
import { LocalAssistant, TurnContext, buildTurnContext } from "./local.js";
import { Lesson, LessonStore, appendLesson, extractCites, loadLessons, newLessonId } from "./memory.js";

/**
 * One conversation, and the thing that makes this feel like a chat assistant
 * rather than a pile of CLI commands.
 *
 * Each turn is routed: the local models answer what they can (recall, lookup,
 * completion) in single-digit milliseconds with no network, and Claude answers
 * the rest. Either way the answer is written to memory, indexed on the next
 * train, and available locally from then on. The loop is the point - the local
 * half's coverage grows with use, and `pca stats --router` shows where the line
 * currently sits.
 */

export type Route = "local" | "claude";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  at: string;
  route?: Route;
  kind?: string;
  sources?: string[];
  costUsd?: number;
  /** Router probability that this turn needed Claude, at the time it was asked. */
  pClaude?: number;
}

/** The slice of the Claude client a session needs, so tests can substitute one. */
export interface ChatBackend {
  readonly model: string;
  stream(opts: StreamOptions): Promise<StreamResult>;
}

export interface AskOptions {
  /** Override the router for this turn. */
  force?: Route | null;
  effort?: Effort;
  showThinking?: boolean;
  onRoute?: (route: Route, pClaude: number, why: string) => void;
  onText?: (delta: string) => void;
  onThinking?: (delta: string) => void;
}

export interface AskResult {
  route: Route;
  text: string;
  kind: string;
  confidence: number;
  sources: string[];
  pClaude: number;
  costUsd: number;
  lessonId: string | null;
  /** Grounding the turn used, for `/why`. */
  grounding: string[];
}

/** Chat-specific addendum to the shared system prompt. Static, so the cache hits. */
const CHAT_ROLE = `You are answering inside an interactive chat session in the developer's terminal.

- Keep answers conversational and short unless depth was asked for.
- Earlier turns are real context; refer back to them rather than repeating yourself.
- A <memory> block holds answers previously given about this same codebase. Treat it as your own prior work: build on it, and correct it explicitly if it is now wrong.`;

export class ChatSession {
  readonly bundle: Bundle;
  readonly id: string;
  readonly turns: ChatTurn[] = [];
  private lessons: LessonStore;
  private local: LocalAssistant;
  private client: ChatBackend | null;
  private persist: boolean;
  /** Context of the most recent user turn, kept so `/claude` can re-route it. */
  private lastContext: TurnContext | null = null;
  private lastRoute: Route | null = null;
  private lastAnsweredLocally = false;
  totalCostUsd = 0;

  constructor(
    bundle: Bundle,
    opts: { id?: string; persist?: boolean; client?: ChatBackend } = {},
  ) {
    this.bundle = bundle;
    this.id = opts.id ?? new Date().toISOString().replace(/[:.]/g, "-");
    this.persist = opts.persist ?? true;
    this.client = opts.client ?? null;
    this.lessons = loadLessons(bundle);
    this.local = new LocalAssistant(bundle);
  }

  get memorySize(): number {
    return this.lessons.size;
  }

  /** Reload memory from disk - used after a retrain folds in new lessons. */
  refreshMemory(): void {
    this.lessons = loadLessons(this.bundle);
  }

  async ask(question: string, opts: AskOptions = {}): Promise<AskResult> {
    // Asking something new instead of retrying means the previous local answer
    // was good enough. Record that before the context is replaced.
    this.confirmLocalAnswer();
    const ctx = buildTurnContext(this.bundle, this.lessons, question);
    this.lastContext = ctx;
    this.turns.push({ role: "user", content: question, at: new Date().toISOString() });

    const router = this.bundle.router;
    const pClaude = router.probability(ctx.features);
    const keyed = this.client !== null || hasCredentials();
    let route: Route;
    let why: string;
    if (opts.force) {
      route = opts.force;
      why = `forced ${route}`;
    } else if (!keyed) {
      route = "local";
      why = "no ANTHROPIC_API_KEY, local only";
    } else {
      route = router.decide(ctx.features);
      why = routeReason(ctx, pClaude);
    }
    if (route === "claude" && !keyed) {
      route = "local";
      why = "no ANTHROPIC_API_KEY, local only";
    }
    opts.onRoute?.(route, pClaude, why);
    this.lastRoute = route;

    const result =
      route === "local" ? this.answerLocally(ctx, pClaude) : await this.answerWithClaude(ctx, pClaude, opts);

    this.turns.push({
      role: "assistant",
      content: result.text,
      at: new Date().toISOString(),
      route: result.route,
      kind: result.kind,
      sources: result.sources,
      costUsd: result.costUsd,
      pClaude,
    });
    this.save();
    return result;
  }

  /**
   * Re-answer the last question with Claude. This is also the router's training
   * signal: the user just told us the local answer was not good enough.
   */
  async escalate(opts: AskOptions = {}): Promise<AskResult> {
    const ctx = this.lastContext;
    if (!ctx) throw new Error("Nothing to escalate - ask something first.");
    if (this.lastAnsweredLocally) {
      this.bundle.router.learn(ctx.features, 1);
      this.bundle.saveRouter();
    }
    // Drop the local attempt so the transcript shows one answer per question.
    if (this.turns.length > 0 && this.turns[this.turns.length - 1].role === "assistant") {
      this.turns.pop();
    }
    const result = await this.answerWithClaude(ctx, this.bundle.router.probability(ctx.features), opts);
    this.turns.push({
      role: "assistant",
      content: result.text,
      at: new Date().toISOString(),
      route: "claude",
      kind: result.kind,
      sources: result.sources,
      costUsd: result.costUsd,
    });
    this.save();
    return result;
  }

  /**
   * Called when the user asks something new after a local answer: they did not
   * retry, so the route was right. Weak but free supervision.
   */
  confirmLocalAnswer(): void {
    if (!this.lastAnsweredLocally || !this.lastContext) return;
    this.bundle.router.learn(this.lastContext.features, 0);
    this.bundle.saveRouter();
    this.lastAnsweredLocally = false;
  }

  private answerLocally(ctx: TurnContext, pClaude: number): AskResult {
    const answer = this.local.answer(ctx);
    this.lastAnsweredLocally = true;
    // Local answers that recalled a prior lesson are not new knowledge, so they
    // are not written back - that would compound paraphrases of paraphrases.
    return {
      route: "local",
      text: answer.text,
      kind: answer.kind,
      confidence: answer.confidence,
      sources: answer.sources,
      pClaude,
      costUsd: 0,
      lessonId: null,
      grounding: ctx.hits.slice(0, 4).map((h) => `${h.file}:${h.chunk.startLine}`),
    };
  }

  private async answerWithClaude(
    ctx: TurnContext,
    pClaude: number,
    opts: AskOptions,
  ): Promise<AskResult> {
    if (!this.client) this.client = new ClaudeClient();
    this.lastAnsweredLocally = false;

    const grounding = ctx.hits.slice(0, 8);
    const messages = this.buildMessages(ctx, grounding.length);
    const result = await this.client.stream({
      system: systemPrompt(this.bundle.style, CHAT_ROLE),
      messages,
      effort: opts.effort ?? "high",
      showThinking: opts.showThinking,
      onText: opts.onText,
      onThinking: opts.onThinking,
    });
    this.totalCostUsd += result.estimatedCostUsd;

    const groundingPaths = grounding.map((h) => `${h.file}:${h.chunk.startLine}`);
    const lesson: Lesson = {
      id: newLessonId(),
      at: new Date().toISOString(),
      question: ctx.question,
      answer: result.text,
      source: "claude",
      model: result.model,
      cites: extractCites(result.text),
      grounding: groundingPaths,
      tokensIn: result.usage.inputTokens,
      tokensOut: result.usage.outputTokens,
      costUsd: result.estimatedCostUsd,
    };
    // Only remember substantive answers; a one-line "I need more context" is
    // noise in both retrieval and the training corpus.
    let lessonId: string | null = null;
    if (result.text.trim().length >= 80 && result.stopReason !== "refusal") {
      appendLesson(this.bundle, lesson);
      this.lessons = loadLessons(this.bundle);
      lessonId = lesson.id;
    }

    return {
      route: "claude",
      text: result.text,
      kind: "claude",
      confidence: 1,
      sources: lesson.cites.length > 0 ? lesson.cites : groundingPaths,
      pClaude,
      costUsd: result.estimatedCostUsd,
      lessonId,
      grounding: groundingPaths,
    };
  }

  /**
   * Conversation history for the API, with retrieval and memory attached to the
   * final user turn only. Earlier turns keep their plain text so the prefix
   * stays byte-stable and the prompt cache keeps hitting as the chat grows.
   */
  private buildMessages(ctx: TurnContext, groundingCount: number) {
    const messages: Array<{ role: "user" | "assistant"; content: string }> = [];
    for (let i = 0; i < this.turns.length - 1; i++) {
      const turn = this.turns[i];
      messages.push({ role: turn.role, content: turn.content });
    }
    const blocks: string[] = [];
    if (groundingCount > 0) blocks.push(renderContext(ctx.hits.slice(0, groundingCount)));
    const memory = ctx.recalls.filter((r) => r.score > 0.12).slice(0, 2);
    if (memory.length > 0) {
      const rendered = memory
        .map(
          (r) =>
            `--- asked ${r.lesson.at.slice(0, 10)}: "${r.lesson.question.trim()}"\n${r.lesson.answer.trim()}`,
        )
        .join("\n\n");
      blocks.push(`<memory>\nPrevious answers about this codebase.\n\n${rendered}\n</memory>`);
    }
    blocks.push(ctx.question);
    messages.push({ role: "user", content: blocks.filter(Boolean).join("\n\n") });
    // The API rejects two consecutive turns with the same role; a local answer
    // that was escalated can leave the history in that shape.
    return collapseRoles(messages);
  }

  transcriptPath(): string {
    return statePath(this.bundle.ws, "chat", `${this.id}.json`);
  }

  save(): void {
    if (!this.persist) return;
    const file = this.transcriptPath();
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(
      file,
      JSON.stringify({ id: this.id, turns: this.turns, costUsd: this.totalCostUsd }, null, 2),
      "utf8",
    );
  }

  clear(): void {
    this.turns.length = 0;
    this.lastContext = null;
    this.lastRoute = null;
    this.lastAnsweredLocally = false;
  }

  get route(): Route | null {
    return this.lastRoute;
  }
}

function collapseRoles(
  messages: Array<{ role: "user" | "assistant"; content: string }>,
): Array<{ role: "user" | "assistant"; content: string }> {
  const out: Array<{ role: "user" | "assistant"; content: string }> = [];
  for (const m of messages) {
    const prev = out[out.length - 1];
    if (prev && prev.role === m.role) prev.content = `${prev.content}\n\n${m.content}`;
    else out.push({ ...m });
  }
  // A conversation must start with a user turn.
  while (out.length > 0 && out[0].role === "assistant") out.shift();
  return out;
}

function routeReason(ctx: TurnContext, pClaude: number): string {
  const top = ctx.recalls[0];
  if (top && top.score >= 0.45) return `memory hit ${(top.score * 100).toFixed(0)}%`;
  if (ctx.looksLikeCode) return "looks like code to continue";
  if (pClaude >= 0.5) return `router p(claude)=${pClaude.toFixed(2)}`;
  return `router p(claude)=${pClaude.toFixed(2)}, ${ctx.hits.length} index hits`;
}
