import { Bundle } from "../core/bundle.js";
import { Effort, hasCredentials } from "../llm/claude.js";
import { ChatBackend, ChatSession, ChatTurn, Route } from "./session.js";
import { ThreadMeta, ThreadStore, assertSafeThreadId } from "./threads.js";

/**
 * The part that makes this messageable rather than merely interactive.
 *
 * A phone is a hostile client: it locks the screen, backgrounds the tab, drops
 * the connection when you walk into a lift, and the user expects the reply to be
 * there anyway. So a turn does not belong to a request. You post a message, the
 * hub answers it in the background - streaming to whoever happens to be attached
 * - and writes it to the conversation on disk. Reattach later, from anything, and
 * a snapshot brings you up to date: finished answers whole, an in-flight one with
 * the text it has produced so far.
 *
 * That design also gives multi-device for free: the browser on your phone and
 * the browser on your laptop subscribe to the same thread and see the same
 * stream, because neither of them is where the conversation lives.
 *
 * One turn at a time per conversation (further messages queue, like an unsent
 * text), unlimited conversations in parallel.
 */

export type TurnStatus = "queued" | "routing" | "thinking" | "streaming" | "done" | "error";

/** A conversation message as the wire sees it: a turn plus its position. */
export interface Message extends ChatTurn {
  seq: number;
}

export interface PendingTurn {
  /** Client-supplied or generated id, so a sender can follow its own message. */
  id: string;
  question: string;
  status: TurnStatus;
  route: Route | null;
  why: string | null;
  pClaude: number | null;
  /** Answer text so far. Complete when status is "done". */
  text: string;
  thinking: string;
  error: string | null;
  startedAt: string;
}

export interface Snapshot {
  thread: ThreadMeta;
  messages: Message[];
  /** The turn being answered right now, if any. */
  pending: PendingTurn | null;
  /** Messages accepted but not started, oldest first. */
  queued: Array<{ id: string; question: string }>;
}

/**
 * Every event carries the id of the message it belongs to, so a client can tell
 * its own turn from one a different device started in the same conversation.
 */
export type HubEvent =
  | { type: "snapshot"; snapshot: Snapshot }
  | { type: "user"; id: string; message: Message }
  | { type: "queued"; id: string; question: string }
  | { type: "route"; id: string; route: Route; why: string; pClaude: number }
  | { type: "thinking"; id: string; text: string }
  | { type: "text"; id: string; text: string }
  | { type: "done"; id: string; message: Message; lessonId: string | null; thread: ThreadMeta }
  | { type: "error"; id: string; message: string }
  | { type: "thread"; thread: ThreadMeta };

export interface SendOptions {
  force?: Route | null;
  effort?: Effort;
  showThinking?: boolean;
  /** Echoed back on the events for this turn, so a client can match them up. */
  id?: string;
}

interface Job {
  id: string;
  question: string;
  opts: SendOptions;
  /** Set for an escalation of the previous answer rather than a new question. */
  escalate?: boolean;
}

interface Live {
  session: ChatSession;
  subscribers: Set<(event: HubEvent) => void>;
  queue: Job[];
  pending: PendingTurn | null;
  running: boolean;
  lastTouched: number;
}

/** Conversations kept warm in memory; the rest are reloaded from disk on demand. */
const MAX_LIVE = 8;

export class ChatHub {
  readonly bundle: Bundle;
  readonly threads: ThreadStore;
  private live = new Map<string, Live>();
  private backend: ChatBackend | null;
  private defaults: { effort?: Effort; showThinking?: boolean };

  constructor(
    bundle: Bundle,
    opts: { backend?: ChatBackend; effort?: Effort; showThinking?: boolean } = {},
  ) {
    this.bundle = bundle;
    this.threads = new ThreadStore(bundle.ws);
    this.backend = opts.backend ?? null;
    this.defaults = { effort: opts.effort, showThinking: opts.showThinking };
  }

  get claudeAvailable(): boolean {
    return this.backend !== null || hasCredentials();
  }

  list(): ThreadMeta[] {
    // In-memory conversations can be ahead of disk for a few milliseconds during
    // a turn; merge so a list request never shows a stale title or count.
    const metas = new Map(this.threads.list().map((m) => [m.id, m]));
    for (const [id, live] of this.live) {
      const meta = this.metaOfLive(live);
      if (meta.messages > 0 || metas.has(id)) metas.set(id, meta);
    }
    return [...metas.values()].sort((a, b) =>
      a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0,
    );
  }

  create(title?: string): ThreadMeta {
    return this.threads.create(title);
  }

  rename(id: string, title: string): ThreadMeta | null {
    assertSafeThreadId(id);
    const live = this.live.get(id);
    if (live) {
      live.session.title = title.trim().slice(0, 200);
      live.session.save();
      const meta = this.metaOfLive(live);
      this.emit(id, { type: "thread", thread: meta });
      return meta;
    }
    return this.threads.rename(id, title);
  }

  /** Forget a conversation. An in-flight turn is left to finish into the void. */
  remove(id: string): boolean {
    assertSafeThreadId(id);
    const live = this.live.get(id);
    // A turn still running would otherwise save the file back into existence.
    if (live) live.session.disablePersistence();
    this.live.delete(id);
    return this.threads.remove(id);
  }

  snapshot(id: string): Snapshot | null {
    assertSafeThreadId(id);
    const live = this.live.get(id);
    if (live) return this.snapshotOf(live);
    const record = this.threads.load(id);
    if (!record) return null;
    return {
      thread: this.threads.meta(id)!,
      messages: record.turns.map((turn, seq) => ({ ...turn, seq })),
      pending: null,
      queued: [],
    };
  }

  /**
   * Accept a message and answer it in the background.
   *
   * Returns as soon as the message is recorded, which is the whole point: the
   * HTTP request that posted it can end, the phone can sleep, and the answer
   * still arrives.
   */
  send(id: string, question: string, opts: SendOptions = {}): { id: string; status: TurnStatus } {
    const text = question.trim();
    if (text.length === 0) throw new Error("A message cannot be empty.");
    const live = this.ensureLive(id);
    const job: Job = { id: opts.id ?? randomId(), question: text, opts };
    live.queue.push(job);
    if (live.running) this.emit(id, { type: "queued", id: job.id, question: text });
    this.pump(id);
    return { id: job.id, status: this.statusOf(live, job.id) };
  }

  /** Re-answer the last question with Claude, and teach the router from it. */
  escalate(id: string, opts: SendOptions = {}): { id: string; status: TurnStatus } {
    const live = this.ensureLive(id);
    const last = [...live.session.turns].reverse().find((t) => t.role === "user");
    if (!last) throw new Error("Nothing to escalate - send a message first.");
    const job: Job = { id: opts.id ?? randomId(), question: last.content, opts, escalate: true };
    live.queue.push(job);
    if (live.running) this.emit(id, { type: "queued", id: job.id, question: last.content });
    this.pump(id);
    return { id: job.id, status: this.statusOf(live, job.id) };
  }

  /**
   * Where a just-accepted message got to. A local turn is answered
   * synchronously, so by the time `send` returns it can already be done.
   */
  private statusOf(live: Live, jobId: string): TurnStatus {
    if (live.queue.some((job) => job.id === jobId)) return "queued";
    if (live.pending && live.pending.id === jobId) return live.pending.status;
    return "done";
  }

  /**
   * Attach to a conversation. The callback gets a snapshot immediately, then
   * every event until the returned function is called.
   */
  subscribe(id: string, listener: (event: HubEvent) => void): () => void {
    const live = this.ensureLive(id);
    live.subscribers.add(listener);
    listener({ type: "snapshot", snapshot: this.snapshotOf(live) });
    return () => {
      live.subscribers.delete(listener);
    };
  }

  /** After a retrain: every open conversation should see the new memory. */
  refreshMemory(): void {
    for (const live of this.live.values()) live.session.refreshMemory();
  }

  /** Conversations currently held in memory, for `/health`. */
  get liveCount(): number {
    return this.live.size;
  }

  private ensureLive(id: string): Live {
    assertSafeThreadId(id);
    const existing = this.live.get(id);
    if (existing) {
      existing.lastTouched = Date.now();
      return existing;
    }
    const session = ChatSession.open(this.bundle, id, {
      ...(this.backend ? { client: this.backend } : {}),
    });
    const live: Live = {
      session,
      subscribers: new Set(),
      queue: [],
      pending: null,
      running: false,
      lastTouched: Date.now(),
    };
    this.live.set(id, live);
    this.evictIdle();
    return live;
  }

  /** Bound memory: drop the least recently touched idle conversations. */
  private evictIdle(): void {
    if (this.live.size <= MAX_LIVE) return;
    const candidates = [...this.live.entries()]
      .filter(([, l]) => !l.running && l.subscribers.size === 0 && l.queue.length === 0)
      .sort((a, b) => a[1].lastTouched - b[1].lastTouched);
    for (const [id] of candidates) {
      if (this.live.size <= MAX_LIVE) break;
      this.live.delete(id);
    }
  }

  private emit(id: string, event: HubEvent): void {
    const live = this.live.get(id);
    if (!live) return;
    // A copy: a listener is allowed to unsubscribe (a closed socket does exactly
    // that) while the event it was handed is still being delivered to the others.
    for (const listener of [...live.subscribers]) {
      try {
        listener(event);
      } catch {
        // A dead subscriber (closed socket) must not stop the turn or the others.
      }
    }
  }

  private snapshotOf(live: Live): Snapshot {
    return {
      thread: this.metaOfLive(live),
      messages: live.session.turns.map((turn, seq) => ({ ...turn, seq })),
      pending: live.pending ? { ...live.pending } : null,
      queued: live.queue.map((job) => ({ id: job.id, question: job.question })),
    };
  }

  private metaOfLive(live: Live): ThreadMeta {
    const session = live.session;
    const turns = session.turns;
    const last = turns[turns.length - 1];
    const lastAnswer = [...turns].reverse().find((t) => t.role === "assistant");
    return {
      id: session.id,
      title: session.title,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt,
      messages: turns.length,
      costUsd: session.totalCostUsd,
      preview: live.pending
        ? live.pending.question.replace(/\s+/g, " ").trim().slice(0, 60)
        : last
          ? last.content.replace(/\s+/g, " ").trim().slice(0, 60)
          : "",
      lastRoute: lastAnswer?.route,
    };
  }

  /** Run the next queued job, if one is queued and nothing is running. */
  private pump(id: string): void {
    const live = this.live.get(id);
    if (!live || live.running) return;
    const job = live.queue.shift();
    if (!job) return;
    live.running = true;
    live.lastTouched = Date.now();
    void this.run(id, live, job).finally(() => {
      live.running = false;
      // Drain the queue one turn at a time, so answers stay in order.
      this.pump(id);
    });
  }

  private async run(id: string, live: Live, job: Job): Promise<void> {
    const pending: PendingTurn = {
      id: job.id,
      question: job.question,
      status: "routing",
      route: job.escalate ? "claude" : null,
      why: job.escalate ? "you asked for Claude" : null,
      pClaude: null,
      text: "",
      thinking: "",
      error: null,
      startedAt: new Date().toISOString(),
    };
    live.pending = pending;

    const askOptions = {
      force: job.opts.force ?? null,
      effort: job.opts.effort ?? this.defaults.effort,
      showThinking: job.opts.showThinking ?? this.defaults.showThinking,
      onRoute: (route: Route, pClaude: number, why: string) => {
        pending.route = route;
        pending.why = why;
        pending.pClaude = pClaude;
        pending.status = route === "claude" ? "thinking" : "streaming";
        this.emit(id, { type: "route", id: job.id, route, why, pClaude });
      },
      onThinking: (text: string) => {
        pending.thinking += text;
        this.emit(id, { type: "thinking", id: job.id, text });
      },
      onText: (text: string) => {
        pending.text += text;
        pending.status = "streaming";
        this.emit(id, { type: "text", id: job.id, text });
      },
      onUser: (turn: ChatTurn, seq: number) => {
        this.emit(id, { type: "user", id: job.id, message: { ...turn, seq } });
      },
    };

    try {
      // An escalation re-answers the question already in the transcript, and
      // replaces the answer at the same position, so no user event is emitted.
      const result = job.escalate
        ? await live.session.escalate(askOptions)
        : await live.session.ask(job.question, askOptions);
      pending.status = "done";
      pending.text = result.text;
      this.finish(id, live, job.id, result.lessonId);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      pending.status = "error";
      pending.error = message;
      live.pending = null;
      this.emit(id, { type: "error", id: job.id, message });
      this.emit(id, { type: "thread", thread: this.metaOfLive(live) });
    }
  }

  private finish(id: string, live: Live, jobId: string, lessonId: string | null): void {
    const seq = live.session.turns.length - 1;
    const turn = live.session.turns[seq];
    live.pending = null;
    const thread = this.metaOfLive(live);
    if (turn) {
      this.emit(id, { type: "done", id: jobId, message: { ...turn, seq }, lessonId, thread });
    }
    // A new lesson is knowledge for every conversation, not just this one.
    if (lessonId) {
      for (const [otherId, other] of this.live) {
        if (otherId !== id) other.session.refreshMemory();
      }
    }
  }
}

function randomId(): string {
  return Math.random().toString(36).slice(2, 10);
}
