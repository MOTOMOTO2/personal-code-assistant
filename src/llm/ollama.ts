/**
 * The local language model: an open-weights model served by Ollama on this
 * machine.
 *
 * This is the half of the assistant that reasons in prose. The from-scratch
 * models in src/ml are good at what a model trained on one person's code can
 * be good at - completion, lookup, recall - but a model that size cannot hold
 * a conversation. An 8B open model on the local GPU can, and it never leaves
 * the machine: no API key, no cloud, nothing sent anywhere but localhost.
 *
 * Only Ollama's own HTTP API is used (/api/chat, /api/tags), streamed as
 * newline-delimited JSON, so there is no dependency to install.
 */

export const DEFAULT_LOCAL_MODEL = "qwen3:8b";
export const DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434";

/** A tool the model may call, described the way Ollama expects. */
export interface ToolSpec {
  name: string;
  description: string;
  /** JSON Schema for the arguments object. */
  parameters: Record<string, unknown>;
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

/** One message in Ollama's chat format. */
export interface ModelMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  thinking?: string;
  tool_calls?: Array<{ function: { name: string; arguments: Record<string, unknown> } }>;
  tool_name?: string;
}

export interface TurnOptions {
  messages: ModelMessage[];
  tools?: ToolSpec[];
  /** Let the model reason before it answers, if it can. */
  think?: boolean;
  onText?: (delta: string) => void;
  onThinking?: (delta: string) => void;
}

/** What the model produced in one request: words, reasoning, or tool calls. */
export interface ModelTurn {
  content: string;
  thinking: string;
  toolCalls: ToolCall[];
  /** Tokens generated, as Ollama counts them. */
  evalTokens: number;
}

export class LocalModelError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LocalModelError";
  }
}

/**
 * Context window to ask for. Ollama's default is too small for a few tool
 * results, and the weights plus an 8K cache is what fits an 8 GB card.
 */
export const DEFAULT_CONTEXT_TOKENS = 8192;

export class OllamaClient {
  readonly host: string;
  readonly model: string;
  readonly contextTokens: number;
  private fetchImpl: typeof fetch;
  /** Set once the model has said it cannot think, so later turns skip asking. */
  private thinkingUnsupported = false;

  constructor(opts: { host?: string; model?: string; contextTokens?: number; fetchImpl?: typeof fetch } = {}) {
    this.host = (opts.host ?? process.env.OLLAMA_HOST ?? DEFAULT_OLLAMA_HOST).replace(/\/+$/, "");
    if (!/^https?:\/\//.test(this.host)) this.host = `http://${this.host}`;
    this.model = opts.model || process.env.PCA_MODEL || DEFAULT_LOCAL_MODEL;
    this.contextTokens = opts.contextTokens ?? (Number(process.env.PCA_CONTEXT) || DEFAULT_CONTEXT_TOKENS);
    this.fetchImpl = opts.fetchImpl ?? fetch;
  }

  /** The names of the models pulled into this Ollama, or null if it is not running. */
  async installedModels(): Promise<string[] | null> {
    try {
      const res = await this.fetchImpl(`${this.host}/api/tags`, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) return null;
      const body = (await res.json()) as { models?: Array<{ name?: string; model?: string }> };
      return (body.models ?? []).map((m) => m.name ?? m.model ?? "").filter(Boolean);
    } catch {
      return null;
    }
  }

  /** Whether `model` is pulled. `qwen3:8b` and `qwen3:8b-q4_K_M` are different models; `qwen3` means `qwen3:latest`. */
  static hasModel(installed: string[], model: string): boolean {
    const want = model.includes(":") ? model : `${model}:latest`;
    return installed.some((name) => name === want || name === model);
  }

  /**
   * One request: stream the reply, and hand back whatever tool calls the model
   * made. Running the tools and asking again is the caller's loop.
   */
  async turn(opts: TurnOptions): Promise<ModelTurn> {
    const think = Boolean(opts.think) && !this.thinkingUnsupported;
    try {
      return await this.request(opts, think);
    } catch (err) {
      // Not every model can think. Asking one that cannot is an error, not a
      // no-op, so fall back once and remember.
      if (think && err instanceof LocalModelError && /think/i.test(err.message)) {
        this.thinkingUnsupported = true;
        return this.request(opts, false);
      }
      throw err;
    }
  }

  private async request(opts: TurnOptions, think: boolean): Promise<ModelTurn> {
    let res: Response;
    try {
      res = await this.fetchImpl(`${this.host}/api/chat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: this.model,
          messages: opts.messages,
          tools: opts.tools?.length
            ? opts.tools.map((t) => ({ type: "function", function: t }))
            : undefined,
          think,
          stream: true,
          options: { num_ctx: this.contextTokens },
        }),
      });
    } catch (err) {
      throw new LocalModelError(
        `Ollama is not answering at ${this.host} (${(err as Error).message}). Start it with \`ollama serve\`.`,
      );
    }

    if (!res.ok || !res.body) {
      const detail = await res.text().catch(() => "");
      let message = detail;
      try {
        message = (JSON.parse(detail) as { error?: string }).error ?? detail;
      } catch {
        // not JSON - use the text as it came
      }
      throw new LocalModelError(`Ollama answered ${res.status}: ${message.trim().slice(0, 300)}`);
    }

    const out: ModelTurn = { content: "", thinking: "", toolCalls: [], evalTokens: 0 };
    const decoder = new TextDecoder();
    let held = "";

    const take = (line: string) => {
      if (!line.trim()) return;
      const chunk = JSON.parse(line) as {
        error?: string;
        message?: ModelMessage;
        eval_count?: number;
      };
      if (chunk.error) throw new LocalModelError(chunk.error);
      const m = chunk.message;
      if (m?.thinking) {
        out.thinking += m.thinking;
        opts.onThinking?.(m.thinking);
      }
      if (m?.content) {
        out.content += m.content;
        opts.onText?.(m.content);
      }
      // Tool calls arrive whole, not in fragments.
      for (const call of m?.tool_calls ?? []) {
        out.toolCalls.push({ name: call.function.name, arguments: parseArguments(call.function.arguments) });
      }
      if (chunk.eval_count) out.evalTokens = chunk.eval_count;
    };

    for await (const bytes of res.body as unknown as AsyncIterable<Uint8Array>) {
      held += decoder.decode(bytes, { stream: true });
      const lines = held.split("\n");
      held = lines.pop() ?? "";
      for (const line of lines) take(line);
    }
    take(held + decoder.decode());
    return out;
  }
}

/** Ollama sends arguments as an object; some models send a JSON string instead. */
function parseArguments(raw: unknown): Record<string, unknown> {
  if (raw && typeof raw === "object") return raw as Record<string, unknown>;
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (parsed && typeof parsed === "object") return parsed as Record<string, unknown>;
    } catch {
      // fall through
    }
  }
  return {};
}
