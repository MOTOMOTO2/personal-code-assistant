import Anthropic from "@anthropic-ai/sdk";

/**
 * Claude wrapper for the "deep reasoning" half of the assistant.
 *
 * Division of labour: the local models handle everything on the keystroke path,
 * where a network round trip would be unacceptable. Claude handles the work that
 * genuinely needs a frontier model - explaining unfamiliar code, reviewing a
 * diff, planning a refactor - and it gets the local retrieval index as grounding
 * so it answers about *your* code rather than generic patterns.
 */

/** Default model. Opus 5 is the strongest generally-available coding model. */
export const DEFAULT_MODEL = "claude-opus-5";

/** Fallback used if a safety classifier declines the request. */
const FALLBACK_MODEL = "claude-opus-4-8";
const FALLBACK_BETA = "server-side-fallback-2026-06-01";

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";

export interface StreamOptions {
  /** Stable, cacheable instructions. Keep this byte-identical across calls. */
  system: string;
  /** Conversation turns. */
  messages: Anthropic.MessageParam[];
  model?: string;
  maxTokens?: number;
  effort?: Effort;
  /** Surface a summary of Claude's reasoning as it arrives. */
  showThinking?: boolean;
  onText?: (delta: string) => void;
  onThinking?: (delta: string) => void;
}

export interface StreamResult {
  text: string;
  model: string;
  stopReason: string | null;
  refusal: { category: string | null; explanation: string | null } | null;
  usage: {
    inputTokens: number;
    outputTokens: number;
    cacheReadTokens: number;
    cacheCreationTokens: number;
  };
  /** Estimated cost in USD, using the published Opus 5 rates. */
  estimatedCostUsd: number;
}

/** Published per-million-token rates, for the local cost readout. */
const PRICING: Record<string, { input: number; output: number; cacheRead: number }> = {
  "claude-opus-5": { input: 5, output: 25, cacheRead: 0.5 },
  "claude-opus-4-8": { input: 5, output: 25, cacheRead: 0.5 },
  "claude-sonnet-5": { input: 2, output: 10, cacheRead: 0.2 },
  "claude-haiku-4-5": { input: 1, output: 5, cacheRead: 0.1 },
};

export class MissingApiKeyError extends Error {
  constructor() {
    super(
      [
        "No Anthropic credentials found.",
        "",
        "Set an API key:   setx ANTHROPIC_API_KEY sk-ant-...      (Windows)",
        "                  export ANTHROPIC_API_KEY=sk-ant-...    (macOS/Linux)",
        "",
        "Get one at https://console.anthropic.com/settings/keys",
        "Local commands (complete, search, train, eval) work without a key.",
      ].join("\n"),
    );
    this.name = "MissingApiKeyError";
  }
}

export function hasCredentials(): boolean {
  return Boolean(process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN);
}

export class ClaudeClient {
  private client: Anthropic;
  readonly model: string;

  constructor(model = DEFAULT_MODEL) {
    if (!hasCredentials()) throw new MissingApiKeyError();
    // The SDK resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN itself.
    this.client = new Anthropic();
    this.model = model;
  }

  /**
   * Stream a response. Streaming is not optional here - code answers are long
   * enough that a buffered request risks an HTTP timeout, and partial output
   * makes the tool feel responsive.
   */
  async stream(opts: StreamOptions): Promise<StreamResult> {
    const model = opts.model ?? this.model;
    const stream = this.client.beta.messages.stream({
      model,
      max_tokens: opts.maxTokens ?? 32_000,
      // Adaptive thinking: Claude decides how much to reason per request.
      thinking: { type: "adaptive", display: opts.showThinking ? "summarized" : "omitted" },
      output_config: { effort: opts.effort ?? "high" },
      // A refused request is re-run server-side on the fallback model inside the
      // same call, so a classifier false positive does not lose the turn.
      betas: [FALLBACK_BETA],
      fallbacks: [{ model: FALLBACK_MODEL }],
      // The system prompt is the stable prefix, so it carries the cache
      // breakpoint. Retrieved code goes in the user turn, after it.
      system: [{ type: "text", text: opts.system, cache_control: { type: "ephemeral" } }],
      messages: opts.messages,
    });

    let text = "";
    for await (const event of stream) {
      if (event.type !== "content_block_delta") continue;
      if (event.delta.type === "text_delta") {
        text += event.delta.text;
        opts.onText?.(event.delta.text);
      } else if (event.delta.type === "thinking_delta") {
        opts.onThinking?.(event.delta.thinking);
      }
    }

    const final = await stream.finalMessage();
    const usage = {
      inputTokens: final.usage.input_tokens ?? 0,
      outputTokens: final.usage.output_tokens ?? 0,
      cacheReadTokens: final.usage.cache_read_input_tokens ?? 0,
      cacheCreationTokens: final.usage.cache_creation_input_tokens ?? 0,
    };
    const refusal =
      final.stop_reason === "refusal"
        ? {
            category: (final.stop_details as { category?: string } | null)?.category ?? null,
            explanation: (final.stop_details as { explanation?: string } | null)?.explanation ?? null,
          }
        : null;

    return {
      text,
      model: final.model ?? model,
      stopReason: final.stop_reason ?? null,
      refusal,
      usage,
      estimatedCostUsd: estimateCost(final.model ?? model, usage),
    };
  }
}

function estimateCost(
  model: string,
  usage: { inputTokens: number; outputTokens: number; cacheReadTokens: number; cacheCreationTokens: number },
): number {
  const rate = PRICING[model] ?? PRICING[DEFAULT_MODEL];
  const cost =
    (usage.inputTokens / 1e6) * rate.input +
    (usage.outputTokens / 1e6) * rate.output +
    (usage.cacheReadTokens / 1e6) * rate.cacheRead +
    // Cache writes bill at 1.25x the input rate.
    (usage.cacheCreationTokens / 1e6) * rate.input * 1.25;
  return Math.round(cost * 1e6) / 1e6;
}
