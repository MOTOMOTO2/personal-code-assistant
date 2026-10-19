import { ModelMessage, ModelTurn, ToolCall, TurnOptions } from "../llm/ollama.js";
import { Tool } from "./tools.js";

/**
 * The loop that turns a local model into an assistant that can look things up.
 *
 * Ask the model; if it called tools, run them, hand back what they said, and
 * ask again; stop when it answers in words. Small models drift in loops like
 * this in two predictable ways - calling the same tool with the same arguments
 * again, and never stopping - so both are capped here rather than trusted to
 * the prompt.
 */

/** The slice of OllamaClient the loop needs, so tests can substitute one. */
export interface AgentModel {
  turn(opts: TurnOptions): Promise<ModelTurn>;
}

export interface AgentOptions {
  model: AgentModel;
  system: string;
  /** The conversation so far, ending with the message to answer. */
  messages: ModelMessage[];
  tools: Tool[];
  /**
   * Calls to make before the model is asked anything, shown to it as calls it
   * made itself. An 8B model will sometimes answer "what failed on my
   * dashboard?" from imagination rather than look; when the question plainly
   * needs a tool, this makes sure the evidence is already in front of it.
   */
  prefetch?: ToolCall[];
  think?: boolean;
  /** Tool rounds before the model is told to answer with what it has. */
  maxRounds?: number;
  onText?: (delta: string) => void;
  /** A tool is about to run: what it is doing, in a few words. */
  onNote?: (note: string) => void;
}

export interface AgentResult {
  /** The words of the final answer, without anything said between tool calls. */
  text: string;
  /** Every tool that ran, in order. */
  used: string[];
  /** Whether any of them read something that changes (the web, a running job). */
  live: boolean;
  rounds: number;
}

/** Characters of one tool result the model is shown. Enough to answer from; small enough to fit 8K. */
export const TOOL_RESULT_MAX = 6_000;

export async function runAgent(opts: AgentOptions): Promise<AgentResult> {
  const byName = new Map(opts.tools.map((t) => [t.spec.name, t]));
  const specs = opts.tools.map((t) => t.spec);
  const messages: ModelMessage[] = [{ role: "system", content: opts.system }, ...opts.messages];
  const maxRounds = opts.maxRounds ?? 6;
  const seen = new Set<string>();
  const used: string[] = [];
  let live = false;
  let untrustedSeen = false;

  const execute = async (call: ToolCall): Promise<string> => {
    const tool = byName.get(call.name);
    let result: string;
    const key = `${call.name} ${JSON.stringify(call.arguments)}`;
    if (!tool) {
      result = `There is no tool called ${call.name}. The tools are: ${[...byName.keys()].join(", ")}.`;
    } else if (seen.has(key)) {
      result = "You already ran exactly this. Its result is above - use it, or try something different.";
    } else if (tool.acts && untrustedSeen) {
      // Text from the web is not the user. Once some has been read, nothing
      // that changes things runs on the strength of this answer alone.
      result =
        "Not run: this answer has read web content, so nothing that changes anything happens in it. " +
        "Tell the user what you would do, and they can ask for it in their next message.";
    } else {
      seen.add(key);
      used.push(call.name);
      live ||= tool.live;
      untrustedSeen ||= Boolean(tool.untrusted);
      opts.onNote?.(tool.note(call.arguments));
      try {
        result = await tool.run(call.arguments);
      } catch (e) {
        result = `Failed: ${(e as Error).message}`;
      }
    }
    if (result.length > TOOL_RESULT_MAX) {
      result = `${result.slice(0, TOOL_RESULT_MAX)}\n… (cut at ${TOOL_RESULT_MAX} characters)`;
    }
    return result || "(nothing)";
  };

  const record = async (calls: ToolCall[], content = "", thinking = "") => {
    messages.push({
      role: "assistant",
      content,
      thinking: thinking || undefined,
      tool_calls: calls.map((c) => ({ function: { name: c.name, arguments: c.arguments } })),
    });
    for (const call of calls) {
      messages.push({ role: "tool", tool_name: call.name, content: await execute(call) });
    }
  };

  const ahead = (opts.prefetch ?? []).filter((c) => byName.has(c.name));
  if (ahead.length > 0) await record(ahead);

  for (let round = 1; ; round++) {
    const lastChance = round > maxRounds;
    if (lastChance) {
      messages.push({
        role: "system",
        content: "That is enough looking. Answer now with what you have found, and say what you could not find out.",
      });
    }
    const turn = await opts.model.turn({
      messages,
      tools: lastChance ? undefined : specs,
      think: opts.think,
      onText: opts.onText,
    });

    if (turn.toolCalls.length === 0 || lastChance) {
      return { text: turn.content, used, live, rounds: round };
    }

    await record(turn.toolCalls, turn.content, turn.thinking);
  }
}
