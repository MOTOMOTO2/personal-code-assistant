import { SearchHit } from "../index/store.js";
import { StyleProfile, styleToPrompt } from "../core/style.js";

/**
 * Prompt construction.
 *
 * Two rules shape everything here. First, the system prompt is *stable* - it
 * contains only the role and the measured style profile, so it stays
 * byte-identical between calls in a session and the prompt cache actually hits.
 * Second, retrieved code goes in the user turn, clearly delimited and always
 * attributed with `path:line`, so answers can cite real locations instead of
 * inventing them.
 */

const ROLE = `You are a coding assistant embedded in a developer's local toolchain, working on their own codebase.

How to be useful here:
- Answer the question that was asked, at the altitude it was asked. A one-line question gets a one-line answer.
- Ground every claim about this codebase in the provided context. Cite locations as \`path:line\`.
- If the provided context does not contain the answer, say so plainly and name what you would need to look at. Never invent file paths, function names, or APIs.
- When you write code, match the conventions measured from this repository (below). Do not reformat surrounding code or add a framework the project does not already use.
- Prefer showing a small, correct diff or snippet over a long explanation.
- Flag real problems (bugs, race conditions, unhandled errors) when you notice them, briefly, even if they were not the question.
- No preamble, no restating the question, no summary of what you just said.`;

export function systemPrompt(style: StyleProfile, extra?: string): string {
  const parts = [ROLE, "", "## Conventions measured from this repository", styleToPrompt(style)];
  if (extra) parts.push("", extra);
  return parts.join("\n");
}

/** Render retrieved chunks as a context block with real line numbers. */
export function renderContext(hits: SearchHit[]): string {
  if (hits.length === 0) return "";
  const blocks = hits.map((hit) => {
    const header = hit.chunk.symbol
      ? `${hit.file}:${hit.chunk.startLine}  (${hit.chunk.symbol})`
      : `${hit.file}:${hit.chunk.startLine}`;
    const numbered = hit.chunk.text
      .split("\n")
      .map((line, i) => `${String(hit.chunk.startLine + i).padStart(5)} | ${line}`)
      .join("\n");
    return `--- ${header}\n${numbered}`;
  });
  return `<codebase_context>\nRetrieved from the local index, most relevant first.\n\n${blocks.join("\n\n")}\n</codebase_context>`;
}

export function askPrompt(question: string, hits: SearchHit[]): string {
  const context = renderContext(hits);
  return context ? `${context}\n\n${question}` : question;
}

export function explainPrompt(
  path: string,
  code: string,
  hits: SearchHit[],
  question?: string,
): string {
  const numbered = code
    .split("\n")
    .map((line, i) => `${String(i + 1).padStart(5)} | ${line}`)
    .join("\n");
  const ask =
    question ??
    "Explain what this file does: its responsibility, the flow through it, and anything surprising or risky. Lead with the one-sentence version.";
  const related = renderContext(hits.filter((h) => h.file !== path));
  return [`<file path="${path}">\n${numbered}\n</file>`, related, ask].filter(Boolean).join("\n\n");
}

export function reviewPrompt(diff: string, hits: SearchHit[]): string {
  const context = renderContext(hits);
  return [
    context,
    `<diff>\n${diff}\n</diff>`,
    `Review this diff. Report only real problems, most severe first, each as:

**\`path:line\`** - what is wrong, and the concrete input or state that makes it go wrong.

Look for: logic errors, off-by-one and boundary mistakes, unhandled errors and rejected promises, null/undefined paths, race conditions, resource leaks, incorrect async usage, security issues (injection, path traversal, secrets, missing authorization), and behaviour that contradicts a nearby comment or type.

Skip style, formatting, and naming preferences. If the diff is clean, say so in one line rather than inventing findings.`,
  ]
    .filter(Boolean)
    .join("\n\n");
}

export function refactorPrompt(
  path: string,
  code: string,
  instruction: string,
  hits: SearchHit[],
): string {
  const related = renderContext(hits.filter((h) => h.file !== path));
  return [
    related,
    `<file path="${path}">\n${code}\n</file>`,
    `${instruction}

Return the complete revised file in a single fenced code block, then a short bullet list of what changed and why. Preserve existing behaviour unless the instruction says otherwise, and keep the repository's conventions.`,
  ]
    .filter(Boolean)
    .join("\n\n");
}

/**
 * Build a retrieval query from a natural-language request. Strips filler words
 * that BM25 would otherwise waste weight on.
 */
const STOPWORDS = new Set([
  "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "how",
  "what", "where", "when", "why", "who", "which", "can", "could", "should",
  "would", "i", "we", "you", "it", "this", "that", "these", "those", "and",
  "or", "but", "if", "then", "than", "to", "of", "in", "on", "for", "with",
  "about", "into", "my", "our", "your", "me", "us", "please", "help",
]);

export function retrievalQuery(text: string): string {
  const words = text.split(/[^A-Za-z0-9_$]+/).filter((w) => w.length > 1);
  const kept = words.filter((w) => !STOPWORDS.has(w.toLowerCase()));
  // If stripping leaves nothing useful, fall back to the original text.
  return kept.length >= 2 ? kept.join(" ") : text;
}
