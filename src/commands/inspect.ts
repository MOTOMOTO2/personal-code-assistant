import fs from "node:fs";
import { Bundle } from "../core/bundle.js";
import { loadSources } from "../core/sources.js";
import { normalize } from "../ml/embeddings.js";
import { retrievalQuery } from "../llm/prompts.js";
import { hasCredentials } from "../llm/claude.js";
import { ParsedArgs, flagBool, flagNumber } from "../util/args.js";
import { color, info, ok, warn } from "../util/log.js";
import { fileSizeSafe, humanBytes } from "../util/io.js";
import { ARTIFACTS } from "../core/bundle.js";
import { workspace } from "../util/paths.js";
import { walkSources } from "../index/walker.js";
import { loadLessons } from "../chat/memory.js";
import { ThreadStore } from "../chat/threads.js";
import { DEFAULT_PORT, lanAddresses } from "../server/daemon.js";

export async function cmdSearch(args: ParsedArgs): Promise<number> {
  const ws = workspace();
  const bundle = new Bundle(ws);
  const query = args.positional.join(" ");
  if (query.length === 0) throw new Error('Usage: pca search "how do we validate tokens"');
  const index = bundle.index;
  if (!index) throw new Error("No index - run `pca train` first.");

  const embeddings = bundle.embeddings;
  const q = retrievalQuery(query);
  const queryVector = embeddings
    ? normalize(embeddings.sifEmbed(bundle.tokenizer.encode(q)))
    : null;
  const hits = index.search(q, { limit: flagNumber(args, "limit", 8), queryVector });

  if (flagBool(args, "json")) {
    process.stdout.write(
      `${JSON.stringify(
        hits.map((h) => ({
          file: h.file,
          startLine: h.chunk.startLine,
          endLine: h.chunk.endLine,
          symbol: h.chunk.symbol,
          score: h.score,
        })),
        null,
        2,
      )}\n`,
    );
    return 0;
  }

  if (hits.length === 0) {
    info(color.dim("no matches"));
    return 0;
  }

  const showCode = flagBool(args, "code", true);
  const contextLines = flagNumber(args, "lines", 6);
  info("");
  for (const hit of hits) {
    const where = `${hit.file}:${hit.chunk.startLine}`;
    const sym = hit.chunk.symbol ? color.magenta(` ${hit.chunk.symbol}`) : "";
    const how = flagBool(args, "explain")
      ? color.dim(
          ` [bm25 #${hit.lexicalRank ?? "-"} · vector #${hit.vectorRank ?? "-"} · rrf ${hit.score.toFixed(4)}]`,
        )
      : "";
    info(`${color.cyan(where)}${sym}${how}`);
    if (showCode) {
      const lines = hit.chunk.text.split("\n").slice(0, contextLines);
      for (let i = 0; i < lines.length; i++) {
        info(`  ${color.dim(String(hit.chunk.startLine + i).padStart(5))} ${lines[i]}`);
      }
      if (hit.chunk.text.split("\n").length > contextLines) info(color.dim("        …"));
    }
    info("");
  }
  if (!index.hasVectors) {
    info(color.dim("(lexical only - train embeddings with `pca train` to enable semantic search)"));
  }
  return 0;
}

/** Nearest neighbours in the locally-trained embedding space. */
export async function cmdNeighbors(args: ParsedArgs): Promise<number> {
  const ws = workspace();
  const bundle = new Bundle(ws);
  const word = args.positional[0];
  if (!word) throw new Error("Usage: pca neighbors <identifier>");
  const embeddings = bundle.embeddings;
  if (!embeddings) throw new Error("No embeddings - run `pca train` first.");
  const tokenizer = bundle.tokenizer;
  const id = tokenizer.idOf(word);
  if (id === undefined) {
    throw new Error(
      `"${word}" is not in the learned vocabulary (it may be too rare in this codebase).`,
    );
  }
  const neighbors = embeddings.nearest(id, flagNumber(args, "limit", 15));
  info("");
  info(`Tokens the model considers similar to ${color.bold(word)}:`);
  info("");
  for (const n of neighbors) {
    const piece = tokenizer.pieceOf(n.id);
    if (piece.trim().length === 0) continue;
    const bar = "█".repeat(Math.max(1, Math.round(Math.max(0, n.score) * 24)));
    info(`  ${piece.padEnd(24)} ${color.cyan(bar)} ${n.score.toFixed(3)}`);
  }
  info("");
  return 0;
}

export async function cmdStats(args: ParsedArgs): Promise<number> {
  const ws = workspace();
  const bundle = new Bundle(ws);
  const manifest = bundle.manifest;

  info("");
  info(color.bold(`personal-code-assistant  ${color.dim(ws.root)}`));
  info("");

  if (!manifest) {
    warn("Not trained yet. Run `pca train`.");
    return 0;
  }

  info(color.bold("Corpus"));
  info(`  trained            ${new Date(manifest.trainedAt).toLocaleString()}`);
  info(`  files              ${manifest.corpusFiles.toLocaleString()}`);
  info(`  tokens             ${manifest.corpusTokens.toLocaleString()}`);
  info(`  vocabulary         ${manifest.vocabSize.toLocaleString()}`);
  info("");

  info(color.bold("Models"));
  info(`  n-gram             order ${manifest.ngramOrder}  ${sizeOf(bundle, "ngram")}`);
  if (manifest.embeddingDim > 0) {
    info(`  embeddings         ${manifest.embeddingDim}-dim  ${sizeOf(bundle, "embeddings")}`);
  } else {
    info(`  embeddings         ${color.dim("not trained")}`);
  }
  if (manifest.neural) {
    const n = manifest.neural;
    info(
      `  transformer        ${(n.params / 1e6).toFixed(2)}M params · ${n.steps.toLocaleString()} steps · ` +
        `val ppl ${n.valPerplexity.toFixed(1)} · ${(n.trainedSeconds / 60).toFixed(1)} min trained  ${sizeOf(bundle, "neural")}`,
    );
  } else {
    info(`  transformer        ${color.dim("not trained - `pca train --neural --minutes 10`")}`);
  }
  info("");

  const index = bundle.index;
  if (index) {
    const s = index.stats();
    info(color.bold("Retrieval index"));
    info(`  chunks             ${s.chunks.toLocaleString()}`);
    info(`  distinct terms     ${s.terms.toLocaleString()}`);
    info(`  semantic vectors   ${s.hasVectors ? color.green("yes") : color.dim("no")}`);
    info("");
  }

  const r = bundle.reranker;
  info(color.bold("Learned ranking"));
  info(`  feedback events    ${r.accepted} accepted, ${r.rejected} dismissed (${r.updates} updates)`);
  if (flagBool(args, "weights")) {
    info("");
    for (const { feature, weight } of r.describe()) {
      const mag = Math.min(24, Math.round(Math.abs(weight) * 12));
      const bar = (weight >= 0 ? color.green : color.red)("█".repeat(Math.max(1, mag)));
      info(`  ${feature.padEnd(20)} ${weight >= 0 ? " " : "-"}${Math.abs(weight).toFixed(3)} ${bar}`);
    }
  } else {
    info(color.dim("  (pass --weights to see what it learned about your preferences)"));
  }
  info("");

  const memory = loadLessons(bundle);
  const router = bundle.router;
  info(color.bold("Memory and routing"));
  info(`  answers remembered ${memory.size}`);
  info(
    `  route decisions    ${router.toLocal} local, ${router.toClaude} claude (${router.updates} corrections)`,
  );
  if (flagBool(args, "memory") && memory.size > 0) {
    info("");
    for (const lesson of memory.lessons.slice(-8)) {
      info(
        `  ${color.dim(lesson.at.slice(0, 10))} ${lesson.question.replace(/\s+/g, " ").slice(0, 64)}`,
      );
    }
  }
  if (flagBool(args, "router")) {
    info("");
    for (const { feature, weight } of router.describe()) {
      const mag = Math.min(24, Math.round(Math.abs(weight) * 6));
      // Negative means "the local models can take this one".
      const bar = (weight >= 0 ? color.magenta : color.green)("█".repeat(Math.max(1, mag)));
      info(`  ${feature.padEnd(20)} ${weight >= 0 ? " " : "-"}${Math.abs(weight).toFixed(3)} ${bar}`);
    }
    info(color.dim("  magenta pushes a turn to Claude, green keeps it local"));
  } else {
    info(color.dim("  (--router for the learned weights, --memory for recent answers)"));
  }
  info("");

  const style = bundle.style;
  info(color.bold("Measured style"));
  info(
    `  ${style.indentKind === "tabs" ? "tabs" : `${style.indentWidth}-space indent`} · ` +
      `${style.quote} quotes · ${style.semicolons ? "semicolons" : "no semicolons"} · ` +
      `${style.naming} · ~${style.lineWidth} cols · ${style.commentDensity} comments/100 lines`,
  );
  if (style.languages.length > 0) {
    info(`  languages          ${style.languages.map((l) => `${l.ext} (${l.files})`).join(", ")}`);
  }
  info("");
  return 0;
}

export async function cmdDoctor(): Promise<number> {
  const ws = workspace();
  const bundle = new Bundle(ws);
  info("");
  info(color.bold("Environment"));
  info(`  node               ${process.version}`);
  info(`  platform           ${process.platform} ${process.arch}`);
  info(`  project root       ${ws.root}`);
  info(`  state dir          ${ws.stateDir}`);
  info("");

  info(color.bold("Checks"));
  const files = walkSources(ws.root);
  check(files.length > 0, `${files.length} indexable source files found`, "no source files found - is this a code project?");
  check(bundle.isTrained, "local models trained", "not trained yet - run `pca train`");
  if (bundle.isTrained) {
    check(bundle.index !== null, "retrieval index present", "no index - rerun `pca train`");
    check(bundle.embeddings !== null, "embeddings present", "no embeddings - rerun `pca train`");
    check(
      bundle.neural !== null,
      "transformer checkpoint present",
      "no transformer - optional, run `pca train --neural --minutes 10`",
      true,
    );
  }
  check(
    hasCredentials(),
    "Anthropic credentials found (Claude commands enabled)",
    "no ANTHROPIC_API_KEY - `ask`/`review`/`explain` disabled, everything local still works",
    true,
  );
  const memory = loadLessons(bundle);
  check(
    memory.size > 0,
    `${memory.size} answer${memory.size === 1 ? "" : "s"} in memory ` +
      "(folded into the models by `pca learn`)",
    "memory empty - `pca chat` fills it as you use the assistant",
    true,
  );
  const sources = loadSources(ws);
  check(
    sources.length > 0,
    `${sources.length} extra corpus source${sources.length === 1 ? "" : "s"}: ` +
      sources.map((s) => `@${s.name}`).join(", "),
    "training on this project only - `pca corpus add-all <dir>` feeds it your other repos",
    true,
  );
  for (const source of sources) {
    if (!fs.existsSync(source.path)) {
      warn(`source @${source.name} points at a directory that is gone: ${source.path}`);
    }
  }

  const threads = new ThreadStore(ws).list();
  check(
    threads.length > 0,
    `${threads.length} conversation${threads.length === 1 ? "" : "s"} kept ` +
      "(`pca threads`, `pca chat --resume`)",
    "no conversations yet - `pca chat` starts one, `pca phone` opens it on your phone",
    true,
  );
  const addresses = lanAddresses();
  if (addresses.length > 0) {
    info(
      color.dim(
        `  reachable from your phone at http://${addresses[0]}:${DEFAULT_PORT} once \`pca phone\` is running`,
      ),
    );
  }
  info("");
  return 0;
}

function check(pass: boolean, good: string, bad: string, optional = false): void {
  if (pass) ok(good);
  else if (optional) warn(bad);
  else warn(bad);
}

function sizeOf(bundle: Bundle, name: keyof typeof ARTIFACTS): string {
  const size = fileSizeSafe(bundle.path(name));
  return size > 0 ? color.dim(`(${humanBytes(size)})`) : "";
}
