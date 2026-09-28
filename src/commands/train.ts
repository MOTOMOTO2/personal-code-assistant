import fs from "node:fs";
import { Bundle } from "../core/bundle.js";
import { evalLineCompletion, evalNeuralAndHybrid, evalNgram } from "../core/evaluate.js";
import { loadCorpus, trainProject } from "../core/train.js";
import { trainNeural } from "../core/trainNeural.js";
import { isValidationFile } from "../core/train.js";
import { readSource, walkSources } from "../index/walker.js";
import { ParsedArgs, flagBool, flagList, flagNumber } from "../util/args.js";
import { color, info, ok, step } from "../util/log.js";
import { humanBytes } from "../util/io.js";
import { workspace } from "../util/paths.js";

export async function cmdTrain(args: ParsedArgs): Promise<number> {
  const ws = workspace(args.positional[0]);
  const bundle = new Bundle(ws);
  const neuralOnly = flagBool(args, "neural-only");
  const wantNeural = flagBool(args, "neural") || neuralOnly;
  const valFraction = flagNumber(args, "val-fraction", 0.08);
  const extraExtensions = flagList(args, "ext");

  if (!neuralOnly) {
    const report = trainProject(bundle, {
      vocabSize: flagNumber(args, "vocab", 8192),
      order: flagNumber(args, "order", 4),
      maxTokens: flagNumber(args, "max-tokens", 2_000_000),
      embeddingDim: flagNumber(args, "embed-dim", 64),
      embeddingEpochs: flagNumber(args, "embed-epochs", 4),
      valFraction,
      extraExtensions,
      skipIndex: flagBool(args, "skip-index"),
    });
    info("");
    info(`  ${color.dim("files")}      ${report.files} (${humanBytes(report.bytes)})`);
    info(`  ${color.dim("corpus")}     ${report.corpusTokens.toLocaleString()} tokens`);
    info(`  ${color.dim("vocabulary")} ${report.vocabSize}`);
    info(
      `  ${color.dim("n-gram")}     order ${report.ngram.order}, ${report.ngram.ngrams.toLocaleString()} n-grams`,
    );
    info(`  ${color.dim("chunks")}     ${report.chunks.toLocaleString()}`);
    info(`  ${color.dim("held out")}   ${report.valFiles} files for evaluation`);
    info("");
  }

  if (wantNeural) {
    step("Loading corpus for neural training");
    const corpus = loadCorpus(bundle, valFraction, extraExtensions);
    info(`train ${corpus.train.length} files, validation ${corpus.val.length} files`);
    trainNeural(bundle, corpus, {
      minutes: flagNumber(args, "minutes", 5),
      batchSize: flagNumber(args, "batch", 4),
      lr: flagNumber(args, "lr", 3e-3),
      resume: flagBool(args, "resume"),
      config: {
        ...(args.flags.has("d-model") ? { dModel: flagNumber(args, "d-model", 128) } : {}),
        ...(args.flags.has("layers") ? { nLayers: flagNumber(args, "layers", 3) } : {}),
        ...(args.flags.has("heads") ? { nHeads: flagNumber(args, "heads", 4) } : {}),
        ...(args.flags.has("ctx") ? { blockSize: flagNumber(args, "ctx", 96) } : {}),
      },
    });
  } else if (!neuralOnly) {
    info(
      color.dim(
        "Tip: `pca train --neural --minutes 10` also trains the transformer for deeper context.",
      ),
    );
  }

  ok("Done. Try `pca complete --file <path> --line <n>` or `pca eval`.");
  return 0;
}

export async function cmdEval(args: ParsedArgs): Promise<number> {
  const ws = workspace(args.positional[0]);
  const bundle = new Bundle(ws);
  if (!bundle.isTrained) {
    throw new Error("Nothing trained yet - run `pca train` first.");
  }
  const valFraction = flagNumber(args, "val-fraction", 0.08);
  const useNeural = flagBool(args, "neural", true) && bundle.neural !== null;

  step("Loading held-out files");
  const corpus = loadCorpus(bundle, valFraction);
  const valDocs = corpus.val.length > 0 ? corpus.val : corpus.train.slice(0, 5);
  info(`${corpus.valPaths.length} held-out files (${corpus.trainPaths.length} used for training)`);

  step("Next-token metrics: n-gram");
  const ngram = evalNgram(bundle, valDocs, flagNumber(args, "tokens", 20_000));

  let neural = null;
  let hybrid = null;
  if (useNeural) {
    step("Next-token metrics: transformer + hybrid");
    const both = evalNeuralAndHybrid(
      bundle,
      valDocs,
      flagNumber(args, "neural-tokens", 400),
      flagNumber(args, "blend", 0.5),
    );
    if (both) {
      neural = both.neural;
      hybrid = both.hybrid;
    }
  }

  step("End-to-end line completion");
  const texts: Array<{ path: string; text: string }> = [];
  for (const file of walkSources(ws.root)) {
    if (!isValidationFile(file.rel, valFraction)) continue;
    try {
      texts.push({ path: file.rel, text: readSource(file) });
    } catch {
      /* unreadable file - skip */
    }
  }
  const line = evalLineCompletion(bundle, texts.length > 0 ? texts : fallbackTexts(ws.root), {
    samples: flagNumber(args, "samples", 200),
    neural: flagBool(args, "neural-line"),
  });

  const rows: Array<[string, string, string, string]> = [
    ["model", "perplexity", "top-1", "top-5"],
    [
      "n-gram",
      fmt(ngram.perplexity),
      pct(ngram.top1),
      pct(ngram.top5),
    ],
  ];
  if (neural) rows.push(["transformer", fmt(neural.perplexity), pct(neural.top1), pct(neural.top5)]);
  if (hybrid) rows.push(["hybrid", fmt(hybrid.perplexity), pct(hybrid.top1), pct(hybrid.top5)]);

  info("");
  info(color.bold("Next-token prediction on held-out code"));
  printTable(rows);
  if (neural) {
    info(
      color.dim(
        `  n-gram evaluated on ${ngram.tokens.toLocaleString()} tokens, ` +
          `neural/hybrid on ${neural.tokens.toLocaleString()} (each needs a forward pass)`,
      ),
    );
  }

  info("");
  info(color.bold("Line completion from a real mid-line cursor"));
  info(`  ${color.dim("samples")}              ${line.samples}`);
  info(`  ${color.dim("suggestion correct")}   ${pct(line.prefixMatch)}  (offered text was a correct prefix)`);
  info(`  ${color.dim("whole line correct")}   ${pct(line.exactLine)}`);
  info(`  ${color.dim("chars saved / call")}   ${line.meanCorrectChars.toFixed(1)} of ${line.meanOfferedChars.toFixed(1)} offered`);
  info(`  ${color.dim("latency mean / p95")}   ${line.meanLatencyMs.toFixed(2)}ms / ${line.p95LatencyMs.toFixed(2)}ms`);
  info("");

  if (flagBool(args, "json")) {
    process.stdout.write(`${JSON.stringify({ ngram, neural, hybrid, line }, null, 2)}\n`);
  }
  return 0;
}

function fallbackTexts(root: string): Array<{ path: string; text: string }> {
  // Every file landed in the training split (tiny repo): evaluate on training
  // data and say so, rather than reporting nothing.
  const out: Array<{ path: string; text: string }> = [];
  for (const file of walkSources(root).slice(0, 20)) {
    try {
      out.push({ path: file.rel, text: fs.readFileSync(file.abs, "utf8") });
    } catch {
      /* skip */
    }
  }
  return out;
}

function fmt(v: number): string {
  return Number.isFinite(v) ? v.toFixed(1) : "n/a";
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function printTable(rows: Array<[string, string, string, string]>): void {
  const widths = [0, 1, 2, 3].map((c) => Math.max(...rows.map((r) => r[c].length)));
  rows.forEach((row, i) => {
    const line = row.map((cell, c) => (c === 0 ? cell.padEnd(widths[c]) : cell.padStart(widths[c] + 3))).join("");
    info(i === 0 ? `  ${color.dim(line)}` : `  ${line}`);
  });
}
