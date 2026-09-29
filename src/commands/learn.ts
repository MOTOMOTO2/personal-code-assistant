import { Bundle } from "../core/bundle.js";
import { loadCorpus, trainProject } from "../core/train.js";
import { trainNeural } from "../core/trainNeural.js";
import { startWatcher } from "../core/watch.js";
import { loadLessons } from "../chat/memory.js";
import { ParsedArgs, flagBool, flagList, flagNumber, flagString } from "../util/args.js";
import { color, info, ok, step } from "../util/log.js";
import { workspace } from "../util/paths.js";

/**
 * `pca learn` - the training loop, including everything the assistant has been
 * told since the last run.
 *
 * `pca train` builds models from your source. `pca learn` does that *and* folds
 * in the chat memory, which is the part that compounds: each answer Claude
 * gives about this codebase becomes corpus for the n-gram, text for the
 * embeddings, and a chunk in the retrieval index, so the local half can field
 * more of the next session on its own.
 */

export async function cmdLearn(args: ParsedArgs): Promise<number> {
  const ws = workspace(flagString(args, "root") ?? args.positional[0]);
  const bundle = new Bundle(ws);
  const minutes = flagNumber(args, "minutes", 0);
  const valFraction = flagNumber(args, "val-fraction", 0.08);
  const extraExtensions = flagList(args, "ext");
  const trainOpts = {
    valFraction,
    extraExtensions,
    includeLessons: !flagBool(args, "no-memory"),
    includeSources: flagBool(args, "sources", true),
    ...(args.flags.has("vocab") ? { vocabSize: flagNumber(args, "vocab", 8192) } : {}),
    ...(args.flags.has("order") ? { order: flagNumber(args, "order", 4) } : {}),
  };

  const before = bundle.manifest;
  const memory = loadLessons(bundle);
  if (memory.size > 0) {
    info(
      `${color.bold(String(memory.size))} remembered answers` +
        `${before ? ` · last trained ${before.trainedAt.slice(0, 16).replace("T", " ")}` : ""}`,
    );
  }

  if (flagBool(args, "watch")) {
    return watchForever(ws, trainOpts, minutes, flagNumber(args, "debounce", 1500));
  }

  const report = trainProject(bundle, trainOpts);
  info("");
  info(`  ${color.dim("files")}      ${report.files}`);
  info(`  ${color.dim("memory")}     ${report.lessons} answers folded in`);
  info(`  ${color.dim("corpus")}     ${report.corpusTokens.toLocaleString()} tokens`);
  if (report.sourceTokens > 0) {
    info(`  ${color.dim("sources")}    ${report.sourceTokens.toLocaleString()} tokens from ${report.sources.length}`);
  }
  info(`  ${color.dim("chunks")}     ${report.chunks.toLocaleString()}`);
  if (before) {
    const delta = report.corpusTokens - before.corpusTokens;
    if (delta !== 0) {
      info(`  ${color.dim("change")}     ${delta > 0 ? "+" : ""}${delta.toLocaleString()} tokens since last train`);
    }
  }
  info("");

  if (minutes > 0) {
    step(`Continuing transformer training for ${minutes} minute${minutes === 1 ? "" : "s"}`);
    const corpus = loadCorpus(bundle, valFraction, extraExtensions);
    trainNeural(bundle, corpus, {
      minutes,
      resume: !flagBool(args, "restart"),
      batchSize: flagNumber(args, "batch", 4),
      lr: flagNumber(args, "lr", 3e-3),
    });
  } else {
    info(
      color.dim(
        "Tip: `pca learn --minutes 10` also continues transformer training; " +
          "`pca learn --watch` retrains on every save.",
      ),
    );
  }

  ok("Learned. `pca chat` to use it, `pca eval` to measure it.");
  return 0;
}

async function watchForever(
  ws: ReturnType<typeof workspace>,
  trainOpts: Parameters<typeof trainProject>[1],
  neuralMinutes: number,
  debounceMs: number,
): Promise<number> {
  const watcher = startWatcher(ws, {
    debounceMs,
    train: trainOpts,
    neuralMinutes,
    onError: () => {
      /* already reported by the watcher */
    },
  });
  // An initial pass, so the models are current before the first edit lands.
  watcher.trigger("startup");
  const stop = () => watcher.close();
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  await watcher.done;
  ok(`Stopped after ${watcher.cycles} retrain${watcher.cycles === 1 ? "" : "s"}.`);
  return 0;
}
