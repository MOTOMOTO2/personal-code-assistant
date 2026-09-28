import fs from "node:fs";
import path from "node:path";
import { Bundle } from "./bundle.js";
import { TrainOptions, TrainReport, loadCorpus, trainProject } from "./train.js";
import { trainNeural } from "./trainNeural.js";
import { ALWAYS_SKIP_DIRS, CODE_EXTENSIONS } from "../index/walker.js";
import { Workspace } from "../util/paths.js";
import { color, fmtDuration, info, isQuiet, ok, setQuiet, warn } from "../util/log.js";

/**
 * Continuous training.
 *
 * The fast models train in seconds, which changes what training *is*: instead
 * of a thing you remember to run, it becomes a thing that happens whenever you
 * save a file. A model that is three commits out of date cannot complete the
 * function you just wrote, and that function is exactly what you want completed.
 *
 * Changes are debounced rather than handled one-by-one - a rebase or a
 * find-and-replace fires hundreds of events, and retraining once at the end is
 * both cheaper and more correct.
 */

export interface WatchOptions {
  /** Quiet period before retraining, in milliseconds. */
  debounceMs?: number;
  train?: TrainOptions;
  /** Stop after this many retrain cycles. 0 means run until closed. */
  maxCycles?: number;
  /** Continue transformer training for this long after each retrain. */
  neuralMinutes?: number;
  onCycle?: (report: TrainReport, reason: string) => void;
  onError?: (err: Error) => void;
}

export interface Watcher {
  /** Resolves when the watcher stops (cycle limit reached, or `close()`). */
  done: Promise<void>;
  close(): void;
  /** Force a retrain now, bypassing the debounce. Used by `/train` and tests. */
  trigger(reason: string): void;
  cycles: number;
}

/** Whether a path change should trigger a retrain. */
export function isTrainableChange(relPath: string): boolean {
  const parts = relPath.split(/[\\/]+/);
  for (const part of parts.slice(0, -1)) {
    if (ALWAYS_SKIP_DIRS.has(part)) return false;
  }
  const name = parts[parts.length - 1] ?? "";
  if (name.startsWith(".") && !name.startsWith(".env")) return false;
  return CODE_EXTENSIONS.has(path.extname(name).toLowerCase());
}

export function startWatcher(ws: Workspace, opts: WatchOptions = {}): Watcher {
  const bundle = new Bundle(ws);
  const debounceMs = opts.debounceMs ?? 1500;
  const maxCycles = opts.maxCycles ?? 0;

  let timer: NodeJS.Timeout | null = null;
  let pending = new Set<string>();
  let closed = false;
  let running = false;
  let resolveDone: () => void = () => {};
  const done = new Promise<void>((resolve) => {
    resolveDone = resolve;
  });

  const watcher = fs.watch(ws.root, { recursive: true }, (_event, filename) => {
    if (closed || filename === null) return;
    const rel = String(filename).split(path.sep).join("/");
    if (!isTrainableChange(rel)) return;
    pending.add(rel);
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => runCycle(describe(pending)), debounceMs);
    timer.unref?.();
  });

  watcher.on("error", (err: Error) => {
    opts.onError?.(err);
    warn(`watch error: ${err.message}`);
  });

  const state: Watcher = {
    done,
    cycles: 0,
    close() {
      if (closed) return;
      closed = true;
      if (timer) clearTimeout(timer);
      watcher.close();
      resolveDone();
    },
    trigger(reason: string) {
      if (timer) clearTimeout(timer);
      runCycle(reason);
    },
  };

  function runCycle(reason: string): void {
    if (closed || running) return;
    running = true;
    pending = new Set();
    const started = performance.now();
    // The first pass is worth watching - it is the one that takes seconds on a
    // cold repo. After that, a full training log on every file save is noise,
    // so only the one-line summary survives.
    const wasQuiet = isQuiet();
    if (state.cycles > 0) setQuiet(true);
    try {
      const report = trainProject(bundle, opts.train ?? {});
      setQuiet(wasQuiet);
      state.cycles++;
      ok(
        `retrained: ${report.files} files, ${report.corpusTokens.toLocaleString()} tokens, ` +
          `${report.lessons} remembered answers ${color.dim(`(${fmtDuration(performance.now() - started)}, ${reason})`)}`,
      );
      if (opts.neuralMinutes && opts.neuralMinutes > 0) {
        const corpus = loadCorpus(bundle, opts.train?.valFraction ?? 0.08);
        trainNeural(bundle, corpus, { minutes: opts.neuralMinutes, resume: true });
      }
      opts.onCycle?.(report, reason);
    } catch (err) {
      opts.onError?.(err as Error);
      warn(`retrain failed: ${(err as Error).message}`);
    } finally {
      setQuiet(wasQuiet);
      running = false;
    }
    if (maxCycles > 0 && state.cycles >= maxCycles) state.close();
  }

  info(
    color.dim(
      `watching ${ws.root} - retraining ${debounceMs}ms after each change` +
        `${opts.neuralMinutes ? `, plus ${opts.neuralMinutes}min of transformer training per cycle` : ""}`,
    ),
  );
  return state;
}

function describe(changed: Set<string>): string {
  const list = [...changed];
  if (list.length === 0) return "manual";
  if (list.length === 1) return list[0];
  return `${list[0]} +${list.length - 1} more`;
}
