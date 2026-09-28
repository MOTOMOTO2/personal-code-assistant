/**
 * Hyper-parameter sweep for the completion engine, scored on held-out files.
 *
 * The two knobs that matter are how many tokens a suggestion may grow to, and
 * how much n-gram evidence is required to keep growing it. They trade off
 * directly: longer suggestions save more typing when right and waste more time
 * when wrong. This picks the operating point from measurement rather than taste.
 *
 *   node bench/sweep.mjs [--samples 200]
 *
 * Reported per configuration:
 *   correct   fraction of suggestions whose entire offered text was right
 *   chars     mean characters correctly predicted per call (the utility)
 *   offered   mean characters offered per call
 */
import { Bundle } from "../dist/core/bundle.js";
import { evalLineCompletion } from "../dist/core/evaluate.js";
import { isValidationFile } from "../dist/core/train.js";
import { readSource, walkSources } from "../dist/index/walker.js";
import { setQuiet } from "../dist/util/log.js";
import { workspace } from "../dist/util/paths.js";

setQuiet(true);
const args = process.argv.slice(2);
const samples = Number(args[args.indexOf("--samples") + 1]) || 200;
const valFraction = 0.08;

const ws = workspace();
const bundle = new Bundle(ws);
if (!bundle.isTrained) {
  console.error("Not trained. Run `node dist/cli.js train` first.");
  process.exit(1);
}

const texts = [];
for (const file of walkSources(ws.root)) {
  if (!isValidationFile(file.rel, valFraction)) continue;
  try {
    texts.push({ path: file.rel, text: readSource(file) });
  } catch {
    /* skip */
  }
}
if (texts.length === 0) {
  console.error("No held-out files. Lower --val-fraction at train time or add files.");
  process.exit(1);
}

const order = bundle.ngram.order;
const configs = [];
for (const maxTokens of [3, 5, 8, 12]) {
  for (const minEvidenceOrder of [2, 3, Math.min(4, order)]) {
    if (configs.some((c) => c.maxTokens === maxTokens && c.minEvidenceOrder === minEvidenceOrder)) {
      continue;
    }
    configs.push({ maxTokens, minEvidenceOrder });
  }
}

console.log("");
console.log(`  held-out files: ${texts.length}   samples per config: ${samples}   n-gram order: ${order}`);
console.log("");
console.log("  maxTok  minEvid   correct   exact-line    chars  offered   p95 ms");
console.log("  " + "-".repeat(64));

const results = [];
for (const config of configs) {
  const m = evalLineCompletion(bundle, texts, { samples, seed: 4242, ...config });
  results.push({ config, m });
  console.log(
    "  " +
      String(config.maxTokens).padStart(6) +
      String(config.minEvidenceOrder).padStart(9) +
      `${(m.prefixMatch * 100).toFixed(1)}%`.padStart(10) +
      `${(m.exactLine * 100).toFixed(1)}%`.padStart(13) +
      m.meanCorrectChars.toFixed(2).padStart(9) +
      m.meanOfferedChars.toFixed(1).padStart(9) +
      m.p95LatencyMs.toFixed(2).padStart(9),
  );
}

// Rank by mean correct characters - the expected typing actually saved - and
// report the runner-up by trustworthiness so the trade-off stays visible.
const byChars = [...results].sort((a, b) => b.m.meanCorrectChars - a.m.meanCorrectChars);
const byTrust = [...results].sort((a, b) => b.m.prefixMatch - a.m.prefixMatch);
console.log("");
console.log(
  `  most characters saved: maxTokens=${byChars[0].config.maxTokens} minEvidenceOrder=${byChars[0].config.minEvidenceOrder}` +
    ` (${byChars[0].m.meanCorrectChars.toFixed(2)} chars, ${(byChars[0].m.prefixMatch * 100).toFixed(1)}% fully correct)`,
);
console.log(
  `  most trustworthy:      maxTokens=${byTrust[0].config.maxTokens} minEvidenceOrder=${byTrust[0].config.minEvidenceOrder}` +
    ` (${(byTrust[0].m.prefixMatch * 100).toFixed(1)}% fully correct, ${byTrust[0].m.meanCorrectChars.toFixed(2)} chars)`,
);
console.log("");
