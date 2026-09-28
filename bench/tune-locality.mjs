/**
 * Sweep the reranker's locality weight against held-out line completion.
 * The locality feature trades off directly against the language model's own
 * ranking, and the right balance is an empirical question, not a taste question.
 *
 *   node bench/tune-locality.mjs [--samples 200]
 */
import { Bundle } from "../dist/core/bundle.js";
import { evalLineCompletion } from "../dist/core/evaluate.js";
import { isValidationFile } from "../dist/core/train.js";
import { featureIndex } from "../dist/ml/reranker.js";
import { readSource, walkSources } from "../dist/index/walker.js";
import { setQuiet } from "../dist/util/log.js";
import { workspace } from "../dist/util/paths.js";

setQuiet(true);
const args = process.argv.slice(2);
const samples = Number(args[args.indexOf("--samples") + 1]) || 200;

const ws = workspace();
const bundle = new Bundle(ws);
const texts = [];
for (const file of walkSources(ws.root)) {
  if (!isValidationFile(file.rel, 0.08)) continue;
  try {
    texts.push({ path: file.rel, text: readSource(file) });
  } catch {}
}

const idx = featureIndex("localFileHits");
const rankIdx = featureIndex("ngramRank");
console.log("");
console.log("  locality  rank    correct    chars  offered");
console.log("  " + "-".repeat(46));
for (const locality of [0.0, 0.2, 0.3, 0.5, 0.8, 1.2]) {
  for (const rank of [0.15, 0.4]) {
    bundle.reranker.weights[idx] = locality;
    bundle.reranker.weights[rankIdx] = rank;
    const m = evalLineCompletion(bundle, texts, { samples, seed: 4242 });
    console.log(
      "  " +
        locality.toFixed(2).padStart(8) +
        rank.toFixed(2).padStart(6) +
        `${(m.prefixMatch * 100).toFixed(1)}%`.padStart(11) +
        m.meanCorrectChars.toFixed(2).padStart(9) +
        m.meanOfferedChars.toFixed(1).padStart(9),
    );
  }
}
console.log("");
