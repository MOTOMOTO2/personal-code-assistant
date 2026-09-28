# personal-code-assistant

A coding assistant you train on your own code, and a frontier model for the hard
questions. Machine learning models implemented from scratch, running locally at
interactive speed, with Claude wired in for the work that genuinely needs it.

```
$ pca train
✓ Found 33 source files (212KB)
✓ Vocabulary: 2287 tokens
✓ Corpus: 97,200 tokens
✓ n-gram: 88,085 n-grams across 42,837 contexts
✓ Embeddings: 64-dim vectors for 125 chunks
✓ Trained in 2.1s

$ pca complete --file src/ml/params.ts --line 118
  › clipGradNorm(maxNorm: number): num▏
  ▸ ber {                score 2.841  lm -0.42
```

---

## Quickstart: using it right now

This repository is already built and already trained on itself, so the local
half works immediately. The `pca` command is only on your PATH after `npm link`;
until then, invoke it directly:

```bash
node dist/cli.js doctor                       # confirm setup
node dist/cli.js complete "const user = await db."
node dist/cli.js search "where are gradients clipped"
node dist/cli.js neighbors tokenizer
node dist/cli.js stats --weights
```

`npm run pca -- <args>` does the same thing via the package script. To get the
short `pca` form, run `npm link` once (this writes a symlink into your global
npm prefix), after which every example below works verbatim.

If `dist/` is missing or stale, run `npm run build` first. If `.pca/` is missing,
run `node dist/cli.js train` — a few seconds.

**The Claude-backed commands (`ask`, `explain`, `review`, `refactor`) need
`ANTHROPIC_API_KEY` and are disabled without it.** Everything above is local,
offline, and needs no key. `doctor` tells you which half is live.

To use it on a *different* project, `cd` there and run `pca train` — models are
stored per-project in `.pca/`, so nothing is shared between codebases.

---

## Read this first: what this is, and what it is not

You asked for something "amazing with coding like you are". Two honest points up
front, because they shaped every design decision here.

**A model you can train on a laptop will not match a frontier model at writing
code.** Claude is on the order of a trillion parameters trained on a
significant fraction of public code. A model that trains in minutes on your
machine is ~1.6 million parameters trained on your repository. Anyone who tells
you otherwise is selling something. So this project does not pretend: it puts a
local model where a local model genuinely wins, and calls Claude where Claude
genuinely wins.

**Where a local model genuinely wins is latency and specificity.** Inline
completion has a budget of a few milliseconds — no network call can meet it —
and the most useful completion is usually a symbol *you* defined an hour ago,
which no general model has ever seen. That is a real, measurable job, and the
local models here do it: **2.3 ms mean, 4.7 ms p95** on warm inference.

So the split is:

| Job | Runs on | Why |
|---|---|---|
| Inline completion, every keystroke | Local n-gram + reranker | Must be <10 ms; needs *your* symbols |
| Semantic code search | Local embeddings + BM25 | Instant, offline, private |
| "Why does this break?", review, refactor | Claude Opus 5 | Needs real reasoning |
| Learning your preferences | Local online ranker | Trains on your accept/reject, instantly |

The local half works with no API key and no network.

---

## What is actually learned from your code

Five models, all implemented from scratch in TypeScript with no ML dependencies:

**1. A lossless code tokenizer** (`src/tokenizer/`) — learns the identifiers,
keywords and operators your codebase actually uses. Byte fallback means
`decode(encode(x)) === x` for *every* input, so there is no `<unk>` token and
nothing is silently dropped.

**2. An interpolated Kneser-Ney n-gram model** (`src/ml/ngram.ts`) — the
workhorse. Trains in about a second, scores a candidate in microseconds. Code
repeats itself far more than prose does, which is why a well-smoothed count-based
model is genuinely competitive here. Discounts are estimated from the
count-of-counts rather than hardcoded.

**3. A decoder-only transformer** (`src/ml/transformer.ts`) — a real GPT-style
model with a hand-written backward pass. Optional, time-budgeted, and it adds
longer-range context the n-gram structurally cannot see. Every one of its 26
parameter tensors is gradient-checked against finite differences in the test
suite.

**4. Skip-gram embeddings** (`src/ml/embeddings.ts`) — word2vec with negative
sampling, trained on your tokens. These learn relationships specific to your
project, which is what makes semantic search work:

```
$ pca neighbors tokenizer
  bundle          ████████████████  0.671
  encode          █████████████     0.547
  vocabSize       ███████████       0.478
```

**5. An online learning-to-rank reranker** (`src/ml/reranker.ts`) — every
suggestion you accept or ignore is a training example. Accepts become *pairwise*
comparisons against the suggestions you passed over, which is the correct
objective for ranking and learns from far fewer events than classification.
Linear on purpose, so you can read what it learned about you:

```
$ pca stats --weights
  ngramLogProbSum       1.203 ████████████████
  ngramLogProb          1.014 ████████████
  continuesPartial      0.800 ██████████
  localFileHits         0.512 ██████
```

Plus a **style profile** measured from your corpus — indentation, quote style,
semicolons, naming convention, comment density — which is injected into every
Claude prompt so generated code comes back looking like your code, not like a
tutorial.

---

## Install

Requires Node.js 20 or newer. No Python, no CUDA, no build tools.

```bash
git clone https://github.com/MOTOMOTO2/personal-code-assistant.git
cd personal-code-assistant
npm install
npm run build
npm link          # puts `pca` on your PATH
```

Then, in any project you want to work on:

```bash
cd ~/code/my-project
pca train
```

Add `.pca/` to that project's `.gitignore` — the trained models are
machine-local and can be tens of megabytes.

Optional, for the Claude-backed commands:

```bash
# Windows
setx ANTHROPIC_API_KEY sk-ant-...
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

Run `pca doctor` to confirm everything is wired up.

---

## Use it

### Train

```bash
pca train                          # tokenizer, n-gram, index, embeddings (seconds)
pca train --neural --minutes 10    # also train the transformer for 10 minutes
pca train --neural-only --resume   # continue training the transformer
pca eval                           # measure quality on held-out files
```

`pca train` is fast enough to re-run whenever you like — a few seconds on a
mid-size repo. The transformer is the slow part and is strictly optional.

### Complete

```bash
pca complete --file src/auth.ts --line 42 --col 18
pca complete "const user = await db."
echo "function parse" | pca complete --stdin
pca complete --file src/auth.ts --line 42 --neural   # add the transformer
```

### Search

```bash
pca search "where do we retry failed uploads"
pca search "token refresh" --explain     # show how each ranker voted
pca neighbors userId                     # nearest tokens in embedding space
```

### Ask Claude, grounded in your codebase

```bash
pca ask "why does the token refresh run twice?"
pca explain src/auth.ts
pca review --staged                      # review what you are about to commit
pca review --base main
pca refactor src/auth.ts "extract the retry logic into a helper" --write
```

These retrieve relevant chunks from the local index first, so Claude answers
about *your* code with `path:line` citations rather than generic patterns. Each
command prints its token usage and estimated cost.

### Inspect

```bash
pca stats --weights    # what is trained, and what it learned about you
pca doctor             # environment and setup check
```

---

## Editor integration

Start the daemon once per project. It holds the models in memory, which is the
difference between 70 ms and 2 ms per completion:

```bash
pca serve            # http://127.0.0.1:7717, loopback only
```

Then install the VS Code extension:

```bash
cd editors/vscode
npm install && npm run build
# then: code --install-extension . (or press F5 in VS Code to debug it)
```

It provides ghost-text inline completions, a status-bar indicator, and
`PCA: Search Codebase` in the command palette. Accepted suggestions are reported
back to the daemon, which is what trains the ranker on your preferences.

### Any other editor

The daemon is a small JSON API, so hooking up Neovim, Emacs or anything else is
straightforward:

```bash
curl -s localhost:7717/complete -H 'content-type: application/json' \
  -d '{"prefix":"const user = await db.","filePath":"src/a.ts","count":3}'
```

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Readiness, which models are loaded, mean latency |
| `/stats` | GET | Manifest, index stats, learned ranker weights |
| `/complete` | POST | `{prefix, suffix?, filePath?, count?, neural?}` |
| `/search` | POST | `{query, limit?}` |
| `/feedback` | POST | `{accepted: number[], rejected?: number[][]}` |
| `/shutdown` | POST | Stop the daemon |

Set `PCA_TOKEN` before `pca serve` to require `Authorization: Bearer <token>`.
The daemon binds to `127.0.0.1` only and is never intended to be reachable
off-host.

---

## Measured results

On this repository (33 files, 97k tokens, 2.1 s to train), evaluated on
held-out files the models never saw:

| Metric | Value |
|---|---|
| Next-token perplexity (n-gram) | **15.1** |
| Next-token top-1 accuracy (n-gram) | **46.2%** |
| Next-token top-5 accuracy (n-gram) | **70.5%** |
| Suggestions fully correct | **36.9%** |
| Characters saved per call | **2.2** |
| Warm inference, mean / p95 | **2.3 ms / 4.7 ms** |
| Model load, once per process | 70 ms |

Reproduce with `pca eval`, `node bench/latency.mjs`, and `node bench/sweep.mjs`.

**These numbers come from a small repository and a held-out split of only two
files, so treat them as an order of magnitude rather than a precise figure.**
They will improve on a larger corpus — more code means longer n-gram contexts
actually match.

### The transformer, after 5 minutes of training, loses to the n-gram

This is the most important honest result in the project, so it gets its own
table. Same held-out files, same positions:

| Model | Perplexity | top-1 | top-5 |
|---|---|---|---|
| n-gram (1 s to train) | **15.1** | **46.2%** | **70.5%** |
| transformer (5 min to train) | 134.6 | 12.5% | 28.3% |
| 50/50 blend | 25.3 | 38.3% | 61.7% |

Five minutes at ~230 tokens/second is about 70k tokens of training — well under
one epoch over a 120k-token corpus. The model is barely past its initialisation,
and blending it in at equal weight actively *degrades* the n-gram.

This is why neural rescoring is **off by default** (`--neural` is opt-in), and why
`pca train --neural` prints a validation perplexity: the number tells you whether
the model is earning its place yet. Expect to need a substantially larger budget
— hours, not minutes — before the blend beats the n-gram on a corpus this size.
If the transformer is losing on your corpus, leave it off; the n-gram plus the
learned reranker is the configuration that produced every other number here.

### The eval harness earned its keep

"Suggestions fully correct" started at **6.4%** and ended at **36.9%**, and every
step of that came from measuring rather than guessing:

- **Ranking on mean log-probability made long and short candidates look equally
  good**, so the engine over-offered — 15.8 characters per suggestion with only
  2.8 of them right. Adding a *summed* log-probability feature prices length
  honestly: each extra token multiplies in a probability below 1. 6.4% → 31.2%.
- **The locality signal was summing sub-word matches**, so `userId` and
  `invoiceId` both matched on `id` and every candidate saturated it — the feature
  carried no information at all. Switching to the exact identifier: 31.2% → 35.5%.
- **The locality prior was applied only when seeding the beam**, and the very
  next pruning step re-ranked on raw log-probability and discarded those
  candidates. Carrying the bonus with the beam: → 36.9%.
- Suggestion length, required n-gram evidence, and the locality weight are all
  set from sweeps over held-out data (`bench/sweep.mjs`,
  `bench/tune-locality.mjs`), not from taste.

### Known limits, stated plainly

- **An order-4 n-gram cannot see past three tokens.** Given seven handlers that
  differ only in a resource name, `request.` is not enough context to pick the
  right one. That is what the transformer (96-token context) and the feedback
  loop are for.
- **Whole-line completion from a mid-line cursor is ~0%** on this corpus. The
  model reliably finishes the *next few tokens*, not the rest of your thought.
- **The transformer is small and slow to train** — see below.

---

## About the transformer

It is a real GPT-style decoder: pre-LayerNorm blocks, causal multi-head
attention, GELU MLP, tied embeddings, AdamW with a cosine schedule and gradient
clipping. The backward pass is written out by hand rather than taped by an
autograd engine, which is several times faster in plain JavaScript and keeps the
dependency count at zero.

Default config is 3 layers × 128 dim × 4 heads, 96-token context, ≈1.6M
parameters.

**Training it in pure JavaScript is roughly three orders of magnitude slower than
on a GPU.** That is not a bug, it is arithmetic: no SIMD, no BLAS, no CUDA. So
training is wall-clock budgeted — `pca train --neural --minutes 10` trains for
ten minutes and saves the best validation checkpoint it reached. Quality scales
with the budget you give it.

The honest framing: this is a small model that learns *your* code, and at a
realistic training budget on a small corpus **it does not yet beat the n-gram** —
see the measured comparison above. It is not a general coding model. `pca eval`
tells you exactly where it stands on your corpus, and until it wins there, leave
`--neural` off.

It is also the most rigorously tested part of the project. `test/transformer.test.ts`
verifies the gradient of **every** parameter tensor against central finite
differences using directional derivatives — parameters are float32, so
per-element differencing drowns in rounding noise, whereas a directional
derivative along a unit vector gives a signal well above it while still failing
if any single element's gradient is wrong.

---

## How it fits together

```
pca train
  walk (gitignore-aware)  →  tokenizer  →  n-gram
                          →  chunker    →  BM25 index
                          →  embeddings →  chunk vectors
                          →  style profile

pca complete  (2.3 ms warm)
  1 PROPOSE  beam search over the n-gram, plus local identifiers
             from the open buffer                          ~1 ms
  2 RESCORE  one transformer forward pass (optional)     10-100 ms
  3 RANK     linear reranker over both model scores,
             locality, retrieval and embedding features   ~0.5 ms

pca ask / review / explain / refactor
  hybrid retrieval (BM25 ⊕ vectors, fused by RRF)
    → Claude Opus 5, streaming, adaptive thinking, prompt caching
```

```
src/
  tokenizer/   lossless code tokenizer with byte fallback
  ml/          ngram · transformer · embeddings · reranker · linalg · params
  index/       walker · chunker · store (BM25 + vector + RRF)
  core/        bundle · train · trainNeural · complete · evaluate · style
  llm/         claude client · prompt construction
  server/      loopback daemon
  commands/    CLI command implementations
test/          52 tests: gradient checks, round-trips, retrieval, end-to-end
bench/         latency, hyper-parameter sweeps
editors/vscode inline-completion extension
```

### Design decisions worth knowing

**Zero ML dependencies.** No PyTorch, no ONNX, no native modules. `npm install`
pulls one package (the Anthropic SDK) and takes a few seconds. Everything from
the matmul kernels up is in this repository and readable.

**Lazy model loading.** The keystroke path touches the tokenizer, the n-gram and
the reranker and nothing else. The transformer and embedding matrix are paged in
only when something asks for them.

**Deterministic train/validation split by path hash.** The split survives
re-indexing, so perplexity numbers stay comparable between runs.

**Hybrid retrieval by rank fusion.** BM25 and cosine similarity produce scores on
incomparable scales, so they are combined by Reciprocal Rank Fusion over their
*rankings* instead of by tuning a score weight.

**Prompt caching that actually hits.** The Claude system prompt contains only the
role and the measured style profile, so it stays byte-identical across a session
and the cached prefix is reused. Retrieved code goes in the user turn, after the
breakpoint.

---

## Testing

```bash
npm test                              # 52 tests
node bench/latency.mjs                # warm-path latency distribution
node bench/sweep.mjs                  # suggestion length / evidence sweep
node bench/tune-locality.mjs          # locality weight sweep
```

The suite covers gradient correctness for every tensor, tokenizer losslessness
on Unicode/CRLF/NUL/emoji, that Kneser-Ney distributions sum to 1 and never
assign zero mass, save/load round-trips for every model, retrieval ranking, and
an end-to-end test that trains a real temp project and drives the daemon over
HTTP including auth rejection and malformed input.

---

## Assumptions made while building this

Recorded because they were judgement calls, not requirements:

- **TypeScript over Python.** Python has the better ML ecosystem, but the
  requirement was "fast and responsive", and a Python process that imports torch
  cannot start inside an inline-completion budget. Node starts in ~40 ms and
  editor integration is far simpler.
- **A statistical model as the primary completer, not the neural one.** Driven by
  the latency budget. On code, a well-smoothed n-gram is close enough to a small
  transformer to make the 30× speed difference the deciding factor.
- **`claude-opus-5` as the default**, with server-side fallback enabled so a
  safety-classifier false positive retries on a fallback model inside the same
  call rather than losing your turn. Override with `--effort` per command;
  `review` and `refactor` default to higher effort than `ask`.
- **Models stored per-project in `.pca/`**, not globally. Your work codebase and
  your side project should not share a language model.
- **n-gram order 4 and a 2M-token cap by default.** The n-gram tables are held in
  memory during training; raise both with `--order` and `--max-tokens` if you
  have the RAM.

## Not built

- No KV cache in the transformer, so multi-token neural generation re-runs the
  forward pass per token. Fine for rescoring (one pass), too slow for neural
  free-generation.
- No incremental re-indexing — `pca train` re-reads the whole project. At a few
  seconds per run this has not been worth fixing.
- VS Code is the only editor with a first-party client, though the daemon API is
  small enough that others are a short script.
- The reranker learns from real usage, so a fresh install ranks with hand-set
  priors. Those priors were swept on held-out data, but they are not *your*
  preferences until you have used it for a while.

## License

MIT
