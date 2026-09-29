# personal-code-assistant

A coding assistant you train on your own code, and a frontier model for the hard
questions. Machine learning models implemented from scratch, running locally at
interactive speed, with Claude wired in for the work that genuinely needs it.

One chat prompt sits over both. Each turn is routed: the local models answer
what they can, Claude answers the rest, and **every answer Claude gives is
written into the local models on the next train**, so the local half covers more
ground every week.

```
$ pca chat
pca chat - local models first, Claude when it matters
  168,492 tokens trained · 12 answers in memory · Claude available

you › where is the retry logic?

local  router p(claude)=0.11, 8 index hits
From the local index, most relevant first:

**src/lib/database.ts:14** (retryFailedUpload)
…                                                         4ms, no network

you › why does the upload retry twice?

claude router p(claude)=0.88
Because `retryFailedUpload` loops over `attempts` at src/lib/database.ts:14,
and `uploadAll` already wraps it in its own retry at src/jobs/upload.ts:53.
                                    ~$0.0071 · remembered, `/train` to learn it

$ pca learn
✓ Memory: 12 remembered answers folded in
✓ Corpus: 168,492 tokens
✓ Trained in 3.7s
```

Ask that second question again next week and it is answered locally, in
milliseconds, offline, for nothing.

---

## Quickstart: using it right now

This repository is already built and already trained on itself, so the local
half works immediately. The `pca` command is only on your PATH after `npm link`;
until then, invoke it directly:

```bash
node dist/cli.js doctor                       # confirm setup
node dist/cli.js chat                         # the conversation, local + Claude
node dist/cli.js chat "where are gradients clipped?"   # one question, then exit
node dist/cli.js complete "const user = await db."
node dist/cli.js search "where are gradients clipped"
node dist/cli.js neighbors tokenizer
node dist/cli.js stats --weights --router
node dist/cli.js corpus                       # what it is learning from
node dist/cli.js eval --curve                 # whether more code would help
```

`npm run pca -- <args>` does the same thing via the package script. To get the
short `pca` form, run `npm link` once (this writes a symlink into your global
npm prefix), after which every example below works verbatim.

If `dist/` is missing or stale, run `npm run build` first. If `.pca/` is missing,
run `node dist/cli.js train` — a few seconds.

**The Claude-backed commands (`ask`, `explain`, `review`, `refactor`) need
`ANTHROPIC_API_KEY` and are disabled without it.** Everything above is local,
offline, and needs no key — including `pca chat`, which simply stays on the
local half and says so. `doctor` tells you which half is live.

For the browser version of the chat, `pca serve --ui` and open
<http://127.0.0.1:7717>. Same models, same memory, same daemon the editor uses.

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
| Recalling something already answered | Local memory + embeddings | It is on disk; asking again is waste |
| Deciding which of the two answers | Local online router | Learns from your corrections |
| "Why does this break?", review, refactor | Claude Opus 5 | Needs real reasoning |
| Learning your preferences | Local online ranker | Trains on your accept/reject, instantly |

The local half works with no API key and no network.

---

## "so eventually I can use it just like Claude"

That is the goal this chat layer was built for, so here is exactly what it does
and does not deliver.

**It will not turn the local model into Claude.** A 1.6M-parameter n-gram and
transformer trained on one repository cannot reason. Training on Claude's
answers does not transfer Claude's reasoning; that is not how distillation at
this scale works, and any version of this README that claimed otherwise would be
lying to you.

**What it does do is move the boundary, permanently, in one direction.** Every
answer Claude gives about your codebase is a fact about *your* system that now
exists in writing. Store it and it is yours forever:

```
you ask  ──▶  router  ──▶  local models  ──▶  answer (0 ms, $0)
                 │
                 └──────▶  Claude  ──▶  answer  ──▶  .pca/lessons.jsonl
                                                          │
                          ┌───────────────────────────────┘
                          ▼
                     pca learn  ──▶  n-gram · embeddings · index · transformer
                          │
                          └──▶  the same question is local from now on
```

Three separate things happen to a stored answer, and they are worth separating
because only two of them are surprising:

1. **Recall.** Ask the same thing again, or a paraphrase, and the answer comes
   back from disk in milliseconds with its original attribution and date. This
   is the one that pays off immediately.
2. **Retrieval.** Stored answers are indexed next to your source, so they ground
   *future* Claude turns too. Claude reads what Claude already established about
   your system instead of rediscovering it, which makes later answers both
   better and cheaper.
3. **Training.** The tokenizer, n-gram, embeddings and transformer all train on
   the text. This is the weakest of the three — prose about your code is a small
   fraction of a code corpus — but it is the one that makes `pca complete` aware
   of vocabulary that appears in your discussions before it appears in your code.

And the **router learns where the line is**. When it routes locally and you type
`/claude`, that is a labelled example saying *escalate this shape of question*.
When you accept the local answer and move on, that is the opposite label. It is
a linear model, so you can read what it concluded:

```
$ pca stats --router
  memoryTop            -4.000 ████████████████████████
  locateShape          -2.500 ███████████████
  editShape             2.500 ███████████████
  reasonShape           2.000 ████████████
  magenta pushes a turn to Claude, green keeps it local
```

So the honest version of "eventually use it just like Claude" is: **the
interface is the same from day one, and the fraction of it that Claude has to
answer goes down over time.** It does not reach zero. On questions that need
real reasoning about code it has never seen, it should not.

### Measuring whether it is actually working

`pca eval` reports it, because a claim about compounding that nobody measures is
a claim you should not believe:

```
What the local half could handle on its own
  questions            4
  routed local         25.0%  (mean p(claude) 0.59)
  covered by memory    25.0%  (another stored answer already covers it)
  local answer kinds   locate 3 · memory 1
```

Every remembered question is replayed through the router and the local answerer
**with its own answer held out of memory**. Scoring a question against a store
that contains its own answer would report 100% and mean nothing; holding it out
asks the question that matters — does what this assistant already knows cover
the next thing you ask?

It says the local half *would answer*, not that it would answer as well as
Claude. Those are different claims, and only the first is measurable without a
human reading the output. The number to watch is whether it climbs as memory
fills.

This metric earned its keep immediately: the first run reported 75% routed local
while the answer kinds beside it showed only one answer actually coming from
memory. The inconsistency was real — the router was reading weak embedding
similarity as a memory hit, because SIF cosine between any two texts from one
codebase has a floor well above zero. `retrievalTop` and `retrievalCoverage` had
the same problem from the other direction: both sit near 1.0 for any question
about a repository you have indexed, so their weights were acting as a second
bias term and drowning out question shape. Both are fixed, and the two numbers
now agree.

---

## Feeding it: what to train on, and how long until it helps

Short version: **code, your own answers, and more code.** Not videos, not
courses, not tutorials. The reason is mechanical rather than snobbish, and it is
worth understanding before you spend a weekend on the wrong thing.

### Why videos cannot be training data here

Every model in this project predicts **the next token of code**. That is the
entire learning signal. A video has no tokens in it; transcribed, an hour of
someone talking is roughly **10,000 tokens of English prose about code**.

Put that next to what the corpus already holds:

| Source | Tokens | What the model learns from it |
|---|---:|---|
| This repository | ~195,000 | how *your* code is written |
| One hour of video, transcribed | ~10,000 | how English sentences about code are written |
| A mid-size library you depend on | ~400,000 | idioms you are about to type |

A hundred hours of tutorial would be a *fifth* of one repository, and the wrong
fifth: prose teaches a code model the statistics of English, which it will then
happily suggest in the middle of your TypeScript. Video is an excellent way for
**you** to learn and a useless way to train **this**. The same goes for blog
posts and course material. Documentation is a partial exception — `.md` files are
indexed, so prose is genuinely useful for *search* — but it is weak fuel for
completion, which is why the style profile ignores it.

### The three things that actually feed it

```bash
# 1. Your code. Automatic - it is whatever repo you run `pca train` in.
pca train

# 2. Your answers. Every Claude reply is stored and folded back in.
pca learn

# 3. Other people's code, in the languages you actually use.
pca corpus scan ~/src                # what is nearby and what each would add
pca corpus add ../my-other-project
pca corpus add ~/src/some-library --max-tokens 200000
pca corpus                           # see the whole corpus, broken down
```

`pca corpus scan` is the place to start, because "add more code" is useless
advice until you know what you have. It walks a directory looking for projects,
measures each with the same walker training uses — so the token count is the
real one — and prints the `add` command for each. It registers nothing: whether
a codebase resembles yours closely enough to help is a judgement only you can
make.

```
$ pca corpus scan ~/src

Projects under /home/you/src
  api-server                 412 files    1.9MB  ~560,000 tokens (2.9x this corpus)
    pca corpus add "/home/you/src/api-server"
  dashboard                  203 files    780KB  ~223,000 tokens (1.1x this corpus)
    pca corpus add "/home/you/src/dashboard"
```

The third is new, and it is the lever you control. A source is any directory
elsewhere on disk: another project of yours, a library you write against, a
codebase whose style you are trying to absorb. It is walked by the same rules as
your project, folded into the tokenizer, n-gram, embeddings and index, and every
path from it is prefixed `@name/` so a borrowed hit is never mistaken for one of
yours:

```
$ pca search "exponential backoff jitter"
@backoff-lib/src/backoff1.ts:1 retryWithBackoff1
```

Three things stay deliberately project-only:

- **The style profile.** The completer imitates it — quote style, indent width,
  semicolons — so letting a borrowed repo vote would make the assistant write
  like someone else inside your own files.
- **The validation split.** The question `pca eval` answers is how well the
  model predicts *your* code. Scoring it on someone else's would answer a
  different question and flatter the result.
- **Corpus priority.** Under the token cap the order is memory, then your code,
  then borrowed code, so whatever has to be dropped is always the least specific
  to you.

**Pick sources you resemble.** Adding the Linux kernel to a TypeScript project
makes the model worse, not better: it spends vocabulary and probability mass on
patterns you will never type. The rule of thumb is that a source should look
like code you could plausibly have written.

### How long until it is useful — measured, not guessed

`pca eval --curve` retrains the n-gram on growing slices of the corpus and
scores every slice on the same held-out files. This is the only honest way to
answer "would more data help", and the answer for this repository is:

```
$ pca eval --curve

Does more code help? (n-gram retrained on slices of the corpus)
  corpus    tokens   top-1   perplexity
  13%       41,057   41.6%         23.5
  25%       59,581   44.9%         18.1
  50%       96,337   46.9%         15.2
  100%     199,457   50.0%         11.8
  last 2.1x of corpus bought 3.1 points of top-1

  scaling  error ∝ tokens^-0.094  (r² 0.97)
  projected 55.0% top-1 needs ~580k tokens (2.9x this corpus)
  projected 60.0% top-1 needs ~2.0M tokens (10.1x this corpus)
```

Read the exponent, because it is the whole story: **0.094 is shallow.** Doubling
the corpus cuts the error by about 6%, which is roughly three points of top-1 at
this size. Going from 50% to 60% top-1 needs about **ten times** the code — call
it five to ten more repositories the size of this one. That is achievable in an
afternoon of `pca corpus add` if you have the repos, and it is not going to
happen at all by waiting.

So the timeline, in the only units that mean anything:

| When | What works | What it took |
|---|---|---|
| **Immediately** | Inline completion of the next few tokens at ~4 ms; `pca search` over your whole codebase; `pca chat` answering "where is X" locally | one `pca train`, seconds |
| **First session** | Claude answers the hard questions, grounded in your retrieved code; each answer is stored | an API key |
| **After ~10 stored answers** | Repeat and paraphrased questions come back locally, instantly, offline | using `pca chat` normally |
| **After adding 5–10 repos** | ~5–10 points of top-1, noticeably longer correct completions | `pca corpus add`, an afternoon |
| **Never** | Writing a feature from a description; reasoning about code it has not seen | — that is what the Claude half is for, permanently |

The honest summary: **the local half is useful on day one for the jobs a local
model is good at, and it never becomes Claude.** What improves over weeks is not
the model's intelligence — it is the fraction of *your* questions that have
already been answered once and written down. That curve is measured too, by the
leave-one-out routing number in the section above.

The projection is a fit to four points on one machine. It is a planning number,
not a promise, and the tool refuses to print it at all when the points do not
lie on a line (r² below 0.8).

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

**6. An online routing model** (`src/ml/router.ts`) — decides per chat turn
whether the local models can answer. Logistic regression over ten features:
memory similarity, retrieval strength, index coverage of the query's terms, and
the shape of the question (a "where is" is a lookup, a "why does" is not). Its
prior leans towards Claude on purpose — being wrong towards Claude costs a few
cents, being wrong towards local costs a useless answer.

Plus a **style profile** measured from your corpus — indentation, quote style,
semicolons, naming convention, comment density — which is injected into every
Claude prompt so generated code comes back looking like your code, not like a
tutorial.

### What the local half will and will not say

The local models never write prose. An n-gram trained on TypeScript generates
TypeScript, and a fluent-but-wrong English answer would be worse than no answer,
so the local answerer is restricted to four things it is actually good at:

| It says | When |
|---|---|
| A stored answer, verbatim and attributed | Memory matches on ≥2 content words and ≥0.45 similarity |
| Code locations with `path:line` and excerpts | The index has hits |
| Completions for a half-typed line | The message ends mid-expression |
| Nearest tokens in the embedding space | You asked what relates to an identifier |
| "I have nothing local on that" | None of the above |

That last row is the important one. It is why the local half is always either
useful or quiet, and never confidently wrong.

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

### Chat

```bash
pca chat                                   # the conversation
pca chat "where is the retry logic?"       # one question, then exit
pca chat "explain the beam search" --effort max
pca serve --ui                             # the same chat in a browser
echo "where is X?" | pca chat              # piped, one question per line
```

Inside the chat:

| | |
|---|---|
| `/claude` | Re-ask the last question with Claude. Teaches the router. |
| `/claude <question>` | Send a new question straight to Claude |
| `/local <question>` | Force the local models |
| `/why` | How the last turn was routed, and on what evidence |
| `/memory [query]` | What it remembers about this codebase |
| `/train` | Fold new answers into the models, without leaving the chat |
| `/stats` | Corpus, memory, router weights, session cost |
| `/new` | Start a fresh conversation |

Transcripts go to `.pca/chat/`, remembered answers to `.pca/lessons.jsonl`. Both
are plain text you can read, grep, edit or delete.

With no `ANTHROPIC_API_KEY` the chat still runs; every turn stays local and the
banner says so.

### Train

```bash
pca train                          # tokenizer, n-gram, index, embeddings (seconds)
pca learn                          # the same, plus everything chat has remembered
pca learn --watch                  # keep retraining as you edit
pca learn --minutes 10             # ... and continue the transformer for 10 min
pca train --neural --minutes 10    # train the transformer for 10 minutes
pca train --neural-only --resume   # continue training the transformer
pca eval                           # measure quality on held-out files
pca eval --curve                   # ... and whether more code would still help
pca train --no-sources             # skip registered sources for one run
```

### Feed it more code

```bash
pca corpus                         # the corpus, broken down by where it came from
pca corpus scan ~/src              # nearby projects, and what each would add
pca corpus add ../other-repo       # train on another directory too
pca corpus add ~/src/lib --name lib --max-tokens 200000 --no-index
pca corpus remove lib              # stop training on it
```

Sources are recorded in `.pca/sources.json`, which is plain JSON and safe to
edit by hand. Changes take effect on the next `pca train` or `pca learn`. See
[Feeding it](#feeding-it-what-to-train-on-and-how-long-until-it-helps) for which
sources are worth adding and which make the model worse.

`pca train` is fast enough to re-run whenever you like — a few seconds on a
mid-size repo. `pca learn` is `pca train` plus the chat memory, and is the one
worth putting on a keybinding or leaving in `--watch` mode. The transformer is
the slow part and is strictly optional.

`--watch` debounces: a rebase or a find-and-replace fires hundreds of filesystem
events and retrains once, at the end.

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
pca stats --router     # where it draws the local/Claude line, and why
pca stats --memory     # the most recent answers it has stored
pca doctor             # environment and setup check
```

---

## Editor integration

Start the daemon once per project. It holds the models in memory, which is the
difference between 70 ms and 2 ms per completion:

```bash
pca serve            # http://127.0.0.1:7717, loopback only
pca serve --ui       # ... and serve the browser chat client at /
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
| `/` | GET | The browser chat client (needs `--ui`) |
| `/health` | GET | Readiness, which models are loaded, mean latency |
| `/stats` | GET | Manifest, index stats, ranker and router weights, memory size |
| `/memory` | GET | `?q=` to search remembered answers, or the most recent |
| `/complete` | POST | `{prefix, suffix?, filePath?, count?, neural?}` |
| `/search` | POST | `{query, limit?}` |
| `/chat` | POST | `{message, force?, sessionId?}` → server-sent events |
| `/train` | POST | Retrain in place; the daemon reloads its own models |
| `/feedback` | POST | `{accepted: number[], rejected?: number[][]}` |
| `/shutdown` | POST | Stop the daemon |

`/chat` streams `data:` frames of `{type: "route" | "thinking" | "text" | "done"
| "error", ...}`, so any client can drive a conversation in about thirty lines.

Set `PCA_TOKEN` before `pca serve` to require `Authorization: Bearer <token>`
(or `?token=` for the browser, which cannot set a header on a page load). The
daemon binds to `127.0.0.1` only, rejects requests whose `Host` is not a
loopback name — a loopback bind alone does not stop DNS rebinding — and is never
intended to be reachable off-host.

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
- **More data helps slowly.** Measured on this repository, top-1 error falls as
  `tokens^-0.094`: a doubling of the corpus buys about three points. `pca corpus
  add` is worth doing and will not transform the thing — see
  [Feeding it](#feeding-it-what-to-train-on-and-how-long-until-it-helps).
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
pca train / pca learn
  walk (gitignore-aware)  →  tokenizer  →  n-gram
  + .pca/lessons.jsonl    →  chunker    →  BM25 index
                          →  embeddings →  chunk vectors
                          →  style profile

pca complete  (2.3 ms warm)
  1 PROPOSE  beam search over the n-gram, plus local identifiers
             from the open buffer                          ~1 ms
  2 RESCORE  one transformer forward pass (optional)     10-100 ms
  3 RANK     linear reranker over both model scores,
             locality, retrieval and embedding features   ~0.5 ms

pca chat
  1 GROUND   hybrid retrieval + memory recall, one pass, shared    ~3 ms
  2 ROUTE    logistic router over 10 features → local | claude
  3a LOCAL   replay a stored answer · locate code · complete · neighbours
  3b CLAUDE  history + retrieved code + memory → Claude Opus 5,
             streaming, adaptive thinking, prompt caching
  4 RECORD   the answer → .pca/lessons.jsonl → next `pca learn`

pca ask / review / explain / refactor
  hybrid retrieval (BM25 ⊕ vectors, fused by RRF)
    → Claude Opus 5, streaming, adaptive thinking, prompt caching
```

```
src/
  tokenizer/   lossless code tokenizer with byte fallback
  ml/          ngram · transformer · embeddings · reranker · router · linalg
  index/       walker · chunker · store (BM25 + vector + RRF)
  core/        bundle · train · trainNeural · complete · evaluate · style · watch
  chat/        memory (lesson store) · local answerer · session + routing
  llm/         claude client · prompt construction
  server/      loopback daemon (completion, search, chat, train)
  web/         the browser chat client, one self-contained page
  commands/    CLI command implementations
test/          62 tests: gradient checks, round-trips, retrieval, chat, end-to-end
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
and the cached prefix is reused. Retrieved code and recalled memory go in the
*latest* user turn, after the breakpoint, so a growing conversation keeps hitting
the cache instead of invalidating it.

**Memory is append-only JSONL, not a database.** The write happens at the end of
a streamed answer, where a crash is entirely possible; appending a line cannot
corrupt the lines before it, and a half-written line is skipped on load. It is
also greppable, editable and deletable with the tools you already have.

**The router escalates on doubt.** Its prior leans towards Claude, and it only
replays a stored answer when at least two content words match. Being wrong
towards Claude costs a few cents. Being wrong towards local costs you a
confident, useless answer, which is the failure mode that would make you stop
trusting the tool.

**Stopwords are stripped from memory recall but not from code search.**
Questions are prose, where "where is X" and "why is X" share every word but the
one that matters; identifiers are not. Scoring recall over raw words let a
question about Kneser-Ney discounts match a stored answer about the locality
weight on "where", "is" and "the" alone — caught by a smoke test against the
real index, not by the unit tests, which is why that case is now one.

---

## Testing

```bash
npm test                              # 62 tests
node bench/latency.mjs                # warm-path latency distribution
node bench/sweep.mjs                  # suggestion length / evidence sweep
node bench/tune-locality.mjs          # locality weight sweep
```

The suite covers gradient correctness for every tensor, tokenizer losslessness
on Unicode/CRLF/NUL/emoji, that Kneser-Ney distributions sum to 1 and never
assign zero mass, save/load round-trips for every model, retrieval ranking, and
an end-to-end test that trains a real temp project and drives the daemon over
HTTP including auth rejection and malformed input.

The chat layer is tested the same way, against a real trained fixture rather
than mocks: that a remembered answer is recalled by a paraphrase *and not* by an
unrelated question, that remembered answers actually reach the corpus and the
index, that routing follows question shape and only follows memory when the
recall is real, that routing coverage is measured leave-one-out so an answer
cannot recall itself, that escalating a turn moves the router's weights and
persists them, that the file watcher retrains on a genuine filesystem event, and
that the daemon serves the UI, streams a turn, rejects a non-loopback `Host`,
and retrains on request. The Claude path runs against an injected fake backend,
so the whole suite passes offline with no API key.

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
- **"Connect my AI so I can use it just like Claude" was read as: put both
  engines behind one chat interface, and close the loop so the local half grows
  into more of it.** The alternative readings — train the transformer until it
  can hold a conversation, or just add a `pca ask` alias — were rejected as,
  respectively, not achievable at this scale and not an answer to the question.
  What was built is the conventional middle: routing, memory, and training on
  what memory collects. See ["so eventually I can use it just like
  Claude"](#so-eventually-i-can-use-it-just-like-claude) for what that does and
  does not get you.
- **Local answers are not written to memory, only Claude's are.** Storing the
  local half's own output would compound paraphrases of paraphrases, and a
  recalled answer is already in memory by definition.
- **Retrieval and memory are computed once per turn and shared** by the router,
  the local answerer and the Claude prompt. One index query per message, not
  three.

## Not built

- No KV cache in the transformer, so multi-token neural generation re-runs the
  forward pass per token. Fine for rescoring (one pass), too slow for neural
  free-generation.
- No incremental re-indexing — `pca train` re-reads the whole project, and every
  registered source with it. At a few seconds per run this has not been worth
  fixing, but a source of a few million tokens would make it worth fixing.
- **The learning curve only sweeps the n-gram.** It trains by counting, so four
  slices cost about what one `pca train` costs. Sweeping the transformer would
  take hours and answer a question nobody asked, so the projection describes the
  model that actually serves inline completion and not the optional one.
- VS Code is the only editor with a first-party client, though the daemon API is
  small enough that others are a short script.
- The reranker learns from real usage, so a fresh install ranks with hand-set
  priors. Those priors were swept on held-out data, but they are not *your*
  preferences until you have used it for a while.
- **The router's priors have not been swept**, unlike the reranker's — that
  needs a corpus of real routing decisions, which only exists after the tool has
  been used. They are reasoned defaults, sanity-checked against the routing
  eval on a handful of real questions, and they move as you correct them.
- **No long-run number for how far memory moves the local share.** The metric
  exists (`pca eval`, above) and the mechanism is tested, but quoting a headline
  figure from four seeded answers would be meaningless. It needs weeks of real
  use; `pca eval` and `pca stats --router` let you watch your own.
- Memory recall is lexical overlap plus SIF-embedding cosine, which is adequate
  for hundreds of answers. At tens of thousands it would want a real ANN index.
- Chat has no tool use: it reads your codebase and answers, but it does not edit
  files. `pca refactor --write` is still the way to apply a change.

## License

MIT
