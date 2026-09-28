import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { Bundle } from "../src/core/bundle.js";
import { ChatBackend, ChatSession } from "../src/chat/session.js";
import {
  LessonStore,
  appendLesson,
  extractCites,
  loadLessons,
  newLessonId,
} from "../src/chat/memory.js";
import {
  LocalAssistant,
  buildTurnContext,
  isCodeContinuation,
  isStrongRecall,
  recallStrength,
} from "../src/chat/local.js";
import { Router, routerFeatureIndex } from "../src/ml/router.js";
import { StreamOptions, StreamResult } from "../src/llm/claude.js";
import { evalRouting } from "../src/core/evaluate.js";
import { loadCorpus, trainProject } from "../src/core/train.js";
import { isTrainableChange, startWatcher } from "../src/core/watch.js";
import { createDaemon } from "../src/server/daemon.js";
import { setQuiet } from "../src/util/log.js";
import { workspace } from "../src/util/paths.js";

setQuiet(true);

/**
 * A fixture with one obviously-findable subject, so a retrieval assertion is
 * about ranking rather than luck.
 */
function writeFixtureProject(root: string): void {
  fs.mkdirSync(path.join(root, "src"), { recursive: true });
  fs.writeFileSync(path.join(root, "package.json"), '{\n  "name": "fixture"\n}\n');
  fs.writeFileSync(
    path.join(root, "src", "uploader.ts"),
    `export class Uploader {
  async retryFailedUpload(uploadId: string, attempts: number) {
    for (let attempt = 0; attempt < attempts; attempt++) {
      const uploaded = await this.attemptUpload(uploadId);
      if (uploaded) return true;
    }
    return false;
  }

  private async attemptUpload(uploadId: string) {
    return uploadId.length > 0;
  }
}
`,
  );
  fs.writeFileSync(
    path.join(root, "src", "session.ts"),
    `export interface SessionRecord {
  sessionId: string;
  expiresAt: number;
}

export function isExpired(record: SessionRecord, now: number) {
  return record.expiresAt <= now;
}
`,
  );
}

function setupProject(t: { after(fn: () => void): void }): Bundle {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pca-chat-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  writeFixtureProject(root);
  const bundle = new Bundle(workspace(root));
  trainProject(bundle, { vocabSize: 1024, embeddingDim: 16, embeddingEpochs: 2, valFraction: 0 });
  return bundle;
}

function lesson(question: string, answer: string) {
  return {
    id: newLessonId(),
    at: new Date().toISOString(),
    question,
    answer,
    source: "claude" as const,
    model: "claude-opus-5",
    cites: extractCites(answer),
    grounding: [] as string[],
  };
}

/** GET /health with an arbitrary Host header, which `fetch` refuses to send. */
function statusWithHost(port: number, host: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: "127.0.0.1", port, path: "/health", method: "GET", headers: { Host: host } },
      (res) => {
        res.resume();
        res.on("end", () => resolve(res.statusCode ?? 0));
      },
    );
    req.on("error", reject);
    req.end();
  });
}

/** A backend that returns canned text, so the Claude path is testable offline. */
function fakeBackend(text: string): ChatBackend & { calls: StreamOptions[] } {
  const calls: StreamOptions[] = [];
  return {
    model: "claude-opus-5",
    calls,
    async stream(opts: StreamOptions): Promise<StreamResult> {
      calls.push(opts);
      for (const ch of text) opts.onText?.(ch);
      return {
        text,
        model: "claude-opus-5",
        stopReason: "end_turn",
        refusal: null,
        usage: { inputTokens: 100, outputTokens: 50, cacheReadTokens: 0, cacheCreationTokens: 0 },
        estimatedCostUsd: 0.001,
      };
    },
  };
}

test("memory round-trips and recalls by meaning as well as by word", (t) => {
  const bundle = setupProject(t);
  appendLesson(
    bundle,
    lesson(
      "why does the upload retry twice?",
      "Because `retryFailedUpload` loops over `attempts` at src/uploader.ts:2 and the caller already retries once itself.",
    ),
  );
  appendLesson(bundle, lesson("how do sessions expire?", "`isExpired` compares expiresAt to now."));

  const store = loadLessons(bundle);
  assert.equal(store.size, 2);

  const recalls = store.recall("why does the upload retry twice?", {
    tokenizer: bundle.tokenizer,
    embeddings: bundle.embeddings,
  });
  assert.ok(recalls.length > 0, "expected a recall");
  assert.match(recalls[0].lesson.question, /upload retry/);
  assert.ok(recalls[0].score > 0.45, `exact question should score high, got ${recalls[0].score}`);

  // The unrelated lesson must not outrank the relevant one.
  const other = store.recall("session expiry", { tokenizer: bundle.tokenizer });
  assert.match(other[0].lesson.question, /sessions expire/);

  // Stopwords must not carry a recall. "where/is/the" are shared with both
  // stored questions, and scoring over raw words let that alone clear the
  // replay threshold when the store was small.
  const unrelated = store.recall("where is the Kneser-Ney discount estimated?", {
    tokenizer: bundle.tokenizer,
    embeddings: bundle.embeddings,
  });
  for (const r of unrelated) {
    assert.ok(
      !isStrongRecall(r),
      `"${r.lesson.question}" should not be replayed (score ${r.score.toFixed(2)}, ` +
        `${r.matchedTerms} terms)`,
    );
  }

  assert.deepEqual(extractCites("see src/uploader.ts:2 and src/uploader.ts:2"), ["src/uploader.ts:2"]);
});

test("remembered answers become part of the trained corpus and index", (t) => {
  const bundle = setupProject(t);
  const before = bundle.manifest!.corpusTokens;

  appendLesson(
    bundle,
    lesson(
      "what is the quarantine policy?",
      "Quarantined uploads are held for fourteen days in the quarantine bucket, then purged by the nightly sweeper. This is a policy decision, not something the code states anywhere.",
    ),
  );

  const report = trainProject(bundle, {
    vocabSize: 1024,
    embeddingDim: 16,
    embeddingEpochs: 2,
    valFraction: 0,
  });
  assert.equal(report.lessons, 1, "the lesson should have been folded in");
  assert.ok(report.corpusTokens > before, "corpus should grow with remembered answers");

  // And it must be retrievable - this is the part that makes the local half
  // able to answer a question whose answer exists nowhere in the source.
  bundle.reload();
  const hits = bundle.index!.search("quarantine policy purge", { limit: 5 });
  assert.ok(
    hits.some((h) => h.file.startsWith("~memory/")),
    `expected a memory hit, got ${hits.map((h) => h.file).join(", ")}`,
  );

  const corpus = loadCorpus(bundle, 0, undefined, true);
  assert.ok(
    corpus.trainPaths.some((p) => p.startsWith("~memory/")),
    "neural training corpus should include memory",
  );
  const without = loadCorpus(bundle, 0, undefined, false);
  assert.ok(!without.trainPaths.some((p) => p.startsWith("~memory/")));
});

test("the local assistant answers from memory, from the index, or not at all", (t) => {
  const bundle = setupProject(t);
  appendLesson(
    bundle,
    lesson(
      "why does the upload retry twice?",
      "Because the caller retries once and `retryFailedUpload` retries again.",
    ),
  );
  const store = loadLessons(bundle);
  const local = new LocalAssistant(bundle);

  const remembered = local.answer(
    buildTurnContext(bundle, store, "why does the upload retry twice?"),
  );
  assert.equal(remembered.kind, "memory");
  assert.match(remembered.text, /caller retries once/);
  assert.match(remembered.text, /Recalled from memory/);

  const located = local.answer(buildTurnContext(bundle, store, "where is attemptUpload defined"));
  assert.equal(located.kind, "locate");
  assert.ok(located.sources.length > 0, "a locate answer must cite something");
  assert.ok(
    located.sources.some((src) => src.startsWith("src/uploader.ts:")),
    `expected uploader.ts, got ${located.sources.join(", ")}`,
  );

  // Nothing indexed, nothing remembered: say so rather than inventing.
  const empty = new LocalAssistant(bundle).answer(
    buildTurnContext(bundle, new LessonStore([]), "zzqqxx wombat protocol"),
  );
  assert.ok(["unsure", "locate"].includes(empty.kind));

  assert.equal(isCodeContinuation("const user = await db."), true);
  assert.equal(isCodeContinuation("why does the upload retry?"), false);
  assert.equal(isCodeContinuation("where is the uploader"), false);
});

test("the router learns from corrections and stays interpretable", (t) => {
  const bundle = setupProject(t);
  const router = new Router();
  const features = new Float64Array(routerFeatureIndex("lengthTokens") + 1);
  features[routerFeatureIndex("bias")] = 1;
  features[routerFeatureIndex("reasonShape")] = 1;

  const before = router.probability(features);
  for (let i = 0; i < 30; i++) router.learn(features, 0);
  const after = router.probability(features);
  assert.ok(after < before, `escalation probability should fall: ${before} -> ${after}`);
  assert.equal(router.toLocal, 30);

  for (let i = 0; i < 60; i++) router.learn(features, 1);
  assert.ok(router.probability(features) > after, "and rise again when corrected the other way");

  router.save(bundle.path("router"));
  const loaded = Router.load(bundle.path("router"));
  assert.equal(loaded.updates, router.updates);
  assert.ok(Math.abs(loaded.probability(features) - router.probability(features)) < 1e-12);

  const described = loaded.describe();
  assert.equal(described.length, loaded.weights.length);
  assert.ok(
    Math.abs(described[0].weight) >= Math.abs(described[described.length - 1].weight),
    "describe() should sort by magnitude",
  );
});

test("routing reflects question shape, and memory only when it is real", (t) => {
  const bundle = setupProject(t);
  appendLesson(
    bundle,
    lesson(
      "why does the upload retry twice?",
      "Because the caller retries once and `retryFailedUpload` retries again at src/uploader.ts:2.",
    ),
  );
  const store = loadLessons(bundle);
  const router = bundle.router;
  const p = (question: string, lessons = store) =>
    router.probability(buildTurnContext(bundle, lessons, question).features);

  // A lookup the index can serve stays local; reasoning and edits do not.
  assert.ok(p("where is attemptUpload defined") < 0.5, "a lookup should stay local");
  assert.ok(p("add a backoff to the uploader") > 0.5, "an edit needs Claude");
  assert.ok(
    p("why does isExpired compare to now?") > 0.5,
    "a why-question with no memory needs Claude, however good the retrieval hit is",
  );
  // The same shape of question, once it has been answered, does not.
  assert.ok(p("why does the upload retry twice?") < 0.5, "a real memory hit should pull local");

  // A weak similarity must contribute nothing: SIF cosine over one codebase has
  // a floor well above zero, and a raw score would make every question look
  // like a partial hit.
  const weak = store.recall("session expiry semantics", {
    tokenizer: bundle.tokenizer,
    embeddings: bundle.embeddings,
  });
  assert.equal(recallStrength(weak[0]), 0, "a sub-threshold recall must not pull local");
  const exact = store.recall("why does the upload retry twice?", { tokenizer: bundle.tokenizer });
  assert.ok(recallStrength(exact[0]) > 0.5, "an exact hit should pull hard");
});

test("routing coverage is measured leave-one-out, not against itself", (t) => {
  const bundle = setupProject(t);
  const one = lesson(
    "why does the upload retry twice?",
    "Because the caller retries once and `retryFailedUpload` retries again at src/uploader.ts:2.",
  );
  appendLesson(bundle, one);

  // A single answer cannot cover its own question - that would be the metric
  // measuring storage rather than coverage.
  const alone = evalRouting(bundle, loadLessons(bundle).lessons);
  assert.equal(alone.questions, 1);
  assert.equal(alone.coveredByOthers, 0, "a lesson must not be allowed to recall itself");

  // A paraphrase of it, though, is covered by the original.
  appendLesson(
    bundle,
    lesson(
      "why does the upload retry two times?",
      "The caller retries once, and `retryFailedUpload` retries again at src/uploader.ts:2.",
    ),
  );
  const pair = evalRouting(bundle, loadLessons(bundle).lessons);
  assert.equal(pair.questions, 2);
  assert.ok(pair.coveredByOthers > 0, "a paraphrase should be covered by the original");
  assert.ok(pair.local > 0, "and therefore routed locally");
  // The report must be internally consistent: anything counted as covered was
  // answered from memory.
  assert.ok((pair.byKind.memory ?? 0) >= pair.coveredByOthers * pair.questions);
});

test("a chat session routes, remembers, and learns from an escalation", async (t) => {
  const bundle = setupProject(t);
  const backend = fakeBackend(
    "The retry lives in `retryFailedUpload` at src/uploader.ts:2, which loops over `attempts` and returns on the first success. Nothing else retries, so a double retry means the caller is retrying too.",
  );
  const session = new ChatSession(bundle, { client: backend, persist: true });

  // A lookup the local half can serve: no API call should happen.
  const located = await session.ask("where is attemptUpload defined");
  assert.equal(located.route, "local");
  assert.equal(backend.calls.length, 0);
  assert.equal(located.costUsd, 0);

  // Escalating teaches the router that this shape needed Claude.
  const updatesBefore = bundle.router.updates;
  const escalated = await session.escalate();
  assert.equal(escalated.route, "claude");
  assert.equal(backend.calls.length, 1);
  assert.ok(bundle.router.updates > updatesBefore, "escalation must train the router");
  assert.ok(fs.existsSync(bundle.path("router")), "router weights must be persisted");

  // The answer is now in memory...
  assert.ok(escalated.lessonId, "a substantive answer should be remembered");
  assert.equal(loadLessons(bundle).size, 1);

  // ...and the same question is answered locally from it next time.
  const again = await session.ask("where is attemptUpload defined");
  assert.equal(again.route, "local");
  assert.equal(again.kind, "memory");
  assert.match(again.text, /retryFailedUpload/);
  assert.equal(backend.calls.length, 1, "recall must not hit the API again");

  // Grounding and memory ride in the user turn, not the system prompt, so the
  // cached prefix stays byte-stable.
  const sent = backend.calls[0];
  const lastTurn = sent.messages[sent.messages.length - 1];
  assert.equal(lastTurn.role, "user");
  assert.match(String(lastTurn.content), /<codebase_context>/);
  assert.doesNotMatch(sent.system, /<codebase_context>/);
  // Never two turns of the same role in a row.
  for (let i = 1; i < sent.messages.length; i++) {
    assert.notEqual(sent.messages[i].role, sent.messages[i - 1].role);
  }

  const transcript = JSON.parse(fs.readFileSync(session.transcriptPath(), "utf8")) as {
    turns: Array<{ role: string }>;
  };
  assert.ok(transcript.turns.length >= 4, "the transcript should be on disk");
});

test("a forced local turn works with no credentials and costs nothing", async (t) => {
  const bundle = setupProject(t);
  const session = new ChatSession(bundle, { persist: false });
  const result = await session.ask("explain the entire architecture and rewrite it", {
    force: "local",
  });
  assert.equal(result.route, "local");
  assert.equal(result.costUsd, 0);
});

test("the watcher retrains on a real file change", async (t) => {
  const bundle = setupProject(t);
  const cycles: string[] = [];
  const watcher = startWatcher(bundle.ws, {
    debounceMs: 60,
    maxCycles: 1,
    train: { vocabSize: 1024, embeddingDim: 0, valFraction: 0 },
    onCycle: (_report, reason) => cycles.push(reason),
  });
  t.after(() => watcher.close());

  fs.writeFileSync(
    path.join(bundle.ws.root, "src", "added.ts"),
    "export const addedLater = 42;\n",
  );
  // fs.watch is the only thing here that is not deterministic; the cycle limit
  // resolves `done`, and the test times out if it never fires.
  await watcher.done;
  assert.equal(watcher.cycles, 1, "expected exactly one retrain");
  assert.equal(cycles.length, 1);

  bundle.reload();
  assert.ok(
    bundle.index!.files.some((f) => f.path === "src/added.ts"),
    "the new file should be in the index",
  );

  assert.equal(isTrainableChange("src/a.ts"), true);
  assert.equal(isTrainableChange(".pca/ngram.json.gz"), false);
  assert.equal(isTrainableChange("node_modules/x/index.js"), false);
  assert.equal(isTrainableChange("README.md"), true);
  assert.equal(isTrainableChange("src/logo.png"), false);
});

test("the daemon serves the chat UI, streams a turn, and retrains on request", async (t) => {
  const bundle = setupProject(t);
  const daemon = createDaemon(bundle.ws, { port: 0, ui: true });
  const port = await daemon.listen(0);
  t.after(async () => {
    await daemon.close();
  });
  const base = `http://127.0.0.1:${port}`;

  const page = await fetch(`${base}/`);
  assert.equal(page.status, 200);
  assert.match(page.headers.get("content-type") ?? "", /text\/html/);
  assert.match(await page.text(), /pca chat/);

  // DNS rebinding: a loopback bind alone would let this through. `fetch` will
  // not let a caller set Host, so this goes out over raw http.
  assert.equal(await statusWithHost(port, "evil.example.com"), 403);
  assert.equal(await statusWithHost(port, `localhost:${port}`), 200);

  const chat = await fetch(`${base}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message: "where is attemptUpload defined", force: "local" }),
  });
  assert.equal(chat.status, 200);
  assert.match(chat.headers.get("content-type") ?? "", /text\/event-stream/);
  const events = (await chat.text())
    .split("\n\n")
    .filter((frame) => frame.startsWith("data: "))
    .map((frame) => JSON.parse(frame.slice(6)) as { type: string; route?: string; text?: string });
  assert.ok(events.some((e) => e.type === "route" && e.route === "local"));
  const done = events.find((e) => e.type === "done");
  assert.ok(done, "stream must end with a done event");
  assert.match(String(done!.text), /uploader\.ts/);

  const stats = (await (await fetch(`${base}/stats`)).json()) as {
    router: { weights: unknown[] };
    memory: { lessons: number };
  };
  assert.ok(stats.router.weights.length > 0);
  assert.equal(stats.memory.lessons, 0);

  const trained = await fetch(`${base}/train`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ options: { vocabSize: 1024, embeddingDim: 0, valFraction: 0 } }),
  });
  assert.equal(trained.status, 200);
  const report = (await trained.json()) as { files: number; seconds: number };
  assert.ok(report.files > 0);
});
