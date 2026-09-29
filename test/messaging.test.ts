import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { Bundle } from "../src/core/bundle.js";
import { ChatBackend, ChatSession } from "../src/chat/session.js";
import { ChatHub, HubEvent } from "../src/chat/hub.js";
import { ThreadStore, isSafeThreadId, newThreadId, titleFrom } from "../src/chat/threads.js";
import { StreamOptions, StreamResult } from "../src/llm/claude.js";
import { trainProject } from "../src/core/train.js";
import { createDaemon, isLoopbackHost } from "../src/server/daemon.js";
import { setQuiet } from "../src/util/log.js";
import { workspace } from "../src/util/paths.js";

setQuiet(true);

/**
 * These tests are about the messaging properties specifically: that a
 * conversation outlives the process, that an answer is not tied to the
 * connection that asked for it, and that two clients in one conversation see the
 * same thing. Model quality is covered elsewhere.
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
}

function setupProject(t: { after(fn: () => void): void }): Bundle {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pca-msg-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  writeFixtureProject(root);
  const bundle = new Bundle(workspace(root));
  trainProject(bundle, { vocabSize: 1024, embeddingDim: 16, embeddingEpochs: 2, valFraction: 0 });
  return bundle;
}

/** A backend that streams canned text, so the Claude path is testable offline. */
function fakeBackend(text: string): ChatBackend & { calls: StreamOptions[] } {
  const calls: StreamOptions[] = [];
  return {
    model: "claude-opus-5",
    calls,
    async stream(opts: StreamOptions): Promise<StreamResult> {
      calls.push(opts);
      // Yield to the event loop first, so a caller that does not await the turn
      // genuinely observes the "in flight" state.
      await new Promise((resolve) => setTimeout(resolve, 1));
      for (const chunk of text.match(/.{1,16}/gs) ?? []) opts.onText?.(chunk);
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

/** Wait for a hub event matching `want`, or fail the test on timeout. */
function waitFor(
  hub: ChatHub,
  id: string,
  want: (event: HubEvent) => boolean,
  label: string,
): Promise<HubEvent> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      stop();
      reject(new Error(`timed out waiting for ${label}`));
    }, 5000);
    const stop = hub.subscribe(id, (event) => {
      if (!want(event)) return;
      clearTimeout(timer);
      // Unsubscribing from inside a callback is allowed; the hub delivers each
      // event over a copy of its subscriber set.
      stop();
      resolve(event);
    });
  });
}

test("conversations outlive the process, and are listed, renamed and deleted", (t) => {
  const bundle = setupProject(t);
  const store = new ThreadStore(bundle.ws);
  assert.deepEqual(store.list(), []);

  const first = new ChatSession(bundle, { title: "before" });
  first.turns.push({ role: "user", content: "where is attemptUpload", at: new Date().toISOString() });
  first.save();

  // Reopening by id must bring the turns back - this is what makes the phone and
  // the terminal the same conversation rather than two.
  const reopened = ChatSession.open(bundle, first.id);
  assert.equal(reopened.turns.length, 1);
  assert.equal(reopened.turns[0].content, "where is attemptUpload");
  assert.equal(reopened.title, "before");

  const listed = store.list();
  assert.equal(listed.length, 1);
  assert.equal(listed[0].id, first.id);
  assert.equal(listed[0].messages, 1);
  assert.match(listed[0].preview, /attemptUpload/);

  assert.equal(store.rename(first.id, "  renamed  ")!.title, "renamed");
  assert.equal(store.load(first.id)!.title, "renamed");
  assert.equal(store.latest()!.id, first.id);
  assert.equal(store.remove(first.id), true);
  assert.deepEqual(store.list(), []);
  assert.equal(store.load(first.id), null);

  // A transcript written before conversations had names still opens, and gets a
  // name from its first message.
  const legacyId = "2026-09-28T20-46-01-123Z";
  fs.mkdirSync(path.join(bundle.ws.stateDir, "chat"), { recursive: true });
  fs.writeFileSync(
    path.join(bundle.ws.stateDir, "chat", `${legacyId}.json`),
    JSON.stringify({
      id: legacyId,
      turns: [{ role: "user", content: "why does the upload retry twice?", at: "2026-09-28T20:46:01.000Z" }],
      costUsd: 0.5,
    }),
  );
  const legacy = store.load(legacyId)!;
  assert.equal(legacy.title, "why does the upload retry twice?");
  assert.equal(legacy.costUsd, 0.5);
  assert.equal(legacy.createdAt, "2026-09-28T20:46:01.000Z");

  // Unreadable files are skipped rather than taking the list down with them.
  fs.writeFileSync(path.join(bundle.ws.stateDir, "chat", "broken.json"), "{ not json");
  assert.equal(store.load("broken"), null);
  assert.equal(store.list().length, 1);
});

test("ids that arrive over the network cannot escape the state directory", (t) => {
  const bundle = setupProject(t);
  const store = new ThreadStore(bundle.ws);

  assert.equal(isSafeThreadId("web-abc123"), true);
  assert.equal(isSafeThreadId(newThreadId()), true);
  assert.equal(isSafeThreadId("2026-09-28T20-46-01-123Z"), true);
  for (const bad of ["../../evil", "..", "a/b", "a\\b", "", ".hidden", "x".repeat(65), "a:b"]) {
    assert.equal(isSafeThreadId(bad), false, `${JSON.stringify(bad)} must be rejected`);
    assert.throws(() => store.file(bad), /Invalid conversation id/);
  }
  assert.equal(store.remove("../../evil"), false);
  assert.equal(store.load("../../evil"), null);

  assert.equal(titleFrom("  a\n  b  "), "a b");
  assert.equal(titleFrom("x".repeat(90)).length, 60);
  assert.equal(titleFrom("   "), "New conversation");
});

test("a message is answered whether or not anyone is still listening", async (t) => {
  const bundle = setupProject(t);
  const backend = fakeBackend(
    "The retry lives in `retryFailedUpload` at src/uploader.ts:2, which loops over `attempts`.",
  );
  const hub = new ChatHub(bundle, { backend });
  const thread = hub.create();

  // Post and immediately walk away, as a phone does when the screen locks.
  const accepted = hub.send(thread.id, "why does the upload retry twice?", { force: "claude" });
  assert.ok(accepted.id);
  assert.notEqual(accepted.status, "done");

  const done = await waitFor(hub, thread.id, (e) => e.type === "done", "the answer");
  assert.equal(done.type, "done");
  if (done.type !== "done") return;
  assert.equal(done.id, accepted.id, "events carry the id of the message they answer");
  assert.equal(done.message.route, "claude");
  assert.match(done.message.content, /retryFailedUpload/);
  assert.ok(done.message.costUsd && done.message.costUsd > 0);

  // Attaching afterwards - the "I came back to the app" case - must produce the
  // finished conversation, not an empty one.
  const snapshot = hub.snapshot(thread.id)!;
  assert.equal(snapshot.pending, null);
  assert.equal(snapshot.messages.length, 2);
  assert.equal(snapshot.messages[0].role, "user");
  assert.equal(snapshot.messages[1].role, "assistant");
  assert.deepEqual(
    snapshot.messages.map((m) => m.seq),
    [0, 1],
  );
  assert.equal(snapshot.thread.title, "why does the upload retry twice?");

  // And it is on disk, so a daemon restart does not lose it.
  const stored = new ThreadStore(bundle.ws).load(thread.id)!;
  assert.equal(stored.turns.length, 2);
  assert.match(stored.turns[1].content, /retryFailedUpload/);
  assert.ok(stored.costUsd > 0);
});

test("a second device sees the same conversation, live", async (t) => {
  const bundle = setupProject(t);
  const hub = new ChatHub(bundle, { backend: fakeBackend("Because the caller retries too.") });
  const thread = hub.create();

  const phone: string[] = [];
  const laptop: string[] = [];
  const stopPhone = hub.subscribe(thread.id, (e) => phone.push(e.type));
  const stopLaptop = hub.subscribe(thread.id, (e) => laptop.push(e.type));
  t.after(() => {
    stopPhone();
    stopLaptop();
  });

  // Both got a snapshot on attach.
  assert.deepEqual(phone, ["snapshot"]);
  assert.deepEqual(laptop, ["snapshot"]);

  hub.send(thread.id, "why does the upload retry twice?", { force: "claude" });
  await waitFor(hub, thread.id, (e) => e.type === "done", "the answer");

  assert.deepEqual(phone, laptop, "both clients must see the same event sequence");
  assert.deepEqual(phone.slice(0, 3), ["snapshot", "user", "route"]);
  assert.ok(phone.includes("text"), "streamed text reaches every attached client");
  assert.equal(phone[phone.length - 1], "done");
});

test("messages sent in a burst queue up and are answered in order", async (t) => {
  const bundle = setupProject(t);
  const hub = new ChatHub(bundle, { backend: fakeBackend("An answer long enough to be kept.") });
  const thread = hub.create();

  const first = hub.send(thread.id, "first question", { force: "claude" });
  const second = hub.send(thread.id, "second question", { force: "claude" });
  const third = hub.send(thread.id, "third question", { force: "claude" });
  assert.equal(second.status, "queued");
  assert.equal(third.status, "queued");

  const queued = hub.snapshot(thread.id)!.queued;
  assert.deepEqual(
    queued.map((q) => q.question),
    ["second question", "third question"],
  );

  let answers = 0;
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("timed out waiting for three answers")), 8000);
    const stop = hub.subscribe(thread.id, (event) => {
      if (event.type !== "done") return;
      if (++answers < 3) return;
      clearTimeout(timer);
      setTimeout(stop, 0);
      resolve();
    });
  });

  const snapshot = hub.snapshot(thread.id)!;
  assert.equal(snapshot.pending, null);
  assert.deepEqual(snapshot.queued, []);
  assert.deepEqual(
    snapshot.messages.filter((m) => m.role === "user").map((m) => m.content),
    ["first question", "second question", "third question"],
  );
  assert.equal(snapshot.messages.length, 6, "one answer per question, in order");
  // The first message was not queued behind anything: it was already being
  // answered by the time `send` returned.
  assert.notEqual(first.status, "queued");
});

test("a local answer can be escalated to Claude in place", async (t) => {
  const bundle = setupProject(t);
  const backend = fakeBackend(
    "It retries because `retryFailedUpload` loops over `attempts` at src/uploader.ts:2 and the caller retries once as well.",
  );
  const hub = new ChatHub(bundle, { backend });
  const thread = hub.create();

  hub.send(thread.id, "where is attemptUpload defined", { force: "local" });
  await waitFor(hub, thread.id, (e) => e.type === "done", "the local answer");
  let snapshot = hub.snapshot(thread.id)!;
  assert.equal(snapshot.messages.length, 2);
  assert.equal(snapshot.messages[1].route, "local");
  assert.equal(backend.calls.length, 0, "the local half must not call the API");

  const updatesBefore = bundle.router.updates;
  hub.escalate(thread.id);
  const done = await waitFor(hub, thread.id, (e) => e.type === "done", "the escalated answer");
  assert.equal(done.type === "done" && done.message.route, "claude");
  assert.equal(backend.calls.length, 1);
  assert.ok(bundle.router.updates > updatesBefore, "escalating teaches the router");

  // The answer replaces the local attempt at the same position rather than
  // appending, so the conversation keeps one answer per question.
  snapshot = hub.snapshot(thread.id)!;
  assert.equal(snapshot.messages.length, 2);
  assert.equal(snapshot.messages[1].route, "claude");
  assert.equal(done.type === "done" && done.message.seq, 1);

  // Nothing to escalate in a conversation with no messages.
  const empty = hub.create();
  assert.throws(() => hub.escalate(empty.id), /send a message first/);
  assert.throws(() => hub.send(thread.id, "   "), /cannot be empty/);
  assert.throws(() => hub.send("../../evil", "hello"), /Invalid conversation id/);
});

test("an answer remembered in one conversation is available in the others", async (t) => {
  const bundle = setupProject(t);
  const hub = new ChatHub(bundle, {
    backend: fakeBackend(
      "Uploads are quarantined for fourteen days in the quarantine bucket, then purged by the nightly sweeper. That is a policy decision, not something the code states.",
    ),
  });
  const a = hub.create();
  const b = hub.create();

  // Open both, so the second one has already loaded memory from disk.
  hub.subscribe(b.id, () => {})();
  hub.send(a.id, "what is the quarantine policy?", { force: "claude" });
  await waitFor(hub, a.id, (e) => e.type === "done", "the answer in the first conversation");

  hub.send(b.id, "what is the quarantine policy?");
  const done = await waitFor(hub, b.id, (e) => e.type === "done", "the answer in the second");
  assert.equal(done.type === "done" && done.message.route, "local");
  assert.equal(done.type === "done" && done.message.kind, "memory");
  assert.match(done.type === "done" ? done.message.content : "", /fourteen days/);
});

test("deleting a conversation mid-answer does not bring it back", async (t) => {
  const bundle = setupProject(t);
  const hub = new ChatHub(bundle, { backend: fakeBackend("An answer nobody is waiting for.") });
  const thread = hub.create();
  const store = new ThreadStore(bundle.ws);

  hub.send(thread.id, "a question that is still being answered", { force: "claude" });
  assert.equal(hub.remove(thread.id), true);
  assert.equal(store.load(thread.id), null);

  // The turn is already in flight and will finish; its final save must not
  // recreate the file for the conversation that was just deleted.
  await new Promise((resolve) => setTimeout(resolve, 200));
  assert.equal(store.load(thread.id), null, "a deleted conversation must stay deleted");
  assert.equal(store.list().length, 0);
});

test("the daemon serves conversations over HTTP, and streams one as it happens", async (t) => {
  const bundle = setupProject(t);
  const daemon = createDaemon(bundle.ws, { port: 0, ui: true });
  const port = await daemon.listen(0);
  t.after(async () => {
    await daemon.close();
  });
  const base = `http://127.0.0.1:${port}`;
  const json = async (method: string, route: string, body?: unknown) => {
    const res = await fetch(`${base}${route}`, {
      method,
      headers: { "content-type": "application/json" },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    return { status: res.status, body: (await res.json()) as Record<string, any> };
  };

  // First paint: everything the client needs in one request.
  const state = await json("GET", "/api/state");
  assert.equal(state.status, 200);
  assert.equal(state.body.trained, true);
  assert.deepEqual(state.body.threads, []);
  assert.ok(state.body.manifest);

  const created = await json("POST", "/api/threads", { title: "on the sofa" });
  assert.equal(created.status, 200);
  const id = created.body.thread.id as string;
  assert.equal(created.body.thread.title, "on the sofa");
  const listed = await json("GET", "/api/threads");
  assert.deepEqual(
    (listed.body.threads as Array<{ id: string }>).map((thread) => thread.id),
    [id],
  );

  // Attach to the stream first, then post - the order a client uses.
  const events: Array<Record<string, any>> = [];
  const stream = await fetch(`${base}/api/threads/${id}/events`);
  assert.equal(stream.status, 200);
  assert.match(stream.headers.get("content-type") ?? "", /text\/event-stream/);
  const reader = stream.body!.getReader();
  const decoder = new TextDecoder();
  const pump = (async () => {
    let buffer = "";
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) return;
      buffer += decoder.decode(chunk.value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        if (!frame.startsWith("data: ")) continue;
        events.push(JSON.parse(frame.slice(6)));
        if (events[events.length - 1].type === "done") return;
      }
    }
  })();

  const posted = await json("POST", `/api/threads/${id}/messages`, {
    text: "where is attemptUpload defined",
    force: "local",
  });
  assert.equal(posted.status, 202, "posting a message returns immediately");
  assert.ok(posted.body.id);

  await pump;
  await reader.cancel().catch(() => {});
  assert.equal(events[0].type, "snapshot");
  assert.deepEqual(events.map((e) => e.type).slice(1), ["user", "route", "done"]);
  const done = events[events.length - 1];
  assert.equal(done.message.route, "local");
  assert.match(String(done.message.content), /uploader\.ts/);
  assert.equal(done.id, posted.body.id, "the stream says which message it answered");

  // Catch-up: the same conversation fetched cold has both messages.
  const snapshot = await json("GET", `/api/threads/${id}`);
  assert.equal(snapshot.status, 200);
  assert.equal((snapshot.body.messages as unknown[]).length, 2);
  assert.equal(snapshot.body.pending, null);

  // Renaming, deleting, and the refusal to touch anything outside .pca/chat/.
  assert.equal((await json("PATCH", `/api/threads/${id}`, { title: "renamed" })).body.thread.title, "renamed");
  assert.equal((await json("GET", "/api/threads/..%2F..%2Fevil")).status, 400);
  assert.equal((await json("POST", "/api/threads/..%2F..%2Fevil/messages", { text: "hi" })).status, 400);
  assert.equal((await json("GET", "/api/threads/nope-not-here")).status, 404);
  assert.equal((await json("POST", `/api/threads/${id}/messages`, { text: "  " })).status, 400);
  assert.equal((await json("DELETE", `/api/threads/${id}`)).body.removed, true);
  assert.deepEqual((await json("GET", "/api/threads")).body.threads, []);

  // Installable: the manifest and icon the home-screen shortcut needs.
  const manifest = await fetch(`${base}/manifest.webmanifest`);
  assert.equal(manifest.status, 200);
  assert.equal(((await manifest.json()) as { start_url: string }).start_url, "/");
  const icon = await fetch(`${base}/icon.svg`);
  assert.equal(icon.status, 200);
  assert.match(icon.headers.get("content-type") ?? "", /image\/svg/);
});

test("off loopback, a token is mandatory and a cookie is enough to carry it", async (t) => {
  const bundle = setupProject(t);

  // Binding anywhere but loopback must not be possible without a secret, so one
  // is generated when the caller does not supply it. (Not listened to: the point
  // is the policy, not the socket.)
  const exposed = createDaemon(bundle.ws, { port: 0, host: "0.0.0.0", ui: true });
  assert.ok(exposed.token && exposed.token.length >= 32, "a LAN daemon must have a token");
  assert.equal(exposed.host, "0.0.0.0");
  assert.equal(isLoopbackHost("127.0.0.1"), true);
  assert.equal(isLoopbackHost("0.0.0.0"), false);
  assert.match(exposed.urls(7717)[0], /^http:\/\/[\d.]+:7717\/\?token=/);

  const daemon = createDaemon(bundle.ws, { port: 0, ui: true, token: "sekrit" });
  const port = await daemon.listen(0);
  t.after(async () => {
    await daemon.close();
  });
  const base = `http://127.0.0.1:${port}`;
  assert.equal((await fetch(`${base}/api/threads`)).status, 401);
  assert.equal((await fetch(`${base}/api/threads?token=wrong`)).status, 401);
  assert.equal((await fetch(`${base}/api/threads?token=sekrit`)).status, 200);
  assert.equal(
    (await fetch(`${base}/api/threads`, { headers: { authorization: "Bearer sekrit" } })).status,
    200,
  );

  // The page load hands the token to the browser as a cookie, so an installed
  // home-screen app that starts at "/" is not locked out.
  const page = await fetch(`${base}/?token=sekrit`);
  assert.equal(page.status, 200);
  assert.match(page.headers.get("set-cookie") ?? "", /pca_token=sekrit; Path=\/;.*HttpOnly/);
  assert.equal((await fetch(`${base}/`, { headers: { cookie: "pca_token=sekrit" } })).status, 200);
  assert.equal((await fetch(`${base}/`, { headers: { cookie: "pca_token=no" } })).status, 401);
  // The web manifest and icon are fetched by the browser without credentials, so
  // they must not be behind the token or the app cannot be installed.
  assert.equal((await fetch(`${base}/manifest.webmanifest`)).status, 200);
  assert.equal((await fetch(`${base}/icon.svg`)).status, 200);
  assert.equal((await fetch(`${base}/health`)).status, 401, "everything else still needs it");

  // A plain loopback page load sets no cookie, because it needed no token.
  const plain = createDaemon(bundle.ws, { port: 0, ui: true });
  const plainPort = await plain.listen(0);
  t.after(async () => {
    await plain.close();
  });
  const plainPage = await fetch(`http://127.0.0.1:${plainPort}/`);
  assert.equal(plainPage.headers.get("set-cookie"), null);
});
