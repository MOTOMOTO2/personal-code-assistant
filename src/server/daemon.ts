import crypto from "node:crypto";
import http from "node:http";
import os from "node:os";
import { Bundle } from "../core/bundle.js";
import { CompletionEngine, CompletionRequest } from "../core/complete.js";
import { ChatHub, HubEvent } from "../chat/hub.js";
import { Route } from "../chat/session.js";
import { isSafeThreadId } from "../chat/threads.js";
import { loadLessons } from "../chat/memory.js";
import { trainProject } from "../core/train.js";
import { normalize } from "../ml/embeddings.js";
import { Effort, hasCredentials } from "../llm/claude.js";
import { retrievalQuery } from "../llm/prompts.js";
import { CHAT_HTML, ICON_SVG, WEB_MANIFEST } from "../web/ui.js";
import { Workspace } from "../util/paths.js";
import { color, info, ok, warn } from "../util/log.js";

/**
 * HTTP daemon: one warm process serving the editor, the browser and your phone.
 *
 * Model load is the dominant cost for a one-shot CLI invocation - the n-gram
 * tables are tens of megabytes. Keeping one warm process means the editor sees
 * only inference time, which is what makes inline completion viable at all. The
 * chat rides on the same warm models, so a turn from the sofa costs no more than
 * an editor keystroke does.
 *
 * Security has two modes, because reachability and safety pull against each
 * other:
 *
 *   - Bound to loopback (the default), it is unauthenticated but unreachable
 *     from the network. The Host header is still checked, because a loopback
 *     bind alone does not stop a public DNS name from resolving to 127.0.0.1 and
 *     a page on that origin reaching this port.
 *   - Bound anywhere else - which is what makes it messageable from a phone - a
 *     token is mandatory. One is generated if you do not supply it, and every
 *     request needs it, so the Host check stops being the thing holding the door.
 *
 * It is still plain HTTP on your own network: fine for a laptop and a phone on
 * the same Wi-Fi, not something to port-forward to the internet.
 */

export const DEFAULT_PORT = 7717;
/** SSE keepalive. Phones and proxies drop connections that go quiet. */
const PING_MS = 25_000;

export interface DaemonOptions {
  port?: number;
  /** Interface to bind. Defaults to loopback; "0.0.0.0" exposes it to the LAN. */
  host?: string;
  /** Shared secret required in `Authorization: Bearer <token>` or `?token=`. */
  token?: string;
  /** Exit after this many idle milliseconds. 0 disables the timeout. */
  idleTimeoutMs?: number;
  /** Serve the browser chat client at `/`. */
  ui?: boolean;
  /** Default reasoning effort for Claude turns started over HTTP. */
  effort?: Effort;
}

interface FeedbackBody {
  /** Feature vector of the suggestion that was accepted. */
  accepted?: number[];
  /** Feature vectors of suggestions that were shown but not accepted. */
  rejected?: number[][];
  /** Set when a suggestion was explicitly dismissed with nothing accepted. */
  dismissed?: number[];
  filePath?: string;
}

export function isLoopbackHost(host: string): boolean {
  return host === "127.0.0.1" || host === "localhost" || host === "::1" || host === "[::1]";
}

/** IPv4 addresses this machine can be reached on, best guess first. */
export function lanAddresses(): string[] {
  const out: string[] = [];
  for (const entries of Object.values(os.networkInterfaces())) {
    for (const entry of entries ?? []) {
      if (entry.family !== "IPv4" || entry.internal) continue;
      out.push(entry.address);
    }
  }
  // 192.168.* is the likeliest home network; put those first.
  return out.sort((a, b) => Number(b.startsWith("192.168.")) - Number(a.startsWith("192.168.")));
}

function timingSafeEqual(a: string, b: string): boolean {
  const left = Buffer.from(a);
  const right = Buffer.from(b);
  if (left.length !== right.length) return false;
  return crypto.timingSafeEqual(left, right);
}

export function createDaemon(ws: Workspace, opts: DaemonOptions = {}) {
  const bundle = new Bundle(ws);
  const engine = new CompletionEngine(bundle);
  const host = opts.host ?? "127.0.0.1";
  const loopbackOnly = isLoopbackHost(host);
  // Off loopback the token is not optional: generate one rather than serving the
  // whole conversation history to anything on the network.
  const token =
    opts.token ??
    process.env.PCA_TOKEN ??
    (loopbackOnly ? null : crypto.randomBytes(16).toString("hex"));
  const idleTimeoutMs = opts.idleTimeoutMs ?? 0;
  const ui = opts.ui ?? false;
  const hub = new ChatHub(bundle, opts.effort ? { effort: opts.effort } : {});

  let lastActivity = Date.now();
  let requests = 0;
  let totalLatency = 0;

  const server = http.createServer((req, res) => {
    lastActivity = Date.now();
    const send = (status: number, body: unknown) => {
      const payload = JSON.stringify(body);
      res.writeHead(status, {
        "content-type": "application/json; charset=utf-8",
        "content-length": Buffer.byteLength(payload),
        "cache-control": "no-store",
      });
      res.end(payload);
    };

    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    const method = req.method ?? "GET";

    // Loopback bind: only loopback origins. Blocks DNS rebinding, which the bind
    // does not - a public hostname can resolve here, and the browser would send
    // its own name in Host. Off loopback the token does this job instead.
    if (loopbackOnly) {
      const hostHeader = (req.headers.host ?? "").split(":")[0];
      if (
        hostHeader &&
        hostHeader !== "127.0.0.1" &&
        hostHeader !== "localhost" &&
        hostHeader !== "[::1]" &&
        hostHeader !== "::1"
      ) {
        send(403, { error: "forbidden host" });
        return;
      }
    }

    // The token can arrive three ways, and all three are needed: a header for
    // the editor and `fetch`, a query parameter for the first page load and for
    // EventSource (which cannot set headers), and a cookie so an installed
    // home-screen app that launches at "/" is not locked out.
    // The manifest and the icon are exempt: a browser fetches a web manifest
    // without credentials, so gating them would break "Add to Home Screen", and
    // neither is secret.
    const publicAsset =
      method === "GET" && (url.pathname === "/manifest.webmanifest" || url.pathname === "/icon.svg");
    let authedByQuery = false;
    if (token && !publicAsset) {
      const auth = req.headers.authorization ?? "";
      const bearer = auth.startsWith("Bearer ") ? auth.slice(7) : "";
      const query = url.searchParams.get("token") ?? "";
      authedByQuery = timingSafeEqual(query, token);
      const cookie = readCookie(req.headers.cookie, "pca_token");
      if (!authedByQuery && !timingSafeEqual(bearer, token) && !timingSafeEqual(cookie, token)) {
        send(401, { error: "unauthorized" });
        return;
      }
    }

    if (method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
      if (!ui) {
        send(404, { error: "not found", hint: "start with `pca serve --ui` for the chat client" });
        return;
      }
      res.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "content-length": Buffer.byteLength(CHAT_HTML),
        "cache-control": "no-store",
        // Remember the token for later launches. SameSite=Strict so another
        // site cannot borrow it, HttpOnly because only the browser needs it.
        ...(authedByQuery && token
          ? {
              "set-cookie": `pca_token=${token}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict`,
            }
          : {}),
      });
      res.end(CHAT_HTML);
      return;
    }

    // Enough for "Add to Home Screen" to install it as its own app rather than
    // a bookmark, which is most of what makes it feel like a messaging client.
    if (method === "GET" && url.pathname === "/manifest.webmanifest") {
      const body = JSON.stringify(WEB_MANIFEST);
      res.writeHead(200, {
        "content-type": "application/manifest+json; charset=utf-8",
        "content-length": Buffer.byteLength(body),
      });
      res.end(body);
      return;
    }
    if (method === "GET" && url.pathname === "/icon.svg") {
      res.writeHead(200, {
        "content-type": "image/svg+xml; charset=utf-8",
        "content-length": Buffer.byteLength(ICON_SVG),
      });
      res.end(ICON_SVG);
      return;
    }

    if (method === "GET" && url.pathname === "/health") {
      send(200, {
        ok: true,
        root: ws.root,
        trained: bundle.isTrained,
        neural: bundle.isTrained ? bundle.neural !== null : false,
        embeddings: bundle.isTrained ? bundle.embeddings !== null : false,
        index: bundle.isTrained ? bundle.index !== null : false,
        requests,
        meanLatencyMs: requests > 0 ? Math.round((totalLatency / requests) * 100) / 100 : 0,
        conversations: hub.liveCount,
      });
      return;
    }

    if (method === "GET" && url.pathname === "/stats") {
      send(200, statsPayload());
      return;
    }

    if (method === "GET" && url.pathname === "/memory") {
      const store = loadLessons(bundle);
      const q = url.searchParams.get("q");
      const recalls = q
        ? store.recall(q, {
            limit: 8,
            tokenizer: bundle.isTrained ? bundle.tokenizer : undefined,
            embeddings: bundle.isTrained ? bundle.embeddings : null,
          })
        : store.lessons.slice(-8).reverse().map((lesson) => ({ lesson, score: 1 }));
      send(200, {
        total: store.size,
        lessons: recalls.map((r) => ({
          id: r.lesson.id,
          at: r.lesson.at,
          question: r.lesson.question,
          answer: r.lesson.answer,
          model: r.lesson.model,
          cites: r.lesson.cites,
          score: r.score,
        })),
      });
      return;
    }

    // Everything the chat client needs for a first paint, in one request.
    if (method === "GET" && url.pathname === "/api/state") {
      send(200, { ...statsPayload(), threads: hub.list(), trained: bundle.isTrained });
      return;
    }

    if (method === "GET" && url.pathname === "/api/threads") {
      send(200, { threads: hub.list() });
      return;
    }

    const threadRoute = matchThread(url.pathname);
    if (threadRoute) {
      if (!isSafeThreadId(threadRoute.id)) {
        send(400, { error: "invalid conversation id" });
        return;
      }
      if (method === "GET" && threadRoute.action === null) {
        const snapshot = hub.snapshot(threadRoute.id);
        if (!snapshot) {
          send(404, { error: "no such conversation" });
          return;
        }
        send(200, snapshot);
        return;
      }
      if (method === "GET" && threadRoute.action === "events") {
        streamEvents(threadRoute.id, req, res);
        return;
      }
      if (method === "DELETE" && threadRoute.action === null) {
        send(200, { removed: hub.remove(threadRoute.id) });
        return;
      }
    }

    const mutating = method === "POST" || method === "PATCH" || method === "PUT";
    if (!mutating) {
      send(404, { error: "not found" });
      return;
    }

    readBody(req, 4 * 1024 * 1024)
      .then((raw) => {
        let body: unknown;
        try {
          body = raw.length > 0 ? JSON.parse(raw) : {};
        } catch {
          send(400, { error: "invalid JSON body" });
          return;
        }

        if (url.pathname === "/complete") {
          const t0 = performance.now();
          const request = body as CompletionRequest;
          if (typeof request.prefix !== "string") {
            send(400, { error: "prefix is required" });
            return;
          }
          if (!bundle.isTrained) {
            send(409, { error: "not trained", hint: "run `pca train` in this project" });
            return;
          }
          try {
            const result = engine.complete(request);
            requests++;
            totalLatency += performance.now() - t0;
            send(200, result);
          } catch (err) {
            send(500, { error: (err as Error).message });
          }
          return;
        }

        if (url.pathname === "/feedback") {
          const fb = body as FeedbackBody;
          const reranker = bundle.reranker;
          if (Array.isArray(fb.accepted)) {
            reranker.learnPairwise(fb.accepted, (fb.rejected ?? []).map((r) => r));
            reranker.learnPointwise(fb.accepted, 1);
          } else if (Array.isArray(fb.dismissed)) {
            reranker.learnPointwise(fb.dismissed, 0);
          } else {
            send(400, { error: "expected `accepted` or `dismissed` feature vector" });
            return;
          }
          bundle.saveReranker();
          bundle.appendFeedback({ at: new Date().toISOString(), ...fb });
          send(200, { ok: true, updates: reranker.updates });
          return;
        }

        if (url.pathname === "/search") {
          const { query, limit } = body as { query?: string; limit?: number };
          if (typeof query !== "string" || query.length === 0) {
            send(400, { error: "query is required" });
            return;
          }
          const index = bundle.index;
          if (!index) {
            send(409, { error: "no index", hint: "run `pca train` in this project" });
            return;
          }
          const embeddings = bundle.embeddings;
          const q = retrievalQuery(query);
          const queryVector = embeddings
            ? normalize(embeddings.sifEmbed(bundle.tokenizer.encode(q)))
            : null;
          send(200, {
            hits: index.search(q, { limit: limit ?? 8, queryVector }).map((hit) => ({
              file: hit.file,
              startLine: hit.chunk.startLine,
              endLine: hit.chunk.endLine,
              symbol: hit.chunk.symbol,
              score: hit.score,
              text: hit.chunk.text,
            })),
          });
          return;
        }

        if (url.pathname === "/api/threads" && method === "POST") {
          const { title } = body as { title?: string };
          send(200, { thread: hub.create(title) });
          return;
        }

        const route = matchThread(url.pathname);
        if (route) {
          if (!isSafeThreadId(route.id)) {
            send(400, { error: "invalid conversation id" });
            return;
          }
          if (route.action === null && method === "PATCH") {
            const { title } = body as { title?: string };
            if (typeof title !== "string" || title.trim().length === 0) {
              send(400, { error: "title is required" });
              return;
            }
            const thread = hub.rename(route.id, title);
            if (!thread) {
              send(404, { error: "no such conversation" });
              return;
            }
            send(200, { thread });
            return;
          }
          if (route.action === "messages" && method === "POST") {
            const { text, force, effort, id } = body as {
              text?: string;
              force?: Route | null;
              effort?: Effort;
              id?: string;
            };
            if (typeof text !== "string" || text.trim().length === 0) {
              send(400, { error: "text is required" });
              return;
            }
            try {
              // Returns as soon as the message is accepted. The answer arrives on
              // the event stream, whether or not this client is still listening.
              const accepted = hub.send(route.id, text, {
                force: force ?? null,
                effort,
                id,
              });
              send(202, accepted);
            } catch (err) {
              send(400, { error: (err as Error).message });
            }
            return;
          }
          if (route.action === "escalate" && method === "POST") {
            try {
              const { effort, id } = body as { effort?: Effort; id?: string };
              send(202, hub.escalate(route.id, { effort, id }));
            } catch (err) {
              send(400, { error: (err as Error).message });
            }
            return;
          }
        }

        if (url.pathname === "/chat") {
          // Kept for the editor and older clients: post a message and hold the
          // connection open until that one turn finishes.
          const { message, force, sessionId } = body as {
            message?: string;
            force?: Route | null;
            sessionId?: string;
          };
          if (typeof message !== "string" || message.trim().length === 0) {
            send(400, { error: "message is required" });
            return;
          }
          const id = typeof sessionId === "string" && sessionId.length > 0 ? sessionId : "default";
          if (!isSafeThreadId(id)) {
            send(400, { error: "invalid sessionId" });
            return;
          }
          legacyChat(id, message, force ?? null, res);
          return;
        }

        if (url.pathname === "/train") {
          // Synchronous and seconds-long: it blocks this daemon's event loop on
          // purpose, because serving completions from half-written artifacts
          // would be worse than a brief stall.
          try {
            const report = trainProject(bundle, (body as { options?: object }).options ?? {});
            bundle.reload();
            engine.warm({ retrieval: true });
            hub.refreshMemory();
            send(200, report);
          } catch (err) {
            send(500, { error: (err as Error).message });
          }
          return;
        }

        if (url.pathname === "/shutdown") {
          send(200, { ok: true });
          setTimeout(() => server.close(), 10);
          return;
        }

        send(404, { error: "not found" });
      })
      .catch((err: Error) => send(413, { error: err.message }));
  });

  function statsPayload() {
    return {
      manifest: bundle.manifest,
      index: bundle.isTrained ? (bundle.index?.stats() ?? null) : null,
      reranker: {
        updates: bundle.reranker.updates,
        accepted: bundle.reranker.accepted,
        rejected: bundle.reranker.rejected,
        weights: bundle.reranker.describe(),
      },
      router: {
        updates: bundle.router.updates,
        toLocal: bundle.router.toLocal,
        toClaude: bundle.router.toClaude,
        weights: bundle.router.describe(),
      },
      memory: { lessons: loadLessons(bundle).size },
      claudeAvailable: hasCredentials(),
      style: bundle.style,
    };
  }

  /** Server-sent events for one conversation: snapshot first, then live turns. */
  function streamEvents(id: string, req: http.IncomingMessage, res: http.ServerResponse): void {
    res.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-store",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    res.flushHeaders?.();
    const write = (event: HubEvent) => {
      if (!res.writableEnded) res.write(`data: ${JSON.stringify(event)}\n\n`);
    };
    const unsubscribe = hub.subscribe(id, write);
    const ping = setInterval(() => {
      if (!res.writableEnded) res.write(": ping\n\n");
    }, PING_MS);
    ping.unref();
    const stop = () => {
      clearInterval(ping);
      unsubscribe();
      if (!res.writableEnded) res.end();
    };
    req.on("close", stop);
    req.on("error", stop);
  }

  /** The pre-thread `/chat` shape, expressed on top of the hub. */
  function legacyChat(id: string, message: string, force: Route | null, res: http.ServerResponse) {
    res.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-store",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    const emit = (event: unknown) => {
      if (!res.writableEnded) res.write(`data: ${JSON.stringify(event)}\n\n`);
    };
    // Only this request's own turn: another device could be mid-answer in the
    // same conversation, and this client asked about one message.
    let jobId: string | null = null;
    const mine = (event: { id?: string }) => jobId === null || event.id === jobId;
    const unsubscribe = hub.subscribe(id, (event) => {
      switch (event.type) {
        case "route":
          if (mine(event)) emit({ type: "route", route: event.route, pClaude: event.pClaude, why: event.why });
          break;
        case "thinking":
          if (mine(event)) emit({ type: "thinking", text: event.text });
          break;
        case "text":
          if (mine(event)) emit({ type: "text", text: event.text });
          break;
        case "done":
          if (!mine(event)) break;
          emit({
            type: "done",
            route: event.message.route,
            kind: event.message.kind,
            // Local answers are not streamed, so the text arrives here.
            text: event.message.route === "local" ? event.message.content : "",
            sources: event.message.sources ?? [],
            costUsd: event.message.costUsd ?? 0,
            lessonId: event.lessonId,
            confidence: event.message.route === "claude" ? 1 : undefined,
          });
          finish();
          break;
        case "error":
          if (!mine(event)) break;
          emit({ type: "error", message: event.message });
          finish();
          break;
        default:
          break;
      }
    });
    const finish = () => {
      unsubscribe();
      if (!res.writableEnded) res.end();
    };
    res.on("close", () => unsubscribe());
    try {
      // A local turn can finish inside this call, before `jobId` is assigned -
      // which is why `mine()` treats "not yet known" as "mine".
      jobId = hub.send(id, message, { force }).id;
    } catch (err) {
      emit({ type: "error", message: (err as Error).message });
      finish();
    }
  }

  let idleTimer: NodeJS.Timeout | null = null;
  if (idleTimeoutMs > 0) {
    idleTimer = setInterval(() => {
      if (Date.now() - lastActivity > idleTimeoutMs) {
        info("idle timeout reached, shutting down");
        server.close();
      }
    }, 10_000);
    idleTimer.unref();
  }

  server.on("close", () => {
    if (idleTimer) clearInterval(idleTimer);
  });

  return {
    server,
    hub,
    token,
    host,
    /** URLs this daemon can be opened at, token included when there is one. */
    urls(port: number): string[] {
      const query = token ? `/?token=${token}` : "/";
      if (loopbackOnly) return [`http://127.0.0.1:${port}${query}`];
      const addresses = lanAddresses();
      return (addresses.length > 0 ? addresses : ["127.0.0.1"]).map(
        (address) => `http://${address}:${port}${query}`,
      );
    },
    listen(port = opts.port ?? DEFAULT_PORT): Promise<number> {
      return new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(port, host, () => {
          const address = server.address();
          const actual = typeof address === "object" && address ? address.port : port;
          if (bundle.isTrained) {
            // Page the models in now so the first editor keystroke is not the
            // request that pays for loading them.
            const t0 = performance.now();
            engine.warm({ neural: true, retrieval: true });
            info(color.dim(`models warm in ${Math.round(performance.now() - t0)}ms`));
          }
          ok(`pca daemon listening on ${color.bold(`http://${host}:${actual}`)}`);
          if (!bundle.isTrained) {
            warn("this project is not trained yet - run `pca train`");
          }
          if (loopbackOnly && !token) {
            info(color.dim("loopback only; set PCA_TOKEN or use --lan to reach it from a phone"));
          }
          resolve(actual);
        });
      });
    },
    close(): Promise<void> {
      return new Promise((resolve) => server.close(() => resolve()));
    },
  };
}

/** `/api/threads/<id>` and `/api/threads/<id>/<action>`. */
function matchThread(pathname: string): { id: string; action: string | null } | null {
  const parts = pathname.split("/").filter((p) => p.length > 0);
  if (parts.length < 3 || parts[0] !== "api" || parts[1] !== "threads") return null;
  if (parts.length > 4) return null;
  return { id: decodeURIComponent(parts[2]), action: parts[3] ?? null };
}

function readCookie(header: string | undefined, name: string): string {
  if (!header) return "";
  for (const part of header.split(";")) {
    const eq = part.indexOf("=");
    if (eq < 0) continue;
    if (part.slice(0, eq).trim() === name) return part.slice(eq + 1).trim();
  }
  return "";
}

function readBody(req: http.IncomingMessage, limit: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error("request body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}
