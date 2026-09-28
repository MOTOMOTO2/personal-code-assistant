import http from "node:http";
import { Bundle } from "../core/bundle.js";
import { CompletionEngine, CompletionRequest } from "../core/complete.js";
import { ChatSession, Route } from "../chat/session.js";
import { loadLessons } from "../chat/memory.js";
import { trainProject } from "../core/train.js";
import { normalize } from "../ml/embeddings.js";
import { hasCredentials } from "../llm/claude.js";
import { retrievalQuery } from "../llm/prompts.js";
import { CHAT_HTML } from "../web/ui.js";
import { Workspace } from "../util/paths.js";
import { color, info, ok, warn } from "../util/log.js";

/**
 * Loopback HTTP daemon: one warm process serving the editor and the browser.
 *
 * Model load is the dominant cost for a one-shot CLI invocation - the n-gram
 * tables are tens of megabytes. Keeping one warm process means the editor sees
 * only inference time, which is what makes inline completion viable at all. The
 * chat UI rides on the same warm models, so a browser turn costs no more than
 * an editor keystroke does.
 *
 * Binds to 127.0.0.1 exclusively and, when PCA_TOKEN is set, requires it as a
 * bearer token. Because a browser now talks to it, the Host header is checked
 * as well: a loopback bind alone does not stop a public DNS name from resolving
 * to 127.0.0.1 and a page on that origin reaching this port.
 */

export const DEFAULT_PORT = 7717;

export interface DaemonOptions {
  port?: number;
  /** Shared secret required in `Authorization: Bearer <token>`. */
  token?: string;
  /** Exit after this many idle milliseconds. 0 disables the timeout. */
  idleTimeoutMs?: number;
  /** Serve the browser chat client at `/`. */
  ui?: boolean;
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

export function createDaemon(ws: Workspace, opts: DaemonOptions = {}) {
  const bundle = new Bundle(ws);
  const engine = new CompletionEngine(bundle);
  const token = opts.token ?? process.env.PCA_TOKEN ?? null;
  const idleTimeoutMs = opts.idleTimeoutMs ?? 0;
  const ui = opts.ui ?? false;
  /** One conversation per browser tab, keyed by the id the client sends. */
  const sessions = new Map<string, ChatSession>();

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
        // The daemon is loopback-only; no browser should be talking to it.
        "cache-control": "no-store",
      });
      res.end(payload);
    };

    const url = new URL(req.url ?? "/", "http://127.0.0.1");

    // Only loopback origins. Blocks DNS rebinding, which a 127.0.0.1 bind does
    // not: a public hostname can resolve here, and the browser would send its
    // own name in Host.
    const host = (req.headers.host ?? "").split(":")[0];
    if (host && host !== "127.0.0.1" && host !== "localhost" && host !== "[::1]" && host !== "::1") {
      send(403, { error: "forbidden host" });
      return;
    }

    if (token) {
      // The query parameter is for the browser client, which cannot set a
      // header on the initial page load. Loopback-only, so the URL never
      // leaves this machine.
      const auth = req.headers.authorization;
      if (auth !== `Bearer ${token}` && url.searchParams.get("token") !== token) {
        send(401, { error: "unauthorized" });
        return;
      }
    }

    if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
      if (!ui) {
        send(404, { error: "not found", hint: "start with `pca serve --ui` for the chat client" });
        return;
      }
      res.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "content-length": Buffer.byteLength(CHAT_HTML),
        "cache-control": "no-store",
      });
      res.end(CHAT_HTML);
      return;
    }

    if (req.method === "GET" && url.pathname === "/health") {
      send(200, {
        ok: true,
        root: ws.root,
        trained: bundle.isTrained,
        neural: bundle.isTrained ? bundle.neural !== null : false,
        embeddings: bundle.isTrained ? bundle.embeddings !== null : false,
        index: bundle.isTrained ? bundle.index !== null : false,
        requests,
        meanLatencyMs: requests > 0 ? Math.round((totalLatency / requests) * 100) / 100 : 0,
      });
      return;
    }

    if (req.method === "GET" && url.pathname === "/stats") {
      send(200, {
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
      });
      return;
    }

    if (req.method === "GET" && url.pathname === "/memory") {
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

    if (req.method !== "POST") {
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

        if (url.pathname === "/chat") {
          const { message, force, sessionId } = body as {
            message?: string;
            force?: Route | null;
            sessionId?: string;
          };
          if (typeof message !== "string" || message.trim().length === 0) {
            send(400, { error: "message is required" });
            return;
          }
          const key = typeof sessionId === "string" && sessionId.length > 0 ? sessionId : "default";
          let session = sessions.get(key);
          if (!session) {
            session = new ChatSession(bundle, { id: key });
            sessions.set(key, session);
          }
          // Server-sent events: one JSON object per `data:` frame. The turn can
          // take tens of seconds, and a buffered response would look hung.
          res.writeHead(200, {
            "content-type": "text/event-stream; charset=utf-8",
            "cache-control": "no-store",
            connection: "keep-alive",
            "x-accel-buffering": "no",
          });
          const emit = (event: unknown) => {
            if (!res.writableEnded) res.write(`data: ${JSON.stringify(event)}\n\n`);
          };
          session
            .ask(message, {
              force: force ?? null,
              onRoute: (route, pClaude, why) => emit({ type: "route", route, pClaude, why }),
              onThinking: (text) => emit({ type: "thinking", text }),
              onText: (text) => emit({ type: "text", text }),
            })
            .then((result) => {
              emit({
                type: "done",
                route: result.route,
                kind: result.kind,
                // Local answers are not streamed, so the text arrives here.
                text: result.route === "local" ? result.text : "",
                sources: result.sources,
                costUsd: result.costUsd,
                lessonId: result.lessonId,
                confidence: result.confidence,
              });
              res.end();
            })
            .catch((err: Error) => {
              emit({ type: "error", message: err.message });
              res.end();
            });
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
            for (const session of sessions.values()) session.refreshMemory();
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
    listen(port = opts.port ?? DEFAULT_PORT): Promise<number> {
      return new Promise((resolve, reject) => {
        server.once("error", reject);
        // Loopback only - never 0.0.0.0.
        server.listen(port, "127.0.0.1", () => {
          const address = server.address();
          const actual = typeof address === "object" && address ? address.port : port;
          if (bundle.isTrained) {
            // Page the models in now so the first editor keystroke is not the
            // request that pays for loading them.
            const t0 = performance.now();
            engine.warm({ neural: true, retrieval: true });
            info(color.dim(`models warm in ${Math.round(performance.now() - t0)}ms`));
          }
          ok(`pca daemon listening on ${color.bold(`http://127.0.0.1:${actual}`)}`);
          if (!bundle.isTrained) {
            warn("this project is not trained yet - run `pca train`");
          }
          if (!token) {
            info(color.dim("set PCA_TOKEN to require a bearer token"));
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
