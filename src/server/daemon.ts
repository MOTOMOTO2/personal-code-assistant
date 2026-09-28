import http from "node:http";
import { Bundle } from "../core/bundle.js";
import { CompletionEngine, CompletionRequest } from "../core/complete.js";
import { normalize } from "../ml/embeddings.js";
import { retrievalQuery } from "../llm/prompts.js";
import { Workspace } from "../util/paths.js";
import { color, info, ok, warn } from "../util/log.js";

/**
 * Loopback HTTP daemon.
 *
 * Model load is the dominant cost for a one-shot CLI invocation - the n-gram
 * tables are tens of megabytes. Keeping one warm process means the editor sees
 * only inference time, which is what makes inline completion viable at all.
 *
 * Binds to 127.0.0.1 exclusively and, when PCA_TOKEN is set, requires it as a
 * bearer token. Nothing here should ever be exposed off-host.
 */

export const DEFAULT_PORT = 7717;

export interface DaemonOptions {
  port?: number;
  /** Shared secret required in `Authorization: Bearer <token>`. */
  token?: string;
  /** Exit after this many idle milliseconds. 0 disables the timeout. */
  idleTimeoutMs?: number;
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

    if (token) {
      const auth = req.headers.authorization;
      if (auth !== `Bearer ${token}`) {
        send(401, { error: "unauthorized" });
        return;
      }
    }

    const url = new URL(req.url ?? "/", "http://127.0.0.1");

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
        index: bundle.index?.stats() ?? null,
        reranker: {
          updates: bundle.reranker.updates,
          accepted: bundle.reranker.accepted,
          rejected: bundle.reranker.rejected,
          weights: bundle.reranker.describe(),
        },
        style: bundle.style,
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
