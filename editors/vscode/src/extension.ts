import * as vscode from "vscode";

/**
 * Thin editor client for the `pca serve` daemon.
 *
 * The extension is deliberately dumb: it sends the text around the cursor and
 * renders what comes back. All the modelling lives in the daemon, so retraining
 * or swapping models needs no extension reload, and the completion path stays
 * a single localhost round trip.
 *
 * Accepted suggestions are reported back to `/feedback`, which is what trains
 * the ranker on your preferences over time.
 */

/** Characters of context sent either side of the cursor. */
const CONTEXT_BEFORE = 4000;
const CONTEXT_AFTER = 1000;

interface Suggestion {
  text: string;
  score: number;
  features: number[];
}

interface CompletionResponse {
  suggestions: Suggestion[];
  timing: { total: number };
}

interface HealthResponse {
  ok: boolean;
  root: string;
  trained: boolean;
  neural: boolean;
  index: boolean;
  requests: number;
  meanLatencyMs: number;
}

function config() {
  const c = vscode.workspace.getConfiguration("pca");
  return {
    port: c.get<number>("port", 7717),
    token: c.get<string>("token", ""),
    enabled: c.get<boolean>("enabled", true),
    neural: c.get<boolean>("neural", false),
    debounceMs: c.get<number>("debounceMs", 40),
    maxSuggestions: c.get<number>("maxSuggestions", 3),
  };
}

async function callDaemon<T>(
  route: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<T | null> {
  const { port, token } = config();
  try {
    const res = await fetch(`http://127.0.0.1:${port}${route}`, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      signal,
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    // Daemon down, or the request was superseded by a newer keystroke.
    return null;
  }
}

export function activate(context: vscode.ExtensionContext): void {
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  status.command = "pca.status";
  context.subscriptions.push(status);

  let enabled = config().enabled;
  let daemonUp = false;
  /** Features of the suggestions most recently shown, for feedback. */
  let lastShown: Suggestion[] = [];

  const render = () => {
    if (!enabled) {
      status.text = "$(circle-slash) pca off";
      status.tooltip = "Inline completions disabled. Run 'PCA: Toggle Inline Completions'.";
    } else if (!daemonUp) {
      status.text = "$(debug-disconnect) pca";
      status.tooltip = "Daemon not reachable. Run `pca serve` in your project.";
    } else {
      status.text = `$(sparkle) pca${config().neural ? "+nn" : ""}`;
      status.tooltip = "Personal Code Assistant is serving completions.";
    }
    status.show();
  };

  const checkHealth = async () => {
    const health = await callDaemon<HealthResponse>("/health", undefined);
    daemonUp = health?.ok === true && health.trained;
    render();
  };
  void checkHealth();
  const healthTimer = setInterval(() => void checkHealth(), 15_000);
  context.subscriptions.push({ dispose: () => clearInterval(healthTimer) });

  const provider: vscode.InlineCompletionItemProvider = {
    async provideInlineCompletionItems(document, position, _ctx, cancelToken) {
      if (!enabled || !daemonUp) return { items: [] };
      const settings = config();

      // Debounce: a burst of keystrokes should produce one request, not ten.
      const settled = await new Promise<boolean>((resolve) => {
        const timer = setTimeout(() => resolve(true), settings.debounceMs);
        cancelToken.onCancellationRequested(() => {
          clearTimeout(timer);
          resolve(false);
        });
      });
      if (!settled || cancelToken.isCancellationRequested) return { items: [] };

      const offset = document.offsetAt(position);
      const text = document.getText();
      const controller = new AbortController();
      cancelToken.onCancellationRequested(() => controller.abort());

      const response = await callDaemon<CompletionResponse>(
        "/complete",
        {
          prefix: text.slice(Math.max(0, offset - CONTEXT_BEFORE), offset),
          suffix: text.slice(offset, offset + CONTEXT_AFTER),
          filePath: vscode.workspace.asRelativePath(document.uri, false),
          neural: settings.neural,
          retrieval: true,
          count: settings.maxSuggestions,
        },
        controller.signal,
      );
      if (!response || response.suggestions.length === 0) return { items: [] };

      lastShown = response.suggestions;
      return {
        items: response.suggestions.map((suggestion, rank) => {
          const item = new vscode.InlineCompletionItem(
            suggestion.text,
            new vscode.Range(position, position),
          );
          // Report the accept so the ranker can learn from it.
          item.command = {
            command: "pca.accepted",
            title: "pca accepted",
            arguments: [rank],
          };
          return item;
        }),
      };
    },
  };

  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, provider),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("pca.accepted", async (rank: number) => {
      const accepted = lastShown[rank];
      if (!accepted) return;
      await callDaemon("/feedback", {
        accepted: accepted.features,
        rejected: lastShown.filter((_, i) => i !== rank).map((s) => s.features),
      });
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("pca.toggle", () => {
      enabled = !enabled;
      void vscode.workspace
        .getConfiguration("pca")
        .update("enabled", enabled, vscode.ConfigurationTarget.Global);
      render();
      vscode.window.setStatusBarMessage(
        `Personal Code Assistant ${enabled ? "enabled" : "disabled"}`,
        2000,
      );
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("pca.status", async () => {
      const health = await callDaemon<HealthResponse>("/health", undefined);
      if (!health) {
        const choice = await vscode.window.showWarningMessage(
          "pca daemon is not running. Start it with `pca serve` in your project.",
          "Show me how",
        );
        if (choice) {
          const terminal = vscode.window.createTerminal("pca serve");
          terminal.sendText("pca serve");
          terminal.show();
        }
        return;
      }
      void vscode.window.showInformationMessage(
        `pca: ${health.trained ? "trained" : "NOT trained"} · ` +
          `transformer ${health.neural ? "yes" : "no"} · index ${health.index ? "yes" : "no"} · ` +
          `${health.requests} completions served, ${health.meanLatencyMs}ms mean`,
      );
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("pca.search", async () => {
      const query = await vscode.window.showInputBox({
        prompt: "Search your codebase (semantic + lexical)",
        placeHolder: "where do we retry failed uploads",
      });
      if (!query) return;
      const result = await callDaemon<{
        hits: Array<{ file: string; startLine: number; symbol: string | null; text: string }>;
      }>("/search", { query, limit: 15 });
      if (!result || result.hits.length === 0) {
        void vscode.window.showInformationMessage("No matches.");
        return;
      }
      const picked = await vscode.window.showQuickPick(
        result.hits.map((hit) => ({
          label: `${hit.symbol ?? hit.file}`,
          description: `${hit.file}:${hit.startLine}`,
          detail: hit.text.split("\n").slice(0, 2).join(" ").trim().slice(0, 120),
          hit,
        })),
        { placeHolder: query, matchOnDescription: true, matchOnDetail: true },
      );
      if (!picked) return;
      const root = vscode.workspace.workspaceFolders?.[0];
      if (!root) return;
      const uri = vscode.Uri.joinPath(root.uri, picked.hit.file);
      const doc = await vscode.workspace.openTextDocument(uri);
      const editor = await vscode.window.showTextDocument(doc);
      const line = Math.max(0, picked.hit.startLine - 1);
      editor.selection = new vscode.Selection(line, 0, line, 0);
      editor.revealRange(
        new vscode.Range(line, 0, line, 0),
        vscode.TextEditorRevealType.InCenter,
      );
    }),
  );

  render();
}

export function deactivate(): void {
  /* nothing to tear down: the daemon owns all state */
}
