import { ThreadStore } from "../chat/threads.js";
import { ParsedArgs, flagBool, flagString } from "../util/args.js";
import { color, info, warn } from "../util/log.js";
import { workspace } from "../util/paths.js";

/**
 * `pca threads` - the conversations you have had with this project, as listed by
 * the drawer in the browser client and resumable from either side.
 */
export async function cmdThreads(args: ParsedArgs): Promise<number> {
  const ws = workspace(flagString(args, "root"));
  const store = new ThreadStore(ws);
  const [verb, ...rest] = args.positional;

  if (verb === "rm" || verb === "delete" || verb === "remove") {
    if (rest.length === 0) {
      warn("Usage: pca threads rm <id>");
      return 1;
    }
    let removed = 0;
    for (const id of rest) {
      if (store.remove(id)) removed++;
      else warn(`no conversation ${id}`);
    }
    info(`${removed} deleted`);
    return removed === rest.length ? 0 : 1;
  }

  if (verb === "rename") {
    const [id, ...words] = rest;
    const title = words.join(" ").trim();
    if (!id || title.length === 0) {
      warn('Usage: pca threads rename <id> "new title"');
      return 1;
    }
    const meta = store.rename(id, title);
    if (!meta) {
      warn(`no conversation ${id}`);
      return 1;
    }
    info(`${meta.id}  ${meta.title}`);
    return 0;
  }

  const threads = store.list();
  if (flagBool(args, "json")) {
    process.stdout.write(`${JSON.stringify(threads, null, 2)}\n`);
    return 0;
  }
  if (threads.length === 0) {
    info("No conversations yet. `pca chat` starts one, `pca serve --lan` opens it on your phone.");
    return 0;
  }
  info(`${threads.length} conversation${threads.length === 1 ? "" : "s"} in this project:\n`);
  for (const thread of threads) {
    const route = thread.lastRoute ? ` ${routeTag(thread.lastRoute)}` : "";
    info(
      `  ${color.bold(thread.id)}  ${thread.title}\n` +
        color.dim(
          `    ${thread.messages} messages · ${ago(thread.updatedAt)}` +
            `${thread.costUsd > 0 ? ` · $${thread.costUsd.toFixed(4)}` : ""}${route}`,
        ),
    );
  }
  info(color.dim("\n  pca chat --resume            continue the most recent"));
  info(color.dim("  pca chat --thread <id>       continue a specific one"));
  return 0;
}

function routeTag(route: "local" | "claude"): string {
  return route === "claude" ? color.magenta("claude") : color.green("local");
}

function ago(iso: string): string {
  const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  if (mins < 60 * 24) return `${Math.round(mins / 60)}h ago`;
  return `${Math.round(mins / 1440)}d ago`;
}
