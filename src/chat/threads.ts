import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { ChatTurn } from "./session.js";
import { Workspace, statePath } from "../util/paths.js";

/**
 * Conversations that outlive the process, so this can be *messaged* rather than
 * merely run.
 *
 * A REPL session is gone when you close the terminal, and the old browser client
 * minted a fresh session id on every page load - which meant a phone locking its
 * screen lost the conversation. Every turn has always been written to
 * `.pca/chat/<id>.json`; this module makes those files a first-class store you
 * can list, resume, retitle and delete, and that both the terminal and the
 * browser open by id. Ask something from the phone, keep going from the terminal.
 *
 * Ids arrive from the network, so they are validated as filenames here rather
 * than at each call site: `../../..` reaching `path.join` would write outside the
 * project.
 */

/** Same shape the daemon and the CLI list conversations with. */
export interface ThreadMeta {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: number;
  costUsd: number;
  /** First line of the most recent message, for a list row. */
  preview: string;
  /** Route of the last answer, so a list row can show where it came from. */
  lastRoute?: "local" | "claude";
}

export interface ThreadRecord {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  costUsd: number;
  turns: ChatTurn[];
}

const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

/** True for ids that are safe to use as a filename inside `.pca/chat/`. */
export function isSafeThreadId(id: string): boolean {
  return ID_RE.test(id) && !id.includes("..");
}

export function assertSafeThreadId(id: string): string {
  if (!isSafeThreadId(id)) throw new Error(`Invalid conversation id: ${JSON.stringify(id)}`);
  return id;
}

/**
 * Lexicographically sortable and unique: the timestamp orders conversations by
 * creation even on a filesystem with coarse mtimes, and the suffix keeps two
 * threads started in the same millisecond apart.
 */
export function newThreadId(now = new Date()): string {
  const stamp = now.toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "");
  return `${stamp}-${crypto.randomBytes(3).toString("hex")}`;
}

/** A conversation's name, derived from its opening message when unset. */
export function deriveTitle(turns: ChatTurn[]): string {
  const first = turns.find((t) => t.role === "user");
  if (!first) return "New conversation";
  return titleFrom(first.content);
}

export function titleFrom(text: string): string {
  const one = text.replace(/\s+/g, " ").trim();
  if (one.length === 0) return "New conversation";
  return one.length <= 60 ? one : `${one.slice(0, 59)}…`;
}

export class ThreadStore {
  readonly ws: Workspace;

  constructor(ws: Workspace) {
    this.ws = ws;
  }

  private dir(): string {
    return statePath(this.ws, "chat");
  }

  file(id: string): string {
    return path.join(this.dir(), `${assertSafeThreadId(id)}.json`);
  }

  exists(id: string): boolean {
    return isSafeThreadId(id) && fs.existsSync(this.file(id));
  }

  /** Every conversation on disk, most recently updated first. */
  list(): ThreadMeta[] {
    let names: string[];
    try {
      names = fs.readdirSync(this.dir());
    } catch {
      return [];
    }
    const out: ThreadMeta[] = [];
    for (const name of names) {
      if (!name.endsWith(".json")) continue;
      const id = name.slice(0, -5);
      if (!isSafeThreadId(id)) continue;
      const record = this.load(id);
      if (record) out.push(metaOf(record));
    }
    return out.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0));
  }

  /** null when the file is missing or unreadable, so a bad file cannot hide the rest. */
  load(id: string): ThreadRecord | null {
    if (!isSafeThreadId(id)) return null;
    const file = this.file(id);
    let raw: string;
    try {
      raw = fs.readFileSync(file, "utf8");
    } catch {
      return null;
    }
    let parsed: Partial<ThreadRecord> & { turns?: unknown };
    try {
      parsed = JSON.parse(raw) as Partial<ThreadRecord>;
    } catch {
      return null;
    }
    const turns = Array.isArray(parsed.turns) ? (parsed.turns as ChatTurn[]).filter(isTurn) : [];
    // Transcripts written before conversations had names, and before mtime was
    // recorded in the file, still have to open: fall back to the turns.
    const stat = statSafe(file);
    const createdAt = parsed.createdAt ?? turns[0]?.at ?? stat ?? new Date().toISOString();
    const updatedAt = parsed.updatedAt ?? turns[turns.length - 1]?.at ?? stat ?? createdAt;
    return {
      id,
      title: typeof parsed.title === "string" && parsed.title.length > 0 ? parsed.title : deriveTitle(turns),
      createdAt,
      updatedAt,
      costUsd: typeof parsed.costUsd === "number" ? parsed.costUsd : 0,
      turns,
    };
  }

  meta(id: string): ThreadMeta | null {
    const record = this.load(id);
    return record ? metaOf(record) : null;
  }

  save(record: ThreadRecord): void {
    const file = this.file(record.id);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    // Atomic: a phone that drops mid-write must not be able to truncate a
    // conversation that is also open in a terminal.
    const tmp = `${file}.tmp${process.pid}`;
    fs.writeFileSync(tmp, JSON.stringify(record, null, 2), "utf8");
    fs.renameSync(tmp, file);
  }

  /** An empty conversation, on disk, so it shows up in the list immediately. */
  create(title?: string): ThreadMeta {
    const now = new Date();
    const record: ThreadRecord = {
      id: newThreadId(now),
      title: title && title.trim().length > 0 ? titleFrom(title) : "New conversation",
      createdAt: now.toISOString(),
      updatedAt: now.toISOString(),
      costUsd: 0,
      turns: [],
    };
    this.save(record);
    return metaOf(record);
  }

  rename(id: string, title: string): ThreadMeta | null {
    const record = this.load(id);
    if (!record) return null;
    record.title = titleFrom(title);
    this.save(record);
    return metaOf(record);
  }

  remove(id: string): boolean {
    if (!isSafeThreadId(id)) return false;
    try {
      fs.unlinkSync(this.file(id));
      return true;
    } catch {
      return false;
    }
  }

  /** Most recently updated conversation, for `pca chat --resume`. */
  latest(): ThreadMeta | null {
    return this.list()[0] ?? null;
  }
}

export function metaOf(record: ThreadRecord): ThreadMeta {
  const last = record.turns[record.turns.length - 1];
  const lastAnswer = [...record.turns].reverse().find((t) => t.role === "assistant");
  return {
    id: record.id,
    title: record.title,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    messages: record.turns.length,
    costUsd: record.costUsd,
    preview: last ? titleFrom(last.content) : "",
    lastRoute: lastAnswer?.route,
  };
}

function isTurn(value: unknown): value is ChatTurn {
  if (typeof value !== "object" || value === null) return false;
  const turn = value as Partial<ChatTurn>;
  return (
    (turn.role === "user" || turn.role === "assistant") &&
    typeof turn.content === "string" &&
    typeof turn.at === "string"
  );
}

function statSafe(file: string): string | null {
  try {
    return fs.statSync(file).mtime.toISOString();
  } catch {
    return null;
  }
}
