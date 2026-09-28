import fs from "node:fs";
import path from "node:path";
import os from "node:os";

/** Name of the per-project state directory. */
export const STATE_DIR = ".pca";

/**
 * Walk up from `start` looking for a project root: a directory that already has
 * a `.pca/` state dir, or a `.git/`, or a `package.json`. Falls back to `start`.
 */
export function findProjectRoot(start: string = process.cwd()): string {
  let dir = path.resolve(start);
  const markers = [STATE_DIR, ".git", "package.json", "pyproject.toml", "go.mod", "Cargo.toml"];
  for (;;) {
    for (const m of markers) {
      if (fs.existsSync(path.join(dir, m))) return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) return path.resolve(start);
    dir = parent;
  }
}

export interface Workspace {
  root: string;
  stateDir: string;
}

export function workspace(start?: string): Workspace {
  const root = findProjectRoot(start);
  return { root, stateDir: path.join(root, STATE_DIR) };
}

export function ensureStateDir(ws: Workspace): string {
  fs.mkdirSync(ws.stateDir, { recursive: true });
  return ws.stateDir;
}

export function statePath(ws: Workspace, ...parts: string[]): string {
  return path.join(ws.stateDir, ...parts);
}

/** Global (cross-project) config location, e.g. for the API key. */
export function globalConfigDir(): string {
  return path.join(os.homedir(), ".pca");
}

export function relative(ws: Workspace, abs: string): string {
  return path.relative(ws.root, abs).split(path.sep).join("/");
}
