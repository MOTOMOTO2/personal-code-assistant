/**
 * Minimal argv parser. A dependency for this would be absurd, and keeping the
 * tool dependency-free means `npm install` is fast and the supply chain is one
 * package deep.
 *
 * Supports: `--flag`, `--no-flag`, `--key value`, `--key=value`, `-abc` short
 * clusters, `--` to stop parsing, and positional arguments.
 */
export interface ParsedArgs {
  positional: string[];
  flags: Map<string, string | boolean>;
  rest: string[];
}

export function parseArgs(argv: string[]): ParsedArgs {
  const positional: string[] = [];
  const flags = new Map<string, string | boolean>();
  const rest: string[] = [];
  let i = 0;
  for (; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--") {
      rest.push(...argv.slice(i + 1));
      break;
    }
    if (arg.startsWith("--")) {
      const body = arg.slice(2);
      const eq = body.indexOf("=");
      if (eq >= 0) {
        flags.set(body.slice(0, eq), body.slice(eq + 1));
        continue;
      }
      if (body.startsWith("no-")) {
        flags.set(body.slice(3), false);
        continue;
      }
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith("-")) {
        flags.set(body, next);
        i++;
      } else {
        flags.set(body, true);
      }
      continue;
    }
    if (arg.startsWith("-") && arg.length > 1) {
      for (const ch of arg.slice(1)) flags.set(ch, true);
      continue;
    }
    positional.push(arg);
  }
  return { positional, flags, rest };
}

export function flagString(args: ParsedArgs, name: string, fallback: string): string;
export function flagString(args: ParsedArgs, name: string): string | undefined;
export function flagString(args: ParsedArgs, name: string, fallback?: string): string | undefined {
  const v = args.flags.get(name);
  if (typeof v === "string") return v;
  return fallback;
}

export function flagNumber(args: ParsedArgs, name: string, fallback: number): number {
  const v = args.flags.get(name);
  if (typeof v === "string") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
    throw new Error(`--${name} expects a number, got "${v}"`);
  }
  return fallback;
}

export function flagBool(args: ParsedArgs, name: string, fallback = false): boolean {
  const v = args.flags.get(name);
  if (typeof v === "boolean") return v;
  if (typeof v === "string") return v !== "false" && v !== "0";
  return fallback;
}

export function flagList(args: ParsedArgs, name: string): string[] {
  const v = args.flags.get(name);
  if (typeof v !== "string") return [];
  return v
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}
