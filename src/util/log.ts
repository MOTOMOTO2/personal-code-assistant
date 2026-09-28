const useColor = process.stdout.isTTY === true && !process.env.NO_COLOR;

const wrap = (code: string) => (s: string) => (useColor ? `\u001b[${code}m${s}\u001b[0m` : s);

export const color = {
  dim: wrap("2"),
  bold: wrap("1"),
  red: wrap("31"),
  green: wrap("32"),
  yellow: wrap("33"),
  blue: wrap("34"),
  magenta: wrap("35"),
  cyan: wrap("36"),
};

let quiet = false;
export function setQuiet(q: boolean): void {
  quiet = q;
}

export function isQuiet(): boolean {
  return quiet;
}

/** Human-facing status output. Goes to stderr so stdout stays pipe-clean. */
export function info(msg: string): void {
  if (!quiet) process.stderr.write(`${msg}\n`);
}

export function step(msg: string): void {
  if (!quiet) process.stderr.write(`${color.cyan("›")} ${msg}\n`);
}

export function ok(msg: string): void {
  if (!quiet) process.stderr.write(`${color.green("✓")} ${msg}\n`);
}

export function warn(msg: string): void {
  process.stderr.write(`${color.yellow("!")} ${msg}\n`);
}

export function fail(msg: string): void {
  process.stderr.write(`${color.red("✗")} ${msg}\n`);
}

/** Single-line progress indicator that overwrites itself on a TTY. */
export function progress(msg: string): void {
  if (quiet) return;
  if (process.stderr.isTTY) {
    process.stderr.write(`\r\u001b[2K${color.dim(msg)}`);
  }
}

export function progressDone(msg?: string): void {
  if (quiet) return;
  if (process.stderr.isTTY) process.stderr.write("\r\u001b[2K");
  if (msg) process.stderr.write(`${msg}\n`);
}

export function fmtDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m${s.toString().padStart(2, "0")}s`;
}
