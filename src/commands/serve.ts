import { DEFAULT_PORT, createDaemon, isLoopbackHost, lanAddresses } from "../server/daemon.js";
import { QR_MAX_BYTES, qrForTerminal } from "../web/qr.js";
import { Effort } from "../llm/claude.js";
import { ParsedArgs, flagBool, flagNumber, flagString } from "../util/args.js";
import { color, info, warn } from "../util/log.js";
import { workspace } from "../util/paths.js";

/**
 * `pca serve` - the warm daemon, and the thing you point a phone at.
 *
 * `--lan` is the whole feature in one flag: bind every interface, generate a
 * token if there is not one, turn the browser client on, and print the URL both
 * as text and as a QR code so getting it onto a phone is one camera scan rather
 * than typing a 32-character secret with your thumbs.
 */

const VALID_EFFORT = new Set(["low", "medium", "high", "xhigh", "max"]);

export async function cmdServe(args: ParsedArgs): Promise<number> {
  const ws = workspace(flagString(args, "root"));
  const lan = flagBool(args, "lan");
  const host = flagString(args, "host") ?? (lan ? "0.0.0.0" : "127.0.0.1");
  const loopback = isLoopbackHost(host);
  // Reaching it from a phone is pointless without the client, so --lan implies it.
  const ui = flagBool(args, "ui", !loopback);
  const effort = flagString(args, "effort") as Effort | undefined;
  if (effort && !VALID_EFFORT.has(effort)) {
    throw new Error(`--effort must be one of: ${[...VALID_EFFORT].join(", ")}`);
  }

  const daemon = createDaemon(ws, {
    port: flagNumber(args, "port", DEFAULT_PORT),
    host,
    ui,
    token: flagString(args, "token"),
    idleTimeoutMs: flagNumber(args, "idle-timeout", 0),
    effort,
  });
  const port = await daemon.listen();
  announce(daemon.urls(port), daemon.token, {
    loopback,
    ui,
    qr: flagBool(args, "qr", !loopback),
  });

  // Resolve only when the server closes, so the process stays up.
  await new Promise<void>((resolve) => {
    daemon.server.once("close", resolve);
    const stop = () => {
      void daemon.close();
    };
    process.once("SIGINT", stop);
    process.once("SIGTERM", stop);
  });
  return 0;
}

/** `pca phone` - `serve --lan --ui`, because that is the thing people want. */
export async function cmdPhone(args: ParsedArgs): Promise<number> {
  if (!args.flags.has("host")) args.flags.set("lan", true);
  if (!args.flags.has("ui")) args.flags.set("ui", true);
  return cmdServe(args);
}

function announce(
  urls: string[],
  token: string | null,
  opts: { loopback: boolean; ui: boolean; qr: boolean },
): void {
  if (!opts.ui) {
    if (token) info(color.dim(`token ${token}`));
    return;
  }
  const url = urls[0];
  if (opts.loopback) {
    info(`  chat at ${color.bold(url)}`);
    info(color.dim("  add --lan to reach it from your phone on this network"));
    return;
  }

  if (lanAddresses().length === 0) {
    warn("no network interface found - the address below may not be reachable");
  }
  info(`\n  ${color.bold("Open this on your phone")} ${color.dim("(same Wi-Fi as this machine)")}`);
  for (const candidate of urls) info(`    ${color.bold(candidate)}`);
  if (token) info(color.dim(`\n  token ${token}`));

  if (!opts.qr) return;
  // Colour is not decoration here: a QR code rendered in the terminal's own
  // colours can come out inverted, and an inverted symbol does not scan.
  const colorOk = process.stderr.isTTY === true && !process.env.NO_COLOR;
  if (!colorOk) {
    info(color.dim("  (no colour available for a QR code - use the URL above)"));
    return;
  }
  const qr = qrForTerminal(url, { quietZone: 2 });
  if (!qr) {
    info(color.dim(`  (URL is over ${QR_MAX_BYTES} bytes, too long for a QR code)`));
    return;
  }
  info(color.dim("\n  or scan:"));
  info(`\n${qr}`);
  info(color.dim("  Ctrl-C stops the server. Conversations are kept in .pca/chat/.\n"));
}
