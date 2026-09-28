import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

/** Write JSON, gzipped, atomically (tmp file + rename). */
export function writeJsonGz(file: string, value: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const buf = zlib.gzipSync(Buffer.from(JSON.stringify(value), "utf8"), { level: 6 });
  const tmp = `${file}.tmp${process.pid}`;
  fs.writeFileSync(tmp, buf);
  fs.renameSync(tmp, file);
}

export function readJsonGz<T>(file: string): T {
  const buf = zlib.gunzipSync(fs.readFileSync(file));
  return JSON.parse(buf.toString("utf8")) as T;
}

export function writeBinary(file: string, buf: Buffer | Uint8Array): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp${process.pid}`;
  fs.writeFileSync(tmp, buf);
  fs.renameSync(tmp, file);
}

export function readBinary(file: string): Buffer {
  return fs.readFileSync(file);
}

export function exists(file: string): boolean {
  return fs.existsSync(file);
}

/** Pack a Float32Array to a Buffer (little-endian, platform native on x64/arm64). */
export function packF32(a: Float32Array): Buffer {
  return Buffer.from(a.buffer, a.byteOffset, a.byteLength);
}

export function unpackF32(buf: Buffer): Float32Array {
  // Copy so the result is aligned and independent of the Buffer pool.
  const out = new Float32Array(buf.byteLength / 4);
  Buffer.from(out.buffer).set(buf);
  return out;
}

export function fileSizeSafe(file: string): number {
  try {
    return fs.statSync(file).size;
  } catch {
    return 0;
  }
}

export function humanBytes(n: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 10 || i === 0 ? v.toFixed(0) : v.toFixed(1)}${units[i]}`;
}
