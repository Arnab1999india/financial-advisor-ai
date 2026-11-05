/**
 * Node smoke tests for speech text joining (no browser APIs).
 * Run: node frontend/speech-utils.test.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";

const dir = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(dir, "voice-recognition.js"), "utf8");
const sandbox = { window: {}, globalThis: {} };
sandbox.globalThis = sandbox.window;
vm.runInNewContext(source, sandbox);

const { joinSpeechParts } = sandbox.window.VoiceDictation;

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

assert(
  joinSpeechParts("Hello", "world", "") === "Hello world",
  "base + committed"
);
assert(
  joinSpeechParts("", "etf", " funds") === "etf funds",
  "interim append"
);
assert(
  joinSpeechParts("Line one", "", "partial") === "Line one partial",
  "base + interim"
);

console.log("speech-utils.test.mjs: all passed");
