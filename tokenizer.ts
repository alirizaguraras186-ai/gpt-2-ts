import fs from "node:fs";
import path from "node:path";

export type Tokenizer = {
  encoder: Record<string, number>;
  decoder: Map<number, string>;
  bpeRanks: Map<string, number>;
  cache: Map<string, string[]>;
  byteEncoder: Map<number, string>;
  byteDecoder: Map<string, number>;
};

const PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode(): Pick<Tokenizer, "byteEncoder" | "byteDecoder"> {
  const bs: number[] = [];
  for (let i = "!".charCodeAt(0); i <= "~".charCodeAt(0); i++) bs.push(i);
  for (let i = "¡".charCodeAt(0); i <= "¬".charCodeAt(0); i++) bs.push(i);
  for (let i = "®".charCodeAt(0); i <= "ÿ".charCodeAt(0); i++) bs.push(i);

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const byteEncoder = new Map<number, string>();
  const byteDecoder = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const b = bs[i];
    const c = cs[i];
    if (b === undefined || c === undefined) throw new Error("Invalid byte/unicode table");
    const s = String.fromCodePoint(c);
    byteEncoder.set(b, s);
    byteDecoder.set(s, b);
  }
  return { byteEncoder, byteDecoder };
}

export class GPT2Tokenizer {
  state: Tokenizer;

  constructor(tokenizerDir = "tensors/tokenizer") {
    this.state = loadTokenizer(tokenizerDir);
  }

  encode(text: string): number[] {
    return encode(this.state, text);
  }

  decode(ids: number[]): string {
    return decode(this.state, ids);
  }
}

export function loadTokenizer(tokenizerDir = "tensors/tokenizer"): Tokenizer {
  const encoderPath = path.join(tokenizerDir, "encoder.json");
  const vocabPath = path.join(tokenizerDir, "vocab.bpe");

  const encoder: Record<string, number> = JSON.parse(fs.readFileSync(encoderPath, "utf8"));
  const decoder = new Map<number, string>();
  for (const [tok, id] of Object.entries(encoder)) decoder.set(id, tok);

  const bpeRanks = new Map<string, number>();
  const lines = fs.readFileSync(vocabPath, "utf8").split(/\r?\n/);
  let rank = 0;
  for (const line of lines) {
    if (!line || line.startsWith("#")) continue;
    const parts = line.split(" ");
    if (parts.length !== 2) continue;
    bpeRanks.set(parts[0] + "\u0000" + parts[1], rank++);
  }

  return { encoder, decoder, bpeRanks, cache: new Map(), ...bytesToUnicode() };
}

export function encode(tok: Tokenizer, text: string): number[] {
  const ids: number[] = [];
  const chunks = text.match(PATTERN) ?? [];

  for (const chunk of chunks) {
    const bytes = Buffer.from(chunk, "utf8");
    let token = "";
    for (const b of bytes) token += tok.byteEncoder.get(b)!;

    for (const bpeToken of bpe(tok, token)) {
      const id = tok.encoder[bpeToken];
      if (id === undefined) throw new Error(`Token not in encoder: ${JSON.stringify(bpeToken)}`);
      ids.push(id);
    }
  }
  return ids;
}

export function decode(tok: Tokenizer, ids: number[]): string {
  let text = "";
  for (const id of ids) {
    const token = tok.decoder.get(id);
    if (token === undefined) throw new Error(`Unknown token id: ${id}`);
    text += token;
  }

  const bytes: number[] = [];
  for (const ch of Array.from(text)) {
    const b = tok.byteDecoder.get(ch);
    if (b === undefined) throw new Error(`Unknown byte-unicode char: ${JSON.stringify(ch)}`);
    bytes.push(b);
  }
  return Buffer.from(bytes).toString("utf8");
}

function bpe(tok: Tokenizer, token: string): string[] {
  const cached = tok.cache.get(token);
  if (cached) return cached;

  const parts = Array.from(token);
  while (parts.length > 1) {
    let bestRank = Infinity;
    let bestIndex = -1;

    for (let i = 0; i < parts.length - 1; i++) {
      const rank = tok.bpeRanks.get(parts[i] + "\u0000" + parts[i + 1]);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIndex = i;
      }
    }

    if (bestIndex < 0) break;
    const left = parts[bestIndex];
    const right = parts[bestIndex + 1];
    if (left === undefined || right === undefined) throw new Error("Invalid BPE merge index");
    parts.splice(bestIndex, 2, left + right);
  }

  tok.cache.set(token, parts);
  return parts;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const text = process.argv.slice(2).join(" ") || "Hello world";
  const tok = new GPT2Tokenizer();
  const ids = tok.encode(text);
  console.log(JSON.stringify(ids));
  console.log(tok.decode(ids));
}
