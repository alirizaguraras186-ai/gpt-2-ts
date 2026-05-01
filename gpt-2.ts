import path from "node:path";
import {
	add,
	addInPlace,
	geluInPlace,
	layerNorm,
	loadTensor,
	matvec,
	matvecRows,
	row,
	type Tensor,
} from "./tensor.ts";
import { GPT2Tokenizer } from "./tokenizer.ts";

const N_LAYER = 12;
const N_HEAD = 12;
const N_EMBD = 768;
const N_CTX = 1024;
const N_VOCAB = 50257;
const N_MLP = 3072;
const EOS_TOKEN = 50256;
const HEAD_DIM = N_EMBD / N_HEAD;

export type Block = {
	ln1Gamma: Tensor;
	ln1Beta: Tensor;
	attnQkvWeight: Tensor;
	attnQkvBias: Tensor;
	attnProjWeight: Tensor;
	attnProjBias: Tensor;
	ln2Gamma: Tensor;
	ln2Beta: Tensor;
	mlpFcWeight: Tensor;
	mlpFcBias: Tensor;
	mlpProjWeight: Tensor;
	mlpProjBias: Tensor;
};

export type GPT2 = {
	tokenizer: GPT2Tokenizer;
	tokenEmbedding: Tensor;
	positionEmbedding: Tensor;
	lnFGamma: Tensor;
	lnFBeta: Tensor;
	blocks: Block[];
};

export type KVCache = {
	// One flat [N_CTX, N_EMBD] K and V array per block.
	k: Float32Array[];
	v: Float32Array[];
};

export type Workspace = {
	x: Float32Array;
	norm: Float32Array;
	qkv: Float32Array;
	attn: Float32Array;
	proj: Float32Array;
	fc: Float32Array;
	logits: Float32Array;
	scores: Float32Array;
};

function loadBlock(root: string, i: number): Block {
	const dir = path.join(
		root,
		"tensors",
		`block_${i.toString().padStart(2, "0")}`,
	);
	const block: Block = {
		ln1Gamma: loadTensor(path.join(dir, "ln_1_gamma.tensor")),
		ln1Beta: loadTensor(path.join(dir, "ln_1_beta.tensor")),
		attnQkvWeight: loadTensor(path.join(dir, "attn_qkv_weight.tensor")),
		attnQkvBias: loadTensor(path.join(dir, "attn_qkv_bias.tensor")),
		attnProjWeight: loadTensor(path.join(dir, "attn_proj_weight.tensor")),
		attnProjBias: loadTensor(path.join(dir, "attn_proj_bias.tensor")),
		ln2Gamma: loadTensor(path.join(dir, "ln_2_gamma.tensor")),
		ln2Beta: loadTensor(path.join(dir, "ln_2_beta.tensor")),
		mlpFcWeight: loadTensor(path.join(dir, "mlp_fc_weight.tensor")),
		mlpFcBias: loadTensor(path.join(dir, "mlp_fc_bias.tensor")),
		mlpProjWeight: loadTensor(path.join(dir, "mlp_proj_weight.tensor")),
		mlpProjBias: loadTensor(path.join(dir, "mlp_proj_bias.tensor")),
	};

	return block;
}

export function loadGPT2(root = "."): GPT2 {
	const tokenEmbedding = loadTensor(
		path.join(root, "tensors/token_embedding.tensor"),
	);
	const positionEmbedding = loadTensor(
		path.join(root, "tensors/position_embedding.tensor"),
	);
	const lnFGamma = loadTensor(path.join(root, "tensors/ln_f_gamma.tensor"));
	const lnFBeta = loadTensor(path.join(root, "tensors/ln_f_beta.tensor"));

	const blocks: Block[] = [];
	for (let i = 0; i < N_LAYER; i++) blocks.push(loadBlock(root, i));

	return {
		tokenizer: new GPT2Tokenizer(path.join(root, "tensors/tokenizer")),
		tokenEmbedding,
		positionEmbedding,
		lnFGamma,
		lnFBeta,
		blocks,
	};
}

export function createKVCache(): KVCache {
	return {
		k: Array.from({ length: N_LAYER }, () => new Float32Array(N_CTX * N_EMBD)),
		v: Array.from({ length: N_LAYER }, () => new Float32Array(N_CTX * N_EMBD)),
	};
}

export function createWorkspace(): Workspace {
	return {
		x: new Float32Array(N_EMBD),
		norm: new Float32Array(N_EMBD),
		qkv: new Float32Array(3 * N_EMBD),
		attn: new Float32Array(N_EMBD),
		proj: new Float32Array(N_EMBD),
		fc: new Float32Array(N_MLP),
		logits: new Float32Array(N_VOCAB),
		scores: new Float32Array(N_CTX),
	};
}

export function encodeText(model: GPT2, text: string): number[] {
	return model.tokenizer.encode(text);
}

export function embedToken(
	out: Float32Array,
	model: GPT2,
	tokenId: number,
	position: number,
): void {
	add(
		out,
		row(model.tokenEmbedding, tokenId),
		row(model.positionEmbedding, position),
	);
}

function attention(
	out: Float32Array,
	qkv: Float32Array,
	cache: KVCache,
	layer: number,
	pos: number,
	scores: Float32Array,
): void {
	const kCache = cache.k[layer];
	const vCache = cache.v[layer];
	if (kCache === undefined || vCache === undefined)
		throw new Error("Invalid KV cache layer");

	// Store this token's K,V in cache. qkv layout is [Q all heads][K all heads][V all heads].
	kCache.set(qkv.subarray(N_EMBD, 2 * N_EMBD), pos * N_EMBD);
	vCache.set(qkv.subarray(2 * N_EMBD, 3 * N_EMBD), pos * N_EMBD);
	out.fill(0);

	for (let h = 0; h < N_HEAD; h++) {
		const head = h * HEAD_DIM;
		let maxScore = -Infinity;

		for (let t = 0; t <= pos; t++) {
			let s = 0;
			const kBase = t * N_EMBD + head;
			for (let d = 0; d < HEAD_DIM; d++)
				s += qkv[head + d]! * kCache[kBase + d]!;
			s /= Math.sqrt(HEAD_DIM);
			scores[t] = s;
			if (s > maxScore) maxScore = s;
		}

		let sumExp = 0;
		for (let t = 0; t <= pos; t++) {
			const e = Math.exp(scores[t]! - maxScore);
			scores[t] = e;
			sumExp += e;
		}

		for (let t = 0; t <= pos; t++) {
			const a = scores[t]! / sumExp;
			const vBase = t * N_EMBD + head;
			for (let d = 0; d < HEAD_DIM; d++)
				out[head + d] = out[head + d]! + a * vCache[vBase + d]!;
		}
	}
}

function runBlock(
	x: Float32Array,
	model: GPT2,
	block: Block,
	cache: KVCache,
	layer: number,
	pos: number,
	ws: Workspace,
): void {
	// x -> LN -> QKV -> attention -> projection -> residual add.
	layerNorm(ws.norm, x, block.ln1Gamma, block.ln1Beta);
	matvec(ws.qkv, ws.norm, block.attnQkvWeight, block.attnQkvBias);
	attention(ws.attn, ws.qkv, cache, layer, pos, ws.scores);
	matvec(ws.proj, ws.attn, block.attnProjWeight, block.attnProjBias);
	addInPlace(x, ws.proj);

	// x -> LN -> MLP up -> GELU -> MLP down -> residual add.
	layerNorm(ws.norm, x, block.ln2Gamma, block.ln2Beta);
	matvec(ws.fc, ws.norm, block.mlpFcWeight, block.mlpFcBias);
	geluInPlace(ws.fc);
	matvec(ws.proj, ws.fc, block.mlpProjWeight, block.mlpProjBias);
	addInPlace(x, ws.proj);
}

// Runs one token at absolute position pos and writes logits for the next token.
export function forwardToken(
	model: GPT2,
	tokenId: number,
	pos: number,
	cache: KVCache,
	ws: Workspace,
): Float32Array {
	embedToken(ws.x, model, tokenId, pos);
	for (let i = 0; i < N_LAYER; i++) {
		const block = model.blocks[i];
		if (block === undefined) throw new Error("Missing block");
		runBlock(ws.x, model, block, cache, i, pos, ws);
	}
	layerNorm(ws.norm, ws.x, model.lnFGamma, model.lnFBeta);
	matvecRows(ws.logits, ws.norm, model.tokenEmbedding);
	return ws.logits;
}

export function argmax(x: Float32Array): number {
	let best = 0;
	for (let i = 1; i < x.length; i++) if (x[i]! > x[best]!) best = i;
	return best;
}

type SampleOptions = {
	topK: number;
	temperature: number;
	stopOnEos: boolean;
	onToken?: (tokenId: number) => void;
};

export function sampleTopK(
	logits: Float32Array,
	options: SampleOptions,
): number {
	const k = Math.max(1, Math.min(options.topK, logits.length));
	const temperature = Math.max(options.temperature, 1e-6);
	const topIds = new Int32Array(k);
	const topVals = new Float32Array(k);
	topVals.fill(-Infinity);

	// Keep the k largest logits. Simple O(vocab*k), fine for small k.
	for (let id = 0; id < logits.length; id++) {
		const v = logits[id]!;
		if (v <= topVals[k - 1]!) continue;
		let j = k - 1;
		while (j > 0 && v > topVals[j - 1]!) {
			topVals[j] = topVals[j - 1]!;
			topIds[j] = topIds[j - 1]!;
			j--;
		}
		topVals[j] = v;
		topIds[j] = id;
	}

	let max = -Infinity;
	for (let i = 0; i < k; i++) {
		topVals[i] = topVals[i]! / temperature;
		if (topVals[i]! > max) max = topVals[i]!;
	}

	let sum = 0;
	for (let i = 0; i < k; i++) {
		const p = Math.exp(topVals[i]! - max);
		topVals[i] = p;
		sum += p;
	}

	let r = Math.random() * sum;
	for (let i = 0; i < k; i++) {
		r -= topVals[i]!;
		if (r <= 0) return topIds[i]!;
	}
	return topIds[k - 1]!;
}

export function generate(
	model: GPT2,
	prompt: string,
	maxNewTokens: number,
	options: SampleOptions = { topK: 40, temperature: 0.9, stopOnEos: true },
): string {
	const ids = encodeText(model, prompt);
	const cache = createKVCache();
	const ws = createWorkspace();
	let logits = ws.logits;

	for (let pos = 0; pos < ids.length; pos++) {
		const id = ids[pos];
		if (id === undefined) throw new Error("Invalid prompt ids");
		logits = forwardToken(model, id, pos, cache, ws);
	}

	for (let i = 0; i < maxNewTokens; i++) {
		const next =
			options.topK <= 1 ? argmax(logits) : sampleTopK(logits, options);
		if (options.stopOnEos && next === EOS_TOKEN) break;
		ids.push(next);
		options.onToken?.(next);
		logits = forwardToken(model, next, ids.length - 1, cache, ws);
	}

	return model.tokenizer.decode(ids);
}

function arg(name: string, fallback: string): string {
	const prefix = `--${name}=`;
	return (
		process.argv.find((x) => x.startsWith(prefix))?.slice(prefix.length) ??
		fallback
	);
}

if (import.meta.url === `file://${process.argv[1]}`) {
	const positional = process.argv.slice(2).filter((x) => !x.startsWith("--"));
	const prompt = positional.join(" ") || "Hello world";
	const maxTokens = Number(arg("tokens", "20"));
	const topK = Number(arg("top-k", "40"));
	const temperature = Number(arg("temperature", "0.9"));
	const stopOnEos = arg("stop-eos", "1") !== "0";
	const stream = arg("stream", "1") !== "0";
	const model = loadGPT2();
	if (stream) {
		process.stdout.write(prompt);
		generate(model, prompt, maxTokens, {
			topK,
			temperature,
			stopOnEos,
			onToken: (id) => process.stdout.write(model.tokenizer.decode([id])),
		});
		process.stdout.write("\n");
	} else {
		console.log(generate(model, prompt, maxTokens, { topK, temperature, stopOnEos }));
	}
}
