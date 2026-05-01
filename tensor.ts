import fs from "node:fs";

const HEADER_SIZE = 64;
const MAGIC = "GPT2TNS\0";

export type Tensor = {
  shape: number[];
  data: Float32Array;
};

export function loadTensor(file: string): Tensor {
  const buf = fs.readFileSync(file);
  if (buf.length < HEADER_SIZE) throw new Error(`Tensor too small: ${file}`);

  const magic = buf.subarray(0, 8).toString("latin1");
  if (magic !== MAGIC) throw new Error(`Bad tensor magic in ${file}`);

  const version = buf.readUInt32LE(8);
  const dtype = buf.readUInt32LE(12);
  const ndim = buf.readUInt32LE(16);
  const nbytes = Number(buf.readBigUInt64LE(56));

  if (version !== 1) throw new Error(`Unsupported tensor version ${version} in ${file}`);
  if (dtype !== 1) throw new Error(`Unsupported dtype ${dtype} in ${file}; expected float32`);
  if (ndim > 4) throw new Error(`Invalid ndim ${ndim} in ${file}`);
  if (buf.length !== HEADER_SIZE + nbytes) throw new Error(`Size mismatch in ${file}`);

  const shape64 = [
    buf.readBigUInt64LE(24),
    buf.readBigUInt64LE(32),
    buf.readBigUInt64LE(40),
    buf.readBigUInt64LE(48),
  ];
  const shape = shape64.slice(0, ndim).map((x) => Number(x));

  let count = 1;
  for (const dim of shape) count *= dim;
  if (count * 4 !== nbytes) throw new Error(`Shape/nbytes mismatch in ${file}`);

  // Copy into an ArrayBuffer owned/aligned for Float32Array.
  const payload = buf.subarray(HEADER_SIZE);
  const arrayBuffer = payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength);
  return { shape, data: new Float32Array(arrayBuffer) };
}

export function expectShape(t: Tensor, shape: number[], name = "tensor"): void {
  if (t.shape.length !== shape.length || t.shape.some((dim, i) => dim !== shape[i])) {
    throw new Error(`${name}: expected shape [${shape}], got [${t.shape}]`);
  }
}

export function row(t: Tensor, rowIndex: number): Float32Array {
  if (t.shape.length !== 2) throw new Error(`Expected 2D tensor, got [${t.shape}]`);
  const [rows, cols] = t.shape;
  if (rows === undefined || cols === undefined) throw new Error("Invalid tensor shape");
  if (rowIndex < 0 || rowIndex >= rows) throw new Error(`Row ${rowIndex} out of range 0..${rows - 1}`);
  return t.data.subarray(rowIndex * cols, (rowIndex + 1) * cols);
}

export function add(out: Float32Array, a: Float32Array, b: Float32Array): void {
  if (out.length !== a.length || out.length !== b.length) throw new Error("Vector length mismatch");
  for (let i = 0; i < out.length; i++) out[i] = a[i]! + b[i]!;
}

export function addInPlace(a: Float32Array, b: Float32Array): void {
  if (a.length !== b.length) throw new Error("Vector length mismatch");
  for (let i = 0; i < a.length; i++) a[i] = a[i]! + b[i]!;
}

export function layerNorm(out: Float32Array, x: Float32Array, gamma: Tensor, beta: Tensor): void {
  if (out.length !== x.length || gamma.data.length !== x.length || beta.data.length !== x.length) {
    throw new Error("LayerNorm length mismatch");
  }

  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i]!;
  mean /= x.length;

  let variance = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i]! - mean;
    variance += d * d;
  }
  variance /= x.length;

  const scale = 1 / Math.sqrt(variance + 1e-5);
  for (let i = 0; i < x.length; i++) out[i] = (x[i]! - mean) * scale * gamma.data[i]! + beta.data[i]!;
}

// out[j] = bias[j] + sum_i x[i] * w[i,j], w shape [in,out].
export function matvec(out: Float32Array, x: Float32Array, w: Tensor, bias?: Tensor): void {
  if (w.shape.length !== 2) throw new Error("matvec weight must be 2D");
  const [inDim, outDim] = w.shape;
  if (inDim === undefined || outDim === undefined) throw new Error("Invalid matvec weight shape");
  if (x.length !== inDim || out.length !== outDim) throw new Error("matvec shape mismatch");
  if (bias !== undefined && bias.data.length !== outDim) throw new Error("matvec bias shape mismatch");

  for (let j = 0; j < outDim; j++) out[j] = bias?.data[j] ?? 0;
  for (let i = 0; i < inDim; i++) {
    const xi = x[i]!;
    const base = i * outDim;
    for (let j = 0; j < outDim; j++) out[j] = out[j]! + xi * w.data[base + j]!;
  }
}

// logits[row] = dot(x, embedding[row]), embedding shape [rows,dim].
export function matvecRows(out: Float32Array, x: Float32Array, embedding: Tensor): void {
  const [rows, dim] = embedding.shape;
  if (rows === undefined || dim === undefined || x.length !== dim || out.length !== rows) {
    throw new Error("matvecRows shape mismatch");
  }

  for (let r = 0; r < rows; r++) {
    let sum = 0;
    const base = r * dim;
    for (let i = 0; i < dim; i++) sum += x[i]! * embedding.data[base + i]!;
    out[r] = sum;
  }
}

export function geluInPlace(x: Float32Array): void {
  for (let i = 0; i < x.length; i++) {
    const v = x[i]!;
    x[i] = 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
  }
}
