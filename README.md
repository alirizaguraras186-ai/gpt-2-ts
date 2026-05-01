# gpt-2-ts

https://github.com/user-attachments/assets/27c98071-38f1-406d-a653-88b91766fe4e

A tiny GPT-2 124M inference implementation in TypeScript.

The runtime is intentionally simple and C-like: fixed tensor files, `Float32Array` buffers, token-by-token inference, KV cache, top-k sampling, and streaming output.

## Requirements

- Node.js with built-in TypeScript type stripping, tested with Node `v25.2.1`
- `uv` for the Python conversion tools

## Setup

Install TypeScript typings/compiler:

```bash
npm install
```

Download and convert OpenAI's original GPT-2 checkpoint:

```bash
cd convert
uv sync
uv run python download_model.py 124M
uv run python convert.py --model 124M --clean
cd ..
```

Other original GPT-2 sizes should also work:

```bash
uv run python download_model.py 355M   # or 774M, 1558M
uv run python convert.py --model 355M --clean
```

The runtime reads `tensors/hparams.json`, so model dimensions are not hardcoded. The 1558M model needs many GB of RAM/disk.

This creates local, gitignored directories:

```text
models/   # original OpenAI TensorFlow checkpoint files
tensors/  # converted .tensor files + tokenizer files
```

## Run

```bash
node gpt-2.ts "Hello, my name is" --tokens=50 --top-k=40 --temperature=0.9
```

Streaming is on by default. Disable it with:

```bash
node gpt-2.ts "Hello" --tokens=20 --stream=0
```

Other options:

```text
--tokens=N          number of new tokens, default 20
--top-k=K           top-k sampling; use 1 for greedy, default 40
--temperature=T     sampling temperature, default 0.9
--stop-eos=0        do not stop on <|endoftext|>
--stream=0          print only final text
```

## Check

```bash
npm run check
```

## Tensor format

Each `.tensor` file is a 64-byte fixed little-endian header followed by raw row-major `float32` data:

```c
typedef struct {
    char magic[8];       // "GPT2TNS\0"
    uint32_t version;    // 1
    uint32_t dtype;      // 1 = float32
    uint32_t ndim;
    uint32_t reserved;
    uint64_t shape[4];
    uint64_t nbytes;
} TensorHeader;
```

See `convert/README.md` for conversion details.
