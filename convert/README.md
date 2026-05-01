# Conversion tools

This folder contains the Python-only setup for downloading OpenAI's original GPT-2 TensorFlow checkpoint and converting it to the simple `.tensor` format used by the TypeScript runtime.

## Setup

```bash
cd ~/Desktop/gpt-2/convert
uv sync
```

## Download model files

```bash
uv run python download_model.py 124M
```

Writes to:

```text
../models/124M/
```

## Convert to `.tensor` files

```bash
uv run python convert.py --model 124M --clean
```

Writes to:

```text
../tensors/
```

The converter also copies tokenizer/config files into:

```text
../tensors/tokenizer/encoder.json
../tensors/tokenizer/vocab.bpe
../tensors/hparams.json
```

## `.tensor` file format

Each tensor file is a 64-byte fixed little-endian header followed by raw row-major `float32` data.

```c
typedef struct {
    char magic[8];       // "GPT2TNS\0"
    uint32_t version;    // 1
    uint32_t dtype;      // 1 = float32
    uint32_t ndim;       // number of used shape entries
    uint32_t reserved;   // 0
    uint64_t shape[4];   // unused entries are 0
    uint64_t nbytes;     // payload bytes after this header
} TensorHeader;
```

By default, TensorFlow Conv1D leading singleton dimensions are stripped:

```text
[1, 768, 2304] -> [768, 2304]
```

Use `--keep-leading-one` to preserve exact TensorFlow shapes.
