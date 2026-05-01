#!/usr/bin/env python3
"""Convert OpenAI GPT-2 TensorFlow checkpoint weights to simple tensor files.

Output tensor file format, little-endian:

    offset  size  field
    0       8     magic bytes: b"GPT2TNS\0"
    8       4     uint32 version, currently 1
    12      4     uint32 dtype, 1 = float32
    16      4     uint32 ndim
    20      4     uint32 reserved, zero
    24      32    uint64 shape[4], unused entries zero
    56      8     uint64 nbytes
    64      ...   raw row-major float32 payload

By default this strips GPT-2's TensorFlow Conv1D leading singleton dimension,
e.g. [1, 768, 2304] becomes [768, 2304].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

MAGIC = b"GPT2TNS\0"
VERSION = 1
DTYPE_F32 = 1
HEADER_SIZE = 64


def nice_path(tf_name: str) -> Path:
    """Map TensorFlow checkpoint names to readable output paths."""
    if tf_name == "model/wte":
        return Path("token_embedding.tensor")
    if tf_name == "model/wpe":
        return Path("position_embedding.tensor")
    if tf_name == "model/ln_f/g":
        return Path("ln_f_gamma.tensor")
    if tf_name == "model/ln_f/b":
        return Path("ln_f_beta.tensor")

    m = re.fullmatch(r"model/h(\d+)/(.*)", tf_name)
    if not m:
        return Path(tf_name.replace("/", "__") + ".tensor")

    block = int(m.group(1))
    rest = m.group(2)
    names = {
        "ln_1/g": "ln_1_gamma.tensor",
        "ln_1/b": "ln_1_beta.tensor",
        "ln_2/g": "ln_2_gamma.tensor",
        "ln_2/b": "ln_2_beta.tensor",
        "attn/c_attn/w": "attn_qkv_weight.tensor",
        "attn/c_attn/b": "attn_qkv_bias.tensor",
        "attn/c_proj/w": "attn_proj_weight.tensor",
        "attn/c_proj/b": "attn_proj_bias.tensor",
        "mlp/c_fc/w": "mlp_fc_weight.tensor",
        "mlp/c_fc/b": "mlp_fc_bias.tensor",
        "mlp/c_proj/w": "mlp_proj_weight.tensor",
        "mlp/c_proj/b": "mlp_proj_bias.tensor",
    }
    leaf = names.get(rest, rest.replace("/", "__") + ".tensor")
    return Path(f"block_{block:02d}") / leaf


def write_tensor(path: Path, arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype="<f4", order="C")
    shape = list(arr.shape)
    if len(shape) > 4:
        raise ValueError(f"{path}: too many dimensions: {shape}")

    padded_shape = shape + [0] * (4 - len(shape))
    data = arr.tobytes(order="C")
    header = struct.pack(
        "<8sIIII4QQ",
        MAGIC,
        VERSION,
        DTYPE_F32,
        len(shape),
        0,
        padded_shape[0],
        padded_shape[1],
        padded_shape[2],
        padded_shape[3],
        len(data),
    )
    assert len(header) == HEADER_SIZE

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(data)

    sha256 = hashlib.sha256()
    sha256.update(header)
    sha256.update(data)

    return {
        "file": str(path),
        "dtype": "float32",
        "ndim": len(shape),
        "shape": shape,
        "nbytes": len(data),
        "sha256": sha256.hexdigest(),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="124M", help="GPT-2 model size, e.g. 124M")
    p.add_argument("--checkpoint", default=None, help="checkpoint prefix, not .data file")
    p.add_argument("--out", default=str(root / "tensors"), help="output directory")
    p.add_argument("--clean", action="store_true", help="delete output directory first")
    p.add_argument(
        "--keep-leading-one",
        action="store_true",
        help="keep TF Conv1D leading singleton dimension [1, in, out] instead of writing [in, out]",
    )
    args = p.parse_args()

    checkpoint = args.checkpoint or str(root / "models" / args.model / "model.ckpt")
    model_dir = root / "models" / args.model
    out_dir = Path(args.out)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = tf.train.load_checkpoint(checkpoint)
    shapes = reader.get_variable_to_shape_map()

    manifest: dict[str, Any] = {
        "format": {
            "magic": "GPT2TNS\\0",
            "version": VERSION,
            "header_size": HEADER_SIZE,
            "endianness": "little",
            "dtype_codes": {"1": "float32"},
        },
        "checkpoint": checkpoint,
        "strip_leading_singleton_dimension": not args.keep_leading_one,
        "tensors": [],
    }

    for tf_name in sorted(shapes.keys(), key=lambda n: nice_path(n).as_posix()):
        arr = reader.get_tensor(tf_name)
        original_shape = list(arr.shape)

        if not args.keep_leading_one and arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        rel = nice_path(tf_name)
        info = write_tensor(out_dir / rel, arr)
        info["file"] = str(rel)
        info["tf_name"] = tf_name
        info["tf_shape"] = original_shape
        manifest["tensors"].append(info)
        print(f"{tf_name:28s} {original_shape!s:18s} -> {rel} {info['shape']}")

    tokenizer_dir = out_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_dir / "encoder.json", tokenizer_dir / "encoder.json")
    shutil.copy2(model_dir / "vocab.bpe", tokenizer_dir / "vocab.bpe")
    shutil.copy2(model_dir / "hparams.json", out_dir / "hparams.json")
    manifest["tokenizer"] = {
        "type": "gpt2_bpe",
        "encoder_json": "tokenizer/encoder.json",
        "vocab_bpe": "tokenizer/vocab.bpe",
    }
    manifest["hparams"] = "hparams.json"

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nWrote {len(manifest['tensors'])} tensors to {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
