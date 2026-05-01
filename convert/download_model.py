#!/usr/bin/env python3
"""Download the original OpenAI GPT-2 TensorFlow checkpoint files.

Usage:
  cd ~/Desktop/gpt-2/convert
  uv run python download_model.py 124M
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

FILES = [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
]
BASE = "https://openaipublic.blob.core.windows.net/gpt-2/models"


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    resume_at = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r, tmp.open("ab" if resume_at else "wb") as f:
        total_header = r.headers.get("Content-Length")
        total = (int(total_header) + resume_at) if total_header else 0
        done = resume_at
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r{dst.name:35s} {done / 1e6:8.1f}/{total / 1e6:8.1f} MB {pct:5.1f}%", end="")
            else:
                print(f"\r{dst.name:35s} {done / 1e6:8.1f} MB", end="")
    print()
    tmp.replace(dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="124M", help="124M, 355M, 774M, or 1558M")
    p.add_argument("--out", default=None, help="models directory; default is ../models")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_base = Path(args.out) if args.out else root / "models"
    out_dir = out_base / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"{name:35s} already exists")
            continue
        url = f"{BASE}/{args.model}/{name}"
        try:
            download(url, dst)
        except Exception as e:
            print(f"\nfailed: {url}: {e}", file=sys.stderr)
            raise

    print(f"Downloaded {args.model} to {out_dir}")


if __name__ == "__main__":
    main()
