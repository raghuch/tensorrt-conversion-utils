#!/usr/bin/env python3
"""Convert a multilingual CTC checkpoint to ONNX or TensorRT.

This wrapper keeps the repo's existing conversion flow while adding the extra
language-aware metadata needed by the multilingual Triton ensemble.

Example:
  python3 pt2onnx2trt/multilingual_model_export.py onnx \
      --checkpoint /path/to/model.nemo \
      --artifacts-dir /tmp/onnx_run

  python3 pt2onnx2trt/multilingual_model_export.py trt \
      --checkpoint /path/to/model.nemo \
      --artifacts-dir /tmp/trt_run \
      --language-mask-file /path/to/language_masks.json \
      --vocab-file /path/to/vocab.json
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def export_checkpoint(checkpoint: str, mode: str, artifacts_dir: str, *,
                      onnx_path: str = None, folded_path: str = None,
                      engine_path: str = None, precision: str = "fp16",
                      workspace_size: str = "4096", batchsize_range: str = "1,4,64",
                      shape_ranges: str = "1,1001,3001"):
    if mode not in {"onnx", "trt", "all"}:
        raise ValueError(f"Unsupported mode: {mode}")

    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(SCRIPT_DIR / "nemo2onnx2trt.py")]

    if mode == "onnx":
        cmd.append("onnx")
    elif mode == "trt":
        cmd.append("trt")
    else:
        cmd.append("all")

    cmd.append(checkpoint)
    cmd.extend(["--artifacts-dir", str(artifacts)])
    if onnx_path:
        cmd.extend(["--onnx-path", onnx_path])
    if folded_path:
        cmd.extend(["--onnx-folded-path", folded_path])
    if engine_path:
        cmd.extend(["--engine-path", engine_path])
    if precision:
        cmd.extend(["--precision", precision])
    if workspace_size:
        cmd.extend(["--workspace-size", str(workspace_size)])
    if batchsize_range:
        cmd.extend(["--batchsize-range", batchsize_range])
    if shape_ranges:
        cmd.extend(["--shape-ranges", shape_ranges])

    _run(cmd)

    return str(artifacts)


def copy_language_metadata(artifacts_dir: str, language_mask_file: str = None, vocab_file: str = None):
    target = Path(artifacts_dir)
    if language_mask_file:
        src = Path(language_mask_file)
        dst = target / "language_masks.json"
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    if vocab_file:
        src = Path(vocab_file)
        dst = target / "vocab.json"
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


def add_arguments(parser):
    parser.add_argument("mode", choices=("onnx", "trt", "all"), help="export mode")
    parser.add_argument("--checkpoint", required=True, help="Path to the .nemo checkpoint")
    parser.add_argument("--artifacts-dir", default="./models_latest_multilingual", help="Output directory for ONNX and TensorRT files")
    parser.add_argument("--onnx-path", help="Exact ONNX output path")
    parser.add_argument("--engine-path", help="Exact engine output path")
    parser.add_argument("--folded-path", help="Exact folded ONNX path")
    parser.add_argument("--precision", default="fp16", choices=("fp32", "fp16"))
    parser.add_argument("--workspace-size", default="4096")
    parser.add_argument("--batchsize-range", default="1,4,64")
    parser.add_argument("--shape-ranges", default="1,1001,3001")
    parser.add_argument("--language-mask-file", help="JSON mapping language -> class mask")
    parser.add_argument("--vocab-file", help="JSON mapping language -> vocabulary list")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()

    export_checkpoint(
        args.checkpoint,
        args.mode,
        args.artifacts_dir,
        onnx_path=args.onnx_path,
        folded_path=args.folded_path,
        engine_path=args.engine_path,
        precision=args.precision,
        workspace_size=args.workspace_size,
        batchsize_range=args.batchsize_range,
        shape_ranges=args.shape_ranges,
    )

    if args.language_mask_file or args.vocab_file:
        copy_language_metadata(args.artifacts_dir, args.language_mask_file, args.vocab_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
