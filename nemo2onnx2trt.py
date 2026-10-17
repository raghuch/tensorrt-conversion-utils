#!/usr/bin/env python3
"""NeMo Conformer-CTC -> ONNX -> TensorRT conversion pipeline.

Modular replacement for the notebook cell-by-cell flow in
``01_nemo2onnx2trt_devanagari.ipynb``::

    gpus   report the local GPUs and the engines that would be built
    onnx   restore the .nemo model and export ONNX
    fold   fold constants with polygraphy surgeon sanitize
    trt    build one engine per local GPU architecture
    all    onnx -> fold -> trt

Point it at a .nemo model and everything else is derived::

    python3 nemo2onnx2trt.py all Conformer-CTC-BPE.nemo

The GPU is detected from the machine, so the same command works on a V100,
A100 or L40S box. TensorRT engines only run on the GPU model they were built
on, so one engine per architecture is produced and each plan is tagged with
its GPU, e.g. ``model_l40s.plan`` or ``model_v100-sxm2-32gb.plan``. Devices
that share a compute capability are covered by a single engine. Run the same
command on each GPU machine to get an engine for every GPU you have.

The default optimization profile covers 0.1s to 30s of audio (3001 frames at
the feature extractor's 10ms hop), and the TensorRT build workspace defaults
to 4096 MiB -- the notebook's 30000 was a generous ceiling, not a
requirement, and build workspace does not affect run-time memory.

The TensorRT step reuses ``converter.py`` (imported, never modified) so the
network fixes that keep Conformer accurate in FP16 stay in a single place.
All third-party imports are lazy so ``--help`` works outside the NeMo
container.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from types import SimpleNamespace

from deps import require
from paths import default_paths, model_name

# converter.py is vendored beside this script, so it resolves from sys.path[0]
# without any path juggling. See VENDORED.md.

DEFAULT_ONNX_OPSET = 18
DEFAULT_PRECISION = "fp16"
DEFAULT_BATCHSIZE_RANGE = "1,4,64"
# 3001 frames at the 10ms hop of the feature extractor (window_stride=0.01 in
# 02_nemofe2ts.ipynb) is 30.01s of audio, so the default profile already
# covers 30s utterances.
DEFAULT_SHAPE_RANGES = ["1,1001,3001"]
SUPPORTED_PRECISIONS = ("fp32", "fp16")

# TensorRT workspace for the *build* only; it caps the scratch memory the
# builder may use to pick tactics, and does not affect the memory the engine
# needs at run time. The notebook used 30000, which is a ceiling rather than a
# requirement, so keep a much smaller default and raise it only if a build
# complains.
DEFAULT_WORKSPACE_SIZE = 4096

# Fraction of a GPU's memory used when --workspace-size auto is requested.
AUTO_WORKSPACE_RATIO = 0.6
AUTO_WORKSPACE_MAX_MIB = 30000

# Compute capability fallback for GPUs whose driver does not report one.
KNOWN_COMPUTE_CAPABILITIES = {
    "v100": "7.0",
    "t4": "7.5",
    "a100": "8.0",
    "a30": "8.0",
    "a10": "8.6",
    "a40": "8.9",
    "l4": "8.9",
    "l40": "8.9",
    "4090": "8.9",
    "h100": "9.0",
    "h200": "9.0",
}
VENDOR_PREFIXES = ("nvidia", "tesla", "geforce", "instinct", "amd", "radeon")


def print_with_datetime(msg):
    print(f"{datetime.now()} {msg}")


def gpu_tag(name):
    """Short, filename-safe identifier for a GPU model name."""
    tag = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    parts = [p for p in tag.split("-") if p not in VENDOR_PREFIXES]
    return "-".join(parts) or tag


def _query_nvidia_smi(fields):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "nvidia-smi was not found. This script reads the local GPU to decide "
            "which engines to build; run it on a machine with an NVIDIA driver, "
            "or start a container with '--gpus all'."
        ) from None
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return result.stdout


def _visible_physical_indices():
    """Physical GPU indices torch can reach, honouring CUDA_VISIBLE_DEVICES.

    nvidia-smi ignores CUDA_VISIBLE_DEVICES, so without this the script would
    try to build on a GPU the process cannot see.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    try:
        return [int(part) for part in raw.split(",") if part.strip() != ""]
    except ValueError:
        return None


def detect_gpus():
    """Describe the reachable GPUs as
    {index, ordinal, name, tag, capability, memory_mib}.

    `index` is the physical nvidia-smi index, `ordinal` is the index torch
    uses (they differ when CUDA_VISIBLE_DEVICES selects a subset).
    """
    output = _query_nvidia_smi("index,name,compute_cap,memory.total")
    reachable = _visible_physical_indices()
    gpus = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        index, name, capability, memory = (f.strip() for f in line.split(",", 3))
        index = int(index)
        if reachable is None:
            ordinal = index
        elif index in reachable:
            ordinal = reachable.index(index)
        else:
            continue
        if capability in ("", "[N/A]"):
            lowered = name.lower()
            for model, cap in KNOWN_COMPUTE_CAPABILITIES.items():
                if model in lowered:
                    capability = cap
                    break
        gpus.append(
            {
                "index": index,
                "ordinal": ordinal,
                "name": name,
                "tag": gpu_tag(name),
                "capability": capability,
                "memory_mib": int(float(memory)),
            }
        )
    if not gpus:
        raise RuntimeError(
            "No reachable GPU found. Check nvidia-smi and CUDA_VISIBLE_DEVICES"
            + (f"={os.environ.get('CUDA_VISIBLE_DEVICES')!r}." if
               os.environ.get("CUDA_VISIBLE_DEVICES") else ".")
        )
    return gpus


def group_gpus_by_capability(gpus):
    """Group GPUs that share a compute capability, preserving device order.

    Engines are compatible across devices with the same compute capability, so
    one engine per group covers every device in that group.
    """
    groups = []
    for gpu in gpus:
        for group in groups:
            if group["capability"] == gpu["capability"]:
                group["gpus"].append(gpu)
                break
        else:
            groups.append({"capability": gpu["capability"], "gpus": [gpu]})
    return groups


def tagged_engine_path(engine_path, tag):
    """models_latest/conformer_trt/model.plan -> .../model_l40s.plan"""
    stem, ext = os.path.splitext(engine_path)
    return f"{stem}_{tag}{ext}"


def select_device(gpu):
    """Make `gpu` the device that TensorRT will build for."""
    try:
        import torch

        torch.cuda.set_device(gpu["ordinal"])
        return f"using CUDA device {gpu['ordinal']} ({gpu['name']})"
    except Exception as e:  # torch missing or no CUDA support
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        return f"CUDA_VISIBLE_DEVICES={gpu['index']} ({type(e).__name__})"


def resolve_workspace_size(value, gpu):
    """Turn 'auto' into a MiB value that fits the target GPU."""
    if str(value).lower() != "auto":
        return int(value)
    if gpu is None:
        return AUTO_WORKSPACE_MAX_MIB
    fitted = int(gpu["memory_mib"] * AUTO_WORKSPACE_RATIO)
    return min(fitted, AUTO_WORKSPACE_MAX_MIB)


def setup_device(cuda_visible_devices=None):
    """Select the device for the NeMo export step.

    The GPU is detected automatically; CUDA_VISIBLE_DEVICES is only touched
    when the caller passes it explicitly, so an existing setting in the
    environment is respected. It must be set before torch is imported, which
    is why the import lives in this function.
    """
    if cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    torch = require("torch", "to run the export")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def restore_model(nemo_path, device):
    nemo_asr = require("nemo.collections.asr", "to restore the .nemo model")

    print_with_datetime(f"Restore {nemo_path}")
    model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_path)
    model = model.to(device)
    model.eval()
    return model


def export_onnx(model, onnx_path, opset):
    ensure_parent_dir(onnx_path)
    print_with_datetime(f"Export ONNX (opset {opset}) to {onnx_path}")
    model.export(onnx_path, onnx_opset_version=opset)


def fold_onnx(onnx_path, folded_path, polygraphy_bin="polygraphy"):
    ensure_parent_dir(folded_path)
    print_with_datetime(f"Fold constants: {onnx_path} -> {folded_path}")
    subprocess.run(
        [
            polygraphy_bin,
            "surgeon",
            "sanitize",
            onnx_path,
            "--fold-constants",
            "-o",
            folded_path,
        ],
        check=True,
    )


def _to_converter_args(args):
    """Adapt argparse results to what converter.create_config expects."""
    workspace = str(args.workspace_size).strip()
    if workspace.lower() == "auto":
        # Overwritten per GPU by build_trt_engine.
        workspace_size = AUTO_WORKSPACE_MAX_MIB
    else:
        workspace_size = int(workspace)
    return SimpleNamespace(
        precision=args.precision,
        noTF32=args.noTF32,
        workspace_size=workspace_size,
        batchsize_range=args.batchsize_range,
        shape_ranges=args.shape_ranges,
        use_variable_batchsize=(not args.no_variable_batchsize),
        use_internal_fix=args.use_internal_fix,
    )


def build_trt_engine(args, onnx_path, engine_path, gpu=None):
    require("polygraphy", "to build the engine")
    from polygraphy.backend.trt import (
        engine_from_network,
        network_from_onnx_path,
        save_engine,
    )
    from polygraphy.logger import G_LOGGER as PG_LOGGER

    converter = require("converter", "to build the engine")

    ensure_parent_dir(engine_path)
    conv_args = _to_converter_args(args)
    if str(args.workspace_size).strip().lower() == "auto":
        conv_args.workspace_size = resolve_workspace_size("auto", gpu)
    print_with_datetime(
        f"Workspace size: {conv_args.workspace_size} MiB"
        + (f" (device {gpu['index']}: {gpu['name']})" if gpu else "")
    )

    verbosity = PG_LOGGER.EXTRA_VERBOSE if args.verbose else PG_LOGGER.WARNING
    with PG_LOGGER.verbosity(verbosity):
        print_with_datetime("Try to parse ONNX.")
        builder, network, parser = network_from_onnx_path(onnx_path)

        print_with_datetime("Verify given args.")
        converter.verify_args(conv_args, builder)
        if not builder.platform_has_tf32 and not conv_args.noTF32:
            conv_args.noTF32 = True
            print_with_datetime("This GPU has no TF32, disabling it.")
        print_with_datetime("Setup optimization configuration.")
        trt_config = converter.create_config(conv_args, network)

        print_with_datetime(
            "Apply some fixes and tricks to "
            "keep Conformer model accuracy in FP16."
        )
        if args.precision != converter.PRECISION_NAME_FP32:
            if args.use_internal_fix:
                network = converter.fix_fp16_network(network, PG_LOGGER)
            else:
                network = converter.add_extra_fp32(network)

        print_with_datetime("Run optimization.")
        try:
            trt_engine = engine_from_network(
                (builder, network, parser),
                config=trt_config,
            )
        except Exception as e:
            raise RuntimeError(
                f"TensorRT could not build an engine with a workspace of "
                f"{conv_args.workspace_size} MiB on {gpu['name'] if gpu else 'this GPU'}. "
                f"If the log above mentions tactics, workspace or memory, retry with "
                f"a larger --workspace-size (for example "
                f"--workspace-size {conv_args.workspace_size * 4}). "
                f"Original error: {e}"
            ) from e
        print_with_datetime("Save model.")
        save_engine(trt_engine, engine_path)
    print_with_datetime(f"Saved {engine_path}")


def install_engine(engine_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(engine_path))
    print_with_datetime(f"Copy {engine_path} -> {dest}")
    shutil.copy2(engine_path, dest)
    print_with_datetime(f"Installed {dest}")


def build_engines_per_gpu(args, onnx_path):
    """Build one engine per distinct compute capability among local GPUs."""
    groups = group_gpus_by_capability(detect_gpus())
    print_with_datetime(
        f"Found {len(groups)} distinct GPU architecture(s) to build for"
    )

    built = []
    for group in groups:
        members = group["gpus"]
        gpu = members[0]
        tag = gpu["tag"]
        engine_path = tagged_engine_path(args.engine_path, tag)

        indices = ",".join(str(m["index"]) for m in members)
        print_with_datetime(
            f"--- {gpu['name']} (sm_{group['capability'].replace('.', '')}, "
            f"{gpu['memory_mib']} MiB) on device(s) {indices} ---"
        )
        if len(members) > 1:
            print_with_datetime(
                "These devices share a compute capability, one engine covers them."
            )
        print_with_datetime(select_device(gpu))

        started = time.perf_counter()
        build_trt_engine(args, onnx_path, engine_path, gpu)
        elapsed = time.perf_counter() - started
        built.append((tag, indices, engine_path, elapsed))

        if args.install_to:
            install_engine(engine_path, args.install_to)

    print_with_datetime("Summary")
    for tag, indices, engine_path, elapsed in built:
        size_mib = os.path.getsize(engine_path) / (1 << 20)
        print(f"  {tag:24s} device(s) {indices:8s} {size_mib:8.1f} MiB "
              f"{elapsed / 60:6.1f} min  {engine_path}")
    print_with_datetime(f"done. {len(built)} engine(s) built.")


def report_gpus(args):
    """Show local GPUs and the engines a --per-gpu run would produce."""
    gpus = detect_gpus()
    print_with_datetime(f"{len(gpus)} visible GPU(s)")
    for gpu in gpus:
        print(
            f"  device {gpu['index']}  {gpu['name']:28s} "
            f"sm_{gpu['capability'].replace('.', ''):4s} "
            f"{gpu['memory_mib']:7d} MiB  "
            f"workspace {resolve_workspace_size('auto', gpu)} MiB"
        )
    groups = group_gpus_by_capability(gpus)
    print(f"{len(groups)} distinct architecture(s) -> {len(groups)} engine(s)")
    for group in groups:
        members = group["gpus"]
        indices = ",".join(str(m["index"]) for m in members)
        print(f"  {members[0]['name']:28s} device(s) {indices:8s} -> "
              f"{os.path.basename(tagged_engine_path(args.engine_path, members[0]['tag']))}")


def add_paths_arguments(parser):
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help=(
            "Base directory for the ONNX, folded ONNX and plan files. "
            "Defaults to ./models_latest_<model-name>, so separate checkpoints "
            "never collide. Individual files can still be overridden with "
            "--onnx-path / --onnx-folded-path / --engine-path."
        ),
    )
    parser.add_argument(
        "--onnx-path",
        default=None,
        help="Path to write the exported ONNX (default: derived).",
    )
    parser.add_argument(
        "--onnx-folded-path",
        default=None,
        help="Path to write the folded ONNX (default: derived from the model).",
    )
    parser.add_argument(
        "--engine-path",
        default=None,
        help="Path to write the TensorRT engine (default: derived from the model).",
    )
    parser.add_argument(
        "--onnx-opset",
        default=DEFAULT_ONNX_OPSET,
        type=int,
        help="ONNX opset version used on export.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help=(
            "Override CUDA_VISIBLE_DEVICES for the export step. "
            "Left unset by default so the GPU is detected and any existing "
            "setting is respected."
        ),
    )


def add_trt_arguments(parser):
    parser.add_argument(
        "--precision",
        default=DEFAULT_PRECISION,
        choices=SUPPORTED_PRECISIONS,
        help="Target precision.",
    )
    parser.add_argument(
        "--noTF32",
        action="store_true",
        help="Disable TF32.",
    )
    parser.add_argument(
        "--workspace-size",
        default=DEFAULT_WORKSPACE_SIZE,
        help=(
            f"TensorRT build workspace in MiB (default {DEFAULT_WORKSPACE_SIZE}), "
            "or 'auto' to take a fraction "
            f"{AUTO_WORKSPACE_RATIO} of the target GPU's memory, at most "
            f"{AUTO_WORKSPACE_MAX_MIB} MiB. Raise this if a build fails "
            "complaining about tactics, workspace or memory."
        ),
    )
    parser.add_argument(
        "--no-variable-batchsize",
        action="store_true",
        help=(
            "Build one optimization profile per batch size in "
            "--batchsize-range instead of a single variable-batch profile. "
            "Variable batchsize is the default."
        ),
    )
    parser.add_argument(
        "--batchsize-range",
        default=DEFAULT_BATCHSIZE_RANGE,
        help=(
            "min_batch,avg_batch,max_batch. "
            "Used together with --shape-ranges, e.g. 1,4,64."
        ),
    )
    parser.add_argument(
        "--shape-ranges",
        default=DEFAULT_SHAPE_RANGES,
        nargs="+",
        help=(
            "min_len,opt_len,max_len per optimization config. "
            "(e.g., 1,1001,3001 means 0.1s to 30s, optimum 10s.)"
        ),
    )
    parser.add_argument(
        "--use-internal-fix",
        action="store_true",
        help="Switch a network fix method to NV internal version.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose log.",
    )


def add_model_argument(parser):
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help=(
            "Path to the .nemo model. Output paths are derived from its name "
            "unless overridden."
        ),
    )


def build_parser():
    paths_parent = argparse.ArgumentParser(add_help=False)
    add_paths_arguments(paths_parent)

    trt_parent = argparse.ArgumentParser(add_help=False)
    add_trt_arguments(trt_parent)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_help = (
        "After building, copy each engine into this Triton model repository "
        "directory (e.g. ./model_repository/conformer_trt/1)."
    )

    gpus = subparsers.add_parser(
        "gpus",
        parents=[paths_parent],
        help="Report the local GPUs and the engines that would be built.",
    )
    gpus.add_argument("model", nargs="?", default=None, help=argparse.SUPPRESS)

    onnx = subparsers.add_parser(
        "onnx",
        parents=[paths_parent],
        help="Restore the .nemo model and export ONNX.",
    )
    add_model_argument(onnx)

    fold = subparsers.add_parser(
        "fold",
        parents=[paths_parent],
        help="Fold constants in the exported ONNX model.",
    )
    fold.add_argument("model", nargs="?", default=None, help=argparse.SUPPRESS)

    trt = subparsers.add_parser(
        "trt",
        parents=[paths_parent, trt_parent],
        help="Build one engine per local GPU architecture from the folded ONNX.",
    )
    trt.add_argument("--install-to", default=None, help=install_help)
    trt.add_argument("model", nargs="?", default=None, help=argparse.SUPPRESS)

    run_all = subparsers.add_parser(
        "all",
        parents=[paths_parent, trt_parent],
        help="Run onnx, fold and trt in sequence.",
    )
    run_all.add_argument("--install-to", default=None, help=install_help)
    add_model_argument(run_all)

    return parser


def resolve_paths(args):
    """Fill in the output paths from --artifacts-dir or the model file name."""
    nemo_path = args.model
    if nemo_path is not None and not os.path.isfile(nemo_path):
        raise SystemExit(f"No such model file: {nemo_path}")
    args.nemo_model = nemo_path

    if nemo_path is None and args.artifacts_dir is None and args.onnx_path is None:
        if args.command != "gpus":
            raise SystemExit(
                f"'{args.command}' needs to know where to write. Pass the .nemo "
                "model, or --artifacts-dir DIR, or set the paths explicitly "
                "with --onnx-path / --onnx-folded-path / --engine-path."
            )
    if args.artifacts_dir is None and nemo_path is None:
        # `gpus` only needs a name to show; keep it obvious that it is a guess.
        print_with_datetime(
            "No model or --artifacts-dir given; showing placeholder output names."
        )

    onnx, folded, engine = default_paths(
        nemo_path or "model.nemo", args.artifacts_dir
    )
    if args.onnx_path is None:
        args.onnx_path = onnx
    if args.onnx_folded_path is None:
        args.onnx_folded_path = folded
    if args.engine_path is None:
        args.engine_path = engine
    return args


def main():
    args = resolve_paths(build_parser().parse_args())
    print_with_datetime(args)

    if args.command == "gpus":
        report_gpus(args)
        return

    if args.command in ("onnx", "all"):
        if args.nemo_model is None:
            raise SystemExit(
                f"{args.command} needs a .nemo model, "
                f"e.g. python3 nemo2onnx2trt.py {args.command} Conformer-CTC-BPE.nemo"
            )
        device = setup_device(args.cuda_visible_devices)
        model = restore_model(args.nemo_model, device)
        export_onnx(model, args.onnx_path, args.onnx_opset)

    if args.command in ("fold", "all"):
        fold_onnx(args.onnx_path, args.onnx_folded_path)

    if args.command in ("trt", "all"):
        build_engines_per_gpu(args, args.onnx_folded_path)

    print_with_datetime("done.")


if __name__ == "__main__":
    main()
