#!/usr/bin/env python3
"""Export the Conformer frontend (log-mel feature extractor) to TorchScript.

Modular replacement for 02_nemofe2ts.ipynb.

    python3 nemo_fe2ts.py Conformer-CTC-BPE-hi-nasim.nemo
    python3 nemo_fe2ts.py model.nemo --out ./model_repository/feature-extractor/1/model.pt

Two deliberate differences from the notebook:

1. The featurizer is built from the .nemo checkpoint's own model_config.yaml
   instead of hand-written keyword arguments. The notebook hardcodes them, and
   it has already drifted: the checkpoint says `dither: 1.0e-05` while the
   notebook passes `dither=0`. Reading the config removes that whole class of
   bug, and needs neither NeMo nor the model weights.

2. The export is verified instead of assumed. The notebook traces the module
   and then checks the output shape at the one shape it traced at. This script
   additionally proves the traced graph is shape-generic (one module must serve
   every batch and length, since config.pbtxt declares dims [-1]) and that the
   exported module is numerically identical to the eager one across several
   lengths, including a ragged batch.

The notebook's choice of torch.jit.trace over torch.jit.script is correct and
kept: BaseFeatures overrides __call__ rather than forward, so torch.jit.script
returns a RecursiveScriptModule with no entry point. Measured, not assumed --
see the module-level docs on export_module.
"""

import argparse
import os
import sys
import tarfile
from datetime import datetime

from deps import require
from paths import default_artifacts_dir

# features.py is vendored beside this script, so FilterbankFeatures resolves
# from sys.path[0] without any path juggling. See VENDORED.md.


# Keys in the checkpoint's preprocessor block that are named differently in
# FilterbankFeatures.__init__.
CONFIG_KEY_ALIASES = {
    "features": "n_filt",
    "pad_to": "pad_align",
}
# FilterbankFeatures defaults for anything the checkpoint does not specify.
FEATURIZER_DEFAULTS = {
    "sample_rate": 16000,
    "window_size": 0.025,
    "window_stride": 0.01,
    "window": "hann",
    "normalize": "per_feature",
    "n_fft": 512,
    "log": True,
    "frame_splicing": 1,
    "dither": 0.0,
    "preemph": 0.97,
    "lowfreq": 0,
    "highfreq": None,
    "pad_align": 0,
}
# Never taken from the config: these must be off or the exported graph would
# bake in random masking / fixed padding.
ALWAYS_OVERRIDDEN = {"spec_augment": None, "cutout_augment": None,
                     "pad_to_max_duration": False}


def print_with_datetime(msg):
    print(f"{datetime.now()} {msg}")


def read_checkpoint_config(nemo_path):
    """Return the parsed model_config.yaml from inside a .nemo tarball."""
    yaml = require("yaml", "to read model_config.yaml from the .nemo")

    with tarfile.open(nemo_path, "r") as tar:
        member = next((m for m in tar.getmembers()
                       if m.name.endswith("model_config.yaml")), None)
        if member is None:
            raise SystemExit(f"{nemo_path} has no model_config.yaml")
        config = yaml.safe_load(tar.extractfile(member))
    return config


def featurizer_kwargs(config, dither="keep"):
    """Map the checkpoint's preprocessor block onto FilterbankFeatures args."""
    pre = config.get("preprocessor") or {}
    if not pre:
        raise SystemExit(
            "This checkpoint has no `preprocessor` block, so the frontend "
            "settings are unknown. Pass them explicitly instead of guessing."
        )

    kwargs = dict(FEATURIZER_DEFAULTS)
    for key, value in pre.items():
        if key.startswith("_") or key == "pad_value":
            continue  # _target_, pad_value: not FilterbankFeatures arguments
        kwargs[CONFIG_KEY_ALIASES.get(key, key)] = value
    kwargs.update(ALWAYS_OVERRIDDEN)

    if dither == "zero":
        kwargs["dither"] = 0.0
    return kwargs


def build_featurizer(kwargs, device):
    FilterbankFeatures = require("features", "to build the featurizer").FilterbankFeatures
    return FilterbankFeatures(**kwargs).to(device)


def export_module(featurizer, example_args, method, device):
    """Export to TorchScript, preferring torch.jit.trace.

    Trace rather than script, because features.py's BaseFeatures overrides
    __call__ instead of forward. torch.jit.script compiles from forward, so it
    happily returns a RecursiveScriptModule with no forward attribute and no
    entry point -- the export "succeeds" and then raises
    "'RecursiveScriptModule' object has no attribute 'forward'" on first call.
    Trace calls the module the way Triton will, which is what is wanted here.

    The result is still shape-generic: the tracer records the x.size(0) /
    x.size(-1) reads in calculate_features as dynamic size ops rather than
    constants, so one traced module serves every batch and length. That is
    asserted after export rather than assumed -- see check_generic.
    """
    import torch

    if method in ("script", "auto"):
        try:
            with torch.no_grad():
                candidate = torch.jit.script(featurizer)
                # script() can succeed and still be uncallable, so probe it.
                candidate(*example_args)
            print_with_datetime("Exported with torch.jit.script")
            return candidate, "script"
        except Exception as e:
            if method == "script":
                raise SystemExit(
                    "torch.jit.script cannot export this featurizer: "
                    f"{type(e).__name__}: {str(e).splitlines()[0][:200]}\n"
                    "  features.py's BaseFeatures overrides __call__, not "
                    "forward, so TorchScript has no entry point to compile.\n"
                    "  Use --export-method trace (the default), which works."
                ) from None
            print_with_datetime("torch.jit.script unusable here, using trace:")
            print(f"    {type(e).__name__}: {str(e).splitlines()[0][:300]}")

    with torch.no_grad():
        module = torch.jit.trace(featurizer, example_args)
    print_with_datetime("Exported with torch.jit.trace")
    return module, "trace"


def make_case(batch, samples, ragged, device, seed=0):
    """Synthetic audio plus true lengths, mimicking a padded serving batch."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    if ragged:
        # Varied true lengths, zero-padded up to `samples`, as Triton sends it.
        lengths = torch.randint(
            max(1, samples // 2), samples + 1, (batch,), generator=generator
        )
    else:
        lengths = torch.full((batch,), samples, dtype=torch.int32)

    audio = torch.zeros(batch, samples, dtype=torch.float32)
    for i, length in enumerate(lengths.tolist()):
        audio[i, :length] = (
            torch.randn(length, generator=generator, dtype=torch.float32) * 0.1
        )
    return audio.to(device), lengths.to(device)


def frame_count(samples, hop_length):
    """Frames that torch.stft(center=True) actually produces: 1 + n // hop.

    Deliberately not ceil(n / hop) + 1, which is what
    FilterbankFeatures.get_seq_len returns for the *lengths* output. The two
    agree whenever the input is frame-aligned and differ by at most one
    otherwise (12345 samples -> 78 frames vs seq_len 79), which is the
    off-by-one features.py's stack_subsample_frames asserts about.
    """
    return 1 + samples // hop_length


def verify_module(module, featurizer, cases, device, hop_length, deterministic):
    """Fail unless the exported module matches eager at every shape.

    With dither=0 the comparison is exact and repeated runs must be
    bit-identical. With --dither keep the graph holds a randn_like, so exact
    equality is impossible and the comparison falls back to a tolerance.
    """
    import torch

    for batch, samples, ragged in cases:
        audio, lengths = make_case(batch, samples, ragged, device)
        with torch.no_grad():
            want, want_lens = featurizer(audio, lengths)
            got, got_lens = module(audio, lengths)

        expected_frames = frame_count(samples, hop_length)
        where = f"batch={batch} samples={samples} ragged={ragged}"
        if got.shape != want.shape:
            raise SystemExit(
                f"FAIL {where}: exported module returned {tuple(got.shape)}, "
                f"eager returns {tuple(want.shape)}"
            )
        if deterministic:
            if not torch.equal(got, want):
                diff = (got - want).abs().max().item()
                raise SystemExit(
                    f"FAIL {where}: max |exported - eager| = {diff:g}"
                )
        else:
            # dither>0: the graph itself is stochastic, so only require that
            # the two agree to within the noise that dither introduces.
            if not torch.allclose(got, want, atol=5e-2, rtol=5e-2):
                diff = (got - want).abs().max().item()
                raise SystemExit(
                    f"FAIL {where}: max |exported - eager| = {diff:g}, which "
                    f"is more than dither can explain"
                )
        if got.shape[2] != expected_frames:
            print_with_datetime(
                f"WARNING {where}: {got.shape[2]} frames, expected "
                f"{expected_frames} for {samples} samples at hop "
                f"{hop_length}"
            )
        print_with_datetime(
            f"  OK {where}: {tuple(got.shape)} frames, "
            f"lengths {tuple(got_lens.shape)}"
        )
    return True


def check_determinism(module, device, hop_length, deterministic):
    """A serving feature extractor must give the same answer twice."""
    import torch

    audio = torch.zeros(2, 16000, device=device)
    lengths = torch.full((2,), 16000, dtype=torch.int32, device=device)
    with torch.no_grad():
        first, _ = module(audio, lengths)
        second, _ = module(audio, lengths)
    if deterministic and not torch.equal(first, second):
        raise SystemExit(
            "FAIL: the exported module returns different features for the "
            "same input. Something stochastic survived the export (dither?); "
            "Triton clients would see non-reproducible transcriptions."
        )
    if not deterministic:
        print_with_datetime(
            "NOTE: --dither keep, so the module embeds a random op and is not "
            "bit-reproducible across calls."
        )
    else:
        print_with_datetime("Determinism check passed (two calls, identical)")


def check_generic(module, device, hop_length):
    """A traced graph pinned to one shape fails here rather than in production."""
    import torch

    saved = {}
    for batch, samples in ((1, 16000), (5, 8000), (7, 40000)):
        audio = torch.zeros(batch, samples, device=device)
        lengths = torch.full((batch,), samples, dtype=torch.int32, device=device)
        want = frame_count(samples, hop_length)
        try:
            with torch.no_grad():
                out, _ = module(audio, lengths)
        except Exception as e:
            saved[batch] = f"raised {type(e).__name__}"
            continue
        if tuple(out.shape) != (batch, out.shape[1], want):
            saved[batch] = (
                f"returned {tuple(out.shape)}, expected {want} frames"
            )
    if saved:
        detail = "; ".join(f"batch {b}: {why}" for b, why in saved.items())
        raise SystemExit(
            "FAIL: the exported module is pinned to the shape it was built at "
            f"({detail}).\n"
            "A Triton feature extractor must accept any batch and length, since "
            "config.pbtxt declares dims [-1].\n"
            "Re-run with --export-method script, or fix the shape-dependent "
            "arithmetic in features.py:calculate_features (the mask built from "
            "x.size(0) / x.size(-1))."
        )
    print_with_datetime("Shape-generality check passed (batch 1/5/7, 1s-2.5s)")


def add_arguments(parser):
    parser.add_argument("model", help="Path to the .nemo checkpoint.")
    parser.add_argument(
        "--artifacts-dir", default=None,
        help=(
            "Base directory for build artifacts. Defaults to "
            "./models_latest_<model-name>, the same default as "
            "nemo2onnx2trt.py, so one --artifacts-dir covers the ONNX, the "
            "plan and the feature extractor."
        ),
    )
    parser.add_argument(
        "--out", default=None,
        help=(
            "Where to write the TorchScript module. Defaults to "
            "<artifacts-dir>/feature-extractor/1/model.pt, which is the layout "
            "setup_model_repository.py expects."
        ),
    )
    parser.add_argument(
        "--export-method", default="trace", choices=("auto", "script", "trace"),
        help=(
            "trace (default) uses torch.jit.trace, which is what this module "
            "supports -- BaseFeatures overrides __call__, not forward, so "
            "torch.jit.script yields an uncallable module. auto tries script "
            "first and falls back to trace."
        ),
    )
    parser.add_argument(
        "--dither", default="zero", choices=("zero", "keep"),
        help=(
            "zero (default) exports with dither=0 even though the checkpoint "
            "says 1e-05, so the traced graph contains no randn_like and the "
            "served features are bit-reproducible. keep honours the "
            "checkpoint, which embeds a random op in the graph: trace-check "
            "then reports a large mismatch and two identical requests can "
            "differ. The 1e-05 amplitude is training-time noise and is "
            "amplified by the log and per-feature normalisation in near-silent "
            "frames, so it is not worth carrying into serving."
        ),
    )
    parser.add_argument(
        "--verify-samples", default="1600,16000,480000",
        help=(
            "Comma-separated audio lengths (samples) used to verify the "
            "export. 480000 is 30s at 16kHz. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--verify-batch", default="4",
        help="Batch size used for the numerical comparison. Default: %(default)s",
    )
    parser.add_argument(
        "--cuda-visible-devices", default=None,
        help=(
            "Override CUDA_VISIBLE_DEVICES. Left unset by default so the "
            "existing environment is respected."
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # torch is imported only after the environment is set, and only here, so
    # --help works without torch installed.
    torch = require("torch", "to export the feature extractor")

    print_with_datetime(args)
    if not os.path.isfile(args.model):
        raise SystemExit(f"No such checkpoint: {args.model}")

    artifacts = default_artifacts_dir(args.model, args.artifacts_dir)
    if args.out is None:
        args.out = f"{artifacts}/feature-extractor/1/model.pt"
        print_with_datetime(f"Output defaults to {args.out}")

    config = read_checkpoint_config(args.model)
    print_with_datetime(
        f"Checkpoint: {config.get('target', 'unknown class')} "
        f"(saved by NeMo {config.get('nemo_version', 'unknown')})"
    )

    kwargs = featurizer_kwargs(config, dither=args.dither)
    print_with_datetime(
        "Featurizer config from checkpoint: "
        + ", ".join(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
    )
    if args.dither == "zero" and kwargs["dither"] == 0.0:
        print_with_datetime("dither forced to 0.0 (--dither zero)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    featurizer = build_featurizer(kwargs, device)
    hop_length = featurizer.hop_length
    print_with_datetime(f"Device: {device} (hop_length {hop_length})")

    lengths_to_check = [int(v) for v in args.verify_samples.split(",")]
    example_audio, example_lengths = make_case(
        int(args.verify_batch), max(lengths_to_check), False, device
    )
    module, method = export_module(
        featurizer, (example_audio, example_lengths), args.export_method, device
    )

    cases = [
        (int(args.verify_batch), n, False) for n in lengths_to_check
    ] + [(int(args.verify_batch), lengths_to_check[-1], True)]
    deterministic = kwargs["dither"] == 0
    print_with_datetime(
        f"Verifying with method={method} over "
        f"{[c[1] for c in cases]} samples (dither={kwargs['dither']})"
    )
    verify_module(module, featurizer, cases, device, hop_length, deterministic)
    check_generic(module, device, hop_length)
    check_determinism(module, device, hop_length, deterministic)

    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    module.save(args.out)
    print_with_datetime(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    print_with_datetime(
        "config.pbtxt I/O: input__0 FP32 [-1], input__1 INT32 [1] (reshape []) "
        f"-> output__0 FP32 [{kwargs['n_filt']}, -1], output__1 INT32 [1] "
        "(reshape [])"
    )
    print_with_datetime(
        f"Frames = 1 + floor(samples / {hop_length}), so 30s of audio = "
        f"{frame_count(30 * kwargs['sample_rate'], hop_length)} frames -- the "
        "3001 in the engine's --shape-ranges."
    )
    print_with_datetime("done.")


if __name__ == "__main__":
    main()
