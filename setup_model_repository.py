#!/usr/bin/env python3
"""Generate the Triton model repository's config.pbtxt files.

Modular replacement for 04_triton_server_setup.ipynb, for the pyctcdecode flow
(no topk model, no separate beam-decoder TorchScript):

    feature-extractor-ts-trace   pytorch_libtorch   wav -> log-mel features
    conformer_trt                tensorrt_plan     features -> log-probs
    beam-decoder                 python            log-probs -> transcript
    conformer_trt_ensemble       ensemble          wav -> transcripts

Usage:
    python3 setup_model_repository.py --repo ./model_repository_topk_hi_dn
    python3 setup_model_repository.py --repo ./model_repository_topk_hi_dn \\
        --decoder beam --kenlm-model /models/lm/4gram.bin

Then copy the artifacts in (the plan from nemo2onnx2trt.py, the feature
extractor from nemo_fe2ts.py, the engine's vocabulary) and start Triton:

    tritonserver --model-repository=./model_repository_topk_hi_dn
"""

import argparse
import os
import shutil
import sys
import tarfile
from datetime import datetime

from deps import require

DEFAULT_REPO = "./model_repository_topk_hi_dn"
DEFAULT_MAX_BATCH = 64  # the engine profile is batch 1..64; see --max-batch

N_FILT = 80       # mel bins, from the checkpoint's preprocessor block

# Fallback only. The real number is the engine's channel count, which is
# len(decoder.vocabulary) + 1 -- the vocabulary plus the CTC blank that
# ConvASRDecoder appends as the LAST channel. 129 is what the Conformer
# checkpoint in this repo emits (128 tokens). It is used when no vocabulary is
# available to read; see resolve_num_classes().
DEFAULT_NUM_CLASSES = 129


def print_with_datetime(msg):
    print(f"{datetime.now()} {msg}")


def feature_extractor_config(args):
    if args.fe_io_names == "real":
        # The traced module's real graph input names (see nemo_fe2ts.py).
        # Outputs stay output__0/output__1 because the traced forward returns
        # a single tuple, which Triton flattens.
        audio_in, length_in = "audio", "seq_len"
    else:
        audio_in, length_in = "input__0", "input__1"
    return f"""
name: "feature-extractor"
platform: "pytorch_libtorch"
default_model_filename: "model.pt"
max_batch_size: {args.max_batch}

input [
  {{
    name: "{audio_in}"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }},
  {{
    name: "{length_in}"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape {{ shape: [] }}
  }}
]

output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {N_FILT}, -1 ]
  }},
  {{
    name: "output__1"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape {{ shape: [] }}
  }}
]
"""


def conformer_trt_config(args):
    return f"""
name: "conformer_trt"
platform: "tensorrt_plan"
default_model_filename: "{args.plan_filename}"

# Must not exceed the maximum batch in the engine's optimization profile, or
# Triton will assemble a request the plan rejects.
max_batch_size: {args.max_batch}

input [
  {{
    name: "audio_signal"
    data_type: TYPE_FP32
    dims: [ {N_FILT}, -1 ]
  }},
  {{
    name: "length"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape {{ shape: [] }}
  }}
]

output [
  {{
    name: "logprobs"
    data_type: TYPE_FP32
    dims: [ -1, {args.num_classes} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 4, 8, 16, 32, 64 ]
}}
"""


def beam_decoder_config(args):
    parameters = [
        ("decoder", args.decoder),
        ("vocab_file", args.vocab_filename),
        ("blank_id", str(args.num_classes - 1)),
    ]
    if args.decoder == "beam":
        parameters += [("alpha", str(args.alpha)), ("beta", str(args.beta))]
    blocks = "\n".join(
        f'parameters: {{ key: "{key}" value: {{ string_value: "{value}" }} }}'
        for key, value in parameters
    )
    if args.kenlm_model:
        blocks += (
            '\nparameters: { key: "kenlm_model_path" '
            f'value: {{ string_value: "{args.kenlm_model}" }} }}'
        )
    return f"""
name: "beam-decoder"
backend: "python"
max_batch_size: {args.max_batch}

input [
  {{
    name: "LOG_PROBS"
    data_type: TYPE_FP32
    dims: [ -1, {args.num_classes} ]
  }}
]

# Optional: the feature extractor's frame count, converted to output frames by
# the decoder to trim padding. Omit it and every returned frame is decoded.
input [
  {{
    name: "NUM_TIME_STEPS"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }}
]

output [
  {{
    name: "TRANSCRIPT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

{blocks}

instance_group [ {{ kind: KIND_CPU }} ]
"""


def ensemble_config(args):
    return f"""
name: "conformer_trt_ensemble"
platform: "ensemble"
max_batch_size: {args.max_batch}

input {{
  name: "AUDIO_SIGNAL"
  data_type: TYPE_FP32
  dims: [ -1 ]
}}
input {{
  name: "NUM_SAMPLES"
  data_type: TYPE_INT32
  dims: [ 1 ]
}}
output {{
  name: "TRANSCRIPTS"
  data_type: TYPE_STRING
    dims: [ 1 ]
}}

ensemble_scheduling {{
  step [
    {{
      model_name: "feature-extractor"
      model_version: -1
      input_map {{
        key: "{args.fe_audio_key}"
        value: "AUDIO_SIGNAL"
      }}
      input_map {{
        key: "{args.fe_length_key}"
        value: "NUM_SAMPLES"
      }}
      output_map {{
        key: "output__0"
        value: "audio_features"
      }}
      output_map {{
        key: "output__1"
        value: "NUM_TIME_STEPS"
      }}
    }},
    {{
      model_name: "conformer_trt"
      model_version: -1
      input_map {{
        key: "audio_signal"
        value: "audio_features"
      }}
      input_map {{
        key: "length"
        value: "NUM_TIME_STEPS"
      }}
      output_map {{
        key: "logprobs"
        value: "logits"
      }}
    }},
    {{
      model_name: "beam-decoder"
      model_version: -1
      input_map {{
        key: "LOG_PROBS"
        value: "logits"
      }}
      input_map {{
        key: "NUM_TIME_STEPS"
        value: "NUM_TIME_STEPS"
      }}
      output_map {{
        key: "TRANSCRIPT"
        value: "TRANSCRIPTS"
      }}
    }}
  ]
}}
"""


def add_arguments(parser):
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Model repository root.")
    parser.add_argument(
        "--max-batch", type=int, default=DEFAULT_MAX_BATCH,
        help=(
            "max_batch_size for the batched models. Must not exceed the "
            f"engine profile maximum ({DEFAULT_MAX_BATCH} with the default "
            "nemo2onnx2trt.py settings)."
        ),
    )
    parser.add_argument("--plan-filename", default="model.plan")
    parser.add_argument(
        "--nemo-model",
        help=(
            "Path to the .nemo checkpoint. Used to generate "
            "beam-decoder/1/vocab.txt from decoder.vocabulary; without it the "
            "vocabulary is left alone."
        ),
    )
    parser.add_argument(
        "--skip-beam-decoder", action="store_true",
        help="Do not copy the python_backend model.py into the repository.",
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help=(
            "Engine output channels = vocabulary tokens + 1 CTC blank. "
            "Normally left unset: it is read from --nemo-model's "
            f"decoder.vocabulary, or from an existing vocab.txt, and only "
            f"falls back to {DEFAULT_NUM_CLASSES} when neither is available. "
            "Set it explicitly only when building a repository by hand."
        ),
    )
    parser.add_argument("--vocab-filename", default="vocab.txt")
    parser.add_argument("--decoder", default="greedy", choices=("greedy", "beam"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--kenlm-model", default="", help="ARPA/binary LM for beam search.")
    parser.add_argument(
        "--fe-io-names", default="generic", choices=("generic", "real"),
        help=(
            "generic uses input__0/input__1, which is what the previous "
            "notebooks used. real uses the traced module's own names "
            "(audio/seq_len). Try generic first; switch to real if Triton "
            "reports a missing input name."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the configs, write nothing."
    )


def configs_for(args):
    args.fe_audio_key = "input__0" if args.fe_io_names == "generic" else "audio"
    args.fe_length_key = "input__1" if args.fe_io_names == "generic" else "seq_len"
    return {
        "feature-extractor": feature_extractor_config(args),
        "conformer_trt": conformer_trt_config(args),
        "beam-decoder": beam_decoder_config(args),
        "conformer_trt_ensemble": ensemble_config(args),
    }


def check_repository(repo, configs, num_classes):
    """Warn about the mistakes that make Triton refuse to start."""
    problems = []
    for name in configs:
        if not os.path.isdir(os.path.join(repo, name)):
            problems.append(f"{name}/ does not exist")

    version_dir = os.path.join(repo, "conformer_trt", "1")
    if not os.path.isdir(version_dir):
        found = []
        conformer = os.path.join(repo, "conformer_trt")
        if os.path.isdir(conformer):
            found = sorted(
                d for d in os.listdir(conformer)
                if os.path.isdir(os.path.join(conformer, d))
            )
        problems.append(
            f"conformer_trt/1/ does not exist (versions found: {found or 'none'})"
        )
    elif not os.path.isfile(os.path.join(version_dir, "model.plan")):
        problems.append("conformer_trt/1/model.plan is missing")

    fe_version = os.path.join(repo, "feature-extractor", "1")
    if os.path.isdir(fe_version) and not os.path.isfile(
        os.path.join(fe_version, "model.pt")
    ):
        problems.append("feature-extractor/1/model.pt is missing")

    vocab = os.path.join(repo, "beam-decoder", "1", "vocab.txt")
    if os.path.isfile(vocab):
        with open(vocab, "r", encoding="utf-8") as handle:
            count = len(handle.read().splitlines())
        if count != num_classes - 1:
            problems.append(
                f"beam-decoder/1/vocab.txt has {count} tokens, but the "
                f"configs were written for {num_classes - 1} "
                f"({num_classes} channels - 1 blank). Re-run with "
                "--nemo-model so both come from the same vocabulary."
            )
    else:
        problems.append("beam-decoder/1/vocab.txt is missing")

    if not os.path.isfile(os.path.join(repo, "beam-decoder", "1", "model.py")):
        problems.append("beam-decoder/1/model.py is missing")

    return problems


def read_vocabulary(nemo_path):
    """The decoder's token list, in output-channel order, from the checkpoint.

    The `vocab.txt` inside a .nemo is the SentencePiece vocabulary (127 '##'
    style pieces) and does NOT line up with the model's 128 decoder tokens.
    The authoritative list is `decoder.vocabulary` in model_config.yaml, so
    that is what is used here.
    """
    yaml = require("yaml", "to read the vocabulary from the .nemo")

    with tarfile.open(nemo_path, "r") as tar:
        member = next((m for m in tar.getmembers()
                       if m.name.endswith("model_config.yaml")), None)
        if member is None:
            raise SystemExit(f"{nemo_path} has no model_config.yaml")
        config = yaml.safe_load(tar.extractfile(member))

    vocabulary = ((config.get("decoder") or {}).get("vocabulary")) or []
    if not vocabulary:
        raise SystemExit(
            f"{nemo_path} has no decoder.vocabulary, so the label list for the "
            "beam decoder is unknown. Pass --nemo-model for a checkpoint that "
            "has one."
        )
    return vocabulary


def resolve_num_classes(args):
    """The engine's channel count: vocabulary tokens + 1 for the CTC blank.

    Sources, in priority order:

      1. --num-classes, if passed.
      2. len(decoder.vocabulary) + 1 from --nemo-model.
      3. token count + 1 of a vocab.txt already in the repository.
      4. DEFAULT_NUM_CLASSES (129), with a note saying so.

    Every config.pbtxt dimension and the blank_id come from this, so a model
    with a different vocabulary does not need the source edited.
    """
    if args.num_classes is not None:
        if args.num_classes < 2:
            raise SystemExit(
                f"--num-classes {args.num_classes} is too small; it is the "
                "vocabulary size plus the blank, so it must be at least 2."
            )
        source = "--num-classes"
        if args.nemo_model:
            actual = len(read_vocabulary(args.nemo_model)) + 1
            if actual != args.num_classes:
                raise SystemExit(
                    f"--num-classes {args.num_classes} disagrees with "
                    f"{args.nemo_model}, whose vocabulary gives "
                    f"{actual} channels ({actual - 1} tokens + 1 blank). "
                    "Trust the checkpoint: drop the flag."
                )
    elif args.nemo_model:
        count = len(read_vocabulary(args.nemo_model))
        args.num_classes = count + 1
        source = f"{args.nemo_model} (decoder.vocabulary)"
    else:
        existing = os.path.join(
            args.repo, "beam-decoder", "1", args.vocab_filename
        )
        if os.path.isfile(existing):
            with open(existing, "r", encoding="utf-8") as handle:
                count = len(handle.read().splitlines())
            if count:
                args.num_classes = count + 1
                source = existing
            else:
                args.num_classes = DEFAULT_NUM_CLASSES
                source = f"default ({DEFAULT_NUM_CLASSES}); {existing} is empty"
        else:
            args.num_classes = DEFAULT_NUM_CLASSES
            source = (
                f"default ({DEFAULT_NUM_CLASSES}); no --nemo-model and no "
                f"{args.vocab_filename} in the repository"
            )

    print_with_datetime(
        f"{args.num_classes} output channels = "
        f"{args.num_classes - 1} vocabulary tokens + 1 CTC blank, from {source}"
    )
    if args.num_classes == DEFAULT_NUM_CLASSES and source.startswith("default"):
        print(
            "  If the engine emits a different number of channels, re-run with "
            "--nemo-model MODEL.nemo so the count is read from the checkpoint."
        )
    return args.num_classes


def install_beam_decoder(args):
    """Copy the python_backend model in and write vocab.txt from the .nemo."""
    target_dir = os.path.join(args.repo, "beam-decoder", "1")
    os.makedirs(target_dir, exist_ok=True)

    source = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "beam_decoder", "1", "model.py")
    destination = os.path.join(target_dir, "model.py")
    shutil.copy2(source, destination)
    print_with_datetime(f"Copied {source} -> {destination}")

    if args.nemo_model:
        vocabulary = read_vocabulary(args.nemo_model)
        vocab_path = os.path.join(target_dir, args.vocab_filename)
        with open(vocab_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(vocabulary) + "\n")
        print_with_datetime(
            f"Wrote {vocab_path} ({len(vocabulary)} tokens, blank at "
            f"index {len(vocabulary)})"
        )
    elif not os.path.isfile(os.path.join(target_dir, args.vocab_filename)):
        print_with_datetime(
            f"No --nemo-model given and {args.vocab_filename} is absent; "
            "re-run with --nemo-model to generate it."
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()

    resolve_num_classes(args)
    configs = configs_for(args)
    if args.max_batch > DEFAULT_MAX_BATCH:
        print_with_datetime(
            f"WARNING --max-batch {args.max_batch} exceeds the default engine "
            f"profile maximum of {DEFAULT_MAX_BATCH}. Either rebuild the "
            "engine with a larger --batchsize-range or lower this."
        )

    for name, text in configs.items():
        target = os.path.join(args.repo, name, "config.pbtxt")
        if args.dry_run:
            print_with_datetime(f"--- {target} (dry run) ---")
            print(text)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text.lstrip("\n"))
        print_with_datetime(f"Wrote {target}")

    if args.dry_run:
        return 0

    if not args.skip_beam_decoder:
        install_beam_decoder(args)

    problems = check_repository(args.repo, configs, args.num_classes)
    if problems:
        print_with_datetime("Repository is not complete yet:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nTriton will fail to load until these are fixed. The two build "
            "artifacts are the only things left to copy in:\n"
            "  conformer_trt/1/model.plan     <- nemo2onnx2trt.py trt ...\n"
            "  feature-extractor/1/model.pt   <- nemo_fe2ts.py ...\n"
            "(beam-decoder/1/model.py and vocab.txt are handled by this script)"
        )
        return 1

    print_with_datetime(f"All four configs written and the repository looks complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
