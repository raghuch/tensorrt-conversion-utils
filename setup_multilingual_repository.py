#!/usr/bin/env python3
"""Create a Triton model repository for a multilingual CTC model.

This keeps the published ONNX contract: one encoder + one CTC head, and a
language-aware decode step that selects the right token mask and vocabulary
before argmax and BLANK removal.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from multilingual_packaging import (
    DEFAULT_BLANK_ID,
    DEFAULT_NUM_CLASSES,
    build_multilingual_decoder_config,
    build_multilingual_ensemble_config,
)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_metadata(repo_dir: Path, language_mask_file: str = None, vocab_file: str = None):
    repo_dir.mkdir(parents=True, exist_ok=True)
    if language_mask_file:
        src = Path(language_mask_file)
        dst = repo_dir / "language_masks.json"
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    if vocab_file:
        src = Path(vocab_file)
        dst = repo_dir / "vocab.json"
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


def build_repository(args):
    repo_dir = Path(args.repo)
    ensure_metadata(repo_dir, args.language_mask_file, args.vocab_file)

    feature_extractor = """
name: "feature-extractor"
platform: "pytorch_libtorch"
default_model_filename: "model.pt"
max_batch_size: 64

input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }},
  {{
    name: "input__1"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape {{ shape: [] }}
  }}
]

output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 80, -1 ]
  }},
  {{
    name: "output__1"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape {{ shape: [] }}
  }}
]
""".lstrip()

    conformer_trt = """
name: "conformer_trt"
platform: "tensorrt_plan"
default_model_filename: "model.plan"
max_batch_size: 64

input [
  {{
    name: "audio_signal"
    data_type: TYPE_FP32
    dims: [ 80, -1 ]
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
""".lstrip()

    decoder_config = build_multilingual_decoder_config(
        num_classes=args.num_classes,
        language_mask_file=args.language_mask_file_name,
        vocab_file=args.vocab_file_name,
        blank_id=args.blank_id,
    )
    ensemble_config = build_multilingual_ensemble_config(
        repo_dir=str(repo_dir),
        num_classes=args.num_classes,
        language_mask_file=args.language_mask_file_name,
        vocab_file=args.vocab_file_name,
        blank_id=args.blank_id,
    )

    write_text(repo_dir / "feature-extractor" / "config.pbtxt", feature_extractor)
    write_text(repo_dir / "conformer_trt" / "config.pbtxt", conformer_trt)
    write_text(repo_dir / "multilingual_decoder" / "config.pbtxt", decoder_config)
    write_text(repo_dir / "multilingual_ensemble" / "config.pbtxt", ensemble_config)

    backend = '''#!/usr/bin/env python3
import json
import os
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_dir = args.get("model_repository")
        self.model_dir = model_dir
        self.blank_id = int(os.environ.get("BLANK_ID", 256))
        self.mask_path = os.path.join(model_dir, "language_masks.json")
        self.vocab_path = os.path.join(model_dir, "vocab.json")
        with open(self.mask_path, "r", encoding="utf-8") as handle:
            self.language_masks = json.load(handle)
        with open(self.vocab_path, "r", encoding="utf-8") as handle:
            self.vocab = json.load(handle)

    def _decode_language(self, logits, language):
        if language not in self.language_masks:
            raise ValueError(f"Unsupported language: {language}")
        if language not in self.vocab:
            raise ValueError(f"No vocabulary for: {language}")
        mask = self.language_masks[language]
        vocab = self.vocab[language]
        selected = logits[:, :, mask]
        argmax = np.argmax(selected, axis=-1)
        collapsed = []
        prev = None
        for token_id in np.asarray(argmax).reshape(-1).tolist():
            if prev != token_id:
                collapsed.append(int(token_id))
            prev = token_id
        text = "".join(vocab[token_id] for token_id in collapsed if token_id != self.blank_id)
        return text.replace("▁", " ").strip()

    def execute(self, requests):
        responses = []
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "LOG_PROBS").as_numpy()
            lang_tensor = pb_utils.get_input_tensor_by_name(request, "LANGUAGE")
            language = lang_tensor.as_numpy().reshape(-1)[0].decode("utf-8")
            transcript = self._decode_language(logits, language)
            out = pb_utils.Tensor("TRANSCRIPT", np.asarray([transcript], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        pass
'''

    backend_path = repo_dir / "multilingual_decoder" / "1" / "model.py"
    backend_path.parent.mkdir(parents=True, exist_ok=True)
    backend_path.write_text(backend, encoding="utf-8")
    print(f"Wrote {backend_path}")


def add_arguments(parser):
    parser.add_argument("--repo", default="./model_repository_multilingual", help="Target Triton model repository")
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES, help="Output classes for the CTC head")
    parser.add_argument("--blank-id", type=int, default=DEFAULT_BLANK_ID, help="CTC blank id")
    parser.add_argument("--language-mask-file", default="language_masks.json", help="JSON mapping language -> mask")
    parser.add_argument("--vocab-file", default="vocab.json", help="JSON mapping language -> vocabulary")
    parser.add_argument("--dry-run", action="store_true", help="Print config but do not write files")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()

    args.language_mask_file_name = os.path.basename(args.language_mask_file)
    args.vocab_file_name = os.path.basename(args.vocab_file)

    if args.dry_run:
        print(build_multilingual_decoder_config(
            num_classes=args.num_classes,
            language_mask_file=args.language_mask_file_name,
            vocab_file=args.vocab_file_name,
            blank_id=args.blank_id,
        ))
        print(build_multilingual_ensemble_config(
            repo_dir=args.repo,
            num_classes=args.num_classes,
            language_mask_file=args.language_mask_file_name,
            vocab_file=args.vocab_file_name,
            blank_id=args.blank_id,
        ))
        return 0

    build_repository(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
