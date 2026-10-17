#!/usr/bin/env python3
"""Helpers for multilingual CTC TensorRT/Triton packaging.

The model is multilingual at decode time: one shared encoder/CTC head is used,
while language-specific masks and vocabularies are applied before argmax and
final text conversion. This preserves the exact logic used by the published
ONNX wrapper for multi-language CTC decoding.
"""

import json
from typing import Dict, Iterable, List, Tuple

DEFAULT_BLANK_ID = 256
DEFAULT_NUM_CLASSES = 5633


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_multilingual_decoder_config(
    num_classes: int = DEFAULT_NUM_CLASSES,
    language_mask_file: str = "language_masks.json",
    vocab_file: str = "vocab.json",
    blank_id: int = DEFAULT_BLANK_ID,
):
    return f"""
name: "multilingual_decoder"
backend: "python"
max_batch_size: 8

input [
  {{
    name: "LOG_PROBS"
    data_type: TYPE_FP32
    dims: [ -1, {num_classes} ]
  }},
  {{
    name: "LANGUAGE"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "TRANSCRIPT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

parameters: {{ key: "language_masks_file" value: {{ string_value: \"{language_mask_file}\" }} }}
parameters: {{ key: "vocab_file" value: {{ string_value: \"{vocab_file}\" }} }}
parameters: {{ key: "blank_id" value: {{ string_value: \"{blank_id}\" }} }}

instance_group [ {{ kind: KIND_CPU }} ]
""".lstrip()


def build_multilingual_ensemble_config(
    repo_dir: str,
    num_classes: int = DEFAULT_NUM_CLASSES,
    language_mask_file: str = "language_masks.json",
    vocab_file: str = "vocab.json",
    blank_id: int = DEFAULT_BLANK_ID,
):
    return f"""
name: "multilingual_ensemble"
platform: "ensemble"
max_batch_size: 64

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
input {{
  name: "LANGUAGE"
  data_type: TYPE_STRING
  dims: [ 1 ]
}}

output {{
  name: "TRANSCRIPTS"
  data_type: TYPE_STRING
  dims: [ 1 ]
}}

parameters: {{ key: "language_masks_file" value: {{ string_value: \"{language_mask_file}\" }} }}
parameters: {{ key: "vocab_file" value: {{ string_value: \"{vocab_file}\" }} }}
parameters: {{ key: "blank_id" value: {{ string_value: \"{blank_id}\" }} }}

ensemble_scheduling {{
  step [
    {{
      model_name: "feature-extractor"
      model_version: -1
      input_map {{
        key: "input__0"
        value: "AUDIO_SIGNAL"
      }}
      input_map {{
        key: "input__1"
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
      model_name: "multilingual_decoder"
      model_version: -1
      input_map {{
        key: "LOG_PROBS"
        value: "logits"
      }}
      input_map {{
        key: "LANGUAGE"
        value: "LANGUAGE"
      }}
      output_map {{
        key: "TRANSCRIPT"
        value: "TRANSCRIPTS"
      }}
    }}
  ]
}}
""".lstrip()


def multilingual_decode(
    logits,
    language: str,
    language_masks: Dict[str, List[int]],
    vocab: Dict[str, List[str]],
    blank_id: int = DEFAULT_BLANK_ID,
):
    """Decode CTC logits for a single language without an LM.

    This mirrors the reference logic in the published ONNX wrapper:
      - select the language-specific slice of the logits
      - argmax over classes
      - collapse repeated tokens
      - remove blank
      - map token ids back through that language's vocabulary
    """
    if language not in language_masks:
        raise KeyError(f"Language '{language}' not found in language_masks")
    if language not in vocab:
        raise KeyError(f"Language '{language}' not found in vocab")

    mask = language_masks[language]
    locale_vocab = vocab[language]
    if isinstance(logits, list):
        logits = __import__("numpy").asarray(logits)

    selected = logits[:, :, mask]
    if selected.shape[-1] == 0:
        return ""

    np = __import__("numpy")
    idx = np.argmax(selected, axis=-1)
    reduced = []
    prev = None
    for token_id in idx[0].tolist():
        if prev != token_id:
            reduced.append(int(token_id))
        prev = token_id
    text = "".join(locale_vocab[tok] for tok in reduced if tok != blank_id)
    return text.replace("▁", " ").strip()


def decode_from_files(logits, language: str, mask_path: str, vocab_path: str, blank_id: int = DEFAULT_BLANK_ID):
    masks = load_json(mask_path)
    vocab = load_json(vocab_path)
    return multilingual_decode(logits, language, masks, vocab, blank_id=blank_id)
