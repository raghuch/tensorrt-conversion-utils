#!/usr/bin/env python3
"""Reference transcripts straight from the .nemo model, for WER comparison.

Modular replacement for the NeMo half of 05_triton_client_ensemble.ipynb
(cell 3). The Triton client can diff its predictions against the JSON this
produces, which is the only way to tell whether the whole
nemo -> onnx -> TensorRT chain changed any words.

    # greedy NeMo decode, the baseline the engine must match
    python3 nemo_reference.py --model Conformer-CTC-BPE-hi-nasim.nemo \\
        --manifest sample_test_manifest.json \\
        --data-dir /tts_data/code/nemo_models/ \\
        --output nemo_reference.json

    # with an n-gram LM beam search
    python3 nemo_reference.py --model m.nemo --manifest m.json \\
        --kenlm-model lm/4gram.bin --output nemo_reference.json

    # then compare
    python3 triton_client.py --manifest m.json --data-dir wavs/ \\
        --reference nemo_reference.json

Note on dither: like nemo_fe2ts.py, this forces the featurizer's dither to 0.
The checkpoint stores 1e-05, which is training-time noise; leaving it on makes
the reference itself non-reproducible and would show up as phantom WER.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from deps import require


def print_with_datetime(msg):
    print(f"{datetime.now()} {msg}")


def read_manifest(manifest, data_dir, limit):
    entries = []
    with open(manifest, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  skipping {manifest}:{line_no}: {e}", file=sys.stderr)
                continue
            relative = record.get("audio_filepath")
            if not relative:
                continue
            path = relative if os.path.isabs(relative) else os.path.join(
                data_dir, relative
            )
            entries.append((path, record.get("text")))
            if limit and len(entries) >= limit:
                break
    return entries


def softmax(x):
    import numpy as np

    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])


def beam_search_with_lm(model, logits, vocabulary, lm_path, beam_width,
                        alpha, beta):
    """NeMo's BeamSearchDecoderWithLM over a pyctcdecode n-gram LM.

    The decoder wants one character per token id, so ids are mapped through
    chr() and the resulting characters are mapped back with the tokenizer.
    This is the same trick 05 used, kept because NeMo's LM scorer is built for
    that interface.
    """
    nemo_asr = require("nemo.collections.asr", "for beam search")
    import numpy as np

    offset = 0
    chars = [chr(i + offset) for i in range(len(vocabulary))]
    beam_searcher = nemo_asr.modules.BeamSearchDecoderWithLM(
        vocab=chars,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        lm_path=lm_path,
        num_cpus=max(os.cpu_count() or 1, 1),
        input_tensor=False,
    )
    ids_to_text = model.tokenizer.ids_to_text
    texts = []
    for item in logits:
        probs = softmax(item)
        beams = beam_searcher.forward(
            log_probs=probs.reshape(1, probs.shape[0], probs.shape[1]),
            log_probs_length=None,
        )
        candidate = beams[0][0][1]
        if ids_to_text is not None:
            texts.append(ids_to_text([ord(c) - offset for c in candidate]))
        else:
            texts.append(candidate)
    return texts


def add_arguments(parser):
    parser.add_argument("--model", required=True, help="Path to the .nemo model")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="nemo_reference.json")
    parser.add_argument(
        "--kenlm-model", default="",
        help="Optional n-gram LM (from pyctcdecoder). Omit for greedy decode.",
    )
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--cuda-visible-devices", default=None)


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

    # Imported after the environment is set.
    torch = require("torch", "to run NeMo inference")
    nemo_asr = require("nemo.collections.asr", "to run NeMo inference")

    if not os.path.isfile(args.model):
        raise SystemExit(f"No such model: {args.model}")
    entries = read_manifest(args.manifest, args.data_dir, args.limit)
    if not entries:
        raise SystemExit(f"no usable entries in {args.manifest}")
    paths = [p for p, _ in entries]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_with_datetime(f"Restoring {args.model} on {device}")
    model = nemo_asr.models.ASRModel.restore_from(restore_path=args.model)
    model = model.to(device)
    model.eval()
    # See the module docstring: the checkpoint's 1e-05 dither is training noise.
    model.preprocessor.featurizer.dither = 0

    print_with_datetime(f"Transcribing {len(paths)} utterance(s)")
    started = time.perf_counter()
    with torch.no_grad():
        logits = list(model.transcribe(paths))
    print_with_datetime(f"forward pass done in {time.perf_counter() - started:.1f}s")

    if args.kenlm_model:
        print_with_datetime(f"Beam search with LM {args.kenlm_model}")
        texts = beam_search_with_lm(
            model, logits, model.decoder.vocabulary, args.kenlm_model,
            args.beam_width, args.alpha, args.beta,
        )
    else:
        texts = [str(item) for item in logits]

    records = [
        {"audio_filepath": path, "text": text}
        for (path, _), text in zip(entries, texts)
    ]
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print_with_datetime(f"Wrote {args.output} ({len(records)} utterances)")

    for record in records[:5]:
        print(f"  {os.path.basename(record['audio_filepath']):40s} {record['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
