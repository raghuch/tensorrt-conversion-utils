#!/usr/bin/env python3
"""Send audio to the Triton ensemble and print transcripts.

Modular replacement for the Triton half of 05_triton_client_ensemble.ipynb
(cells 11 and 15-17), for the pyctcdecode flow where the server does the
decoding. The client no longer needs a vocabulary, a tokenizer or pyctcdecode
-- it sends raw audio and gets text back.

    # one file
    python3 triton_client.py --audio sounds.wav

    # a manifest, in batches, with WER against its text field
    python3 triton_client.py --manifest sample_test_manifest.json \\
        --data-dir /tts_data/code/nemo_models/ --batch-size 32

    # compare against NeMo reference transcripts
    python3 triton_client.py --manifest m.json --data-dir wavs/ \\
        --reference nemo_reference.json

    # HTTP instead of gRPC
    python3 triton_client.py --audio a.wav --protocol http --url localhost:8000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from deps import require

# metrics.py (used for WER) is vendored beside this script, so it resolves from
# sys.path[0] without any path juggling. See VENDORED.md.

DEFAULT_MODEL = "conformer_trt_ensemble"
DEFAULT_URL = "localhost:8001"
EXPECTED_SAMPLE_RATE = 16000


def print_with_datetime(msg):
    print(f"{datetime.now()} {msg}")


def read_manifest(manifest, data_dir, limit, text_field):
    """Return [(absolute audio path, reference text or None)]."""
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
                print(f"  skipping {manifest}:{line_no}: no audio_filepath",
                      file=sys.stderr)
                continue
            path = relative if os.path.isabs(relative) else os.path.join(
                data_dir, relative
            )
            entries.append((path, record.get(text_field)))
            if limit and len(entries) >= limit:
                break
    return entries


def pad_batch(samples):
    """Pad a list of 1-D float arrays to (batch, max_len) plus true lengths.

    Matches what the ensemble expects: AUDIO_SIGNAL is a padded 2-D batch and
    NUM_SAMPLES carries each row's real length so the feature extractor can
    mask the padding.
    """
    import numpy as np

    lengths = np.asarray([len(s) for s in samples], dtype=np.int32)
    width = int(lengths.max())
    batch = np.zeros((len(samples), width), dtype=np.float32)
    for row, sample in enumerate(samples):
        batch[row, : len(sample)] = sample
    return batch, lengths.reshape(-1, 1)


def load_audio(path, target_rate, warn=True):
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != target_rate:
        if warn:
            print(f"  {os.path.basename(path)}: {rate} Hz, expected "
                  f"{target_rate} Hz -- resampling is not done here",
                  file=sys.stderr)
        raise ValueError(
            f"{path} is {rate} Hz but the model expects {target_rate} Hz"
        )
    return np.ascontiguousarray(audio, dtype=np.float32)


def make_client(protocol, url, verbose):
    client = require(f"tritonclient.{protocol}", "to talk to Triton")
    return client.InferenceServerClient(url=url, verbose=verbose)


def infer_batch(client, model, batch, lengths, protocol):
    transport = require(f"tritonclient.{protocol}", "to send inference requests")

    audio = transport.InferInput("AUDIO_SIGNAL", list(batch.shape), "FP32")
    audio.set_data_from_numpy(batch)
    samples = transport.InferInput("NUM_SAMPLES", list(lengths.shape), "INT32")
    samples.set_data_from_numpy(lengths)
    output = transport.InferRequestedOutput("TRANSCRIPTS")
    response = client.infer(
        model, model_version="1", inputs=[audio, samples], outputs=[output]
    )
    raw = response.as_numpy("TRANSCRIPTS")
    if raw is None:
        raise RuntimeError(
            "the server returned no TRANSCRIPTS. Check that the ensemble's "
            "output is named TRANSCRIPTS and that beam-decoder loaded."
        )
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in raw.reshape(-1)]


def wait_for_server(client, protocol, model, attempts=30, delay=2.0):
    """Poll until the model reports ready.

    Uses the client's own is_model_ready; there is no module-level
    model_ready in either transport.
    """
    for attempt in range(1, attempts + 1):
        try:
            if client.is_model_ready(model):
                return True
        except Exception:
            pass
        if attempt == 1 or attempt % 5 == 0:
            print_with_datetime(f"waiting for {model} ({attempt}/{attempts})")
        time.sleep(delay)
    return False


def word_error_rate(hypotheses, references):
    """WER via the repo's metrics.py, or None if it is unavailable."""
    try:
        metrics = require("metrics", "to score transcripts")
    except SystemExit:
        return None
    wer, errors, words = metrics.word_error_rate(hypotheses, references)
    return wer, errors, words


def add_arguments(parser):
    parser.add_argument("--url", default=DEFAULT_URL, help="host:port")
    parser.add_argument("--protocol", default="grpc", choices=("grpc", "http"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio", action="append", default=[],
                        help="wav file; repeatable")
    parser.add_argument("--manifest", help="JSONL manifest with audio_filepath")
    parser.add_argument("--data-dir", default=".",
                        help="prefix for relative paths in the manifest")
    parser.add_argument("--text-field", default="text",
                        help="manifest field holding the reference transcript")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="only send the first N files (0 = all)")
    parser.add_argument("--sample-rate", type=int, default=EXPECTED_SAMPLE_RATE)
    parser.add_argument("--reference",
                        help="JSON from nemo_reference.py, for WER")
    parser.add_argument("--output", help="write predictions to this JSON file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-models", action="store_true",
                        help="ask the server what it has loaded, then exit")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()

    client = make_client(args.protocol, args.url, args.verbose)
    if not client.is_server_live():
        raise SystemExit(f"no Triton server at {args.url} ({args.protocol})")
    if args.list_models:
        # get_model_repository_index() returns the raw protobuf response; the
        # repeated `models` field is what carries the names.
        response = client.get_model_repository_index()
        models = getattr(response, "models", response)
        try:
            entries = list(models)
        except TypeError:
            entries = []
        for entry in entries:
            versions = entry.version or "-"
            print(f"{entry.name:28s} {versions}")
        return 0
    if not wait_for_server(client, args.protocol, args.model):
        raise SystemExit(f"{args.model} never became ready on {args.url}")

    entries = []
    for path in args.audio:
        entries.append((path, None))
    if args.manifest:
        entries += read_manifest(
            args.manifest, args.data_dir, args.limit, args.text_field
        )
    if args.limit and len(entries) > args.limit:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit("nothing to send: pass --audio and/or --manifest")

    predictions = []
    started = time.perf_counter()
    for offset in range(0, len(entries), args.batch_size):
        chunk = entries[offset : offset + args.batch_size]
        samples, names, failed = [], [], []
        for path, _ in chunk:
            try:
                samples.append(load_audio(path, args.sample_rate))
                names.append(path)
            except Exception as e:
                failed.append((path, str(e)))
        if not samples:
            for path, why in failed:
                print(f"  SKIP {os.path.basename(path)}: {why}", file=sys.stderr)
            continue
        batch, lengths = pad_batch(samples)
        transcripts = infer_batch(
            client, args.model, batch, lengths, args.protocol
        )
        for name, text in zip(names, transcripts):
            predictions.append({"audio_filepath": name, "text": text})
            print(f"  {os.path.basename(name):40s} {text}")
        for path, why in failed:
            print(f"  SKIP {os.path.basename(path)}: {why}", file=sys.stderr)

    elapsed = time.perf_counter() - started
    audio_seconds = 0.0
    for item in predictions:
        try:
            audio_seconds += os.path.getsize(item["audio_filepath"]) / (
                2 * args.sample_rate
            )
        except OSError:
            pass
    print_with_datetime(
        f"{len(predictions)} utterances in {elapsed:.1f}s"
        + (f" ({len(predictions) / max(elapsed, 1e-9):.1f} utt/s)" if elapsed else "")
    )

    references = [ref for _, ref in entries if ref]
    if references and len(references) == len(predictions):
        result = word_error_rate([p["text"] for p in predictions], references)
        if result is None:
            print("  (metrics.py not importable, skipping WER)")
        else:
            wer, errors, words = result
            print_with_datetime(
                f"WER {wer:.4f}  ({errors} errors / {words} words)"
            )
    elif references:
        print(f"  {len(predictions)} predictions vs {len(references)} references "
              f"with {len(entries) - len(references)} untagged files -- skipping WER")

    if args.reference:
        with open(args.reference, "r", encoding="utf-8") as handle:
            reference = json.load(handle)
        by_path = {os.path.basename(r["audio_filepath"]): r["text"] for r in reference}
        paired = [
            (p["text"], by_path[os.path.basename(p["audio_filepath"])])
            for p in predictions
            if os.path.basename(p["audio_filepath"]) in by_path
        ]
        if paired:
            result = word_error_rate([h for h, _ in paired], [r for _, r in paired])
            if result is not None:
                wer, errors, words = result
                print_with_datetime(
                    f"WER vs NeMo reference {wer:.4f} "
                    f"({errors}/{words} words, {len(paired)} utterances)"
                )
        else:
            print("  no overlap between predictions and the reference file")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle, ensure_ascii=False, indent=2)
        print_with_datetime(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
