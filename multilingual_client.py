#!/usr/bin/env python3
"""Send audio and a language parameter to the multilingual Triton ensemble."""

import argparse
import os
import sys

import numpy as np

DEFAULT_MODEL = "multilingual_ensemble"
DEFAULT_URL = "localhost:8001"
EXPECTED_SAMPLE_RATE = 16000


def load_audio(path, target_rate=EXPECTED_SAMPLE_RATE):
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != target_rate:
        raise ValueError(f"{path}: expected {target_rate} Hz, got {rate} Hz")
    return np.ascontiguousarray(audio, dtype=np.float32)


def pad_batch(samples):
    lengths = np.asarray([len(s) for s in samples], dtype=np.int32)
    width = int(lengths.max())
    batch = np.zeros((len(samples), width), dtype=np.float32)
    for row, sample in enumerate(samples):
        batch[row, : len(sample)] = sample
    return batch, lengths.reshape(-1, 1)


def make_inference_client(url, protocol="grpc"):
    import tritonclient.grpc as grpc_client

    if protocol != "grpc":
        raise NotImplementedError("Only gRPC is implemented in this helper; use HTTP if you extend it.")
    return grpc_client.InferenceServerClient(url=url, verbose=False)


def infer(client, model_name, audio_path, language, protocol="grpc"):
    audio = load_audio(audio_path)
    batch, lengths = pad_batch([audio])
    input_audio = client.get_inference_input(name="AUDIO_SIGNAL") if False else None
    transport = __import__("tritonclient.grpc")
    audio_input = transport.grpc.InferInput("AUDIO_SIGNAL", list(batch.shape), "FP32")
    audio_input.set_data_from_numpy(batch)

    length_input = transport.grpc.InferInput("NUM_SAMPLES", list(lengths.shape), "INT32")
    length_input.set_data_from_numpy(lengths)

    lang_input = transport.grpc.InferInput("LANGUAGE", [1], "BYTES")
    lang_input.set_data_from_numpy(np.asarray([language], dtype=object))

    response = client.infer(
        model_name,
        model_version="1",
        inputs=[audio_input, length_input, lang_input],
        outputs=[transport.grpc.InferRequestedOutput("TRANSCRIPTS")],
    )
    raw = response.as_numpy("TRANSCRIPTS")
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw.reshape(-1)]


def add_arguments(parser):
    parser.add_argument("--audio", required=True, help="Input .wav file")
    parser.add_argument("--language", required=True, help="Language code, e.g. hi, en, mr, bn")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Triton model name")
    parser.add_argument("--url", default=DEFAULT_URL, help="Triton host:port")
    parser.add_argument("--protocol", default="grpc", choices=("grpc",))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()

    client = make_inference_client(args.url, args.protocol)
    if not client.is_server_live():
        raise SystemExit(f"No Triton server is live at {args.url}")
    results = infer(client, args.model, args.audio, args.language, args.protocol)
    print(results[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
