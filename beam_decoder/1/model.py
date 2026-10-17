# Triton python_backend model: CTC log-probs -> transcript string.
#
# Replaces the topk + beam-decoder pair. It takes the raw 129-channel log-probs
# straight from the conformer_trt engine and decodes them here, so there is no
# separate topk TorchScript model to build (see 03_decoder2ts.ipynb).
#
# Channel layout, taken from the checkpoint's model_config.yaml:
#   decoder.num_classes : 128          (the vocabulary)
#   add_blank           : True         (NeMo ConvASRDecoder appends 1)
#   -> 129 output channels, blank_id = 128, i.e. the LAST channel.
#
# That blank position is why the beam path below rotates the channel axis:
# pyctcdecode assumes labels[0] is the blank and offers no blank_id argument,
# so the blank is moved to the front before handing the log-probs over.
#
# Decoding is greedy by default because it needs no extra dependency and is
# deterministic. Set decoder: "beam" in config.pbtxt for pyctcdecode beam
# search (optionally with a KenLM model).
#
# SentencePiece detokenisation: pieces are concatenated with no separator and
# the U+2581 LOWER ONE EIGHTH BLOCK mark becomes a space. Joining with spaces
# instead would split every sub-word piece, which is wrong for Devanagari.

import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils

SPM_WORD_MARK = "▁"


def _out_frames(frames_in):
    """Output frames for `frames_in` input frames.

    The conformer_trt ONNX declares its output as
    floor(floor(T / 2 - 1/2) / 2) + 1 because the encoder subsamples twice.
    NUM_TIME_STEPS from the feature extractor is a count of *input* frames, so
    it has to be converted before it can be used to trim the output.
    """
    return int(np.floor(np.floor(frames_in / 2 - 0.5) / 2) + 1)


def _detokenize(pieces):
    """SentencePiece pieces -> text: concatenate, then turn U+2581 into space."""
    return "".join(pieces).replace(SPM_WORD_MARK, " ").strip()


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"]).get("parameters", {})

        def text(key, default=None):
            entry = config.get(key)
            return entry["string_value"] if entry else default

        self.decoder_kind = text("decoder", "greedy")
        self.lm_path = text("kenlm_model_path", "")
        self.alpha = float(text("alpha", "0.5"))
        self.beta = float(text("beta", "1.5"))

        vocab_file = text("vocab_file", "vocab.txt")
        path = self._resolve(args, vocab_file)
        with open(path, "r", encoding="utf-8") as handle:
            self.labels = handle.read().splitlines()
        if not self.labels:
            raise ValueError(f"{path} is empty")

        # NeMo appends the blank as the final channel.
        self.blank_id = int(text("blank_id", str(len(self.labels))))
        self.num_channels = len(self.labels) + 1

        self.beam = None
        if self.decoder_kind == "beam":
            try:
                from pyctcdecode import build_ctcdecoder
            except ImportError as e:
                raise RuntimeError(
                    "decoder: \"beam\" needs pyctcdecode, which this model "
                    "repository does not ship. Either `pip install "
                    "pyctcdecode` in the server image, or set "
                    "parameters { key: \"decoder\" value: "
                    "{ string_value: \"greedy\" } } in config.pbtxt, which "
                    "needs no extra package. Original error: " + str(e)
                ) from None

            # pyctcdecode appends its own blank to the end of the vocabulary
            # when one is not present, which is the same convention NeMo's
            # ConvASRDecoder uses (num_classes + 1, blank last). So the 128
            # tokens are passed unchanged and no channel permutation is needed:
            # both sides put the blank at index 128. It also leaves the
            # SentencePiece U+2581 marks alone, so _detokenize still applies.
            self.beam = build_ctcdecoder(
                self.labels,
                kenlm_model_path=self.lm_path or None,
                alpha=self.alpha,
                beta=self.beta,
            )
            # Fail at load time rather than on the first request if the
            # vocabulary and the engine's channel count disagree.
            probe = np.zeros((1, self.num_channels), dtype=np.float32)
            self.beam.decode(probe)
        elif self.decoder_kind != "greedy":
            raise ValueError(f"unknown decoder {self.decoder_kind!r}")

    @staticmethod
    def _resolve(args, filename):
        """Locate a file shipped with the model.

        The python backend's `model_repository` is the model's own directory
        (e.g. /models/beam-decoder) in some Triton releases and the repository
        root in others, so both layouts are tried instead of guessing.
        """
        candidates = [
            os.path.join(args["model_repository"], filename),
            os.path.join(
                args["model_repository"],
                args.get("model_version", "1"),
                filename,
            ),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"{filename} not found; looked in " + " and ".join(candidates)
        )

    def _greedy(self, log_probs, frames):
        best = log_probs[:frames].argmax(axis=1)
        pieces = []
        previous = -1
        for index in best:
            index = int(index)
            if index != previous and index != self.blank_id:
                pieces.append(self.labels[index])
            previous = index
        return _detokenize(pieces)

    def _decode(self, log_probs, frames):
        if frames <= 0:
            return ""
        if frames < len(log_probs):
            log_probs = log_probs[:frames]
        if self.decoder_kind == "greedy":
            return self._greedy(log_probs, frames)
        return _detokenize(self.beam.decode(log_probs))

    def execute(self, requests):
        responses = []
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "LOG_PROBS")
            log_probs = np.asarray(tensor.as_numpy(), dtype=np.float32)

            if log_probs.ndim != 3:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(
                        "LOG_PROBS must have shape [batch, time, channels], "
                        f"got {log_probs.shape}."
                    ),
                ))
                continue

            if log_probs.shape[2] != self.num_channels:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(
                        f"LOG_PROBS has {log_probs.shape[2]} channels, expected "
                        f"{self.num_channels} ({len(self.labels)} tokens + 1 blank). "
                        "vocab.txt and the conformer_trt engine disagree."
                    ),
                ))
                continue

            batch_size, time_steps, _ = log_probs.shape
            frames = [time_steps] * batch_size
            length_tensor = pb_utils.get_input_tensor_by_name(
                request, "NUM_TIME_STEPS"
            )
            if length_tensor is not None:
                lengths = np.asarray(length_tensor.as_numpy()).reshape(-1)
                if lengths.size != batch_size:
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "NUM_TIME_STEPS must contain one length per sample, "
                            f"got {lengths.size} for batch size {batch_size}."
                        ),
                    ))
                    continue
                frames = [min(time_steps, _out_frames(int(length)))
                          for length in lengths]

            transcripts = [self._decode(sample, count)
                           for sample, count in zip(log_probs, frames)]
            out = pb_utils.Tensor(
                "TRANSCRIPT", np.asarray(transcripts, dtype=object).reshape(-1, 1)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        self.beam = None
