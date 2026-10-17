#!/usr/bin/env python3
import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_dir = args.get("model_repository")
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
            token_id = int(token_id)
            if prev != token_id:
                collapsed.append(token_id)
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
            output = pb_utils.Tensor("TRANSCRIPT", np.asarray([transcript], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
        return responses

    def finalize(self):
        pass
