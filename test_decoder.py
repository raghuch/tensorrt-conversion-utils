import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent


class Tensor:
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def as_numpy(self):
        return self.values


BACKEND = SimpleNamespace(
    Tensor=Tensor,
    TritonError=ValueError,
    InferenceResponse=lambda output_tensors, error=None: SimpleNamespace(
        output_tensors=output_tensors, error=error
    ),
    get_input_tensor_by_name=lambda request, name: request.get(name),
)


def load_script(name):
    with patch.dict(sys.modules, {"triton_python_backend_utils": BACKEND}), \
            patch.object(sys, "path", [str(PACKAGE_DIR)] + sys.path):
        return runpy.run_path(str(PACKAGE_DIR / name))


def logits(tokens):
    values = np.full((len(tokens), 3), -10.0, dtype=np.float32)
    values[np.arange(len(tokens)), tokens] = 0.0
    return values


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "vocab.txt").write_text(
            "\u2581a\nb\n", encoding="utf-8"
        )
        self.model_class = load_script("beam_decoder/1/model.py")["TritonPythonModel"]

    def model(self, parameters=None):
        model = self.model_class()
        config = {"name": "beam-decoder"}
        if parameters is not None:
            config["parameters"] = {
                key: {"string_value": str(value)}
                for key, value in parameters.items()
            }
        model.initialize({
            "model_config": json.dumps(config),
            "model_repository": self.directory.name,
            "model_version": "1",
        })
        return model

    def request(self, values, lengths=None):
        request = {"LOG_PROBS": Tensor("LOG_PROBS", values)}
        if lengths is not None:
            request["NUM_TIME_STEPS"] = Tensor(
                "NUM_TIME_STEPS", np.asarray(lengths, dtype=np.int32)
            )
        return request

    def test_standard_config_and_single_sample(self):
        model = self.model({"decoder": "greedy", "blank_id": 2})
        response = model.execute([
            self.request(logits([0, 0, 2, 0, 1])[None])
        ])[0]
        self.assertIsNone(response.error)
        output = response.output_tensors[0]
        self.assertEqual(output.name, "TRANSCRIPT")
        self.assertEqual(output.as_numpy().shape, (1, 1))
        self.assertEqual(output.as_numpy().tolist(), [["a ab"]])
        self.assertEqual(self.model().decoder_kind, "greedy")

    def test_ragged_batch_uses_each_length(self):
        values = np.stack([logits([0, 1, 2, 1]), logits([0, 2, 0, 1])])
        response = self.model().execute([
            self.request(values, [[5], [9]])
        ])[0]
        self.assertIsNone(response.error)
        self.assertEqual(response.output_tensors[0].as_numpy().tolist(),
                         [["ab"], ["a a"]])

    def test_batch_without_lengths_and_multiple_requests(self):
        values = np.stack([logits([0, 2, 1]), logits([1, 1, 2])])
        responses = self.model().execute([
            self.request(values), self.request(values[:1])
        ])
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].output_tensors[0].as_numpy().tolist(),
                         [["ab"], ["b"]])
        self.assertEqual(responses[1].output_tensors[0].as_numpy().shape, (1, 1))

    def test_invalid_requests_do_not_drop_other_responses(self):
        values = logits([0, 1])[None]
        responses = self.model().execute([
            self.request(values[0]),
            self.request(values[:, :, :2]),
            self.request(np.repeat(values, 2, axis=0), [[5]]),
            self.request(values),
        ])
        self.assertEqual(len(responses), 4)
        for response in responses[:3]:
            self.assertIsNotNone(response.error)
        self.assertIsNone(responses[3].error)

    def test_beam_parameters_and_per_sample_trimming(self):
        for lm_path in ("", "/models/lm.bin"):
            with self.subTest(lm_path=lm_path):
                calls = []
                options = {}

                def decode(values):
                    calls.append(values.shape)
                    return "beam text"

                def build(labels, **kwargs):
                    options.update(kwargs)
                    self.assertEqual(labels, ["\u2581a", "b"])
                    return SimpleNamespace(decode=decode)

                with patch.dict(sys.modules, {
                    "pyctcdecode": SimpleNamespace(build_ctcdecoder=build)
                }):
                    model = self.model({
                        "decoder": "beam", "kenlm_model_path": lm_path,
                        "alpha": 0.7, "beta": 1.2,
                    })
                values = np.stack([logits([0, 1, 2, 1])] * 2)
                response = model.execute([self.request(values, [[5], [9]])])[0]
                self.assertEqual(calls, [(1, 3), (2, 3), (3, 3)])
                self.assertEqual(options, {
                    "kenlm_model_path": lm_path or None, "alpha": 0.7, "beta": 1.2
                })
                self.assertEqual(response.output_tensors[0].as_numpy().tolist(),
                                 [["beam text"], ["beam text"]])

    def test_generated_batch_contract_and_client_output(self):
        setup = load_script("setup_model_repository.py")
        parser = argparse.ArgumentParser()
        setup["add_arguments"](parser)
        args = parser.parse_args(["--max-batch", "8"])
        args.num_classes = 3
        configs = setup["configs_for"](args)
        self.assertIn("max_batch_size: 8", configs["beam-decoder"])
        self.assertIn("dims: [ -1, 3 ]", configs["beam-decoder"])
        self.assertRegex(configs["conformer_trt_ensemble"],
                 r'name: "TRANSCRIPTS"\s+data_type: TYPE_STRING\s+dims: \[ 1 \]')
        client_script = load_script("triton_client.py")
        transport = SimpleNamespace(
            InferInput=lambda *args: SimpleNamespace(set_data_from_numpy=lambda values: None),
            InferRequestedOutput=lambda name: name,
        )
        client = SimpleNamespace(infer=lambda *args, **kwargs: SimpleNamespace(
            as_numpy=lambda name: np.asarray([[b"one"], [b"two"]], dtype=object)
        ))
        with patch.dict(client_script["infer_batch"].__globals__, {
            "require": lambda *args: transport
        }):
            transcripts = client_script["infer_batch"](
                client, "conformer_trt_ensemble", np.zeros((2, 100), np.float32),
                np.full((2, 1), 100, np.int32), "grpc"
            )
        self.assertEqual(transcripts, ["one", "two"])


if __name__ == "__main__":
    unittest.main()