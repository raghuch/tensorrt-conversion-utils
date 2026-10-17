"""Import a heavy dependency, or say which requirements file provides it.

The scripts import torch, NeMo, TensorRT, tritonclient and PyYAML lazily so
that `--help` works on a machine with none of them. The flip side is that a
missing package otherwise surfaces as a bare ModuleNotFoundError traceback,
which does not tell a first-time user which file to install. This keeps that
message actionable.
"""

import importlib

# module -> the file that provides it
PROVIDERS = {
    "torch": "requirements-convert-v100.txt / -convert-tensorrt10.txt",
    "nemo": "requirements-convert-v100.txt / -convert-tensorrt10.txt",
    "nemo.collections.asr": "requirements-convert-v100.txt / -convert-tensorrt10.txt",
    "tensorrt": "requirements-convert-v100.txt / -convert-tensorrt10.txt",
    "polygraphy": "requirements-convert-v100.txt / -convert-tensorrt10.txt",
    "onnxruntime": "requirements-convert-v100.txt / -convert-tensorrt10.txt",
    "yaml": "requirements-convert-v100.txt / -convert-tensorrt10.txt",
    "soundfile": "requirements.txt",
    "tritonclient": "requirements.txt",
    "tritonclient.grpc": "requirements.txt",
    "tritonclient.http": "requirements.txt",
    "librosa": "nemo_toolkit[asr]",
    "kenlm": "nemo_toolkit[asr]",
    "pyctcdecode": "pip install pyctcdecode",
    "riva_asrlib.decoder": "pip install riva-asrlib-decoder",
}


def require(module_name, why=""):
    """Return the imported module, or exit with an actionable message."""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        provider = PROVIDERS.get(module_name, PROVIDERS.get(
            module_name.split(".")[0], "the requirements files"))
        first = provider.split(" / ")[0]
        container = (
            "Dockerfile.trt10" if first == "requirements.txt"
            else "Dockerfile.v100"
        )
        need = f" It is needed {why}." if why else ""
        raise SystemExit(
            f"error: '{module_name}' is not installed.{need}\n"
            f"  provided by: {provider}\n"
            f"  e.g. pip install -r pt2onnx2trt/{first}\n"
            f"  or use the container image built from pt2onnx2trt/{container}\n"
            f"  (original error: {e})"
        ) from None
