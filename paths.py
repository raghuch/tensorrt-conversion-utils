"""Where build artifacts go, shared by nemo2onnx2trt.py and nemo_fe2ts.py.

Both scripts name artifact directories after the checkpoint, so the rule lives
here once rather than being copied into each entry point.
"""

import os
import re

SUBDIRS = ("conformer_onnx", "conformer_onnx_folded", "conformer_trt")


def model_name(nemo_path):
    """Conformer-CTC-BPE.nemo -> conformer-ctc-bpe

    Used as the default artifact directory name so two checkpoints never
    collide.
    """
    stem = os.path.splitext(os.path.basename(nemo_path))[0]
    return re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower() or "model"


def default_artifacts_dir(nemo_path, artifacts_dir=None):
    """The base directory: --artifacts-dir if given, else named after the model."""
    return artifacts_dir or f"./models_latest_{model_name(nemo_path)}"


def default_paths(nemo_path, artifacts_dir=None):
    """The ONNX, folded ONNX and plan locations under the base directory.

        <base>/conformer_onnx/model.onnx
        <base>/conformer_onnx_folded/model.onnx
        <base>/conformer_trt/model<gpu tag>.plan
    """
    root = default_artifacts_dir(nemo_path, artifacts_dir)
    return (
        f"{root}/conformer_onnx/model.onnx",
        f"{root}/conformer_onnx_folded/model.onnx",
        f"{root}/conformer_trt/model.plan",
    )
