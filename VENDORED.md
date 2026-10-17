# Vendored helper modules

These three files are **copies**, kept here so `pt2onnx2trt/` runs standalone
with no dependency on the rest of the repository.

| file | origin | used by | what it does |
|---|---|---|---|
| `converter.py` | NVIDIA's Conformer/TensorRT conversion script (upstream of this repo) | `nemo2onnx2trt.py` | `network_from_onnx_path`, `CreateConfig`/`Profile` with the batch/length optimization profiles, and the FP32-layer fixes that keep Conformer accurate in FP16 |
| `features.py` | a copy of NeMo's `FilterbankFeatures`, made TorchScript-friendly | `nemo_fe2ts.py` | log-mel filterbank used to trace the feature extractor |
| `metrics.py` | NVIDIA's WER helper (upstream of this repo) | `triton_client.py` | `word_error_rate` for scoring transcripts |

Identical copies also live in the repository root, because the original
notebooks still import them from there — `02_nemofe2ts.ipynb` does
`from features import FilterbankFeatures`, and the `05_*` notebooks use
`metrics.py`.

**That means there are two copies, and they can drift.** If you change one,
resync the other:

```bash
for f in converter.py features.py metrics.py; do
  cp "$f" "pt2onnx2trt/$f"
done
```

The copies here are byte-identical to the root ones as packaged. Nothing in
this package edits them.

## Not vendored, and why

- `speech_utils.py` and `conformer-client.py` belong to the old client flow
  that `triton_client.py` replaces. `speech_utils.py:159` also hardcodes a
  128-token vocabulary in the wrong order (index 2 is `▁ह,` there but `▁स` in
  the checkpoint), so copying it in would spread that bug. They are left
  untouched in the repository root.
