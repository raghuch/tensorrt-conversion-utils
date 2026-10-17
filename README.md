# pt2onnx2trt

NeMo Conformer-CTC → ONNX → TensorRT → Triton, packaged as scripts.

This replaces the notebook cell-by-cell flow of `01_nemo2onnx2trt_devanagari.ipynb`,
`02_nemofe2ts.ipynb`, `04_triton_server_setup.ipynb` and `05_triton_client_ensemble.ipynb`.
The `03_decoder2ts.ipynb` topk model is intentionally **not** reproduced: decoding
happens server-side in `beam-decoder/1/model.py` with pyctcdecode instead.

## Layout

```
pt2onnx2trt/
  nemo2onnx2trt.py            .nemo -> ONNX -> folded ONNX -> TensorRT engines
  nemo_fe2ts.py               feature extractor -> TorchScript
  setup_model_repository.py   writes the Triton config.pbtxt files
  triton_client.py            sends audio to Triton, prints transcripts, WER
  nemo_reference.py           reference transcripts from NeMo, for WER
  deps.py                     actionable "you are missing package X" errors
  paths.py                    where the ONNX / plan / featurizer go
  beam_decoder/1/model.py     Triton python_backend CTC decoder (pyctcdecode)
  converter.py  features.py  metrics.py      vendored, see VENDORED.md
  requirements.txt                       client-side environment
  requirements-convert-v100.txt          V100   (TensorRT 8.6.3)
  requirements-convert-tensorrt10.txt    A100/L40/L40S (TensorRT 10.8.0.43)
  Dockerfile.v100                        build + serve, V100
  Dockerfile.trt10                       build + serve, A100 / L40S
```

`converter.py`, `features.py` and `metrics.py` are **vendored copies** in this
directory, so the package runs standalone with nothing else on `PYTHONPATH`.
Identical copies stay in the repository root because the original notebooks
import them from there — see `VENDORED.md`, which also explains how to resync
them if you change one.

## GPU support, in one table

| GPU | SM | TensorRT | Triton container | Works because |
|---|---|---|---|---|
| V100 | 7.0 | 8.6.3 | `24.04` | last TensorRT/Triton that runs Volta |
| A100 | 8.0 | 10.8.0.43 | `25.01` / `25.02` | same TensorRT build as Triton ships |
| L40 / L40S | 8.9 | 10.8.0.43 | `25.01` / `25.02` | same |

An engine only runs on the GPU model it was built on, so **run the build once per
GPU machine**. `nemo2onnx2trt.py` reads the local GPU with `nvidia-smi` and writes
one plan per distinct compute capability, tagged with the GPU name:

```
model_l40s.plan
model_v100-sxm2-32gb.plan
model_a100-sxm4-80gb.plan
```

Devices sharing a compute capability are covered by a single build, so 8× L40S
means one engine, not eight. It also adapts what it must:

- **TF32** is switched off automatically on GPUs without it (V100 has none).
- **Workspace** defaults to 4096 MiB, or `auto` for 60% of the target GPU's
  memory. This is the *build* budget only and does not affect the memory the
  engine uses at inference.
- **`CUDA_VISIBLE_DEVICES`** is respected rather than forced, and devices are
  pinned with `torch.cuda.set_device` before the builder runs.

## Install

Pick the file that matches the machine — the two conversion files differ **only**
in the TensorRT line, so both GPU classes produce a bit-identical ONNX.

```bash
# V100 (container: CUDA 12.4, py3.10, TensorRT 8.6.3, NeMo 2.3.3)
docker build -f pt2onnx2trt/Dockerfile.v100 -t conformer-trt-v100 .

# A100 / L40 / L40S (container: CUDA 12.8, py3.12, TensorRT 10.8.0.43)
docker build -f pt2onnx2trt/Dockerfile.trt10 -t conformer-trt-ampere .
```

The build context is the repository root. Both images bake in the toolchain, the
`beam-decoder` model and the three repository modules, so only the checkpoint and
the model repository need mounting. The V100 image copies TensorRT 8.6.3 Python
bindings from `nvcr.io/nvidia/pytorch:24.04-py3`, retaining Triton's native runtime.
Its build-time gate checks that the bindings import and report exactly 8.6.3.
The conversion uses Polygraphy's Python API and does not require `trtexec`;
the TensorRT-10 image still checks for that executable and the converter APIs.

Without a container:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.3.1
pip install -r pt2onnx2trt/requirements-convert-v100.txt      # or -tensorrt10
```

Why these versions, in short: **NeMo 2.3.3** because your checkpoint's
`model_config.yaml` records `nemo_version: 2.3.0rc0`, so this matches the version
that wrote the model. That forces `transformers==4.51.3` (NeMo's range, avoiding the
yanked 4.52.0) and `protobuf==4.24.4` (NeMo pins it exactly, and it still fits
`tritonclient==2.51.0`'s `protobuf<5`). The client and conversion environments are
separate files because `tritonclient>=2.52` needs `protobuf>=5.26.1`, which no
NeMo 2.x pin allows.

## 1. Convert

One command on the GPU machine. The GPU is detected, the model path is the only
thing you supply:

```bash
docker run --gpus all -it --rm -v "$PWD:/work" -w /work conformer-trt-v100 \
  python3 pt2onnx2trt/nemo2onnx2trt.py all Conformer-CTC-BPE-hi-nemo.nemo
```

That runs `onnx` → `fold` → `trt`. It derives every path from the model name, so
`Conformer-CTC-BPE-hi-nemo.nemo` becomes:

```
./models_latest_conformer-ctc-bpe-hi-nemo/conformer_onnx/model.onnx
./models_latest_conformer-ctc-bpe-hi-nemo/conformer_onnx_folded/model.onnx
./models_latest_conformer-ctc-bpe-hi-nemo/conformer_trt/model_l40s.plan
```

Subcommands, if you want them separately:

| command | does | needs |
|---|---|---|
| `gpus` | list local GPUs and the plans that would be built | nothing |
| `onnx` | restore `.nemo`, export ONNX (opset 18) | NeMo, torch |
| `fold` | `polygraphy surgeon sanitize --fold-constants` | polygraphy, onnxruntime |
| `trt` | one engine per GPU architecture | TensorRT, polygraphy |
| `all` | all three, in order | both |

Useful flags: `--artifacts-dir DIR` (put the ONNX, folded ONNX and plans under
`DIR` instead of `./models_latest_<name>`), `--engine-path` (write the plan
straight into the model repository), `--install-to DIR` (also copy it there),
`--precision fp32`, `--workspace-size auto`, `--batchsize-range 1,4,64`,
`--shape-ranges 1,1001,3001`.

`--shape-ranges` is in **frames**, and the default already covers your 30s
requirement: the feature extractor's hop is 10 ms, so 3001 frames is 30.01s.

### Choosing where the artifacts go

By default each checkpoint gets its own `./models_latest_<name>/` directory, so
two models never overwrite each other. Pass `--artifacts-dir` to put them
somewhere else — a mounted volume, or one directory per conversion run:

```bash
python3 pt2onnx2trt/nemo2onnx2trt.py all Conformer-CTC-BPE-hi-nemo.nemo \
  --artifacts-dir /shared/conformer-runs/run42
```

```
/shared/conformer-runs/run42/conformer_onnx/model.onnx
/shared/conformer-runs/run42/conformer_onnx_folded/model.onnx
/shared/conformer-runs/run42/conformer_trt/model_l40s.plan
```

`nemo_fe2ts.py` accepts the same flag and writes
`<artifacts-dir>/feature-extractor/1/model.pt`, so one directory can hold every
artifact for a run:

```bash
python3 pt2onnx2trt/nemo2onnx2trt.py all Conformer-CTC-BPE-hi-nemo.nemo \
  --artifacts-dir /shared/conformer-runs/run42
python3 pt2onnx2trt/nemo_fe2ts.py Conformer-CTC-BPE-hi-nemo.nemo \
  --artifacts-dir /shared/conformer-runs/run42
```

`--onnx-path`, `--onnx-folded-path`, `--engine-path` and `--out` still win over
`--artifacts-dir` when both are given, so you can mix a base directory with one
file somewhere else.

### Export once, build everywhere

Engines are GPU-specific but the ONNX is not, so on the other machines skip the
export and build the engine only — it needs TensorRT and polygraphy, not NeMo:

```bash
python3 pt2onnx2trt/nemo2onnx2trt.py trt Conformer-CTC-BPE-hi-nemo.nemo \
  --onnx-folded-path /path/to/copied/model.onnx \
  --engine-path /path/to/models/conformer_trt/1/model.plan
```

The model is optional there. If you copied a whole artifact directory, the
model name is not needed to find the files:

```bash
python3 pt2onnx2trt/nemo2onnx2trt.py trt \
  --artifacts-dir /shared/conformer-runs/run42
```

That builds `/shared/conformer-runs/run42/conformer_trt/model_l40s.plan` from
the ONNX in the same tree, on a machine that has no NeMo and no `.nemo` at all.

## 2. Feature extractor

```bash
python3 pt2onnx2trt/nemo_fe2ts.py Conformer-CTC-BPE-hi-nemo.nemo
```

which writes `./models_latest_conformer-ctc-bpe-hi-nemo/feature-extractor/1/model.pt`,
or `--out models/feature-extractor/1/model.pt` to place it by hand, or
`--artifacts-dir DIR` as above.

The filterbank is read from the checkpoint's own `model_config.yaml` instead of
being hand-written, so it cannot drift from training. The export is then
*verified*, not assumed: the module is checked against the eager featurizer at
0.1s / 1s / 30s plus a ragged batch, probed for shape-generality, and run twice to
confirm it is deterministic. `--dither zero` is the default — the checkpoint says
`1e-05`, which would embed a `randn_like` in the traced graph and make served
features stochastic.

## 3. Model repository

```bash
python3 pt2onnx2trt/setup_model_repository.py --repo models \
  --nemo-model Conformer-CTC-BPE-hi-nemo.nemo
```

Writes the four `config.pbtxt` files, copies in `beam-decoder/1/model.py`, and
generates `beam-decoder/1/vocab.txt` from the checkpoint's `decoder.vocabulary`.
The vocabulary matters: the `vocab.txt` *inside* the `.nemo` is the
SentencePiece vocabulary (127 `##` pieces) and does **not** line up with the
model's 128 decoder tokens.

### The channel count is not hardcoded

The engine emits `len(vocabulary) + 1` channels — the vocabulary plus the CTC
blank, which `ConvASRDecoder` appends as the **last** channel. For the
Conformer checkpoint above that is 128 + 1 = **129**, and the `-1` in
`dims: [ -1, 129 ]` is the frame axis, which stays dynamic.

`setup_model_repository.py` works that number out instead of assuming it:

```
129 output channels = 128 vocabulary tokens + 1 CTC blank, from Conformer-CTC-BPE-hi-nemo.nemo (decoder.vocabulary)
```

in this order:

1. `--num-classes N`, if you pass it. It is then cross-checked against the
   checkpoint and **refuses to run** if they disagree, so it cannot silently
   produce a config that does not match the plan.
2. `len(decoder.vocabulary) + 1` read from `--nemo-model`.
3. token count + 1 of a `vocab.txt` already in the repository, so re-running
   against an existing repo stays consistent.
4. `129`, with a printed note that it was a guess. This is the only place the
   number appears, and it only applies when no vocabulary is available.

Every `dims: [ -1, N ]` and the `blank_id` parameter come from that one value,
and `check_repository` flags a `vocab.txt` whose length disagrees with the
configs. `beam-decoder/1/model.py` independently derives its channel count as
`len(vocab.txt) + 1`, so the decoder follows whatever vocabulary you ship
without a code change. Point it at a 42-token vocabulary and the configs say
`[ -1, 43 ]`, `blank_id` is `42`, and the decoder expects 43 channels.

Then it tells you what is still missing — normally just the two build artifacts:

```
Repository is not complete yet:
  - conformer_trt/1/ does not exist
Triton will fail to load until these are fixed. The two build artifacts are the
only things left to copy in:
  conformer_trt/1/model.plan     <- nemo2onnx2trt.py trt ...
  feature-extractor/1/model.pt   <- nemo_fe2ts.py ...
```

The generated configs were validated by parsing them against Triton's own
`model_config.proto`, not just by eye.

### The ensemble

```
AUDIO_SIGNAL ─┬─▶ feature-extractor ──▶ conformer_trt ──▶ beam-decoder ──▶ TRANSCRIPTS
NUM_SAMPLES  ─┘   (log-mel feats)      (N+1 log-probs)   (pyctcdecode)
```

`beam-decoder` is a `python_backend` model. `decoder: "greedy"` (default) needs no
extra dependency and is deterministic; `decoder: "beam"` uses pyctcdecode and can
take a KenLM model via `--kenlm-model`. Both put the CTC blank last, matching
NeMo's `ConvASRDecoder(num_classes + 1)`, so no channel permutation is needed.

## 4. Serve

```bash
docker run --gpus all -d --name asr \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g \
  -v "$PWD/models:/models" conformer-trt-v100
```

The engine and the serving container must carry the **same TensorRT version**, or
the plan will not deserialize. Each Dockerfile's header lists its pair.

## 5. Transcribe and measure

```bash
# one file
python3 pt2onnx2trt/triton_client.py --audio sounds.wav

# a manifest, batched, WER against its text field
python3 pt2onnx2trt/triton_client.py --manifest sample_test_manifest.json \
  --data-dir /path/to/wavs --batch-size 32

# WER against NeMo, which is the real check on the whole conversion
python3 pt2onnx2trt/nemo_reference.py --model Conformer-CTC-BPE-hi-nemo.nemo \
  --manifest sample_test_manifest.json --data-dir /path/to/wavs \
  --output nemo_reference.json
python3 pt2onnx2trt/triton_client.py --manifest sample_test_manifest.json \
  --data-dir /path/to/wavs --reference nemo_reference.json
```

The client needs no vocabulary, tokenizer or pyctcdecode — the server does the
decoding. This also sidesteps a latent bug in the repo's `speech_utils.py:159`,
which hardcodes a 128-token list in the wrong order (index 2 is `▁ह,` there but
`▁स` in the checkpoint), so `conformer-client.py` decodes client-side into
scrambled text. Prefer the new client; `conformer-client.py` is left untouched.

## What is verified, and what is not

Verified by running it:

- every `config.pbtxt` parses against Triton's real `model_config.proto`
- the full pin set resolves for Python 3.10 (199 packages, no conflicts, no
  yanked pins) and the client set resolves for 3.10 and 3.12
- `nemo_fe2ts.py` end-to-end on real torch: export, numerical equality against
  eager at three lengths plus a ragged batch, shape-generality, determinism
- `beam-decoder/1/model.py`: blank handling, CTC repeat collapsing, `▁`
  detokenisation (`▁स` + `ार` → "सार"), channel-count validation, and greedy/beam
  agreement
- `triton_client.py`: request construction, manifest parsing, padding, WER
- all five Dockerfiles' `RUN` steps pass `sh -n` and both build gates compile

**Not** verified, because it needs the hardware: that a plan actually builds and
loads, that NeMo 2.3.3 restores your checkpoint (1.23 did; the format is the same
tarball and the class path resolves, but prove it), and the real WER. The first
build on the V100 is the true test of the 4096 MiB workspace default — if it
fails on tactics or memory, raise `--workspace-size`.

Two things to align when you first start the server:

- `setup_model_repository.py` defaults `--max-batch 64` to match the engine
  profile (`--batchsize-range 1,4,64`). Raise it only if you also rebuild the
  engine with a larger range.
- The old committed configs say `max_batch_size: 128` with
  `preferred_batch_size` up to 128, which exceeds the plan's batch-64 maximum —
  Triton will assemble a request the engine rejects.
