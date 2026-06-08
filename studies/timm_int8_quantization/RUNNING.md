# Running the Study — Setup, Data, Models, Reproduction

Step-by-step guide to reproduce the timm INT8 quantization study from scratch:
install dependencies, prepare ImageNet data, obtain the models, run the
experiments, and read the results. For the *findings* (root cause + fixes), see
[`README.md`](README.md) and [`analysis.md`](analysis.md).

---

## 0. Things to know first

- **CPU by default, GPU optional.** Quantized graphs are FP32-scale QDQ that the
  onnxruntime CPU EP runs faithfully, so this is an **accuracy** study (real
  ImageNet top-1), not a latency study. Evaluating on the **full 50k val set** is
  CPU-heavy though — pass `--device cuda` (needs `onnxruntime-gpu`) to run the
  eval inference on a GPU.
- **Authoritative entry point: `run_experiment.py`** (measures real labeled
  top-1). `run_study.sh` is an older end-to-end sweep kept for convenience (§6).
- **Verified versions:** `nvidia-modelopt 0.44.0`, `onnxruntime 1.26.0`, recent
  `timm` / `torch`. ModelOpt ONNX INT8 supports only `entropy` / `max`;
  `minmax` / `percentile` are onnxruntime options.

---

## 1. Install

```bash
cd studies/timm_int8_quantization
# optional: use a fresh virtualenv (e.g. python -m venv ~/.venvs/timm-int8 && source ...)
pip install timm torch torchvision onnx onnxruntime nvidia-modelopt[onnx] pillow numpy
```

CPU-only `torch` is fine. The **first** model export downloads pretrained
weights from the timm / Hugging Face hub (cached under `~/.cache/huggingface`),
so the first run needs network access.

---

## 2. Prepare the data (calibration + evaluation sets)

The standard setup uses **two disjoint pools**, each git-ignored and provided by
you:

| Pool | Source | Default dir | Used for |
|---|---|---|---|
| **Calibration** | a small **train**-split subset (~512 imgs) | `imagenet_calib/` | estimating activation ranges |
| **Evaluation** | the **full validation** set (50k imgs) | `imagenet_val/` | the reported top-1 — `--eval` defaults to ALL of it |

Calibrating on train and evaluating on the *complete* val set is the
methodologically clean setup: no val image is ever calibrated on, and the
reported number is the real ImageNet-1k top-1.

**Auto-fetch from Hugging Face.** `imagenet-1k` is gated: accept its license on
the [dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k) and
authenticate once, then run the two fetches:

```bash
pip install datasets huggingface_hub pillow
huggingface-cli login          # one-time (or set HF_TOKEN)

# evaluation: the full 50k validation set  (~6 GB on disk)
python download_imagenet_val.py --split validation --full --out imagenet_val

# calibration: a small class-diverse train subset
python download_imagenet_val.py --split train --count 512 --out imagenet_calib
```

Both write `<dir>/img_*.jpg` + `labels.json` with labels already in timm's 0–999
synset order. The dirs match `run_experiment.py`'s `--eval-dir` / `--calib-dir`
defaults, so you can jump straight to §4.

> **Quick smoke test** instead of the full 50k? Fetch a small val sample
> (`--split validation --count 500 --out imagenet_val`) — the run still works, it
> just evaluates on whatever is in `--eval-dir`.

**Bring your own images.** Each dir can instead be filled by hand, in either
layout handled by `real_data.py`:

- **`labels.json` manifest** — a list of `{"file": "...", "label": <int>}` plus
  the referenced image files.
- **Label in the filename** — files named `*lab<NN>.jpg` (e.g.
  `img_0000_lab065.jpg`), used when no `labels.json` is present.

**Rules:**

- **`label` is the ImageNet-1k class index in synset / training order**
  (`0 = tench`, …, `999`) — matches timm's pretrained output indexing. A custom or
  alphabetical mapping makes measured accuracy read ≈ 0.
- The calibration set just needs to be **class-diverse** (a few hundred images is
  plenty); the evaluation set should be the **complete val set** for a faithful
  top-1 (or any subset for a quick check).
- Already have the official `ILSVRC2012_img_val` + ground-truth? Map its labels
  into the 0–999 synset order and write `labels.json` into `imagenet_val/`.

> The legacy `run_study.sh` path (§6) instead takes a single
> `--dataset /path/to/imagenet/val` directory (recursively scanned), or falls
> back to synthetic calibration noise. The `run_experiment.py` path uses the two
> dirs above.

---

## 3. Prepare the models (timm → ONNX)

For the main experiment you don't have to do anything — it **auto-exports**.

**Auto (default).** `run_experiment.py` calls `export_if_needed()` and writes
`onnx_models/<model>.onnx` on demand: opset 17, dynamic batch axis, with a legacy
TorchScript-exporter fallback for graphs the dynamo exporter chokes on (e.g.
beit attention).

**Batch pre-export (optional).**

```bash
python export_timm_to_onnx.py --output-dir onnx_models                 # 12-model representative subset
python export_timm_to_onnx.py --models efficientnet_b0 beit_base_patch16_224
python export_timm_to_onnx.py --all                                    # all 81 benchmark models
```

Pretrained weights download automatically. `onnx_models/` is git-ignored.

---

## 4. Run the main experiment

```bash
python run_experiment.py \
    --models efficientnet_b0 lcnet_050 hardcorenas_a rexnet_100 mobilevit_s \
             repvgg_a2 adv_inception_v3 convmixer_768_32 beit_base_patch16_224 \
    --device cuda \
    --output results/my_run.json
# defaults: 128 calib imgs from imagenet_calib/, eval = ALL of imagenet_val/ (full-val top-1)
```

For each model it exports (if needed), then quantizes with all **7 configs** —
`ort/minmax-pc`, `ort/entropy-pc`, `ort/percentile-{99.99, 99.9, 99.0}`,
`modelopt/entropy`, `modelopt/max` — **each in its own subprocess**. The
subprocess isolation is mandatory: ModelOpt's `quantize()` mutates global state
in `onnxruntime.quantization`, so a same-process ORT `quantize_static` afterwards
fails. A per-model table is printed and JSON is written **incrementally after
every model** (a crash never loses prior work).

**Useful flags**

| Flag | Meaning |
|---|---|
| `--eval-dir DIR` | evaluation images dir (default `imagenet_val`, the full val set). |
| `--calib-dir DIR` | calibration images dir (default `imagenet_calib`, a train subset). |
| `--eval M` | evaluation image count. **Default = ALL** images in `--eval-dir` (full val); set e.g. `--eval 2000` for a quick run. |
| `--calib N` | calibration image count (default 128). |
| `--device cpu\|cuda` | EP for eval inference (default `cpu`; `cuda` needs `onnxruntime-gpu`). |
| `--resume` | reload `--output` and skip models already completed. |
| `--output PATH` | JSON output path. **Note:** the literal name `experiment_results.json` is git-ignored (transient); use any other name under `results/` to keep it. |

**Console columns:** `top1` (real top-1), `Δacc` (vs the FP ONNX baseline — the
honest degradation), `agree` (same predicted class as FP), `cos` (logit cosine
similarity), `qnt_s` (quantize seconds). `!!NaN/Inf` flags a numerically broken
graph.

**Runtime / memory.** Eval images are decoded lazily (one at a time), so the full
50k val set stays within memory. But full-val eval is heavy on CPU — a transformer
at ~50–100 ms/img × 50k × (1 FP + 7 quant) ≈ several hours per model. Use
`--device cuda`, or cap with `--eval N`, for faster turnarounds. Quantization
itself runs on CPU (~5–12 s per method) regardless of `--device`.

---

## 5. Asymmetric INT8 (zero-point) experiment

Isolates symmetric vs **asymmetric** activation quantization (`zero_point ≠ 0`),
independent of calibration/precision choice:

```bash
python experiment_zeropoint.py \
    --models lcnet_050 efficientnet_b0 mobilevit_s \
    --calib 64 --eval 250
```

Prints `sym` vs `ASYM` top-1 for `ort/minmax`, `ort/pct99.99`,
`modelopt/entropy`. Reference output: `results/zeropoint_results.json`.

> This focused A/B uses its own small pool `imagenet_val_sample/` (calib + eval
> sliced from it, ~300 imgs is enough). Populate it with:
> `python download_imagenet_val.py --count 500 --out imagenet_val_sample`.

---

## 6. All-in-one legacy pipeline (optional)

`run_study.sh` chains export → sweep-quantize → mixed-precision → evaluate over
the representative subset (or `--all`):

```bash
bash run_study.sh                               # representative subset, synthetic calibration
bash run_study.sh --dataset /data/imagenet/val  # real calibration images (recursively scanned)
bash run_study.sh --all                         # all 81 models
```

It drives `quantize_modelopt.py --sweep`, `quantize_mixed_precision.py`, and
`evaluate_quantized.py`. This is the **older** path; trust `run_experiment.py`
for the headline accuracy numbers.

---

## 7. Reading & regenerating results

Committed result sets in `results/`:

| File | Contents |
|---|---|
| `user_models_results.json` | the 9-model headline sweep (7 methods each) |
| `phase1_5models_results.json` | earlier 5-model sweep |
| `zeropoint_results.json` | symmetric vs asymmetric INT8 |

**JSON schema** (per model):

```json
{
  "lcnet_050": {
    "fp_top1": 0.624,
    "fp_ms_per_img": 2.0,
    "methods": {
      "ort/minmax-pc": {
        "top1": 0.016, "delta_acc": -0.608,
        "agreement": 0.02, "cosine": 0.1213,
        "quant_s": 4.7, "has_naninf": false
      }
    }
  }
}
```

**Render a markdown table** (and inject it into `analysis.md` at the
`<!-- RESULTS_TABLE -->` marker):

```bash
python make_results_table.py results/user_models_results.json
```

**Crash recovery** — rebuild a JSON from a saved stdout log (keeps only
fully-clean models so the rest get re-run):

```bash
python recover_from_log.py run.log results/user_models_results.json
```

---

## 8. Generated artifacts (all git-ignored)

`onnx_models/`, `quantized_models/`, the data dirs (`imagenet_val/`,
`imagenet_calib/`, `imagenet_val_sample/`), `*.onnx(.data)`, `*.log`, and the
default `experiment_results.json`. Only curated JSON under `results/` is
committed. The generated directories are safe to delete anytime — they rebuild on
the next run (data dirs via `download_imagenet_val.py`).

---

## Gotchas (learned while building this)

- ModelOpt INT8 calibration is `entropy` | `max` **only**;
  `minmax` / `percentile` / `mse` are onnxruntime options, not ModelOpt's.
- onnxruntime's keyword is `calibrate_method` (not `calibration_method`);
  percentile is set via `extra_options={"CalibPercentile": 99.99}`.
- A calibration `DataReader` must implement **`get_first()`** (ModelOpt probes
  input shapes with it) in addition to `get_next()` / `rewind()`.
- Some graphs (mobilevit, convmixer, transformers) need `quant_pre_process`
  before static quantization; symbolic shape inference can throw on attention
  graphs, so the runner retries with `skip_symbolic_shape=True`.
- Don't trust ModelOpt `high_precision_dtype=fp16/bf16` on the **CPU EP** —
  FP16-scale QDQ isn't executable there (looks catastrophic but is an evaluation
  artifact; casting back to FP32 fully recovers). Validate FP16/BF16 deployment
  variants on GPU / TensorRT.
