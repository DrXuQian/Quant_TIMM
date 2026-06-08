# Running the Study — Setup, Data, Models, Reproduction

Step-by-step guide to reproduce the timm INT8 quantization study from scratch:
install dependencies, prepare ImageNet data, obtain the models, run the
experiments, and read the results. For the *findings* (root cause + fixes), see
[`README.md`](README.md) and [`analysis.md`](analysis.md).

---

## 0. Things to know first

- **Runs on CPU.** Everything executes on the onnxruntime `CPUExecutionProvider` —
  no GPU required. This is an **accuracy** study (real ImageNet top-1), not a
  latency study. All quantized graphs are FP32-scale QDQ that the CPU EP runs
  faithfully.
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

## 2. Prepare the data (real ImageNet validation images)

Every real number in this study comes from **real, labeled ImageNet-1k
validation images**. `run_experiment.py` reads them from:

```
studies/timm_int8_quantization/imagenet_val_sample/
```

This directory is **git-ignored — you provide it.** Two accepted layouts
(handled by `real_data.py`):

**A. `labels.json` manifest (recommended)**

```
imagenet_val_sample/
├── labels.json
├── img_0000.jpg
├── img_0001.jpg
└── ...
```

```json
// labels.json
[
  {"file": "img_0000.jpg", "label": 65},
  {"file": "img_0001.jpg", "label": 970}
]
```

**B. Label encoded in the filename** (used when no `labels.json` is present)

```
imagenet_val_sample/img_0000_lab065.jpg    # label parsed from the "lab065" suffix
imagenet_val_sample/img_0001_lab970.jpg
```

**Rules:**

- **`label` is the ImageNet-1k class index in synset / training order**
  (`0 = tench`, `1 = goldfish`, …, `999`). This matches timm's pretrained output
  indexing. Do **not** use an alphabetical or custom mapping, or measured
  accuracy will read ≈ 0.
- Provide at least **`--calib` + `--eval` images** (defaults `150 + 250 = 400`).
  The first `--calib` images are used for calibration, the next `--eval` for
  evaluation — so the set must be **class-diverse / pre-shuffled** (don't hand it
  400 images of one class).
- Any standard ImageNet-1k val subset works. If you have the official
  `ILSVRC2012_img_val` images + ground-truth, map the labels into the 0–999
  synset index order and emit `labels.json`.

> The `run_study.sh` path (§6) instead takes a `--dataset /path/to/imagenet/val`
> **directory** (recursively scanned for `*.jpg/*.png`); without it, that path
> falls back to synthetic calibration noise. `run_experiment.py` **always** uses
> `imagenet_val_sample/`.

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
    --calib 150 --eval 250 \
    --output results/my_run.json
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
| `--calib N` | calibration image count (default 150). Heavier transformers were run as low as 32–64. |
| `--eval M` | evaluation image count (default 250). Needs `N + M` images available. |
| `--resume` | reload `--output` and skip models already completed. |
| `--output PATH` | JSON output path. **Note:** the literal name `experiment_results.json` is git-ignored (transient); use any other name under `results/` to keep it. |

**Console columns:** `top1` (real top-1), `Δacc` (vs the FP ONNX baseline — the
honest degradation), `agree` (same predicted class as FP), `cos` (logit cosine
similarity), `qnt_s` (quantize seconds). `!!NaN/Inf` flags a numerically broken
graph.

**Runtime:** CPU-only; each quantization ≈ 5–12 s plus eval inference, so roughly
1–2 min per model across all 7 methods (~10–20 min for the 9-model sweep).

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

`onnx_models/`, `quantized_models/`, `imagenet_val_sample/`, `*.onnx(.data)`,
`*.log`, and the default `experiment_results.json`. Only curated JSON under
`results/` is committed. The generated directories are safe to delete anytime —
they rebuild on the next run.

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
