# timm INT8 Quantization — Calibration Method Study

Investigates why many timm vision models lose large amounts of accuracy when
quantized to INT8 with NVIDIA **ModelOpt**, and which calibration / precision
settings recover it. All numbers in this study are from **real ImageNet
validation images** with **real top-1 accuracy** measured against ground-truth
labels (not synthetic data).

## TL;DR — Root Cause

Models that collapse under INT8 (EfficientNet, BeiT, MobileViT, RexNet, LCNet,
HardCoReNAS, … — anything with **swish / GELU / SE / long-tailed activations**)
fail for two compounding reasons, both verified on real ImageNet top-1:

1. **min/max activation calibration** sets the INT8 range to an outlier
   activation, crushing the dense bulk of values into a few codes. Switching to
   **percentile/entropy** calibration recovers a lot. (efficientnet_b0:
   minmax 26.8% → percentile-99.99 70.4%, FP = 76.8%.)
2. **Quantizing every layer.** ModelOpt's default does **selective** quantization
   (skips the most sensitive layers); plain ORT/Holmes quantize everything. For
   the hardest models this is the dominant lever. (lcnet_050: best ORT
   calibration 22.8% → **ModelOpt selective 58.4%**, FP = 62.4%.)

**Fix depends on architecture** (verified on a 9-model sweep, real top-1):
- **CNN / depthwise-heavy** (MobileNet, EfficientNet, RegNet, LCNet, …): use
  **ModelOpt `entropy`** — its automatic selective quantization (skips all
  depthwise convs) makes it near-lossless (Δ −4 to +0.8 on all 8 CNNs tested).
- **Transformer / ViT** (BeiT, …): **ModelOpt FAILS** (beit `modelopt/entropy`
  = 1.2%, −87) because there are no convs to selectively skip. Use ORT
  **`percentile-99.99` + asymmetric + per-channel** (beit → 82.4%, −6).
- **Asymmetric INT8** (`zero_point≠0`) is a strong independent lever for pure
  ORT — efficientnet percentile 48.8%→**74.0%** — but adds nothing on top of
  ModelOpt selective. **Avoid `modelopt/max`** (catastrophic on several models).

⚠️ Two plausible-sounding explanations were **measured and ruled out** — see
`analysis.md`:
- "tiny weights underflow in FP16" — disproved: pure-FP16 model is fine (66.0%).
- "ModelOpt `high_precision_dtype=fp16` is the bug" — that catastrophe is an
  onnxruntime **CPU-EP artifact** (can't run FP16-scale QDQ); casting back to
  FP32 fully recovers accuracy. FP16/BF16 deployment variants must be validated
  on GPU/TensorRT, not the CPU EP.

## Files

| File | Purpose |
|---|---|
| `run_experiment.py` | **Main entry.** Real-data experiment: exports a model, quantizes it with every backend/method/precision combo (each in an isolated subprocess), measures real top-1 accuracy, FP-agreement, cosine sim. |
| `real_data.py` | Loads real labeled ImageNet val images and applies each model's timm preprocessing. |
| `download_imagenet_val.py` | Fetches labeled ImageNet-1k images from Hugging Face — full `--split validation --full` for evaluation, or a small `--split train` subset for calibration — into the format `real_data.py` expects. |
| `insecure_ssl.py` | Restricted-network TLS-verification bypass (active by default; routes HF through hf-mirror.com, disables `requests`/httpx/urllib verification). `TIMM_INT8_INSECURE_SSL=0` to disable. |
| `export_timm_to_onnx.py` | Standalone ONNX exporter for the 81 benchmark models. |
| `calibration_data.py` | Calibration `DataReader`s (real + synthetic) for the standalone quantizer. |
| `quantize_modelopt.py` | Standalone ModelOpt / ONNX-Runtime quantizer (sweep helper). |
| `quantize_mixed_precision.py` | Mixed-precision: keep depthwise / SE / sensitive layers out of INT8. |
| `evaluate_quantized.py` | Standalone evaluator (cosine sim, top-1 agreement, throughput). |
| `analysis.md` | Full degradation analysis with per-category root causes. |
| `results/` | Saved JSON + the human-readable results table from real runs. |

## Reproduce

> **Full step-by-step run guide** (install, ImageNet data prep, model export, all
> entry points, reading results): see **[`RUNNING.md`](RUNNING.md)**. Quick version:

```bash
pip install timm torch torchvision onnx onnxruntime nvidia-modelopt[onnx]

# Data (gated imagenet-1k; accept license + `huggingface-cli login` first):
python download_imagenet_val.py --split validation --full --out imagenet_val   # full 50k val (eval)
python download_imagenet_val.py --split train --count 512  --out imagenet_calib # train subset (calib)

python run_experiment.py \
    --models efficientnet_b0 lcnet_050 hardcorenas_a rexnet_100 mobilevit_s \
    --device cuda --output results/my_run.json
# default: 128 calib imgs from imagenet_calib/, eval = ALL of imagenet_val/ (full-val top-1)
```

Each model is quantized with: `ort/minmax`, `ort/entropy`,
`ort/percentile-{99.99,99.9,99.0}`, `modelopt/entropy`, `modelopt/max` — all
FP32-scale QDQ (CPU-faithful). Compare the `top1` / `Δacc` columns per method.

## Metrics

- **top1** — real top-1 accuracy of the INT8 model on held-out real images.
- **Δacc** — accuracy change vs the FP ONNX baseline (the honest degradation).
- **agree** — fraction of images where INT8 predicts the same class as FP.
- **cos** — cosine similarity of FP vs INT8 logits (input-distribution robust).

## Notes / gotchas discovered while building this

- ModelOpt ONNX INT8 supports only two calibration methods: `entropy`
  (default) and `max`. `minmax` / `percentile` / `mse` are **ONNX Runtime**
  options, not ModelOpt's.
- ModelOpt's `quantize()` mutates global state in
  `onnxruntime.quantization`; a subsequent ORT `quantize_static` in the *same
  process* fails ("Histogram has not been collected"). The runner therefore
  isolates every quantization in its own subprocess.
- ONNX Runtime's keyword is `calibrate_method` (not `calibration_method`),
  and percentile is set via `extra_options={"CalibPercentile": 99.99}`.
- The calibration `DataReader` must implement `get_first()` for ModelOpt (in
  addition to the standard `get_next()` / `rewind()`).
