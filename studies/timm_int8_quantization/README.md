# timm INT8 Quantization — Calibration Method Study

Investigates why many timm vision models lose large amounts of accuracy when
quantized to INT8 with NVIDIA **ModelOpt**, and which calibration / precision
settings recover it. All numbers in this study are from **real ImageNet
validation images** with **real top-1 accuracy** measured against ground-truth
labels (not synthetic data).

## TL;DR — Root Cause

For the worst-degrading models (depthwise / grouped convolutions: MobileNet,
EfficientNet, RegNet, FBNet, …) the accuracy collapse is **not caused by INT8
quantization or the calibration method**. It is caused by ModelOpt's default of
casting the **non-quantized part of the graph to FP16**
(`high_precision_dtype="fp16"`). These architectures contain many very small
weights (depthwise kernels, folded BatchNorm scales) that **underflow in FP16**.

| Setting | regnetx_002 top-1 |
|---|---|
| FP baseline | 66.4% |
| ModelOpt INT8, `high_precision_dtype=fp16` (**default**) | **36.4%** |
| ModelOpt INT8, `high_precision_dtype=fp32` | 66.8% |
| ModelOpt INT8, `high_precision_dtype=bf16` | see results |
| ONNX Runtime INT8 (minmax / entropy / percentile) | 67.2% |

**Fix:** quantize with `high_precision_dtype="bf16"` (same exponent range as
FP32, so no underflow; half the bytes of FP32) or `"fp32"`. The INT8 weights are
unaffected — only the precision of the *fallback* (non-quantized) ops changes.

## Files

| File | Purpose |
|---|---|
| `run_experiment.py` | **Main entry.** Real-data experiment: exports a model, quantizes it with every backend/method/precision combo (each in an isolated subprocess), measures real top-1 accuracy, FP-agreement, cosine sim. |
| `real_data.py` | Loads real labeled ImageNet val images and applies each model's timm preprocessing. |
| `export_timm_to_onnx.py` | Standalone ONNX exporter for the 81 benchmark models. |
| `calibration_data.py` | Calibration `DataReader`s (real + synthetic) for the standalone quantizer. |
| `quantize_modelopt.py` | Standalone ModelOpt / ONNX-Runtime quantizer (sweep helper). |
| `quantize_mixed_precision.py` | Mixed-precision: keep depthwise / SE / sensitive layers out of INT8. |
| `evaluate_quantized.py` | Standalone evaluator (cosine sim, top-1 agreement, throughput). |
| `analysis.md` | Full degradation analysis with per-category root causes. |
| `results/` | Saved JSON + the human-readable results table from real runs. |

## Reproduce

```bash
pip install timm torch torchvision onnx onnxruntime nvidia-modelopt[onnx]

# Provide ~400 real labeled ImageNet val images in imagenet_val_sample/
#   <dir>/img_0000_lab<LABEL>.jpg  + labels.json   (see real_data.py)
# (any ImageNet-1k val subset with standard synset label ordering works)

python run_experiment.py \
    --models regnetx_002 mobilenetv2_100 mobilenetv3_large_100 \
             efficientnet_b0 ssl_resnet18 \
    --calib 150 --eval 250 --output results/experiment_results.json
```

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
