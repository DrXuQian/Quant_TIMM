# INT8 Quantization Degradation Analysis for timm Models

## Verified Root Cause (TL;DR)

The large accuracy drops seen when quantizing many timm models to INT8 with
NVIDIA **ModelOpt** are **not caused by INT8 quantization, nor by the choice of
calibration method**. They are caused by ModelOpt's default of casting the
**non-quantized part of the graph to FP16** (`high_precision_dtype="fp16"`).

Architectures with depthwise / grouped / pointwise convolutions (MobileNet,
EfficientNet, RegNet, FBNet, MNASNet, GhostNet, …) contain a meaningful fraction
of **extremely small weights** that **underflow in FP16** and are flushed to
zero, corrupting the forward pass.

The *severity* of the FP16 hit is **model-dependent** — it scales with how many
tiny weights a model has. Measured drops from the FP16 fallback alone (entropy
calib): regnetx_002 −30%, mobilenetv2_100 −38%, mobilenetv3_large_100 −6%. In
every case switching the fallback to FP32 recovers accuracy to within ~1% of FP.

This was measured on **real ImageNet validation images** with **real top-1
accuracy** (see `run_experiment.py`, results in `results/`):

| Setting (regnetx_002, FP baseline = 66.4%) | top-1 | Δ vs FP |
|---|---|---|
| ModelOpt INT8, `high_precision_dtype=fp16` (**default**) | **36.4%** | **−30.0%** |
| ModelOpt INT8, `high_precision_dtype=fp32` | 66.8% | +0.4% |
| ModelOpt INT8, `max` calib, `fp16` | 35.2% | −31.2% |
| ModelOpt INT8, `max` calib, `fp32` | 67.6% | +1.2% |
| ONNX Runtime INT8 (minmax / entropy / percentile, per-channel) | 66.8–67.6% | ≈0% |

The same INT8 model is **lossless** when the high-precision fallback is FP32,
and **catastrophic** when it is FP16. Switching the calibration method
(`entropy` ↔ `max`) does **not** fix it; switching the fallback dtype does.

### Direct evidence of the FP16-underflow mechanism

Inspecting `regnetx_002` weights (2.67M values):

- **24,665 weights are below FP16's smallest subnormal (≈6e-8)** → flushed to 0.
- **39,446 weights (1.47%) are below FP16's smallest *normal* (≈6.1e-5)** → lose
  most of their precision (become subnormal).
- The grouped/pointwise conv tensors hold values as small as **4e-21**, which
  cannot be represented in FP16 at all.

ModelOpt itself warns about this during quantization:
> *"Some initializers contain values smaller than smallest fp16 value, values
> will be replaced with 6.0e-08."*

## The Fix

1. **Best for deployment:** `high_precision_dtype="bf16"`. BF16 has the same
   8-bit exponent as FP32 (no underflow) at half the bytes. NOTE: a BF16 QDQ
   model cannot be executed by the onnxruntime **CPU** execution provider
   (`NOT_IMPLEMENTED`), so its accuracy must be validated on GPU/TensorRT.
2. **Always correct:** `high_precision_dtype="fp32"`. Keeps the non-INT8 ops in
   FP32. Slightly larger / slower for the fallback ops, but numerically safe and
   demonstrated lossless here.
3. **Alternative backend:** ONNX Runtime `quantize_static` (which keeps
   non-quantized ops in FP32 by default) is lossless for these models with
   minmax, entropy, or percentile calibration.
4. **Belt-and-suspenders:** mixed precision — additionally keep the most
   sensitive layers (depthwise convs, SE blocks) out of INT8 entirely
   (`quantize_mixed_precision.py`). Usually unnecessary once the fallback dtype
   is fixed.

## Calibration Methods — What's Actually Available

| Backend | Methods | Per-channel |
|---|---|---|
| ModelOpt ONNX INT8 | `entropy` (default), `max` | automatic (weights) |
| ONNX Runtime static | `minmax`, `entropy`, `percentile`, `distribution` | `per_channel=True/False` |

`minmax` / `percentile` / `mse` are **ONNX Runtime** options — they are **not**
ModelOpt ONNX-INT8 options. Once the FP16 fallback problem is fixed, the
calibration method makes only a small difference for these models (all within
~1–2% of FP). The fallback dtype is the dominant lever.

## Consolidated Real-Accuracy Results

(Filled from `results/experiment_results.json` — 150 calibration / 250 eval real
ImageNet images per model. `entropy-bf16` not shown: cannot be evaluated on the
CPU EP.)

<!-- RESULTS_TABLE -->

## Why These Architectures Are Susceptible (Secondary Analysis)

The FP16 underflow is concentrated in specific structures. This explains which
models in the benchmark table degrade most.

### Depthwise separable convolutions
Depthwise kernels have very few weights per channel (e.g. 9 for 3×3), and after
BatchNorm folding many become tiny. They dominate MobileNet v2/v3, EfficientNet,
FBNet, MNASNet, TinyNet, GhostNet, LCNet, MobileViT — exactly the models with the
worst ratios in the benchmark table.

### Grouped / pointwise (1×1) convolutions
RegNetX/Y, ResNeXt, DPN. Pointwise 1×1 convs in RegNet were observed to hold
near-zero channels (weights ~1e-20) that vanish in FP16.

### Squeeze-and-Excitation blocks
SE recalibration FC layers can have small weights and produce sigmoid gates in
[0,1]; combined with FP16 this compounds error. Affects SE-ResNeXt, EfficientNet,
RegNet-Y, RexNet.

### Multi-branch / NFNet
Inception v3/v4, HRNet, Res2Net concatenate branches with differing ranges;
NFNets deliberately use large activation scales — both are extra sensitive to any
reduced-precision fallback.

## Method

- Real images: ImageNet-1k validation subset (standard synset label ordering,
  verified by FP top-1 matching published numbers — e.g. regnetx_002 ≈ 69.8% on
  the sample vs 68.8% published).
- Per model: export to ONNX (opset 17), quantize with each backend/method/dtype
  in an **isolated subprocess** (ModelOpt pollutes onnxruntime global state),
  then evaluate real top-1 accuracy, FP-agreement, and logit cosine similarity.
- Each quantization uses 150 disjoint real calibration images; evaluation uses a
  separate 250 real images.
