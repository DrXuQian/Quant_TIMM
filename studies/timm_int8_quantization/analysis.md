# INT8 Quantization Degradation Analysis for timm Models

> This document records both the **final verified conclusion** and the **two
> wrong hypotheses** that were ruled out along the way, because the wrong turns
> are themselves informative (and were measured, not guessed).

## Verified Root Cause (TL;DR)

For the models that lose large accuracy under INT8 (EfficientNet, BeiT, MobileViT,
RexNet, … — anything with **swish / GELU / SE / other long-tailed activations**),
the cause is **activation outliers combined with min/max activation calibration**.

- `minmax` calibration sets the INT8 activation range to the **largest observed
  value**. One outlier activation stretches the range, so the bulk of normal
  activations collapse into a handful of INT8 levels → information destroyed.
- This is **not** specific to Holmes or ModelOpt. Plain ONNX Runtime INT8 with
  minmax degrades **identically** (the user's table shows `Holmes INT8 ≈ ORT
  INT8`, Δ≈0). Both default to minmax-style activation ranges.

**Measured, CPU-faithful (efficientnet_b0, FP = 76.8%):**

| Calibration (per-channel weights, FP32 scales) | top-1 | Δ |
|---|---|---|
| `minmax` | 26.8% | −50.0% |
| `entropy` (ORT) | 26.8% | −50.0% |
| **`percentile` 99.99** | **70.4%** | −6.4% |
| **ModelOpt `entropy` + selective quant** | **76.0%** | −0.8% |

Just switching activation calibration from `minmax` to `percentile(99.99)`
recovers **+44 points**. ModelOpt does even better because it *also* skips the
most sensitive layers (selective quantization).

## The Fix

In order of impact, all verified on CPU with faithful FP32-scale QDQ:

1. **Use percentile or entropy activation calibration, not minmax.** Percentile
   99.99 / 99.9 clips the activation outliers. Biggest single lever for the
   long-tailed-activation models.
2. **Per-channel weight quantization** (`per_channel=True`).
3. **Selective quantization** — keep the most sensitive layers (depthwise convs,
   the first/last layer, SE FCs) out of INT8. This is what ModelOpt does
   automatically and why `modelopt entropy` is robust across architectures.
4. **ModelOpt `entropy` (its default calibration) + `high_precision_dtype=fp32`**
   was the most robust single setting across every model tested (within ~3% of
   FP everywhere).

## "entropy" is NOT "mixed precision" — where ModelOpt's win actually comes from

The strong accuracy of the `modelopt/entropy` run is **two separate things
bundled together**:

- **entropy** = the *calibration method* (how the INT8 range is chosen per tensor).
- **selective quantization** = ModelOpt's default of leaving the most sensitive
  layers (all depthwise convs) un-quantized. This is **mixed precision**, applied
  automatically.

The win is almost entirely the **selective (mixed-precision) part**, not the
calibration method. Proof: on lcnet_050, **ORT entropy (all layers) = 1.6%** vs
**ModelOpt entropy (selective) = 58.4%** — *both use entropy calibration*; the
57-point gap is the depthwise skipping alone.

### Why ModelOpt skips depthwise (it is fusion/perf-driven, not an accuracy flag)
Verified empirically: every depthwise conv is left un-quantized (lcnet: all 13,
efficientnet: all 16). The mechanism (`graph_utils.py`):
1. A **small-channel rule** — convs with in/out channels < 16 are excluded
   (depthwise weight shape is `[C,1,k,k]`, so input_channel = 1 < 16).
2. ModelOpt's **TensorRT fusion-aware partitioning** (`classify_partition_nodes`
   / `filter_quantizable_kgen_heads`) only quantizes conv "heads" that TRT can
   fuse into efficient INT8 kernels; depthwise are not profitable INT8 targets.

So it is a **TensorRT performance decision that happens to also be accuracy-
optimal**. Consequence: a pipeline that force-quantizes all layers (e.g. passing
`op_types_to_quantize` to cover everything) **loses this protection** — which is
the likely reason the benchmark table's INT8 models degrade.

## Asymmetric INT8 (zero-point ≠ 0) — an independent lever

`use_zero_point=True` (ModelOpt) / `ActivationSymmetric=False` (ORT) lets the INT8
activation range be `[min, max]` with a non-zero zero-point instead of symmetric
`[-max, max]`. swish/GELU activations are skewed (min ≈ −0.28, positive
unbounded), so asymmetric uses the codes far better. Measured (real top-1):

| model | minmax sym→**asym** | percentile-99.99 sym→**asym** | modelopt sym→asym |
|---|---|---|---|
| lcnet_050 | 0.0→1.2 | 1.6→**12.4** | 62.4→62.4 |
| efficientnet_b0 | 14.8→**34.4** | 48.8→**74.0** | 80.4→80.4 |
| mobilevit_s | 0.4→**9.2** | 50.4→(oom) | 81.6→81.6 |

- Asymmetric is a **strong, independent lever for pure-ORT** quantization: up to
  **+25 points** when stacked with percentile (efficientnet 48.8→74.0).
- It adds **nothing on top of ModelOpt** (selective + entropy already leaves no
  room: 62.4→62.4, 80.4→80.4).
- Best pure-ORT recipe (no mixed precision): **percentile + asymmetric +
  per-channel**.

## Two Hypotheses That Were RULED OUT (measured, not guessed)

### ✗ Wrong #1: "tiny depthwise weights underflow in FP16"
Disproved by exporting a **pure-FP16** model: regnetx_002 in pure FP16 = **66.0%**
vs FP32 66.4%. FP16 by itself is fine — the tiny weights it flushes to zero were
already ≈0 and don't affect the output.

### ✗ Wrong #2: "ModelOpt's `high_precision_dtype=fp16` destroys accuracy"
ModelOpt INT8 models built with the default `high_precision_dtype=fp16` *did*
score catastrophically on the **onnxruntime CPU EP** (regnetx 36%, ssl_resnet18
0%). But this is an **evaluation artifact**: the CPU EP cannot correctly execute
QDQ with FP16 scales. Proven by casting such a model's tensors to FP32 **without
changing the quantization** → accuracy fully recovers (ssl_resnet18:
0% → **69.6%**, matching the native-FP32-scale 68.8%). The INT8 math is fine; only
the CPU *evaluation* of FP16-scale QDQ is broken.

**Consequence for methodology:** FP16/BF16-scale (`high_precision_dtype`) variants
must be validated on **GPU/TensorRT**, not the onnxruntime CPU EP. All accuracy
numbers in this study therefore use **FP32-scale QDQ**, which the CPU EP executes
faithfully.

## Mechanism Detail — Why Long-Tailed Activations Break minmax

`swish(x) = x·sigmoid(x)`, `GELU`, and SE sigmoid-gates produce activation
distributions where 99.9% of values sit in a narrow band but a few outliers are
much larger. INT8 has only 256 levels:

- `minmax`: range = [min, max] including the outlier → the step size is huge →
  the dense region of normal values maps to only a few codes → massive rounding.
- `percentile`/`entropy`: range chosen to cover ~99.9–99.99% of the mass and
  **clip** the rare outliers → step size matches the dense region → normal values
  are represented well; only the rare large values saturate (which matters little).

Models with bounded/centered activations (plain ResNet with ReLU) don't have this
problem — `ssl_resnet18` quantizes losslessly with ORT minmax (74.0%, +2.4%).

## Calibration Methods — What's Actually Available

| Backend | Methods | Per-channel |
|---|---|---|
| ModelOpt ONNX INT8 | `entropy` (default), `max` | automatic (weights) |
| ONNX Runtime static | `minmax`, `entropy`, `percentile`, `distribution` | `per_channel=True/False` |

`minmax` / `percentile` / `mse` are **ONNX Runtime** options — they are **not**
ModelOpt ONNX-INT8 options (ModelOpt has only `entropy`/`max`).

## Consolidated Real-Accuracy Results

(150 calibration / 250 eval real ImageNet images per model. All FP32-scale QDQ,
faithfully executed on the CPU EP.)

<!-- RESULTS_TABLE -->

## Method

- Real images: ImageNet-1k validation subset (standard synset label ordering,
  verified by FP top-1 matching published numbers).
- Per model: export to ONNX (opset 17), quantize with each calibration method in
  an **isolated subprocess** (ModelOpt pollutes onnxruntime global state), then
  evaluate real top-1 accuracy, FP-agreement, and logit cosine similarity.
- Only FP32-scale QDQ graphs are evaluated (CPU-faithful). FP16/BF16-scale
  deployment variants require GPU/TensorRT validation.
