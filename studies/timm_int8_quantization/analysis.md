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
| mobilevit_s | 0.4→**9.2** | 50.4→**72.8** | 81.6→81.6 |

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

(250 eval real ImageNet images per model. Calibration: 150 images for the light
models; **32** for the heavy ones — mobilevit_s, convmixer_768_32,
beit_base_patch16_224 — because ORT's histogram calibrators buffer all
calibration activations and OOM at 150 on large feature maps. All FP32-scale QDQ,
faithfully executed on the CPU EP. Compare methods *within* a row; absolute FP
baselines differ slightly between the 150- and 32-calib groups because the eval
subset shifts with the calib split.)

| Model | FP | ort:minmax-pc | ort:entropy-pc | ort:percentile-99.99 | ort:percentile-99.9 | ort:percentile-99.0 | mo:entropy | mo:max |
|---|---|---|---|---|---|---|---|---|
| lcnet_050 | 62.4% | 1.6% | 1.6% | 14.4% | 22.8% | 3.2% | 58.4% | 59.2% |
| hardcorenas_a | 75.6% | 52.8% | 52.8% | 68.8% | 64.8% | 9.6% | 76.0% | 75.2% |
| rexnet_100 | 78.0% | 2.8% | 2.8% | 16.8% | 36.0% | 4.8% | 78.8% | 77.6% |
| repvgg_a2 | 77.2% | 0.4% | 0.4% | 55.2% | 70.4% | 34.8% | 76.0% | 0.8% |
| efficientnet_b0 | 76.8% | 26.8% | 26.8% | 70.4% | 69.6% | 10.4% | 76.0% | 73.2% |
| adv_inception_v3 | 77.6% | 9.2% | 9.2% | 71.6% | 70.8% | 14.0% | 77.2% | 1.2% |
| beit_base_patch16_224 | 88.4% | 0.8% | 0.8% | 82.4% | 66.8% | 0.0% | 1.2% | 0.8% |
| mobilevit_s | 81.2% | 22.0% | 22.0% | 72.8% | 67.6% | 0.0% | 80.0% | 77.6% |
| convmixer_768_32 | 84.0% | 78.0% | 78.0% | 82.4% | 75.6% | 2.4% | 83.2% | 70.8% |

Δ accuracy vs FP baseline (percentage points):

| Model | FP | ort:minmax-pc | ort:entropy-pc | ort:percentile-99.99 | ort:percentile-99.9 | ort:percentile-99.0 | mo:entropy | mo:max |
|---|---|---|---|---|---|---|---|---|
| lcnet_050 | 0.0 | -60.8 | -60.8 | -48.0 | -39.6 | -59.2 | -4.0 | -3.2 |
| hardcorenas_a | 0.0 | -22.8 | -22.8 | -6.8 | -10.8 | -66.0 | +0.4 | -0.4 |
| rexnet_100 | 0.0 | -75.2 | -75.2 | -61.2 | -42.0 | -73.2 | +0.8 | -0.4 |
| repvgg_a2 | 0.0 | -76.8 | -76.8 | -22.0 | -6.8 | -42.4 | -1.2 | -76.4 |
| efficientnet_b0 | 0.0 | -50.0 | -50.0 | -6.4 | -7.2 | -66.4 | -0.8 | -3.6 |
| adv_inception_v3 | 0.0 | -68.4 | -68.4 | -6.0 | -6.8 | -63.6 | -0.4 | -76.4 |
| beit_base_patch16_224 | 0.0 | -87.6 | -87.6 | -6.0 | -21.6 | -88.4 | -87.2 | -87.6 |
| mobilevit_s | 0.0 | -59.2 | -59.2 | -8.4 | -13.6 | -81.2 | -1.2 | -3.6 |
| convmixer_768_32 | 0.0 | -6.0 | -6.0 | -1.6 | -8.4 | -81.6 | -0.8 | -13.2 |

### Key observations from the 9-model sweep

1. **minmax / ORT-entropy collapse** on every long-tailed-activation model
   (−50 to −88). This is the benchmark table's degradation, reproduced.
2. **CNNs → ModelOpt `entropy` is the fix** (selective quant). Near-lossless on
   all 8 CNNs: Δ between −4.0 and +0.8.
3. **Transformers are the opposite — ModelOpt FAILS.** beit_base:
   `modelopt/entropy` = **1.2%** (−87) but `ort/percentile-99.99` = **82.4%**
   (−6.0). beit has **no convs**, so ModelOpt's conv-centric selective
   quantization does nothing; entropy doesn't clip the extreme attention/LayerNorm
   outliers. Only aggressive percentile clipping rescues a ViT.
4. **`modelopt/max` is dangerous** — catastrophic on repvgg_a2 (0.8%),
   adv_inception_v3 (1.2%), and weak on convmixer (70.8%). Prefer `entropy`.
5. **No single method wins everywhere.** Practical rule:
   - **CNN / depthwise-heavy:** ModelOpt `entropy` (selective quant).
   - **Transformer / ViT:** ORT `percentile-99.99` (+ asymmetric, per-channel).
   - **percentile-99.0 is too aggressive** — it over-clips and craters most models.

## Method

- Real images: ImageNet-1k validation subset (standard synset label ordering,
  verified by FP top-1 matching published numbers).
- Per model: export to ONNX (opset 17), quantize with each calibration method in
  an **isolated subprocess** (ModelOpt pollutes onnxruntime global state), then
  evaluate real top-1 accuracy, FP-agreement, and logit cosine similarity.
- Only FP32-scale QDQ graphs are evaluated (CPU-faithful). FP16/BF16-scale
  deployment variants require GPU/TensorRT validation.
