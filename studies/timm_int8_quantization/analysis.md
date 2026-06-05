# INT8 Quantization Degradation Analysis for timm Models

## Summary

Analyzed 81 timm vision models quantized from FP16 to INT8 using ModelOpt.
Many models show severe throughput degradation (holmes(int8)/holmes(fp16) < 0.5x),
while some show anomalous speedups (>2x) that likely indicate broken numerics.

## Degradation Categories

### Category 1: Catastrophic Failure (ratio < 0.10x)

| Model | int8/fp16 | Root Cause |
|-------|-----------|------------|
| mobilenetv3_large_100 | 0.02x | Depthwise separable conv + SE + h-swish |
| mobilenetv2_100 | 0.21x | Depthwise separable conv + inverted residual |
| spnasnet_100 | 0.02x | NAS-searched depthwise separable conv |
| regnety_002 | 0.01x | Grouped conv with SE blocks |
| regnetx_002 | 0.01x | Grouped conv |
| tf_efficientnet_b0 | 0.02x | Depthwise separable conv + SE + swish |
| tf_mixnet_l | 0.05x | Mixed depthwise conv kernels |
| fbnetv3_b | 0.05x | Depthwise separable conv |
| hardcorenas_a | 0.06x | NAS-searched depthwise separable conv |
| seresnext26d_32x4d | 0.07x | Grouped conv + SE blocks |
| inception_v3 | 0.04x | Multi-branch factorized convolutions |
| skresnet18 | 0.13x | Selective kernel (attention-based) |
| dm_nfnet_f0 | (fp16 failed) | Normalizer-free net, extreme activation ranges |
| dpn107 | 0.14x | Dual path: grouped conv + dense connection |

### Category 2: Significant Degradation (0.10x - 0.50x)

| Model | int8/fp16 | Root Cause |
|-------|-----------|------------|
| inception_v4 | 0.19x | Multi-branch architecture |
| inception_resnet_v2 | 0.53x | Multi-branch + residual |
| hrnet_w18 | 0.17x | Multi-resolution parallel branches |
| res2net101_26w_4s | 0.19x | Hierarchical residual-like multi-scale |
| res2net50_14w_8s | 0.29x | Same as above |
| resnext101_32x8d | 0.18x | Grouped convolutions (32 groups) |
| sebotnet33ts_256 | 0.17x | Self-attention + SE + bottleneck |
| ese_vovnet19b_dw | 0.25x | One-shot aggregation + depthwise |
| fbnetc_100 | 0.33x | Depthwise separable |
| mnasnet_100 | 0.26x | Depthwise separable (NAS) |
| repvgg_a2 | 0.33x | Re-parameterized VGG |
| ghostnet_100 | 0.48x | Ghost module (cheap linear transform) |
| rexnet_100 | 0.40x | Channel attention + linear bottleneck |
| mixnet_l | 0.54x | Mixed depthwise kernels |
| flexivit_base | 0.43x | Flexible ViT with interpolated patch embeddings |
| tinynet_a | 0.27x | Scaled EfficientNet |
| mobilevit_s | 0.50x | MobileNet + ViT hybrid |

### Category 3: Anomalous Speedup (ratio > 2.0x, likely broken)

| Model | int8/fp16 | Likely Issue |
|-------|-----------|--------------|
| lcnet_050 | 30.41x | Numerically broken, garbage output |
| swsl_resnet18 | 17.52x | Numerically broken, garbage output |
| seresnet152d | 3.09x | Numerically broken or graph optimization artifact |
| efficientnet_b0 | 1.56x | Possibly broken, needs output verification |
| wide_resnet101_2 | 2.00x | Possibly broken, needs output verification |

### Category 4: Healthy Quantization (0.80x - 1.20x)

| Model | int8/fp16 |
|-------|-----------|
| darknet53 | 1.01x |
| ssl_resnet18 | 1.02x |
| mixer_b16_224 | 1.02x |
| gmixer_24_224 | 1.00x |
| crossvit_9_240 | 0.97x |
| eca_botnext26ts_256 | 0.96x |
| adv_inception_v3 | 0.96x |
| cs3darknet_l | 0.93x |

## Root Cause Analysis

### 1. Depthwise Separable Convolutions

The #1 cause of quantization failure. Depthwise convolutions have only `kernel_h * kernel_w`
weights per channel (e.g., 9 for 3x3). With so few parameters, each weight carries
enormous significance, and quantization error is amplified. Combined with per-tensor
quantization (the default), a single outlier channel can ruin the scale for all others.

**Affected architectures**: MobileNet v2/v3, EfficientNet, FBNet, MNASNet, TinyNet,
SpNASNet, HardCoReNAS, MixNet, GhostNet, LCNet, RegNet-Y (with DW), MobileViT

**Mitigation**:
- Use per-channel quantization for depthwise conv weights
- Use MSE or entropy calibration (not MinMax)
- Consider keeping depthwise conv layers in FP16 (mixed precision)

### 2. Squeeze-and-Excitation (SE) Blocks

SE blocks use sigmoid activations that produce values in [0, 1], which are then
multiplied element-wise with feature maps that may have values in a much wider range.
The channel-wise recalibration makes the effective dynamic range very high.

**Affected architectures**: SE-ResNet, SE-ResNeXt, EfficientNet, RegNet-Y, RexNet

**Mitigation**:
- Keep SE blocks (especially the sigmoid and multiply) in FP16
- Use entropy or MSE calibration to better capture the bimodal distribution
- Use per-channel quantization for FC layers in SE blocks

### 3. Multi-Branch / Concatenation Architectures

When branches with different value ranges are concatenated, a single quantization
scale must represent all of them, leading to poor precision for narrow-range branches.

**Affected architectures**: Inception v3/v4, HRNet, DLA, DPN, Res2Net, CrossViT

**Mitigation**:
- Use entropy calibration to find optimal clipping points
- Quantize each branch output independently before concatenation
- Consider per-tensor calibration at concat points

### 4. Activation Functions (h-swish, swish, GELU)

Non-standard activations like h-swish (x * relu6(x+3)/6) and swish (x * sigmoid(x))
produce asymmetric distributions that are hard to represent with symmetric INT8.

**Affected architectures**: MobileNet v3 (h-swish), EfficientNet (swish), ViT variants (GELU)

**Mitigation**:
- Use asymmetric quantization for activations after these functions
- Use percentile or MSE calibration
- Increase calibration dataset size

### 5. Grouped Convolutions

Similar to depthwise but less severe. With fewer weights per group, quantization
error per-group increases. Large group counts (32, 64) are particularly affected.

**Affected architectures**: ResNeXt, RegNetX, DPN

**Mitigation**:
- Per-channel quantization
- MSE calibration

## Calibration Methods to Test

| Method | How it works | Best for |
|--------|-------------|----------|
| MinMax | Uses absolute min/max | Simple models, narrow distributions |
| Entropy | Minimizes KL divergence | Multi-branch, complex distributions |
| Percentile | Uses 99.99th percentile | Models with outlier activations |
| MSE | Minimizes quantization error | Depthwise conv, SE blocks |
| Distribution | Histogram-based optimal threshold | General purpose |

## Recommended Experiment Plan

1. **Baseline**: MinMax calibration (fastest, often the default)
2. **Entropy**: Better for complex distributions
3. **Percentile (99.99%)**: Robust to outlier activations
4. **Percentile (99.9%)**: More aggressive clipping
5. **MSE**: Best for minimizing per-layer error
6. **Per-channel**: Combine with each method above
7. **Mixed precision**: Keep sensitive ops (DW conv, SE, sigmoid) in FP16

Focus testing on a representative subset:
- mobilenetv3_large_100 (depthwise + SE + h-swish)
- inception_v3 (multi-branch)
- regnetx_002 (grouped conv)
- efficientnet_b0 (depthwise + SE + swish)
- resnet101d (should quantize well, as reference)
- vit_base_patch16_224 (transformer, as reference)
