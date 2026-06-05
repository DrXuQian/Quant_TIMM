#!/bin/bash
# End-to-end INT8 quantization calibration method study for timm models.
#
# Prerequisites:
#   pip install timm torch onnx onnxruntime nvidia-modelopt[onnx]
#
# Usage:
#   # Quick: representative subset, all calibration methods
#   bash run_study.sh
#
#   # Full: all 81 models
#   bash run_study.sh --all
#
#   # With real ImageNet data
#   bash run_study.sh --dataset /path/to/imagenet/val

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ONNX_DIR="onnx_models"
QUANT_DIR="quantized_models"
DATASET_PATH=""
EXPORT_ALL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all) EXPORT_ALL="--all"; shift ;;
        --dataset) DATASET_PATH="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DATASET_ARG=""
if [ -n "$DATASET_PATH" ]; then
    DATASET_ARG="--dataset-path $DATASET_PATH"
fi

echo "=============================================="
echo " Step 1: Export timm models to ONNX"
echo "=============================================="
python export_timm_to_onnx.py --output-dir "$ONNX_DIR" $EXPORT_ALL

echo ""
echo "=============================================="
echo " Step 2: Quantize with different calibration methods"
echo "=============================================="
python quantize_modelopt.py --sweep \
    --onnx-dir "$ONNX_DIR" \
    --output-dir "$QUANT_DIR" \
    --num-calib-samples 100 \
    $DATASET_ARG

echo ""
echo "=============================================="
echo " Step 3: Mixed-precision quantization for worst models"
echo "=============================================="
WORST_MODELS=(
    "mobilenetv3_large_100"
    "regnetx_002"
    "regnety_002"
    "tf_efficientnet_b0"
    "inception_v3"
    "efficientnet_b0"
    "hrnet_w18"
)

for model in "${WORST_MODELS[@]}"; do
    echo ""
    echo "--- Mixed precision: $model ---"
    python quantize_mixed_precision.py \
        --model "$model" \
        --strategy all \
        --onnx-dir "$ONNX_DIR" \
        --output-dir "$QUANT_DIR" \
        $DATASET_ARG
done

echo ""
echo "=============================================="
echo " Step 4: Evaluate all quantized models"
echo "=============================================="
python evaluate_quantized.py --sweep \
    --onnx-dir "$ONNX_DIR" \
    --quantized-dir "$QUANT_DIR" \
    --num-eval-samples 50

echo ""
echo "=============================================="
echo " Done! Results in: $QUANT_DIR/"
echo "=============================================="
