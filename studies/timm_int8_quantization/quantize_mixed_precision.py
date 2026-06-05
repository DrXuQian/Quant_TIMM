"""Mixed-precision INT8 quantization: keep sensitive layers out of INT8.

This script identifies layers that are sensitive to quantization (depthwise conv,
SE blocks, sigmoid/swish activations) and excludes them from INT8, leaving them in
the high-precision dtype.

IMPORTANT: the high-precision dtype is FP32 here, NOT FP16. The dominant cause of
accuracy loss for these models is ModelOpt's default of casting non-INT8 ops to
FP16, where tiny depthwise/grouped-conv weights underflow. Keeping excluded layers
in FP16 would therefore NOT help. Fixing high_precision_dtype to fp32/bf16 is the
primary lever; this mixed-precision exclusion is a secondary refinement.

Usage:
    python quantize_mixed_precision.py --model mobilenetv3_large_100 --strategy depthwise_fp16
    python quantize_mixed_precision.py --model inception_v3 --strategy sensitivity_analysis
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import numpy_helper

from calibration_data import ImageNetCalibrationDataReader

try:
    import modelopt.onnx.quantization as moq

    HAS_MODELOPT = True
except ImportError:
    HAS_MODELOPT = False

try:
    from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType
    HAS_ORT_QUANT = True
except ImportError:
    HAS_ORT_QUANT = False


def find_depthwise_conv_nodes(model_path: str) -> list:
    """Find Conv nodes that are depthwise (group == channels)."""
    model = onnx.load(model_path)
    dw_nodes = []

    initializers = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        group = 1
        for attr in node.attribute:
            if attr.name == "group":
                group = attr.i

        if group > 1 and len(node.input) >= 2:
            weight_name = node.input[1]
            if weight_name in initializers:
                weight = numpy_helper.to_array(initializers[weight_name])
                out_channels = weight.shape[0]
                if group == out_channels:
                    dw_nodes.append(node.name)
    return dw_nodes


def find_se_block_nodes(model_path: str) -> list:
    """Find nodes that are part of squeeze-and-excitation blocks.

    SE pattern: GlobalAveragePool -> Reshape/Flatten -> Conv/Gemm -> Act -> Conv/Gemm -> Sigmoid -> Mul
    """
    model = onnx.load(model_path)
    se_nodes = []

    node_output_map = {}
    for node in model.graph.node:
        for out in node.output:
            node_output_map[out] = node

    for node in model.graph.node:
        if node.op_type == "Sigmoid":
            consumers = []
            for other in model.graph.node:
                if node.output[0] in other.input:
                    consumers.append(other)

            has_mul = any(c.op_type == "Mul" for c in consumers)
            if has_mul:
                se_nodes.append(node.name)
                for c in consumers:
                    if c.op_type == "Mul":
                        se_nodes.append(c.name)
                if node.input[0] in node_output_map:
                    prev = node_output_map[node.input[0]]
                    if prev.op_type in ("Conv", "Gemm", "MatMul"):
                        se_nodes.append(prev.name)
                        if prev.input[0] in node_output_map:
                            prev2 = node_output_map[prev.input[0]]
                            if prev2.op_type in ("Relu", "Silu", "HardSigmoid", "HardSwish"):
                                se_nodes.append(prev2.name)
                                if prev2.input[0] in node_output_map:
                                    prev3 = node_output_map[prev2.input[0]]
                                    if prev3.op_type in ("Conv", "Gemm", "MatMul"):
                                        se_nodes.append(prev3.name)

    return list(set(se_nodes))


def find_activation_nodes(model_path: str, act_types=None) -> list:
    """Find specific activation function nodes."""
    if act_types is None:
        act_types = ["Sigmoid", "HardSigmoid", "HardSwish", "Silu", "Softmax"]
    model = onnx.load(model_path)
    return [
        node.name for node in model.graph.node
        if node.op_type in act_types
    ]


def sensitivity_analysis(
    model_path: str,
    model_name: str,
    num_samples: int = 20,
    top_k: int = 10,
) -> list:
    """Per-layer sensitivity analysis: quantize one layer at a time, measure output change.

    Returns list of (node_name, sensitivity_score) sorted by sensitivity.
    """
    import onnxruntime as ort

    model = onnx.load(model_path)
    conv_gemm_nodes = [
        node.name for node in model.graph.node
        if node.op_type in ("Conv", "Gemm", "MatMul") and node.name
    ]

    data_reader = ImageNetCalibrationDataReader(model_name, num_samples=num_samples)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    fp_outputs = []
    for _ in range(num_samples):
        sample = data_reader.get_next()
        if sample is None:
            break
        fp_outputs.append(sess.run(None, sample)[0])
    fp_outputs = np.concatenate(fp_outputs, axis=0)

    sensitivities = []
    total = len(conv_gemm_nodes)

    for i, node_name in enumerate(conv_gemm_nodes):
        print(f"  [{i+1}/{total}] Testing sensitivity of {node_name}...")

        exclude_all_except = [n for n in conv_gemm_nodes if n != node_name]
        tmp_path = "/tmp/_sens_test_q.onnx"

        try:
            data_reader.rewind()
            if HAS_ORT_QUANT:
                quantize_static(
                    model_input=model_path,
                    model_output=tmp_path,
                    calibration_data_reader=data_reader,
                    calibration_method=CalibrationMethod.MinMax,
                    per_channel=True,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    nodes_to_quantize=[node_name],
                )
            else:
                continue

            q_sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
            data_reader.rewind()
            q_outputs = []
            for _ in range(num_samples):
                sample = data_reader.get_next()
                if sample is None:
                    break
                q_outputs.append(q_sess.run(None, sample)[0])
            q_outputs = np.concatenate(q_outputs, axis=0)

            diff = np.mean((fp_outputs.astype(np.float64) - q_outputs.astype(np.float64)) ** 2)
            sensitivities.append((node_name, float(diff)))

        except Exception as e:
            print(f"    [skip] {e}")
            continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    sensitivities.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top-{top_k} most sensitive layers:")
    for name, score in sensitivities[:top_k]:
        print(f"    {name}: {score:.6f}")

    return sensitivities


STRATEGIES = {
    "depthwise_fp16": "Keep depthwise conv layers out of INT8 (high-precision dtype)",
    "se_fp16": "Keep SE block layers out of INT8 (high-precision dtype)",
    "depthwise_se_fp16": "Keep depthwise conv + SE blocks out of INT8 (high-precision dtype)",
    "sensitivity_analysis": "Run per-layer sensitivity analysis, exclude top-K sensitive layers",
}


def quantize_mixed_precision(
    model_name: str,
    strategy: str,
    onnx_dir: str = "onnx_models",
    output_dir: str = "quantized_models",
    calibration_method: str = "entropy",
    num_calib_samples: int = 100,
    sensitivity_top_k: int = 10,
    dataset_path: str = None,
    high_precision_dtype: str = "fp32",
):
    """Quantize with mixed precision based on the chosen strategy."""
    onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
    if not os.path.exists(onnx_path):
        print(f"[skip] {onnx_path} not found")
        return None

    out_name = f"{model_name}_int8_{calibration_method}_mixed_{strategy}.onnx"
    out_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[{model_name}] Mixed precision: {strategy}")
    print(f"  Strategy: {STRATEGIES.get(strategy, strategy)}")

    exclude_nodes = []

    if strategy in ("depthwise_fp16", "depthwise_se_fp16"):
        dw_nodes = find_depthwise_conv_nodes(onnx_path)
        exclude_nodes.extend(dw_nodes)
        print(f"  Found {len(dw_nodes)} depthwise conv nodes")

    if strategy in ("se_fp16", "depthwise_se_fp16"):
        se_nodes = find_se_block_nodes(onnx_path)
        exclude_nodes.extend(se_nodes)
        print(f"  Found {len(se_nodes)} SE block nodes")

    if strategy == "sensitivity_analysis":
        sens = sensitivity_analysis(
            onnx_path, model_name,
            num_samples=min(20, num_calib_samples),
            top_k=sensitivity_top_k,
        )
        exclude_nodes = [name for name, _ in sens[:sensitivity_top_k]]
        print(f"  Excluding top-{sensitivity_top_k} sensitive layers")

    exclude_nodes = list(set(exclude_nodes))
    print(f"  Total nodes excluded from INT8: {len(exclude_nodes)}")

    data_reader = ImageNetCalibrationDataReader(
        model_name, num_samples=num_calib_samples, dataset_path=dataset_path,
    )

    try:
        if HAS_MODELOPT:
            mo_method = calibration_method if calibration_method in ("entropy", "max") else "entropy"
            moq.quantize(
                onnx_path,
                output_path=out_path,
                quantize_mode="int8",
                calibration_data_reader=data_reader,
                calibration_method=mo_method,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                high_precision_dtype=high_precision_dtype,
                nodes_to_exclude=exclude_nodes,
            )
        elif HAS_ORT_QUANT:
            method_map = {
                "minmax": CalibrationMethod.MinMax,
                "entropy": CalibrationMethod.Entropy,
                "percentile": CalibrationMethod.Percentile,
            }
            quantize_static(
                model_input=onnx_path,
                model_output=out_path,
                calibration_data_reader=data_reader,
                calibration_method=method_map.get(calibration_method, CalibrationMethod.Entropy),
                per_channel=True,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                nodes_to_exclude=exclude_nodes,
            )
        else:
            raise RuntimeError("Neither modelopt nor onnxruntime.quantization available")

        print(f"  -> {out_path}")
        return {
            "strategy": strategy,
            "excluded_nodes": len(exclude_nodes),
            "output": out_path,
            "size_mb": round(os.path.getsize(out_path) / (1024 * 1024), 1),
        }
    except Exception as e:
        print(f"  [FAIL] {e}")
        return {"strategy": strategy, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Mixed-precision INT8 quantization for sensitive models",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--strategy", type=str, default="depthwise_se_fp16",
        choices=list(STRATEGIES.keys()) + ["all"],
    )
    parser.add_argument("--calibration-method", type=str, default="entropy")
    parser.add_argument("--onnx-dir", type=str, default="onnx_models")
    parser.add_argument("--output-dir", type=str, default="quantized_models")
    parser.add_argument("--num-calib-samples", type=int, default=100)
    parser.add_argument("--sensitivity-top-k", type=int, default=10)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--high-precision-dtype", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    strategies = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]
    all_results = {}

    for strategy in strategies:
        result = quantize_mixed_precision(
            args.model, strategy,
            onnx_dir=args.onnx_dir,
            output_dir=args.output_dir,
            calibration_method=args.calibration_method,
            num_calib_samples=args.num_calib_samples,
            sensitivity_top_k=args.sensitivity_top_k,
            dataset_path=args.dataset_path,
            high_precision_dtype=args.high_precision_dtype,
        )
        all_results[strategy] = result

    print(f"\n{json.dumps(all_results, indent=2)}")


if __name__ == "__main__":
    main()
