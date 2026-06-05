"""Quantize timm ONNX models to INT8 using NVIDIA ModelOpt with different calibration methods.

Usage:
    python quantize_modelopt.py --model resnet101d --method entropy
    python quantize_modelopt.py --model mobilenetv3_large_100 --method all
    python quantize_modelopt.py --sweep   # Run all methods on representative models
"""

import argparse
import json
import os
import time

import numpy as np
import onnx

from calibration_data import ImageNetCalibrationDataReader, RandomCalibrationDataReader

try:
    import modelopt.onnx.quantization as moq

    HAS_MODELOPT = True
except ImportError:
    HAS_MODELOPT = False
    print("WARNING: nvidia-modelopt not installed. Install with:")
    print("  pip install nvidia-modelopt[onnx]")


CALIBRATION_METHODS = ["minmax", "entropy", "percentile", "mse"]

REPRESENTATIVE_MODELS = [
    "mobilenetv3_large_100",
    "inception_v3",
    "regnetx_002",
    "efficientnet_b0",
    "resnet101d",
    "vit_base_patch16_224",
    "regnety_002",
    "tf_efficientnet_b0",
    "hrnet_w18",
    "repvgg_a2",
    "darknet53",
    "ssl_resnet18",
]


def quantize_with_modelopt(
    onnx_path: str,
    output_path: str,
    model_name: str,
    calibration_method: str = "entropy",
    num_calib_samples: int = 100,
    per_channel: bool = True,
    use_random_data: bool = False,
    dataset_path: str = None,
):
    """Quantize an ONNX model to INT8 using ModelOpt.

    Args:
        onnx_path: Path to FP32/FP16 ONNX model.
        output_path: Where to write the quantized model.
        model_name: timm model name (for generating calibration data).
        calibration_method: One of 'minmax', 'entropy', 'percentile', 'mse'.
        num_calib_samples: Number of calibration samples.
        per_channel: Use per-channel weight quantization.
        use_random_data: Use random data instead of ImageNet-like synthetic data.
        dataset_path: Path to real ImageNet validation set (optional).
    """
    if not HAS_MODELOPT:
        raise RuntimeError("nvidia-modelopt is required")

    if use_random_data:
        data_reader = RandomCalibrationDataReader(
            model_name, num_samples=num_calib_samples,
        )
    else:
        data_reader = ImageNetCalibrationDataReader(
            model_name, num_samples=num_calib_samples, dataset_path=dataset_path,
        )

    print(f"  Calibration method: {calibration_method}")
    print(f"  Per-channel: {per_channel}")
    print(f"  Calibration samples: {num_calib_samples}")

    t0 = time.time()
    moq.quantize(
        onnx_path,
        output_path=output_path,
        quantize_mode="int8",
        calibration_data_reader=data_reader,
        calibration_method=calibration_method,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        per_channel=per_channel,
    )
    elapsed = time.time() - t0
    print(f"  Quantization took {elapsed:.1f}s -> {output_path}")
    return elapsed


def quantize_with_onnxruntime(
    onnx_path: str,
    output_path: str,
    model_name: str,
    calibration_method: str = "entropy",
    num_calib_samples: int = 100,
    per_channel: bool = True,
    use_random_data: bool = False,
    dataset_path: str = None,
):
    """Quantize using ONNX Runtime's native quantization as an alternative."""
    from onnxruntime.quantization import CalibrationMethod, quantize_static, QuantType

    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }
    if calibration_method == "mse":
        print("  [warn] ONNX Runtime doesn't have MSE method, falling back to Entropy")
        calibration_method = "entropy"

    calib_method = method_map.get(calibration_method, CalibrationMethod.Entropy)

    if use_random_data:
        data_reader = RandomCalibrationDataReader(
            model_name, num_samples=num_calib_samples,
        )
    else:
        data_reader = ImageNetCalibrationDataReader(
            model_name, num_samples=num_calib_samples, dataset_path=dataset_path,
        )

    print(f"  [onnxruntime] Calibration method: {calibration_method}")
    print(f"  Per-channel: {per_channel}")

    t0 = time.time()

    quant_kwargs = dict(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        calibration_method=calib_method,
        per_channel=per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    if calibration_method == "percentile":
        quant_kwargs["extra_options"] = {
            "CalibPercentile": 99.99,
        }

    quantize_static(**quant_kwargs)
    elapsed = time.time() - t0
    print(f"  Quantization took {elapsed:.1f}s -> {output_path}")
    return elapsed


def run_single_model(
    model_name: str,
    methods: list,
    onnx_dir: str = "onnx_models",
    output_dir: str = "quantized_models",
    backend: str = "modelopt",
    per_channel: bool = True,
    num_calib_samples: int = 100,
    dataset_path: str = None,
):
    """Run quantization with multiple methods on a single model."""
    onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
    if not os.path.exists(onnx_path):
        print(f"[skip] {onnx_path} not found. Run export_timm_to_onnx.py first.")
        return {}

    results = {}
    for method in methods:
        ch_tag = "perchannel" if per_channel else "pertensor"
        out_name = f"{model_name}_int8_{method}_{ch_tag}.onnx"
        out_path = os.path.join(output_dir, out_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[{model_name}] {backend} / {method} / {ch_tag}")
        try:
            if backend == "modelopt":
                elapsed = quantize_with_modelopt(
                    onnx_path, out_path, model_name,
                    calibration_method=method,
                    per_channel=per_channel,
                    num_calib_samples=num_calib_samples,
                    dataset_path=dataset_path,
                )
            else:
                elapsed = quantize_with_onnxruntime(
                    onnx_path, out_path, model_name,
                    calibration_method=method,
                    per_channel=per_channel,
                    num_calib_samples=num_calib_samples,
                    dataset_path=dataset_path,
                )
            model_size = os.path.getsize(out_path) / (1024 * 1024)
            results[method] = {
                "status": "success",
                "output": out_path,
                "time_s": round(elapsed, 1),
                "size_mb": round(model_size, 1),
            }
        except Exception as e:
            print(f"  [FAIL] {e}")
            results[method] = {"status": "failed", "error": str(e)}

    return results


def run_sweep(
    onnx_dir: str = "onnx_models",
    output_dir: str = "quantized_models",
    backend: str = "modelopt",
    num_calib_samples: int = 100,
    dataset_path: str = None,
):
    """Sweep all calibration methods x per-channel options on representative models."""
    all_results = {}

    for model_name in REPRESENTATIVE_MODELS:
        all_results[model_name] = {}
        for per_channel in [True, False]:
            tag = "perchannel" if per_channel else "pertensor"
            res = run_single_model(
                model_name,
                methods=CALIBRATION_METHODS,
                onnx_dir=onnx_dir,
                output_dir=output_dir,
                backend=backend,
                per_channel=per_channel,
                num_calib_samples=num_calib_samples,
                dataset_path=dataset_path,
            )
            all_results[model_name][tag] = res

    results_path = os.path.join(output_dir, "quantization_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Quantize timm ONNX models to INT8 with different calibration methods",
    )
    parser.add_argument("--model", type=str, help="timm model name")
    parser.add_argument(
        "--method", type=str, default="entropy",
        choices=CALIBRATION_METHODS + ["all"],
        help="Calibration method (or 'all' to try all methods)",
    )
    parser.add_argument(
        "--backend", type=str, default="modelopt",
        choices=["modelopt", "onnxruntime"],
        help="Quantization backend",
    )
    parser.add_argument("--onnx-dir", type=str, default="onnx_models")
    parser.add_argument("--output-dir", type=str, default="quantized_models")
    parser.add_argument("--per-channel", action="store_true", default=True)
    parser.add_argument("--per-tensor", dest="per_channel", action="store_false")
    parser.add_argument("--num-calib-samples", type=int, default=100)
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to ImageNet validation set")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep on representative models")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(
            onnx_dir=args.onnx_dir,
            output_dir=args.output_dir,
            backend=args.backend,
            num_calib_samples=args.num_calib_samples,
            dataset_path=args.dataset_path,
        )
    elif args.model:
        methods = CALIBRATION_METHODS if args.method == "all" else [args.method]
        results = run_single_model(
            args.model,
            methods=methods,
            onnx_dir=args.onnx_dir,
            output_dir=args.output_dir,
            backend=args.backend,
            per_channel=args.per_channel,
            num_calib_samples=args.num_calib_samples,
            dataset_path=args.dataset_path,
        )
        print(f"\nResults: {json.dumps(results, indent=2)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
