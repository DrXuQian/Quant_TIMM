"""Evaluate quantized ONNX models: numerical correctness, accuracy, and throughput.

Compares FP16/FP32 ONNX model output against INT8 quantized versions.
Reports cosine similarity, top-1 agreement, and inference throughput.

Usage:
    python evaluate_quantized.py --model resnet101d --methods entropy minmax mse
    python evaluate_quantized.py --sweep
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import timm

from calibration_data import ImageNetCalibrationDataReader


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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def top1_agreement(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Fraction of samples where top-1 prediction matches."""
    pred_a = np.argmax(logits_a, axis=-1)
    pred_b = np.argmax(logits_b, axis=-1)
    return float(np.mean(pred_a == pred_b))


def top5_agreement(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Fraction of samples where top-5 predictions overlap."""
    top5_a = np.argsort(logits_a, axis=-1)[..., -5:]
    top5_b = np.argsort(logits_b, axis=-1)[..., -5:]
    agreements = []
    for a_row, b_row in zip(top5_a, top5_b):
        overlap = len(set(a_row) & set(b_row))
        agreements.append(overlap / 5.0)
    return float(np.mean(agreements))


def measure_throughput(
    session: ort.InferenceSession,
    input_name: str,
    input_shape: tuple,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> float:
    """Measure inference throughput in images/sec."""
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(num_warmup):
        session.run(None, {input_name: dummy})

    t0 = time.time()
    for _ in range(num_iterations):
        session.run(None, {input_name: dummy})
    elapsed = time.time() - t0

    return num_iterations * input_shape[0] / elapsed


def evaluate_model(
    model_name: str,
    onnx_dir: str = "onnx_models",
    quantized_dir: str = "quantized_models",
    methods: list = None,
    num_eval_samples: int = 50,
):
    """Evaluate a model's INT8 quantization quality across calibration methods."""
    if methods is None:
        methods = CALIBRATION_METHODS

    fp_path = os.path.join(onnx_dir, f"{model_name}.onnx")
    if not os.path.exists(fp_path):
        print(f"[skip] FP model not found: {fp_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    fp_session = ort.InferenceSession(fp_path, sess_opts, providers=["CPUExecutionProvider"])
    input_name = fp_session.get_inputs()[0].name
    input_shape = fp_session.get_inputs()[0].shape
    input_shape = [1 if isinstance(d, str) else d for d in input_shape]

    data_reader = ImageNetCalibrationDataReader(
        model_name, num_samples=num_eval_samples,
    )

    fp_outputs = []
    for _ in range(num_eval_samples):
        sample = data_reader.get_next()
        if sample is None:
            break
        out = fp_session.run(None, sample)
        fp_outputs.append(out[0])
    fp_outputs = np.concatenate(fp_outputs, axis=0)

    fp_throughput = measure_throughput(fp_session, input_name, tuple(input_shape))
    print(f"  FP model throughput: {fp_throughput:.1f} img/s")

    results = {"fp_throughput": round(fp_throughput, 1)}

    for method in methods:
        for ch_mode in ["perchannel", "pertensor"]:
            q_name = f"{model_name}_int8_{method}_{ch_mode}.onnx"
            q_path = os.path.join(quantized_dir, q_name)
            if not os.path.exists(q_path):
                continue

            tag = f"{method}/{ch_mode}"
            print(f"\n  --- {tag} ---")

            try:
                q_session = ort.InferenceSession(
                    q_path, sess_opts, providers=["CPUExecutionProvider"],
                )

                data_reader.rewind()
                q_outputs = []
                for _ in range(num_eval_samples):
                    sample = data_reader.get_next()
                    if sample is None:
                        break
                    out = q_session.run(None, sample)
                    q_outputs.append(out[0])
                q_outputs = np.concatenate(q_outputs, axis=0)

                cos_sim = cosine_similarity(fp_outputs, q_outputs)
                mse_val = mse(fp_outputs, q_outputs)
                top1_agr = top1_agreement(fp_outputs, q_outputs)
                top5_agr = top5_agreement(fp_outputs, q_outputs)
                q_throughput = measure_throughput(q_session, input_name, tuple(input_shape))

                has_nan = bool(np.any(np.isnan(q_outputs)) or np.any(np.isinf(q_outputs)))

                print(f"  Cosine similarity:  {cos_sim:.6f}")
                print(f"  MSE:                {mse_val:.6f}")
                print(f"  Top-1 agreement:    {top1_agr:.2%}")
                print(f"  Top-5 agreement:    {top5_agr:.2%}")
                print(f"  INT8 throughput:    {q_throughput:.1f} img/s")
                print(f"  Has NaN/Inf:        {has_nan}")

                results[tag] = {
                    "cosine_sim": round(cos_sim, 6),
                    "mse": round(mse_val, 6),
                    "top1_agreement": round(top1_agr, 4),
                    "top5_agreement": round(top5_agr, 4),
                    "throughput": round(q_throughput, 1),
                    "has_nan": has_nan,
                    "size_mb": round(os.path.getsize(q_path) / (1024 * 1024), 1),
                }

            except Exception as e:
                print(f"  [FAIL] {e}")
                results[tag] = {"status": "failed", "error": str(e)}

    return results


def print_summary_table(all_results: dict):
    """Print a comparison table across models and methods."""
    print(f"\n{'='*80}")
    print("SUMMARY: Cosine Similarity (FP vs INT8)")
    print(f"{'='*80}")

    header = f"{'Model':<30}"
    methods_ch = []
    for m in CALIBRATION_METHODS:
        for ch in ["perchannel", "pertensor"]:
            tag = f"{m}/{ch}"
            methods_ch.append(tag)
            short = f"{m[:4]}/{'pch' if ch == 'perchannel' else 'pt'}"
            header += f" {short:>10}"
    print(header)
    print("-" * len(header))

    for model_name, results in all_results.items():
        if results is None:
            continue
        row = f"{model_name:<30}"
        for tag in methods_ch:
            if tag in results and isinstance(results[tag], dict) and "cosine_sim" in results[tag]:
                val = results[tag]["cosine_sim"]
                row += f" {val:>10.4f}"
            else:
                row += f" {'---':>10}"
        print(row)

    print(f"\n{'='*80}")
    print("SUMMARY: Top-1 Agreement (FP vs INT8)")
    print(f"{'='*80}")

    print(header)
    print("-" * len(header))

    for model_name, results in all_results.items():
        if results is None:
            continue
        row = f"{model_name:<30}"
        for tag in methods_ch:
            if tag in results and isinstance(results[tag], dict) and "top1_agreement" in results[tag]:
                val = results[tag]["top1_agreement"]
                row += f" {val:>9.1%} "
            else:
                row += f" {'---':>10}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate INT8 quantized models")
    parser.add_argument("--model", type=str, help="Model to evaluate")
    parser.add_argument("--methods", type=str, nargs="+", default=CALIBRATION_METHODS)
    parser.add_argument("--onnx-dir", type=str, default="onnx_models")
    parser.add_argument("--quantized-dir", type=str, default="quantized_models")
    parser.add_argument("--num-eval-samples", type=int, default=50)
    parser.add_argument("--sweep", action="store_true", help="Evaluate all representative models")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    all_results = {}
    if args.sweep:
        for name in REPRESENTATIVE_MODELS:
            all_results[name] = evaluate_model(
                name, args.onnx_dir, args.quantized_dir,
                args.methods, args.num_eval_samples,
            )
    elif args.model:
        all_results[args.model] = evaluate_model(
            args.model, args.onnx_dir, args.quantized_dir,
            args.methods, args.num_eval_samples,
        )
    else:
        parser.print_help()
        return

    print_summary_table(all_results)

    out_path = args.output or os.path.join(args.quantized_dir, "evaluation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
