"""Authoritative INT8 calibration-method experiment on timm models with REAL ImageNet data.

For each model and calibration method, measures real top-1 accuracy (vs ground truth),
top-1 agreement vs FP, and output cosine similarity.

Backends / methods actually supported (verified against installed versions:
modelopt 0.44.0, onnxruntime 1.26.0):
  - ModelOpt ONNX int8:  'entropy' (default), 'max'
  - ONNX Runtime static:  'minmax', 'entropy', 'percentile'  (per-channel or per-tensor)

IMPORTANT: ModelOpt's quantize() pollutes global state in onnxruntime.quantization,
breaking any subsequent ORT quantize_static in the SAME process. Each quantization is
therefore run in an isolated subprocess (this same file invoked with --worker).

Usage:
    python run_experiment.py --models regnetx_002 mobilenetv3_large_100
    python run_experiment.py --models regnetx_002 --calib 150 --eval 250
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import onnxruntime as ort

from real_data import load_image_label_pairs, RealImageNetDataReader

warnings.filterwarnings("ignore")
ort.set_default_logger_severity(3)

ONNX_DIR = "onnx_models"
QUANT_DIR = "quantized_models"

# (label, backend, method, per_channel, high_precision_dtype, percentile)
#
# All methods here produce FP32-scale QDQ graphs that the onnxruntime CPU EP
# executes correctly, so the measured accuracy is faithful. We deliberately do
# NOT include modelopt high_precision_dtype=fp16/bf16 variants: the CPU EP cannot
# execute FP16-scale QDQ (gives garbage / NOT_IMPLEMENTED), which is an evaluation
# artifact, not a real accuracy loss (verified by casting such a model back to
# fp32 -> accuracy fully recovers). Those must be validated on GPU/TensorRT.
#
# The real, CPU-faithful lever for these models is the ACTIVATION calibration
# method (minmax vs entropy vs percentile) plus selective quantization.
EXPERIMENTS = [
    ("ort/minmax-pc", "ort", "minmax", True, "fp32", 99.99),       # baseline (≈ORT/Holmes default)
    ("ort/entropy-pc", "ort", "entropy", True, "fp32", 99.99),
    ("ort/percentile-99.99", "ort", "percentile", True, "fp32", 99.99),
    ("ort/percentile-99.9", "ort", "percentile", True, "fp32", 99.9),
    ("ort/percentile-99.0", "ort", "percentile", True, "fp32", 99.0),
    ("modelopt/entropy", "modelopt", "entropy", True, "fp32", 99.99),  # selective + entropy
    ("modelopt/max", "modelopt", "max", True, "fp32", 99.99),
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def export_if_needed(model_name):
    path = os.path.join(ONNX_DIR, f"{model_name}.onnx")
    if os.path.exists(path):
        return path
    import timm
    import torch

    os.makedirs(ONNX_DIR, exist_ok=True)
    model = timm.create_model(model_name, pretrained=True).eval()
    size = model.default_cfg.get("input_size", (3, 224, 224))
    dummy = torch.randn(1, *size)
    # The torch 2.x dynamo exporter fails to decompose some graphs (e.g. beit's
    # attention). Fall back to the legacy TorchScript exporter (dynamo=False),
    # which is more robust for these timm models.
    try:
        torch.onnx.export(
            model, dummy, path, opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
    except Exception:
        torch.onnx.export(
            model, dummy, path, opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    return path


def get_input_name(onnx_path):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.get_inputs()[0].name


def build_calib_samples(model_name, calib_count, input_name):
    pairs = load_image_label_pairs()[:calib_count]
    reader = RealImageNetDataReader(model_name, pairs, input_name=input_name)
    return [{input_name: t} for t in reader.tensors]


# --------------------------------------------------------------------------- #
# Worker: performs exactly ONE quantization in an isolated process
# --------------------------------------------------------------------------- #
class _ListReader:
    def __init__(self, samples):
        self.samples = samples
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        out = self.samples[self.index]
        self.index += 1
        return out

    def get_first(self):
        return self.samples[0] if self.samples else None

    def rewind(self):
        self.index = 0


def worker_main(args):
    onnx_path = args.onnx
    out_path = args.out
    model_name = args.model
    input_name = get_input_name(onnx_path)
    calib_samples = build_calib_samples(model_name, args.calib, input_name)

    if args.backend == "modelopt":
        import modelopt.onnx.quantization as moq

        moq.quantize(
            onnx_path,
            quantize_mode="int8",
            calibration_method=args.method,        # 'entropy' or 'max'
            calibration_data_reader=_ListReader(calib_samples),
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            calibration_eps=["cpu"],
            high_precision_dtype=args.high_precision,
            use_zero_point=args.zero_point,        # True => asymmetric INT8
            output_path=out_path,
        )
    else:
        from onnxruntime.quantization import (
            CalibrationMethod, QuantType, quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process

        method_map = {
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile,
            "distribution": CalibrationMethod.Distribution,
        }
        extra = {}
        if args.method == "percentile":
            extra["CalibPercentile"] = args.percentile
        # zero_point=True => asymmetric activations (weights stay symmetric/per-channel).
        extra["ActivationSymmetric"] = not args.zero_point
        extra["WeightSymmetric"] = True
        # Some graphs (mobilevit, convmixer, transformers) need shape inference +
        # optimization before static quantization, otherwise entropy/percentile
        # fail ("run pre-processing before quantization"). minmax tolerates its
        # absence; the others don't. Symbolic shape inference itself throws an
        # AssertionError on attention graphs (mobilevit), so fall back to
        # skip_symbolic_shape=True, which still does the optimization that the
        # quantizer needs.
        prepped = out_path + ".prep.onnx"
        for skip_sym in (False, True):
            try:
                quant_pre_process(onnx_path, prepped, skip_symbolic_shape=skip_sym)
                break
            except Exception:
                prepped = onnx_path  # last resort: raw model
        quantize_static(
            model_input=prepped,
            model_output=out_path,
            calibration_data_reader=_ListReader(calib_samples),
            calibrate_method=method_map[args.method],
            per_channel=args.per_channel,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            extra_options=extra,
        )


def spawn_quantize(model_name, onnx_path, out_path, backend, method, per_channel,
                   calib, high_precision, percentile):
    cmd = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--onnx", onnx_path, "--out", out_path, "--model", model_name,
        "--backend", backend, "--method", method, "--calib", str(calib),
        "--high-precision", high_precision, "--percentile", str(percentile),
    ]
    if per_channel:
        cmd.append("--per-channel")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        msg = tail[-1] if tail else f"exit {proc.returncode}"
        raise RuntimeError(msg)


# --------------------------------------------------------------------------- #
# Evaluation (pure inference, safe in parent process)
# --------------------------------------------------------------------------- #
def run_session_predictions(onnx_path, reader):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    reader.rewind()
    logits, t0, n = [], time.time(), 0
    while True:
        s = reader.get_next()
        if s is None:
            break
        logits.append(sess.run(None, s)[0])
        n += 1
    elapsed = (time.time() - t0) / max(n, 1)
    return np.concatenate(logits, axis=0), elapsed


def accuracy(logits, labels):
    return float(np.mean(np.argmax(logits, -1) == np.array(labels)))


def agreement(a, b):
    return float(np.mean(np.argmax(a, -1) == np.argmax(b, -1)))


def cosine(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_model(model_name, calib_count, eval_count):
    onnx_path = export_if_needed(model_name)
    input_name = get_input_name(onnx_path)

    pairs = load_image_label_pairs()
    eval_pairs = pairs[calib_count:calib_count + eval_count]
    eval_reader = RealImageNetDataReader(model_name, eval_pairs, input_name=input_name)

    fp_logits, fp_time = run_session_predictions(onnx_path, eval_reader)
    fp_acc = accuracy(fp_logits, eval_reader.labels)

    print(f"\n{'='*74}")
    print(f"{model_name}  | FP top-1: {fp_acc*100:.1f}%  ({fp_time*1000:.1f} ms/img, "
          f"{calib_count} calib / {eval_count} eval)")
    print(f"{'='*74}")
    print(f"{'method':<22}{'top1':>8}{'Δacc':>8}{'agree':>8}{'cos':>9}{'qnt_s':>8}")
    print("-" * 74)

    results = {"fp_top1": round(fp_acc, 4), "fp_ms_per_img": round(fp_time * 1000, 2),
               "methods": {}}
    os.makedirs(QUANT_DIR, exist_ok=True)

    for label, backend, method, per_channel, high_precision, percentile in EXPERIMENTS:
        safe = label.replace("/", "_")
        out_path = os.path.join(QUANT_DIR, f"{model_name}_{safe}.onnx")
        try:
            t0 = time.time()
            spawn_quantize(model_name, onnx_path, out_path, backend, method,
                           per_channel, calib_count, high_precision, percentile)
            qnt_s = time.time() - t0

            q_logits, _ = run_session_predictions(out_path, eval_reader)
            bad = not np.isfinite(q_logits).all()
            q_acc = accuracy(q_logits, eval_reader.labels)
            agr = agreement(fp_logits, q_logits)
            cos = cosine(fp_logits, q_logits)
            results["methods"][label] = {
                "top1": round(q_acc, 4), "delta_acc": round(q_acc - fp_acc, 4),
                "agreement": round(agr, 4), "cosine": round(cos, 4),
                "quant_s": round(qnt_s, 1), "has_naninf": bool(bad),
            }
            flag = "  !!NaN/Inf" if bad else ""
            print(f"{label:<22}{q_acc*100:>7.1f}%{(q_acc-fp_acc)*100:>+7.1f}%"
                  f"{agr*100:>7.1f}%{cos:>9.4f}{qnt_s:>8.1f}{flag}")
        except Exception as e:
            results["methods"][label] = {"error": str(e)[:200]}
            print(f"{label:<22}  FAILED: {str(e)[:60]}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--onnx"); ap.add_argument("--out"); ap.add_argument("--model")
    ap.add_argument("--backend"); ap.add_argument("--method")
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--high-precision", default="fp32")
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--zero-point", action="store_true",
                    help="asymmetric INT8 activations (zero_point != 0)")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--calib", type=int, default=150)
    ap.add_argument("--eval", type=int, default=250)
    ap.add_argument("--output", type=str, default="experiment_results.json")
    ap.add_argument("--resume", action="store_true",
                    help="load existing --output and skip models already completed")
    args = ap.parse_args()

    if args.worker:
        worker_main(args)
        return

    if not args.models:
        ap.error("--models is required")

    pairs = load_image_label_pairs()
    if len(pairs) < args.calib + 1:
        raise RuntimeError(f"Only {len(pairs)} real images available")
    print(f"Real images: {len(pairs)} total -> {args.calib} calib, "
          f"{min(args.eval, len(pairs)-args.calib)} eval")

    # Resume: keep previously completed models, skip them this run.
    all_results = {}
    if args.resume and os.path.exists(args.output):
        all_results = json.load(open(args.output))
        done = [m for m, r in all_results.items()
                if isinstance(r, dict) and "fp_top1" in r]
        print(f"Resuming: {len(done)} models already in {args.output}, will skip them")

    for m in args.models:
        if args.resume and m in all_results and "fp_top1" in all_results.get(m, {}):
            print(f"[{m}] skip (already done)")
            continue
        try:
            all_results[m] = run_model(m, args.calib, args.eval)
        except Exception as e:
            print(f"[{m}] FAILED: {e}")
            all_results[m] = {"error": str(e)}
        # Incremental save after EVERY model so a crash never loses prior work.
        json.dump(all_results, open(args.output, "w"), indent=2)
        print(f"  [saved {len([1 for r in all_results.values() if 'fp_top1' in r])} models -> {args.output}]")

    print(f"\nDone -> {args.output}")


if __name__ == "__main__":
    main()
