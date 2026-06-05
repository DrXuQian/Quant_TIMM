"""Isolate the effect of asymmetric INT8 (zero_point != 0) on accuracy.

For each model and calibration method, quantizes both symmetric (zero_point=0)
and asymmetric (zero_point!=0) and reports real top-1, so we can see whether
asymmetric activation quantization helps the long-tailed-activation models
*independently* of mixed precision / calibration choice.

Each quantization runs in an isolated subprocess (run_experiment.py --worker),
which already supports --zero-point.

Usage:
    python experiment_zeropoint.py --models lcnet_050 efficientnet_b0 mobilevit_s
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

import run_experiment as R
from real_data import load_image_label_pairs, RealImageNetDataReader

ort.set_default_logger_severity(3)

# (label, backend, method, percentile, zero_point)
EXPERIMENTS = [
    ("ort/minmax sym", "ort", "minmax", 99.99, False),
    ("ort/minmax ASYM", "ort", "minmax", 99.99, True),
    ("ort/pct99.99 sym", "ort", "percentile", 99.99, False),
    ("ort/pct99.99 ASYM", "ort", "percentile", 99.99, True),
    ("modelopt/entropy sym", "modelopt", "entropy", 99.99, False),
    ("modelopt/entropy ASYM", "modelopt", "entropy", 99.99, True),
]


def spawn(model_name, onnx_path, out_path, backend, method, percentile, zero_point, calib):
    cmd = [
        sys.executable, os.path.abspath("run_experiment.py"), "--worker",
        "--onnx", onnx_path, "--out", out_path, "--model", model_name,
        "--backend", backend, "--method", method, "--calib", str(calib),
        "--high-precision", "fp32", "--percentile", str(percentile), "--per-channel",
    ]
    if zero_point:
        cmd.append("--zero-point")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        raise RuntimeError(tail[-1] if tail else f"exit {proc.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["lcnet_050", "efficientnet_b0", "mobilevit_s"])
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--eval", type=int, default=250)
    args = ap.parse_args()

    pairs = load_image_label_pairs()
    eval_pairs = pairs[args.calib:args.calib + args.eval]
    os.makedirs("quantized_models", exist_ok=True)

    for model in args.models:
        onnx_path = R.export_if_needed(model)
        input_name = R.get_input_name(onnx_path)
        ev = RealImageNetDataReader(model, eval_pairs, input_name=input_name)
        fp_logits, _ = R.run_session_predictions(onnx_path, ev)
        fp_acc = R.accuracy(fp_logits, ev.labels)

        print(f"\n{'='*60}\n{model}  | FP top-1: {fp_acc*100:.1f}%  "
              f"(calib {args.calib})\n{'='*60}")
        print(f"{'method':<24}{'top1':>8}{'Δacc':>9}")
        print("-" * 60)
        for label, backend, method, pct, zp in EXPERIMENTS:
            safe = label.replace("/", "_").replace(" ", "_")
            out = f"quantized_models/{model}_zp_{safe}.onnx"
            try:
                spawn(model, onnx_path, out, backend, method, pct, zp, args.calib)
                q, _ = R.run_session_predictions(out, ev)
                acc = R.accuracy(q, ev.labels)
                print(f"{label:<24}{acc*100:>7.1f}%{(acc-fp_acc)*100:>+8.1f}%")
            except Exception as e:
                print(f"{label:<24}  FAILED: {str(e)[:30]}")


if __name__ == "__main__":
    main()
