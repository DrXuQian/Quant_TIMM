#!/usr/bin/env python3
"""INT8 per-channel activation quantization via modelopt.torch on 9 timm models.

Replicates the 'int8_pc' strategy: per-channel input activation (axis=1 NCHW)
+ per-channel weights (axis=0) + max calibration.

Calibration: training-set subset (imagenet_calib/).
Evaluation:  full 50k validation set (imagenet_val/).

Uses DataLoader with multiple workers for fast data loading.

Usage:
    python run_torch_int8_pc.py
    python run_torch_int8_pc.py --device cuda --batch-size 128
    python run_torch_int8_pc.py --export-onnx
"""

import argparse
import copy
import json
import os
import time
import warnings

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform
from torch.utils.data import Dataset, DataLoader

import modelopt.torch.quantization as mtq

warnings.filterwarnings("ignore")

MODELS = [
    "beit_base_patch16_224",
    "adv_inception_v3",
    "mobilevit_s",
    "rexnet_100",
    "hardcorenas_a",
    "lcnet_050",
    "efficientnet_b0",
    "convmixer_768_32",
    "repvgg_a2",
]

CALIB_DIR = "imagenet_calib"
EVAL_DIR = "imagenet_val"


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ImageLabelDataset(Dataset):
    """Dataset that reads (filepath, label) pairs and applies a transform."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label


def load_labels_json(data_dir):
    """Load [(filepath, label), ...] from labels.json."""
    labels_json = os.path.join(data_dir, "labels.json")
    meta = json.load(open(labels_json))
    return [(os.path.join(data_dir, m["file"]), int(m["label"])) for m in meta]


def get_transform(model_name):
    """Get timm preprocessing transform for a model."""
    model_tmp = timm.create_model(model_name, pretrained=False)
    cfg = resolve_data_config(model_tmp.default_cfg)
    transform = create_transform(**cfg, is_training=False)
    del model_tmp
    return transform


# --------------------------------------------------------------------------- #
# Quantization config
# --------------------------------------------------------------------------- #
def build_int8_pc_config():
    """INT8 config with per-channel activation quantization (axis=1 for NCHW)."""
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    for entry in cfg["quant_cfg"]:
        if entry.get("quantizer_name") == "*input_quantizer":
            entry["cfg"]["axis"] = 1  # per-channel on C dimension (NCHW)
    cfg["algorithm"] = "max"
    return cfg


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a DataLoader, return top-1 accuracy."""
    model.eval()
    correct = 0
    total = 0
    for batch_t, batch_l in loader:
        logits = model(batch_t.to(device))
        preds = logits.argmax(dim=-1).cpu()
        correct += int((preds == batch_l).sum())
        total += batch_l.size(0)
        if total % 10000 < batch_t.size(0):
            print(f"    eval {total}/{len(loader.dataset)}  "
                  f"running acc: {correct/total*100:.2f}%", flush=True)
    return correct / total


# --------------------------------------------------------------------------- #
# Per-model runner
# --------------------------------------------------------------------------- #
def run_single_model(model_name, calib_loader, eval_loader, device,
                     export_onnx=False, onnx_dir="onnx_models_torch_int8pc"):
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}", flush=True)

    model = timm.create_model(model_name, pretrained=True).eval().to(device)

    # FP32 baseline
    t0 = time.time()
    fp_acc = evaluate(model, eval_loader, device)
    fp_time = time.time() - t0
    print(f"  FP32 top-1: {fp_acc*100:.2f}%  ({fp_time:.1f}s, "
          f"{fp_time/len(eval_loader.dataset)*1000:.1f} ms/img)", flush=True)

    # Quantize
    config = build_int8_pc_config()

    def forward_loop(m):
        with torch.no_grad():
            for batch_t, _ in calib_loader:
                m(batch_t.to(device))

    t0 = time.time()
    qmodel = mtq.quantize(model, config, forward_loop=forward_loop)
    quant_time = time.time() - t0
    print(f"  Quantization: {quant_time:.1f}s", flush=True)

    # INT8_PC evaluation
    t0 = time.time()
    q_acc = evaluate(qmodel, eval_loader, device)
    q_time = time.time() - t0
    delta = q_acc - fp_acc
    print(f"  INT8_PC top-1: {q_acc*100:.2f}%  (Δ={delta*100:+.2f}%)  "
          f"({q_time:.1f}s)", flush=True)

    result = {
        "fp_top1": round(fp_acc, 4),
        "int8_pc_top1": round(q_acc, 4),
        "delta": round(delta, 4),
        "quant_s": round(quant_time, 1),
        "fp_eval_s": round(fp_time, 1),
        "int8_eval_s": round(q_time, 1),
    }

    if export_onnx:
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_path = os.path.join(onnx_dir, f"{model_name}_int8pc.onnx")
        input_size = model.default_cfg.get("input_size", (3, 224, 224))
        dummy = torch.randn(1, *input_size, device=device)
        try:
            torch.onnx.export(
                qmodel, dummy, onnx_path, opset_version=17,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
            sz = os.path.getsize(onnx_path) / 1e6
            print(f"  ONNX export: {onnx_path} ({sz:.1f} MB)")
            result["onnx_path"] = onnx_path
        except Exception as e:
            print(f"  ONNX export FAILED: {e}")
            result["onnx_export_error"] = str(e)[:200]

    del model, qmodel
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--calib-dir", default=CALIB_DIR)
    ap.add_argument("--eval-dir", default=EVAL_DIR)
    ap.add_argument("--calib-count", type=int, default=512)
    ap.add_argument("--eval-count", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--output", default="results/torch_int8_pc_50k.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    calib_items = load_labels_json(args.calib_dir)[:args.calib_count]
    eval_items = load_labels_json(args.eval_dir)
    if args.eval_count:
        eval_items = eval_items[:args.eval_count]

    print(f"Calibration: {len(calib_items)} from {args.calib_dir}")
    print(f"Evaluation:  {len(eval_items)} from {args.eval_dir}")
    print(f"Device: {device}, batch={args.batch_size}, workers={args.num_workers}")
    print(f"Strategy: INT8 per-channel activation (axis=1) + max calibration")

    all_results = {}
    if args.resume and os.path.exists(args.output):
        all_results = json.load(open(args.output))
        done = [m for m in all_results if "fp_top1" in all_results[m]]
        print(f"Resuming: {len(done)} models done, will skip")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for model_name in args.models:
        if (args.resume and model_name in all_results
                and "fp_top1" in all_results.get(model_name, {})):
            print(f"\n[{model_name}] skip (already done)")
            continue

        try:
            transform = get_transform(model_name)

            calib_ds = ImageLabelDataset(calib_items, transform)
            eval_ds = ImageLabelDataset(eval_items, transform)

            calib_loader = DataLoader(
                calib_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            eval_loader = DataLoader(
                eval_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )

            result = run_single_model(
                model_name, calib_loader, eval_loader, device,
                export_onnx=args.export_onnx,
            )
            all_results[model_name] = result

        except Exception as e:
            print(f"\n[{model_name}] FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

        torch.cuda.empty_cache()
        json.dump(all_results, open(args.output, "w"), indent=2)
        n = len([1 for r in all_results.values() if "fp_top1" in r])
        print(f"  [saved {n}/{len(args.models)} -> {args.output}]")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Model':<28}{'FP32':>8}{'INT8_PC':>10}{'Δ':>10}")
    print("-" * 70)
    for m in args.models:
        r = all_results.get(m, {})
        if "fp_top1" in r:
            print(f"{m:<28}{r['fp_top1']*100:>7.2f}%{r['int8_pc_top1']*100:>9.2f}%"
                  f"{r['delta']*100:>+9.2f}%")
        else:
            print(f"{m:<28}  {'FAILED':>8}")
    print(f"{'='*70}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
