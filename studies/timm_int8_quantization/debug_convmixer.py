#!/usr/bin/env python3
"""Debug convmixer_768_32: try multiple calibration algorithms with int8_pc.

Tests: max, mse, histogram (entropy), smoothquant — all with per-channel
activation (axis=1). No mixed precision (all layers quantized).

Usage:
    python debug_convmixer.py
"""

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

MODEL = "convmixer_768_32"
CALIB_DIR = "imagenet_calib"
EVAL_DIR = "imagenet_val"


class ImageLabelDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_labels_json(data_dir):
    meta = json.load(open(os.path.join(data_dir, "labels.json")))
    return [(os.path.join(data_dir, m["file"]), int(m["label"])) for m in meta]


def build_config(algorithm, pc_activation=True):
    """Build INT8 config with a given calibration algorithm."""
    if algorithm == "smoothquant":
        cfg = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    else:
        cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        cfg["algorithm"] = algorithm  # "max", "mse", or "histogram"

    if pc_activation:
        for entry in cfg["quant_cfg"]:
            if entry.get("quantizer_name") == "*input_quantizer":
                entry["cfg"]["axis"] = 1
    return cfg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch_t, batch_l in loader:
        preds = model(batch_t.to(device)).argmax(dim=-1).cpu()
        correct += int((preds == batch_l).sum())
        total += batch_l.size(0)
    return correct / total


def main():
    device = torch.device("cuda")

    calib_items = load_labels_json(CALIB_DIR)[:512]
    eval_items = load_labels_json(EVAL_DIR)

    model_tmp = timm.create_model(MODEL, pretrained=False)
    cfg = resolve_data_config(model_tmp.default_cfg)
    transform = create_transform(**cfg, is_training=False)
    del model_tmp

    calib_loader = DataLoader(
        ImageLabelDataset(calib_items, transform),
        batch_size=256, num_workers=8, pin_memory=True,
    )
    eval_loader = DataLoader(
        ImageLabelDataset(eval_items, transform),
        batch_size=256, num_workers=8, pin_memory=True,
    )

    # FP32 baseline
    model = timm.create_model(MODEL, pretrained=True).eval().to(device)
    fp_acc = evaluate(model, eval_loader, device)
    print(f"FP32 baseline: {fp_acc*100:.2f}%\n")
    del model
    torch.cuda.empty_cache()

    # Strategies to test
    strategies = [
        ("max + pc_act",        "max",         True),
        ("mse + pc_act",        "mse",         True),
        ("histogram + pc_act",  "histogram",   True),
        ("smoothquant + pc_act","smoothquant",  True),
        ("max + pt_act",        "max",         False),   # per-tensor activation (baseline)
        ("mse + pt_act",        "mse",         False),
        ("histogram + pt_act",  "histogram",   False),
    ]

    results = {}
    for label, algo, pc_act in strategies:
        print(f"\n--- {label} ---", flush=True)
        model = timm.create_model(MODEL, pretrained=True).eval().to(device)
        config = build_config(algo, pc_activation=pc_act)

        def forward_loop(m):
            with torch.no_grad():
                for batch_t, _ in calib_loader:
                    m(batch_t.to(device))

        t0 = time.time()
        try:
            qmodel = mtq.quantize(model, config, forward_loop=forward_loop)
            quant_s = time.time() - t0
            q_acc = evaluate(qmodel, eval_loader, device)
            delta = q_acc - fp_acc
            print(f"  {label}: {q_acc*100:.2f}%  (Δ={delta*100:+.2f}%, quant={quant_s:.1f}s)",
                  flush=True)
            results[label] = {
                "top1": round(q_acc, 4),
                "delta": round(delta, 4),
                "quant_s": round(quant_s, 1),
            }
        except Exception as e:
            print(f"  {label}: FAILED — {e}", flush=True)
            results[label] = {"error": str(e)[:200]}

        del model
        if 'qmodel' in dir():
            del qmodel
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*65}")
    print(f"convmixer_768_32  FP32: {fp_acc*100:.2f}%")
    print(f"{'='*65}")
    print(f"{'Strategy':<25}{'Top-1':>8}{'Δ':>10}{'Quant':>8}")
    print("-" * 65)
    for label, _, _ in strategies:
        r = results.get(label, {})
        if "top1" in r:
            print(f"{label:<25}{r['top1']*100:>7.2f}%{r['delta']*100:>+9.2f}%{r['quant_s']:>7.1f}s")
        else:
            print(f"{label:<25}  FAILED")
    print(f"{'='*65}")

    results["fp_top1"] = round(fp_acc, 4)
    json.dump(results, open("results/debug_convmixer.json", "w"), indent=2)
    print(f"Saved -> results/debug_convmixer.json")


if __name__ == "__main__":
    main()
