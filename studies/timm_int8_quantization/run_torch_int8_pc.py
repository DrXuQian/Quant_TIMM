#!/usr/bin/env python3
"""INT8 quantization with per-channel activation on depthwise convolutions.

Strategy:
  - Depthwise Conv2d: per-channel activation (axis=1) + per-channel weight (axis=0)
  - Regular Conv2d:   per-tensor activation (axis=None) + per-channel weight (axis=0)
  - Linear:           per-tensor activation (axis=None) + per-channel weight (axis=0)

Per-channel activation only benefits depthwise conv (each channel computed
independently). Regular conv mixes all input channels so per-channel activation
is unnecessary and breaks some backends (e.g. Holmes).

Data sources:
  A) Local dirs:       --calib-dir imagenet_calib  --eval-dir imagenet_val
  B) HuggingFace:      --hf-dataset ILSVRC/imagenet-1k

Usage:
    python run_torch_int8_pc.py --models efficientnet_b0 --device cuda
    python run_torch_int8_pc.py --models convmixer_768_32 --calibrator percentile --percentile 99.9
    python run_torch_int8_pc.py --hf-dataset ILSVRC/imagenet-1k --models efficientnet_b0
    python run_torch_int8_pc.py --models efficientnet_b0 --act-mode per-tensor  # no DW per-channel
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
from modelopt.torch.quantization.nn import TensorQuantizer

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
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class HFImageNetDataset(Dataset):
    def __init__(self, hf_dataset_id, split, transform, max_count=None):
        from datasets import load_dataset
        print(f"  Loading HF {hf_dataset_id} [{split}] ...", flush=True)
        ds = load_dataset(hf_dataset_id, split=split, streaming=True)
        self.images, self.labels = [], []
        for i, ex in enumerate(ds):
            if max_count and i >= max_count:
                break
            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            self.images.append(img)
            self.labels.append(int(ex["label"]))
            if (i + 1) % 5000 == 0:
                print(f"    loaded {i+1}/{max_count or '?'} images", flush=True)
        self.transform = transform
        print(f"  Loaded {len(self.images)} images from HF", flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


def load_labels_json(data_dir):
    labels_json = os.path.join(data_dir, "labels.json")
    meta = json.load(open(labels_json))
    return [(os.path.join(data_dir, m["file"]), int(m["label"])) for m in meta]


def get_transform(model_name):
    model_tmp = timm.create_model(model_name, pretrained=False)
    cfg = resolve_data_config(model_tmp.default_cfg)
    transform = create_transform(**cfg, is_training=False)
    del model_tmp
    return transform


# --------------------------------------------------------------------------- #
# Find depthwise conv layers
# --------------------------------------------------------------------------- #
def find_depthwise_conv_names(model):
    """Return set of module names that are depthwise Conv2d (groups == in_channels > 1)."""
    names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and mod.groups == mod.in_channels and mod.groups > 1:
            names.add(name)
    return names


# --------------------------------------------------------------------------- #
# Quantization config
# --------------------------------------------------------------------------- #
def build_int8_config(model, act_mode="dw-per-channel"):
    """Build INT8 quantization config.

    act_mode:
      "dw-per-channel" — depthwise conv activation per-channel (axis=1),
                          everything else per-tensor (default, best accuracy)
      "per-channel"    — ALL activations per-channel (axis=1)
                          (may break some backends)
      "per-tensor"     — ALL activations per-tensor (axis=None)
                          (most compatible, lower accuracy on DW-heavy models)
    """
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    cfg["algorithm"] = "max"

    if act_mode == "per-channel":
        # All input_quantizer → axis=1
        for entry in cfg["quant_cfg"]:
            if entry.get("quantizer_name") == "*input_quantizer":
                entry["cfg"]["axis"] = 1

    elif act_mode == "dw-per-channel":
        # Default: per-tensor for all (axis=None already in INT8_DEFAULT_CFG)
        # Then append per-name overrides for depthwise convs
        dw_names = find_depthwise_conv_names(model)
        for dw in dw_names:
            cfg["quant_cfg"].append({
                "quantizer_name": f"{dw}.input_quantizer",
                "cfg": {"num_bits": 8, "axis": 1},
            })
        if dw_names:
            print(f"  DW per-channel activation on {len(dw_names)} layers", flush=True)

    # act_mode == "per-tensor": no changes needed (default is axis=None)

    return cfg


# --------------------------------------------------------------------------- #
# Per-channel percentile re-calibration
# --------------------------------------------------------------------------- #
def recalibrate_percentile(qmodel, calib_loader, device, percentile=99.9,
                           num_bins=2048):
    """Re-calibrate input quantizers using per-channel percentile.

    Works on quantizers that have axis != None (i.e. depthwise conv input
    quantizers when act_mode="dw-per-channel"). Per-tensor quantizers are
    left unchanged.
    """
    # Only re-calibrate per-channel quantizers (axis is not None)
    input_qs = [(n, q) for n, q in qmodel.named_modules()
                if isinstance(q, TensorQuantizer) and "input" in n
                and hasattr(q, "_amax") and q.axis is not None]
    if not input_qs:
        print("  [warn] no per-channel input quantizers found, skipping percentile")
        return

    orig_amaxes = {n: q._amax.data.clone() for n, q in input_qs}

    # Try patched HistogramCalibrator
    try:
        from modelopt.torch.quantization.calib.histogram import HistogramCalibrator
        HistogramCalibrator(num_bits=8, axis=1, num_bins=16)
        use_histogram = True
    except (NotImplementedError, TypeError):
        try:
            import patch_histogram_perchannel  # noqa: F401
            from modelopt.torch.quantization.calib.histogram import HistogramCalibrator
            HistogramCalibrator(num_bits=8, axis=1, num_bins=16)
            use_histogram = True
        except Exception:
            use_histogram = False

    if use_histogram:
        for _, q in input_qs:
            q._calibrator = HistogramCalibrator(num_bits=8, axis=q.axis,
                                                num_bins=num_bins)
            q.enable_calib()
            q.disable_quant()

        with torch.no_grad():
            for batch_t, _ in calib_loader:
                qmodel(batch_t.to(device))

        replaced = 0
        for n, q in input_qs:
            new_amax = q._calibrator.compute_amax("percentile",
                                                   percentile=percentile)
            if new_amax is not None:
                q._amax.data.copy_(new_amax.to(q._amax.device))
                replaced += 1
            else:
                q._amax.data.copy_(orig_amaxes[n])
            q.disable_calib()
            q.enable_quant()
        print(f"  Percentile {percentile}: {replaced}/{len(input_qs)} DW quantizers "
              f"(histogram path)", flush=True)
    else:
        # Fallback: torch.quantile
        print("  [info] using torch.quantile fallback", flush=True)
        activations = {n: [] for n, _ in input_qs}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                if input and isinstance(input[0], torch.Tensor):
                    activations[name].append(input[0].detach())
            return hook_fn

        for _, q in input_qs:
            q.disable_quant()
        parent_map = {}
        for n, q in input_qs:
            parent_name = n.rsplit(".", 1)[0]
            parent_mod = dict(qmodel.named_modules()).get(parent_name)
            if parent_mod and parent_name not in parent_map:
                parent_map[parent_name] = n
                hooks.append(parent_mod.register_forward_hook(make_hook(n)))

        with torch.no_grad():
            for batch_t, _ in calib_loader:
                qmodel(batch_t.to(device))

        for h in hooks:
            h.remove()

        replaced = 0
        for n, q in input_qs:
            if not activations[n]:
                q._amax.data.copy_(orig_amaxes[n])
            else:
                cat = torch.cat(activations[n], dim=0).abs()
                axis = q.axis if q.axis is not None else 1
                flat = cat.movedim(axis, 0).flatten(start_dim=1)
                pct_amax = torch.quantile(flat.float(), percentile / 100.0, dim=1)
                shape = [1] * cat.ndim
                shape[axis] = pct_amax.numel()
                q._amax.data.copy_(pct_amax.reshape(shape).to(q._amax.device))
                replaced += 1
            q.enable_quant()
        print(f"  Percentile {percentile}: {replaced}/{len(input_qs)} DW quantizers "
              f"(torch.quantile fallback)", flush=True)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch_t, batch_l in loader:
        preds = model(batch_t.to(device)).argmax(dim=-1).cpu()
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
                     act_mode="dw-per-channel", calibrator="max",
                     percentile=99.9, export_onnx=False,
                     onnx_dir="onnx_models_torch_int8pc"):
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}", flush=True)

    model = timm.create_model(model_name, pretrained=True).eval().to(device)

    # FP32 baseline
    t0 = time.time()
    fp_acc = evaluate(model, eval_loader, device)
    fp_time = time.time() - t0
    print(f"  FP32 top-1: {fp_acc*100:.2f}%  ({fp_time:.1f}s)", flush=True)

    # Build config (inspects model for depthwise conv names)
    config = build_int8_config(model, act_mode=act_mode)

    def forward_loop(m):
        with torch.no_grad():
            for batch_t, _ in calib_loader:
                m(batch_t.to(device))

    t0 = time.time()
    qmodel = mtq.quantize(model, config, forward_loop=forward_loop)
    quant_time = time.time() - t0
    print(f"  Quantization (max): {quant_time:.1f}s", flush=True)

    # Optional: percentile re-calibration (only affects DW per-channel quantizers)
    if calibrator == "percentile":
        recalibrate_percentile(qmodel, calib_loader, device,
                               percentile=percentile)

    # Evaluate
    tag = f"INT8({act_mode},{calibrator})"
    t0 = time.time()
    q_acc = evaluate(qmodel, eval_loader, device)
    q_time = time.time() - t0
    delta = q_acc - fp_acc
    print(f"  {tag}: {q_acc*100:.2f}%  (Δ={delta*100:+.2f}%)  "
          f"({q_time:.1f}s)", flush=True)

    result = {
        "fp_top1": round(fp_acc, 4),
        "int8_top1": round(q_acc, 4),
        "delta": round(delta, 4),
        "act_mode": act_mode,
        "calibrator": calibrator,
        "quant_s": round(quant_time, 1),
    }

    if export_onnx:
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_path = os.path.join(onnx_dir, f"{model_name}_int8.onnx")
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

    del model, qmodel
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="INT8 quantization with per-channel activation on depthwise conv")

    data = ap.add_argument_group("data source")
    data.add_argument("--calib-dir", default=CALIB_DIR)
    data.add_argument("--eval-dir", default=EVAL_DIR)
    data.add_argument("--hf-dataset", default=None,
                      help="HuggingFace dataset id (e.g. ILSVRC/imagenet-1k)")

    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--calib-count", type=int, default=512)
    ap.add_argument("--eval-count", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--act-mode", choices=["dw-per-channel", "per-channel", "per-tensor"],
                    default="dw-per-channel",
                    help="activation quantization mode: "
                         "dw-per-channel = per-channel only on depthwise conv (default), "
                         "per-channel = all layers per-channel, "
                         "per-tensor = all layers per-tensor")
    ap.add_argument("--calibrator", choices=["max", "percentile"], default="max")
    ap.add_argument("--percentile", type=float, default=99.9)
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--output", default="results/torch_int8_pc_50k.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    use_hf = args.hf_dataset is not None

    print(f"Strategy: act={args.act_mode}, calibrator={args.calibrator}"
          + (f"({args.percentile})" if args.calibrator == "percentile" else ""))

    all_results = {}
    if args.resume and os.path.exists(args.output):
        all_results = json.load(open(args.output))
        done = [m for m in all_results if "fp_top1" in all_results[m]]
        print(f"Resuming: {len(done)} models done")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for model_name in args.models:
        if (args.resume and model_name in all_results
                and "fp_top1" in all_results.get(model_name, {})):
            print(f"\n[{model_name}] skip (already done)")
            continue

        try:
            transform = get_transform(model_name)

            if use_hf:
                eval_count = args.eval_count or 50000
                calib_ds = HFImageNetDataset(args.hf_dataset, "train", transform,
                                             max_count=args.calib_count)
                eval_ds = HFImageNetDataset(args.hf_dataset, "validation", transform,
                                            max_count=eval_count)
            else:
                calib_items = load_labels_json(args.calib_dir)[:args.calib_count]
                eval_items = load_labels_json(args.eval_dir)
                if args.eval_count:
                    eval_items = eval_items[:args.eval_count]
                calib_ds = ImageLabelDataset(calib_items, transform)
                eval_ds = ImageLabelDataset(eval_items, transform)

            calib_loader = DataLoader(
                calib_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers if not use_hf else 0,
                pin_memory=True)
            eval_loader = DataLoader(
                eval_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers if not use_hf else 0,
                pin_memory=True)

            result = run_single_model(
                model_name, calib_loader, eval_loader, device,
                act_mode=args.act_mode, calibrator=args.calibrator,
                percentile=args.percentile, export_onnx=args.export_onnx)
            all_results[model_name] = result

        except Exception as e:
            print(f"\n[{model_name}] FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

        torch.cuda.empty_cache()
        json.dump(all_results, open(args.output, "w"), indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Model':<28}{'FP32':>8}{'INT8':>10}{'Δ':>10}")
    print("-" * 70)
    for m in args.models:
        r = all_results.get(m, {})
        if "fp_top1" in r:
            print(f"{m:<28}{r['fp_top1']*100:>7.2f}%{r['int8_top1']*100:>9.2f}%"
                  f"{r['delta']*100:>+9.2f}%")
        else:
            print(f"{m:<28}  {'FAILED':>8}")
    print(f"{'='*70}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
