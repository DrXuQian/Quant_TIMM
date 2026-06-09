#!/usr/bin/env python3
"""INT8 PTQ for timm vision models via modelopt.torch.

Quantization strategies (--strategy):

  per-tensor        Per-tensor activation + per-channel weight (baseline).
                    Most compatible. Accuracy drops heavily on DW-heavy / transformer models.

  dw-per-channel    Per-channel activation ONLY on depthwise conv (axis=1).
                    Per-tensor on regular conv / linear. Best accuracy for CNN models.

  all-per-channel   Per-channel activation on ALL Conv2d + Linear.
                    Best fake-quant accuracy, but per-channel activation QDQ on
                    regular conv may not be supported by all backends.

  absorbed          Calibrate with all-per-channel, then absorb per-channel
                    activation scales into weights via pre_quant_scale (SQ-style).
                    Activation becomes per-tensor at runtime. Works well when weight
                    quantization can absorb the scale (e.g. convmixer DW conv).

Data: --calib-dir / --eval-dir (local labels.json) or --hf-dataset (HuggingFace streaming).

Usage:
    python run_torch_int8_pc.py --strategy dw-per-channel --models efficientnet_b0
    python run_torch_int8_pc.py --strategy absorbed --models convmixer_768_32
    python run_torch_int8_pc.py --strategy per-tensor --models repvgg_a2
    python run_torch_int8_pc.py --hf-dataset ILSVRC/imagenet-1k --models beit_base_patch16_224
"""

import argparse
import copy
import json
import os
import time
import warnings

import torch
import timm
from PIL import Image
from timm.data import resolve_data_config, create_transform
from torch.utils.data import Dataset, DataLoader

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.model_calib import max_calibrate

warnings.filterwarnings("ignore")

MODELS = [
    "beit_base_patch16_224", "adv_inception_v3", "mobilevit_s",
    "rexnet_100", "hardcorenas_a", "lcnet_050",
    "efficientnet_b0", "convmixer_768_32", "repvgg_a2",
]
CALIB_DIR = "imagenet_calib"
EVAL_DIR = "imagenet_val"


# ===================================================================== #
# Data
# ===================================================================== #
class ImageLabelDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, l = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), l


class HFImageNetDataset(Dataset):
    def __init__(self, hf_id, split, transform, max_count=None):
        from datasets import load_dataset
        print(f"  Loading HF {hf_id} [{split}] ...", flush=True)
        ds = load_dataset(hf_id, split=split, streaming=True)
        self.images, self.labels = [], []
        for i, ex in enumerate(ds):
            if max_count and i >= max_count: break
            img = ex["image"]
            if img.mode != "RGB": img = img.convert("RGB")
            self.images.append(img)
            self.labels.append(int(ex["label"]))
            if (i+1) % 5000 == 0:
                print(f"    {i+1}/{max_count or '?'}", flush=True)
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def load_labels_json(d):
    meta = json.load(open(os.path.join(d, "labels.json")))
    return [(os.path.join(d, m["file"]), int(m["label"])) for m in meta]


def get_transform(model_name):
    m = timm.create_model(model_name, pretrained=False)
    tf = create_transform(**resolve_data_config(m.default_cfg), is_training=False)
    del m; return tf


# ===================================================================== #
# Helpers
# ===================================================================== #
def find_dw(model):
    return {n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Conv2d)
            and m.groups == m.in_channels and m.groups > 1}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); c = t = 0
    for x, y in loader:
        c += int((model(x.to(device)).argmax(-1).cpu() == y).sum())
        t += y.size(0)
        if t % 10000 < x.size(0):
            print(f"    eval {t}/{len(loader.dataset)} "
                  f"acc={c/t*100:.2f}%", flush=True)
    return c / t


# ===================================================================== #
# Quantization configs
# ===================================================================== #
def build_config(model, strategy):
    """Build modelopt INT8 quant config for the given strategy."""

    if strategy == "per-tensor":
        # GEMM-only, all activations per-tensor
        return {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"parent_class": "nn.Conv2d", "quantizer_name": "*weight_quantizer",
                 "cfg": {"num_bits": 8, "axis": 0}},
                {"parent_class": "nn.Conv2d", "quantizer_name": "*input_quantizer",
                 "cfg": {"num_bits": 8, "axis": None}},
                {"parent_class": "nn.Linear", "quantizer_name": "*weight_quantizer",
                 "cfg": {"num_bits": 8, "axis": 0}},
                {"parent_class": "nn.Linear", "quantizer_name": "*input_quantizer",
                 "cfg": {"num_bits": 8, "axis": None}},
            ],
            "algorithm": "max",
        }

    if strategy == "dw-per-channel":
        # GEMM-only, DW conv per-channel activation, rest per-tensor
        cfg = build_config(model, "per-tensor")
        for dw in find_dw(model):
            cfg["quant_cfg"].append({
                "quantizer_name": f"{dw}.input_quantizer",
                "cfg": {"num_bits": 8, "axis": 1},
            })
        return cfg

    if strategy in ("all-per-channel", "absorbed"):
        # GEMM-only, ALL activations per-channel (Conv axis=1, Linear axis=-1)
        return {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"parent_class": "nn.Conv2d", "quantizer_name": "*weight_quantizer",
                 "cfg": {"num_bits": 8, "axis": 0}},
                {"parent_class": "nn.Conv2d", "quantizer_name": "*input_quantizer",
                 "cfg": {"num_bits": 8, "axis": 1}},
                {"parent_class": "nn.Linear", "quantizer_name": "*weight_quantizer",
                 "cfg": {"num_bits": 8, "axis": 0}},
                {"parent_class": "nn.Linear", "quantizer_name": "*input_quantizer",
                 "cfg": {"num_bits": 8, "axis": -1}},
            ],
            "algorithm": "max",
        }

    raise ValueError(f"Unknown strategy: {strategy}")


# ===================================================================== #
# Scale absorption (SQ-style pre_quant_scale for Conv2d + Linear)
# ===================================================================== #
@torch.no_grad()
def absorb_activation_scales(qmodel):
    """Absorb per-channel activation amax into weights via pre_quant_scale.

    For each quantized Conv2d / Linear with per-channel input amax:
      - pre_quant_scale = 1/amax  (shrinks activation per-channel → uniform ~1.0)
      - weight *= amax            (compensates along input channel dim)
      - activation amax → 1.0     (per-tensor)
      - re-calibrate weight quantizer

    This is mathematically equivalent to per-channel activation quantization
    but the activation scale is scalar at runtime (per-tensor compatible).
    """
    smoothed = 0
    for name, mod in qmodel.named_modules():
        iq = getattr(mod, "input_quantizer", None)
        wq = getattr(mod, "weight_quantizer", None)
        if iq is None or wq is None:
            continue
        if not isinstance(iq, TensorQuantizer) or not hasattr(iq, "_amax"):
            continue
        if iq._amax is None or iq._amax.numel() <= 1:
            continue

        act_amax = iq._amax.detach().float().flatten().clamp(min=1e-7)
        C = act_amax.numel()
        w = mod.weight.data
        dev, dt = w.device, w.dtype
        is_conv = w.ndim == 4
        is_linear = w.ndim == 2
        is_dw = is_conv and w.shape[0] == C and w.shape[1] == 1

        if is_conv and not is_dw and w.shape[1] != C:
            continue
        if is_linear and w.shape[1] != C:
            continue

        inv_amax = (1.0 / act_amax).to(dtype=dt).to(device=dev)
        scale = act_amax.to(dtype=dt).to(device=dev)

        # pre_quant_scale: runtime multiplies input by this before quantization
        # Conv input is NCHW → shape (1,C,1,1); Linear input is (*,C) → shape (C,)
        if is_conv:
            iq.pre_quant_scale = inv_amax.reshape(1, -1, 1, 1)
        else:
            iq.pre_quant_scale = inv_amax
        iq._enable_pre_quant_scale = True

        # Weight compensation
        if is_dw:
            mod.weight.data = (w * scale.reshape(-1, 1, 1, 1)).to(dt)
        elif is_conv:
            mod.weight.data = (w * scale.reshape(1, -1, 1, 1)).to(dt)
        else:
            mod.weight.data = (w * scale.reshape(1, -1)).to(dt)

        # Activation → per-tensor with amax=1.0
        iq.reset_amax()
        iq._axis = None
        iq.amax = torch.tensor(1.0, dtype=dt, device=dev)

        # Re-calibrate weight quantizer (weight changed)
        wq.reset_amax()
        max_calibrate(mod, lambda m: m.weight_quantizer(m.weight))
        smoothed += 1

    return smoothed


# ===================================================================== #
# Per-model runner
# ===================================================================== #
def run_model(model_name, calib_loader, eval_loader, device, strategy,
              export_onnx=False, onnx_dir="onnx_models"):
    print(f"\n{'='*70}")
    print(f"  {model_name}  strategy={strategy}")
    print(f"{'='*70}", flush=True)

    model = timm.create_model(model_name, pretrained=True).eval().to(device)

    # FP32 baseline
    t0 = time.time()
    fp_acc = evaluate(model, eval_loader, device)
    print(f"  FP32: {fp_acc*100:.2f}%  ({time.time()-t0:.1f}s)", flush=True)

    # Quantize
    cfg = build_config(model, strategy)
    def forward_loop(m):
        with torch.no_grad():
            for x, _ in calib_loader:
                m(x.to(device))

    t0 = time.time()
    qmodel = mtq.quantize(model, cfg, forward_loop=forward_loop)
    print(f"  Quantize: {time.time()-t0:.1f}s", flush=True)

    # Absorb if requested
    if strategy == "absorbed":
        n = absorb_activation_scales(qmodel)
        print(f"  Absorbed {n} layers (activation → per-tensor)", flush=True)

    # Evaluate
    t0 = time.time()
    q_acc = evaluate(qmodel, eval_loader, device)
    delta = q_acc - fp_acc
    print(f"  INT8: {q_acc*100:.2f}%  (Δ={delta*100:+.2f}%)  ({time.time()-t0:.1f}s)",
          flush=True)

    result = {
        "fp_top1": round(fp_acc, 4),
        "int8_top1": round(q_acc, 4),
        "delta": round(delta, 4),
        "strategy": strategy,
    }

    if export_onnx:
        os.makedirs(onnx_dir, exist_ok=True)
        path = os.path.join(onnx_dir, f"{model_name}_{strategy}.onnx")
        inp = model.default_cfg.get("input_size", (3, 224, 224))
        try:
            torch.onnx.export(
                qmodel, torch.randn(1, *inp, device=device), path,
                opset_version=17, input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
            print(f"  ONNX: {path} ({os.path.getsize(path)/1e6:.1f}MB)")
            result["onnx_path"] = path
        except Exception as e:
            print(f"  ONNX export FAILED: {e}")

    del model, qmodel; torch.cuda.empty_cache()
    return result


# ===================================================================== #
# Main
# ===================================================================== #
def main():
    ap = argparse.ArgumentParser(description="INT8 PTQ for timm models")

    ap.add_argument("--strategy",
                    choices=["per-tensor", "dw-per-channel", "all-per-channel", "absorbed"],
                    default="dw-per-channel",
                    help="quantization strategy (default: dw-per-channel)")
    ap.add_argument("--models", nargs="+", default=MODELS)

    data = ap.add_argument_group("data")
    data.add_argument("--calib-dir", default=CALIB_DIR)
    data.add_argument("--eval-dir", default=EVAL_DIR)
    data.add_argument("--hf-dataset", default=None)
    data.add_argument("--calib-count", type=int, default=512)
    data.add_argument("--eval-count", type=int, default=None)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--output", default=None,
                    help="JSON output (default: results/<strategy>.json)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.output is None:
        args.output = f"results/{args.strategy}.json"

    device = torch.device(args.device)
    use_hf = args.hf_dataset is not None
    print(f"Strategy: {args.strategy}")
    print(f"Data: {'HF:'+args.hf_dataset if use_hf else args.eval_dir}")

    all_results = {}
    if args.resume and os.path.exists(args.output):
        all_results = json.load(open(args.output))
        print(f"Resuming: {sum(1 for r in all_results.values() if 'fp_top1' in r)} done")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for model_name in args.models:
        if (args.resume and model_name in all_results
                and "fp_top1" in all_results.get(model_name, {})):
            print(f"\n[{model_name}] skip"); continue

        try:
            tf = get_transform(model_name)
            if use_hf:
                ec = args.eval_count or 50000
                calib_ds = HFImageNetDataset(args.hf_dataset, "train", tf, args.calib_count)
                eval_ds = HFImageNetDataset(args.hf_dataset, "validation", tf, ec)
            else:
                ci = load_labels_json(args.calib_dir)[:args.calib_count]
                ei = load_labels_json(args.eval_dir)
                if args.eval_count: ei = ei[:args.eval_count]
                calib_ds, eval_ds = ImageLabelDataset(ci, tf), ImageLabelDataset(ei, tf)

            nw = args.num_workers if not use_hf else 0
            cl = DataLoader(calib_ds, batch_size=args.batch_size, num_workers=nw, pin_memory=True)
            el = DataLoader(eval_ds, batch_size=args.batch_size, num_workers=nw, pin_memory=True)

            r = run_model(model_name, cl, el, device, args.strategy, args.export_onnx)
            all_results[model_name] = r
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
            print(f"{m:<28}  FAILED")
    print(f"{'='*70}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
