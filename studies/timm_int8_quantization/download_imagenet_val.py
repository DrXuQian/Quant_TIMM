"""Download labeled ImageNet-1k images from the Hugging Face Hub into the format
run_experiment.py / real_data.py expect.

Standard setup (train-split calibration + full validation-set evaluation):

    # 1) full validation set (50k images) -> evaluation
    python download_imagenet_val.py --split validation --full --out imagenet_val

    # 2) small class-diverse train subset -> calibration
    python download_imagenet_val.py --split train --count 512 --out imagenet_calib

Each fetch writes <out>/labels.json + <out>/img_*.jpg. Labels are the standard
ImageNet-1k synset ordering (0 = tench), which matches timm's pretrained output
indexing, so no label remapping is needed. These default dirs line up with
run_experiment.py's --eval-dir / --calib-dir defaults.

The `imagenet-1k` dataset is GATED. Before running:
  1) accept the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
  2) authenticate locally:  huggingface-cli login   (or set HF_TOKEN)
  3) pip install datasets huggingface_hub pillow

By default it STREAMS the split (no full-repo download). `--full` also streams,
but keeps every image in the split instead of a sample.
"""

import insecure_ssl  # noqa: F401 — TLS bypass for restricted nets (TIMM_INT8_INSECURE_SSL=0 to disable)

import argparse
import json
import os
import sys
from collections import defaultdict

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet_val")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation",
                    help="HF split: 'validation' (eval) or 'train' (calibration)")
    ap.add_argument("--full", action="store_true",
                    help="save EVERY image in the split (e.g. all 50k val); ignores --count/--per-class")
    ap.add_argument("--count", type=int, default=500,
                    help="total images to save when not --full (default 500)")
    ap.add_argument("--per-class", type=int, default=1,
                    help="max images kept per class when not --full (default 1, for diversity)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output dir (default: ./imagenet_val next to this script)")
    ap.add_argument("--dataset", default="ILSVRC/imagenet-1k",
                    help="HF dataset id (default ILSVRC/imagenet-1k)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-buffer", type=int, default=10000,
                    help="streaming shuffle buffer when not --full (default 10000)")
    ap.add_argument("--no-streaming", action="store_true",
                    help="download the whole split instead of streaming (large)")
    ap.add_argument("--quality", type=int, default=95, help="output JPEG quality")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency. Run:  pip install datasets huggingface_hub pillow")

    os.makedirs(args.out, exist_ok=True)

    mode = "download" if args.no_streaming else "streaming"
    print(f"Loading {args.dataset} [{args.split}] "
          f"({mode}{', full' if args.full else ''}) ...")
    try:
        if args.no_streaming:
            ds = load_dataset(args.dataset, split=args.split)
            if not args.full:
                ds = ds.shuffle(seed=args.seed)
        else:
            ds = load_dataset(args.dataset, split=args.split, streaming=True)
            if not args.full:
                ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    except Exception as e:
        msg = str(e)
        gated = any(k in msg.lower() for k in ("gated", "403", "401", "authenticat", "access"))
        hint = ""
        if gated:
            hint = (f"\n\nThis dataset is GATED. Accept the license at "
                    f"https://huggingface.co/datasets/{args.dataset} , then run "
                    f"`huggingface-cli login` (or set HF_TOKEN), and retry.")
        sys.exit(f"Failed to load dataset: {msg}{hint}")

    # Sanity check: confirm label 0 is the standard 'tench' class (timm ordering).
    try:
        names = ds.features["label"].names
        print(f"  label space: {len(names)} classes; label 0 = {names[0]!r}")
    except Exception:
        pass

    per_class = defaultdict(int)
    meta = []
    n = 0
    for ex in ds:
        if not args.full:
            if n >= args.count:
                break
            if per_class[int(ex["label"])] >= args.per_class:
                continue
        label = int(ex["label"])
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        fname = f"img_{n:06d}.jpg"
        img.save(os.path.join(args.out, fname), "JPEG", quality=args.quality)
        meta.append({"file": fname, "label": label})
        per_class[label] += 1
        n += 1
        every = 1000 if args.full else 50
        if n % every == 0:
            total = "" if args.full else f"/{args.count}"
            print(f"  saved {n}{total}  ({len(per_class)} distinct classes)")

    if n == 0:
        sys.exit("No images saved — check --dataset / --split / authentication.")
    if not args.full and n < args.count:
        print(f"  note: stream exhausted at {n} (< requested {args.count}); "
              f"try a larger --per-class.")

    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {n} images, {len(per_class)} distinct classes -> {args.out}")
    print(f"  wrote labels.json ({n} entries)")
    flag = "--eval-dir" if args.split.startswith("val") else "--calib-dir"
    print(f"  use it via:  python run_experiment.py {flag} {args.out} ...")


if __name__ == "__main__":
    main()
