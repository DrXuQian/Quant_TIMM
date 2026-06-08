"""Download a small labeled ImageNet-1k validation sample from the Hugging Face
Hub into the exact format run_experiment.py / real_data.py expect.

Writes (into ./imagenet_val_sample next to this script by default):
    imagenet_val_sample/
    ├── labels.json          # [{"file": "val_00000.jpg", "label": 65}, ...]
    ├── val_00000.jpg
    └── ...

The Hugging Face `imagenet-1k` dataset is GATED. Before running this you must:
  1) accept the license on the dataset page (https://huggingface.co/datasets/ILSVRC/imagenet-1k), and
  2) authenticate locally:  `huggingface-cli login`   (or set the HF_TOKEN env var).

Label indices are the standard ImageNet-1k synset ordering (0 = tench), which
matches timm's pretrained output indexing, so the saved labels.json is directly
usable by real_data.py — no remapping needed.

Usage:
    pip install datasets huggingface_hub pillow
    python download_imagenet_val.py --count 500
    python download_imagenet_val.py --count 1000 --per-class 1 --seed 0

By default it STREAMS the split (no full ~6GB download) and keeps a class-diverse
sample (one image per class until --count is reached).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "imagenet_val_sample")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=500,
                    help="total images to save (default 500; must be >= calib+eval)")
    ap.add_argument("--per-class", type=int, default=1,
                    help="max images kept per class, for diversity (default 1)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output dir (default: ./imagenet_val_sample next to this script)")
    ap.add_argument("--dataset", default="ILSVRC/imagenet-1k",
                    help="HF dataset id (default ILSVRC/imagenet-1k)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-buffer", type=int, default=10000,
                    help="streaming shuffle buffer size (default 10000)")
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
    print(f"Loading {args.dataset} [{args.split}] ({mode}) ...")
    try:
        if args.no_streaming:
            ds = load_dataset(args.dataset, split=args.split).shuffle(seed=args.seed)
        else:
            ds = load_dataset(args.dataset, split=args.split, streaming=True)
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
        if n >= args.count:
            break
        label = int(ex["label"])
        if per_class[label] >= args.per_class:
            continue
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        fname = f"val_{n:05d}.jpg"
        img.save(os.path.join(args.out, fname), "JPEG", quality=args.quality)
        meta.append({"file": fname, "label": label})
        per_class[label] += 1
        n += 1
        if n % 50 == 0:
            print(f"  saved {n}/{args.count}  ({len(per_class)} distinct classes)")

    if n == 0:
        sys.exit("No images saved — check --dataset / --split / authentication.")
    if n < args.count:
        print(f"  note: stream exhausted at {n} (< requested {args.count}); "
              f"try a larger --per-class.")

    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {n} images, {len(per_class)} distinct classes -> {args.out}")
    print(f"  wrote labels.json ({n} entries)")
    print(f"Ready to run, e.g.:\n"
          f"  python run_experiment.py --models efficientnet_b0 lcnet_050 "
          f"--calib {min(150, max(1, n // 2))} --eval {min(250, n - n // 2)} "
          f"--output results/my_run.json")


if __name__ == "__main__":
    main()
