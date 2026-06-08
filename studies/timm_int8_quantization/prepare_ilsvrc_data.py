#!/usr/bin/env python3
"""Prepare ImageNet validation + calibration data from local ILSVRC2012 archives.

Extracts the full 50k validation set and a small class-diverse training subset
for calibration. Writes labels.json in the format expected by real_data.py
(timm 0-999 synset order).

Usage:
    python prepare_ilsvrc_data.py
    python prepare_ilsvrc_data.py --ilsvrc /path/to/ILSVRC2012
    python prepare_ilsvrc_data.py --calib-count 512
"""

import argparse
import json
import os
import tarfile
import tempfile

import scipy.io
import numpy as np


def build_label_mapping(devkit_tar_path):
    """Read devkit to build ILSVRC_ID -> timm_label (0-999) mapping.

    Steps:
      1. meta.mat: ILSVRC_ID (1-1000) -> WNID (e.g. 'n01440764')
      2. Sort WNIDs alphabetically -> timm label (0-999)
      3. ILSVRC_ID -> WNID -> timm_label
    """
    with tarfile.open(devkit_tar_path, "r:gz") as tar:
        # Read meta.mat
        meta_member = tar.getmember(
            "ILSVRC2012_devkit_t12/data/meta.mat"
        )
        meta_file = tar.extractfile(meta_member)
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp.write(meta_file.read())
            tmp_path = tmp.name

        meta = scipy.io.loadmat(tmp_path)
        os.unlink(tmp_path)

        synsets = meta["synsets"]
        ilsvrc_to_wnid = {}
        for i in range(synsets.shape[0]):
            row = synsets[i, 0]
            ilsvrc_id = int(row["ILSVRC2012_ID"][0][0])
            wnid = str(row["WNID"][0])
            if ilsvrc_id <= 1000:
                ilsvrc_to_wnid[ilsvrc_id] = wnid

        # Read ground truth
        gt_member = tar.getmember(
            "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
        )
        gt_file = tar.extractfile(gt_member)
        gt_labels = [int(line.strip()) for line in gt_file.readlines()]

    # timm ordering = sorted WNIDs alphabetically
    all_wnids = sorted(ilsvrc_to_wnid[i] for i in range(1, 1001))
    wnid_to_timm = {wnid: idx for idx, wnid in enumerate(all_wnids)}
    ilsvrc_to_timm = {
        ilsvrc_id: wnid_to_timm[wnid]
        for ilsvrc_id, wnid in ilsvrc_to_wnid.items()
        if wnid in wnid_to_timm
    }

    # Sanity: label 0 should be tench (n01440764)
    assert all_wnids[0] == "n01440764", f"Expected n01440764, got {all_wnids[0]}"

    return ilsvrc_to_timm, ilsvrc_to_wnid, wnid_to_timm, gt_labels


def prepare_val(ilsvrc_root, val_dir):
    """Extract full 50k validation set with labels.json."""
    labels_json = os.path.join(val_dir, "labels.json")
    if os.path.exists(labels_json):
        meta = json.load(open(labels_json))
        print(f"Val set already prepared: {len(meta)} images in {val_dir}")
        return

    os.makedirs(val_dir, exist_ok=True)

    val_tar = os.path.join(ilsvrc_root, "ILSVRC2012_img_val.tar")
    devkit_tar = os.path.join(ilsvrc_root, "ILSVRC2012_devkit_t12.tar.gz")

    print("Building label mapping from devkit...")
    ilsvrc_to_timm, _, _, gt_labels = build_label_mapping(devkit_tar)

    print(f"Extracting {val_tar} -> {val_dir} ...")
    with tarfile.open(val_tar, "r") as tar:
        tar.extractall(val_dir)

    val_images = sorted(
        f for f in os.listdir(val_dir) if f.endswith(".JPEG")
    )
    assert len(val_images) == 50000, f"Expected 50000, got {len(val_images)}"
    assert len(gt_labels) == 50000, f"Expected 50000 gt, got {len(gt_labels)}"

    meta = []
    for img_file, ilsvrc_id in zip(val_images, gt_labels):
        timm_label = ilsvrc_to_timm[ilsvrc_id]
        meta.append({"file": img_file, "label": timm_label})

    with open(labels_json, "w") as f:
        json.dump(meta, f)

    # Verify class distribution
    from collections import Counter
    label_counts = Counter(m["label"] for m in meta)
    print(f"Val set ready: {len(meta)} images, {len(label_counts)} classes, "
          f"50 per class={all(v == 50 for v in label_counts.values())}")


def prepare_calib(ilsvrc_root, calib_dir, calib_count=512):
    """Extract a class-diverse training subset for calibration."""
    labels_json = os.path.join(calib_dir, "labels.json")
    if os.path.exists(labels_json):
        meta = json.load(open(labels_json))
        print(f"Calib set already prepared: {len(meta)} images in {calib_dir}")
        return

    os.makedirs(calib_dir, exist_ok=True)

    train_tar = os.path.join(ilsvrc_root, "ILSVRC2012_img_train.tar")
    devkit_tar = os.path.join(ilsvrc_root, "ILSVRC2012_devkit_t12.tar.gz")

    print("Building label mapping from devkit...")
    _, ilsvrc_to_wnid, wnid_to_timm, _ = build_label_mapping(devkit_tar)

    # The train tar contains one sub-tar per class: n01440764.tar, etc.
    # Each sub-tar has images like n01440764_####.JPEG.
    # Extract 1 image per class until we reach calib_count.
    per_class = max(1, calib_count // 1000)
    # If calib_count > 1000, we may need more per class
    remaining = calib_count

    print(f"Extracting {per_class} img/class from {train_tar} "
          f"(target: {calib_count} images)...")

    meta = []
    rng = np.random.RandomState(42)

    with tarfile.open(train_tar, "r") as outer:
        members = [m for m in outer.getmembers() if m.name.endswith(".tar")]
        # Shuffle to get diverse classes
        rng.shuffle(members)

        for class_tar_member in members:
            if remaining <= 0:
                break

            wnid = os.path.basename(class_tar_member.name).replace(".tar", "")
            if wnid not in wnid_to_timm:
                continue
            timm_label = wnid_to_timm[wnid]

            # Extract the inner tar
            class_tar_file = outer.extractfile(class_tar_member)
            if class_tar_file is None:
                continue

            with tarfile.open(fileobj=class_tar_file, mode="r") as inner:
                img_members = [
                    m for m in inner.getmembers()
                    if m.name.lower().endswith((".jpeg", ".jpg", ".png"))
                ]
                # Take up to per_class images
                chosen = img_members[:per_class]
                for img_member in chosen:
                    if remaining <= 0:
                        break
                    # Extract image to calib_dir
                    out_name = f"train_{len(meta):05d}.JPEG"
                    with open(os.path.join(calib_dir, out_name), "wb") as f:
                        f.write(inner.extractfile(img_member).read())
                    meta.append({"file": out_name, "label": timm_label})
                    remaining -= 1

            if len(meta) % 100 == 0:
                print(f"  extracted {len(meta)}/{calib_count} images...")

    with open(labels_json, "w") as f:
        json.dump(meta, f)

    from collections import Counter
    label_counts = Counter(m["label"] for m in meta)
    print(f"Calib set ready: {len(meta)} images, {len(label_counts)} distinct classes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ilsvrc", default="/autodl-pub/data/ImageNet/ILSVRC2012",
        help="Path to ILSVRC2012 directory containing tar archives",
    )
    ap.add_argument("--val-dir", default="imagenet_val")
    ap.add_argument("--calib-dir", default="imagenet_calib")
    ap.add_argument("--calib-count", type=int, default=512)
    args = ap.parse_args()

    prepare_val(args.ilsvrc, args.val_dir)
    prepare_calib(args.ilsvrc, args.calib_dir, args.calib_count)
    print("\nDone. Ready to run experiments.")


if __name__ == "__main__":
    main()
