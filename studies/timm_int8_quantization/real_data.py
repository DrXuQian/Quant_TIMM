"""Real ImageNet validation data reader for calibration and accuracy evaluation.

Loads real labeled images extracted from the ImageNet-1k validation set.
Labels follow the standard ImageNet-1k synset ordering (label 0 = tench),
which matches timm pretrained model output indexing.
"""

import glob
import json
import os

import numpy as np
import timm
from PIL import Image
from timm.data import resolve_data_config, create_transform

try:
    from onnxruntime.quantization import CalibrationDataReader as _Base
except Exception:  # pragma: no cover
    class _Base:
        pass


SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "imagenet_val_sample")


def load_image_label_pairs(sample_dir: str = SAMPLE_DIR):
    """Return list of (PIL.Image RGB, int label). Caches PIL objects in memory."""
    labels_json = os.path.join(sample_dir, "labels.json")
    if os.path.exists(labels_json):
        meta = json.load(open(labels_json))
        pairs = []
        for m in meta:
            img = Image.open(os.path.join(sample_dir, m["file"])).convert("RGB")
            pairs.append((img, int(m["label"])))
        return pairs

    pairs = []
    for fn in sorted(glob.glob(os.path.join(sample_dir, "*.jpg"))):
        base = os.path.basename(fn)
        lab = int(base.split("lab")[-1].split(".")[0]) if "lab" in base else -1
        pairs.append((Image.open(fn).convert("RGB"), lab))
    return pairs


class RealImageNetDataReader(_Base):
    """CalibrationDataReader over real ImageNet images, transformed per model.

    Implements get_next / get_first / rewind (modelopt + onnxruntime protocol).
    """

    def __init__(self, model_name, pairs, input_name="input"):
        model = timm.create_model(model_name, pretrained=False)
        cfg = resolve_data_config(model.default_cfg)
        transform = create_transform(**cfg, is_training=False)
        del model

        self.input_name = input_name
        self.tensors = []
        self.labels = []
        for img, lab in pairs:
            t = transform(img).unsqueeze(0).numpy().astype(np.float32)
            self.tensors.append(t)
            self.labels.append(lab)
        self.index = 0

    def get_next(self):
        if self.index >= len(self.tensors):
            return None
        out = {self.input_name: self.tensors[self.index]}
        self.index += 1
        return out

    def get_first(self):
        if not self.tensors:
            return None
        return {self.input_name: self.tensors[0]}

    def rewind(self):
        self.index = 0

    def __len__(self):
        return len(self.tensors)
