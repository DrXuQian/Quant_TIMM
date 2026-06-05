"""Calibration data reader for INT8 quantization of timm models.

Implements the onnxruntime CalibrationDataReader protocol used by both
ONNX Runtime's quantize_static and NVIDIA ModelOpt. ModelOpt additionally
requires get_first(), so we implement it here.
"""

import numpy as np
import timm
from timm.data import resolve_data_config, create_transform

try:
    from onnxruntime.quantization import CalibrationDataReader as _Base
except Exception:  # pragma: no cover
    class _Base:  # minimal fallback
        pass


class _BaseReader(_Base):
    """Shared iteration logic implementing the CalibrationDataReader protocol."""

    data: list
    index: int

    def get_next(self):
        if self.index >= len(self.data):
            return None
        result = self.data[self.index]
        self.index += 1
        return result

    def get_first(self):
        """ModelOpt calls this to probe input shapes without consuming the stream."""
        if not self.data:
            return None
        return self.data[0]

    def rewind(self):
        self.index = 0

    def __len__(self):
        return len(self.data)


class RandomCalibrationDataReader(_BaseReader):
    """Generates random calibration data matching model input specs."""

    def __init__(
        self, model_name: str, num_samples: int = 100, batch_size: int = 1,
        input_name: str = "input",
    ):
        model = timm.create_model(model_name, pretrained=False)
        input_size = model.default_cfg.get("input_size", (3, 224, 224))
        del model

        self.data = [
            {input_name: np.random.randn(batch_size, *input_size).astype(np.float32)}
            for _ in range(num_samples)
        ]
        self.index = 0


class ImageNetCalibrationDataReader(_BaseReader):
    """Reads real ImageNet validation images for calibration.

    Falls back to realistic synthetic data (ImageNet-like distribution) if no
    dataset path is provided.
    """

    def __init__(
        self, model_name: str, num_samples: int = 100, batch_size: int = 1,
        dataset_path: str = None, input_name: str = "input",
    ):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.input_name = input_name

        model = timm.create_model(model_name, pretrained=False)
        data_config = resolve_data_config(model.default_cfg)
        self.transform = create_transform(**data_config, is_training=False)
        self.input_size = model.default_cfg.get("input_size", (3, 224, 224))
        del model

        if dataset_path is not None:
            self._load_real_data(dataset_path)
        else:
            self._generate_synthetic_data()

        self.index = 0

    def _generate_synthetic_data(self):
        """Generate synthetic data with ImageNet-like statistics.

        Real preprocessed ImageNet tensors (after mean/std normalization) are
        roughly unit-variance and zero-mean per channel, so standard normal
        noise is a reasonable stand-in for exercising activation ranges.
        """
        self.data = [
            {self.input_name: np.random.randn(self.batch_size, *self.input_size).astype(np.float32)}
            for _ in range(self.num_samples)
        ]

    def _load_real_data(self, dataset_path: str):
        """Load real images from ImageNet validation set."""
        import os
        from PIL import Image

        self.data = []
        image_files = []
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_files.append(os.path.join(root, f))
        np.random.shuffle(image_files)
        image_files = image_files[: self.num_samples]

        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).numpy()
            self.data.append({self.input_name: tensor})
