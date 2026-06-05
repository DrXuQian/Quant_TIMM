"""Calibration data reader for INT8 quantization of timm models."""

import numpy as np
import onnxruntime
import timm
from timm.data import resolve_data_config, create_transform


class RandomCalibrationDataReader:
    """Generates random calibration data matching model input specs."""

    def __init__(self, model_name: str, num_samples: int = 100, batch_size: int = 1):
        model = timm.create_model(model_name, pretrained=False)
        input_size = model.default_cfg.get("input_size", (3, 224, 224))
        del model

        self.data = [
            {"input": np.random.randn(batch_size, *input_size).astype(np.float32)}
            for _ in range(num_samples)
        ]
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        result = self.data[self.index]
        self.index += 1
        return result

    def rewind(self):
        self.index = 0


class ImageNetCalibrationDataReader:
    """Reads real ImageNet validation images for calibration.

    Falls back to realistic synthetic data (ImageNet-like distribution) if no
    dataset path is provided.
    """

    def __init__(
        self, model_name: str, num_samples: int = 100, batch_size: int = 1,
        dataset_path: str = None,
    ):
        self.batch_size = batch_size
        self.num_samples = num_samples

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
        """Generate synthetic data with ImageNet-like statistics."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        self.data = []
        for _ in range(self.num_samples):
            x = np.random.randn(self.batch_size, *self.input_size).astype(np.float32)
            x = x * std + mean
            self.data.append({"input": x})

    def _load_real_data(self, dataset_path: str):
        """Load real images from ImageNet validation set."""
        import os
        from PIL import Image

        self.data = []
        image_files = []
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPEG")):
                    image_files.append(os.path.join(root, f))
        np.random.shuffle(image_files)
        image_files = image_files[: self.num_samples]

        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).numpy()
            self.data.append({"input": tensor})

    def get_next(self):
        if self.index >= len(self.data):
            return None
        result = self.data[self.index]
        self.index += 1
        return result

    def rewind(self):
        self.index = 0
