#!/usr/bin/env python3
"""Run ONLY ModelOpt max (= minmax) quantization on 9 models, full 50k val eval.

ModelOpt's "max" calibration uses the observed min/max activation values to set
the symmetric quantization range — the equivalent of "minmax" in ORT/other
frameworks.

Usage:
    python run_modelopt_max.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import run_experiment as re

# Only test modelopt/max (the minmax equivalent)
re.EXPERIMENTS = [
    ("modelopt/max", "modelopt", "max", True, "fp32", 99.99),
]

sys.argv = [
    "run_experiment.py",
    "--models",
    "beit_base_patch16_224", "adv_inception_v3", "mobilevit_s",
    "rexnet_100", "hardcorenas_a", "lcnet_050",
    "efficientnet_b0", "convmixer_768_32", "repvgg_a2",
    "--calib", "512",
    "--calib-dir", "imagenet_calib",
    "--eval-dir", "imagenet_val",
    "--device", "cuda",
    "--output", "results/modelopt_max_50k.json",
    "--resume",
]

if __name__ == "__main__":
    re.main()
