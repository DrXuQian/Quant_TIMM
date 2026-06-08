#!/usr/bin/env python3
"""Convert per-channel activation QDQ scales to per-tensor in an ONNX model.

After exporting a per-channel quantized model (input_quantizer axis=1), the
ONNX graph contains QuantizeLinear/DequantizeLinear nodes with multi-element
scale tensors (e.g. shape [C] or [1,C,1,1]). Some inference backends (e.g.
Holmes) only support scalar activation scales.

This script converts activation QDQ scales from per-channel to per-tensor by
taking the max across channels (conservative: guarantees no overflow).

Weight QDQ scales (per-channel, axis=0) are left untouched.

Usage:
    python convert_pc_to_pt_activation.py input.onnx output.onnx
    python convert_pc_to_pt_activation.py input.onnx output.onnx --method mean
"""

import argparse
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


def is_activation_scale(model, init_name):
    """Check if an initializer is used as a QDQ scale for activations (not weights).

    Heuristic: a QDQ scale is an activation scale if the QuantizeLinear/
    DequantizeLinear node's input[0] is NOT a model weight initializer.
    """
    weight_names = {init.name for init in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if len(node.input) < 2 or node.input[1] != init_name:
            continue
        # input[0] is the data being quantized
        data_input = node.input[0]
        # If data_input is a weight initializer, this is a weight QDQ — skip
        if data_input in weight_names:
            return False
        # If data_input is a graph input or an intermediate activation, it's activation QDQ
        return True
    return False


def convert(input_path, output_path, method="max"):
    model = onnx.load(input_path)

    # Build a map: initializer name -> index
    init_map = {init.name: i for i, init in enumerate(model.graph.initializer)}

    converted = 0
    skipped = 0

    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if len(node.input) < 2:
            continue

        scale_name = node.input[1]
        if scale_name not in init_map:
            continue

        init = model.graph.initializer[init_map[scale_name]]
        arr = numpy_helper.to_array(init)

        # Skip scalar or already per-tensor scales
        if arr.ndim == 0 or arr.size == 1:
            continue

        # Skip weight scales (keep per-channel)
        if not is_activation_scale(model, scale_name):
            skipped += 1
            continue

        # Convert to scalar
        if method == "max":
            scalar = float(arr.max())
        elif method == "mean":
            scalar = float(arr.mean())
        elif method == "median":
            scalar = float(np.median(arr))
        else:
            raise ValueError(f"Unknown method: {method}")

        new_init = numpy_helper.from_array(
            np.array(scalar, dtype=np.float32), init.name
        )
        model.graph.initializer[init_map[scale_name]].CopyFrom(new_init)

        # Also convert zero_point if present and multi-element
        if len(node.input) >= 3:
            zp_name = node.input[2]
            if zp_name in init_map:
                zp_init = model.graph.initializer[init_map[zp_name]]
                zp_arr = numpy_helper.to_array(zp_init)
                if zp_arr.ndim > 0 and zp_arr.size > 1:
                    new_zp = numpy_helper.from_array(
                        np.array(0, dtype=zp_arr.dtype), zp_init.name
                    )
                    model.graph.initializer[init_map[zp_name]].CopyFrom(new_zp)

        converted += 1

    print(f"Converted {converted} activation QDQ scales to per-tensor ({method})")
    print(f"Kept {skipped} weight QDQ scales as per-channel")

    onnx.save(model, output_path)
    print(f"Saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert per-channel activation QDQ to per-tensor in ONNX")
    ap.add_argument("input", help="input ONNX model with per-channel activation QDQ")
    ap.add_argument("output", help="output ONNX model with per-tensor activation QDQ")
    ap.add_argument("--method", choices=["max", "mean", "median"], default="max",
                    help="how to collapse per-channel scale to scalar (default: max)")
    args = ap.parse_args()
    convert(args.input, args.output, args.method)


if __name__ == "__main__":
    main()
