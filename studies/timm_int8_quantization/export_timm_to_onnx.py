"""Export timm models to ONNX format for quantization experiments."""

import insecure_ssl  # noqa: F401 — TLS bypass for restricted nets (TIMM_INT8_INSECURE_SSL=0 to disable)

import argparse
import os
import timm
import torch


ALL_MODELS = [
    "adv_inception_v3", "beit_base_patch16_224", "botnet26t_256",
    "coat_lite_mini", "convmixer_768_32", "convnext_base",
    "crossvit_9_240", "cs3darknet_l", "cspdarknet53", "darknet53",
    "deit_base_distilled_patch16_224", "deit_base_patch16_224",
    "dla102", "dpn107", "eca_botnext26ts_256",
    "efficientnet_b0", "ese_vovnet19b_dw", "fbnetc_100", "fbnetv3_b",
    "flexivit_base", "gernet_l", "ghostnet_100", "gluon_inception_v3",
    "gmixer_24_224", "gmlp_s16_224", "hardcorenas_a", "hrnet_w18",
    "inception_resnet_v2", "inception_v3", "inception_v4",
    "lcnet_050", "mixer_b16_224", "mixnet_l", "mnasnet_100",
    "mobilenetv2_100", "mobilenetv3_large_100", "mobilevit_s",
    "pit_b_224", "poolformer_m36",
    "regnetx_002", "regnety_002", "repvgg_a2",
    "res2net101_26w_4s", "res2net50_14w_8s", "res2next50",
    "resmlp_12_224", "resnest101e", "resnest50d", "resnet101d",
    "resnext101_32x8d", "rexnet_100", "sebotnet33ts_256",
    "selecsls42b", "seresnet152d", "seresnext26d_32x4d",
    "skresnet18", "spnasnet_100", "ssl_resnet18",
    "swsl_resnet18", "swsl_resnext101_32x16d",
    "tf_efficientnet_b0", "tf_mixnet_l", "tinynet_a",
    "twins_pcpvt_base", "visformer_small",
    "vit_base_patch16_224", "vit_large_patch16_224",
    "wide_resnet101_2",
]

REPRESENTATIVE_MODELS = [
    "mobilenetv3_large_100",
    "inception_v3",
    "regnetx_002",
    "efficientnet_b0",
    "resnet101d",
    "vit_base_patch16_224",
    "regnety_002",
    "tf_efficientnet_b0",
    "hrnet_w18",
    "repvgg_a2",
    "darknet53",
    "ssl_resnet18",
]


def get_input_size(model_name: str) -> tuple:
    """Get the expected input size for a timm model."""
    model = timm.create_model(model_name, pretrained=False)
    config = model.default_cfg
    input_size = config.get("input_size", (3, 224, 224))
    del model
    return input_size


def export_model(model_name: str, output_dir: str, opset: int = 17):
    """Export a single timm model to ONNX."""
    output_path = os.path.join(output_dir, f"{model_name}.onnx")
    if os.path.exists(output_path):
        print(f"  [skip] {output_path} already exists")
        return output_path

    print(f"  Exporting {model_name}...")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    input_size = model.default_cfg.get("input_size", (3, 224, 224))
    dummy_input = torch.randn(1, *input_size)

    # Prefer the legacy TorchScript exporter (dynamo=False): the torch 2.x dynamo
    # exporter fails to decompose some graphs (e.g. beit attention). Fall back to
    # the plain call on older torch that doesn't accept the `dynamo` kwarg.
    try:
        torch.onnx.export(
            model, dummy_input, output_path, opset_version=opset,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )
    except Exception:
        torch.onnx.export(
            model, dummy_input, output_path, opset_version=opset,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
    print(f"  [done] {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export timm models to ONNX")
    parser.add_argument(
        "--output-dir", type=str, default="onnx_models",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Specific models to export. If not set, exports representative subset.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Export all 81 models from the benchmark table",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        models = ALL_MODELS
    elif args.models:
        models = args.models
    else:
        models = REPRESENTATIVE_MODELS

    print(f"Exporting {len(models)} models to {args.output_dir}/")
    for name in models:
        try:
            export_model(name, args.output_dir, args.opset)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")


if __name__ == "__main__":
    main()
