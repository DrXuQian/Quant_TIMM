#!/usr/bin/env python3
"""auto_quantize sweep: find min eff_bits for < 1% on lcnet_050 and adv_inception_v3."""
import torch, timm, copy, json, sys, warnings
from PIL import Image
from timm.data import resolve_data_config, create_transform
from torch.utils.data import Dataset, DataLoader
import modelopt.torch.quantization as mtq
warnings.filterwarnings('ignore')

device = torch.device('cuda')

class DS(Dataset):
    def __init__(s, items, tf): s.items, s.tf = items, tf
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p,l = s.items[i]
        return s.tf(Image.open(p).convert('RGB')), l

meta = json.load(open('imagenet_val/labels.json'))
eval_items = [('imagenet_val/'+m['file'], m['label']) for m in meta]
cmeta = json.load(open('imagenet_calib/labels.json'))
calib_items = [('imagenet_calib/'+m['file'], m['label']) for m in cmeta][:512]

@torch.no_grad()
def acc(model, ld):
    model.eval(); c=t=0
    for x,y in ld:
        c += int((model(x.to(device)).argmax(-1).cpu()==y).sum()); t += y.size(0)
    return c/t

FP = {'lcnet_050': 0.6313, 'adv_inception_v3': 0.7759}

# GEMM-only INT8 pc config
int8_pc_cfg = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"parent_class": "nn.Conv2d", "quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"parent_class": "nn.Conv2d", "quantizer_name": "*input_quantizer",  "cfg": {"num_bits": 8, "axis": 1}},
        {"parent_class": "nn.Linear", "quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"parent_class": "nn.Linear", "quantizer_name": "*input_quantizer",  "cfg": {"num_bits": 8, "axis": 1}},
    ],
    "algorithm": "max",
}

for MODEL in ['lcnet_050', 'adv_inception_v3']:
    fp = FP[MODEL]
    m_tmp = timm.create_model(MODEL, pretrained=False)
    tf = create_transform(**resolve_data_config(m_tmp.default_cfg), is_training=False)
    del m_tmp
    eval_ld = DataLoader(DS(eval_items, tf), batch_size=256, num_workers=8, pin_memory=True)
    calib_ld = DataLoader(DS(calib_items, tf), batch_size=64, num_workers=4, pin_memory=True)

    print(f'\n{"="*55}', flush=True)
    print(f'{MODEL}  FP32={fp*100:.2f}%', flush=True)
    print(f'{"="*55}', flush=True)

    for eff_bits in [8, 8.5, 9, 9.5, 10, 11, 12]:
        model = timm.create_model(MODEL, pretrained=True).eval().to(device)
        best_model, _ = mtq.auto_quantize(
            model,
            constraints={"effective_bits": eff_bits},
            quantization_formats=[int8_pc_cfg],
            data_loader=calib_ld,
            forward_step=lambda m, batch: m(batch[0].to(device)),
            method="kl_div",
            num_calib_steps=64,
            num_score_steps=32,
            verbose=False,
        )
        a = acc(best_model, eval_ld)
        delta = (a - fp) * 100
        ok = "OK" if abs(delta) < 1.0 else ""
        print(f'  eff_bits={eff_bits:>4}: {a*100:.2f}% (Δ={delta:+.2f}%) {ok}', flush=True)
        del model, best_model; torch.cuda.empty_cache()
        if abs(delta) < 0.5:
            break

if __name__ == "__main__":
    pass
