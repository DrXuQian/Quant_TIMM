"""Recover completed-model results from a run_experiment.py stdout log into JSON.

Used after a crash: the per-model result rows are in the log even though the
final JSON was never written. Only fully-clean models (no FAILED method) are
recovered; partial/failed models are left out so they get re-run.
"""

import json
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "user_models_run.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/user_models_results.json"

fp_re = re.compile(r"^(\S+)\s+\|\s+FP top-1:\s+([\d.]+)%\s+\(([\d.]+) ms/img")
# e.g. "ort/minmax-pc            1.6%  -60.8%    2.0%   0.1213     4.7"
row_re = re.compile(
    r"^((?:ort|modelopt)/\S+)\s+([\d.]+)%\s+([+\-][\d.]+)%\s+([\d.]+)%\s+([\-\d.]+)\s+([\d.]+)"
)
fail_re = re.compile(r"^((?:ort|modelopt)/\S+)\s+FAILED")

results = {}
cur = None
for line in open(LOG):
    line = line.rstrip("\n")
    m = fp_re.match(line)
    if m:
        cur = m.group(1)
        results[cur] = {
            "fp_top1": round(float(m.group(2)) / 100, 4),
            "fp_ms_per_img": float(m.group(3)),
            "methods": {},
            "_had_failure": False,
        }
        continue
    if cur is None:
        continue
    f = fail_re.match(line)
    if f:
        results[cur]["methods"][f.group(1)] = {"error": "FAILED"}
        results[cur]["_had_failure"] = True
        continue
    r = row_re.match(line)
    if r:
        label, top1, dacc, agree, cos, qnt = r.groups()
        results[cur]["methods"][label] = {
            "top1": round(float(top1) / 100, 4),
            "delta_acc": round(float(dacc) / 100, 4),
            "agreement": round(float(agree) / 100, 4),
            "cosine": float(cos),
            "quant_s": float(qnt),
            "has_naninf": top1.endswith("!"),
        }

# Keep only fully-clean models (7 methods, no failure) so others get re-run.
clean = {}
for model, r in results.items():
    n = len([1 for v in r["methods"].values() if "error" not in v])
    if not r["_had_failure"] and n >= 7:
        r.pop("_had_failure", None)
        clean[model] = r
    else:
        print(f"  skip {model}: {n} clean methods, had_failure={r['_had_failure']}")

json.dump(clean, open(OUT, "w"), indent=2)
print(f"Recovered {len(clean)} clean models -> {OUT}: {list(clean)}")
