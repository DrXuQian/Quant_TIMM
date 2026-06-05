"""Render experiment_results.json into a markdown table and inject it into analysis.md."""

import json
import os
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results/experiment_results.json"
ANALYSIS = os.path.join(os.path.dirname(__file__), "analysis.md")

# Columns we show (skip bf16: not CPU-evaluable)
METHODS = [
    "modelopt/entropy-fp16",
    "modelopt/entropy-fp32",
    "modelopt/max-fp16",
    "modelopt/max-fp32",
    "ort/minmax-pc",
    "ort/entropy-pc",
    "ort/percentile-pc",
]


def cell(rec):
    if not rec or "error" in rec:
        return "—"
    if rec.get("has_naninf"):
        return f"{rec['top1']*100:.1f}%!"
    return f"{rec['top1']*100:.1f}%"


def main():
    data = json.load(open(RESULTS))

    lines = []
    header = "| Model | FP | " + " | ".join(m.replace("modelopt/", "mo:").replace("ort/", "ort:")
                                             for m in METHODS) + " |"
    sep = "|" + "---|" * (len(METHODS) + 2)
    lines.append(header)
    lines.append(sep)

    for model, res in data.items():
        if not isinstance(res, dict) or "fp_top1" not in res:
            continue
        fp = f"{res['fp_top1']*100:.1f}%"
        row = [model, fp]
        for m in METHODS:
            row.append(cell(res.get("methods", {}).get(m)))
        lines.append("| " + " | ".join(row) + " |")

    # Delta table (Δ vs FP) — the honest degradation
    lines.append("")
    lines.append("Δ accuracy vs FP baseline (percentage points):")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for model, res in data.items():
        if not isinstance(res, dict) or "fp_top1" not in res:
            continue
        row = [model, "0.0"]
        for m in METHODS:
            rec = res.get("methods", {}).get(m)
            if not rec or "error" in rec:
                row.append("—")
            else:
                row.append(f"{rec['delta_acc']*100:+.1f}")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    print(table)

    # Inject into analysis.md between the marker
    if os.path.exists(ANALYSIS):
        txt = open(ANALYSIS).read()
        marker = "<!-- RESULTS_TABLE -->"
        if marker in txt:
            txt = txt.replace(marker, table)
            open(ANALYSIS, "w").write(txt)
            print(f"\nInjected table into {ANALYSIS}")


if __name__ == "__main__":
    main()
