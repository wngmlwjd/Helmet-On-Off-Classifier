import os
import re
import numpy as np

from train.utils import SUB_DIRS

def compute_avg_var_from_results(results_txt_path):
    patterns = {
        "accuracy":  re.compile(r"Accuracy\s*:\s*([0-9.]+)"),
        "precision": re.compile(r"Precision\s*:\s*([0-9.]+)"),
        "recall":    re.compile(r"Recall\s*:\s*([0-9.]+)"),
        "f1":        re.compile(r"F1-score\s*:\s*([0-9.]+)")
    }

    with open(results_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    values = {
        k: np.array([float(v) for v in p.findall(text)])
        for k, p in patterns.items()
    }

    avg = {k: v.mean() for k, v in values.items()}
    var = {k: v.var()  for k, v in values.items()}

    return avg, var, len(values["accuracy"])

def write_avg_result(
    save_path,
    color,
    strategy,
    runs,
    avg,
    var
):
    with open(save_path, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("=== Test Results (AVERAGE + VARIANCE) ===\n")
        f.write(f"Color      : {color}\n")
        f.write(f"Strategy   : {strategy}\n")
        f.write(f"Runs       : {runs}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy   : {avg['accuracy']:.4f} ({var['accuracy']:.6f})\n")
        f.write(f"Precision  : {avg['precision']:.4f} ({var['precision']:.6f})\n")
        f.write(f"Recall     : {avg['recall']:.4f} ({var['recall']:.6f})\n")
        f.write(f"F1-score   : {avg['f1']:.4f} ({var['f1']:.6f})\n")
        f.write("\n")

COLORS = ["gray", "color"]
STRATEGIES = [1, 2, 3, 4]
date = "20251230_04"
write_result_txt = os.path.join("./results", f"{date}.txt")
os.makedirs(os.path.dirname(write_result_txt), exist_ok=True)

for color in COLORS:
    for strategy in STRATEGIES:
        BASE_MODEL_SAVE_DIR = os.path.join("./model", color, SUB_DIRS[strategy], date)

        results_txt = os.path.join(BASE_MODEL_SAVE_DIR, "results.txt")
        avg, var, runs = compute_avg_var_from_results(results_txt)

        write_avg_result(
            write_result_txt,
            color,
            strategy,
            runs,
            avg,
            var
        )