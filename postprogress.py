import re
import numpy as np
from collections import defaultdict
import os

# =========================
# 사용자 설정 부분
# =========================
input_files = [
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_33.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_34.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_35.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_36.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_37.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_38.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_39.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_40.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_41.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260122_42.txt",
]

output_path = r"./accuracy_summary.txt"

# =========================
# 정규식 패턴
# =========================
header_pattern = re.compile(
    r"\[COLOR=(?P<color>\w+)\s*\|\s*STRATEGY=(?P<strategy>\d+)\]"
)

accuracy_pattern = re.compile(
    r"Accuracy\s*:\s*(?P<acc>[\d.]+)$"
)

error_pattern = re.compile(
    r"Error-rate\s*:\s*(?P<err>[\d.]+)$"
)

# =========================
# 결과 저장용 dict
# =========================
accuracy_dict = defaultdict(list)
error_dict = defaultdict(list)

# =========================
# 파일 파싱
# =========================
for file_path in input_files:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_color = None
    current_strategy = None

    for line in lines:
        header_match = header_pattern.search(line)
        if header_match:
            current_color = header_match.group("color")
            current_strategy = int(header_match.group("strategy"))
            continue

        if current_color is None:
            continue

        acc_match = accuracy_pattern.search(line)
        if acc_match:
            acc_value = float(acc_match.group("acc"))  # 0~1
            key = (current_color, current_strategy)
            accuracy_dict[key].append(acc_value)
            continue

        err_match = error_pattern.search(line)
        if err_match:
            err_value = float(err_match.group("err"))  # 0~1
            key = (current_color, current_strategy)
            error_dict[key].append(err_value)

# =========================
# 결과 계산 및 저장
# =========================
with open(output_path, "w", encoding="utf-8") as out:
    out.write("Accuracy / Error-rate Summary (Mean / Variance)\n")
    out.write("=" * 60 + "\n\n")

    color_order = {"gray": 0, "color": 1}

    for (color, strategy) in sorted(
        accuracy_dict.keys(),
        key=lambda x: (color_order.get(x[0], 99), x[1])
    ):
        acc_values = np.array(accuracy_dict[(color, strategy)])
        err_values = np.array(error_dict[(color, strategy)])

        acc_mean = acc_values.mean()
        acc_var = acc_values.var(ddof=0)

        err_mean = err_values.mean()
        err_var = err_values.var(ddof=0)

        out.write(f"[COLOR={color} | STRATEGY={strategy}]\n")
        out.write(f"Accuracy   : {acc_mean:.6f} ({acc_var:.6f})\n")
        out.write(f"Error-rate : {err_mean:.6f} ({err_var:.6f})\n")
        out.write("-" * 40 + "\n")

    # =========================
    # 사용한 파일 목록
    # =========================
    out.write("\n")
    out.write("=" * 60 + "\n")
    out.write(f"Used Input Files (Count: {len(input_files)})\n")
    out.write("=" * 60 + "\n")

    for file_path in input_files:
        out.write(f"- {os.path.basename(file_path)}\n")
