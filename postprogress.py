import re
import numpy as np
from collections import defaultdict
import os

# =========================
# 사용자 설정 부분
# =========================
input_files = [
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260109_01.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260109_02.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260109_03.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260109_04.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260109_05.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260109_06.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260109_07.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260109_08.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260110_01.txt",
    r"/Users/wngmlwjd/workspace/github/helmet-on-off-classifier/results/20260110_02.txt",
]

output_path = r"./accuracy_summary.txt"

# =========================
# 정규식 패턴
# =========================
header_pattern = re.compile(
    r"\[COLOR=(?P<color>\w+)\s*\|\s*STRATEGY=(?P<strategy>\d+)\]"
)

accuracy_pattern = re.compile(
    r"Accuracy\s*:\s*(?P<acc>[0-9.]+)"
)

# =========================
# 데이터 수집
# =========================
accuracy_dict = defaultdict(list)

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

        acc_match = accuracy_pattern.search(line)
        if acc_match and current_color is not None:
            acc_value = float(acc_match.group("acc"))
            key = (current_color, current_strategy)
            accuracy_dict[key].append(acc_value)

# =========================
# 결과 계산 및 저장
# =========================
with open(output_path, "w", encoding="utf-8") as out:
    out.write("Accuracy Summary (Mean / Variance)\n")
    out.write("=" * 50 + "\n\n")

    count = 0
    color_order = {"gray": 0, "color": 1}

    for (color, strategy), values in sorted(accuracy_dict.items(), key=lambda x: (color_order.get(x[0][0], 99), x[0][1])):
        values_np = np.array(values)
        mean = values_np.mean()
        var = values_np.var(ddof=0)  # 모집단 분산

        out.write(f"[COLOR={color} | STRATEGY={strategy}]\n")
        out.write(f"Accuracy: {mean:.6f} ({var:.6f})\n")
        out.write("-" * 40 + "\n")
        
        count = len(values)
        
    # =========================
    # 사용한 파일명 목록 추가
    # =========================
    out.write("\n")
    out.write("=" * 50 + "\n")
    out.write(f"Used Input Files (Count: {count})\n")
    out.write("=" * 50 + "\n")

    for file_path in input_files:
        out.write(f"- {os.path.basename(file_path)}\n")
