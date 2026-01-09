import os
import importlib
from datetime import datetime
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict

import train.utils as U

# ===============================
# 실험 설정
# ===============================
COLORS = ["gray", "color"]
STRATEGIES = [1, 2, 3, 4]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===============================
# 결과 파일 생성
# ===============================
today_str = datetime.now().strftime("%Y%m%d")
seq = 1
while True:
    result_filename = f"{today_str}_{seq:02d}.txt"
    RESULTS_PATH = os.path.join(RESULTS_DIR, result_filename)
    if not os.path.exists(RESULTS_PATH):
        break
    seq += 1

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write(f"통합 실험 결과\n")
    f.write(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"모델: 4\n")
    f.write("=" * 60 + "\n\n")

# ===============================
# 전체 실험 루프
# ===============================
for color in COLORS:
    for strategy in STRATEGIES:
        print("\n" + "=" * 60)
        print(f"🚀 START: COLOR={color.upper()}, STRATEGY={strategy}")
        print("=" * 60)

        # -------------------------------------------------
        # utils 전역 설정 갱신
        # -------------------------------------------------
        U.SELECTED_COLOR = color
        U.SELECTED_STRATEGY = strategy

        U.TRAIN_IMAGE_DIR = os.path.join(
            U.PREPROCESSED_DIR,
            color,
            "train",
            U.SUB_DIRS[strategy],
            "images"
        )
        U.TRAIN_LABEL_DIR = os.path.join(
            U.PREPROCESSED_DIR,
            color,
            "train",
            U.SUB_DIRS[strategy],
            "labels"
        )

        print(f"📂 IMAGE DIR : {U.TRAIN_IMAGE_DIR}")
        print(f"📂 LABEL DIR : {U.TRAIN_LABEL_DIR}")
        
        # 모델 저장 디렉토리 생성
        BASE_MODEL_SAVE_DIR = U.get_train_model_save_dir(color, strategy)
        
        metric_values = defaultdict(lambda: {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        })

        for run in range(U.RUN_TIMES):
            U.SEED = 42 + run

            random.seed(U.SEED)
            np.random.seed(U.SEED)
            tf.random.set_seed(U.SEED)
            
            TRAIN_MODEL_SAVE_DIR = os.path.join(BASE_MODEL_SAVE_DIR, f"set_{run+1:02d}")    
            print(f"💾 SAVE DIR  : {TRAIN_MODEL_SAVE_DIR}")

            # =====================
            # 학습 (Cross Validation)
            # =====================
            train_mod = importlib.import_module("train.train_cross_validation")
            importlib.reload(train_mod)

            fold_results = train_mod.train_model(
                selected_color=color,
                selected_strategy=strategy,
                model_save_dir=TRAIN_MODEL_SAVE_DIR,
                n_folds=U.N_FOLDS
            )

            # -------------------------------------------------
            # 최고 성능 fold 선택
            # -------------------------------------------------
            best_fold = max(fold_results, key=lambda x: x["best_val_acc"])
            best_fold_idx = best_fold["fold"]
            best_epoch = best_fold["best_epoch"]
            best_val_acc = best_fold["best_val_acc"]

            print(
                f"🏆 BEST MODEL → Fold {best_fold_idx}, "
                f"Epoch {best_epoch}, Val Acc {best_val_acc:.4f}"
            )

            best_model_path = os.path.join(
                TRAIN_MODEL_SAVE_DIR,
                f"fold_{best_fold_idx}",
                f"epoch_{best_epoch:02d}.h5"
            )

            # =====================
            # 테스트
            # =====================
            test_mod = importlib.import_module("test.test_cross_validation")
            importlib.reload(test_mod)

            test_metrics = test_mod.test_model(
                selected_color=color,
                selected_strategy=strategy,
                model_path=best_model_path,
                epoch=best_epoch,
                set_idx=run+1,
                fold_idx=best_fold_idx,
                results_txt_dir=TRAIN_MODEL_SAVE_DIR,
                confusion_matrix_dir=TRAIN_MODEL_SAVE_DIR
            )
                        
            key = (color, strategy)

            metric_values[key]["accuracy"].append(test_metrics["accuracy"])
            metric_values[key]["precision"].append(test_metrics["precision"])
            metric_values[key]["recall"].append(test_metrics["recall"])
            metric_values[key]["f1"].append(test_metrics["f1"])

            # =====================
            # 결과 누적 기록
            # =====================
            results_txt_path = os.path.join(
                TRAIN_MODEL_SAVE_DIR,
                f"results_color_{color}_strategy_{strategy}_epoch_{best_epoch}.txt"
            )

            with open(os.path.join(BASE_MODEL_SAVE_DIR, "results.txt"), "a", encoding="utf-8") as f:
                f.write("=" * 40 + "\n")
                f.write(f"[SET {run+1:02d}]\n")
                f.write(f"Best Fold : {best_fold_idx}\n")
                f.write(f"Best Epoch: {best_epoch}\n")
                f.write(f"Val Acc   : {best_val_acc:.4f}\n\n")
                f.write("--- Test Metrics ---\n")
                f.write(f"Accuracy  : {test_metrics['accuracy']:.4f}\n")
                f.write(f"Precision : {test_metrics['precision']:.4f}\n")
                f.write(f"Recall    : {test_metrics['recall']:.4f}\n")
                f.write(f"F1-score  : {test_metrics['f1']:.4f}\n")
                f.write("=" * 40 + "\n\n")
        
        vals = metric_values[(color, strategy)]

        avg_acc  = np.mean(vals["accuracy"])
        avg_prec = np.mean(vals["precision"])
        avg_rec  = np.mean(vals["recall"])
        avg_f1   = np.mean(vals["f1"])

        var_acc  = np.var(vals["accuracy"])
        var_prec = np.var(vals["precision"])
        var_rec  = np.var(vals["recall"])
        var_f1   = np.var(vals["f1"])
                
        avg_result_path = os.path.join(
            BASE_MODEL_SAVE_DIR,
            f"results_color_{color}_strategy_{strategy}_AVG.txt"
        )

        with open(avg_result_path, "w", encoding="utf-8") as f:
            f.write("=== Test Results (AVERAGE + VARIANCE) ===\n")
            f.write(f"Color      : {color}\n")
            f.write(f"Strategy   : {strategy}\n")
            f.write(f"Runs       : {U.RUN_TIMES}\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Accuracy   : {avg_acc:.4f} ({var_acc:.6f})\n")
            f.write(f"Precision  : {avg_prec:.4f} ({var_prec:.6f})\n")
            f.write(f"Recall     : {avg_rec:.4f} ({var_rec:.6f})\n")
            f.write(f"F1-score   : {avg_f1:.4f} ({var_f1:.6f})\n")

        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(f"[COLOR={color} | STRATEGY={strategy}]\n")
            f.write(f"Runs       : {U.RUN_TIMES}\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Accuracy   : {avg_acc:.4f} ({var_acc:.6f})\n")
            f.write(f"Precision  : {avg_prec:.4f} ({var_prec:.6f})\n")
            f.write(f"Recall     : {avg_rec:.4f} ({var_rec:.6f})\n")
            f.write(f"F1-score   : {avg_f1:.4f} ({var_f1:.6f})\n")
            f.write("\n" + "=" * 60 + "\n\n")

print(f"\n✅ 모든 실험 완료")
with open(RESULTS_PATH, "a", encoding="utf-8") as f:
    f.write(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
print(f"📄 통합 결과 파일: {RESULTS_PATH}")
