import os
import importlib
from datetime import datetime
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict

from train.utils import get_train_model_save_dir

from data_prep.seperating import split_fold, split_train_test
from train.train import train_model
from test.test import test_model

# ===============================
# 실험 설정
# ===============================
COLORS = ["gray", "color"]
STRATEGIES = [1, 2, 3, 4]

N_FOLDS = 5
VAL_SPLIT_RATIO = 0.2

for _ in range(5):
    # ===============================
    # 결과 파일 생성
    # ===============================
    today_str = datetime.now().strftime("%Y%m%d")
    seq = 1
    while True:
        result_filename = f"{today_str}_{seq:02d}.txt"
        RESULTS_PATH = os.path.join("./results", result_filename)
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
            # 데이터 분할
            # -------------------------------------------------
            split_fold(color, strategy, N_FOLDS)
            
            # 모델 저장 디렉토리 생성
            BASE_MODEL_SAVE_DIR = get_train_model_save_dir(color, strategy, today_str)
            
            metric_values = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": []
            }

            for run in range(N_FOLDS):
                run += 1
                    
                split_train_test(run)
                
                TRAIN_MODEL_SAVE_DIR = os.path.join(BASE_MODEL_SAVE_DIR, f"fold_{run:02d}")    
                print(f"💾 SAVE DIR  : {TRAIN_MODEL_SAVE_DIR}")

                # =====================
                # 학습
                # =====================
                train_results = train_model(color, strategy, TRAIN_MODEL_SAVE_DIR, VAL_SPLIT_RATIO)

                best_epoch = train_results["best_epoch"]
                best_val_acc = train_results["best_val_acc"]
                
                best_model_path = os.path.join(TRAIN_MODEL_SAVE_DIR, f"epoch_{best_epoch:02d}.h5")

                # =====================
                # 테스트
                # =====================
                test_results = test_model(color, strategy, best_model_path, best_epoch)

                metric_values["accuracy"].append(test_results["accuracy"])
                metric_values["precision"].append(test_results["precision"])
                metric_values["recall"].append(test_results["recall"])
                metric_values["f1"].append(test_results["f1"])

                # =====================
                # 결과 누적 기록
                # =====================
                with open(os.path.join(BASE_MODEL_SAVE_DIR, "results.txt"), "a", encoding="utf-8") as f:
                    f.write("=" * 40 + "\n")
                    f.write(f"[fold_{run:02d}]\n")
                    f.write(f"Best Epoch: {best_epoch}\n")
                    f.write(f"Val Acc   : {best_val_acc:.4f}\n\n")
                    f.write("--- Test Metrics ---\n")
                    f.write(f"Accuracy  : {test_results['accuracy']:.4f}\n")
                    f.write(f"Precision : {test_results['precision']:.4f}\n")
                    f.write(f"Recall    : {test_results['recall']:.4f}\n")
                    f.write(f"F1-score  : {test_results['f1']:.4f}\n")
                    f.write("=" * 40 + "\n\n")

            avg_acc  = np.mean(metric_values["accuracy"])
            avg_prec = np.mean(metric_values["precision"])
            avg_rec  = np.mean(metric_values["recall"])
            avg_f1   = np.mean(metric_values["f1"])

            var_acc  = np.var(metric_values["accuracy"])
            var_prec = np.var(metric_values["precision"])
            var_rec  = np.var(metric_values["recall"])
            var_f1   = np.var(metric_values["f1"])
                    
            avg_result_path = os.path.join(BASE_MODEL_SAVE_DIR, f"results_AVG_VAR.txt")

            with open(avg_result_path, "w", encoding="utf-8") as f:
                f.write("=== Test Results (AVERAGE + VARIANCE) ===\n")
                f.write(f"Color      : {color}\n")
                f.write(f"Strategy   : {strategy}\n")
                f.write(f"{'-'*40}\n")
                f.write(f"Accuracy   : {avg_acc:.4f} ({var_acc:.6f})\n")
                f.write(f"Precision  : {avg_prec:.4f} ({var_prec:.6f})\n")
                f.write(f"Recall     : {avg_rec:.4f} ({var_rec:.6f})\n")
                f.write(f"F1-score   : {avg_f1:.4f} ({var_f1:.6f})\n")

            with open(RESULTS_PATH, "a", encoding="utf-8") as f:
                f.write(f"[COLOR={color} | STRATEGY={strategy}]\n")
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
